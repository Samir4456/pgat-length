"""PGAT-v2 translation model: PGAT variable-length prefix + mBART.

Video path:
    PGAT tokenizer -> temporal tokens + articulator + global summary
    -> concat prefix [B, K+12, H]
    -> projection to mBART hidden dim [B, K+12, D_m]
    -> mBART encoder (inputs_embeds; visual tokens replace word ids)

Text path:
    labels are the mBART tokenizer ids for the German reference
    (already produced offline by scripts/04_build_text.py).

Loss:
    Label-smoothed cross-entropy over the decoder logits.

Generation:
    model.generate() with beam search on the mBART decoder using
    encoder outputs from the visual path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import MBartForConditionalGeneration

from pgat_length.models.articulator import BiasedArticulatorAttention
from pgat_length.models.global_summary import GlobalSummaryAttention
from pgat_length.models.projection import PgatMbartProjection
from pgat_length.models.tokenizer import EncoderConfig, PgatVariableTokenizer


@dataclass(frozen=True)
class TranslationConfig:
    encoder: EncoderConfig
    text_model_name: str
    mbart_hidden_dim: int
    articulator_queries: int
    global_queries: int
    hf_cache: Path | None = None
    label_smoothing: float = 0.1


class PgatMbartTranslationModel(nn.Module):
    def __init__(self, config: TranslationConfig) -> None:
        super().__init__()
        self.config = config
        # Encoder side (same building blocks as alignment).
        self.tokenizer = PgatVariableTokenizer(config.encoder)
        self.articulator = BiasedArticulatorAttention(
            hidden_dim=config.encoder.hidden_dim,
            num_queries=config.articulator_queries,
            num_heads=config.encoder.transformer_heads,
            dropout=config.encoder.dropout,
        )
        self.global_summary = GlobalSummaryAttention(
            hidden_dim=config.encoder.hidden_dim,
            num_queries=config.global_queries,
            num_heads=config.encoder.transformer_heads,
            dropout=config.encoder.dropout,
        )
        self.projection = PgatMbartProjection(
            pgat_dim=config.encoder.hidden_dim,
            mbart_dim=config.mbart_hidden_dim,
            layernorm=True,
            bias=True,
        )
        # Full mBART for conditional generation (encoder + decoder + lm_head).
        cache_dir = str(config.hf_cache) if config.hf_cache else None
        self.mbart = MBartForConditionalGeneration.from_pretrained(
            config.text_model_name, cache_dir=cache_dir
        )

    # ---- visual prefix ------------------------------------------------

    def encode_visual_prefix(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (encoder_inputs [B, K+12, D_m], attention_mask [B, K+12])."""
        temporal_tokens, segment_valid = self.tokenizer(
            spatial_features=batch["spatial_features"],
            spatial_valid=batch["spatial_valid"],
            motion_features=batch["motion_features"],
            motion_centers=batch["motion_centers"],
            pose_descriptor=batch["pose_descriptor"],
            pose_confidence=batch["pose_confidence"],
            pose_motion=batch["pose_motion"],
            segment_valid=batch["segment_valid"],
        )
        art_tokens = self.articulator(
            temporal_tokens=temporal_tokens,
            segment_valid=segment_valid,
            pose_confidence=batch["pose_confidence"],
            pose_motion=batch["pose_motion"],
        )  # [B, 8, H]
        glob_tokens = self.global_summary(
            temporal_tokens=temporal_tokens,
            segment_valid=segment_valid,
        )  # [B, 4, H]
        # Assemble the full visual memory: [B, K_MAX + 12, H].
        # Articulator + global summary tokens are ALWAYS valid; temporal tokens
        # take their validity from segment_valid.
        B = temporal_tokens.shape[0]
        summary_valid = torch.ones(
            (B, art_tokens.shape[1] + glob_tokens.shape[1]),
            dtype=torch.bool,
            device=temporal_tokens.device,
        )
        prefix_tokens = torch.cat((temporal_tokens, art_tokens, glob_tokens), dim=1)
        prefix_valid = torch.cat((segment_valid, summary_valid), dim=1)
        # Project to mBART hidden dim.
        prefix_embeds = self.projection(prefix_tokens)
        return prefix_embeds, prefix_valid

    # ---- forward: training loss --------------------------------------

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        """Returns (loss, stats). Uses label-smoothed cross-entropy."""
        prefix_embeds, prefix_mask = self.encode_visual_prefix(batch)
        labels = batch["input_ids"].to(torch.long)
        # mBART's forward with labels shifts them internally for teacher forcing.
        # We compute logits without built-in loss so we can apply label smoothing.
        outputs = self.mbart(
            inputs_embeds=prefix_embeds,
            attention_mask=prefix_mask.to(torch.long),
            labels=labels,
        )
        logits = outputs.logits  # [B, T, V]
        # Mask out pad positions in labels (attention_mask == False).
        attn_mask = batch["attention_mask"].to(torch.bool)
        target = labels.clone()
        target[~attn_mask] = -100
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            ignore_index=-100,
            label_smoothing=self.config.label_smoothing,
        )
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = ((preds == target) & (target != -100)).float().sum()
            total = (target != -100).float().sum().clamp(min=1.0)
            token_acc = float((correct / total).item())
        return loss, {"loss": float(loss.item()), "token_acc": token_acc}

    # ---- generation ---------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        batch: dict,
        tokenizer,
        max_new_tokens: int = 96,
        num_beams: int = 3,
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
    ) -> list[str]:
        """Return list of decoded German strings, one per batch element."""
        self.eval()
        prefix_embeds, prefix_mask = self.encode_visual_prefix(batch)
        # Prepare the decoder_start_token_id: for mBART the target-language code.
        tgt_lang = tokenizer.tgt_lang or "de_DE"
        decoder_start_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        forced_bos_token_id = decoder_start_token_id  # mBART expects lang id as first decoded token
        outputs = self.mbart.generate(
            inputs_embeds=prefix_embeds,
            attention_mask=prefix_mask.to(torch.long),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            decoder_start_token_id=decoder_start_token_id,
            forced_bos_token_id=forced_bos_token_id,
        )
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [t.strip() for t in texts]
