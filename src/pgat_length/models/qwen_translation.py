"""PGAT variable-length prefix + Qwen2.5-3B QLoRA decoder.

Video path is identical to models/translation.py through the visual memory
(temporal tokens + articulator + global summary, variable pi in [24, 44]),
but the projection targets Qwen's hidden dim (2048) and the decoder is a
causal LM conditioned on [prompt | visual prefix | target] via inputs_embeds.

Loss: causal-LM next-token prediction with prompt/prefix positions labelled
-100 (only the target section contributes gradient).

Generation: models/qwen_translation.py::generate_qwen (see qwen_generate.py
in evaluation/).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from pgat_length.models.articulator import BiasedArticulatorAttention
from pgat_length.models.global_summary import GlobalSummaryAttention
from pgat_length.models.projection import PgatQwenProjection
from pgat_length.models.tokenizer import EncoderConfig, PgatVariableTokenizer


@dataclass(frozen=True)
class QwenTranslationConfig:
    encoder: EncoderConfig
    qwen_hidden_dim: int
    articulator_queries: int
    global_queries: int
    qlora: Mapping[str, Any] = field(default_factory=dict)
    hf_cache: Path | None = None


class PgatQwenTranslationModel(nn.Module):
    def __init__(self, config: QwenTranslationConfig) -> None:
        super().__init__()
        self.config = config
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
        self.projection = PgatQwenProjection(
            pgat_dim=config.encoder.hidden_dim,
            qwen_dim=config.qwen_hidden_dim,
            layernorm=True,
            bias=True,
        )
        # QLoRA decoder is loaded lazily by attach_qlora_decoder so we can
        # keep model construction CPU-safe (the alignment warm-start reads
        # state before we go to GPU).
        self.decoder: nn.Module | None = None
        self._decoder_dtype: torch.dtype | None = None

    def attach_qlora_decoder(self, decoder: nn.Module, decoder_dtype: torch.dtype) -> None:
        embedding_dim = int(decoder.get_input_embeddings().embedding_dim)
        if embedding_dim != self.config.qwen_hidden_dim:
            raise ValueError(
                f"Qwen embedding dim {embedding_dim} does not match "
                f"projection dim {self.config.qwen_hidden_dim}"
            )
        self.decoder = decoder
        self._decoder_dtype = decoder_dtype

    # ---- visual prefix ------------------------------------------------

    def encode_visual_prefix(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
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
        )
        glob_tokens = self.global_summary(
            temporal_tokens=temporal_tokens,
            segment_valid=segment_valid,
        )
        B = temporal_tokens.shape[0]
        summary_valid = torch.ones(
            (B, art_tokens.shape[1] + glob_tokens.shape[1]),
            dtype=torch.bool,
            device=temporal_tokens.device,
        )
        prefix_tokens = torch.cat((temporal_tokens, art_tokens, glob_tokens), dim=1)
        prefix_valid = torch.cat((segment_valid, summary_valid), dim=1)
        prefix_embeds = self.projection(prefix_tokens)
        return prefix_embeds, prefix_valid

    # ---- assemble [prompt | prefix | target] --------------------------

    def _assemble(
        self,
        prefix_embeds: torch.Tensor,
        prefix_mask: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor | None,
        target_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.decoder is None:
            raise RuntimeError("Qwen decoder not attached; call attach_qlora_decoder first")
        embed = self.decoder.get_input_embeddings()
        prompt_embeds = embed(prompt_input_ids.long())
        target_embeds: torch.Tensor | None = None
        if target_input_ids is not None:
            target_embeds = embed(target_input_ids.long())
        prefix_embeds = prefix_embeds.to(dtype=prompt_embeds.dtype, device=prompt_embeds.device)
        parts_embeds = [prompt_embeds, prefix_embeds]
        parts_mask = [
            prompt_attention_mask.to(device=prompt_embeds.device, dtype=torch.long),
            prefix_mask.to(device=prompt_embeds.device, dtype=torch.long),
        ]
        if target_embeds is not None and target_attention_mask is not None:
            parts_embeds.append(target_embeds.to(device=prompt_embeds.device))
            parts_mask.append(target_attention_mask.to(device=prompt_embeds.device, dtype=torch.long))
        inputs_embeds = torch.cat(parts_embeds, dim=1)
        attention_mask = torch.cat(parts_mask, dim=1)
        labels: torch.Tensor | None = None
        if target_input_ids is not None and target_attention_mask is not None:
            B = prompt_input_ids.shape[0]
            device = prompt_input_ids.device
            ignored_prompt = torch.full_like(prompt_input_ids, -100, dtype=torch.long)
            ignored_prefix = torch.full(
                (B, prefix_embeds.shape[1]), -100, dtype=torch.long, device=device
            )
            tgt_labels = target_input_ids.long().masked_fill(
                ~target_attention_mask.to(dtype=torch.bool), -100
            )
            labels = torch.cat([ignored_prompt, ignored_prefix, tgt_labels], dim=1)
        return inputs_embeds, attention_mask, labels

    # ---- forward: training loss --------------------------------------

    def forward(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        prefix_embeds, prefix_mask = self.encode_visual_prefix(batch)
        inputs_embeds, attention_mask, labels = self._assemble(
            prefix_embeds=prefix_embeds,
            prefix_mask=prefix_mask,
            prompt_input_ids=batch["prompt_input_ids"],
            prompt_attention_mask=batch["prompt_attention_mask"],
            target_input_ids=batch["target_input_ids"],
            target_attention_mask=batch["target_attention_mask"],
        )
        assert self.decoder is not None
        output = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        loss = output.loss
        with torch.no_grad():
            logits = output.logits[:, :-1, :]
            shifted_labels = labels[:, 1:]
            preds = logits.argmax(dim=-1)
            valid = shifted_labels != -100
            correct = ((preds == shifted_labels) & valid).float().sum()
            total = valid.float().sum().clamp(min=1.0)
            token_acc = float((correct / total).item())
        return loss, {"loss": float(loss.item()), "token_acc": token_acc}
