"""Video-text alignment model for stage 1 contrastive training.

Video path:
    PGAT variable tokenizer -> temporal tokens
    + articulator + global summary
    -> mean over 4 global tokens
    -> project to alignment_dim
    -> L2-normalize
    -> unit sphere embedding

Text path:
    mBART encoder (frozen at stage 1) on the tokenized German reference
    -> mean-pool over valid tokens
    -> project to alignment_dim
    -> L2-normalize
    -> unit sphere embedding

Loss:
    InfoNCE with temperature 0.07, symmetric (v2t + t2v).
    Optional hard-negative loss added with a configurable weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import MBartModel

from pgat_length.models.articulator import BiasedArticulatorAttention
from pgat_length.models.global_summary import GlobalSummaryAttention
from pgat_length.models.tokenizer import EncoderConfig, PgatVariableTokenizer


@dataclass(frozen=True)
class AlignmentConfig:
    encoder: EncoderConfig
    text_model_name: str
    text_hidden_dim: int
    alignment_dim: int
    hf_cache: Path | None = None
    articulator_queries: int = 8
    global_queries: int = 4


class TextEncoder(nn.Module):
    """Wraps the mBART encoder for contrastive alignment.

    We use only the encoder side of mBART. It's kept in eval mode with
    parameters frozen at stage 1 (translation stage 2 will unfreeze).
    """

    def __init__(self, model_name: str, hf_cache: Path | None = None, freeze: bool = True) -> None:
        super().__init__()
        cache_dir = str(hf_cache) if hf_cache else None
        base = MBartModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.encoder = base.get_encoder()
        del base
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    def forward(
        self,
        input_ids: torch.Tensor,        # [B, T] int32
        attention_mask: torch.Tensor,   # [B, T] bool
    ) -> torch.Tensor:
        """Returns [B, hidden] mean-pooled encoder output over valid tokens."""
        attn_int = attention_mask.to(torch.long)
        outputs = self.encoder(
            input_ids=input_ids.to(torch.long),
            attention_mask=attn_int,
        )
        hidden = outputs.last_hidden_state  # [B, T, hidden]
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)  # [B, T, 1]
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom  # [B, hidden]


class PgatAlignmentModel(nn.Module):
    def __init__(self, config: AlignmentConfig) -> None:
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
        self.text_encoder = TextEncoder(
            config.text_model_name, hf_cache=config.hf_cache, freeze=True
        )
        # Projection heads: video hidden -> alignment_dim, text hidden -> alignment_dim.
        self.video_projection = nn.Sequential(
            nn.LayerNorm(config.encoder.hidden_dim),
            nn.Linear(config.encoder.hidden_dim, config.alignment_dim, bias=True),
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(config.text_hidden_dim),
            nn.Linear(config.text_hidden_dim, config.alignment_dim, bias=True),
        )

    # ---- video ---------------------------------------------------------

    def encode_video(self, batch: dict) -> torch.Tensor:
        """Returns L2-normalized [B, alignment_dim] video embedding."""
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
        # Use articulator + global summary to condense to fixed 4 global tokens.
        _ = self.articulator(
            temporal_tokens=temporal_tokens,
            segment_valid=segment_valid,
            pose_confidence=batch["pose_confidence"],
            pose_motion=batch["pose_motion"],
        )
        global_tokens = self.global_summary(temporal_tokens, segment_valid)  # [B, 4, H]
        pooled = global_tokens.mean(dim=1)  # [B, H]
        projected = self.video_projection(pooled)
        return F.normalize(projected, p=2, dim=-1)

    # ---- text ----------------------------------------------------------

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pooled = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        projected = self.text_projection(pooled)
        return F.normalize(projected, p=2, dim=-1)


class InfoNceWithHardNegatives(nn.Module):
    def __init__(self, temperature: float = 0.07, hard_weight: float = 0.5, top_k: int = 4) -> None:
        super().__init__()
        self.temperature = temperature
        self.hard_weight = hard_weight
        self.top_k = top_k

    def forward(
        self,
        video: torch.Tensor,   # [B, D] L2-normalized
        text: torch.Tensor,    # [B, D] L2-normalized
    ) -> tuple[torch.Tensor, dict[str, float]]:
        B = video.shape[0]
        logits = video @ text.T / self.temperature  # [B, B]
        labels = torch.arange(B, device=logits.device)
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.T, labels)
        infonce = 0.5 * (loss_v2t + loss_t2v)

        # Hard negatives: mask out positive, then pick top-k highest per anchor.
        with torch.no_grad():
            neg_mask = torch.ones_like(logits, dtype=torch.bool)
            neg_mask[torch.arange(B), torch.arange(B)] = False
        neg_logits = logits.masked_fill(~neg_mask, float("-inf"))
        # top-k hardest negatives per anchor (from v-to-t direction).
        top_k = min(self.top_k, B - 1)
        _, top_idx = neg_logits.topk(top_k, dim=-1)  # [B, top_k]
        # For each anchor i, compute InfoNCE where negatives are only the top_k.
        hard_negatives_scores = torch.gather(logits, 1, top_idx)  # [B, top_k]
        pos_scores = logits.diagonal().unsqueeze(-1)  # [B, 1]
        stacked = torch.cat((pos_scores, hard_negatives_scores), dim=-1)  # [B, top_k+1]
        hard_labels = torch.zeros(B, dtype=torch.long, device=logits.device)
        hard_loss = F.cross_entropy(stacked, hard_labels)

        total = infonce + self.hard_weight * hard_loss
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            v2t_acc = (preds == labels).float().mean().item()
            preds_t2v = logits.T.argmax(dim=-1)
            t2v_acc = (preds_t2v == labels).float().mean().item()
        stats = {
            "loss_infonce": float(infonce.item()),
            "loss_hard": float(hard_loss.item()),
            "loss_total": float(total.item()),
            "acc_v2t": float(v2t_acc),
            "acc_t2v": float(t2v_acc),
        }
        return total, stats
