"""PGAT-v2 tokenizer with variable K_temporal per sample.

Emits ``[B, max_K, hidden_dim]`` temporal tokens plus a validity mask where
``max_K`` is the batch-max real K (padded samples zero out attention). All
downstream modules (articulator, global summary, alignment head) consume
this padded/masked representation.

Design (matches docs/ARCHITECTURE.md §5):
- Six candidate sources per segment: global crop, left/right/mouth safe
  fallbacks, motion (interpolated to K), pose descriptor.
- Confidence-safe fallback for hands/mouth: c*local + (1-c)*global.
- Learned gate over the 6 sources, conditioned on
  [pose_descriptor, confidence, motion].
- Temporal Transformer encoder over K tokens with padding mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EncoderConfig:
    spatial_dim: int = 768
    motion_dim: int = 768
    pose_descriptor_dim: int = 64
    articulator_views: int = 4
    hidden_dim: int = 512
    transformer_layers: int = 4
    transformer_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.10


def _view_projection(spatial_dim: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(spatial_dim),
        nn.Linear(spatial_dim, hidden_dim, bias=True),
    )


class PgatVariableTokenizer(nn.Module):
    """Six-source gated fusion + temporal transformer with variable K per sample."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        H = config.hidden_dim

        # Per-view projections (global + 3 articulator views share dim).
        self.view_projections = nn.ModuleList(
            [_view_projection(config.spatial_dim, H) for _ in range(config.articulator_views)]
        )

        # Pose descriptor and motion projections.
        self.pose_descriptor_projection = nn.Sequential(
            nn.LayerNorm(config.pose_descriptor_dim),
            nn.Linear(config.pose_descriptor_dim, H, bias=True),
        )
        self.motion_projection = nn.Sequential(
            nn.LayerNorm(config.motion_dim),
            nn.Linear(config.motion_dim, H, bias=True),
        )

        # Gate over the six sources.
        # Gate input: pose_descriptor (64) + confidence (4) + motion (4) = 72.
        gate_in = config.pose_descriptor_dim + 2 * config.articulator_views
        self.gate = nn.Sequential(
            nn.LayerNorm(gate_in),
            nn.Linear(gate_in, H, bias=True),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(H, 6, bias=True),
        )

        # Learned positional embedding for up to K_MAX temporal positions.
        # Note: K_MAX is bounded by the feature-side constant (32).
        self.max_temporal_positions = 64
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, self.max_temporal_positions, H)
        )
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H,
            nhead=config.transformer_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers,
            norm=nn.LayerNorm(H),
        )

    # ---- helpers -------------------------------------------------------

    def _interpolate_motion(
        self,
        motion_features: torch.Tensor,   # [B, M, D_motion]
        motion_centers: torch.Tensor,    # [B, M] in [0, 1]
        target_k: int,
    ) -> torch.Tensor:
        """Linearly interpolate M motion clip embeddings to target_k positions.

        Uniform grid at target_k / (K-1) fractions along [0, 1].
        """
        B, M, D = motion_features.shape
        device = motion_features.device
        query = torch.linspace(0.0, 1.0, target_k, device=device).unsqueeze(0).expand(B, -1)  # [B, K]
        # For each batch element, index into motion_centers with linear interp.
        # motion_centers is monotonic; assume [B, M] with values in [0, 1].
        # For simplicity use torch's grid_sample-style interpolation manually:
        # Find left index and blend weight for each query position.
        centers = motion_centers.to(query.dtype)  # [B, M]
        # Clamp query into range of centers (avoid extrapolation beyond ends).
        c0 = centers[:, :1]  # [B, 1]
        c1 = centers[:, -1:]  # [B, 1]
        query = torch.clamp(query, min=c0, max=c1)  # [B, K]
        # For each query, find idx_l such that centers[b, idx_l] <= q < centers[b, idx_l+1].
        # Vectorised via searchsorted.
        idx_r = torch.searchsorted(centers, query)  # [B, K]
        idx_r = torch.clamp(idx_r, min=1, max=M - 1)
        idx_l = idx_r - 1
        # Blend weight.
        left_center = torch.gather(centers, 1, idx_l)   # [B, K]
        right_center = torch.gather(centers, 1, idx_r)  # [B, K]
        denom = (right_center - left_center).clamp(min=1e-6)
        w = ((query - left_center) / denom).unsqueeze(-1)  # [B, K, 1]
        left = torch.gather(motion_features, 1, idx_l.unsqueeze(-1).expand(-1, -1, D))
        right = torch.gather(motion_features, 1, idx_r.unsqueeze(-1).expand(-1, -1, D))
        return left * (1.0 - w) + right * w  # [B, K, D]

    # ---- forward -------------------------------------------------------

    def forward(
        self,
        spatial_features: torch.Tensor,   # [B, K_MAX, 4, D_spatial]  fp16 stored, cast to model dtype
        spatial_valid: torch.Tensor,      # [B, K_MAX, 4]  bool
        motion_features: torch.Tensor,    # [B, 8, D_motion]
        motion_centers: torch.Tensor,     # [B, 8]  in [0, 1]
        pose_descriptor: torch.Tensor,    # [B, K_MAX, 64]  fp16
        pose_confidence: torch.Tensor,    # [B, K_MAX, 4]  fp16
        pose_motion: torch.Tensor,        # [B, K_MAX, 4]  fp16
        segment_valid: torch.Tensor,      # [B, K_MAX]  bool  (True for real segments)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (temporal_tokens [B, max_K, H], valid_mask [B, max_K])."""
        B, K_MAX, V, _ = spatial_features.shape
        device = spatial_features.device
        param_dtype = self.pose_descriptor_projection[1].weight.dtype

        # Cast inputs to the model's compute dtype.
        spatial = spatial_features.to(param_dtype)
        pose_desc = pose_descriptor.to(param_dtype)
        pose_conf = pose_confidence.to(param_dtype)
        pose_mot = pose_motion.to(param_dtype)
        motion = motion_features.to(param_dtype)
        motion_ctr = motion_centers.to(param_dtype)

        # Per-view projections. [B, K_MAX, H] each.
        projected: list[torch.Tensor] = []
        for view_idx in range(V):
            x = spatial[:, :, view_idx, :]  # [B, K_MAX, D_spatial]
            projected.append(self.view_projections[view_idx](x))

        global_view = projected[0]  # [B, K_MAX, H]
        # Confidence-safe fallback for views 1..3.
        safe_views: list[torch.Tensor] = []
        for view_idx in range(1, V):
            c = pose_conf[:, :, view_idx].unsqueeze(-1)  # [B, K_MAX, 1]
            v = projected[view_idx]
            safe = c * v + (1.0 - c) * global_view
            safe_views.append(safe)

        # Pose descriptor projection.
        pose_proj = self.pose_descriptor_projection(pose_desc)  # [B, K_MAX, H]

        # Motion projection then interpolation to K_MAX positions.
        motion_full = self.motion_projection(motion)  # [B, 8, H]
        motion_interp = self._interpolate_motion(motion_full, motion_ctr, K_MAX)  # [B, K_MAX, H]

        # Assemble the 6 sources: global, left_safe, right_safe, mouth_safe, motion, pose.
        sources = torch.stack(
            (global_view, safe_views[0], safe_views[1], safe_views[2], motion_interp, pose_proj),
            dim=-2,
        )  # [B, K_MAX, 6, H]

        # Gate input: concat pose_desc + confidence + motion channels.
        gate_input = torch.cat((pose_desc, pose_conf, pose_mot), dim=-1)  # [B, K_MAX, 72]
        gate_logits = self.gate(gate_input)  # [B, K_MAX, 6]
        gate_weights = F.softmax(gate_logits, dim=-1)  # [B, K_MAX, 6]

        # Fused segment tokens.
        fused = (gate_weights.unsqueeze(-1) * sources).sum(dim=-2)  # [B, K_MAX, H]

        # Zero out padded segments explicitly (safety; encoder mask does the rest).
        seg_valid = segment_valid.to(dtype=torch.bool)
        fused = fused * seg_valid.unsqueeze(-1).to(fused.dtype)

        # Add positional embeddings (learned).
        pos = self.positional_embedding[:, :K_MAX, :].to(fused.dtype)
        fused = fused + pos

        # Transformer expects key_padding_mask=True at PADDING positions.
        key_padding_mask = ~seg_valid  # [B, K_MAX]
        # PyTorch Transformer requires bool key padding mask; True = pad.
        encoded = self.temporal_encoder(fused, src_key_padding_mask=key_padding_mask)
        # Zero out padded positions after encoding as a defensive measure.
        encoded = encoded * seg_valid.unsqueeze(-1).to(encoded.dtype)
        return encoded, seg_valid
