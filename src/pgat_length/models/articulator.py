"""Biased articulator attention (8 learned queries).

Attends from 8 articulator queries into the encoded temporal tokens. Attention
logits are biased by pose confidence and motion magnitude (per query group),
so each articulator query is pulled toward timesteps where its own articulator
is well-observed and moving.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


# 8 queries divided across 4 groups (2 per group):
#   {left_hand, right_hand, mouth, upper_body}
# The confidence/motion channels are indexed:
#   0 = left, 1 = right, 2 = mouth, 3 = body
QUERY_GROUP_INDICES = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)


class BiasedArticulatorAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_queries: int = 8,
        num_heads: int = 8,
        dropout: float = 0.10,
        confidence_scale_init: float = 1.0,
        motion_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        if num_queries != 8:
            raise ValueError("BiasedArticulatorAttention expects 8 queries")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        if self.head_dim * num_heads != hidden_dim:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.queries = nn.Parameter(torch.zeros(num_queries, hidden_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Learnable positive scalars for the confidence and motion bias.
        self.confidence_scale = nn.Parameter(torch.tensor(confidence_scale_init))
        self.motion_scale = nn.Parameter(torch.tensor(motion_scale_init))

        # Post-attention FFN + LayerNorm (residual around cross-attention).
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim, bias=True),
        )

        # Register query group indices as a buffer so it moves with .to(device).
        self.register_buffer("query_groups", QUERY_GROUP_INDICES, persistent=False)

    def forward(
        self,
        temporal_tokens: torch.Tensor,   # [B, K_MAX, H]
        segment_valid: torch.Tensor,     # [B, K_MAX] bool
        pose_confidence: torch.Tensor,   # [B, K_MAX, 4]
        pose_motion: torch.Tensor,       # [B, K_MAX, 4]
    ) -> torch.Tensor:
        B, K, H = temporal_tokens.shape
        param_dtype = self.q_proj.weight.dtype
        temporal_tokens = temporal_tokens.to(param_dtype)
        pose_conf = pose_confidence.to(param_dtype).clamp(min=1e-6, max=1.0)
        pose_mot = pose_motion.to(param_dtype)

        Q = 8
        queries = self.queries.to(param_dtype).unsqueeze(0).expand(B, -1, -1)  # [B, 8, H]

        # Project.
        q = self.q_proj(queries).view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(temporal_tokens).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(temporal_tokens).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        # Standard scaled dot product logits: [B, heads, Q, K].
        scale = 1.0 / (self.head_dim ** 0.5)
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H_heads, Q, K]

        # Bias by pose confidence/motion for each query's own articulator group.
        # confidence_bias[b, q, k] = lambda_c * log(pose_conf[b, k, group(q)])
        groups = self.query_groups.to(logits.device)  # [Q]
        # gather along the articulator dim: [B, K, 4] -> [B, K, Q]
        conf_bias = pose_conf.index_select(-1, groups)   # [B, K, Q]
        mot_bias = pose_mot.index_select(-1, groups)     # [B, K, Q]
        conf_bias = self.confidence_scale.to(param_dtype) * torch.log(conf_bias)
        mot_bias = self.motion_scale.to(param_dtype) * mot_bias
        bias = (conf_bias + mot_bias).transpose(1, 2).unsqueeze(1)  # [B, 1, Q, K]

        logits = logits + bias
        # Key padding mask: -inf at padded positions.
        pad_mask = (~segment_valid.to(torch.bool)).unsqueeze(1).unsqueeze(1)  # [B, 1, 1, K]
        logits = logits.masked_fill(pad_mask, float("-inf"))

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)  # [B, heads, Q, head_dim]
        context = context.transpose(1, 2).contiguous().view(B, Q, H)
        context = self.out_proj(context)
        context = self.dropout(context)

        # Residual + FFN.
        out = self.norm1(queries + context)
        out = self.norm2(out + self.ffn(out))
        return out  # [B, 8, H]
