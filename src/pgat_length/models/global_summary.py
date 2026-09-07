"""Global summary attention (4 learned queries).

Standard multi-head attention: 4 learned global queries attend into the
K encoded temporal tokens with segment_valid as key padding mask. Residual
+ LayerNorm + feed-forward.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GlobalSummaryAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_queries: int = 4,
        num_heads: int = 8,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
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

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim, bias=True),
        )

    def forward(
        self,
        temporal_tokens: torch.Tensor,   # [B, K_MAX, H]
        segment_valid: torch.Tensor,     # [B, K_MAX] bool
    ) -> torch.Tensor:
        B, K, H = temporal_tokens.shape
        Q = self.queries.shape[0]
        param_dtype = self.q_proj.weight.dtype
        temporal_tokens = temporal_tokens.to(param_dtype)

        queries = self.queries.to(param_dtype).unsqueeze(0).expand(B, -1, -1)

        q = self.q_proj(queries).view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(temporal_tokens).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(temporal_tokens).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / (self.head_dim ** 0.5)
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale
        pad_mask = (~segment_valid.to(torch.bool)).unsqueeze(1).unsqueeze(1)
        logits = logits.masked_fill(pad_mask, float("-inf"))
        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Q, H)
        context = self.out_proj(context)
        context = self.dropout(context)

        out = self.norm1(queries + context)
        out = self.norm2(out + self.ffn(out))
        return out  # [B, 4, H]
