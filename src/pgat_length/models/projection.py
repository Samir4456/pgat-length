"""Projection from PGAT hidden dim (512) to a decoder hidden dim.

Both mBART (1024) and Qwen2.5-3B (2048) reuse the same shape: LayerNorm on
the PGAT feature dim, then a single linear projection to the decoder dim.
The projection preserves the variable-length prefix axis (any T).
"""

from __future__ import annotations

import torch
from torch import nn


class PgatMbartProjection(nn.Module):
    def __init__(self, pgat_dim: int = 512, mbart_dim: int = 1024, layernorm: bool = True, bias: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(pgat_dim) if layernorm else nn.Identity()
        self.linear = nn.Linear(pgat_dim, mbart_dim, bias=bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, pgat_dim] -> [B, T, mbart_dim]
        return self.linear(self.norm(tokens))


class PgatQwenProjection(nn.Module):
    """PGAT -> Qwen hidden dim (Qwen2.5-3B = 2048)."""

    def __init__(self, pgat_dim: int = 512, qwen_dim: int = 2048, layernorm: bool = True, bias: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(pgat_dim) if layernorm else nn.Identity()
        self.linear = nn.Linear(pgat_dim, qwen_dim, bias=bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, pgat_dim] -> [B, T, qwen_dim]; T is variable across samples.
        return self.linear(self.norm(tokens))
