"""Projection from PGAT hidden dim (512) to mBART hidden dim (1024)."""

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
