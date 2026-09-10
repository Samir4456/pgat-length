"""Bidirectional selective state-space adapter (Mamba-flavored, pure PyTorch).

Mirror of the pgat-length adapter (src/pgat_length/models/ssm_adapter.py),
adapted for TSPNet's fairseq transformer encoder. Same math, same
sequential scan, no external CUDA-kernel dependency (no mamba-ssm).

Insertion point in TSPNet: TransformerEncoderSign.forward, immediately
after the 3 per-level features are concatenated along the sequence dim
and BEFORE the (T, B, C) transpose:

    x = torch.cat(x_lvls, dim=1)   # (B, T_concat, C)
    # -- adapter here --
    x = self.ssm_adapter(x, mask=(~encoder_padding_mask))
    # -----------------
    x = x.transpose(0, 1)          # (T, B, C) for fairseq

The adapter is opt-in via --use-ssm-adapter; when disabled the encoder
matches the published TSPNet baseline exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SSMAdapterConfig:
    d_model: int = 1024
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    num_layers: int = 1        # per direction (forward and backward each)
    dropout: float = 0.1
    dt_min: float = 0.001
    dt_max: float = 0.1
    residual_scale: float = 1.0


class _SelectiveSSMBlock(nn.Module):
    """One-directional selective SSM block (Mamba-flavored)."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)

        # Input projection -> (x_ssm, z_gate)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Depthwise 1D conv on x_ssm (padding=d_conv-1, trimmed to T after).
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # x -> (delta, B_param, C_param)
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + 2 * d_state, bias=False)

        # Softplus bias for delta so initial dt ~ log-uniform in [dt_min, dt_max].
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        # Inverse softplus so softplus(dt_bias) == dt at init.
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # A_log: per-channel, per-state, negative dynamics (S4D-lin init).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # Skip parameter.
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection back to d_model.
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,           # [B, T, D]
        mask: torch.Tensor | None = None,  # [B, T] bool; True = valid position
    ) -> torch.Tensor:
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype

        xz = self.in_proj(x)                          # [B, T, 2*d_inner]
        x_ssm, z = xz.chunk(2, dim=-1)                # each [B, T, d_inner]

        y_conv = x_ssm.transpose(1, 2)                # [B, d_inner, T]
        y_conv = self.conv1d(y_conv)[:, :, :T]        # trim padding
        y_conv = F.silu(y_conv.transpose(1, 2))       # [B, T, d_inner]

        x_dbl = self.x_proj(y_conv)                   # [B, T, d_inner + 2*d_state]
        delta, B_param, C_param = torch.split(
            x_dbl, [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(delta + self.dt_bias)      # [B, T, d_inner]

        A = -torch.exp(self.A_log.to(dtype))          # [d_inner, d_state]

        A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        B_bar = delta.unsqueeze(-1) * B_param.unsqueeze(-2)

        h = torch.zeros(B, self.d_inner, self.d_state, device=device, dtype=dtype)
        outputs = []
        y_input = y_conv.unsqueeze(-1)               # [B, T, d_inner, 1]
        for t in range(T):
            h = A_bar[:, t] * h + B_bar[:, t] * y_input[:, t]
            y_t = torch.einsum("bds,bs->bd", h, C_param[:, t])
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)              # [B, T, d_inner]

        y = y + self.D.to(dtype) * y_conv
        y = y * F.silu(z)
        out = self.out_proj(y)                       # [B, T, d_model]

        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


class BidirectionalSSMAdapter(nn.Module):
    """Bidirectional SSM adapter with residual + LayerNorm.

    Applies num_layers forward blocks and num_layers backward blocks (reversed
    input), sums the two directions, applies dropout, and adds a residual to
    the original tokens before a final LayerNorm.
    """

    def __init__(self, config: SSMAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_layers = nn.ModuleList([
            _SelectiveSSMBlock(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                dt_min=config.dt_min,
                dt_max=config.dt_max,
            )
            for _ in range(config.num_layers)
        ])
        self.backward_layers = nn.ModuleList([
            _SelectiveSSMBlock(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                dt_min=config.dt_min,
                dt_max=config.dt_max,
            )
            for _ in range(config.num_layers)
        ])
        self.pre_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.post_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,           # [B, T, d_model]
        mask: torch.Tensor | None = None,  # [B, T] bool; True=valid
    ) -> torch.Tensor:
        normed = self.pre_norm(x)

        fwd = normed
        for layer in self.forward_layers:
            fwd = layer(fwd, mask=mask)

        bwd_input = torch.flip(normed, dims=[1])
        bwd_mask = torch.flip(mask, dims=[1]) if mask is not None else None
        bwd = bwd_input
        for layer in self.backward_layers:
            bwd = layer(bwd, mask=bwd_mask)
        bwd = torch.flip(bwd, dims=[1])

        combined = fwd + bwd
        combined = self.dropout(combined)

        out = self.post_norm(x + self.config.residual_scale * combined)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out
