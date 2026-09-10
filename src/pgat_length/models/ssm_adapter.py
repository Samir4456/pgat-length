"""Bidirectional selective state-space adapter (Mamba-flavored, pure PyTorch).

Purpose: insert as a small module between the PGAT temporal tokenizer's output
and the downstream articulator / global-summary heads (see
`models/translation.py`). Adds long-range temporal context via selective SSM
dynamics without changing the rest of the pipeline. Kept in pure PyTorch so
there is no `mamba-ssm` CUDA-kernel build dependency; sequences are short
(K in [12, 32]) so the sequential scan is negligible overhead.

Design summary (per direction):
    x [B, T, D]
        |-> in_proj -> (x_ssm, z_gate)  each [B, T, d_inner]
        |-> depthwise conv on x_ssm (mixes short-range context, matching Mamba)
        |-> SiLU
        |-> x_proj -> (delta, B_param, C_param)
        |-> discretize A_bar = exp(delta * A_diag);  B_bar = delta * B_param
        |-> sequential scan h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
        |                   y_t = C_param_t . h_t
        |-> y = y + D * x_ssm     (skip)
        |-> y = y * SiLU(z_gate)  (gating)
        |-> out_proj -> [B, T, D]

Bidirectional wrapper runs the block forwards and backwards over the same
input, sums the two outputs, dropouts and layer-norms with a residual to the
original tokens.

Reference: Mamba (Gu & Dao, 2023) simplified without the parallel-scan CUDA
kernel, plus S4D-lin initialization for A.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SSMAdapterConfig:
    d_model: int = 512
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    num_layers: int = 1        # per direction
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

        # Depthwise 1D conv on x_ssm (padding=d_conv-1, we trim to T after).
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # x -> (delta, B_param, C_param)
        # delta: [d_inner], B/C: [d_state]
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + 2 * d_state, bias=False)

        # Softplus bias for delta so initial dt ~ log-uniform in [dt_min, dt_max].
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        # Inverse softplus so softplus(dt_bias) == dt at init.
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # A_log: per-channel, per-state, negative dynamics.
        # S4D-lin init: A[i, n] = -(n+1)  for n = 0..d_state-1.
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

        # Input projection.
        xz = self.in_proj(x)                          # [B, T, 2*d_inner]
        x_ssm, z = xz.chunk(2, dim=-1)                # each [B, T, d_inner]

        # Depthwise conv on x_ssm. Trim padding.
        y_conv = x_ssm.transpose(1, 2)                # [B, d_inner, T]
        y_conv = self.conv1d(y_conv)[:, :, :T]        # trim padding
        y_conv = F.silu(y_conv.transpose(1, 2))       # [B, T, d_inner]

        # Compute per-timestep SSM parameters.
        x_dbl = self.x_proj(y_conv)                   # [B, T, d_inner + 2*d_state]
        delta, B_param, C_param = torch.split(
            x_dbl, [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(delta + self.dt_bias)      # [B, T, d_inner]

        # A: negative diagonal per channel.
        A = -torch.exp(self.A_log.to(dtype))          # [d_inner, d_state]

        # Discretization (ZOH-like, first-order).
        # A_bar: [B, T, d_inner, d_state]
        A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # B_bar: [B, T, d_inner, d_state]
        B_bar = delta.unsqueeze(-1) * B_param.unsqueeze(-2)

        # Sequential scan.  T is small (<=32) so a Python loop is fine.
        # h: [B, d_inner, d_state]
        h = torch.zeros(B, self.d_inner, self.d_state, device=device, dtype=dtype)
        outputs = []
        # y_conv has shape [B, T, d_inner]; unsqueeze last dim for broadcast with h.
        y_input = y_conv.unsqueeze(-1)               # [B, T, d_inner, 1]
        for t in range(T):
            h = A_bar[:, t] * h + B_bar[:, t] * y_input[:, t]  # [B, d_inner, d_state]
            # Output: C_param[t] dot h across d_state.
            y_t = torch.einsum("bds,bs->bd", h, C_param[:, t])  # [B, d_inner]
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)              # [B, T, d_inner]

        # Skip term.
        y = y + self.D.to(dtype) * y_conv

        # Gating.
        y = y * F.silu(z)

        # Output projection.
        out = self.out_proj(y)                       # [B, T, d_model]

        # Zero-out padded positions in the output so they can't leak into the
        # residual sum done in the wrapper.
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


class BidirectionalSSMAdapter(nn.Module):
    """Bidirectional SSM adapter with residual + LayerNorm.

    Applies num_layers forward blocks and num_layers backward blocks (reversed
    input), sums the two directions, applies dropout, and adds a residual to
    the *original* tokens before a final LayerNorm.

    Padded positions in the input (mask==False) contribute zero to both
    directions and are zeroed in the output.
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
        mask: torch.Tensor | None = None,  # [B, T] bool
    ) -> torch.Tensor:
        # LayerNorm before the SSMs (standard pre-norm structure).
        normed = self.pre_norm(x)

        # Forward direction.
        fwd = normed
        for layer in self.forward_layers:
            fwd = layer(fwd, mask=mask)

        # Backward direction: flip along time, run, flip output back.
        # For padded (right-padded) sequences, flip moves padding to the start.
        # Zeroed padded inputs contribute zero state update; masking the output
        # again ensures we do not carry over any spurious values.
        bwd_input = torch.flip(normed, dims=[1])
        bwd_mask = torch.flip(mask, dims=[1]) if mask is not None else None
        bwd = bwd_input
        for layer in self.backward_layers:
            bwd = layer(bwd, mask=bwd_mask)
        bwd = torch.flip(bwd, dims=[1])

        combined = fwd + bwd
        combined = self.dropout(combined)

        # Residual + post-norm, scaled by residual_scale (default 1.0).
        out = self.post_norm(x + self.config.residual_scale * combined)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


def build_ssm_adapter(cfg: dict | None) -> BidirectionalSSMAdapter | None:
    """Instantiate the adapter from a config dict; return None if disabled."""
    if not cfg:
        return None
    if not bool(cfg.get("enabled", True)):
        return None
    return BidirectionalSSMAdapter(SSMAdapterConfig(
        d_model=int(cfg["d_model"]),
        d_state=int(cfg.get("d_state", 16)),
        d_conv=int(cfg.get("d_conv", 4)),
        expand=int(cfg.get("expand", 2)),
        num_layers=int(cfg.get("num_layers", 1)),
        dropout=float(cfg.get("dropout", 0.1)),
        dt_min=float(cfg.get("dt_min", 0.001)),
        dt_max=float(cfg.get("dt_max", 0.1)),
        residual_scale=float(cfg.get("residual_scale", 1.0)),
    ))
