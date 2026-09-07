"""Best-only checkpoint helpers.

Storage rules (100 GB quota):
- Save best.pt only. Never write last.pt.
- Only save trainable parameters (frozen mBART/DINOv2 base weights stay in
  the HF cache, deduplicated across runs).
- Atomic write via temporary file plus os.replace.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


def trainable_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return only parameters with requires_grad=True."""
    return {
        name: p.detach().cpu().clone()
        for name, p in module.named_parameters()
        if p.requires_grad
    }


def save_best(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_best(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)
