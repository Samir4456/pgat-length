"""Batch collator for PhoenixCachedDataset.

All feature banks already store per-sample tensors padded to K_MAX (32),
so batching is a straight torch.stack — no per-batch length trimming here.
The tokenizer's transformer uses `segment_valid` as the key padding mask.

Text arrays are already padded to max_target_tokens by the text builder.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


BATCH_TENSOR_KEYS: tuple[str, ...] = (
    "pose_descriptor",
    "pose_confidence",
    "pose_motion",
    "segment_bounds",
    "anchor_positions",
    "segment_valid",
    "spatial_features",
    "spatial_valid",
    "motion_features",
    "motion_centers",
    "input_ids",
    "attention_mask",
)

BATCH_SCALAR_KEYS: tuple[str, ...] = ("k_temporal", "text_length")


def collate_alignment_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("empty batch")
    batch: dict[str, Any] = {}
    for key in BATCH_TENSOR_KEYS:
        stacked = np.stack([s[key] for s in samples], axis=0)
        batch[key] = torch.from_numpy(stacked)
    for key in BATCH_SCALAR_KEYS:
        batch[key] = torch.tensor([s[key] for s in samples], dtype=torch.long)
    # Metadata (lists, not tensors).
    batch["uid"] = [s["uid"] for s in samples]
    batch["sample_id"] = [s["sample_id"] for s in samples]
    batch["reference"] = [s["reference"] for s in samples]
    return batch


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor entries to device; leave list metadata alone."""
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved
