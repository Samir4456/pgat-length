"""Qwen-flavoured Dataset wrapping PhoenixCachedDataset for the QLoRA path.

Reads the same plans / spatial / motion / manifest banks as
PhoenixCachedDataset, but replaces the mBART text bank with the npz
cache produced by scripts/13_build_text_qwen.py:

    prompt_input_ids       [P] int32       (identical per sample)
    prompt_attention_mask  [P] bool
    target_input_ids       [max_target] int32
    target_attention_mask  [max_target] bool
    target_length          int

The visual prefix is not part of the dataset -- the model assembles
[prompt | prefix | target] at forward time (models/qwen_translation.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pgat_length.data.dataset import PhoenixCachedDataset
from pgat_length.features.shards import K_MAX


class QwenPhoenixDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        plans_root: Path,
        spatial_root: Path,
        motion_root: Path,
        qwen_text_root: Path,
        manifest_path: Path,
        split: str,
    ) -> None:
        self.split = split
        # Reuse the visual/manifest reader, but bypass its mBART text bank
        # by constructing a slim PhoenixCachedDataset-like reader ourselves.
        # We can't easily instantiate PhoenixCachedDataset without a text
        # bank, so we build only the parts we need.
        from pgat_length.data.dataset import _BankReader

        self.plans = _BankReader(plans_root, split, "plans")
        self.spatial = _BankReader(spatial_root, split, "spatial")
        self.motion = _BankReader(motion_root, split, "motion")

        plan_uids = set(self.plans.uids())
        for name, reader in (("spatial", self.spatial), ("motion", self.motion)):
            other = set(reader.uids())
            if plan_uids != other:
                raise RuntimeError(
                    f"{name} bank uids differ from plans; "
                    f"missing={sorted(plan_uids - other)[:3]}"
                )

        cache_path = qwen_text_root / f"qwen_text_{split}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Qwen text cache missing: {cache_path} "
                f"(run scripts/13_build_text_qwen.py --split {split}"
                f"{' --allow-test' if split == 'test' else ''})"
            )
        loaded = np.load(cache_path, allow_pickle=True)
        cache_uids = [str(u) for u in loaded["uids"]]
        if set(cache_uids) != plan_uids:
            missing = sorted(plan_uids - set(cache_uids))[:3]
            raise RuntimeError(
                f"Qwen text cache uids differ from plans; missing example={missing}"
            )
        self.prompt_input_ids = np.asarray(loaded["prompt_input_ids"], dtype=np.int32)
        self.prompt_attention_mask = np.asarray(loaded["prompt_attention_mask"], dtype=np.bool_)
        self._target_ids = np.asarray(loaded["target_input_ids"], dtype=np.int32)
        self._target_mask = np.asarray(loaded["target_attention_mask"], dtype=np.bool_)
        self._target_len = np.asarray(loaded["target_length"], dtype=np.int32)
        self._uid_to_row = {u: i for i, u in enumerate(cache_uids)}
        meta_raw = str(loaded["meta"])
        self.meta = json.loads(meta_raw) if meta_raw else {}
        self.uids: list[str] = sorted(plan_uids)

        manifest = pd.read_pickle(manifest_path)
        manifest = manifest.loc[manifest["split"].eq(split)].set_index("uid", drop=False)
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        uid = self.uids[index]
        plan = self.plans.get(uid)
        spatial = self.spatial.get(uid)
        motion = self.motion.get(uid)
        manifest_row = self.manifest.loc[uid]
        row = self._uid_to_row[uid]

        k_temporal = int(np.asarray(plan["k_temporal"]))
        segment_valid = np.zeros(K_MAX, dtype=np.bool_)
        segment_valid[:k_temporal] = True

        return {
            "uid": uid,
            "sample_id": str(manifest_row["sample_id"]),
            "reference": str(manifest_row["translation"]),
            "k_temporal": k_temporal,
            "pose_descriptor": np.asarray(plan["pose_descriptor"], dtype=np.float32),
            "pose_confidence": np.asarray(plan["pose_confidence"], dtype=np.float32),
            "pose_motion": np.asarray(plan["pose_motion"], dtype=np.float32),
            "segment_bounds": np.asarray(plan["segment_bounds"], dtype=np.int32),
            "anchor_positions": np.asarray(plan["anchor_positions"], dtype=np.int32),
            "segment_valid": segment_valid,
            "spatial_features": np.asarray(spatial["spatial_features"], dtype=np.float32),
            "spatial_valid": np.asarray(spatial["spatial_valid"], dtype=np.bool_),
            "motion_features": np.asarray(motion["motion_features"], dtype=np.float32),
            "motion_centers": np.asarray(motion["motion_centers"], dtype=np.float32),
            "prompt_input_ids": self.prompt_input_ids.copy(),
            "prompt_attention_mask": self.prompt_attention_mask.copy(),
            "target_input_ids": self._target_ids[row].copy(),
            "target_attention_mask": self._target_mask[row].copy(),
            "target_length": int(self._target_len[row]),
        }


QWEN_TENSOR_KEYS: tuple[str, ...] = (
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
    "prompt_input_ids",
    "prompt_attention_mask",
    "target_input_ids",
    "target_attention_mask",
)
QWEN_SCALAR_KEYS: tuple[str, ...] = ("k_temporal", "target_length")


def collate_qwen_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("empty batch")
    batch: dict[str, Any] = {}
    for key in QWEN_TENSOR_KEYS:
        stacked = np.stack([s[key] for s in samples], axis=0)
        batch[key] = torch.from_numpy(stacked)
    for key in QWEN_SCALAR_KEYS:
        batch[key] = torch.tensor([s[key] for s in samples], dtype=torch.long)
    batch["uid"] = [s["uid"] for s in samples]
    batch["sample_id"] = [s["sample_id"] for s in samples]
    batch["reference"] = [s["reference"] for s in samples]
    return batch


__all__ = ["QwenPhoenixDataset", "collate_qwen_batch"]
