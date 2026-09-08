"""Dataset variant for mpnet-alignment training.

Same visual banks as PhoenixCachedDataset, but the text side is a
precomputed mpnet sentence embedding per sample (npz from scripts/17).
The mBART text bank is not required (and not touched) for this path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pgat_length.data.dataset import _BankReader
from pgat_length.features.shards import K_MAX
from pgat_length.features.text_mpnet import load_mpnet_cache


class MpnetPhoenixDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        plans_root: Path,
        spatial_root: Path,
        motion_root: Path,
        mpnet_text_root: Path,
        manifest_path: Path,
        split: str,
    ) -> None:
        self.split = split
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

        cache_path = mpnet_text_root / f"mpnet_text_{split}.npz"
        cache_uids, embeddings, self.meta = load_mpnet_cache(cache_path)
        if set(cache_uids) != plan_uids:
            missing = sorted(plan_uids - set(cache_uids))[:3]
            raise RuntimeError(f"mpnet cache uids differ from plans; missing example={missing}")
        self._uid_to_row = {u: i for i, u in enumerate(cache_uids)}
        self._embeddings = embeddings
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
            "mpnet_embedding": self._embeddings[self._uid_to_row[uid]].copy(),
        }


MPNET_TENSOR_KEYS: tuple[str, ...] = (
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
    "mpnet_embedding",
)
MPNET_SCALAR_KEYS: tuple[str, ...] = ("k_temporal",)


def collate_mpnet_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("empty batch")
    batch: dict[str, Any] = {}
    for key in MPNET_TENSOR_KEYS:
        stacked = np.stack([s[key] for s in samples], axis=0)
        batch[key] = torch.from_numpy(stacked)
    for key in MPNET_SCALAR_KEYS:
        batch[key] = torch.tensor([s[key] for s in samples], dtype=torch.long)
    batch["uid"] = [s["uid"] for s in samples]
    batch["sample_id"] = [s["sample_id"] for s in samples]
    batch["reference"] = [s["reference"] for s in samples]
    return batch


__all__ = ["MpnetPhoenixDataset", "collate_mpnet_batch"]
