"""Cached-feature Dataset that reads all four PGAT-length banks by uid.

Given a plans root, spatial root, motion root, and text root plus a split,
yields per-sample dicts:
    {
      "uid": str, "sample_id": str, "k_temporal": int, "reference": str,
      "spatial_features": [K_MAX, 4, 768] fp32,
      "spatial_valid":    [K_MAX, 4] bool,
      "motion_features":  [8, 768] fp32,
      "motion_centers":   [8] fp32,
      "pose_descriptor":  [K_MAX, 64] fp32,
      "pose_confidence":  [K_MAX, 4] fp32,
      "pose_motion":      [K_MAX, 4] fp32,
      "segment_bounds":   [K_MAX+1] int32,
      "anchor_positions": [K_MAX] int32,
      "segment_valid":    [K_MAX] bool,   # first k_temporal True, rest False
      "input_ids":        [max_target_tokens] int32,
      "attention_mask":   [max_target_tokens] bool,
      "text_length":      int,
    }

Segment_valid is derived from k_temporal: positions in [0, K) are valid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open

from pgat_length.features.shards import K_MAX


class _BankReader:
    """Read per-uid rows from a bank's shards via its index.jsonl."""

    def __init__(self, root: Path, split: str, expected_bank: str) -> None:
        self.root = root
        self.split = split
        self.expected_bank = expected_bank
        self.index_path = root / split / "index.jsonl"
        if not self.index_path.is_file():
            raise FileNotFoundError(f"{expected_bank} index missing: {self.index_path}")
        self.by_uid: dict[str, dict[str, Any]] = {}
        with self.index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                self.by_uid[str(row["uid"])] = row
        self._shard_cache: dict[str, dict[str, np.ndarray]] = {}

    def _load_shard(self, shard_name: str) -> dict[str, np.ndarray]:
        cached = self._shard_cache.get(shard_name)
        if cached is not None:
            return cached
        shard_path = self.root / self.split / shard_name
        with safe_open(shard_path, framework="np") as handle:
            arrays = {name: handle.get_tensor(name) for name in handle.keys()}
        self._shard_cache[shard_name] = arrays
        return arrays

    def get(self, uid: str) -> dict[str, np.ndarray]:
        entry = self.by_uid.get(uid)
        if entry is None:
            raise KeyError(f"{self.expected_bank}: uid {uid!r} not in index")
        arrays = self._load_shard(str(entry["shard"]))
        offset = int(entry["offset"])
        return {name: arrays[name][offset] for name in arrays}

    def uids(self) -> list[str]:
        return list(self.by_uid.keys())


class PhoenixCachedDataset(torch.utils.data.Dataset):
    """Dataset reading all four PGAT-length feature banks for one split."""

    def __init__(
        self,
        plans_root: Path,
        spatial_root: Path,
        motion_root: Path,
        text_root: Path,
        manifest_path: Path,
        split: str,
    ) -> None:
        self.split = split
        self.plans = _BankReader(plans_root, split, "plans")
        self.spatial = _BankReader(spatial_root, split, "spatial")
        self.motion = _BankReader(motion_root, split, "motion")
        self.text = _BankReader(text_root, split, "text")

        # UIDs must be identical across all four banks for the requested split.
        plan_uids = set(self.plans.uids())
        for name, reader in (
            ("spatial", self.spatial),
            ("motion", self.motion),
            ("text", self.text),
        ):
            other = set(reader.uids())
            if plan_uids != other:
                raise RuntimeError(
                    f"{name} bank uids differ from plans; "
                    f"missing={sorted(plan_uids - other)[:3]}"
                )
        self.uids: list[str] = sorted(plan_uids)

        # Manifest for reference text and sample_id metadata.
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
        text = self.text.get(uid)
        manifest_row = self.manifest.loc[uid]

        k_temporal = int(np.asarray(plan["k_temporal"]))
        # Derive segment_valid: first k_temporal positions True, rest False.
        segment_valid = np.zeros(K_MAX, dtype=np.bool_)
        segment_valid[:k_temporal] = True

        return {
            "uid": uid,
            "sample_id": str(manifest_row["sample_id"]),
            "reference": str(manifest_row["translation"]),
            "k_temporal": k_temporal,
            # Plan arrays.
            "pose_descriptor": np.asarray(plan["pose_descriptor"], dtype=np.float32),
            "pose_confidence": np.asarray(plan["pose_confidence"], dtype=np.float32),
            "pose_motion": np.asarray(plan["pose_motion"], dtype=np.float32),
            "segment_bounds": np.asarray(plan["segment_bounds"], dtype=np.int32),
            "anchor_positions": np.asarray(plan["anchor_positions"], dtype=np.int32),
            "segment_valid": segment_valid,
            # Spatial + motion.
            "spatial_features": np.asarray(spatial["spatial_features"], dtype=np.float32),
            "spatial_valid": np.asarray(spatial["spatial_valid"], dtype=np.bool_),
            "motion_features": np.asarray(motion["motion_features"], dtype=np.float32),
            "motion_centers": np.asarray(motion["motion_centers"], dtype=np.float32),
            # Text.
            "input_ids": np.asarray(text["input_ids"], dtype=np.int32),
            "attention_mask": np.asarray(text["attention_mask"], dtype=np.bool_),
            "text_length": int(np.asarray(text["text_length"])),
        }
