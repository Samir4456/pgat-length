"""Read plan shards produced by scripts/01_build_plans.py.

Iterates plan safetensors shards in split order and yields per-sample dicts
that carry the real `k_temporal` plus the padded arrays. Padding is preserved
so downstream extractors can operate on the padded tensors and simply skip
segments beyond `k_temporal`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from safetensors import safe_open


@dataclass(frozen=True)
class PlanRecord:
    uid: str
    split: str
    shard: str
    offset: int
    source_num_frames: int
    k_temporal: int


class PlanBankReader:
    """Read plans from a fixed split under a plans root.

    The reader mmaps each shard exactly once and slices per-sample tensors.
    """

    def __init__(self, plan_root: Path, split: str) -> None:
        self.plan_root = plan_root
        self.split = split
        self.index_path = plan_root / split / "index.jsonl"
        if not self.index_path.is_file():
            raise FileNotFoundError(f"plan index missing: {self.index_path}")
        self.records: list[PlanRecord] = []
        with self.index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") != split:
                    raise ValueError(f"plan index split mismatch in {self.index_path}")
                self.records.append(
                    PlanRecord(
                        uid=str(row["uid"]),
                        split=split,
                        shard=str(row["shard"]),
                        offset=int(row["offset"]),
                        source_num_frames=int(row["source_num_frames"]),
                        k_temporal=int(row["k_temporal"]) if row.get("k_temporal") is not None else -1,
                    )
                )
        if not self.records:
            raise RuntimeError(f"empty plan index: {self.index_path}")
        self._shard_cache: dict[str, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def uids(self) -> list[str]:
        return [record.uid for record in self.records]

    def _load_shard(self, shard_name: str) -> dict[str, np.ndarray]:
        cached = self._shard_cache.get(shard_name)
        if cached is not None:
            return cached
        shard_path = self.plan_root / self.split / shard_name
        with safe_open(shard_path, framework="np") as handle:
            arrays = {name: handle.get_tensor(name) for name in handle.keys()}
        self._shard_cache[shard_name] = arrays
        return arrays

    def get(self, index: int) -> tuple[PlanRecord, dict[str, np.ndarray]]:
        record = self.records[index]
        arrays = self._load_shard(record.shard)
        plan = {name: arrays[name][record.offset] for name in arrays}
        # Reconcile k_temporal from the array (source of truth) with the index.
        stored_k = int(np.asarray(plan["k_temporal"]))
        if record.k_temporal >= 0 and stored_k != record.k_temporal:
            raise RuntimeError(
                f"k_temporal mismatch for {record.uid}: index={record.k_temporal} shard={stored_k}"
            )
        return record, plan

    def iter_range(self, start: int, stop: int) -> Iterator[tuple[PlanRecord, dict[str, np.ndarray]]]:
        for index in range(start, stop):
            yield self.get(index)

    def close(self) -> None:
        self._shard_cache.clear()
