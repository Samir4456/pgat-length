"""Cache storage contracts for pgat-length.

Adapts the PGAT-v1 shard machinery to variable K_temporal per sample.

Design:
- POSITIONS is fixed at 64 (deterministic source-frame count).
- K_MAX is the padded temporal dimension for storage (default 40).
- Each stored plan sample carries a scalar `k_temporal` giving its real K.
- Arrays sized on the temporal axis are padded to K_MAX with zeros /
  bounds sentinel; downstream readers must slice to k_temporal.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


POSITIONS: int = 64
K_MAX: int = 40

# Per-sample tail shapes. Batched arrays have leading dim = sample_count.
PLAN_ARRAY_CONTRACTS: dict[str, tuple[tuple[int, ...], str]] = {
    "frame_indices":     ((POSITIONS,),           "int32"),
    "frame_valid":       ((POSITIONS,),           "bool"),
    "k_temporal":        ((),                     "int16"),
    "segment_bounds":    ((K_MAX + 1,),           "int16"),
    "anchor_positions":  ((K_MAX,),               "int16"),
    "crop_boxes":        ((K_MAX, 3, 4),          "int16"),
    "region_valid":      ((K_MAX, 3),             "bool"),
    "pose_descriptor":   ((K_MAX, 64),            "float16"),
    "pose_confidence":   ((K_MAX, 4),             "float16"),
    "pose_motion":       ((K_MAX, 4),             "float16"),
}

FEATURE_ARRAY_CONTRACTS: dict[str, dict[str, tuple[tuple[int, ...], str]]] = {
    "spatial": {
        "spatial_features": ((K_MAX, 4, 768), "float16"),
        "spatial_valid":    ((K_MAX, 4),      "bool"),
    },
    "motion": {
        "motion_features": ((8, 768), "float16"),
        "motion_centers": ((8,),      "float16"),
    },
}


@dataclass(frozen=True)
class CacheProjection:
    plans_bytes: int
    spatial_bytes: int
    motion_bytes: int


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def uid_order_fingerprint(uids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for uid in uids:
        encoded = str(uid).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _validate_plan_arrays(arrays: Mapping[str, Any], sample_count: int) -> None:
    import numpy as np

    missing = sorted(set(PLAN_ARRAY_CONTRACTS).difference(arrays))
    extra = sorted(set(arrays).difference(PLAN_ARRAY_CONTRACTS))
    if missing or extra:
        raise ValueError(f"plan keys mismatch; missing={missing}, extra={extra}")
    for name, (tail_shape, dtype_name) in PLAN_ARRAY_CONTRACTS.items():
        value = np.asarray(arrays[name])
        expected_shape = (sample_count, *tail_shape)
        if value.shape != expected_shape:
            raise ValueError(f"{name} shape must be {expected_shape}, got {value.shape}")
        if value.dtype.name != dtype_name:
            raise ValueError(f"{name} dtype must be {dtype_name}, got {value.dtype.name}")

    k_temporal = np.asarray(arrays["k_temporal"]).astype(np.int64)
    if np.any(k_temporal < 1) or np.any(k_temporal > K_MAX):
        raise ValueError(f"k_temporal must lie in [1, {K_MAX}]")
    bounds = np.asarray(arrays["segment_bounds"]).astype(np.int64)
    anchors = np.asarray(arrays["anchor_positions"]).astype(np.int64)
    for index in range(sample_count):
        real_k = int(k_temporal[index])
        real_bounds = bounds[index, : real_k + 1]
        if real_bounds[0] != 0 or real_bounds[-1] != POSITIONS:
            raise ValueError(
                f"sample {index}: bounds must start at 0 and end at {POSITIONS}"
            )
        if np.any(np.diff(real_bounds) < 2):
            raise ValueError(
                f"sample {index}: every real segment must contain at least two positions"
            )
        real_anchors = anchors[index, :real_k]
        if np.any(real_anchors < real_bounds[:-1]) or np.any(real_anchors >= real_bounds[1:]):
            raise ValueError(f"sample {index}: anchors must lie inside their segments")


def write_plan_shard(
    output_root: Path,
    split: str,
    shard_number: int,
    uids: Sequence[str],
    arrays: Mapping[str, Any],
    config_fingerprint: str,
    uid_fingerprint: str,
    overwrite: bool = False,
) -> tuple[Path, bool]:
    """Atomically save a validated plan shard.

    Returns ``(path, created)``. A matching complete shard is reused
    unless ``overwrite`` is true. Existing mismatched shards are never
    silently used.
    """

    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if not split or any(character in split for character in "\\/:"):
        raise ValueError(f"unsafe split name: {split!r}")
    if shard_number < 0:
        raise ValueError("shard_number must be non-negative")
    if not uids or len(set(uids)) != len(uids):
        raise ValueError("uids must be non-empty and unique within a shard")
    _validate_plan_arrays(arrays, len(uids))

    split_root = output_root / split
    split_root.mkdir(parents=True, exist_ok=True)
    shard_path = split_root / f"plans-{shard_number:05d}.safetensors"
    metadata = {
        "format_version": "2",
        "bank": "pgat_length_plans",
        "split": split,
        "sample_count": str(len(uids)),
        "k_max": str(K_MAX),
        "positions": str(POSITIONS),
        "config_fingerprint": config_fingerprint,
        "uid_fingerprint": uid_fingerprint,
    }
    if shard_path.exists() and not overwrite:
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            existing = handle.metadata() or {}
            keys = set(handle.keys())
        if existing == metadata and keys == set(PLAN_ARRAY_CONTRACTS):
            return shard_path, False
        raise FileExistsError(
            f"Existing shard does not match the requested fingerprints: {shard_path}"
        )

    tensors = {
        name: torch.from_numpy(np.ascontiguousarray(np.asarray(arrays[name])))
        for name in PLAN_ARRAY_CONTRACTS
    }
    temporary_path = shard_path.with_name(shard_path.name + ".partial")
    save_file(tensors, temporary_path, metadata=metadata)
    validate_saved_plan_shard(temporary_path, len(uids), metadata)
    os.replace(temporary_path, shard_path)
    return shard_path, True


def validate_saved_plan_shard(
    shard_path: Path,
    expected_samples: int | None = None,
    expected_metadata: Mapping[str, str] | None = None,
) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(shard_path, framework="np") as handle:
        arrays = {name: handle.get_tensor(name) for name in handle.keys()}
        metadata = handle.metadata() or {}
    sample_count = expected_samples
    if sample_count is None:
        sample_count = int(metadata.get("sample_count", -1))
    _validate_plan_arrays(arrays, sample_count)
    if expected_metadata is not None and metadata != dict(expected_metadata):
        raise ValueError(f"shard metadata mismatch: {shard_path}")
    return metadata


def _validate_feature_arrays(bank: str, arrays: Mapping[str, Any], sample_count: int) -> None:
    import numpy as np

    if bank not in FEATURE_ARRAY_CONTRACTS:
        raise ValueError(f"Unsupported feature bank: {bank}")
    contracts = FEATURE_ARRAY_CONTRACTS[bank]
    missing = sorted(set(contracts).difference(arrays))
    extra = sorted(set(arrays).difference(contracts))
    if missing or extra:
        raise ValueError(f"{bank} keys mismatch; missing={missing}, extra={extra}")
    for name, (tail_shape, dtype_name) in contracts.items():
        value = np.asarray(arrays[name])
        expected_shape = (sample_count, *tail_shape)
        if value.shape != expected_shape:
            raise ValueError(f"{name} shape must be {expected_shape}, got {value.shape}")
        if value.dtype.name != dtype_name:
            raise ValueError(f"{name} dtype must be {dtype_name}, got {value.dtype.name}")
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    if bank == "spatial":
        features = np.asarray(arrays["spatial_features"])
        valid = np.asarray(arrays["spatial_valid"])
        if np.any(features[~valid] != 0):
            raise ValueError("Invalid spatial views must contain exact zeros")
    elif bank == "motion":
        centers = np.asarray(arrays["motion_centers"], dtype=np.float32)
        if np.any(centers < 0) or np.any(centers > 1):
            raise ValueError("Motion centers must be normalized to [0, 1]")
        if np.any(np.diff(centers, axis=1) <= 0):
            raise ValueError("Motion centers must be strictly increasing")


def validate_saved_feature_shard(
    shard_path: Path,
    bank: str,
    expected_samples: int | None = None,
    expected_metadata: Mapping[str, str] | None = None,
) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(shard_path, framework="np") as handle:
        arrays = {name: handle.get_tensor(name) for name in handle.keys()}
        metadata = handle.metadata() or {}
    sample_count = expected_samples
    if sample_count is None:
        sample_count = int(metadata.get("sample_count", -1))
    _validate_feature_arrays(bank, arrays, sample_count)
    if expected_metadata is not None and metadata != dict(expected_metadata):
        raise ValueError(f"feature shard metadata mismatch: {shard_path}")
    return metadata


def write_feature_shard(
    output_root: Path,
    bank: str,
    split: str,
    shard_number: int,
    uids: Sequence[str],
    arrays: Mapping[str, Any],
    config_fingerprint: str,
    plan_fingerprint: str,
    model_name: str,
    overwrite: bool = False,
) -> tuple[Path, bool]:
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if bank not in FEATURE_ARRAY_CONTRACTS:
        raise ValueError(f"Unsupported feature bank: {bank}")
    if not split or any(character in split for character in "\\/:"):
        raise ValueError(f"unsafe split name: {split!r}")
    if shard_number < 0:
        raise ValueError("shard_number must be non-negative")
    if not uids or len(set(uids)) != len(uids):
        raise ValueError("uids must be non-empty and unique within a shard")
    _validate_feature_arrays(bank, arrays, len(uids))

    split_root = output_root / split
    split_root.mkdir(parents=True, exist_ok=True)
    shard_path = split_root / f"{bank}-{shard_number:05d}.safetensors"
    metadata = {
        "format_version": "2",
        "bank": bank,
        "split": split,
        "sample_count": str(len(uids)),
        "k_max": str(K_MAX),
        "config_fingerprint": config_fingerprint,
        "plan_fingerprint": plan_fingerprint,
        "uid_fingerprint": uid_order_fingerprint(uids),
        "model_name": model_name,
    }
    if shard_path.exists() and not overwrite:
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            existing = handle.metadata() or {}
            keys = set(handle.keys())
        if existing == metadata and keys == set(FEATURE_ARRAY_CONTRACTS[bank]):
            validate_saved_feature_shard(shard_path, bank, len(uids), metadata)
            return shard_path, False
        raise FileExistsError(f"existing feature shard does not match request: {shard_path}")

    tensors = {
        name: torch.from_numpy(np.ascontiguousarray(np.asarray(arrays[name])))
        for name in FEATURE_ARRAY_CONTRACTS[bank]
    }
    temporary_path = shard_path.with_name(shard_path.name + ".partial")
    save_file(tensors, temporary_path, metadata=metadata)
    validate_saved_feature_shard(temporary_path, bank, len(uids), metadata)
    os.replace(temporary_path, shard_path)
    return shard_path, True


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write a UTF-8 JSONL file through a sibling .partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".partial")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)
