"""Step 02 — DINOv2 spatial feature extraction with variable K_temporal.

Usage (via sbatch on ASL):
    sbatch slurm/build_features.sbatch scripts/02_extract_spatial.py \\
        --split train --allow-full
    sbatch slurm/build_features.sbatch scripts/02_extract_spatial.py \\
        --split dev --allow-full

Smoke (local, tiny):
    python scripts/02_extract_spatial.py --split dev --limit 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pgat_length.features.dino_extractor import Dinov2SpatialExtractor  # noqa: E402
from pgat_length.features.frame_paths import (  # noqa: E402
    list_frame_files,
    resolve_frame_directory,
)
from pgat_length.features.plan_reader import PlanBankReader  # noqa: E402
from pgat_length.features.shards import (  # noqa: E402
    K_MAX,
    sha256_fingerprint,
    uid_order_fingerprint,
    validate_saved_feature_shard,
    write_feature_shard,
    write_jsonl_atomic,
)


def _expand(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return payload


def project_or_absolute(root: Path, value: str) -> Path:
    expanded = Path(_expand(str(value)))
    return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()


def _write_bank_metadata(
    output_root: Path,
    model_name: str,
    fingerprint: str,
    plan_fingerprint: str,
    combined_config: dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "metadata.json"
    payload = {
        "format_version": 2,
        "bank": "spatial",
        "k_max": K_MAX,
        "model_name": model_name,
        "config_fingerprint": fingerprint,
        "plan_fingerprint": plan_fingerprint,
        "config": combined_config,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key in ("bank", "model_name", "config_fingerprint", "plan_fingerprint", "k_max"):
            if existing.get(key) != payload[key]:
                raise RuntimeError(f"spatial bank metadata mismatch on {key}: {path}")
        return
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_existing_index(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.exists():
        return {}
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                entries[(str(row["shard"]), int(row["offset"]))] = row
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, default=PROJECT_ROOT / "configs" / "data.yaml")
    parser.add_argument("--features-config", type=Path, default=PROJECT_ROOT / "configs" / "features.yaml")
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--allow-full", action="store_true", help="Process the complete split.")
    scope.add_argument("--limit", type=int, help="Smoke sample count (default: 3).")
    parser.add_argument("--plan-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    data_cfg = load_yaml(args.data_config.resolve())
    features_cfg = load_yaml(args.features_config.resolve())
    combined = {"data": data_cfg, "features": features_cfg}
    config_fingerprint = sha256_fingerprint(combined)

    feature_root = project_or_absolute(PROJECT_ROOT, data_cfg["paths"]["feature_root"])
    hf_cache = project_or_absolute(PROJECT_ROOT, data_cfg["paths"]["hf_cache"])
    manifest_path = project_or_absolute(PROJECT_ROOT, data_cfg["paths"]["manifest"])
    frames_root = project_or_absolute(PROJECT_ROOT, data_cfg["paths"]["raw_frames_root"])

    plan_root = (
        args.plan_root.resolve()
        if args.plan_root is not None
        else feature_root / ("plans" if args.allow_full else "plans_smoke")
    )
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else feature_root / ("spatial" if "smoke" not in plan_root.name else "spatial_smoke")
    )

    plan_meta_path = plan_root / "metadata.json"
    if not plan_meta_path.is_file():
        raise FileNotFoundError(f"plan metadata missing (run step 01 first): {plan_meta_path}")
    plan_metadata = json.loads(plan_meta_path.read_text(encoding="utf-8"))
    plan_fingerprint = str(plan_metadata.get("config_fingerprint", ""))
    if plan_fingerprint != config_fingerprint:
        raise RuntimeError(
            "Plan fingerprint does not match current data+features config. "
            "Rebuild plans (scripts/01) with matching configs."
        )

    reader = PlanBankReader(plan_root, args.split)
    manifest = pd.read_pickle(manifest_path).set_index("uid", drop=False)

    expected = int(data_cfg["splits"][f"{args.split}_size"])
    if len(reader) == expected and not args.allow_full:
        raise RuntimeError(
            f"refusing full-split extraction ({len(reader)} samples) without --allow-full"
        )
    if args.limit is not None and args.limit > 0:
        reader.records = reader.records[: args.limit]

    model_name = str(features_cfg["spatial"]["model_name"])
    feature_dim = int(features_cfg["spatial"]["feature_dim"])
    shard_size = int(features_cfg["storage"]["shard_size"])

    _write_bank_metadata(output_root, model_name, config_fingerprint, plan_fingerprint, combined)

    index_path = output_root / args.split / "index.jsonl"
    existing_index = _load_existing_index(index_path)
    output_index: list[dict[str, Any]] = []
    created_shards = 0
    reused_shards = 0
    started = time.monotonic()

    extractor: Dinov2SpatialExtractor | None = None
    try:
        for shard_number, start in enumerate(range(0, len(reader), shard_size)):
            stop = min(start + shard_size, len(reader))
            records = reader.records[start:stop]
            uids = [record.uid for record in records]
            shard_path = output_root / args.split / f"spatial-{shard_number:05d}.safetensors"
            expected_metadata = {
                "format_version": "2",
                "bank": "spatial",
                "split": args.split,
                "sample_count": str(len(uids)),
                "k_max": str(K_MAX),
                "config_fingerprint": config_fingerprint,
                "plan_fingerprint": plan_fingerprint,
                "uid_fingerprint": uid_order_fingerprint(uids),
                "model_name": model_name,
            }
            if shard_path.exists() and not args.overwrite:
                validate_saved_feature_shard(shard_path, "spatial", len(records), expected_metadata)
                for offset, record in enumerate(records):
                    previous = existing_index.get((shard_path.name, offset))
                    output_index.append(
                        previous
                        if previous is not None and previous.get("uid") == record.uid
                        else {
                            "uid": record.uid,
                            "split": args.split,
                            "shard": shard_path.name,
                            "offset": offset,
                            "bank": "spatial",
                            "k_temporal": record.k_temporal,
                            "model_name": model_name,
                            "config_fingerprint": config_fingerprint,
                            "plan_fingerprint": plan_fingerprint,
                        }
                    )
                reused_shards += 1
                print(f"Reused: {shard_path}")
                continue

            if extractor is None:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                extractor = Dinov2SpatialExtractor(
                    model_name,
                    feature_dim=feature_dim,
                    batch_size=args.batch_size,
                    hf_cache=hf_cache,
                )

            sample_arrays: list[dict[str, np.ndarray]] = []
            for reader_index in range(start, stop):
                record, plan = reader.get(reader_index)
                if record.uid not in manifest.index:
                    raise KeyError(f"plan uid absent from manifest: {record.uid}")
                manifest_row = manifest.loc[record.uid]
                frame_directory, _ = resolve_frame_directory(
                    frames_root, args.split, str(manifest_row["sample_id"])
                )
                frame_files = list_frame_files(frame_directory)
                if len(frame_files) != int(manifest_row["source_num_frames"]):
                    raise RuntimeError(f"frame count mismatch for {record.uid}")
                result = extractor.extract(frame_files, plan)
                sample_arrays.append(
                    {
                        "spatial_features": result.features,
                        "spatial_valid": result.valid,
                    }
                )
                elapsed = max(time.monotonic() - started, 1e-6)
                completed = reader_index + 1
                print(
                    f"[{completed}/{len(reader)}] {record.uid} "
                    f"K={result.k_temporal:2d} rate={completed / elapsed:.3f} samples/s",
                    flush=True,
                )
            arrays = {
                name: np.stack([sample[name] for sample in sample_arrays])
                for name in sample_arrays[0]
            }
            shard_path, created = write_feature_shard(
                output_root=output_root,
                bank="spatial",
                split=args.split,
                shard_number=shard_number,
                uids=uids,
                arrays=arrays,
                config_fingerprint=config_fingerprint,
                plan_fingerprint=plan_fingerprint,
                model_name=model_name,
                overwrite=args.overwrite,
            )
            created_shards += int(created)
            reused_shards += int(not created)
            for offset, record in enumerate(records):
                output_index.append(
                    {
                        "uid": record.uid,
                        "split": args.split,
                        "shard": shard_path.name,
                        "offset": offset,
                        "bank": "spatial",
                        "k_temporal": record.k_temporal,
                        "model_name": model_name,
                        "config_fingerprint": config_fingerprint,
                        "plan_fingerprint": plan_fingerprint,
                    }
                )
            print(f"{'Created' if created else 'Reused'}: {shard_path}")
    finally:
        if extractor is not None:
            extractor.close()

    write_jsonl_atomic(index_path, output_index)
    peak_gib = (
        torch.cuda.max_memory_allocated() / 1024**3
        if torch.cuda.is_available()
        else 0.0
    )
    summary = {
        "bank": "spatial",
        "split": args.split,
        "samples": len(output_index),
        "created_shards": created_shards,
        "reused_shards": reused_shards,
        "peak_cuda_memory_gib": round(peak_gib, 3),
        "index": str(index_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
