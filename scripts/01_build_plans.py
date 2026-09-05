"""Step 01 — build per-sample pose-informed plans with variable K_temporal.

Reads PHOENIX14T frames and the cached manifest; runs MediaPipe pose extraction
on the deterministically-sampled 64 source positions; builds a compact plan with
K_temporal segments (variable per sample); pads each plan to K_MAX and writes
shards + a JSONL index.

Usage (via sbatch on ASL):
    sbatch slurm/build_features.sbatch scripts/01_build_plans.py \\
        --split train --all
    sbatch slurm/build_features.sbatch scripts/01_build_plans.py \\
        --split dev --all

Smoke (small K samples for local testing):
    python scripts/01_build_plans.py --split dev --limit 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pgat_length.features.frame_paths import (  # noqa: E402
    list_frame_files,
    resolve_frame_directory,
)
from pgat_length.features.frame_sampling import sample_source_positions  # noqa: E402
from pgat_length.features.plans import VariableKConfig, build_variable_k_plan  # noqa: E402
from pgat_length.features.pose_plan import extract_selected_landmarks  # noqa: E402
from pgat_length.features.shards import (  # noqa: E402
    K_MAX,
    PLAN_ARRAY_CONTRACTS,
    PLAN_SECTIONS,
    POSITIONS,
    fingerprint_from_sections,
    uid_order_fingerprint,
    validate_saved_plan_shard,
    write_jsonl_atomic,
    write_plan_shard,
)
from pgat_length.pose.extractor import (  # noqa: E402
    MediaPipePoseExtractor,
    PoseExtractorConfig,
)


def _expand_env(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return payload


def project_or_absolute(root: Path, value: str) -> Path:
    expanded = Path(_expand_env(str(value)))
    return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()


def pose_extractor_config(pose_cfg: dict[str, Any]) -> PoseExtractorConfig:
    return PoseExtractorConfig(
        hand_model_path=project_or_absolute(PROJECT_ROOT, pose_cfg["hand_model"]),
        face_model_path=project_or_absolute(PROJECT_ROOT, pose_cfg["face_model"]),
        pose_model_path=project_or_absolute(PROJECT_ROOT, pose_cfg["body_model"]),
        min_hand_detection_confidence=float(pose_cfg["min_hand_detection_confidence"]),
        min_hand_presence_confidence=float(pose_cfg["min_hand_presence_confidence"]),
        min_face_detection_confidence=float(pose_cfg["min_face_detection_confidence"]),
        min_face_presence_confidence=float(pose_cfg["min_face_presence_confidence"]),
        min_pose_detection_confidence=float(pose_cfg["min_pose_detection_confidence"]),
        min_pose_presence_confidence=float(pose_cfg["min_pose_presence_confidence"]),
        smoothing_window=int(pose_cfg["smoothing_window"]),
    )


def variable_k_config(sampling_cfg: dict[str, Any]) -> VariableKConfig:
    return VariableKConfig(
        frames_per_token=int(sampling_cfg["frames_per_token"]),
        k_min=int(sampling_cfg["k_temporal_min"]),
        k_max=int(sampling_cfg["k_temporal_max"]),
        minimum_segment_width=int(sampling_cfg["minimum_segment_width"]),
        hands_weight=float(sampling_cfg["weights"]["hands"]),
        body_weight=float(sampling_cfg["weights"]["body"]),
        mouth_weight=float(sampling_cfg["weights"]["mouth"]),
        uniform_floor=float(sampling_cfg["uniform_floor"]),
        valid_pose_fraction_threshold=float(sampling_cfg["valid_pose_fraction_threshold"]),
    )


def stack_records(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack per-sample tensor dicts along a new leading batch axis."""
    if not records:
        raise ValueError("no records to stack")
    keys = set(records[0])
    for record in records[1:]:
        if set(record) != keys:
            raise ValueError("inconsistent keys across records")
    return {name: np.stack([record[name] for record in records]) for name in keys}


def write_metadata(output_root: Path, config: dict[str, Any], fingerprint: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "metadata.json"
    payload = {
        "format_version": 2,
        "bank": "pgat_length_plans",
        "positions": POSITIONS,
        "k_max": K_MAX,
        "config_fingerprint": fingerprint,
        "config": config,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != fingerprint:
            raise RuntimeError(f"Plan root contains a different configuration: {path}")
        return
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_existing_index(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
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
    scope.add_argument("--all", action="store_true", help="Explicitly process the complete split.")
    scope.add_argument("--limit", type=int, help="Smoke sample count (default: 3).")
    scope.add_argument("--uid", action="append", help="Process specific UIDs; repeatable.")
    parser.add_argument("--output-root", type=Path, help="Override plans root (default: features_root/plans).")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml(args.data_config.resolve())
    features_cfg = load_yaml(args.features_config.resolve())
    combined_for_fingerprint = {
        "data": data_cfg,
        "features": {k: features_cfg[k] for k in PLAN_SECTIONS if k in features_cfg},
    }
    config_fingerprint = fingerprint_from_sections(data_cfg, features_cfg, PLAN_SECTIONS)

    manifest_path = project_or_absolute(PROJECT_ROOT, data_cfg["paths"]["manifest"])
    frames_root = project_or_absolute(PROJECT_ROOT, data_cfg["paths"]["raw_frames_root"])
    default_feature_root = project_or_absolute(PROJECT_ROOT, data_cfg["paths"]["feature_root"])
    default_name = "plans" if args.all else "plans_smoke"
    output_root = args.output_root.resolve() if args.output_root else default_feature_root / default_name
    write_metadata(output_root, combined_for_fingerprint, config_fingerprint)

    manifest = pd.read_pickle(manifest_path)
    rows = manifest.loc[manifest["split"] == args.split].reset_index(drop=True)
    if args.uid:
        requested = set(args.uid)
        rows = rows.loc[rows["uid"].isin(requested)].reset_index(drop=True)
        missing = sorted(requested.difference(rows["uid"].astype(str)))
        if missing:
            raise ValueError(f"UIDs not found in split {args.split}: {missing}")
    elif not args.all:
        rows = rows.iloc[: args.limit if args.limit is not None else 3]
    if rows.empty:
        raise RuntimeError(f"no samples selected for split {args.split}")

    positions = int(features_cfg["sampling"]["positions"])
    if positions != POSITIONS:
        raise ValueError(f"features.yaml sampling.positions must equal {POSITIONS}")
    shard_size = int(features_cfg["storage"]["shard_size"])
    var_config = variable_k_config(features_cfg["sampling"])
    pose_cfg_full = pose_extractor_config(features_cfg["pose"])

    index_rows: list[dict[str, Any]] = []
    index_path = output_root / args.split / "index.jsonl"
    existing_index = load_existing_index(index_path)
    started = time.monotonic()

    with ExitStack() as stack:
        extractor: MediaPipePoseExtractor | None = None
        for shard_number, row_start in enumerate(range(0, len(rows), shard_size)):
            shard_rows = rows.iloc[row_start : row_start + shard_size]
            uids = shard_rows["uid"].astype(str).tolist()
            shard_path = output_root / args.split / f"plans-{shard_number:05d}.safetensors"

            if shard_path.exists() and not args.overwrite:
                metadata = validate_saved_plan_shard(shard_path, len(uids))
                if metadata.get("config_fingerprint") != config_fingerprint:
                    raise RuntimeError(f"Existing shard has a different configuration: {shard_path}")
                if metadata.get("uid_fingerprint") != uid_order_fingerprint(uids):
                    raise RuntimeError(f"Existing shard has a different UID order: {shard_path}")
                for offset, (_, row) in enumerate(shard_rows.iterrows()):
                    previous = existing_index.get((shard_path.name, offset))
                    if previous is not None and previous.get("uid") == str(row["uid"]):
                        index_rows.append(previous)
                    else:
                        index_rows.append(
                            {
                                "uid": str(row["uid"]),
                                "split": args.split,
                                "shard": shard_path.name,
                                "offset": offset,
                                "source_num_frames": int(row["source_num_frames"]),
                                "k_temporal": None,
                                "used_uniform_fallback": None,
                                "valid_pose_fraction": None,
                                "config_fingerprint": config_fingerprint,
                            }
                        )
                print(f"Reused without pose inference: {shard_path}")
                continue

            if extractor is None:
                extractor = stack.enter_context(MediaPipePoseExtractor(pose_cfg_full))

            tensors: list[dict[str, np.ndarray]] = []
            diagnostics: list[tuple[int, bool, float]] = []
            for _, row in shard_rows.iterrows():
                frame_directory, _ = resolve_frame_directory(
                    frames_root, args.split, str(row["sample_id"])
                )
                frame_files = list_frame_files(frame_directory)
                if len(frame_files) != int(row["source_num_frames"]):
                    raise RuntimeError(
                        f"Frame count mismatch for {row['uid']}: "
                        f"manifest={row['source_num_frames']}, disk={len(frame_files)}"
                    )
                frame_indices, frame_valid = sample_source_positions(len(frame_files), positions)
                selected = [frame_files[int(index)] for index in frame_indices]
                landmarks, image_size = extract_selected_landmarks(selected, extractor)
                padded, k_temporal, used_fallback, valid_fraction = build_variable_k_plan(
                    landmarks=landmarks,
                    frame_valid=frame_valid,
                    image_size=image_size,
                    source_num_frames=int(row["source_num_frames"]),
                    config=var_config,
                )
                tensors.append(
                    {
                        "frame_indices": frame_indices.astype(np.int32),
                        "frame_valid": frame_valid.astype(np.bool_),
                        **padded,
                    }
                )
                diagnostics.append((k_temporal, used_fallback, valid_fraction))
                elapsed = max(time.monotonic() - started, 1e-6)
                completed = row_start + len(tensors)
                print(
                    f"[{completed}/{len(rows)}] {row['uid']} "
                    f"K={k_temporal:2d} pose_valid={valid_fraction:.3f} "
                    f"rate={completed / elapsed:.3f} samples/s",
                    flush=True,
                )

            shard_arrays = stack_records(tensors)
            missing = sorted(set(PLAN_ARRAY_CONTRACTS).difference(shard_arrays))
            if missing:
                raise RuntimeError(f"shard {shard_number}: missing plan arrays {missing}")

            shard_path, created = write_plan_shard(
                output_root=output_root,
                split=args.split,
                shard_number=shard_number,
                uids=uids,
                arrays=shard_arrays,
                config_fingerprint=config_fingerprint,
                uid_fingerprint=uid_order_fingerprint(uids),
                overwrite=args.overwrite,
            )
            for offset, (uid, (k, used_fallback, fraction)) in enumerate(zip(uids, diagnostics)):
                index_rows.append(
                    {
                        "uid": uid,
                        "split": args.split,
                        "shard": shard_path.name,
                        "offset": offset,
                        "source_num_frames": int(shard_rows.iloc[offset]["source_num_frames"]),
                        "k_temporal": int(k),
                        "used_uniform_fallback": bool(used_fallback),
                        "valid_pose_fraction": round(float(fraction), 6),
                        "config_fingerprint": config_fingerprint,
                    }
                )
            print(f"{'Created' if created else 'Reused'}: {shard_path}")

    write_jsonl_atomic(index_path, index_rows)
    print(f"Index: {index_path}")
    print(f"Completed {len(index_rows)} {args.split} plans.")


if __name__ == "__main__":
    main()
