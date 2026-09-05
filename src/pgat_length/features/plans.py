"""Pose-informed frame plans with variable K_temporal.

Responsibility:
- For each sample, run MediaPipe to obtain per-frame pose landmarks + confidence.
- Compute pose-motion energy per frame.
- Derive K_temporal = clamp(round(source_num_frames / frames_per_token), K_min, K_max).
- Segment the timeline into K_temporal non-overlapping regions using pose-motion
  boundary strength (higher energy => finer segmentation).
- Per region, store: anchor frame index, four crop boxes (global/left/right/mouth),
  four confidence values, motion summary, valid mask.
- Persist plans as safetensors shards + JSONL UID index.

Public API (to implement):
- build_plan(sample_row: dict, model_config: dict) -> dict
- write_plan_shard(rows: list[dict], out_dir: Path, shard_index: int) -> None
"""

raise NotImplementedError("pgat_length.features.plans: implement in step 01")
