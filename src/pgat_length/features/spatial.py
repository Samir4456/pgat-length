"""DINOv2 spatial feature extraction (four crop views per segment anchor).

Responsibility:
- Load facebook/dinov2-base from HF cache.
- For each plan segment: extract 4 crops (global, left hand, right hand, mouth)
  from the anchor frame; run DINOv2 to get one 768-dim vector per view.
- Store per-sample tensor shape [K_temporal, 4, 768] as FP16 safetensors.
- 192 samples per shard; atomic writes; resume-safe.

Public API (to implement):
- extract_spatial_features(plan: dict, model, processor, device) -> np.ndarray
- run_split(split: str, plan_root: Path, out_root: Path, allow_full: bool) -> None
"""

raise NotImplementedError("pgat_length.features.spatial: implement in step 01")
