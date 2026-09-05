"""Motion features derived from pose landmarks.

Responsibility:
- Compute per-sample motion features summarizing hand/face/body velocity,
  acceleration, and direction change over the sampled frames.
- Output tensor shape [8, 768] per sample (fixed 8 motion centers projected to
  visual dim), consistent with the PGAT tokenizer expectations.
- FP16 safetensors, atomic writes, resume-safe.

Public API (to implement):
- extract_motion_features(plan: dict) -> np.ndarray
- run_split(split: str, plan_root: Path, out_root: Path, allow_full: bool) -> None
"""

raise NotImplementedError("pgat_length.features.motion: implement in step 01")
