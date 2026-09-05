"""Deterministic (and augmented) temporal frame sampling.

Responsibility:
- Given source_num_frames, produce a list of frame indices for feature extraction.
- Endpoint-preserving rounded linspace as the deterministic base.
- Adaptive segmentation: derive K_temporal segments (K in [K_min, K_max])
  from pose-motion boundaries; return segment anchors plus valid mask.
- Training-time augmentation hooks: random start (<= 20% of video),
  frame drop (p=0.10), temporal jitter (<= 2 frames).

Public API (to implement):
- deterministic_sample(source_num_frames: int, target: int) -> np.ndarray
- adaptive_segments(pose_motion: np.ndarray, k_min: int, k_max: int) -> np.ndarray
- augment_indices(indices: np.ndarray, aug_config: dict, rng: np.random.Generator) -> np.ndarray
"""

raise NotImplementedError("pgat_length.data.sampling: implement in step 01")
