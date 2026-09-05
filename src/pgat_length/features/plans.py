"""Variable-K temporal plan builder for pgat-length.

Wraps :func:`pgat_length.features.pose_plan.build_compact_pose_plan` so the
number of temporal segments is chosen per sample from the raw video length.

For a video with ``F`` frames, the plan gets

    K_temporal = clamp(round(F / frames_per_token), K_min, K_max)

temporal segments. Segment bounds and anchors are computed on the sampled
``POSITIONS = 64`` positions (endpoint-preserving linspace); the pose-motion
importance drives the boundary placement exactly as in PGAT-v1, only with a
variable K.

Per-sample outputs are then padded along the temporal axis to K_MAX so shards
can stack samples of different K. The scalar ``k_temporal`` array carried in
the shard tells readers where the padding starts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pgat_length.features.pose_plan import (
    PosePlanConfig,
    build_compact_pose_plan,
)
from pgat_length.features.shards import K_MAX, POSITIONS


@dataclass(frozen=True)
class VariableKConfig:
    frames_per_token: int = 8
    k_min: int = 12
    k_max: int = K_MAX
    minimum_segment_width: int = 2
    hands_weight: float = 0.60
    body_weight: float = 0.25
    mouth_weight: float = 0.15
    uniform_floor: float = 0.05
    valid_pose_fraction_threshold: float = 0.35

    def as_pose_plan_config(self, k_temporal: int) -> PosePlanConfig:
        return PosePlanConfig(
            segments=k_temporal,
            minimum_segment_width=self.minimum_segment_width,
            hands_weight=self.hands_weight,
            body_weight=self.body_weight,
            mouth_weight=self.mouth_weight,
            uniform_floor=self.uniform_floor,
            valid_pose_fraction_threshold=self.valid_pose_fraction_threshold,
        )


def compute_k_temporal(source_num_frames: int, config: VariableKConfig) -> int:
    """K_temporal = clamp(round(F / frames_per_token), K_min, K_max)."""
    if source_num_frames <= 0:
        raise ValueError("source_num_frames must be positive")
    raw = round(source_num_frames / max(1, config.frames_per_token))
    return int(np.clip(raw, config.k_min, config.k_max))


def pad_along_axis(array: np.ndarray, target_size: int, axis: int, fill: float = 0.0) -> np.ndarray:
    """Right-pad ``array`` along ``axis`` to ``target_size`` with ``fill``."""
    current = array.shape[axis]
    if current == target_size:
        return array
    if current > target_size:
        raise ValueError(f"array axis {axis} exceeds target {target_size}: {current}")
    pad_widths = [(0, 0)] * array.ndim
    pad_widths[axis] = (0, target_size - current)
    return np.pad(array, pad_widths, mode="constant", constant_values=fill)


def pad_plan_to_kmax(plan_tensors: dict[str, np.ndarray], k_temporal: int) -> dict[str, np.ndarray]:
    """Pad a per-sample plan's arrays from ``k_temporal`` to ``K_MAX``.

    Padding fill values:
    - ``segment_bounds``  : POSITIONS (the sentinel repeated after the real end)
    - ``anchor_positions``: 0
    - all others           : 0 / False
    """
    if plan_tensors["anchor_positions"].shape[0] != k_temporal:
        raise ValueError("plan segments do not match requested k_temporal")

    padded: dict[str, np.ndarray] = {}
    padded["segment_bounds"] = pad_along_axis(
        plan_tensors["segment_bounds"], K_MAX + 1, axis=0, fill=POSITIONS
    ).astype(np.int16, copy=False)
    padded["anchor_positions"] = pad_along_axis(
        plan_tensors["anchor_positions"], K_MAX, axis=0, fill=0
    ).astype(np.int16, copy=False)
    padded["crop_boxes"] = pad_along_axis(
        plan_tensors["crop_boxes"], K_MAX, axis=0, fill=0
    ).astype(np.int16, copy=False)
    padded["region_valid"] = pad_along_axis(
        plan_tensors["region_valid"], K_MAX, axis=0, fill=False
    ).astype(np.bool_, copy=False)
    padded["pose_descriptor"] = pad_along_axis(
        plan_tensors["pose_descriptor"], K_MAX, axis=0, fill=0.0
    ).astype(np.float16, copy=False)
    padded["pose_confidence"] = pad_along_axis(
        plan_tensors["pose_confidence"], K_MAX, axis=0, fill=0.0
    ).astype(np.float16, copy=False)
    padded["pose_motion"] = pad_along_axis(
        plan_tensors["pose_motion"], K_MAX, axis=0, fill=0.0
    ).astype(np.float16, copy=False)
    return padded


def build_variable_k_plan(
    landmarks: dict[str, np.ndarray],
    frame_valid: np.ndarray,
    image_size: tuple[int, int],
    source_num_frames: int,
    config: VariableKConfig,
) -> tuple[dict[str, np.ndarray], int, bool, float]:
    """Build a padded per-sample plan with variable K_temporal.

    Returns:
        (padded_tensors, k_temporal, used_uniform_fallback, valid_pose_fraction)

    ``padded_tensors`` contains the seven arrays contracted by shards.py plus
    a scalar ``k_temporal`` (as an ndarray of dtype int16, shape ()).
    """
    k_temporal = compute_k_temporal(source_num_frames, config)
    plan = build_compact_pose_plan(
        landmarks=landmarks,
        frame_valid=frame_valid,
        image_size=image_size,
        config=config.as_pose_plan_config(k_temporal),
    )
    padded = pad_plan_to_kmax(plan.as_tensors(), k_temporal)
    padded["k_temporal"] = np.array(k_temporal, dtype=np.int16)
    return padded, k_temporal, plan.used_uniform_fallback, plan.valid_pose_fraction
