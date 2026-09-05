"""Compact pose descriptors, crop boxes, and temporal routing plans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import cv2
import numpy as np

from pgat_length.features.frame_sampling import (
    importance_segment_bounds,
    uniform_segment_bounds,
    weighted_anchor_positions,
)
from pgat_length.pose.constants import (
    FACE_SELECTED_INDICES,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)


GROUP_NAMES = ("left_hand", "right_hand", "mouth", "body")


class FramePoseExtractor(Protocol):
    def extract_frame(self, image_bgr: np.ndarray) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class PosePlanConfig:
    segments: int = 16
    minimum_segment_width: int = 2
    hands_weight: float = 0.60
    body_weight: float = 0.25
    mouth_weight: float = 0.15
    uniform_floor: float = 0.05
    valid_pose_fraction_threshold: float = 0.35
    crop_padding: float = 0.35
    minimum_crop_pixels: int = 24


@dataclass(frozen=True)
class CompactPosePlan:
    segment_bounds: np.ndarray
    anchor_positions: np.ndarray
    crop_boxes: np.ndarray
    region_valid: np.ndarray
    pose_descriptor: np.ndarray
    pose_confidence: np.ndarray
    pose_motion: np.ndarray
    used_uniform_fallback: bool
    valid_pose_fraction: float

    def as_tensors(self) -> dict[str, np.ndarray]:
        return {
            "segment_bounds": self.segment_bounds,
            "anchor_positions": self.anchor_positions,
            "crop_boxes": self.crop_boxes,
            "region_valid": self.region_valid,
            "pose_descriptor": self.pose_descriptor,
            "pose_confidence": self.pose_confidence,
            "pose_motion": self.pose_motion,
        }


def extract_selected_landmarks(
    frame_paths: list[Path],
    extractor: FramePoseExtractor,
) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    """Run MediaPipe once per unique selected source frame and expand repeats."""

    if not frame_paths:
        raise ValueError("frame_paths must not be empty")
    unique_results: dict[Path, Mapping[str, Any]] = {}
    image_size: tuple[int, int] | None = None
    expanded: list[Mapping[str, Any]] = []
    for path in frame_paths:
        key = path.resolve()
        if key not in unique_results:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read frame: {path}")
            height, width = image.shape[:2]
            if image_size is None:
                image_size = (width, height)
            elif image_size != (width, height):
                raise ValueError("All selected frames must have the same dimensions")
            unique_results[key] = extractor.extract_frame(image)
        expanded.append(unique_results[key])

    frame_count = len(expanded)
    arrays = {
        "left_hand": np.zeros((frame_count, 21, 4), dtype=np.float32),
        "right_hand": np.zeros((frame_count, 21, 4), dtype=np.float32),
        "mouth": np.zeros((frame_count, len(FACE_SELECTED_INDICES), 4), dtype=np.float32),
        "body": np.zeros((frame_count, 33, 5), dtype=np.float32),
        "left_valid": np.zeros(frame_count, dtype=np.bool_),
        "right_valid": np.zeros(frame_count, dtype=np.bool_),
        "mouth_valid": np.zeros(frame_count, dtype=np.bool_),
        "body_valid": np.zeros(frame_count, dtype=np.bool_),
    }
    for index, result in enumerate(expanded):
        for source_name, target_name, valid_name in (
            ("left_hand", "left_hand", "left_valid"),
            ("right_hand", "right_hand", "right_valid"),
            ("face_selected", "mouth", "mouth_valid"),
            ("pose_full", "body", "body_valid"),
        ):
            value = result.get(source_name)
            if value is not None:
                arrays[target_name][index] = np.asarray(value, dtype=np.float32)
                arrays[valid_name][index] = True
    assert image_size is not None
    return arrays, image_size


def _centers(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.zeros((values.shape[0], 2), dtype=np.float32)
    output[valid] = values[valid, :, :2].mean(axis=1)
    return output


def _confidence(values: np.ndarray, valid: np.ndarray, body: bool = False) -> np.ndarray:
    output = np.zeros(values.shape[0], dtype=np.float32)
    if body:
        selected = values[:, [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]]
        scores = np.minimum(selected[..., 3], selected[..., 4]).mean(axis=1)
    else:
        scores = values[..., 3].mean(axis=1)
    output[valid] = np.clip(scores[valid], 0.0, 1.0)
    return output


def _displacement(
    centers: np.ndarray,
    valid: np.ndarray,
    frame_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.zeros_like(centers)
    magnitude = np.zeros(centers.shape[0], dtype=np.float32)
    pair_valid = valid[1:] & valid[:-1] & frame_valid[1:] & frame_valid[:-1]
    difference = centers[1:] - centers[:-1]
    delta[1:][pair_valid] = difference[pair_valid]
    magnitude[1:][pair_valid] = np.linalg.norm(difference[pair_valid], axis=1)
    return delta, magnitude


def _box_features(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.zeros((values.shape[0], 4), dtype=np.float32)
    scales = np.zeros(values.shape[0], dtype=np.float32)
    if np.any(valid):
        xy = values[valid, :, :2]
        lower = xy.min(axis=1)
        upper = xy.max(axis=1)
        boxes[valid] = np.concatenate((lower, upper), axis=1)
        scales[valid] = np.sqrt(np.maximum((upper - lower).prod(axis=1), 0.0))
    return boxes, scales


def _hand_descriptor(
    values: np.ndarray,
    valid: np.ndarray,
    frame_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = _centers(values, valid)
    boxes, scales = _box_features(values, valid)
    confidence = _confidence(values, valid)
    delta, speed = _displacement(centers, valid, frame_valid)
    relative = values[..., :3] - values[:, :1, :3]
    radial = np.linalg.norm(relative[..., :2], axis=2)
    wrist_summary = np.stack(
        (
            relative[..., 0].mean(axis=1),
            relative[..., 1].mean(axis=1),
            relative[..., 2].mean(axis=1),
            radial.mean(axis=1),
            radial.std(axis=1),
        ),
        axis=1,
    )
    descriptor = np.concatenate(
        (
            centers,
            boxes,
            scales[:, None],
            confidence[:, None],
            delta,
            speed[:, None],
            wrist_summary,
        ),
        axis=1,
    )
    descriptor[~valid] = 0.0
    return descriptor.astype(np.float32), confidence, speed


def _mouth_descriptor(
    values: np.ndarray,
    valid: np.ndarray,
    frame_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = _centers(values, valid)
    boxes, scales = _box_features(values, valid)
    confidence = _confidence(values, valid)
    delta, speed = _displacement(centers, valid, frame_valid)
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    aspect = np.divide(
        width,
        height,
        out=np.zeros_like(width),
        where=height > 1e-6,
    )
    descriptor = np.concatenate(
        (
            centers,
            boxes,
            scales[:, None],
            confidence[:, None],
            delta,
            aspect[:, None],
            height[:, None],
        ),
        axis=1,
    )
    descriptor[~valid] = 0.0
    return descriptor.astype(np.float32), confidence, speed


def _body_descriptor(
    values: np.ndarray,
    valid: np.ndarray,
    frame_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_shoulder = values[:, LEFT_SHOULDER, :2]
    right_shoulder = values[:, RIGHT_SHOULDER, :2]
    left_hip = values[:, LEFT_HIP, :2]
    right_hip = values[:, RIGHT_HIP, :2]
    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    torso_center = (shoulder_center + (left_hip + right_hip) / 2.0) / 2.0
    shoulder_vector = right_shoulder - left_shoulder
    shoulder_width = np.linalg.norm(shoulder_vector, axis=1)
    angle = np.arctan2(shoulder_vector[:, 1], shoulder_vector[:, 0])
    confidence = _confidence(values, valid, body=True)
    delta, speed = _displacement(torso_center, valid, frame_valid)
    torso_width = np.linalg.norm(right_hip - left_hip, axis=1)
    torso_height = np.linalg.norm(
        shoulder_center - (left_hip + right_hip) / 2.0,
        axis=1,
    )
    wrist_distance = np.stack(
        (
            np.linalg.norm(values[:, LEFT_WRIST, :2] - shoulder_center, axis=1),
            np.linalg.norm(values[:, RIGHT_WRIST, :2] - shoulder_center, axis=1),
        ),
        axis=1,
    )
    shoulder_tilt = shoulder_vector[:, 1:2]
    descriptor = np.concatenate(
        (
            shoulder_center,
            shoulder_width[:, None],
            np.sin(angle)[:, None],
            np.cos(angle)[:, None],
            confidence[:, None],
            delta,
            speed[:, None],
            torso_center,
            torso_width[:, None],
            torso_height[:, None],
            wrist_distance,
            shoulder_tilt,
        ),
        axis=1,
    )
    descriptor[~valid] = 0.0
    return descriptor.astype(np.float32), confidence, speed


def _pixel_crop_box(
    landmarks: np.ndarray,
    width: int,
    height: int,
    padding: float,
    minimum_pixels: int,
) -> np.ndarray:
    xy = landmarks[:, :2]
    lower = xy.min(axis=0) * np.asarray([width, height], dtype=np.float32)
    upper = xy.max(axis=0) * np.asarray([width, height], dtype=np.float32)
    center = (lower + upper) / 2.0
    size = np.maximum(upper - lower, float(minimum_pixels)) * (
        1.0 + 2.0 * padding
    )
    lower = np.floor(center - size / 2.0)
    upper = np.ceil(center + size / 2.0)
    return np.asarray(
        [
            np.clip(lower[0], 0, width - 1),
            np.clip(lower[1], 0, height - 1),
            np.clip(upper[0], 1, width),
            np.clip(upper[1], 1, height),
        ],
        dtype=np.int16,
    )


def build_compact_pose_plan(
    landmarks: Mapping[str, np.ndarray],
    frame_valid: np.ndarray,
    image_size: tuple[int, int],
    config: PosePlanConfig = PosePlanConfig(),
) -> CompactPosePlan:
    frame_mask = np.asarray(frame_valid, dtype=np.bool_)
    frame_count = frame_mask.size
    if (
        frame_mask.ndim != 1
        or frame_count < config.segments * config.minimum_segment_width
    ):
        raise ValueError("frame_valid has an incompatible shape")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive dimensions")

    left_valid = np.asarray(landmarks["left_valid"], dtype=np.bool_) & frame_mask
    right_valid = np.asarray(landmarks["right_valid"], dtype=np.bool_) & frame_mask
    mouth_valid = np.asarray(landmarks["mouth_valid"], dtype=np.bool_) & frame_mask
    body_valid = np.asarray(landmarks["body_valid"], dtype=np.bool_) & frame_mask

    left_desc, left_conf, left_motion = _hand_descriptor(
        landmarks["left_hand"], left_valid, frame_mask
    )
    right_desc, right_conf, right_motion = _hand_descriptor(
        landmarks["right_hand"], right_valid, frame_mask
    )
    mouth_desc, mouth_conf, mouth_motion = _mouth_descriptor(
        landmarks["mouth"], mouth_valid, frame_mask
    )
    body_desc, body_conf, body_motion = _body_descriptor(
        landmarks["body"], body_valid, frame_mask
    )
    confidence = np.stack((left_conf, right_conf, mouth_conf, body_conf), axis=1)
    motion = np.stack((left_motion, right_motion, mouth_motion, body_motion), axis=1)

    per_frame_descriptor = np.concatenate(
        (left_desc, right_desc, mouth_desc, body_desc, confidence), axis=1
    )
    if per_frame_descriptor.shape != (frame_count, 64):
        raise AssertionError(f"unexpected descriptor shape: {per_frame_descriptor.shape}")

    hand_importance = (left_conf * left_motion + right_conf * right_motion) / 2.0
    importance = (
        config.hands_weight * hand_importance
        + config.body_weight * body_conf * body_motion
        + config.mouth_weight * mouth_conf * mouth_motion
        + config.uniform_floor
    )
    importance[~frame_mask] = 0.0
    valid_denominator = max(int(frame_mask.sum()), 1)
    any_pose_valid = body_valid | left_valid | right_valid | mouth_valid
    valid_pose_fraction = float(any_pose_valid.sum()) / valid_denominator
    used_uniform = valid_pose_fraction < config.valid_pose_fraction_threshold
    if used_uniform:
        bounds = uniform_segment_bounds(frame_count, config.segments)
    else:
        bounds = importance_segment_bounds(
            importance,
            config.segments,
            config.minimum_segment_width,
        )
    anchors = weighted_anchor_positions(bounds, importance)

    descriptors = np.zeros((config.segments, 64), dtype=np.float16)
    segment_confidence = np.zeros((config.segments, 4), dtype=np.float16)
    segment_motion = np.zeros((config.segments, 4), dtype=np.float16)
    crop_boxes = np.zeros((config.segments, 3, 4), dtype=np.int16)
    region_valid = np.zeros((config.segments, 3), dtype=np.bool_)
    region_arrays = (
        landmarks["left_hand"],
        landmarks["right_hand"],
        landmarks["mouth"],
    )
    region_masks = (left_valid, right_valid, mouth_valid)

    for segment, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        local_valid = frame_mask[start:end]
        if np.any(local_valid):
            descriptors[segment] = (
                per_frame_descriptor[start:end][local_valid]
                .mean(axis=0)
                .astype(np.float16)
            )
            segment_confidence[segment] = (
                confidence[start:end][local_valid]
                .mean(axis=0)
                .astype(np.float16)
            )
            segment_motion[segment] = (
                motion[start:end][local_valid]
                .mean(axis=0)
                .astype(np.float16)
            )
        anchor = int(anchors[segment])
        for region, (values, valid) in enumerate(zip(region_arrays, region_masks)):
            candidates = np.flatnonzero(valid[start:end]) + int(start)
            if candidates.size == 0:
                continue
            selected = int(candidates[np.argmin(np.abs(candidates - anchor))])
            crop_boxes[segment, region] = _pixel_crop_box(
                values[selected],
                width,
                height,
                config.crop_padding,
                config.minimum_crop_pixels,
            )
            region_valid[segment, region] = True

    return CompactPosePlan(
        segment_bounds=bounds.astype(np.int16),
        anchor_positions=anchors.astype(np.int16),
        crop_boxes=crop_boxes,
        region_valid=region_valid,
        pose_descriptor=descriptors,
        pose_confidence=segment_confidence,
        pose_motion=segment_motion,
        used_uniform_fallback=used_uniform,
        valid_pose_fraction=valid_pose_fraction,
    )
