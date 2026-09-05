from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from pgat_length.pose.constants import (
    FACE_SELECTED_INDICES,
    LEFT_WRIST,
    RIGHT_WRIST,
    TORSO_INDICES,
    UPPER_BODY_INDICES,
)


@dataclass(frozen=True)
class PoseExtractorConfig:
    hand_model_path: Path
    face_model_path: Path
    pose_model_path: Path
    min_hand_detection_confidence: float = 0.35
    min_hand_presence_confidence: float = 0.35
    min_face_detection_confidence: float = 0.35
    min_face_presence_confidence: float = 0.35
    min_pose_detection_confidence: float = 0.35
    min_pose_presence_confidence: float = 0.35
    smoothing_window: int = 3


@dataclass
class HandCandidate:
    landmarks: np.ndarray
    handedness_label: str
    handedness_score: float


class MediaPipePoseExtractor:
    """Extract grouped hand, face and body landmarks from sampled frames."""

    def __init__(self, config: PoseExtractorConfig) -> None:
        self.config = config
        self._validate_model_files()

        base_options = mp.tasks.BaseOptions
        vision = mp.tasks.vision

        hand_options = vision.HandLandmarkerOptions(
            base_options=base_options(
                model_asset_path=str(config.hand_model_path.resolve())
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=(
                config.min_hand_detection_confidence
            ),
            min_hand_presence_confidence=(
                config.min_hand_presence_confidence
            ),
        )
        face_options = vision.FaceLandmarkerOptions(
            base_options=base_options(
                model_asset_path=str(config.face_model_path.resolve())
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=(
                config.min_face_detection_confidence
            ),
            min_face_presence_confidence=(
                config.min_face_presence_confidence
            ),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        pose_options = vision.PoseLandmarkerOptions(
            base_options=base_options(
                model_asset_path=str(config.pose_model_path.resolve())
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=(
                config.min_pose_detection_confidence
            ),
            min_pose_presence_confidence=(
                config.min_pose_presence_confidence
            ),
            output_segmentation_masks=False,
        )

        self.hand_landmarker = vision.HandLandmarker.create_from_options(
            hand_options
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(
            face_options
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(
            pose_options
        )

    def _validate_model_files(self) -> None:
        for path in (
            self.config.hand_model_path,
            self.config.face_model_path,
            self.config.pose_model_path,
        ):
            if not path.exists():
                raise FileNotFoundError(f"MediaPipe model not found: {path}")

    def close(self) -> None:
        self.hand_landmarker.close()
        self.face_landmarker.close()
        self.pose_landmarker.close()

    def __enter__(self) -> "MediaPipePoseExtractor":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @staticmethod
    def _normalised_landmarks_to_array(
        landmarks: list[Any],
        confidence: float | None = None,
        include_visibility_and_presence: bool = False,
    ) -> np.ndarray:
        if include_visibility_and_presence:
            output = np.zeros((len(landmarks), 5), dtype=np.float32)
            for index, landmark in enumerate(landmarks):
                visibility = getattr(landmark, "visibility", None)
                presence = getattr(landmark, "presence", None)
                output[index] = (
                    float(landmark.x or 0.0),
                    float(landmark.y or 0.0),
                    float(landmark.z or 0.0),
                    float(visibility) if visibility is not None else 1.0,
                    float(presence) if presence is not None else 1.0,
                )
            return output

        output = np.zeros((len(landmarks), 4), dtype=np.float32)
        for index, landmark in enumerate(landmarks):
            presence = getattr(landmark, "presence", None)
            landmark_confidence = (
                float(confidence)
                if confidence is not None
                else float(presence) if presence is not None else 1.0
            )
            output[index] = (
                float(landmark.x or 0.0),
                float(landmark.y or 0.0),
                float(landmark.z or 0.0),
                landmark_confidence,
            )
        return output

    @staticmethod
    def _extract_hand_candidates(result: Any) -> list[HandCandidate]:
        candidates: list[HandCandidate] = []
        for index, landmarks in enumerate(result.hand_landmarks):
            categories = (
                result.handedness[index]
                if index < len(result.handedness)
                else []
            )
            category = categories[0] if categories else None
            label = (
                str(getattr(category, "category_name", "Unknown"))
                if category is not None
                else "Unknown"
            )
            score = (
                float(getattr(category, "score", 1.0))
                if category is not None
                else 1.0
            )
            candidates.append(
                HandCandidate(
                    landmarks=(
                        MediaPipePoseExtractor
                        ._normalised_landmarks_to_array(
                            landmarks,
                            confidence=score,
                        )
                    ),
                    handedness_label=label,
                    handedness_score=score,
                )
            )
        return candidates

    @staticmethod
    def _valid_pose_wrist(
        pose_landmarks: np.ndarray | None,
        index: int,
    ) -> np.ndarray | None:
        if pose_landmarks is None or pose_landmarks.shape[0] <= index:
            return None
        landmark = pose_landmarks[index]
        confidence = min(float(landmark[3]), float(landmark[4]))
        if confidence <= 0.05:
            return None
        return landmark[:2]

    @staticmethod
    def _assign_hands(
        candidates: list[HandCandidate],
        pose_landmarks: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Assign detected hands to the signer's left and right sides.

        Pose wrists are preferred because MediaPipe handedness can be affected
        by mirroring conventions. Handedness labels are used as fallback.
        """
        if not candidates:
            return None, None

        left_wrist = MediaPipePoseExtractor._valid_pose_wrist(
            pose_landmarks,
            LEFT_WRIST,
        )
        right_wrist = MediaPipePoseExtractor._valid_pose_wrist(
            pose_landmarks,
            RIGHT_WRIST,
        )

        if left_wrist is not None or right_wrist is not None:
            if len(candidates) == 1:
                candidate = candidates[0]
                hand_wrist = candidate.landmarks[0, :2]
                left_distance = (
                    np.linalg.norm(hand_wrist - left_wrist)
                    if left_wrist is not None
                    else np.inf
                )
                right_distance = (
                    np.linalg.norm(hand_wrist - right_wrist)
                    if right_wrist is not None
                    else np.inf
                )
                if left_distance <= right_distance:
                    return candidate.landmarks, None
                return None, candidate.landmarks

            first, second = candidates[:2]
            first_wrist = first.landmarks[0, :2]
            second_wrist = second.landmarks[0, :2]

            if left_wrist is not None and right_wrist is not None:
                direct_cost = (
                    np.linalg.norm(first_wrist - left_wrist)
                    + np.linalg.norm(second_wrist - right_wrist)
                )
                swapped_cost = (
                    np.linalg.norm(first_wrist - right_wrist)
                    + np.linalg.norm(second_wrist - left_wrist)
                )
                if direct_cost <= swapped_cost:
                    return first.landmarks, second.landmarks
                return second.landmarks, first.landmarks

        left: np.ndarray | None = None
        right: np.ndarray | None = None
        for candidate in sorted(
            candidates,
            key=lambda item: item.handedness_score,
            reverse=True,
        ):
            label = candidate.handedness_label.lower()
            if label == "left" and left is None:
                left = candidate.landmarks
            elif label == "right" and right is None:
                right = candidate.landmarks
            elif left is None:
                left = candidate.landmarks
            elif right is None:
                right = candidate.landmarks
        return left, right

    @staticmethod
    def _moving_average_with_mask(
        values: np.ndarray,
        valid: np.ndarray,
        window: int,
    ) -> np.ndarray:
        if window <= 1:
            return values.copy()

        output = values.copy()
        radius = window // 2
        for frame_index in range(values.shape[0]):
            start = max(0, frame_index - radius)
            end = min(values.shape[0], frame_index + radius + 1)
            local_mask = valid[start:end]
            if np.any(local_mask):
                output[frame_index] = values[start:end][local_mask].mean(
                    axis=0
                )
        return output

    def extract_frame(self, image_bgr: np.ndarray) -> dict[str, Any]:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Input frame is empty.")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = np.ascontiguousarray(image_rgb)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb,
        )

        pose_result = self.pose_landmarker.detect(mp_image)
        hand_result = self.hand_landmarker.detect(mp_image)
        face_result = self.face_landmarker.detect(mp_image)

        pose_landmarks: np.ndarray | None = None
        if pose_result.pose_landmarks:
            pose_landmarks = self._normalised_landmarks_to_array(
                pose_result.pose_landmarks[0],
                include_visibility_and_presence=True,
            )

        candidates = self._extract_hand_candidates(hand_result)
        left_hand, right_hand = self._assign_hands(
            candidates,
            pose_landmarks,
        )

        face_selected: np.ndarray | None = None
        if face_result.face_landmarks:
            face_full = self._normalised_landmarks_to_array(
                face_result.face_landmarks[0]
            )
            maximum_index = max(FACE_SELECTED_INDICES)
            if face_full.shape[0] <= maximum_index:
                raise ValueError(
                    "Face model returned fewer landmarks than expected: "
                    f"{face_full.shape[0]} <= {maximum_index}"
                )
            face_selected = face_full[list(FACE_SELECTED_INDICES)]

        return {
            "left_hand": left_hand,
            "right_hand": right_hand,
            "face_selected": face_selected,
            "pose_full": pose_landmarks,
        }

    def extract_clip(self, frame_paths: list[Path]) -> dict[str, np.ndarray]:
        frame_count = len(frame_paths)
        face_count = len(FACE_SELECTED_INDICES)

        left_hand = np.zeros((frame_count, 21, 4), dtype=np.float32)
        right_hand = np.zeros((frame_count, 21, 4), dtype=np.float32)
        face_selected = np.zeros(
            (frame_count, face_count, 4),
            dtype=np.float32,
        )
        pose_full = np.zeros((frame_count, 33, 5), dtype=np.float32)

        left_valid = np.zeros(frame_count, dtype=np.bool_)
        right_valid = np.zeros(frame_count, dtype=np.bool_)
        face_valid = np.zeros(frame_count, dtype=np.bool_)
        pose_valid = np.zeros(frame_count, dtype=np.bool_)

        for frame_index, frame_path in enumerate(frame_paths):
            image_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Could not read frame: {frame_path}")

            result = self.extract_frame(image_bgr)

            if result["left_hand"] is not None:
                left_hand[frame_index] = result["left_hand"]
                left_valid[frame_index] = True

            if result["right_hand"] is not None:
                right_hand[frame_index] = result["right_hand"]
                right_valid[frame_index] = True

            if result["face_selected"] is not None:
                face_selected[frame_index] = result["face_selected"]
                face_valid[frame_index] = True

            if result["pose_full"] is not None:
                pose_full[frame_index] = result["pose_full"]
                pose_valid[frame_index] = True

        upper_body = pose_full[:, list(UPPER_BODY_INDICES), :]
        torso = pose_full[:, list(TORSO_INDICES), :]

        hand_centroids = np.zeros((frame_count, 4), dtype=np.float32)
        for frame_index in range(frame_count):
            if left_valid[frame_index]:
                hand_centroids[frame_index, 0:2] = left_hand[
                    frame_index, :, 0:2
                ].mean(axis=0)
            if right_valid[frame_index]:
                hand_centroids[frame_index, 2:4] = right_hand[
                    frame_index, :, 0:2
                ].mean(axis=0)

        centroid_valid = np.stack(
            [left_valid, left_valid, right_valid, right_valid],
            axis=1,
        )
        smoothed_centroids = np.zeros_like(hand_centroids)
        smoothed_centroids[:, 0:2] = self._moving_average_with_mask(
            hand_centroids[:, 0:2],
            left_valid,
            self.config.smoothing_window,
        )
        smoothed_centroids[:, 2:4] = self._moving_average_with_mask(
            hand_centroids[:, 2:4],
            right_valid,
            self.config.smoothing_window,
        )
        smoothed_centroids[~centroid_valid] = 0.0

        hand_motion = np.zeros((frame_count, 6), dtype=np.float32)
        for frame_index in range(1, frame_count):
            if left_valid[frame_index] and left_valid[frame_index - 1]:
                delta = (
                    smoothed_centroids[frame_index, 0:2]
                    - smoothed_centroids[frame_index - 1, 0:2]
                )
                hand_motion[frame_index, 0:2] = delta
                hand_motion[frame_index, 2] = np.linalg.norm(delta)

            if right_valid[frame_index] and right_valid[frame_index - 1]:
                delta = (
                    smoothed_centroids[frame_index, 2:4]
                    - smoothed_centroids[frame_index - 1, 2:4]
                )
                hand_motion[frame_index, 3:5] = delta
                hand_motion[frame_index, 5] = np.linalg.norm(delta)

        return {
            "left_hand": left_hand,
            "right_hand": right_hand,
            "face_mouth": face_selected,
            "pose_full": pose_full,
            "upper_body": upper_body,
            "torso": torso,
            "left_hand_valid": left_valid,
            "right_hand_valid": right_valid,
            "face_valid": face_valid,
            "pose_valid": pose_valid,
            "hand_centroids": hand_centroids,
            "hand_centroids_smoothed": smoothed_centroids,
            "hand_motion": hand_motion,
            "face_indices": np.asarray(
                FACE_SELECTED_INDICES,
                dtype=np.int16,
            ),
        }
