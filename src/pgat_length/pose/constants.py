from __future__ import annotations

# MediaPipe PoseLandmarker indices.
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

UPPER_BODY_INDICES = (
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
)

TORSO_INDICES = (
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
)

# A compact face subset containing lips, mouth corners, eyebrows and chin.
# These indices are within the standard MediaPipe face mesh landmark set.
MOUTH_INDICES = (
    0, 13, 14, 17,
    37, 39, 40, 61,
    78, 80, 81, 82,
    84, 87, 88, 91,
    95, 146, 178, 181,
    185, 191, 267, 269,
    270, 291, 308, 310,
    311, 312, 314, 317,
    318, 321, 324, 375,
    402, 405, 409, 415,
)

EYEBROW_INDICES = (
    46, 52, 53, 65, 70,
    276, 282, 283, 295, 300,
)

CHIN_INDICES = (152,)

FACE_SELECTED_INDICES = MOUTH_INDICES + EYEBROW_INDICES + CHIN_INDICES

# Standard 21-point hand skeleton connections.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

POSE_UPPER_BODY_CONNECTIONS = (
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
)
