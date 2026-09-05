"""Endpoint-preserving source-frame sampling for PGAT."""

from __future__ import annotations

import numpy as np


def sample_source_positions(
    number_of_frames: int,
    positions: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source indices and a mask identifying non-padding occurrences.

    Indices are rounded endpoint-preserving linspace positions. When the source
    is shorter than ``positions``, repeated nearest frames are retained in the
    index array, while exactly one occurrence of every source frame is marked
    valid. The first and last positions are preferred for their endpoints.
    """

    if number_of_frames <= 0:
        raise ValueError("number_of_frames must be greater than zero")
    if positions <= 0:
        raise ValueError("positions must be greater than zero")

    indices = np.rint(
        np.linspace(0, number_of_frames - 1, num=positions, dtype=np.float64)
    ).astype(np.int32)
    indices[0] = 0
    indices[-1] = number_of_frames - 1
    indices = np.maximum.accumulate(indices).astype(np.int32, copy=False)

    valid = np.ones(positions, dtype=np.bool_)
    if number_of_frames < positions:
        valid.fill(False)
        chosen: dict[int, int] = {}
        for position, source_index in enumerate(indices.tolist()):
            chosen.setdefault(source_index, position)
        chosen[0] = 0
        chosen[number_of_frames - 1] = positions - 1
        valid[np.fromiter(chosen.values(), dtype=np.int64)] = True

    return indices, valid


def uniform_segment_bounds(
    positions: int = 64,
    segments: int = 16,
) -> np.ndarray:
    if positions <= 0 or segments <= 0:
        raise ValueError("positions and segments must be positive")
    if positions < segments:
        raise ValueError("positions must be at least the number of segments")
    bounds = np.rint(np.linspace(0, positions, segments + 1)).astype(np.int16)
    bounds[0] = 0
    bounds[-1] = positions
    return bounds


def importance_segment_bounds(
    importance: np.ndarray,
    segments: int = 16,
    minimum_width: int = 2,
) -> np.ndarray:
    """Place boundaries at cumulative-importance quantiles with width limits."""

    values = np.asarray(importance, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("importance must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("importance contains non-finite values")
    if np.any(values < 0):
        raise ValueError("importance must be non-negative")
    if segments <= 0 or minimum_width <= 0:
        raise ValueError("segments and minimum_width must be positive")
    if values.size < segments * minimum_width:
        raise ValueError("sequence is too short for the requested minimum width")

    if float(values.sum()) <= 0.0:
        return uniform_segment_bounds(values.size, segments)

    cumulative = np.cumsum(values)
    targets = np.linspace(0.0, float(cumulative[-1]), segments + 1)
    raw = np.searchsorted(cumulative, targets, side="left") + 1
    raw[0] = 0
    raw[-1] = values.size

    bounds = np.zeros(segments + 1, dtype=np.int64)
    bounds[-1] = values.size
    for index in range(1, segments):
        lower = bounds[index - 1] + minimum_width
        upper = values.size - (segments - index) * minimum_width
        bounds[index] = int(np.clip(raw[index], lower, upper))
    return bounds.astype(np.int16)


def weighted_anchor_positions(
    segment_bounds: np.ndarray,
    importance: np.ndarray,
) -> np.ndarray:
    bounds = np.asarray(segment_bounds, dtype=np.int64)
    weights = np.asarray(importance, dtype=np.float64)
    if bounds.ndim != 1 or bounds.size < 2:
        raise ValueError("segment_bounds must be one-dimensional")
    if bounds[0] != 0 or bounds[-1] != weights.size:
        raise ValueError("segment bounds must cover the importance sequence")

    anchors = np.empty(bounds.size - 1, dtype=np.int16)
    for index, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        if end <= start:
            raise ValueError("segment bounds must be strictly increasing")
        local = weights[start:end]
        positions = np.arange(start, end, dtype=np.float64)
        if float(local.sum()) > 0.0:
            center = float(np.average(positions, weights=local))
        else:
            center = (float(start) + float(end - 1)) / 2.0
        anchors[index] = int(np.clip(round(center), start, end - 1))
    return anchors

