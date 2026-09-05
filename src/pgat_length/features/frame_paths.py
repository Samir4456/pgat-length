from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np

FRAME_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def resolve_frame_directory(
    frames_root: Path, split: str, sample: str
) -> tuple[Path, Path]:
    """Resolve a sample below the frame root without duplicating its split.

    PHOENIX annotations commonly store names such as ``train/example`` while
    some converted annotations store only ``example``. The returned relative
    path always has exactly one leading split component.
    """
    sample_path = Path(str(sample).strip().replace("\\", "/"))
    if not sample_path.parts or str(sample_path) == ".":
        raise ValueError("sample must not be empty")
    if sample_path.is_absolute() or ".." in sample_path.parts:
        raise ValueError(f"Unsafe sample path: {sample}")

    if sample_path.parts[0].casefold() == split.casefold():
        relative_path = sample_path
    else:
        relative_path = Path(split) / sample_path
    return frames_root / relative_path, relative_path


def natural_sort_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def list_frame_files(frame_directory: Path) -> list[Path]:
    """List image frames recursively in temporal filename order."""
    if not frame_directory.exists():
        raise FileNotFoundError(
            f"Frame directory does not exist: {frame_directory}"
        )

    if not frame_directory.is_dir():
        raise NotADirectoryError(
            f"Expected a directory: {frame_directory}"
        )

    frame_files = [
        path
        for path in frame_directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in FRAME_EXTENSIONS
    ]

    frame_files.sort(key=natural_sort_key)
    return frame_files


def uniform_sample_indices(number_of_frames: int, target_frames: int) -> np.ndarray:
    if number_of_frames <= 0:
        raise ValueError("number_of_frames must be greater than zero.")
    if target_frames <= 0:
        raise ValueError("target_frames must be greater than zero.")
    if number_of_frames == 1:
        return np.zeros(target_frames, dtype=np.int64)

    values = np.linspace(0, number_of_frames - 1, num=target_frames)
    return np.rint(values).astype(np.int64)


def select_uniform_frames(
    frame_files: Sequence[Path], target_frames: int
) -> tuple[np.ndarray, list[Path]]:
    if not frame_files:
        raise ValueError("No frame files were provided.")
    indices = uniform_sample_indices(len(frame_files), target_frames)
    return indices, [frame_files[int(index)] for index in indices]
