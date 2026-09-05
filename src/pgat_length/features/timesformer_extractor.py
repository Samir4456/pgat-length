"""Frozen TimeSformer motion feature extractor.

Motion features are unaffected by variable K_temporal — every sample gets
a fixed 8 motion clips regardless of video length. The extractor produces
[8, 768] fp16 features plus [8] normalized center positions per sample.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, TimesformerModel

from pgat_length.features.dino_extractor import _select_pooled, torch_dtype_from_name


def motion_clip_positions(
    positions: int = 64,
    clips: int = 8,
    frames_per_clip: int = 8,
    clip_span: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Build overlapping temporal windows and eight sampled frames per clip.

    Returns:
        clip_positions: [clips, frames_per_clip] int16 — indices into the 64
            sampled source positions.
        centers: [clips] float16 — normalized center of each clip in [0, 1].
    """
    if min(positions, clips, frames_per_clip, clip_span) <= 0:
        raise ValueError("motion clip dimensions must be positive")
    if clip_span < frames_per_clip or clip_span > positions:
        raise ValueError("clip_span must be between frames_per_clip and positions")
    starts = np.rint(np.linspace(0, positions - clip_span, clips)).astype(np.int16)
    offsets = np.rint(np.linspace(0, clip_span - 1, frames_per_clip)).astype(np.int16)
    clip_positions = starts[:, None] + offsets[None, :]
    centers = clip_positions.astype(np.float32).mean(axis=1)
    if positions > 1:
        centers /= float(positions - 1)
    if np.any(np.diff(starts) >= clip_span):
        raise AssertionError("motion windows do not overlap")
    return clip_positions.astype(np.int16), centers.astype(np.float16)


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


@dataclass(frozen=True)
class MotionFeatureResult:
    features: np.ndarray  # [8, 768] fp16
    centers: np.ndarray   # [8] fp16


class TimesformerMotionExtractor:
    def __init__(
        self,
        model_name: str,
        feature_dim: int = 768,
        batch_size: int = 2,
        device: str = "cuda",
        compute_dtype: str = "float16",
        hf_cache: Path | None = None,
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA required for TimeSformer extraction")
        self.device = torch.device(device)
        self.compute_dtype = torch_dtype_from_name(compute_dtype)
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        cache_dir = str(hf_cache) if hf_cache else None
        self.processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = TimesformerModel.from_pretrained(
            model_name,
            dtype=self.compute_dtype,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def _prepare_clip(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(0)
        if pixel_values.ndim != 5 or pixel_values.shape[0] != 1:
            raise ValueError(
                f"TimeSformer processor must return [1,T,C,H,W], got {tuple(pixel_values.shape)}"
            )
        return pixel_values

    def extract(
        self,
        frame_files: Sequence[Path],
        plan: dict[str, np.ndarray],
    ) -> MotionFeatureResult:
        frame_indices = np.asarray(plan["frame_indices"])
        clip_positions, centers = motion_clip_positions()
        clips: list[torch.Tensor] = []
        for positions in clip_positions:
            images = [
                _load_rgb(frame_files[int(frame_indices[int(position)])])
                for position in positions
            ]
            try:
                clips.append(self._prepare_clip(images))
            finally:
                for image in images:
                    image.close()
        output = np.zeros((8, self.feature_dim), dtype=np.float16)
        for start in range(0, len(clips), self.batch_size):
            pixel_values = torch.cat(clips[start : start + self.batch_size], dim=0).to(
                self.device, non_blocking=True
            )
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=self.compute_dtype,
                enabled=self.device.type == "cuda",
            ):
                model_output = self.model(pixel_values=pixel_values)
            pooled = _select_pooled(model_output, self.feature_dim)
            values = pooled.float().cpu().numpy().astype(np.float16)
            output[start : start + len(values)] = values
        if not np.isfinite(output).all():
            raise ValueError("TimeSformer produced non-finite motion features")
        return MotionFeatureResult(output, centers)

    def close(self) -> None:
        del self.model, self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
