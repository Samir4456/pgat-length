"""Frozen DINOv2 spatial feature extractor with variable K_temporal per sample.

For each plan, the extractor emits an output tensor of shape [K_MAX, 4, D]
where K = plan.k_temporal is the real segment count and rows [K:] are exact
zeros. Views: 0 = global (always valid), 1 = left hand, 2 = right hand,
3 = mouth (each valid iff the plan says so).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model

from pgat_length.features.shards import K_MAX


SPATIAL_VIEWS = ("global", "left_hand", "right_hand", "mouth")


def torch_dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float32": torch.float32, "fp32": torch.float32,
    }
    key = name.strip().lower()
    if key not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[key]


def _select_pooled(output: Any, expected_dim: int) -> torch.Tensor:
    pooled = getattr(output, "pooler_output", None)
    if not isinstance(pooled, torch.Tensor):
        hidden = getattr(output, "last_hidden_state", None)
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise TypeError("DINO output missing pooler_output and last_hidden_state")
        pooled = hidden[:, 0]
    if pooled.ndim != 2 or pooled.shape[1] != expected_dim:
        raise ValueError(f"pooled features must be [B,{expected_dim}], got {tuple(pooled.shape)}")
    return pooled


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def build_variable_k_views(
    frame_files: Sequence[Path],
    plan: dict[str, np.ndarray],
) -> tuple[list[Image.Image], list[tuple[int, int]], np.ndarray, int]:
    """Return (images, destinations, validity[K_MAX,4], k_temporal).

    Only builds views for the first k_temporal real segments. Segments in
    [k_temporal, K_MAX) are left with validity=False and their crop rows in
    the output tensor stay at zero.
    """
    frame_indices = np.asarray(plan["frame_indices"])
    anchors = np.asarray(plan["anchor_positions"])
    boxes = np.asarray(plan["crop_boxes"])
    local_valid = np.asarray(plan["region_valid"], dtype=np.bool_)
    k_temporal = int(np.asarray(plan["k_temporal"]))

    if frame_indices.shape != (64,):
        raise ValueError(f"plan frame_indices must be (64,), got {frame_indices.shape}")
    if anchors.shape != (K_MAX,):
        raise ValueError(f"plan anchors must be ({K_MAX},), got {anchors.shape}")
    if boxes.shape != (K_MAX, 3, 4):
        raise ValueError(f"plan crop_boxes must be ({K_MAX},3,4), got {boxes.shape}")
    if local_valid.shape != (K_MAX, 3):
        raise ValueError(f"plan region_valid must be ({K_MAX},3), got {local_valid.shape}")
    if not 1 <= k_temporal <= K_MAX:
        raise ValueError(f"k_temporal out of range: {k_temporal}")

    validity = np.zeros((K_MAX, 4), dtype=np.bool_)
    images: list[Image.Image] = []
    destinations: list[tuple[int, int]] = []
    image_cache: dict[int, Image.Image] = {}
    try:
        for segment in range(k_temporal):
            source_index = int(frame_indices[int(anchors[segment])])
            if source_index < 0 or source_index >= len(frame_files):
                raise IndexError(f"source frame index out of range: {source_index}")
            if source_index not in image_cache:
                image_cache[source_index] = _load_rgb(frame_files[source_index])
            full = image_cache[source_index]
            images.append(full.copy())
            destinations.append((segment, 0))
            validity[segment, 0] = True
            for region in range(3):
                if not local_valid[segment, region]:
                    continue
                left, top, right, bottom = (int(v) for v in boxes[segment, region])
                left = max(0, min(left, full.width - 1))
                top = max(0, min(top, full.height - 1))
                right = max(left + 1, min(right, full.width))
                bottom = max(top + 1, min(bottom, full.height))
                images.append(full.crop((left, top, right, bottom)))
                destinations.append((segment, region + 1))
                validity[segment, region + 1] = True
    finally:
        for image in image_cache.values():
            image.close()
    return images, destinations, validity, k_temporal


@dataclass(frozen=True)
class SpatialFeatureResult:
    features: np.ndarray  # [K_MAX, 4, D] fp16
    valid: np.ndarray     # [K_MAX, 4] bool
    k_temporal: int


class Dinov2SpatialExtractor:
    def __init__(
        self,
        model_name: str,
        feature_dim: int = 768,
        batch_size: int = 16,
        device: str = "cuda",
        compute_dtype: str = "bfloat16",
        hf_cache: Path | None = None,
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA required for DINOv2 extraction")
        self.device = torch.device(device)
        self.compute_dtype = torch_dtype_from_name(compute_dtype)
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        cache_dir = str(hf_cache) if hf_cache else None
        self.processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = Dinov2Model.from_pretrained(
            model_name,
            dtype=self.compute_dtype,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def extract(
        self,
        frame_files: Sequence[Path],
        plan: dict[str, np.ndarray],
    ) -> SpatialFeatureResult:
        images, destinations, validity, k_temporal = build_variable_k_views(frame_files, plan)
        output = np.zeros((K_MAX, 4, self.feature_dim), dtype=np.float16)
        try:
            for start in range(0, len(images), self.batch_size):
                batch = images[start : start + self.batch_size]
                inputs = self.processor(images=batch, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)
                with torch.inference_mode(), torch.autocast(
                    device_type=self.device.type,
                    dtype=self.compute_dtype,
                    enabled=self.device.type == "cuda",
                ):
                    model_output = self.model(pixel_values=pixel_values)
                pooled = _select_pooled(model_output, self.feature_dim)
                values = pooled.float().cpu().numpy().astype(np.float16)
                for destination, value in zip(destinations[start : start + len(values)], values):
                    output[destination] = value
        finally:
            for image in images:
                image.close()
        if not np.isfinite(output).all():
            raise ValueError("DINOv2 produced non-finite spatial features")
        if np.any(output[~validity] != 0):
            raise AssertionError("invalid views must remain zero")
        return SpatialFeatureResult(output, validity, k_temporal)

    def close(self) -> None:
        del self.model, self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
