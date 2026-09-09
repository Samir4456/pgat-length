"""MPNet sentence-embedding cache for the alignment text encoder.

Design: one 768-D vector per sample, cached as a single npz per split.
Storage is trivial (7096 * 768 * 4 = ~22 MB train + tiny dev/test), and
alignment training reads the whole cache into memory once.

Why offline: the alignment loop freezes the text encoder anyway, so caching
avoids running mpnet forward every batch and removes an extra
sentence-transformers dependency from the training critical path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MpnetTextConfig:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dim: int = 768
    normalize: bool = False


def compute_mpnet_embeddings(
    texts: list[str],
    config: MpnetTextConfig,
    hf_cache: Path | None = None,
    batch_size: int = 64,
    device: str | None = None,
) -> np.ndarray:
    """Return [N, embedding_dim] float32 mpnet sentence embeddings."""
    import torch
    from sentence_transformers import SentenceTransformer

    device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = str(hf_cache) if hf_cache else None
    model = SentenceTransformer(
        config.model_name,
        cache_folder=cache_dir,
        device=device_str,
    )
    with torch.inference_mode():
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=config.normalize,
            show_progress_bar=True,
        )
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != config.embedding_dim:
        raise RuntimeError(
            f"unexpected embedding shape: {arr.shape}, want (N, {config.embedding_dim})"
        )
    return arr


def save_mpnet_cache(path: Path, uids: list[str], embeddings: np.ndarray, config: MpnetTextConfig) -> None:
    """Atomic npz save with (uids, embeddings, metadata) keys."""
    import json
    import os as _os

    if len(uids) != embeddings.shape[0]:
        raise ValueError(f"uid/embedding count mismatch: {len(uids)} vs {embeddings.shape[0]}")
    meta = {
        "model_name": config.model_name,
        "embedding_dim": config.embedding_dim,
        "normalize": config.normalize,
    }
    # Numpy's savez_compressed auto-appends .npz if the filename does not
    # already end in .npz, so pass an open file handle to bypass that rename.
    tmp_path = path.with_suffix(path.suffix + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "wb") as handle:
        np.savez_compressed(
            handle,
            uids=np.asarray(uids),
            embeddings=embeddings.astype(np.float32),
            meta=np.asarray(json.dumps(meta, ensure_ascii=False, sort_keys=True)),
        )
    _os.replace(tmp_path, path)


def load_mpnet_cache(path: Path) -> tuple[list[str], np.ndarray, dict]:
    """Return (uids, embeddings [N, D] float32, metadata dict)."""
    import json

    if not path.is_file():
        raise FileNotFoundError(f"mpnet cache missing: {path}")
    loaded = np.load(path, allow_pickle=True)
    uids = [str(u) for u in loaded["uids"]]
    embeddings = np.asarray(loaded["embeddings"], dtype=np.float32)
    meta_raw = str(loaded["meta"])
    meta = json.loads(meta_raw) if meta_raw else {}
    return uids, embeddings, meta
