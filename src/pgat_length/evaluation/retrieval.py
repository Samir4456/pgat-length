"""Video->Text retrieval metrics for stage 1 alignment.

Given L2-normalized [N, D] video and text embeddings on the DEV set,
compute similarity matrix and derive:
    - V->T Recall@{1, 5, 10}
    - Median rank of the positive
    - Mean rank of the positive
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RetrievalMetrics:
    r_at_1: float
    r_at_5: float
    r_at_10: float
    median_rank: float
    mean_rank: float

    def to_dict(self) -> dict[str, float]:
        return {
            "v2t_recall_at_1": self.r_at_1,
            "v2t_recall_at_5": self.r_at_5,
            "v2t_recall_at_10": self.r_at_10,
            "v2t_median_rank": self.median_rank,
            "v2t_mean_rank": self.mean_rank,
        }


def video_to_text_retrieval(video: torch.Tensor, text: torch.Tensor) -> RetrievalMetrics:
    """Compute V->T retrieval metrics.

    Both embeddings must be L2-normalized [N, D]. The diagonal is the
    positive: sample i's video should retrieve sample i's text.
    """
    if video.shape != text.shape:
        raise ValueError(f"shape mismatch: video={video.shape} text={text.shape}")
    N = video.shape[0]
    with torch.no_grad():
        sim = (video @ text.T).float().cpu().numpy()  # [N, N]
    # Rank of the positive is 1 + number of negatives with higher similarity.
    positive = np.diag(sim).reshape(-1, 1)  # [N, 1]
    higher = (sim > positive).sum(axis=1)  # [N]
    ranks = higher + 1  # 1-indexed rank
    r1 = float((ranks <= 1).mean()) * 100.0
    r5 = float((ranks <= 5).mean()) * 100.0
    r10 = float((ranks <= 10).mean()) * 100.0
    median = float(np.median(ranks))
    mean = float(np.mean(ranks))
    return RetrievalMetrics(r_at_1=r1, r_at_5=r5, r_at_10=r10, median_rank=median, mean_rank=mean)
