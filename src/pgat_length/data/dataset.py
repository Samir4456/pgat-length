"""Cached-feature Dataset for training and evaluation.

Responsibility:
- Read precomputed feature shards (spatial, motion, text) by uid.
- For each sample, compute per-video K_temporal = clamp(round(F/8), 12, 40).
- Return raw tensors and metadata; leave batching/padding to the collator.

Public API (to implement):
- class PhoenixCachedDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, feature_root, split, model_config, augment=False)
    def __len__(self) -> int
    def __getitem__(self, index: int) -> dict[str, Any]
"""

raise NotImplementedError("pgat_length.data.dataset: implement in step 03")
