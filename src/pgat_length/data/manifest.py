"""PHOENIX14T manifest loading.

Responsibility:
- Load the cached manifest pickle containing per-sample:
  uid, sample_id, split, translation, source_num_frames, sampled_num_frames,
  and cache paths for pose/vision/text.
- Return typed rows for downstream sampling and feature loading.

Public API (to implement):
- load_manifest(path: Path) -> pd.DataFrame
- split_records(frame: pd.DataFrame, split: str) -> pd.DataFrame
- expected_split_sizes() -> dict[str, int]  # {"train": 7096, "dev": 519, "test": 642}
"""

raise NotImplementedError("pgat_length.data.manifest: implement in step 01")
