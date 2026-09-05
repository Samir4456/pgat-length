"""Five-bin reference-length analysis (paper-compatible).

Bins by whitespace-separated reference tokens:
    1-6, 7-12, 13-18, 19-24, 25-31 (paper) or 25-32 (DEV widened by one).

Public API (to implement):
- BIN_EDGES = (("1-6",1,6), ("7-12",7,12), ("13-18",13,18),
               ("19-24",19,24), ("25-32",25,32))
- def bin_label(count: int) -> str
- def per_bin_metrics(records: Sequence[dict]) -> dict[str, dict[str, Any]]
- def render_bin_table(per_bin: dict) -> pd.DataFrame
"""

raise NotImplementedError("pgat_length.evaluation.five_bin: implement in step 06")
