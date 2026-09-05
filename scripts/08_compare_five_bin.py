"""Step 08 — five-bin length-stratified comparison table.

Compares PGAT-v2 predictions against external baseline predictions
(TSPNet, NSLT, GASLT, GFSLT-VLP) placed under external_predictions/.
Emits a summary CSV and a length-curve plot ready for the proposal.

Usage:
    python scripts/08_compare_five_bin.py --candidate outputs/predictions/dev_pgat_v2.jsonl
"""

if __name__ == "__main__":
    from pgat_length.evaluation.five_bin import per_bin_metrics  # noqa: F401
    from pgat_length.evaluation.bootstrap import paired_bootstrap  # noqa: F401
    raise SystemExit("TODO: implement CLI that renders the comparison table + curve")
