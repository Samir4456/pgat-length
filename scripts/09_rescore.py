"""Re-score existing DEV/TEST predictions without regenerating.

Reads a JSONL of {uid, reference, prediction, ...} records and writes
a fresh <split>_metrics.json with overall + five-bin metrics.

Usage:
    python scripts/09_rescore.py \\
        --predictions outputs/predictions/dev_predictions.jsonl \\
        --output outputs/predictions/dev_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pgat_length.evaluation.five_bin import per_bin_metrics  # noqa: E402
from pgat_length.evaluation.metrics import corpus_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    with args.predictions.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    print(f"loaded {len(rows)} predictions from {args.predictions}")

    overall = corpus_metrics(rows)
    bins = per_bin_metrics(rows)
    payload = {
        "split": rows[0].get("split", "unknown") if rows else "unknown",
        "samples": len(rows),
        "predictions_file": str(args.predictions.resolve()),
        "overall": overall,
        "bins": bins,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
