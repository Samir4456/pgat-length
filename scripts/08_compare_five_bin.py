"""Step 08 — render the five-bin length-stratified comparison table.

Takes the DEV metrics JSON produced by scripts/07_evaluate_dev.py and prints
a compact per-bin BLEU-4 / chrF / ROUGE-L table plus the overall row. If
--baseline is passed to another metrics JSON, both are shown side by side
so the length curves can be compared visually.

Usage:
    python scripts/08_compare_five_bin.py \\
        --candidate outputs/predictions/dev_metrics.json \\
        [--baseline external_predictions/pgat_v1_dev_metrics.json]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pgat_length.evaluation.five_bin import BIN_EDGES

BIN_LABELS = [item.label for item in BIN_EDGES]


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    return parser.parse_args()


def _row(bin_label: str, metrics: dict | None) -> dict:
    if metrics is None:
        return {"length_bin": bin_label, "samples": "-", "bleu_4": "-", "chrf": "-", "rouge_l_f1": "-",
                "mean_ref_words": "-", "mean_gen_words": "-"}
    return {
        "length_bin": bin_label,
        "samples": metrics.get("samples", "-"),
        "bleu_4": round(float(metrics.get("bleu_4", 0.0)), 4),
        "chrf": round(float(metrics.get("chrf", 0.0)), 4),
        "rouge_l_f1": round(float(metrics.get("rouge_l_f1", 0.0)), 4),
        "mean_ref_words": round(float(metrics.get("mean_reference_words", 0.0)), 2),
        "mean_gen_words": round(float(metrics.get("mean_generation_words", 0.0)), 2),
    }


def main() -> None:
    args = parse_args()
    cand = load_metrics(args.candidate)
    base = load_metrics(args.baseline) if args.baseline else None

    cand_bins = cand.get("bins", {})
    base_bins = base.get("bins", {}) if base else {}

    print("=" * 96)
    title = f"Candidate: {args.candidate.name}"
    if args.baseline:
        title += f"  vs  Baseline: {args.baseline.name}"
    print(title)
    print("=" * 96)

    header = (
        f"{'bin':>7} {'n':>4} "
        f"{'BLEU4_cand':>10} {'chrF_cand':>10} {'ROUGE_cand':>10} "
    )
    if base is not None:
        header += (
            f"{'BLEU4_base':>10} {'chrF_base':>10} {'ROUGE_base':>10} "
            f"{'ΔBLEU4':>+8} {'ΔchrF':>+8}"
        )
    print(header)

    rows_out: list[dict] = []
    for label in BIN_LABELS:
        c = cand_bins.get(label)
        b = base_bins.get(label) if base else None
        c_row = _row(label, c)
        line = (
            f"{label:>7} {c_row['samples']:>4} "
            f"{c_row['bleu_4']:>10} {c_row['chrf']:>10} {c_row['rouge_l_f1']:>10} "
        )
        if base is not None:
            b_row = _row(label, b)
            if isinstance(c_row["bleu_4"], (int, float)) and isinstance(b_row["bleu_4"], (int, float)):
                d_bleu = c_row["bleu_4"] - b_row["bleu_4"]
                d_chrf = c_row["chrf"] - b_row["chrf"]
                dbleu_s = f"{d_bleu:+.2f}"
                dchrf_s = f"{d_chrf:+.2f}"
            else:
                dbleu_s = dchrf_s = "-"
            line += (
                f"{b_row['bleu_4']:>10} {b_row['chrf']:>10} {b_row['rouge_l_f1']:>10} "
                f"{dbleu_s:>+8} {dchrf_s:>+8}"
            )
            row_dict = {"length_bin": label}
            for k, v in c_row.items():
                if k != "length_bin":
                    row_dict[f"{k}_cand"] = v
            for k, v in b_row.items():
                if k != "length_bin":
                    row_dict[f"{k}_base"] = v
            rows_out.append(row_dict)
        else:
            rows_out.append(c_row)
        print(line)

    print("-" * 96)
    overall = cand.get("overall", {})
    o_row = _row("ALL", overall)
    line = (
        f"{'ALL':>7} {o_row['samples']:>4} "
        f"{o_row['bleu_4']:>10} {o_row['chrf']:>10} {o_row['rouge_l_f1']:>10} "
    )
    if base is not None:
        b_overall = base.get("overall", {})
        b_o = _row("ALL", b_overall)
        d_bleu = o_row['bleu_4'] - b_o['bleu_4']
        d_chrf = o_row['chrf'] - b_o['chrf']
        line += (
            f"{b_o['bleu_4']:>10} {b_o['chrf']:>10} {b_o['rouge_l_f1']:>10} "
            f"{d_bleu:>+8.2f} {d_chrf:>+8.2f}"
        )
    print(line)
    print("=" * 96)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8-sig", newline="") as handle:
            fieldnames = list(rows_out[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"CSV: {args.csv}")


if __name__ == "__main__":
    main()
