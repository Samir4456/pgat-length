"""Render length-curve figures from a pgat-length metrics JSON.

Produces two PNGs suitable for the thesis proposal:
    1) length_curve_bleu_chrf.png  — BLEU-4 and chrF by reference-length bin,
       with the PGAT-v1 DEV baseline overlaid for comparison.
    2) length_calibration.png       — mean generation length vs mean reference
       length per bin, with the diagonal shown. Documents that pgat-length
       fixes the PGAT-v1 premature-EOS behaviour.

Usage:
    python scripts/10_plot_length_curve.py \\
        --metrics $HOME/outputs/predictions/dev_metrics.json \\
        --output-dir $HOME/outputs/figures

Optional --no-baseline drops the PGAT-v1 lines from the plot.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless (no display on the cluster)
import matplotlib.pyplot as plt


# PGAT-v1 DEV baseline numbers (from the exploratory pg-adaptor project).
# Bin edges match ours (1-6, 7-12, 13-18, 19-24, 25-32).
PGAT_V1_DEV_BASELINE: dict[str, dict[str, float]] = {
    "1-6":   {"samples": 33,  "bleu_4": 19.55, "chrf": 37.19, "mean_reference_words": 4.85,  "mean_generation_words": 5.88},
    "7-12":  {"samples": 242, "bleu_4": 17.74, "chrf": 35.60, "mean_reference_words": 9.87,  "mean_generation_words": 9.54},
    "13-18": {"samples": 166, "bleu_4": 4.19,  "chrf": 25.28, "mean_reference_words": 15.19, "mean_generation_words": 12.87},
    "19-24": {"samples": 58,  "bleu_4": 4.73,  "chrf": 25.63, "mean_reference_words": 20.93, "mean_generation_words": 14.93},
    "25-32": {"samples": 20,  "bleu_4": 1.32,  "chrf": 21.67, "mean_reference_words": 26.85, "mean_generation_words": 16.05},
}
PGAT_V1_DEV_OVERALL = {"bleu_4": 9.79, "chrf": 29.19}

BIN_ORDER = ("1-6", "7-12", "13-18", "19-24", "25-32")


@dataclass
class SeriesPoint:
    bin_label: str
    bleu_4: float
    chrf: float
    mean_ref_words: float
    mean_gen_words: float
    samples: int


def collect_series(metrics: dict, baseline: dict[str, dict[str, float]] | None = None) -> tuple[list[SeriesPoint], list[SeriesPoint] | None]:
    ours: list[SeriesPoint] = []
    for label in BIN_ORDER:
        b = metrics["bins"].get(label)
        if b is None:
            continue
        ours.append(
            SeriesPoint(
                bin_label=label,
                bleu_4=float(b["bleu_4"]),
                chrf=float(b["chrf"]),
                mean_ref_words=float(b["mean_reference_words"]),
                mean_gen_words=float(b["mean_generation_words"]),
                samples=int(b["samples"]),
            )
        )
    baseline_points: list[SeriesPoint] | None = None
    if baseline is not None:
        baseline_points = []
        for label in BIN_ORDER:
            b = baseline.get(label)
            if b is None:
                continue
            baseline_points.append(
                SeriesPoint(
                    bin_label=label,
                    bleu_4=float(b["bleu_4"]),
                    chrf=float(b["chrf"]),
                    mean_ref_words=float(b["mean_reference_words"]),
                    mean_gen_words=float(b["mean_generation_words"]),
                    samples=int(b["samples"]),
                )
            )
    return ours, baseline_points


def plot_length_curve(
    ours: list[SeriesPoint],
    baseline: list[SeriesPoint] | None,
    output_path: Path,
    candidate_label: str = "pgat-length",
    baseline_label: str = "PGAT-v1",
) -> None:
    fig, (ax_bleu, ax_chrf) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    x = list(range(len(BIN_ORDER)))
    xtick_labels = [f"{lbl}\n(n={p.samples})" for lbl, p in zip(BIN_ORDER, ours)]

    ours_bleu = [p.bleu_4 for p in ours]
    ours_chrf = [p.chrf for p in ours]

    ax_bleu.plot(x, ours_bleu, marker="o", linewidth=2.4, label=candidate_label, color="#1f77b4")
    ax_chrf.plot(x, ours_chrf, marker="o", linewidth=2.4, label=candidate_label, color="#1f77b4")

    if baseline is not None:
        base_bleu = [p.bleu_4 for p in baseline]
        base_chrf = [p.chrf for p in baseline]
        ax_bleu.plot(x, base_bleu, marker="s", linewidth=2.0, linestyle="--", label=baseline_label, color="#d62728")
        ax_chrf.plot(x, base_chrf, marker="s", linewidth=2.0, linestyle="--", label=baseline_label, color="#d62728")

    for ax, ylab, title in (
        (ax_bleu, "BLEU-4 (corpus)", "BLEU-4 by reference length"),
        (ax_chrf, "chrF (corpus)", "chrF by reference length"),
    ):
        ax.set_xlabel("reference length bin (whitespace tokens)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Length-stratified translation quality on PHOENIX14T DEV",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_length_calibration(
    ours: list[SeriesPoint],
    baseline: list[SeriesPoint] | None,
    output_path: Path,
    candidate_label: str = "pgat-length",
    baseline_label: str = "PGAT-v1",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))

    ref_ours = [p.mean_ref_words for p in ours]
    gen_ours = [p.mean_gen_words for p in ours]
    ax.plot(
        ref_ours,
        gen_ours,
        marker="o",
        linewidth=2.4,
        label=candidate_label,
        color="#1f77b4",
    )
    for p in ours:
        ax.annotate(
            p.bin_label,
            xy=(p.mean_ref_words, p.mean_gen_words),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#1f77b4",
        )

    if baseline is not None:
        ref_base = [p.mean_ref_words for p in baseline]
        gen_base = [p.mean_gen_words for p in baseline]
        ax.plot(
            ref_base,
            gen_base,
            marker="s",
            linewidth=2.0,
            linestyle="--",
            label=baseline_label,
            color="#d62728",
        )
        for p in baseline:
            ax.annotate(
                p.bin_label,
                xy=(p.mean_ref_words, p.mean_gen_words),
                xytext=(4, -12),
                textcoords="offset points",
                fontsize=8,
                color="#d62728",
            )

    # Diagonal (perfect length calibration).
    limit = max(
        max(p.mean_ref_words for p in ours) if ours else 0,
        max(p.mean_gen_words for p in ours) if ours else 0,
        max(p.mean_ref_words for p in baseline) if baseline else 0,
        max(p.mean_gen_words for p in baseline) if baseline else 0,
    ) + 2.0
    ax.plot([0, limit], [0, limit], color="grey", linewidth=1.0, linestyle=":", label="reference = generation")

    ax.set_xlabel("mean reference length (words)")
    ax.set_ylabel("mean generated length (words)")
    ax.set_title("Generation length vs reference length (by bin)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True, help="pgat-length dev_metrics.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-label", type=str, default="pgat-length")
    parser.add_argument("--baseline-label", type=str, default="PGAT-v1")
    parser.add_argument("--no-baseline", action="store_true", help="Omit the PGAT-v1 comparison lines.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    baseline = None if args.no_baseline else PGAT_V1_DEV_BASELINE

    ours, base = collect_series(metrics, baseline)

    curve_path = args.output_dir / "length_curve_bleu_chrf.png"
    calibration_path = args.output_dir / "length_calibration.png"

    plot_length_curve(
        ours=ours,
        baseline=base,
        output_path=curve_path,
        candidate_label=args.candidate_label,
        baseline_label=args.baseline_label,
    )
    plot_length_calibration(
        ours=ours,
        baseline=base,
        output_path=calibration_path,
        candidate_label=args.candidate_label,
        baseline_label=args.baseline_label,
    )

    overall_line = ""
    overall = metrics.get("overall", {})
    if overall:
        overall_line = (
            f"overall: BLEU-4={overall.get('bleu_4', 0):.2f} "
            f"chrF={overall.get('chrf', 0):.2f} "
            f"(baseline PGAT-v1: BLEU-4={PGAT_V1_DEV_OVERALL['bleu_4']}, chrF={PGAT_V1_DEV_OVERALL['chrf']})"
        )
    print(f"length curve   -> {curve_path}")
    print(f"length calibr. -> {calibration_path}")
    if overall_line:
        print(overall_line)


if __name__ == "__main__":
    main()
