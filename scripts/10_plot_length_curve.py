"""Render length-curve figures from a pgat-length metrics JSON.

Produces (into --output-dir):

  1) length_curve_headline.png   — 3-panel: BLEU-4, chrF, ROUGE-L by
     reference-length bin. The single figure most proposal-ready.
  2) length_curve_all.png        — 2x3 grid: BLEU-1..4, chrF, ROUGE-L.
     Full detail for the appendix.
  3) length_calibration.png      — mean generation length vs mean reference
     length per bin, with the y=x diagonal. Documents that pgat-length
     fixes the PGAT-v1 premature-EOS behaviour.

The PGAT-v1 DEV baseline is overlaid for BLEU-4 and chrF (we have per-bin
DEV values). ROUGE-L per-bin baseline is approximated from TEST-derived
numbers (labelled 'PGAT-v1 (TEST-derived)') so it is visually comparable
but not a strict DEV baseline. BLEU-1..3 per-bin baselines are omitted.

Usage:
    python scripts/10_plot_length_curve.py \\
        --metrics $HOME/outputs/predictions/dev_metrics.json \\
        --output-dir $HOME/outputs/figures
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# PGAT-v1 DEV baseline (from the exploratory pg-adaptor project).
PGAT_V1_DEV_BASELINE: dict[str, dict[str, float]] = {
    "1-6":   {"samples": 33,  "bleu_4": 19.55, "chrf": 37.19, "rouge_l_f1": 34.08, "mean_reference_words": 4.85,  "mean_generation_words": 5.88},
    "7-12":  {"samples": 242, "bleu_4": 17.74, "chrf": 35.60, "rouge_l_f1": 31.62, "mean_reference_words": 9.87,  "mean_generation_words": 9.54},
    "13-18": {"samples": 166, "bleu_4": 4.19,  "chrf": 25.28, "rouge_l_f1": 21.00, "mean_reference_words": 15.19, "mean_generation_words": 12.87},
    "19-24": {"samples": 58,  "bleu_4": 4.73,  "chrf": 25.63, "rouge_l_f1": 21.21, "mean_reference_words": 20.93, "mean_generation_words": 14.93},
    "25-32": {"samples": 20,  "bleu_4": 1.32,  "chrf": 21.67, "rouge_l_f1": 17.64, "mean_reference_words": 26.85, "mean_generation_words": 16.05},
}
PGAT_V1_DEV_OVERALL = {"bleu_4": 9.79, "chrf": 29.19, "rouge_l_f1": 26.68}

# Which metrics have per-bin baseline points to overlay.
BASELINE_METRICS_WITH_BINS: set[str] = {"bleu_4", "chrf", "rouge_l_f1"}

BIN_ORDER = ("1-6", "7-12", "13-18", "19-24", "25-32")

# Metric key -> display label.
METRIC_LABELS: dict[str, str] = {
    "bleu_1": "BLEU-1 (corpus)",
    "bleu_2": "BLEU-2 (corpus)",
    "bleu_3": "BLEU-3 (corpus)",
    "bleu_4": "BLEU-4 (corpus)",
    "chrf": "chrF (corpus)",
    "rouge_l_f1": "ROUGE-L F1",
}


@dataclass
class SeriesPoint:
    bin_label: str
    values: dict[str, float]
    mean_ref_words: float
    mean_gen_words: float
    samples: int


def collect_series(
    metrics: dict,
    baseline: dict[str, dict[str, float]] | None = None,
) -> tuple[list[SeriesPoint], list[SeriesPoint] | None]:
    ours: list[SeriesPoint] = []
    for label in BIN_ORDER:
        b = metrics["bins"].get(label)
        if b is None:
            continue
        ours.append(
            SeriesPoint(
                bin_label=label,
                values={key: float(b[key]) for key in METRIC_LABELS if key in b},
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
                    values={key: float(b[key]) for key in METRIC_LABELS if key in b},
                    mean_ref_words=float(b["mean_reference_words"]),
                    mean_gen_words=float(b["mean_generation_words"]),
                    samples=int(b["samples"]),
                )
            )
    return ours, baseline_points


def _plot_metric_panel(
    ax,
    ours: list[SeriesPoint],
    baseline: list[SeriesPoint] | None,
    metric_key: str,
    candidate_label: str,
    baseline_label: str,
) -> None:
    x = list(range(len(ours)))
    xtick_labels = [f"{p.bin_label}\n(n={p.samples})" for p in ours]
    ours_y = [p.values.get(metric_key, 0.0) for p in ours]
    ax.plot(
        x, ours_y,
        marker="o", linewidth=2.4, label=candidate_label, color="#1f77b4",
    )
    if baseline is not None and metric_key in BASELINE_METRICS_WITH_BINS:
        base_y = [p.values.get(metric_key, 0.0) for p in baseline]
        ax.plot(
            x, base_y,
            marker="s", linewidth=2.0, linestyle="--",
            label=baseline_label, color="#d62728",
        )
    ax.set_xlabel("reference length bin (whitespace tokens)")
    ax.set_ylabel(METRIC_LABELS[metric_key])
    ax.set_title(METRIC_LABELS[metric_key])
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", fontsize=9)


def plot_headline(
    ours: list[SeriesPoint],
    baseline: list[SeriesPoint] | None,
    output_path: Path,
    candidate_label: str,
    baseline_label: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for ax, key in zip(axes, ("bleu_4", "chrf", "rouge_l_f1")):
        _plot_metric_panel(ax, ours, baseline, key, candidate_label, baseline_label)
    fig.suptitle(
        "Length-stratified translation quality on PHOENIX14T DEV",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_all_metrics(
    ours: list[SeriesPoint],
    baseline: list[SeriesPoint] | None,
    output_path: Path,
    candidate_label: str,
    baseline_label: str,
) -> None:
    metric_keys = ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "chrf", "rouge_l_f1"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    for ax, key in zip(axes.flat, metric_keys):
        _plot_metric_panel(ax, ours, baseline, key, candidate_label, baseline_label)
    fig.suptitle(
        "Length-stratified translation quality on PHOENIX14T DEV — all metrics",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_length_calibration(
    ours: list[SeriesPoint],
    baseline: list[SeriesPoint] | None,
    output_path: Path,
    candidate_label: str,
    baseline_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ref_ours = [p.mean_ref_words for p in ours]
    gen_ours = [p.mean_gen_words for p in ours]
    ax.plot(
        ref_ours, gen_ours,
        marker="o", linewidth=2.4, label=candidate_label, color="#1f77b4",
    )
    for p in ours:
        ax.annotate(
            p.bin_label,
            xy=(p.mean_ref_words, p.mean_gen_words),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8, color="#1f77b4",
        )
    if baseline is not None:
        ref_base = [p.mean_ref_words for p in baseline]
        gen_base = [p.mean_gen_words for p in baseline]
        ax.plot(
            ref_base, gen_base,
            marker="s", linewidth=2.0, linestyle="--",
            label=baseline_label, color="#d62728",
        )
        for p in baseline:
            ax.annotate(
                p.bin_label,
                xy=(p.mean_ref_words, p.mean_gen_words),
                xytext=(4, -12),
                textcoords="offset points",
                fontsize=8, color="#d62728",
            )
    limit = max(
        max(p.mean_ref_words for p in ours) if ours else 0,
        max(p.mean_gen_words for p in ours) if ours else 0,
        max(p.mean_ref_words for p in baseline) if baseline else 0,
        max(p.mean_gen_words for p in baseline) if baseline else 0,
    ) + 2.0
    ax.plot([0, limit], [0, limit], color="grey", linewidth=1.0, linestyle=":",
            label="reference = generation")
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
    parser.add_argument("--no-baseline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    baseline = None if args.no_baseline else PGAT_V1_DEV_BASELINE

    ours, base = collect_series(metrics, baseline)

    headline_path = args.output_dir / "length_curve_headline.png"
    all_path = args.output_dir / "length_curve_all.png"
    calibration_path = args.output_dir / "length_calibration.png"

    plot_headline(ours, base, headline_path, args.candidate_label, args.baseline_label)
    plot_all_metrics(ours, base, all_path, args.candidate_label, args.baseline_label)
    plot_length_calibration(ours, base, calibration_path, args.candidate_label, args.baseline_label)

    overall = metrics.get("overall", {})
    print(f"headline    -> {headline_path}")
    print(f"all metrics -> {all_path}")
    print(f"calibration -> {calibration_path}")
    if overall:
        print(
            f"overall: BLEU-4={overall.get('bleu_4', 0):.2f} "
            f"chrF={overall.get('chrf', 0):.2f} "
            f"ROUGE-L={overall.get('rouge_l_f1', 0):.2f} "
            f"(baseline PGAT-v1: BLEU-4={PGAT_V1_DEV_OVERALL['bleu_4']}, "
            f"chrF={PGAT_V1_DEV_OVERALL['chrf']}, "
            f"ROUGE-L={PGAT_V1_DEV_OVERALL['rouge_l_f1']})"
        )


if __name__ == "__main__":
    main()
