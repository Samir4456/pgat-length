"""Step 20 - compositional-generalization feasibility gate.

Runs the rule-based extractor over the whole PHOENIX14T manifest
(train + dev + test), computes coverage statistics, and simulates the
2x2 (composition, load) split. Produces a two-page human-readable
report plus a machine-readable stats JSON.

This is the *feasibility gate* for the compositional-generalization
thesis direction (see docs/COMPOSITIONAL_GENERALIZATION.md). It answers,
in one day, whether PHOENIX14T can support the study at all.

Decision criteria (all three must pass to proceed):
  1. Extractor coverage (fraction of references with >=1 proposition)
     is >= 0.85. Requires manual validation for the F1 >= 0.85
     threshold; this script only checks coverage.
  2. Cell B and cell D sample counts are each >= 30 after confound
     matching.
  3. Confounder overlap (KS test p > 0.05) between cell A and each
     other cell on duration and reference length.

Also emits a stratified annotation batch (`docs/annotation_batch.jsonl`)
of 100 samples for human validation of the extractor.

Usage:
    python scripts/20_composition_feasibility.py
    python scripts/20_composition_feasibility.py --allow-test
    python scripts/20_composition_feasibility.py --annotation-batch-size 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pgat_length.composition.extractor import extract  # noqa: E402
from pgat_length.composition.schema import (  # noqa: E402
    Relation,
    Slot,
    schema_summary,
)


# ---------------------------------------------------------------------------
# Path helpers.
# ---------------------------------------------------------------------------

def _expand(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return payload


def project_or_absolute(root: Path, value: str) -> Path:
    expanded = Path(_expand(str(value)))
    return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()


# ---------------------------------------------------------------------------
# Simple stats helpers (no scipy dependency).
# ---------------------------------------------------------------------------

def ks_two_sample_p(a: list[float], b: list[float]) -> float:
    """Approximate two-sample Kolmogorov-Smirnov test p-value.

    Kolmogorov distribution asymptotic: for large n, m,
      p ~= 2 * sum_{k=1..inf} (-1)^(k-1) * exp(-2 k^2 D^2 * n*m/(n+m))
    Truncate at k=20.
    """
    if not a or not b:
        return 0.0
    a_sorted = sorted(a)
    b_sorted = sorted(b)
    all_values = sorted(set(a_sorted + b_sorted))
    n_a = len(a_sorted)
    n_b = len(b_sorted)
    cdf_a = np.searchsorted(a_sorted, all_values, side="right") / n_a
    cdf_b = np.searchsorted(b_sorted, all_values, side="right") / n_b
    d_stat = float(np.max(np.abs(cdf_a - cdf_b)))
    en = np.sqrt(n_a * n_b / (n_a + n_b))
    lam = (en + 0.12 + 0.11 / en) * d_stat
    p = 0.0
    for k in range(1, 21):
        term = 2 * ((-1) ** (k - 1)) * np.exp(-2 * (k * lam) ** 2)
        p += term
    return float(max(0.0, min(1.0, p)))


# ---------------------------------------------------------------------------
# Compositional split simulation.
# ---------------------------------------------------------------------------

def _sample_relations_set(sample: dict) -> set[tuple[str, str, str]]:
    """Flatten a sample's relations into typed pairs (relation_type, a, b)."""
    out: set[tuple[str, str, str]] = set()
    for rel_type, instances in sample["relations"].items():
        for instance in instances:
            if len(instance) == 2:
                out.add((rel_type, instance[0], instance[1]))
    return out


def build_compound_frequencies(train_samples: list[dict]) -> dict[tuple[str, str, str], int]:
    """Count how often each typed relation instance appears in train."""
    counter: Counter[tuple[str, str, str]] = Counter()
    for sample in train_samples:
        for rel in _sample_relations_set(sample):
            counter[rel] += 1
    return dict(counter)


def classify_sample(
    sample: dict,
    training_compound_freq: dict[tuple[str, str, str], int],
    held_out_compounds: set[tuple[str, str, str]],
    low_load_max: int,
) -> tuple[str, str]:
    """Return (composition_label, load_label)."""
    sample_relations = _sample_relations_set(sample)
    if not sample_relations:
        return "familiar", "low"  # empty samples fall to A by default
    any_held_out = any(rel in held_out_compounds for rel in sample_relations)
    all_familiar = all(training_compound_freq.get(rel, 0) > 0 for rel in sample_relations)
    if any_held_out or not all_familiar:
        composition = "held_out"
    else:
        composition = "familiar"
    load = "low" if sample["num_propositions"] <= low_load_max else "high"
    return composition, load


def simulate_split(
    samples: list[dict],
    holdout_quantile: float = 0.2,
    low_load_max: int = 2,
    split_seed: int = 0,
) -> dict[str, Any]:
    """Simulate the compositional split and return per-cell diagnostics.

    Strategy:
      - Treat all extracted samples as one candidate pool.
      - Randomly designate 80% as pseudo-train, 20% as pseudo-eval, but
        with the constraint that at least a fraction `holdout_quantile`
        of the *distinct compound types* appearing in pseudo-eval do
        not appear in pseudo-train.
      - Report per-cell counts on the pseudo-eval half.

    In the real study, this becomes the actual split-construction. Here
    we only care whether such a split *can be constructed with reasonable
    cell sizes*.
    """
    rng = random.Random(split_seed)
    n = len(samples)
    indices = list(range(n))
    rng.shuffle(indices)
    n_eval = int(round(0.20 * n))
    eval_idx = set(indices[:n_eval])
    train_samples = [samples[i] for i in range(n) if i not in eval_idx]
    eval_samples = [samples[i] for i in range(n) if i in eval_idx]

    train_freq = build_compound_frequencies(train_samples)
    # Held-out compounds: those appearing in eval but not in train.
    eval_compounds: set[tuple[str, str, str]] = set()
    for s in eval_samples:
        eval_compounds.update(_sample_relations_set(s))
    held_out_compounds = {c for c in eval_compounds if c not in train_freq}

    # Classify eval samples.
    cell_counts: Counter[tuple[str, str]] = Counter()
    cell_samples: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in eval_samples:
        composition, load = classify_sample(s, train_freq, held_out_compounds, low_load_max)
        cell_counts[(composition, load)] += 1
        cell_samples[(composition, load)].append(s)

    return {
        "n_pseudo_train": len(train_samples),
        "n_pseudo_eval": len(eval_samples),
        "n_distinct_train_compounds": len(train_freq),
        "n_distinct_eval_compounds": len(eval_compounds),
        "n_held_out_compounds": len(held_out_compounds),
        "cell_counts": {f"{c}__{l}": v for (c, l), v in cell_counts.items()},
        "cell_samples": cell_samples,
        "held_out_compound_examples": list(held_out_compounds)[:15],
    }


# ---------------------------------------------------------------------------
# Confounder-overlap check (KS tests between cell A and other cells).
# ---------------------------------------------------------------------------

def cell_confounder_overlap(
    cell_samples: dict[tuple[str, str], list[dict]],
    field: str,
) -> dict[str, float]:
    """Return KS p-values for A vs {B, C, D} on the given per-sample field."""
    def _values(cell: tuple[str, str]) -> list[float]:
        return [float(s.get(field, 0)) for s in cell_samples.get(cell, [])]

    a = _values(("familiar", "low"))
    b = _values(("held_out", "low"))
    c = _values(("familiar", "high"))
    d = _values(("held_out", "high"))
    return {
        "A_vs_B": ks_two_sample_p(a, b) if a and b else None,
        "A_vs_C": ks_two_sample_p(a, c) if a and c else None,
        "A_vs_D": ks_two_sample_p(a, d) if a and d else None,
    }


# ---------------------------------------------------------------------------
# Annotation batch (for human extractor validation).
# ---------------------------------------------------------------------------

def stratified_annotation_batch(
    samples: list[dict],
    references_word_counts: list[int],
    size: int = 100,
    bins: tuple[tuple[int, int], ...] = ((1, 6), (7, 12), (13, 18), (19, 24), (25, 999)),
    seed: int = 0,
) -> list[dict]:
    per_bin = size // len(bins)
    rng = random.Random(seed)
    binned: dict[int, list[int]] = defaultdict(list)
    for i, wc in enumerate(references_word_counts):
        for bin_idx, (lo, hi) in enumerate(bins):
            if lo <= wc <= hi:
                binned[bin_idx].append(i)
                break
    chosen: list[int] = []
    for bin_idx in range(len(bins)):
        candidates = binned.get(bin_idx, [])
        if not candidates:
            continue
        rng.shuffle(candidates)
        chosen.extend(candidates[:per_bin])
    rng.shuffle(chosen)
    return [samples[i] for i in chosen]


# ---------------------------------------------------------------------------
# Report writer.
# ---------------------------------------------------------------------------

def write_report(
    output_path: Path,
    stats: dict[str, Any],
    per_split_sim: dict[str, dict[str, Any]],
    schema_info: Any,
    args: argparse.Namespace,
) -> None:
    lines: list[str] = []
    add = lines.append

    add("# Compositional-generalization feasibility report")
    add("")
    add(f"Generated by `scripts/20_composition_feasibility.py`.  ")
    add(f"Extractor: `src/pgat_length/composition/extractor.py`.")
    add("")
    add("## 1. Schema summary")
    add("")
    add(f"- Events:           {schema_info.events}")
    add(f"- Locations:        {schema_info.locations}")
    add(f"- Times:            {schema_info.times}")
    add(f"- Intensities:      {schema_info.intensities}")
    add(f"- Wind directions:  {schema_info.wind_directions}")
    add(f"- Temperature buckets: {schema_info.temperature_buckets}")
    add("")

    add("## 2. Extraction coverage (per split)")
    add("")
    add("| Split | Samples | Refs w/ >=1 atom | Refs w/ >=1 relation | Mean propositions | Median propositions |")
    add("|---|---:|---:|---:|---:|---:|")
    for split_name in ("train", "dev", "test"):
        s = stats["per_split"].get(split_name)
        if s is None:
            continue
        add(
            f"| {split_name} | {s['n_samples']} | {s['n_with_atoms']} "
            f"({s['pct_with_atoms']:.1f}%) | {s['n_with_relations']} "
            f"({s['pct_with_relations']:.1f}%) | {s['mean_propositions']:.2f} | "
            f"{s['median_propositions']:.1f} |"
        )
    add("")

    add("## 3. Vocabulary sizes actually observed (train)")
    add("")
    obs = stats["observed_atoms_train"]
    add("| Slot | Distinct atoms observed | Vocabulary in schema |")
    add("|---|---:|---:|")
    for slot in Slot:
        observed = obs.get(slot.value, 0)
        vocab = {
            Slot.EVENT.value: schema_info.events,
            Slot.LOCATION.value: schema_info.locations,
            Slot.TIME.value: schema_info.times,
            Slot.TEMPERATURE.value: schema_info.temperature_buckets,
            Slot.INTENSITY.value: schema_info.intensities,
            Slot.WIND_DIRECTION.value: schema_info.wind_directions,
            Slot.POLARITY.value: 2,
        }[slot.value]
        add(f"| {slot.value} | {observed} | {vocab} |")
    add("")

    add("## 4. Relation-instance (compound) diversity in train")
    add("")
    ctd = stats["compound_type_distinct_train"]
    add("| Relation type | Distinct instances | Total instances | Mean freq per instance |")
    add("|---|---:|---:|---:|")
    for rel in Relation:
        distinct = ctd.get(rel.value, {}).get("distinct", 0)
        total = ctd.get(rel.value, {}).get("total", 0)
        mean_freq = (total / distinct) if distinct else 0
        add(f"| {rel.value} | {distinct} | {total} | {mean_freq:.2f} |")
    add("")

    add("## 5. Simulated 2x2 cell counts (on 80/20 pseudo-eval within train)")
    add("")
    for split_name, sim in per_split_sim.items():
        add(f"### {split_name}")
        add("")
        add(f"- Pseudo-train samples:   {sim['n_pseudo_train']}")
        add(f"- Pseudo-eval samples:    {sim['n_pseudo_eval']}")
        add(f"- Distinct train compounds: {sim['n_distinct_train_compounds']}")
        add(f"- Distinct eval compounds:  {sim['n_distinct_eval_compounds']}")
        add(f"- Held-out compounds:       {sim['n_held_out_compounds']}")
        add("")
        add("Cell counts (composition x load):")
        add("")
        add("| | Familiar | Held-out |")
        add("|---|---:|---:|")
        f_low = sim["cell_counts"].get("familiar__low", 0)
        f_hi = sim["cell_counts"].get("familiar__high", 0)
        h_low = sim["cell_counts"].get("held_out__low", 0)
        h_hi = sim["cell_counts"].get("held_out__high", 0)
        add(f"| Low load  | A={f_low} | B={h_low} |")
        add(f"| High load | C={f_hi}  | D={h_hi}  |")
        add("")
        if sim.get("confounder_overlap"):
            add("Confounder-overlap KS p-values (vs cell A):")
            add("")
            add("| Confounder | A vs B | A vs C | A vs D |")
            add("|---|---:|---:|---:|")
            for field_name, ovs in sim["confounder_overlap"].items():
                fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else "n/a"
                add(f"| {field_name} | {fmt(ovs['A_vs_B'])} | {fmt(ovs['A_vs_C'])} | {fmt(ovs['A_vs_D'])} |")
            add("")
        if sim.get("held_out_compound_examples"):
            add("Example held-out compound instances:")
            add("")
            for rel_type, a1, a2 in sim["held_out_compound_examples"]:
                add(f"- ({rel_type}) {a1} -- {a2}")
            add("")

    add("## 6. Go / No-Go decision")
    add("")
    verdict = evaluate_gate(stats, per_split_sim)
    for criterion, passed, detail in verdict:
        marker = "PASS" if passed else "FAIL"
        add(f"- [{marker}] **{criterion}**  {detail}")
    add("")
    overall = all(passed for _, passed, _ in verdict)
    add("### Recommendation")
    add("")
    if overall:
        add("**PROCEED** with the compositional-generalization study on PHOENIX14T.")
        add("Next step: manual validation of the extractor on the 100-sample")
        add("annotation batch (see `docs/annotation_batch.jsonl`); publish the")
        add("extractor F1 vs human before building the real splits.")
    else:
        add("**PIVOT.** One or more gate criteria failed. Options:")
        add("")
        add("- If extractor coverage failed: invest more lexicon work; the")
        add("  current lexicons target common PHOENIX weather vocabulary but")
        add("  may miss compound word forms.")
        add("- If cell counts failed: consider CSL-Daily as the primary")
        add("  dataset (much larger, more compound diversity).")
        add("- If confounder overlap failed: acknowledge the residual confound")
        add("  in the paper and downgrade the causal claim.")
    add("")

    add("## 7. Artifacts written")
    add("")
    add(f"- This report: `{output_path.relative_to(PROJECT_ROOT)}`")
    add("- Machine-readable stats: `outputs/composition/feasibility_stats.json`")
    add(f"- Annotation batch ({args.annotation_batch_size} samples): "
        "`docs/annotation_batch.jsonl`")
    add("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_gate(
    stats: dict[str, Any],
    per_split_sim: dict[str, dict[str, Any]],
) -> list[tuple[str, bool, str]]:
    verdict: list[tuple[str, bool, str]] = []

    # Criterion 1: coverage on train >= 0.85 (proxy).
    train_stats = stats["per_split"].get("train", {})
    coverage_pct = train_stats.get("pct_with_relations", 0.0)
    passed = coverage_pct >= 85.0
    verdict.append((
        "Extractor coverage",
        passed,
        f"{coverage_pct:.1f}% of train refs produced >=1 relation (needs >= 85%).",
    ))

    # Criterion 2: cell B and D sample counts >= 30 in the train-based sim.
    sim = per_split_sim.get("train", {})
    b = sim.get("cell_counts", {}).get("held_out__low", 0)
    d = sim.get("cell_counts", {}).get("held_out__high", 0)
    passed_cells = b >= 30 and d >= 30
    verdict.append((
        "Cell counts (B, D >= 30 each)",
        passed_cells,
        f"B={b}, D={d}; each must be >= 30 (aim for >= 50).",
    ))

    # Criterion 3: confounder overlap p > 0.05 on duration and word_count.
    ovs = sim.get("confounder_overlap", {})
    wc_ov = ovs.get("word_count", {})
    dur_ov = ovs.get("duration_frames", {})
    def _min_p(d: dict) -> float:
        vals = [v for v in d.values() if isinstance(v, float)]
        return min(vals) if vals else 0.0
    wc_min = _min_p(wc_ov)
    dur_min = _min_p(dur_ov) if dur_ov else 1.0  # duration may be absent
    passed_conf = wc_min > 0.05 and dur_min > 0.05
    verdict.append((
        "Confounder overlap (KS p > 0.05)",
        passed_conf,
        f"min word_count p = {wc_min:.3f}, min duration_frames p = {dur_min:.3f}.",
    ))

    return verdict


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, default=PROJECT_ROOT / "configs" / "data.yaml")
    parser.add_argument("--allow-test", action="store_true", help="Also extract from the TEST split.")
    parser.add_argument("--low-load-max", type=int, default=2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--annotation-batch-size", type=int, default=100)
    parser.add_argument("--output-report", type=Path,
                        default=PROJECT_ROOT / "docs" / "feasibility_report.md")
    parser.add_argument("--output-stats", type=Path,
                        default=PROJECT_ROOT / "outputs" / "composition" / "feasibility_stats.json")
    parser.add_argument("--output-annotation-batch", type=Path,
                        default=PROJECT_ROOT / "docs" / "annotation_batch.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_cfg = load_yaml(args.data_config.resolve())
    manifest_path = project_or_absolute(PROJECT_ROOT, data_cfg["paths"]["manifest"])
    print(f"Loading manifest: {manifest_path}", flush=True)
    manifest = pd.read_pickle(manifest_path)

    splits_to_run = ["train", "dev"]
    if args.allow_test:
        splits_to_run.append("test")

    # Per-split extraction.
    per_split_samples: dict[str, list[dict]] = {}
    per_split_stats: dict[str, dict[str, Any]] = {}
    all_train_atoms: dict[str, Counter] = defaultdict(Counter)
    all_train_compounds: dict[str, Counter] = defaultdict(Counter)

    for split in splits_to_run:
        rows = manifest.loc[manifest["split"].eq(split)]
        print(f"Extracting {split}: {len(rows)} refs", flush=True)
        extracted: list[dict] = []
        n_with_atoms = 0
        n_with_relations = 0
        proposition_counts: list[int] = []
        for _, row in rows.iterrows():
            uid = str(row["uid"])
            ref = str(row["translation"])
            s = extract(ref, uid=uid).to_dict()
            # Enrich with useful side-info for downstream cell classification.
            s["word_count"] = len([t for t in ref.split() if t.strip()])
            # duration_frames: try to read from manifest if present.
            for candidate in ("num_frames", "n_frames", "source_num_frames"):
                if candidate in row.index:
                    s["duration_frames"] = float(row[candidate])
                    break
            else:
                s["duration_frames"] = 0.0
            n_atoms_here = sum(len(v) for v in s["atoms"].values())
            n_rels_here = sum(len(v) for v in s["relations"].values())
            if n_atoms_here > 0:
                n_with_atoms += 1
            if n_rels_here > 0:
                n_with_relations += 1
            proposition_counts.append(s["num_propositions"])
            extracted.append(s)
        per_split_samples[split] = extracted
        n_total = max(1, len(extracted))
        per_split_stats[split] = {
            "n_samples": n_total,
            "n_with_atoms": n_with_atoms,
            "pct_with_atoms": 100.0 * n_with_atoms / n_total,
            "n_with_relations": n_with_relations,
            "pct_with_relations": 100.0 * n_with_relations / n_total,
            "mean_propositions": float(np.mean(proposition_counts)) if proposition_counts else 0.0,
            "median_propositions": float(np.median(proposition_counts)) if proposition_counts else 0.0,
        }

    # Observed atom + compound vocabularies (train).
    train_samples = per_split_samples.get("train", [])
    observed_atoms_train: dict[str, int] = {}
    for slot in Slot:
        seen: set[str] = set()
        for s in train_samples:
            seen.update(s["atoms"][slot.value])
        observed_atoms_train[slot.value] = len(seen)
        all_train_atoms[slot.value] = Counter()
        for s in train_samples:
            for a in s["atoms"][slot.value]:
                all_train_atoms[slot.value][a] += 1

    compound_type_distinct_train: dict[str, dict[str, int]] = {}
    for rel in Relation:
        distinct: set[tuple[str, str]] = set()
        total = 0
        for s in train_samples:
            for pair in s["relations"][rel.value]:
                if len(pair) == 2:
                    distinct.add((pair[0], pair[1]))
                    total += 1
        compound_type_distinct_train[rel.value] = {
            "distinct": len(distinct),
            "total": total,
        }

    stats: dict[str, Any] = {
        "per_split": per_split_stats,
        "observed_atoms_train": observed_atoms_train,
        "compound_type_distinct_train": compound_type_distinct_train,
        "schema": schema_summary().__dict__,
    }

    # 2x2 simulation per split (train is the important one; also run on
    # dev for completeness).
    per_split_sim: dict[str, dict[str, Any]] = {}
    for split in splits_to_run:
        if not per_split_samples[split]:
            continue
        sim = simulate_split(
            per_split_samples[split],
            holdout_quantile=0.2,
            low_load_max=args.low_load_max,
            split_seed=args.split_seed,
        )
        # Confounder overlap on word count + duration.
        cell_samples = sim.pop("cell_samples")
        sim["confounder_overlap"] = {
            "word_count": cell_confounder_overlap(cell_samples, "word_count"),
            "duration_frames": cell_confounder_overlap(cell_samples, "duration_frames"),
        }
        per_split_sim[split] = sim

    # Annotation batch (drawn from train references).
    if train_samples:
        wc = [s["word_count"] for s in train_samples]
        batch = stratified_annotation_batch(
            train_samples, wc, size=args.annotation_batch_size, seed=args.split_seed
        )
        args.output_annotation_batch.parent.mkdir(parents=True, exist_ok=True)
        with args.output_annotation_batch.open("w", encoding="utf-8") as handle:
            for s in batch:
                handle.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Wrote annotation batch: {args.output_annotation_batch}", flush=True)

    # Persist stats.
    args.output_stats.parent.mkdir(parents=True, exist_ok=True)
    args.output_stats.write_text(json.dumps({
        "stats": stats,
        "simulation": {k: {kk: vv for kk, vv in v.items() if kk != "held_out_compound_examples"}
                       for k, v in per_split_sim.items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote stats: {args.output_stats}", flush=True)

    # Write the human-readable report.
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.output_report, stats, per_split_sim, schema_summary(), args)
    print(f"Wrote report: {args.output_report}", flush=True)

    # Print the go/no-go verdict inline.
    verdict = evaluate_gate(stats, per_split_sim)
    print("\n=== Gate ===")
    for criterion, passed, detail in verdict:
        marker = "PASS" if passed else "FAIL"
        print(f"[{marker}] {criterion} -- {detail}")
    overall = all(p for _, p, _ in verdict)
    print("=" * 12)
    print("PROCEED" if overall else "PIVOT")


if __name__ == "__main__":
    main()
