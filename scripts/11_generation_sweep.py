"""Sweep beam-search generation configs on the SAME translation checkpoint.

Uses the current translation_best.pt without any retraining. For each
(length_penalty, num_beams) combo, regenerates DEV predictions in a
dedicated output directory, computes metrics, and prints a compact
comparison table.

Total time per config: ~30 min (519 DEV samples at ~5 samples/s).

Usage:
    python scripts/11_generation_sweep.py \\
        --length-penalties 0.7 1.0 1.2 \\
        --num-beams-list 3 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length-penalties", nargs="+", type=float, default=[0.7, 1.0, 1.2])
    parser.add_argument("--num-beams-list", nargs="+", type=int, default=[3])
    parser.add_argument("--output-root", type=Path, default=Path.home() / "outputs" / "sweep")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_one(length_penalty: float, num_beams: int, output_dir: Path, overwrite: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "07_evaluate_dev.py"),
        "--split", "dev",
        "--length-penalty", str(length_penalty),
        "--num-beams", str(num_beams),
        "--output-dir", str(output_dir),
    ]
    if overwrite:
        cmd.append("--overwrite")
    print(f"\n===== lp={length_penalty} beams={num_beams} -> {output_dir} =====", flush=True)
    subprocess.run(cmd, check=True)
    metrics_path = output_dir / "dev_metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    results: list[tuple[float, int, dict]] = []
    for lp in args.length_penalties:
        for beams in args.num_beams_list:
            subdir = args.output_root / f"lp{lp}_beams{beams}"
            metrics = run_one(lp, beams, subdir, args.overwrite)
            results.append((lp, beams, metrics))

    # Print compact table.
    print("\n" + "=" * 96)
    print(f"{'lp':>5} {'beams':>6} {'BLEU-4':>8} {'chrF':>8} {'ROUGE-L':>8} "
          f"{'gen_len':>8} {'ref_len':>8}")
    print("-" * 96)
    for lp, beams, metrics in results:
        o = metrics["overall"]
        print(
            f"{lp:>5.2f} {beams:>6} "
            f"{o['bleu_4']:>8.4f} {o['chrf']:>8.4f} {o['rouge_l_f1']:>8.4f} "
            f"{o['mean_generation_words']:>8.2f} {o['mean_reference_words']:>8.2f}"
        )
    print("=" * 96)

    # Save summary.
    summary_path = args.output_root / "sweep_summary.json"
    summary_path.write_text(
        json.dumps(
            [
                {
                    "length_penalty": lp,
                    "num_beams": beams,
                    "overall": m["overall"],
                }
                for lp, beams, m in results
            ],
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nSweep summary: {summary_path}")


if __name__ == "__main__":
    main()
