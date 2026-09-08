"""Step 12 — safety-gated TEST pipeline: build TEST features + evaluate v2.

TEST is locked in the project's split policy. This script:
  1. Refuses to run without --allow-test.
  2. Confirms the v2 checkpoint exists and is the intended one.
  3. Builds TEST features in order (plans, spatial, motion, text) — each
     step is resume-safe by shard fingerprint.
  4. Runs evaluation with the v2 configs and prints the final metrics.

Prefer running this via slurm/test.sbatch (created alongside), not directly.

Usage:
    sbatch slurm/test.sbatch
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-test", action="store_true", required=True)
    parser.add_argument("--length-penalty", type=float, default=0.7)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path.home() / "outputs" / "translation" / "translation_v2_best.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=Path.home() / "outputs" / "predictions_test_v2")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip TEST feature building (use if already built).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"v2 checkpoint not found: {args.checkpoint}")
    print(f"v2 checkpoint: {args.checkpoint}")
    print(f"output dir:    {args.output_dir}")
    print(f"generation:    beams={args.num_beams} length_penalty={args.length_penalty}")

    python = sys.executable

    if not args.skip_features:
        # 1) plans
        run([python, str(PROJECT_ROOT / "scripts" / "01_build_plans.py"),
             "--split", "test", "--all"])
        # 2) spatial
        run([python, str(PROJECT_ROOT / "scripts" / "02_extract_spatial.py"),
             "--split", "test", "--allow-full"])
        # 3) motion
        run([python, str(PROJECT_ROOT / "scripts" / "03_extract_motion.py"),
             "--split", "test", "--allow-full"])
        # 4) text (train/dev already cached; --all does not include test)
        run([python, str(PROJECT_ROOT / "scripts" / "04_build_text.py"), "--split", "test"])

    # 5) evaluate on TEST with the v2 configs.
    run([
        python, str(PROJECT_ROOT / "scripts" / "07_evaluate_dev.py"),
        "--split", "test",
        "--allow-test",
        "--model-config", str(PROJECT_ROOT / "configs" / "model_v2.yaml"),
        "--translation-config", str(PROJECT_ROOT / "configs" / "translation_v2.yaml"),
        "--checkpoint", str(args.checkpoint),
        "--output-dir", str(args.output_dir),
        "--length-penalty", str(args.length_penalty),
        "--num-beams", str(args.num_beams),
        "--overwrite",
    ])

    metrics_path = args.output_dir / "test_metrics.json"
    print(f"\nDone. Final TEST metrics at: {metrics_path}")


if __name__ == "__main__":
    main()
