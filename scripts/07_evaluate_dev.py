"""Step 07 — generate DEV (or TEST) predictions and compute metrics.

Usage:
    # DEV (default):
    sbatch slurm/evaluate.sbatch --split dev
    # or interactively:
    python scripts/07_evaluate_dev.py --split dev

    # TEST (locked; only after DEV is frozen):
    sbatch slurm/evaluate.sbatch --split test --allow-test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pgat_length.evaluation.generate import generate_dev  # noqa: E402


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return payload


def _expand(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value))


def project_or_absolute(root: Path, value: str) -> Path:
    expanded = Path(_expand(str(value)))
    return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, default=PROJECT_ROOT / "configs" / "data.yaml")
    parser.add_argument("--model-config", type=Path, default=PROJECT_ROOT / "configs" / "model.yaml")
    parser.add_argument("--translation-config", type=Path, default=PROJECT_ROOT / "configs" / "translation.yaml")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-beams", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml(args.data_config.resolve())
    model_cfg = load_yaml(args.model_config.resolve())
    translation_cfg = load_yaml(args.translation_config.resolve())

    checkpoint_path = args.checkpoint or project_or_absolute(
        PROJECT_ROOT, data_cfg["paths"]["output_root"]
    ) / "translation" / str(translation_cfg["checkpoint"]["best_filename"])

    output_dir = args.output_dir or project_or_absolute(
        PROJECT_ROOT, data_cfg["paths"]["output_root"]
    ) / "predictions"

    generation_cfg = translation_cfg.get("generation", {})
    num_beams = args.num_beams or int(generation_cfg.get("num_beams", 3))
    max_new = args.max_new_tokens or int(generation_cfg.get("max_new_tokens", 96))
    no_repeat = int(generation_cfg.get("no_repeat_ngram_size", 3))
    length_penalty = float(generation_cfg.get("length_penalty", 1.0))

    generate_dev(
        data_config=data_cfg,
        model_config=model_cfg,
        translation_config=translation_cfg,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        split=args.split,
        allow_test=args.allow_test,
        max_new_tokens=max_new,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat,
        length_penalty=length_penalty,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
