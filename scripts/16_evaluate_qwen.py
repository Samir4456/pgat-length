"""Step 16 - DEV/TEST generation and scoring for the Qwen QLoRA checkpoint.

Loads translation_qwen_best.pt (PGAT + projection + LoRA state), reattaches
the Qwen base + LoRA adapters, and runs deterministic beam search on the
requested split. Writes predictions.jsonl + metrics.json in the same layout
as scripts/07_evaluate_dev.py.

Usage:
    sbatch slurm/evaluate_qwen.sbatch                       # DEV by default
    sbatch slurm/evaluate_qwen.sbatch --split test --allow-test
    # or interactively:
    python scripts/16_evaluate_qwen.py --split dev
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

from pgat_length.evaluation.qwen_generate import generate_qwen  # noqa: E402


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, default=PROJECT_ROOT / "configs" / "data.yaml")
    parser.add_argument("--model-config", type=Path, default=PROJECT_ROOT / "configs" / "model_qwen.yaml")
    parser.add_argument("--translation-config", type=Path, default=PROJECT_ROOT / "configs" / "translation_qwen.yaml")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path.home() / "outputs" / "translation" / "translation_qwen_best.pt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml(args.data_config.resolve())
    model_cfg = load_yaml(args.model_config.resolve())
    translation_cfg = load_yaml(args.translation_config.resolve())

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        default_name = "predictions_qwen" if args.split == "dev" else "predictions_qwen_test"
        output_dir = (
            project_or_absolute(PROJECT_ROOT, data_cfg["paths"]["output_root"]) / default_name
        )

    generate_qwen(
        data_config=data_cfg,
        model_config=model_cfg,
        translation_config=translation_cfg,
        checkpoint_path=args.checkpoint.resolve(),
        output_dir=output_dir,
        split=args.split,
        allow_test=args.allow_test,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        length_penalty=args.length_penalty,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
