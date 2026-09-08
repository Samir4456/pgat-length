"""Step 15 - stage 2 translation training with Qwen2.5-3B QLoRA decoder.

Warm-starts the PGAT visual encoder from alignment_v2_best.pt (or any
compatible alignment checkpoint) and fine-tunes PGAT + projection + LoRA
adapters. Qwen base weights stay frozen and NF4-quantized on GPU.

Usage:
    sbatch slurm/train_translation_qwen.sbatch --allow-full
    # or interactively:
    python scripts/15_train_translation_qwen.py --allow-full
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

from pgat_length.training.qwen_translation_loop import train_qwen_translation  # noqa: E402


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
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-full", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml(args.data_config.resolve())
    model_cfg = load_yaml(args.model_config.resolve())
    translation_cfg = load_yaml(args.translation_config.resolve())

    output_root = args.output_dir or project_or_absolute(
        PROJECT_ROOT, data_cfg["paths"]["output_root"]
    ) / "translation"

    train_qwen_translation(
        data_config=data_cfg,
        model_config=model_cfg,
        translation_config=translation_cfg,
        output_dir=output_root,
        allow_full=args.allow_full,
    )


if __name__ == "__main__":
    main()
