"""Step 18 - stage 1 alignment training with a precomputed MPNet text encoder.

Prereq: `python scripts/17_build_text_mpnet.py --all` has produced the
mpnet_text_{train,dev}.npz caches under $HOME/pgat-cache/features/text_mpnet/.

Usage:
    sbatch slurm/train_alignment_mpnet.sbatch --allow-full
    # or interactively:
    python scripts/18_train_alignment_mpnet.py --allow-full
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

from pgat_length.training.alignment_mpnet_loop import train_alignment_mpnet  # noqa: E402


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
    parser.add_argument("--features-config", type=Path, default=PROJECT_ROOT / "configs" / "features.yaml")
    parser.add_argument("--model-config", type=Path, default=PROJECT_ROOT / "configs" / "model_v2.yaml")
    parser.add_argument("--alignment-config", type=Path, default=PROJECT_ROOT / "configs" / "alignment_mpnet.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-full", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml(args.data_config.resolve())
    features_cfg = load_yaml(args.features_config.resolve())
    model_cfg = load_yaml(args.model_config.resolve())
    alignment_cfg = load_yaml(args.alignment_config.resolve())

    output_root = args.output_dir or project_or_absolute(
        PROJECT_ROOT, data_cfg["paths"]["output_root"]
    ) / "alignment"

    train_alignment_mpnet(
        data_config=data_cfg,
        features_config=features_cfg,
        model_config=model_cfg,
        alignment_config=alignment_cfg,
        output_dir=output_root,
        allow_full=args.allow_full,
    )


if __name__ == "__main__":
    main()
