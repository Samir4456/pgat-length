"""Step 17 - MPNet sentence-embedding cache for alignment (train/dev; test gated).

Writes one npz per split with (uids, embeddings [N, 768], meta JSON).

Usage:
    python scripts/17_build_text_mpnet.py --all
    python scripts/17_build_text_mpnet.py --split test --allow-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pgat_length.features.text_mpnet import (  # noqa: E402
    MpnetTextConfig,
    compute_mpnet_embeddings,
    save_mpnet_cache,
)


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


def process_split(
    split: str,
    manifest: pd.DataFrame,
    config: MpnetTextConfig,
    output_path: Path,
    hf_cache: Path,
    batch_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        print(f"Reused: {output_path}")
        return {"split": split, "reused": True, "path": str(output_path)}

    rows = manifest.loc[manifest["split"].eq(split), ["uid", "translation"]].reset_index(drop=True)
    if rows.empty:
        raise RuntimeError(f"no samples for split {split!r}")
    uids = [str(u) for u in rows["uid"].tolist()]
    texts = [str(t) for t in rows["translation"].tolist()]

    embeddings = compute_mpnet_embeddings(
        texts,
        config=config,
        hf_cache=hf_cache,
        batch_size=batch_size,
    )
    save_mpnet_cache(output_path, uids=uids, embeddings=embeddings, config=config)
    print(f"Created: {output_path} shape={embeddings.shape}")
    return {"split": split, "reused": False, "path": str(output_path), "samples": len(uids)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, default=PROJECT_ROOT / "configs" / "data.yaml")
    parser.add_argument("--features-config", type=Path, default=PROJECT_ROOT / "configs" / "features.yaml")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--split", choices=("train", "dev", "test"))
    scope.add_argument("--all", action="store_true", help="Process train and dev in one run.")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.split and not args.all:
        args.all = True
    splits: list[str] = ["train", "dev"] if args.all else [args.split]
    if "test" in splits and not args.allow_test:
        raise RuntimeError("--allow-test is required to process the TEST split")

    data_cfg = load_yaml(args.data_config.resolve())
    features_cfg = load_yaml(args.features_config.resolve())
    text_cfg_raw = features_cfg.get("text_mpnet") or {}
    config = MpnetTextConfig(
        model_name=str(text_cfg_raw.get("model_name", "sentence-transformers/all-mpnet-base-v2")),
        embedding_dim=int(text_cfg_raw.get("embedding_dim", 768)),
        normalize=bool(text_cfg_raw.get("normalize", False)),
    )

    feature_root = project_or_absolute(PROJECT_ROOT, data_cfg["paths"]["feature_root"])
    hf_cache = project_or_absolute(PROJECT_ROOT, data_cfg["paths"]["hf_cache"])
    manifest_path = project_or_absolute(PROJECT_ROOT, data_cfg["paths"]["manifest"])
    output_root = args.output_root.resolve() if args.output_root else feature_root / "text_mpnet"

    manifest = pd.read_pickle(manifest_path)

    summaries: list[dict[str, Any]] = []
    for split in splits:
        output_path = output_root / f"mpnet_text_{split}.npz"
        summary = process_split(
            split=split,
            manifest=manifest,
            config=config,
            output_path=output_path,
            hf_cache=hf_cache,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
        summaries.append(summary)

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
