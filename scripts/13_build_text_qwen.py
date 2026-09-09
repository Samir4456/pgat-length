"""Step 13 - Qwen tokenization cache for PHOENIX14T (train/dev; test gated).

Writes one npz per split with:
    uids                    : (N,) str
    prompt_input_ids        : (P,) int32           # shared prompt template
    prompt_attention_mask   : (P,) bool
    target_input_ids        : (N, max_target) int32
    target_attention_mask   : (N, max_target) bool
    target_length           : (N,) int32
    meta                    : json string (model_name, template, sizes)

Design intent: parallel to 04_build_text.py's mBART bank, but standalone
(the mBART shard machinery is schema-bound to input_ids/attention_mask
and does not fit the prompt+target layout). No fingerprinting -- a change
in the prompt template or model name changes the file contents; rerun
with --overwrite to rebuild.

Usage:
    python scripts/13_build_text_qwen.py --all
    python scripts/13_build_text_qwen.py --split test --allow-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pgat_length.features.text_qwen import (  # noqa: E402
    QwenTextTokenizerConfig,
    build_prompt_ids,
    load_qwen_tokenizer,
    tokenize_target,
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
    tokenizer,
    prompt_ids: np.ndarray,
    prompt_mask: np.ndarray,
    config: QwenTextTokenizerConfig,
    output_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        print(f"Reused: {output_path}")
        cached = np.load(output_path, allow_pickle=True)
        return {"split": split, "samples": int(cached["target_length"].shape[0]), "reused": True, "path": str(output_path)}

    rows = manifest.loc[manifest["split"].eq(split), ["uid", "translation"]].reset_index(drop=True)
    if rows.empty:
        raise RuntimeError(f"no samples for split {split!r}")

    uids: list[str] = []
    target_ids_list: list[np.ndarray] = []
    target_mask_list: list[np.ndarray] = []
    target_len_list: list[int] = []
    for _, row in rows.iterrows():
        uids.append(str(row["uid"]))
        t_ids, t_mask, t_len = tokenize_target(
            tokenizer, str(row["translation"]), config.max_target_tokens
        )
        target_ids_list.append(t_ids)
        target_mask_list.append(t_mask)
        target_len_list.append(t_len)

    meta = {
        "model_name": config.model_name,
        "max_prompt_tokens": config.max_prompt_tokens,
        "max_target_tokens": config.max_target_tokens,
        "system_prompt": config.system_prompt,
        "user_prompt": config.user_prompt,
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": int(tokenizer.eos_token_id),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Numpy's savez_compressed auto-appends .npz if the filename does not
    # already end in .npz, so pass an open file handle to bypass that rename.
    tmp_path = output_path.with_suffix(output_path.suffix + ".partial")
    with open(tmp_path, "wb") as handle:
        np.savez_compressed(
            handle,
            uids=np.asarray(uids),
            prompt_input_ids=prompt_ids.astype(np.int32),
            prompt_attention_mask=prompt_mask.astype(np.bool_),
            target_input_ids=np.stack(target_ids_list, axis=0).astype(np.int32),
            target_attention_mask=np.stack(target_mask_list, axis=0).astype(np.bool_),
            target_length=np.asarray(target_len_list, dtype=np.int32),
            meta=np.asarray(json.dumps(meta, ensure_ascii=False, sort_keys=True)),
        )
    os.replace(tmp_path, output_path)
    print(f"Created: {output_path}")
    return {"split": split, "samples": len(uids), "reused": False, "path": str(output_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, default=PROJECT_ROOT / "configs" / "data.yaml")
    parser.add_argument("--features-config", type=Path, default=PROJECT_ROOT / "configs" / "features.yaml")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--split", choices=("train", "dev", "test"))
    scope.add_argument("--all", action="store_true", help="Process train and dev in one run.")
    parser.add_argument("--allow-test", action="store_true", help="Required to process the TEST split.")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-prompt-tokens", type=int, default=96)
    parser.add_argument("--max-target-tokens", type=int, default=96)
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
    text_cfg_raw = features_cfg.get("text_qwen") or {}
    config = QwenTextTokenizerConfig(
        model_name=str(text_cfg_raw.get("model_name", args.model_name)),
        max_prompt_tokens=int(text_cfg_raw.get("max_prompt_tokens", args.max_prompt_tokens)),
        max_target_tokens=int(text_cfg_raw.get("max_target_tokens", args.max_target_tokens)),
    )

    feature_root = project_or_absolute(PROJECT_ROOT, data_cfg["paths"]["feature_root"])
    hf_cache = project_or_absolute(PROJECT_ROOT, data_cfg["paths"]["hf_cache"])
    manifest_path = project_or_absolute(PROJECT_ROOT, data_cfg["paths"]["manifest"])
    output_root = args.output_root.resolve() if args.output_root else feature_root / "text_qwen"
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {config.model_name}")
    tokenizer = load_qwen_tokenizer(config, hf_cache=hf_cache)
    prompt_ids, prompt_mask = build_prompt_ids(tokenizer, config)
    print(f"prompt tokens (real): {int(prompt_mask.sum())} / {config.max_prompt_tokens}")

    manifest = pd.read_pickle(manifest_path)

    summaries: list[dict[str, Any]] = []
    for split in splits:
        output_path = output_root / f"qwen_text_{split}.npz"
        summary = process_split(
            split=split,
            manifest=manifest,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            config=config,
            output_path=output_path,
            overwrite=args.overwrite,
        )
        summaries.append(summary)

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
