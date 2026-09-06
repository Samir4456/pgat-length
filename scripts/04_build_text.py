"""Step 04 — mBART tokenization cache for target German references.

CPU-only; no dependency on plans / spatial / motion. Runs in ~1 minute
on both splits combined.

Usage:
    sbatch slurm/build_features.sbatch scripts/04_build_text.py --all
    # or interactively (no GPU needed):
    python scripts/04_build_text.py --all
    python scripts/04_build_text.py --split dev
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

from pgat_length.features.shards import (  # noqa: E402
    TEXT_SECTIONS,
    fingerprint_from_sections,
    uid_order_fingerprint,
    validate_saved_text_shard,
    write_jsonl_atomic,
    write_text_shard,
)
from pgat_length.features.text import (  # noqa: E402
    TextTokenizerConfig,
    load_mbart_tokenizer,
    tokenize_translation,
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


def _write_bank_metadata(
    output_root: Path,
    tokenizer_name: str,
    src_lang: str,
    tgt_lang: str,
    max_target_tokens: int,
    fingerprint: str,
    combined_config: dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "metadata.json"
    payload = {
        "format_version": 2,
        "bank": "text",
        "tokenizer_name": tokenizer_name,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "max_target_tokens": max_target_tokens,
        "config_fingerprint": fingerprint,
        "config": combined_config,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key in (
            "bank",
            "tokenizer_name",
            "src_lang",
            "tgt_lang",
            "max_target_tokens",
            "config_fingerprint",
        ):
            if existing.get(key) != payload[key]:
                raise RuntimeError(f"text bank metadata mismatch on {key}: {path}")
        return
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_existing_index(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.exists():
        return {}
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                entries[(str(row["shard"]), int(row["offset"]))] = row
    return entries


def process_split(
    split: str,
    manifest: pd.DataFrame,
    tokenizer,
    text_config: TextTokenizerConfig,
    output_root: Path,
    shard_size: int,
    config_fingerprint: str,
    overwrite: bool,
) -> dict[str, Any]:
    rows = manifest.loc[manifest["split"].eq(split), ["uid", "translation"]].reset_index(drop=True)
    if rows.empty:
        raise RuntimeError(f"no samples for split {split!r}")
    index_path = output_root / split / "index.jsonl"
    existing_index = _load_existing_index(index_path)
    output_index: list[dict[str, Any]] = []
    created = 0
    reused = 0

    for shard_number, start in enumerate(range(0, len(rows), shard_size)):
        stop = min(start + shard_size, len(rows))
        shard_rows = rows.iloc[start:stop]
        uids = shard_rows["uid"].astype(str).tolist()
        shard_path = output_root / split / f"text-{shard_number:05d}.safetensors"
        expected_metadata = {
            "format_version": "2",
            "bank": "text",
            "split": split,
            "sample_count": str(len(uids)),
            "max_target_tokens": str(text_config.max_target_tokens),
            "tokenizer_name": text_config.model_name,
            "src_lang": text_config.src_lang,
            "tgt_lang": text_config.tgt_lang,
            "config_fingerprint": config_fingerprint,
            "uid_fingerprint": uid_order_fingerprint(uids),
        }
        if shard_path.exists() and not overwrite:
            validate_saved_text_shard(
                shard_path, len(uids), text_config.max_target_tokens, expected_metadata
            )
            for offset, (_, row) in enumerate(shard_rows.iterrows()):
                previous = existing_index.get((shard_path.name, offset))
                output_index.append(
                    previous
                    if previous is not None and previous.get("uid") == str(row["uid"])
                    else {
                        "uid": str(row["uid"]),
                        "split": split,
                        "shard": shard_path.name,
                        "offset": offset,
                        "bank": "text",
                        "config_fingerprint": config_fingerprint,
                    }
                )
            reused += 1
            print(f"Reused: {shard_path}")
            continue

        input_ids_list: list[np.ndarray] = []
        attn_list: list[np.ndarray] = []
        length_list: list[int] = []
        for _, row in shard_rows.iterrows():
            ids, mask, real_len = tokenize_translation(
                tokenizer, str(row["translation"]), text_config.max_target_tokens
            )
            input_ids_list.append(ids)
            attn_list.append(mask)
            length_list.append(real_len)
        arrays = {
            "input_ids": np.stack(input_ids_list, axis=0).astype(np.int32),
            "attention_mask": np.stack(attn_list, axis=0).astype(np.bool_),
            "text_length": np.asarray(length_list, dtype=np.int32),
        }
        shard_path, was_created = write_text_shard(
            output_root=output_root,
            split=split,
            shard_number=shard_number,
            uids=uids,
            arrays=arrays,
            config_fingerprint=config_fingerprint,
            tokenizer_name=text_config.model_name,
            max_target_tokens=text_config.max_target_tokens,
            src_lang=text_config.src_lang,
            tgt_lang=text_config.tgt_lang,
            overwrite=overwrite,
        )
        created += int(was_created)
        reused += int(not was_created)
        for offset, uid in enumerate(uids):
            output_index.append(
                {
                    "uid": uid,
                    "split": split,
                    "shard": shard_path.name,
                    "offset": offset,
                    "bank": "text",
                    "text_length": int(length_list[offset]),
                    "config_fingerprint": config_fingerprint,
                }
            )
        print(f"{'Created' if was_created else 'Reused'}: {shard_path}")

    write_jsonl_atomic(index_path, output_index)
    return {
        "split": split,
        "samples": len(output_index),
        "created_shards": created,
        "reused_shards": reused,
        "index": str(index_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, default=PROJECT_ROOT / "configs" / "data.yaml")
    parser.add_argument("--features-config", type=Path, default=PROJECT_ROOT / "configs" / "features.yaml")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--split", choices=("train", "dev", "test"))
    scope.add_argument("--all", action="store_true", help="Process train and dev in one run.")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.split and not args.all:
        args.all = True
    splits: list[str] = ["train", "dev"] if args.all else [args.split]

    data_cfg = load_yaml(args.data_config.resolve())
    features_cfg = load_yaml(args.features_config.resolve())
    text_cfg_raw = features_cfg.get("text")
    if not text_cfg_raw:
        raise RuntimeError("features.yaml has no 'text' section")
    text_config = TextTokenizerConfig(
        model_name=str(text_cfg_raw.get("model_name", "facebook/mbart-large-cc25")),
        src_lang=str(text_cfg_raw.get("src_lang", "de_DE")),
        tgt_lang=str(text_cfg_raw.get("tgt_lang", "de_DE")),
        max_target_tokens=int(text_cfg_raw.get("max_target_tokens", 96)),
    )
    config_fingerprint = fingerprint_from_sections(data_cfg, features_cfg, TEXT_SECTIONS)
    combined = {
        "data": data_cfg,
        "features": {k: features_cfg[k] for k in TEXT_SECTIONS if k in features_cfg},
    }

    feature_root = project_or_absolute(PROJECT_ROOT, data_cfg["paths"]["feature_root"])
    hf_cache = project_or_absolute(PROJECT_ROOT, data_cfg["paths"]["hf_cache"])
    manifest_path = project_or_absolute(PROJECT_ROOT, data_cfg["paths"]["manifest"])
    output_root = args.output_root.resolve() if args.output_root else feature_root / "text"

    _write_bank_metadata(
        output_root,
        text_config.model_name,
        text_config.src_lang,
        text_config.tgt_lang,
        text_config.max_target_tokens,
        config_fingerprint,
        combined,
    )

    manifest = pd.read_pickle(manifest_path)
    shard_size = int(features_cfg["storage"]["shard_size"])

    print(f"Loading tokenizer: {text_config.model_name}")
    tokenizer = load_mbart_tokenizer(text_config, hf_cache=hf_cache)

    summaries: list[dict[str, Any]] = []
    for split in splits:
        summary = process_split(
            split=split,
            manifest=manifest,
            tokenizer=tokenizer,
            text_config=text_config,
            output_root=output_root,
            shard_size=shard_size,
            config_fingerprint=config_fingerprint,
            overwrite=args.overwrite,
        )
        summaries.append(summary)

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
