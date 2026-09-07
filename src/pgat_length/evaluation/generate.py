"""DEV/TEST generation from a translation checkpoint.

Loads translation_best.pt, rebuilds the full model, warm-starts from the
saved trainable_state_dict (mBART base weights come from HF cache), runs
beam-search generation over the requested split, and writes predictions
to a resumable JSONL file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from pgat_length.data.collator import batch_to_device, collate_alignment_batch
from pgat_length.data.dataset import PhoenixCachedDataset
from pgat_length.evaluation.metrics import corpus_metrics
from pgat_length.evaluation.five_bin import per_bin_metrics
from pgat_length.training.checkpoint import load_best
from pgat_length.training.translation_loop import build_model


def _rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(payload, encoding="utf-8")


def _load_saved(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    saved: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                saved[str(row["uid"])] = row
    return saved


def generate_dev(
    *,
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    translation_config: dict[str, Any],
    checkpoint_path: Path,
    output_dir: Path,
    split: str = "dev",
    allow_test: bool = False,
    max_new_tokens: int = 96,
    num_beams: int = 3,
    no_repeat_ngram_size: int = 3,
    length_penalty: float = 1.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate + score. Returns overall + per-bin metrics."""
    if split == "test" and not allow_test:
        raise RuntimeError("TEST evaluation requires --allow-test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None

    # Paths.
    import os
    from os.path import expanduser, expandvars

    def resolve(value: str) -> Path:
        return Path(expanduser(expandvars(str(value)))).resolve()

    feature_root = resolve(data_config["paths"]["feature_root"])
    manifest_path = resolve(data_config["paths"]["manifest"])
    hf_cache = resolve(data_config["paths"]["hf_cache"])

    dataset = PhoenixCachedDataset(
        plans_root=feature_root / "plans",
        spatial_root=feature_root / "spatial",
        motion_root=feature_root / "motion",
        text_root=feature_root / "text",
        manifest_path=manifest_path,
        split=split,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_alignment_batch,
    )

    # Rebuild model then load trainable state.
    model = build_model(model_config, hf_cache=hf_cache, label_smoothing=0.0)
    payload = load_best(checkpoint_path, map_location="cpu")
    state_dict = model.state_dict()
    n_loaded = 0
    for name, tensor in payload["trainable_state_dict"].items():
        if name in state_dict and state_dict[name].shape == tensor.shape:
            state_dict[name] = tensor
            n_loaded += 1
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    print(f"loaded {n_loaded} params from {checkpoint_path}", flush=True)

    tokenizer_name = str(model_config["decoder"]["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=str(hf_cache))
    tokenizer.src_lang = "de_DE"
    tokenizer.tgt_lang = "de_DE"

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{split}_predictions.jsonl"
    if overwrite:
        predictions_path.unlink(missing_ok=True)
    saved = _load_saved(predictions_path)

    started = time.monotonic()
    all_rows: list[dict[str, Any]] = []
    with predictions_path.open("a", encoding="utf-8") as handle:
        for index, batch in enumerate(loader):
            uid = batch["uid"][0]
            if uid in saved:
                all_rows.append(saved[uid])
                continue
            device_batch = batch_to_device(batch, device)
            with torch.inference_mode(), (
                torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast_dtype is not None
                else torch.autocast(device_type="cpu", enabled=False)
            ):
                predictions = model.generate(
                    device_batch,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    length_penalty=length_penalty,
                )
            row = {
                "uid": uid,
                "sample_id": batch["sample_id"][0],
                "split": split,
                "reference": batch["reference"][0],
                "prediction": predictions[0],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            saved[uid] = row
            all_rows.append(row)
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"[{index+1}/{len(dataset)}] {uid} rate={(index+1)/elapsed:.2f} samples/s "
                f"pred={predictions[0][:60]!r}",
                flush=True,
            )

    # Ensure order matches dataset iteration order.
    _rewrite_jsonl(predictions_path, all_rows)

    # Compute metrics.
    overall = corpus_metrics(all_rows)
    bins = per_bin_metrics(all_rows)
    metrics_path = output_dir / f"{split}_metrics.json"
    payload_out = {
        "split": split,
        "checkpoint": str(checkpoint_path),
        "samples": len(all_rows),
        "num_beams": num_beams,
        "max_new_tokens": max_new_tokens,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "length_penalty": length_penalty,
        "overall": overall,
        "bins": bins,
    }
    metrics_path.write_text(
        json.dumps(payload_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload_out, indent=2, ensure_ascii=False), flush=True)
    return payload_out
