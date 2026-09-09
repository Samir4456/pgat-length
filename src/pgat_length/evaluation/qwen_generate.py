"""Beam-search generation for the Qwen QLoRA translation model.

Ports the deterministic beam search from pg-adaptor's
training/pgat_translation_runtime.py (kept because HF's generate() does not
support conditioning a causal LM on continuous soft-prefix inputs_embeds
alongside prompt token ids in the specific [prompt | prefix | target]
layout we use here). Batch size is one at generation time.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from pgat_length.data.dataset_qwen import QwenPhoenixDataset, collate_qwen_batch
from pgat_length.evaluation.five_bin import per_bin_metrics
from pgat_length.evaluation.metrics import corpus_metrics
from pgat_length.training.checkpoint import load_best
from pgat_length.training.qwen_runtime import load_lora_state_dict, load_qwen_qlora_decoder
from pgat_length.training.qwen_translation_loop import build_model


def _banned_ngram_tokens(generated: list[int], ngram_size: int) -> set[int]:
    if ngram_size <= 0 or len(generated) + 1 < ngram_size:
        return set()
    prefix_size = ngram_size - 1
    prefix = tuple(generated[-prefix_size:]) if prefix_size else ()
    banned: set[int] = set()
    for start in range(len(generated) - ngram_size + 2):
        ngram = generated[start : start + ngram_size]
        if tuple(ngram[:prefix_size]) == prefix and len(ngram) == ngram_size:
            banned.add(ngram[-1])
    return banned


def _repeat_cache(cache: Any, repeats: int) -> Any:
    if hasattr(cache, "batch_repeat_interleave"):
        cache = deepcopy(cache)
        result = cache.batch_repeat_interleave(repeats)
        return cache if result is None else result
    if isinstance(cache, (tuple, list)):
        return type(cache)(
            type(layer)(
                value.repeat_interleave(repeats, dim=0) if torch.is_tensor(value) else value
                for value in layer
            )
            for layer in cache
        )
    raise TypeError(f"Unsupported decoder cache type: {type(cache)!r}")


def _select_cache(decoder: torch.nn.Module, cache: Any, indices: torch.Tensor) -> Any:
    if hasattr(cache, "batch_select_indices"):
        result = cache.batch_select_indices(indices)
        return cache if result is None else result
    reorder = getattr(decoder, "_reorder_cache", None)
    if reorder is not None:
        return reorder(cache, indices)
    if isinstance(cache, (tuple, list)):
        return type(cache)(
            type(layer)(
                value.index_select(0, indices) if torch.is_tensor(value) else value
                for value in layer
            )
            for layer in cache
        )
    raise TypeError(f"Unsupported decoder cache type: {type(cache)!r}")


@torch.inference_mode()
def beam_generate_from_context(
    decoder: torch.nn.Module,
    context_embeds: torch.Tensor,
    context_mask: torch.Tensor,
    eos_token_id: int,
    max_new_tokens: int,
    num_beams: int,
    no_repeat_ngram_size: int = 0,
    length_penalty: float = 1.0,
) -> torch.Tensor:
    """context_embeds: [1, S, H]; context_mask: [1, S]. Returns [1, L] token ids."""
    if context_embeds.shape[0] != 1:
        raise ValueError("beam_generate_from_context requires batch size 1")
    if num_beams < 1:
        raise ValueError("num_beams must be positive")

    output = decoder(
        inputs_embeds=context_embeds,
        attention_mask=context_mask.long(),
        use_cache=True,
        return_dict=True,
    )
    logits = output.logits[:, -1].float()
    first_scores, first_tokens = torch.log_softmax(logits, dim=-1).topk(num_beams, dim=-1)
    beam_scores = first_scores.squeeze(0)
    sequences: list[list[int]] = [[int(token)] for token in first_tokens.squeeze(0)]
    finished = torch.tensor(
        [tokens[-1] == eos_token_id for tokens in sequences],
        device=logits.device,
        dtype=torch.bool,
    )
    if num_beams == 1:
        # Fast path: no beam expansion.
        cache = output.past_key_values
        attention_mask = context_mask.long()
        for _ in range(1, max_new_tokens):
            if bool(finished.all()):
                break
            attention_mask = torch.cat(
                (attention_mask, torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)),
                dim=1,
            )
            last_token = torch.tensor([[sequences[0][-1]]], device=logits.device, dtype=torch.long)
            output = decoder(
                input_ids=last_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            next_logits = output.logits[:, -1].float()
            banned = _banned_ngram_tokens(sequences[0], no_repeat_ngram_size)
            if banned:
                next_logits[0, list(banned)] = torch.finfo(next_logits.dtype).min
            next_token_id = int(next_logits.argmax(dim=-1).item())
            sequences[0].append(next_token_id)
            cache = output.past_key_values
            if next_token_id == eos_token_id:
                finished[0] = True
        selected = sequences[0]
        if eos_token_id in selected:
            selected = selected[: selected.index(eos_token_id) + 1]
        return torch.tensor([selected], device=logits.device, dtype=torch.long)

    cache = _repeat_cache(output.past_key_values, num_beams)
    attention_mask = context_mask.long().repeat_interleave(num_beams, dim=0)

    for step in range(1, max_new_tokens):
        if bool(finished.all()):
            break
        attention_mask = torch.cat(
            (attention_mask, torch.ones((num_beams, 1), device=attention_mask.device, dtype=attention_mask.dtype)),
            dim=1,
        )
        last_tokens = torch.tensor(
            [[tokens[-1]] for tokens in sequences],
            device=logits.device,
            dtype=torch.long,
        )
        output = decoder(
            input_ids=last_tokens,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        next_logits = output.logits[:, -1].float()
        for i, tokens in enumerate(sequences):
            if finished[i]:
                next_logits[i].fill_(torch.finfo(next_logits.dtype).min)
                next_logits[i, eos_token_id] = 0.0
                continue
            banned = _banned_ngram_tokens(tokens, no_repeat_ngram_size)
            if banned:
                next_logits[i, list(banned)] = torch.finfo(next_logits.dtype).min
        candidate_scores = torch.log_softmax(next_logits, dim=-1) + beam_scores[:, None]
        vocabulary_size = candidate_scores.shape[1]
        next_scores, flat_indices = candidate_scores.flatten().topk(num_beams)
        parent_indices = torch.div(flat_indices, vocabulary_size, rounding_mode="floor")
        token_indices = flat_indices.remainder(vocabulary_size)
        sequences = [
            [*sequences[int(parent)], int(token)]
            for parent, token in zip(parent_indices, token_indices, strict=True)
        ]
        beam_scores = next_scores
        finished = torch.tensor(
            [tokens[-1] == eos_token_id for tokens in sequences],
            device=logits.device,
            dtype=torch.bool,
        )
        cache = _select_cache(decoder, output.past_key_values, parent_indices)
        attention_mask = attention_mask.index_select(0, parent_indices)

    normalized: list[float] = []
    for score, tokens in zip(beam_scores, sequences, strict=True):
        eff_len = next((i + 1 for i, tok in enumerate(tokens) if tok == eos_token_id), len(tokens))
        denom = float(eff_len**length_penalty) if length_penalty else 1.0
        normalized.append(float(score) / denom)
    best = int(np.argmax(normalized))
    selected = sequences[best]
    if eos_token_id in selected:
        selected = selected[: selected.index(eos_token_id) + 1]
    return torch.tensor([selected], device=logits.device, dtype=torch.long)


def _rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(payload, encoding="utf-8")


def generate_qwen(
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
    if split == "test" and not allow_test:
        raise RuntimeError("TEST evaluation requires --allow-test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None

    from os.path import expanduser, expandvars

    def resolve(value: str) -> Path:
        return Path(expanduser(expandvars(str(value)))).resolve()

    feature_root = resolve(data_config["paths"]["feature_root"])
    manifest_path = resolve(data_config["paths"]["manifest"])
    hf_cache = resolve(data_config["paths"]["hf_cache"])

    dataset = QwenPhoenixDataset(
        plans_root=feature_root / "plans",
        spatial_root=feature_root / "spatial",
        motion_root=feature_root / "motion",
        qwen_text_root=feature_root / "text_qwen",
        manifest_path=manifest_path,
        split=split,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_qwen_batch)

    payload = load_best(checkpoint_path, map_location="cpu")
    model = build_model(model_config, translation_config["qlora"], hf_cache=hf_cache)
    trainable_state = payload["trainable_state_dict"]
    missing, unexpected = model.load_state_dict(trainable_state, strict=False)
    # Attach QLoRA decoder + load LoRA weights.
    model = model.to(device)
    decoder, _ = load_qwen_qlora_decoder(translation_config["qlora"], hf_cache_dir=str(hf_cache))
    model.attach_qlora_decoder(decoder, decoder_dtype=torch.bfloat16)
    load_lora_state_dict(model.decoder, payload["lora_state_dict"])  # type: ignore[arg-type]
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(translation_config["qlora"]["decoder_model"]), cache_dir=str(hf_cache))
    eos_id = int(tokenizer.eos_token_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{split}_predictions.jsonl"
    metrics_path = output_dir / f"{split}_metrics.json"
    rows: list[dict[str, Any]] = []
    start = time.monotonic()
    for i, batch in enumerate(loader):
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        if autocast_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                prefix_embeds, prefix_mask = model.encode_visual_prefix(batch)
                context_embeds, context_mask, _ = model._assemble(
                    prefix_embeds=prefix_embeds,
                    prefix_mask=prefix_mask,
                    prompt_input_ids=batch["prompt_input_ids"],
                    prompt_attention_mask=batch["prompt_attention_mask"],
                    target_input_ids=None,
                    target_attention_mask=None,
                )
                generated_ids = beam_generate_from_context(
                    decoder=model.decoder,  # type: ignore[arg-type]
                    context_embeds=context_embeds,
                    context_mask=context_mask,
                    eos_token_id=eos_id,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    length_penalty=length_penalty,
                )
        else:
            prefix_embeds, prefix_mask = model.encode_visual_prefix(batch)
            context_embeds, context_mask, _ = model._assemble(
                prefix_embeds=prefix_embeds,
                prefix_mask=prefix_mask,
                prompt_input_ids=batch["prompt_input_ids"],
                prompt_attention_mask=batch["prompt_attention_mask"],
                target_input_ids=None,
                target_attention_mask=None,
            )
            generated_ids = beam_generate_from_context(
                decoder=model.decoder,  # type: ignore[arg-type]
                context_embeds=context_embeds,
                context_mask=context_mask,
                eos_token_id=eos_id,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=length_penalty,
            )
        text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True).strip()
        rows.append(
            {
                "uid": batch["uid"][0],
                "sample_id": batch["sample_id"][0],
                "reference": batch["reference"][0],
                "prediction": text,
            }
        )
        if (i + 1) % 20 == 0:
            elapsed = time.monotonic() - start
            print(f"  gen {i+1}/{len(loader)}  ({elapsed:.1f}s)", flush=True)

    _rewrite_jsonl(predictions_path, rows)
    overall = corpus_metrics(rows)
    bins = per_bin_metrics(rows)
    summary: dict[str, Any] = {
        "split": split,
        "checkpoint": str(checkpoint_path),
        "samples": len(rows),
        "num_beams": num_beams,
        "max_new_tokens": max_new_tokens,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "length_penalty": length_penalty,
        "overall": overall,
        "bins": bins,
    }
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"overall": overall, "metrics": str(metrics_path)}, indent=2), flush=True)
    return summary
