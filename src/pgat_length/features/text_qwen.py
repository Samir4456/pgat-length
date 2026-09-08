"""Qwen tokenization cache for PHOENIX14T German references.

Stores per uid, for the Qwen prompt+prefix+target layout used by the QLoRA
decoder path (models/qwen_translation.py):

    prompt_input_ids     : the fixed instruction turn tokens (same per split)
    prompt_attention_mask: mask over prompt_input_ids
    target_input_ids     : the German reference tokens + eos (padded)
    target_attention_mask: mask over target_input_ids
    target_length        : real (unpadded) target token count

The prompt is the same string for every sample, so we cache it once in
bank metadata and *also* store it per-row so the reader stays uniform.
The visual prefix is injected between prompt and target at model time
(prepare_prefix_conditioned_batch).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SYSTEM_PROMPT = (
    "You translate German Sign Language videos into natural German sentences."
)
DEFAULT_USER_PROMPT = (
    "Translate the sign-language video into German. Return only the translation."
)


@dataclass(frozen=True)
class QwenTextTokenizerConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    max_prompt_tokens: int = 96
    max_target_tokens: int = 96
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt: str = DEFAULT_USER_PROMPT


def load_qwen_tokenizer(config: QwenTextTokenizerConfig, hf_cache: Path | None = None):
    """Return a fast Qwen tokenizer with pad_token set for right-side padding."""
    from transformers import AutoTokenizer

    cache_dir = str(hf_cache) if hf_cache else None
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, cache_dir=cache_dir, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_prompt_ids(tokenizer, config: QwenTextTokenizerConfig) -> tuple[np.ndarray, np.ndarray]:
    """Render the instruction turn once via the Qwen chat template.

    The generation prompt (add_generation_prompt=True) ends the sequence right
    before the assistant is expected to start speaking — that is where the
    visual prefix + target tokens are appended at model time.

    Returns (input_ids [P], attention_mask [P]) padded to max_prompt_tokens.
    """
    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": config.user_prompt},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        rendered,
        padding="max_length",
        truncation=True,
        max_length=config.max_prompt_tokens,
        return_tensors="np",
        return_attention_mask=True,
        add_special_tokens=False,  # chat template already adds them
    )
    ids = encoded["input_ids"][0].astype(np.int32)
    mask = encoded["attention_mask"][0].astype(np.bool_)
    if int(mask.sum()) < 1:
        raise ValueError("Qwen prompt tokenised to zero real tokens")
    return ids, mask


def tokenize_target(
    tokenizer,
    text: str,
    max_target_tokens: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Encode a German reference as target tokens ending with EOS.

    We add eos explicitly so the causal-LM head learns to stop. Padding is
    right-side; the trailing pad positions get attention_mask = False and
    label = -100 at model time.
    """
    if not text or not text.strip():
        raise ValueError("empty target text")
    encoded = tokenizer(
        text,
        padding=False,
        truncation=True,
        max_length=max_target_tokens - 1,  # leave room for one eos
        return_tensors="np",
        return_attention_mask=True,
        add_special_tokens=False,
    )
    ids = encoded["input_ids"][0].astype(np.int64)
    real_len = int(ids.shape[0]) + 1  # +1 for eos
    pad_id = int(tokenizer.pad_token_id)
    eos_id = int(tokenizer.eos_token_id)
    out_ids = np.full((max_target_tokens,), pad_id, dtype=np.int32)
    out_mask = np.zeros((max_target_tokens,), dtype=np.bool_)
    out_ids[: ids.shape[0]] = ids
    out_ids[ids.shape[0]] = eos_id
    out_mask[:real_len] = True
    if real_len < 1:
        raise ValueError(f"target tokenised to zero real tokens: {text!r}")
    return out_ids, out_mask, real_len
