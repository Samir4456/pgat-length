"""mBART tokenization cache for PHOENIX14T German references.

For each sample, tokenizes the target text with the mBART tokenizer under
the configured source and target language codes, pads to max_target_tokens,
and stores `input_ids` + `attention_mask` + `text_length` per uid.

Text tokenization has no dependency on plans / spatial / motion. The
per-section fingerprint scheme means that adding or editing the text
config never invalidates other banks and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TextTokenizerConfig:
    model_name: str = "facebook/mbart-large-cc25"
    src_lang: str = "de_DE"
    tgt_lang: str = "de_DE"
    max_target_tokens: int = 96


def load_mbart_tokenizer(config: TextTokenizerConfig, hf_cache: Path | None = None):
    """Return an mBART tokenizer with src/tgt language set for German."""
    from transformers import AutoTokenizer

    cache_dir = str(hf_cache) if hf_cache else None
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=cache_dir)
    tokenizer.src_lang = config.src_lang
    tokenizer.tgt_lang = config.tgt_lang
    return tokenizer


def tokenize_translation(
    tokenizer,
    text: str,
    max_target_tokens: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (input_ids, attention_mask, text_length) padded to max_target_tokens.

    mBART's convention for the target side is to open the ids with the language
    code token. We use the tokenizer's as_target_tokenizer context so the
    correct language id is prepended and the eos is appended.
    """
    with tokenizer.as_target_tokenizer():
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_target_tokens,
            return_tensors="np",
            return_attention_mask=True,
        )
    input_ids = encoded["input_ids"][0].astype(np.int32)
    attention_mask = encoded["attention_mask"][0].astype(np.bool_)
    text_length = int(attention_mask.sum())
    if text_length < 1:
        raise ValueError(f"text tokenised to zero real tokens: {text!r}")
    return input_ids, attention_mask, text_length
