"""Target text tokenization cache using the mBART tokenizer.

Responsibility:
- Load facebook/mbart-large-cc25 tokenizer from HF cache.
- Set src_lang and tgt_lang to de_DE.
- Tokenize each PHOENIX14T reference to token ids, truncated to max_target_tokens.
- Persist per-uid arrays plus a JSONL index for random access.

Public API (to implement):
- build_text_cache(manifest: pd.DataFrame, out_root: Path, model_name: str,
                   max_target_tokens: int) -> None
"""

raise NotImplementedError("pgat_length.features.text: implement in step 01")
