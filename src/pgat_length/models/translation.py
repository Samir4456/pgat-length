"""PGAT-v2 translation model: PGAT variable-length prefix + mBART.

Architecture:
    video -> PGAT variable tokenizer -> [B, K+12, 512]
          -> projection to 1024        -> [B, K+12, 1024]
          -> mBART ENCODER (as visual-side encoder replacement)
          -> mBART DECODER (native cross-attention to encoder outputs)
          -> German target tokens

Design notes:
- mBART's encoder is fed our projected PGAT tokens *as if they were embedded
  input tokens*. No mBART tokenizer is used on the visual side.
- Position ids are learned per-token; we bypass mBART's word-embedding lookup
  and instead feed inputs_embeds directly.
- Attention mask marks valid prefix positions (variable per sample, padded to batch max).
- The decoder is the standard mBART decoder; cross-attention operates on encoder outputs.
- Both mBART encoder and decoder are trainable (full fine-tune).

Public API (to implement):
- class PgatMbartTranslationModel(nn.Module):
    encode_prefix(batch) -> (encoder_outputs [B, T, 1024], attention_mask [B, T])
    forward(batch, labels) -> loss
    generate(batch, generation_config) -> token_ids
"""

raise NotImplementedError("pgat_length.models.translation: implement in step 05")
