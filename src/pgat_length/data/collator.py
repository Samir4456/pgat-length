"""Variable-length prefix collator for mBART fine-tuning.

Responsibility:
- Batch samples with DIFFERENT K_temporal values.
- Pad prefix tensors along the temporal axis to the batch max K + 12
  (12 = 8 articulator + 4 global summary tokens).
- Emit a boolean/int attention mask marking valid prefix positions.
- Tokenize German target text via mBART tokenizer with padding to
  max_target_tokens (config).
- Return: {
    "spatial_features", "spatial_valid", "motion_features",
    "pose_descriptor", "pose_confidence", "pose_motion",
    "segment_valid", "prefix_length",
    "labels", "decoder_attention_mask",
    "uid", "sample_id", "reference"
  }

Public API (to implement):
- class PgatVariablePrefixCollator:
    def __init__(self, tokenizer, max_target_tokens: int, model_config: dict) -> None
    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor | list]
"""

raise NotImplementedError("pgat_length.data.collator: implement in step 03")
