"""Beam-search generation for evaluation.

Responsibility:
- Reload the frozen translation_best.pt checkpoint.
- For each sample, encode the PGAT variable-length prefix, then call
  model.mbart.generate(...) with the beam config from configs/translation.yaml.
- Save predictions to JSONL: {uid, sample_id, reference, split, condition,
  prediction, generated_tokens, prefix_length}.
- Resume-safe on interruption.

Public API (to implement):
- def evaluate_split(config: dict, checkpoint_path: Path, split: str,
                     output_dir: Path, allow_test: bool) -> None
"""

raise NotImplementedError("pgat_length.evaluation.generate: implement in step 06")
