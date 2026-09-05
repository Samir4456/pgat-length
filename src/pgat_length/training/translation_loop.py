"""Stage 2: mBART translation fine-tuning with PGAT variable-length prefix.

Responsibility:
- Load PGAT encoder weights from alignment_best.pt (strict warm start).
- Instantiate PgatMbartTranslationModel; unfreeze last N DINOv2 blocks
  (config.encoder.dinov2.unfreeze_last_n_blocks) at low LR.
- Cross-entropy with label smoothing 0.1; teacher forcing.
- Parameter groups with per-module learning rates:
    * PGAT tokenizer: pgat_learning_rate
    * DINOv2 unfrozen blocks: dinov2_learning_rate
    * mBART encoder + decoder: encoder_learning_rate / decoder_learning_rate
    * Projection: encoder_learning_rate
- bf16 mixed precision, gradient checkpointing on both PGAT and mBART.
- Early stop on dev_loss (min), patience 4.
- Save translation_best.pt only. Never save mBART or DINOv2 base weights.

Public API (to implement):
- def train_translation(config: dict, data_config: dict, model_config: dict,
                        output_dir: Path, allow_full: bool) -> None
"""

raise NotImplementedError("pgat_length.training.translation_loop: implement in step 05")
