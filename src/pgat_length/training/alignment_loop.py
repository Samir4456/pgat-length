"""Stage 1: contrastive video-text alignment training loop.

Responsibility:
- Load PhoenixCachedDataset(train) and PhoenixCachedDataset(dev).
- Instantiate PgatAlignmentModel with mBART encoder (frozen).
- InfoNCE + hard-negative loss, temperature 0.07.
- bf16 mixed precision, gradient checkpointing on PGAT transformer.
- Track V->T recall@{1,5,10}, mean/median rank on DEV.
- Early stop on v2t_recall_at_1 (max), patience 4.
- Save alignment_best.pt only (no last.pt).

Public API (to implement):
- def train_alignment(config: dict, data_config: dict, model_config: dict,
                      output_dir: Path, allow_full: bool) -> None
"""

raise NotImplementedError("pgat_length.training.alignment_loop: implement in step 04")
