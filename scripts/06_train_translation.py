"""Step 06 — stage-2 mBART fine-tuning with PGAT variable-length prefix.

Usage:
    sbatch slurm/train_translation.sbatch --allow-full
"""

if __name__ == "__main__":
    from pgat_length.training.translation_loop import train_translation  # noqa: F401
    raise SystemExit("TODO: wire CLI to training.translation_loop.train_translation")
