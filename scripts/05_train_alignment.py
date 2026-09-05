"""Step 05 — stage-1 contrastive video-text alignment.

Usage:
    sbatch slurm/train_alignment.sbatch --allow-full
"""

if __name__ == "__main__":
    from pgat_length.training.alignment_loop import train_alignment  # noqa: F401
    raise SystemExit("TODO: wire CLI to training.alignment_loop.train_alignment")
