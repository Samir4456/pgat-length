"""Step 07 — generate DEV predictions and write metrics.

Usage:
    sbatch slurm/evaluate.sbatch --split dev
"""

if __name__ == "__main__":
    from pgat_length.evaluation.generate import evaluate_split  # noqa: F401
    raise SystemExit("TODO: wire CLI to evaluation.generate.evaluate_split")
