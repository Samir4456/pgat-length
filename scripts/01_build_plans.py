"""Step 01 — build per-sample pose-informed frame plans with variable K_temporal.

Usage (via sbatch):
    sbatch slurm/build_features.sbatch scripts/01_build_plans.py --split train --allow-full
    sbatch slurm/build_features.sbatch scripts/01_build_plans.py --split dev   --allow-full

Writes plan shards under $HOME/pgat-cache/features/plans/<split>/.
"""

if __name__ == "__main__":
    from pgat_length.features.plans import build_plan  # noqa: F401
    raise SystemExit(
        "TODO: wire CLI (argparse + config load + shard writer) to features.plans"
    )
