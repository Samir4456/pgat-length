"""Step 02 — DINOv2 spatial feature extraction (four crop views per segment).

Usage:
    sbatch slurm/build_features.sbatch scripts/02_extract_spatial.py --split train --allow-full
    sbatch slurm/build_features.sbatch scripts/02_extract_spatial.py --split dev   --allow-full
"""

if __name__ == "__main__":
    from pgat_length.features.spatial import run_split  # noqa: F401
    raise SystemExit("TODO: wire CLI to features.spatial.run_split")
