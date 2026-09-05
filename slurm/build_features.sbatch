#!/bin/bash
#SBATCH --job-name=pgat_features
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ASL-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Usage:
#   sbatch slurm/build_features.sbatch scripts/01_build_plans.py --split train
#   sbatch slurm/build_features.sbatch scripts/02_extract_spatial.py --split train --allow-full
#   sbatch slurm/build_features.sbatch scripts/03_extract_motion.py --split train --allow-full
#   sbatch slurm/build_features.sbatch scripts/04_build_text.py --all
#
# ASL limits: one concurrent job per user. Serialize splits and steps as separate submissions.

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate pgat-length

cd "$HOME/pgat-length"
mkdir -p logs

nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

srun python "$@"
