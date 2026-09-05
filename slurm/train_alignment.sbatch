#!/bin/bash
#SBATCH --job-name=pgat_align
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ASL-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00

# Usage:
#   sbatch slurm/train_alignment.sbatch --allow-full

set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate pgat-length

cd "$HOME/pgat-length"
mkdir -p logs "$HOME/outputs/alignment"

nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

srun python scripts/05_train_alignment.py "$@"
