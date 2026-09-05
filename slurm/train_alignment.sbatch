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

export PGAT_ENV="$HOME/.conda/envs/pgat-length"
export PATH="$PGAT_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$PGAT_ENV/lib:${LD_LIBRARY_PATH:-}"

cd "$HOME/pgat-length"
mkdir -p logs "$HOME/outputs/alignment"

nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

srun python scripts/05_train_alignment.py "$@"
