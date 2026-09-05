#!/bin/bash
#SBATCH --job-name=pgat_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ASL-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00

# Usage:
#   sbatch slurm/evaluate.sbatch --split dev
#   sbatch slurm/evaluate.sbatch --split test --allow-test

set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate pgat-length

cd "$HOME/pgat-length"
mkdir -p logs "$HOME/outputs/predictions"

nvidia-smi

srun python scripts/07_evaluate_dev.py "$@"
