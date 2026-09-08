#!/bin/bash
#SBATCH --job-name=pgat_eval_qwen
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ASL-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00

# DEV (default) or TEST evaluation for the Qwen QLoRA checkpoint.
#
# Usage:
#   sbatch slurm/evaluate_qwen.sbatch                             # DEV
#   sbatch slurm/evaluate_qwen.sbatch --split test --allow-test   # TEST

set -euo pipefail

export PGAT_ENV="$HOME/.conda/envs/pgat-length"
export PATH="$PGAT_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$PGAT_ENV/lib:${LD_LIBRARY_PATH:-}"

cd "$HOME/pgat-length"
mkdir -p logs "$HOME/outputs/predictions_qwen" "$HOME/outputs/predictions_qwen_test"

nvidia-smi
python -c "import torch, peft, bitsandbytes; print('CUDA:', torch.cuda.is_available())"

python scripts/16_evaluate_qwen.py "$@"
