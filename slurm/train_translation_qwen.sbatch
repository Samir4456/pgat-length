#!/bin/bash
#SBATCH --job-name=pgat_translate_qwen
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ASL-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=120:00:00

# Stage 2 translation training with Qwen2.5-3B QLoRA decoder.
# Prerequisite: Qwen text bank built via `python scripts/13_build_text_qwen.py --all`.
# Prerequisite: peft + bitsandbytes installed (`uv sync --extra qwen`).
#
# Usage:
#   sbatch slurm/train_translation_qwen.sbatch --allow-full

set -euo pipefail

export PGAT_ENV="$HOME/.conda/envs/pgat-length"
export PATH="$PGAT_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$PGAT_ENV/lib:${LD_LIBRARY_PATH:-}"

cd "$HOME/pgat-length"
mkdir -p logs "$HOME/outputs/translation"

nvidia-smi
python -c "import torch, peft, bitsandbytes; print('CUDA:', torch.cuda.is_available(), 'peft:', peft.__version__, 'bnb:', bitsandbytes.__version__)"

python scripts/15_train_translation_qwen.py "$@"
