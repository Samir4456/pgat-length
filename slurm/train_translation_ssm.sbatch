#!/bin/bash
#SBATCH --job-name=pgat_translate_ssm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ASL-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00

# Stage 2 translation with the bidirectional SSM temporal adapter enabled.
# Reuses the v2 mBART decoder recipe; the only difference vs the v2 run is
# the presence of the SSM adapter (see configs/model_ssm.yaml).
#
# No extra Python deps: the adapter is pure PyTorch (no mamba-ssm needed).
#
# Usage:
#   sbatch slurm/train_translation_ssm.sbatch --allow-full

set -euo pipefail

export PGAT_ENV="$HOME/.conda/envs/pgat-length"
export PATH="$PGAT_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$PGAT_ENV/lib:${LD_LIBRARY_PATH:-}"

cd "$HOME/pgat-length"
mkdir -p logs "$HOME/outputs/translation"

nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

python scripts/06_train_translation.py \
    --model-config "$HOME/pgat-length/configs/model_ssm.yaml" \
    --translation-config "$HOME/pgat-length/configs/translation_ssm.yaml" \
    "$@"
