#!/bin/bash
#SBATCH --job-name=pgat_align_mpnet
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ASL-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00

# Stage 1 alignment with the MPNet cached-embedding text encoder.
# Prereq: `python scripts/17_build_text_mpnet.py --all` has produced the
# mpnet_text_{train,dev}.npz caches. Also install: `uv sync --extra mpnet`.
#
# Usage:
#   sbatch slurm/train_alignment_mpnet.sbatch --allow-full

set -euo pipefail

export PGAT_ENV="$HOME/.conda/envs/pgat-length"
export PATH="$PGAT_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$PGAT_ENV/lib:${LD_LIBRARY_PATH:-}"

cd "$HOME/pgat-length"
mkdir -p logs "$HOME/outputs/alignment"

nvidia-smi
python -c "import torch, sentence_transformers; print('CUDA:', torch.cuda.is_available(), 'st:', sentence_transformers.__version__)"

python scripts/18_train_alignment_mpnet.py "$@"
