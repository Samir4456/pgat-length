#!/bin/bash
#SBATCH --job-name=pgat_test_v2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ASL-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00

# Build TEST features + evaluate pgat-length v2 on TEST split.
# TEST is locked in the split policy; this script explicitly opts in.
#
# Usage:
#   sbatch slurm/test.sbatch
# Or, if TEST features are already built (from an earlier run):
#   sbatch slurm/test.sbatch --skip-features

set -euo pipefail

export PGAT_ENV="$HOME/.conda/envs/pgat-length"
export PATH="$PGAT_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$PGAT_ENV/lib:${LD_LIBRARY_PATH:-}"

cd "$HOME/pgat-length"
mkdir -p logs "$HOME/outputs/predictions_test_v2"

nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

python scripts/12_test_pipeline.py --allow-test "$@"
