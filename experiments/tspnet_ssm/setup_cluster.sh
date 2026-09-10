#!/bin/bash
# One-command cluster setup for the TSPNet + SSM adapter experiment.
#
# What this does:
#   1. Clones TSPNet fresh from upstream into $HOME/tspnet-ssm.
#   2. Copies our patched transformer_from_sign.py + new ssm_adapter.py in.
#   3. Copies the SSM-variant run script into run_scripts/.
#   4. Prints the remaining manual steps (conda env, data download, train).
#
# The patched TSPNet code is opt-in: --use-ssm-adapter enables the module;
# without it the encoder matches the published baseline exactly.
#
# Usage (on the cluster login node):
#   bash experiments/tspnet_ssm/setup_cluster.sh
#
# Idempotent: safe to re-run; existing $HOME/tspnet-ssm is preserved and
# only the patched files are overwritten.

set -euo pipefail

TSPNET_UPSTREAM_URL="https://github.com/verashira/TSPNet.git"
TARGET_DIR="$HOME/tspnet-ssm"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[setup_cluster] SSM_EXPERIMENT dir: $SRC_DIR"
echo "[setup_cluster] Target install:    $TARGET_DIR"

# 1) Clone TSPNet upstream if not already there.
if [ ! -d "$TARGET_DIR/.git" ]; then
    echo "[setup_cluster] cloning TSPNet upstream into $TARGET_DIR"
    git clone "$TSPNET_UPSTREAM_URL" "$TARGET_DIR"
else
    echo "[setup_cluster] $TARGET_DIR already exists (git repo detected); skipping clone"
fi

# 2) Copy our patched + new files in place.
echo "[setup_cluster] copying SSM adapter module"
cp "$SRC_DIR/ssm_adapter.py"                       "$TARGET_DIR/fairseq/models/ssm_adapter.py"

echo "[setup_cluster] replacing transformer_from_sign.py with patched version"
cp "$SRC_DIR/transformer_from_sign.py"             "$TARGET_DIR/fairseq/models/transformer_from_sign.py"

echo "[setup_cluster] copying SSM runner script"
cp "$SRC_DIR/run_phoenix_pos_embed_sp_test_3lvl_ssm.sh" "$TARGET_DIR/run_scripts/run_phoenix_pos_embed_sp_test_3lvl_ssm.sh"
chmod +x "$TARGET_DIR/run_scripts/"*.sh

echo "[setup_cluster] copying the guide"
cp "$SRC_DIR/SSM_EXPERIMENT.md"                    "$TARGET_DIR/SSM_EXPERIMENT.md"

echo ""
echo "======================================================================"
echo "TSPNet+SSM installed at: $TARGET_DIR"
echo ""
echo "Next steps (do these manually on the cluster):"
echo ""
echo "  1) Create a separate conda env (do NOT touch pgat-length env):"
echo ""
echo "     conda create -n tspnet-ssm python=3.8 -y"
echo "     conda activate tspnet-ssm"
echo "     pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \\"
echo "         --extra-index-url https://download.pytorch.org/whl/cu113"
echo "     cd $TARGET_DIR"
echo "     pip install --editable ."
echo "     pip install 'numpy<1.20' sacrebleu==2.2.0 tensorboard==2.9.1 \\"
echo "         bpemb==0.3.4 sentencepiece==0.1.97 lxml==4.9.1 loguru==0.7.0"
echo ""
echo "  2) Download the preprocessed data (~10 GB, ~30-60 min):"
echo ""
echo "     pip install gdown"
echo "     cd $TARGET_DIR"
echo "     gdown --folder https://drive.google.com/drive/folders/1oYV_k1wqGbPUhBrkLRMQb1iWKQp5P3pp"
echo ""
echo "  3) Reproduce the TSPNet baseline (~4-8 h on 1 A6000):"
echo ""
echo "     cd $TARGET_DIR/run_scripts"
echo "     SAVE_DIR=\$HOME/outputs/tspnet_baseline bash run_phoenix_pos_embed_sp_test_3lvl.sh"
echo ""
echo "  4) Train the SSM adapter variant (~4-8 h):"
echo ""
echo "     SAVE_DIR=\$HOME/outputs/tspnet_ssm bash run_phoenix_pos_embed_sp_test_3lvl_ssm.sh"
echo ""
echo "  5) Evaluate both, then rescore under the pgat-length scorer for"
echo "     comparison with the on-pgat-length SSM results."
echo ""
echo "See $TARGET_DIR/SSM_EXPERIMENT.md for the full protocol."
echo "======================================================================"
