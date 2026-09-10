# TSPNet + SSM Adapter Cross-Backbone Experiment

Purpose: test whether the bidirectional selective SSM temporal adapter
(originally developed for pgat-length + mBART; see
`pgat-length/docs/COMPOSITIONAL_GENERALIZATION.md` and
`pgat-length/src/pgat_length/models/ssm_adapter.py`) produces the same
length-flattening effect when inserted into an independent gloss-free
SLT backbone (TSPNet).

**Claim being tested.** The adapter is a **backbone-agnostic
length-flattening module**. If it works on TSPNet as well as on pgat-length,
the paper contribution generalizes across systems.

## What was added to this fork

Two files, one patched file. Fully backward-compatible: the adapter is
opt-in via `--use-ssm-adapter`; without that flag the encoder matches the
published TSPNet baseline exactly.

- `fairseq/models/ssm_adapter.py` -- new. The `BidirectionalSSMAdapter`
  module (pure PyTorch, no `mamba-ssm` dependency).
- `fairseq/models/transformer_from_sign.py` -- patched. Adds
  `--use-ssm-adapter` + six hyperparameter flags; instantiates the
  adapter in `TransformerEncoderSign.__init__`; applies it in
  `TransformerEncoderSign.forward` after the multi-scale features are
  concatenated and before the fairseq `(T, B, C)` transpose.
- `run_scripts/run_phoenix_pos_embed_sp_test_3lvl_ssm.sh` -- new. Copy
  of the baseline runner with the SSM flags appended.

## Cluster setup (ASL)

The cluster env for pgat-length (`pgat-length`) uses PyTorch cu118
which is much newer than TSPNet's expected torch 1.11. Create a
**separate conda env** for this experiment; do not touch pgat-length.

```bash
# On the login node.
conda create -n tspnet-ssm python=3.8 -y
conda activate tspnet-ssm

# PyTorch 1.11 + cu113 to match TSPNet's requirements.
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install their fairseq fork editable so our patches take effect.
cd $HOME
git clone https://github.com/Samir4456/pg-adaptor.git   # or copy from your local
cd pg-adaptor/external_baselines/TSPNet
pip install --editable .

# Extra deps.
pip install sacrebleu==2.2.0 tensorboard==2.9.1 bpemb==0.3.4 sentencepiece==0.1.97 lxml==4.9.1 loguru==0.7.0
```

If any package fails to install (mmcv 0.2.14 is known to fight modern
glibc), install a newer compatible version first and add `--no-deps` to
skip transitive-conflict resolution.

## Download the preprocessed data

TSPNet ships preprocessed I3D features + BPE text via Google Drive:
https://drive.google.com/drive/folders/1oYV_k1wqGbPUhBrkLRMQb1iWKQp5P3pp

On the cluster, use `gdown`:

```bash
pip install gdown

cd $HOME/pg-adaptor/external_baselines/TSPNet

# Full folder pull (~10 GB total).
gdown --folder https://drive.google.com/drive/folders/1oYV_k1wqGbPUhBrkLRMQb1iWKQp5P3pp

# Verify layout matches the README.
ls -la i3d-features/
ls -la data-bin/phoenix2014T/sp25000/
```

Expected layout:

```
TSPNet/
├── i3d-features/
│   ├── span=8_stride=2/
│   ├── span=12_stride=2/
│   └── span=16_stride=2/
├── data-bin/
│   └── phoenix2014T/
│       └── sp25000/
│           ├── train.sign-de.sign, train.sign-de.de, ...
│           ├── test.sign-de.sign,  test.sign-de.de,  ...
│           ├── emb
│           └── dict.de.txt
```

## Reproduce the baseline (Route A step 1)

Run the unmodified baseline once. Verify it converges near the paper's
published BLEU-4 12.98 on TEST.

```bash
cd $HOME/pg-adaptor/external_baselines/TSPNet/run_scripts
SAVE_DIR=$HOME/outputs/tspnet_baseline bash run_phoenix_pos_embed_sp_test_3lvl.sh
```

Wall time on a single A6000: about 4-8 hours for 200 epochs (usually
early-stops earlier via `lr_patience=8` on `reduce_lr_on_plateau`).

## Train the SSM adapter variant (Route A step 2)

```bash
SAVE_DIR=$HOME/outputs/tspnet_ssm bash run_phoenix_pos_embed_sp_test_3lvl_ssm.sh
```

The SSM adapter adds ~15-20 M parameters (d_model=1024, d_state=16,
expand=2, 1 layer per direction). Wall time roughly the same as
baseline (sequential scan is a Python loop; T for concatenated
multi-scale features is a few hundred tokens per sample, still fast).

## Evaluate + compare

Run their eval script for each checkpoint:

```bash
cd $HOME/pg-adaptor/external_baselines/TSPNet/test_scripts
CHECKPOINT=$HOME/outputs/tspnet_baseline/checkpoint_best.pt bash test_phoenix_pos_embed_sp_test_3lvl.sh
CHECKPOINT=$HOME/outputs/tspnet_ssm/checkpoint_best.pt      bash test_phoenix_pos_embed_sp_test_3lvl.sh
```

Then rescore both hypothesis files under the pgat-length scorer (for
apples-to-apples with the SSM-on-pgat-length results):

```bash
# Adapt these paths to where the hypothesis .txt files land.
python $HOME/pg-adaptor/scripts/83_analyze_external_baseline_five_bins.py \
    --model-name tspnet_baseline \
    --predictions <path to baseline hypothesis .txt> \
    --split test \
    --output-root $HOME/outputs/external_baselines/five_bin

python $HOME/pg-adaptor/scripts/83_analyze_external_baseline_five_bins.py \
    --model-name tspnet_ssm \
    --predictions <path to ssm hypothesis .txt> \
    --split test \
    --output-root $HOME/outputs/external_baselines/five_bin
```

## The comparison table to fill in

Once both runs finish and are rescored:

| Bin | Samples | TSPNet BLEU-4 | TSPNet+SSM BLEU-4 | Δ | TSPNet chrF | TSPNet+SSM chrF | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-6 | 42 | | | | | | |
| 7-12 | 286 | | | | | | |
| 13-18 | 220 | | | | | | |
| 19-24 | 78 | | | | | | |
| 25-31 | 16 | | | | | | |

Plus overall (642 samples) and long/short chrF ratio.

**Interpretation targets:**

- If the SSM adapter flattens TSPNet's chrF curve (analogous to
  pgat-length: short/long chrF ratio increases), the cross-backbone
  claim holds and the paper story is "adapter is a general length-
  flattening module."
- If the effect is present but smaller on TSPNet, the story becomes
  "adapter helps but effect scales with backbone characteristics."
- If the effect vanishes on TSPNet, the finding is backbone-specific
  and reported openly.

Any of the three outcomes is publishable; the third is the honest
negative result and is still a legitimate cross-backbone probe.
