# pgat-length

Pose-Guided Adaptive Tokenization with Variable-Length Prefix and mBART Decoder
for Gloss-Free Sign Language Translation on PHOENIX14T.

## Thesis question

Can a pose-guided visual prefix with capacity scaled to source video duration
reduce the long-sentence BLEU-4 cliff in gloss-free sign language translation?

## Target for the proposal

- Overall TEST BLEU-4: >= 17
- Bin 13-18 BLEU-4: >= 10 (baselines cliff around 4-8)
- Bin 19-24 BLEU-4: >= 6
- Comparison baselines: TSPNet, NSLT, GASLT, GFSLT-VLP (published or reproduced)

## Design (PGAT-v2, six locked changes)

1. Pose-guided articulator-aware visual tokens
   (8 articulator queries + 4 global queries, confidence-safe fallback).
2. Adaptive pose-motion temporal segmentation
   (segment boundaries follow motion energy, not uniform grid).
3. Variable-length prefix
   (`K_temporal = clamp(round(source_num_frames / 8), 12, 32)`; total 24-44 tokens).
4. Partial DINOv2 unfreeze (last two blocks, LR 1e-5).
5. Temporal data augmentation
   (random start within first 20%, single-frame drop p=0.1, small jitter; train only).
6. mBART-large-cc25 fully fine-tuned as the decoder
   (PGAT tokens projected to mBART hidden dim, fed to mBART encoder,
   decoder cross-attends natively).

## Repository layout

```
pgat-length/
├── configs/          # data / model / alignment / translation / augmentation YAML
├── slurm/            # sbatch templates for ASL cluster (partition ASL-gpu)
├── src/pgat_length/  # library code
│   ├── data/         # manifest, sampling, collator, dataset
│   ├── features/     # plan / spatial / motion / text extractors
│   ├── models/       # tokenizer, articulator, global summary, alignment, projection, translation
│   ├── training/     # alignment_loop, translation_loop, checkpoint
│   └── evaluation/   # generate, metrics, five_bin, bootstrap
├── scripts/          # numbered CLI entry points 01..08
├── tests/            # unit tests
└── logs/             # sbatch stdout/stderr (gitignored)
```

## Build sequence

Run in this order. Each step is one sbatch job (ASL allows one running job at a time).

1. `scripts/01_build_plans.py`      — deterministic frame plans (pose-informed).
2. `scripts/02_extract_spatial.py`  — DINOv2 crops (global + hands + mouth), variable K.
3. `scripts/03_extract_motion.py`   — motion features.
4. `scripts/04_build_text.py`       — target-text tokenization cache.
5. `scripts/05_train_alignment.py`  — video-text alignment (contrastive).
6. `scripts/06_train_translation.py`— mBART fine-tune with variable-length prefix.
7. `scripts/07_evaluate_dev.py`     — DEV generation + metrics.
8. `scripts/08_compare_five_bin.py` — length-stratified comparison table.

## ASL cluster

Login:

```
ssh -p 44065 dsai-st125989@asl.ait.ac.th
```

Constraints:
- One running job per user.
- One node per job.
- Seven-day maximum runtime per job.
- No `python` on the head node — use `sbatch` or `srun` only.
- 100 GB storage quota.

Submit any training job:

```
sbatch slurm/train_translation.sbatch scripts/06_train_translation.py --allow-full
```

## Storage plan (target: fit in 100 GB)

| Path                                | Purpose                          | Estimate |
|-------------------------------------|----------------------------------|---------:|
| `$HOME/pgat-length/`                | code                             |  ~200 MB |
| `$HOME/pgat-cache/raw/`             | Phoenix14T raw frames            |  ~20 GB  |
| `$HOME/pgat-cache/features/`        | plans + spatial + motion + text  |   ~2 GB  |
| `$HOME/pgat-cache/hf/`              | mBART + DINOv2 base weights      |   ~4 GB  |
| `$HOME/outputs/alignment/`          | best.pt only                     |  ~500 MB |
| `$HOME/outputs/translation/`        | best.pt only                     |   ~1 GB  |
| `$HOME/outputs/predictions/`        | jsonl                            |   ~5 MB  |
| `$HOME/logs/`                       | sbatch stdout/stderr             |  ~200 MB |
| Headroom                            |                                  |  ~70 GB  |

Discipline:
- Save `best.pt` only. Never save `last.pt` unless resuming.
- Never checkpoint mBART or DINOv2 base weights (they live in HF cache, deduplicated).
- Delete alignment `best.pt` after translation training completes.

## Diagnostic evidence supporting the design

Established on the exploratory `pg-adaptor` project (not part of this repo):

- Source32 (doubled DINO anchors, 32 real evidence positions): improved retrieval
  V->T R@1 from 14.84 to 17.92, but did not move long-bin BLEU. Falsifies "more
  encoder evidence" as the mechanism.
- Z1 (min-length decoding sweep on PGAT-v1): forcing longer generation did not
  recover BLEU on long bins, only produced fluent hallucination. Falsifies
  "conservative decoder" as the mechanism.
- Z2 (video length vs generation length correlation): decoder output caps at
  ~16 words regardless of video duration; in bin 25-32 the correlation between
  video frames and generation length is r = -0.26. Confirms fixed-capacity
  prefix compression as the mechanism.

PGAT-v2 addresses that mechanism directly via change 3 (variable-length prefix)
and change 6 (mBART decoder with native cross-attention).
