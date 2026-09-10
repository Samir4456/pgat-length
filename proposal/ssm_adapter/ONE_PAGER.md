# SSM-Adapter — One-Page Proposal Summary

**Working title.** SSM-Adapter: A Bidirectional Selective State-Space Temporal Adapter for Length-Robust Gloss-Free Sign Language Translation.

**Student.** Samir Pokharel (st125989)  |  **Date.** 2026-09-11

## The observation that motivates the study

Gloss-free sign language translation on PHOENIX-2014T shows a sharp drop in translation quality once the reference exceeds 12 words. On our PGAT-v1 reproduction, BLEU-4 falls from 17.39 (bin 7–12 words) to 5.03 (bin 13–18 words) — a 3.5× drop across one bin boundary. The pattern is documented across multiple published systems.

## What we already ruled out

Three architectural interventions, each targeting a different suspect for the cliff, all failed:

| Suspect | Intervention tested | Outcome |
|---|---|---|
| Not enough visual evidence | Source32 (denser DINO features, wider prefix) | Retrieval improved; translation did not. |
| Prefix length too short | pgat-length v2 (variable-length prefix + mBART) | Flatter chrF, worse overall BLEU-4. |
| Decoder too small | Variable prefix + Qwen2.5-3B QLoRA | Mode collapse: two different videos → identical prediction. |

The consistent failure signature across all three: fluent, domain-plausible German with weak dependence on the video.

## What the study proposes

A small **bidirectional selective state-space temporal adapter** inserted at the visual-encoder output. Mamba-flavoured, pure PyTorch (no CUDA-kernel dependency), residual + LayerNorm. Ships as a drop-in module — the backbone architecture is unchanged; the training loss is unchanged.

## What makes it defensible

The adapter is tested on **two independent gloss-free SLT backbones** under matched compute and matched scorers:

- **pgat-length** — mBART decoder + variable-length pose-guided prefix (ours).
- **TSPNet** — I3D features + fairseq transformer decoder (Li et al., NeurIPS 2020).

If the adapter reduces the length cliff on both, the effect is attributed to the adapter rather than to backbone-specific choices. If it reduces the cliff on one but not the other, the finding is reported honestly as backbone-specific.

## Research questions

- **RQ1**. Does the SSM adapter reduce the length cliff on pgat-length?
- **RQ2**. Does the same adapter reproduce the effect on TSPNet?
- **RQ3**. Which hyperparameters matter — state dim, layer count, bidirectionality, residual scale?
- **RQ4**. How does the adapter compare against a **parameter-matched transformer adapter** at the same insertion point?

## Evaluation

- **Primary**: five-bin length-stratified analysis (bins 1–6, 7–12, 13–18, 19–24, 25–32 words). Metrics: BLEU-4, chrF, ROUGE-L F1. Long-over-short ratio as the length-cliff summary.
- **Statistical**: paired bootstrap with 2000 resamples on the four main deltas.
- **Baselines**: E0 (pgat-length no adapter), E2 (pgat-length + transformer adapter), E4 (TSPNet no adapter), E6 (TSPNet + transformer adapter).
- **Ablations**: state dim, layer count, bidirectionality, residual scaling.

## Deliverables

- A released bidirectional selective SSM adapter module in pure PyTorch.
- A patched TSPNet fork demonstrating one-flag opt-in integration.
- A pgat-length training + evaluation harness with the adapter integrated.
- A five-bin length-stratified evaluation protocol and per-system results tables.
- A reproducibility package containing config files, run scripts, and setup scripts for both backbones.

## Progress at the time of writing

- **Adapter implemented** (`src/pgat_length/models/ssm_adapter.py`), tested at both d_model=512 (pgat-length) and d_model=1024 (TSPNet).
- **pgat-length integration complete**. SSM adapter runs end-to-end. Development results: BLEU-4 7.61 (adapter) vs 7.36 (no adapter) — small overall change; chrF long-over-short ratio 0.807 vs 0.772 (flatter with adapter).
- **TSPNet integration complete**. Patched `transformer_from_sign.py` with backward-compatible `--use-ssm-adapter` flag. Baseline training in progress on the cluster; adapter variant queued.
- **Google Drive data downloaded**, all environment compatibility issues resolved (numpy pinned, sacrebleu 1.5.1, PyTorch 1.11 floor-division patches).

## What the professor is asked to comment on

1. Whether cross-backbone verification (two backbones, one dataset) is sufficient scope for a master's thesis contribution, or whether a second dataset (CSL-Daily) is required for defense.
2. Whether the primary claim should be reported when the effect is present on BLEU-4 or when it is only present on chrF and ROUGE-L; the study's current position is that chrF and ROUGE-L flattening is a defensible claim on German due to the well-documented BLEU-4 lag on morphologically rich targets.
3. Whether the parameter-matched transformer control adapter is a sufficient fairness baseline, or whether additional controls (a randomly-initialised frozen adapter of matched parameters) should be added.

## Documents in this proposal package

- `abstract.tex` — 1 page.
- `introduction.tex` — Background, motivation, problem statement, research gap, research questions, scope.
- `methodology.tex` — Approach, system overview, backbone choice, adapter design with math, integration into both backbones.
- `experimental_plan.tex` — Baselines, ablations, metrics, feasibility, risks, decision rules.
- `ONE_PAGER.md` — this document.
- `README.md` — pointers.
