"""Paired bootstrap for BLEU/chrF/ROUGE deltas.

Follows the same protocol as the exploratory project:
- Cache sacrebleu corpus statistics per sample.
- Draw N resamples with replacement (default 2000).
- Report delta, 95% CI, P(candidate > baseline).

Public API (to implement):
- def paired_bootstrap(candidate: Sequence[dict], baseline: Sequence[dict],
                       samples: int = 2000, seed: int = 42) -> dict[str, Any]
"""

raise NotImplementedError("pgat_length.evaluation.bootstrap: implement in step 06")
