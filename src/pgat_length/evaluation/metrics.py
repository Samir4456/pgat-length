"""Corpus and per-sample translation metrics.

Metrics computed on normalized text (lowercase, de-tokenized punctuation).

Corpus metrics:
- BLEU-1..4 (sacrebleu, tokenize=13a, smooth exp)
- chrF (char_order=6, word_order=0, beta=2)
- ROUGE-L F1 (rouge_scorer, no stemmer)
- exact match %
- mean reference words, mean generation words

Public API (to implement):
- def sample_metrics(reference: str, prediction: str) -> dict[str, float]
- def corpus_metrics(records: Sequence[dict]) -> dict[str, Any]
- def normalize_text(text: str) -> str
"""

raise NotImplementedError("pgat_length.evaluation.metrics: implement in step 06")
