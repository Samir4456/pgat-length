"""Corpus + per-sample translation metrics.

Uses SacreBLEU (tokenize=13a, exponential smoothing) for BLEU-1..4,
sacrebleu CHRF (char_order=6, word_order=0, beta=2), and rouge_score
for ROUGE-L F1 with no stemmer. Same scorer as the exploratory project
so numbers are directly comparable to PGAT-v1 / Source32 predictions.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU, CHRF


def normalize_text(text: str) -> str:
    """Lowercase, strip, collapse whitespace, keep basic punctuation."""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def word_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", normalize_text(text))


def sample_metrics(reference: str, prediction: str) -> dict[str, float]:
    """Sentence-level BLEU-4, chrF, ROUGE-L, exact match."""
    ref = normalize_text(reference)
    hyp = normalize_text(prediction)
    bleu = BLEU(tokenize="13a", smooth_method="exp", max_ngram_order=4, effective_order=True)
    chrf = CHRF(char_order=6, word_order=0, beta=2)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return {
        "sentence_bleu4": float(bleu.sentence_score(hyp, [ref]).score),
        "sentence_chrf": float(chrf.sentence_score(hyp, [ref]).score),
        "rouge_l_f1": 100.0 * float(rouge.score(ref, hyp)["rougeL"].fmeasure),
        "exact_match": float(ref == hyp),
    }


def corpus_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Corpus BLEU-1..4, chrF, ROUGE-L F1 over aligned reference/prediction pairs."""
    if not records:
        raise ValueError("no records to score")
    references = [normalize_text(str(row["reference"])) for row in records]
    hypotheses = [normalize_text(str(row["prediction"])) for row in records]
    bleu = BLEU(tokenize="13a", smooth_method="exp", max_ngram_order=4, effective_order=True)
    chrf = CHRF(char_order=6, word_order=0, beta=2)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    bleu_score = bleu.corpus_score(hypotheses, [references])
    chrf_score = chrf.corpus_score(hypotheses, [references])
    rouge_scores = [
        rouge.score(reference, hypothesis)["rougeL"].fmeasure
        for reference, hypothesis in zip(references, hypotheses, strict=True)
    ]
    exact_hits = sum(1 for h, r in zip(hypotheses, references, strict=True) if h == r)
    ref_word_counts = [len(word_tokens(r)) for r in references]
    hyp_word_counts = [len(word_tokens(h)) for h in hypotheses]
    return {
        "samples": len(records),
        "bleu_1": float(bleu_score.precisions[0]),
        "bleu_2": float(bleu_score.precisions[1]),
        "bleu_3": float(bleu_score.precisions[2]),
        "bleu_4": float(bleu_score.score),
        "chrf": float(chrf_score.score),
        "rouge_l_f1": 100.0 * float(np.mean(rouge_scores)),
        "exact_match_percent": 100.0 * exact_hits / len(records),
        "empty_predictions": sum(1 for h in hypotheses if not h),
        "mean_reference_words": float(np.mean(ref_word_counts)),
        "mean_generation_words": float(np.mean(hyp_word_counts)),
    }
