"""Paper-compatible five-bin reference-length analysis.

Bins:
    1-6, 7-12, 13-18, 19-24, 25-32
by whitespace-separated corpus tokens on the raw reference. The last bin
extends to 32 to cover DEV samples that just cross 31.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pgat_length.evaluation.metrics import corpus_metrics


@dataclass(frozen=True)
class LengthBin:
    label: str
    lower: int
    upper: int


BIN_EDGES: tuple[LengthBin, ...] = (
    LengthBin("1-6", 1, 6),
    LengthBin("7-12", 7, 12),
    LengthBin("13-18", 13, 18),
    LengthBin("19-24", 19, 24),
    LengthBin("25-32", 25, 32),
)


def corpus_token_count(text: str) -> int:
    return len(str(text).strip().split())


def bin_label(count: int) -> str:
    if count < 1:
        raise ValueError(f"reference length {count} must be positive")
    for item in BIN_EDGES:
        if item.lower <= count <= item.upper:
            return item.label
    # Anything longer than the top bin still bins into the top bin. The
    # exploratory DEV split had one 33-token reference; TEST tops out at 31.
    return BIN_EDGES[-1].label


def per_bin_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {b.label: [] for b in BIN_EDGES}
    for record in records:
        count = corpus_token_count(str(record["reference"]))
        grouped[bin_label(count)].append(record)
    result: dict[str, dict[str, Any]] = {}
    for item in BIN_EDGES:
        values = grouped[item.label]
        if not values:
            continue
        m = corpus_metrics(values)
        m["minimum_corpus_tokens"] = item.lower
        m["maximum_corpus_tokens"] = item.upper
        result[item.label] = m
    return result
