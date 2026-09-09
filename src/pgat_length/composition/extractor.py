"""Rule-based extractor: German reference -> atoms + relations.

Design goals:
- Deterministic and transparent (a reviewer can read every rule).
- No learned components (no train-test leakage).
- Fast enough to run on the whole PHOENIX manifest in seconds.
- Confidence flag per sample so downstream code can filter uncertain
  extractions from the compositional test set.

Pipeline per reference:
    1. Tokenize (whitespace + light punctuation handling).
    2. Split into clauses on punctuation and coordinating conjunctions.
    3. Per clause:
        a. Lexicon match for each slot type.
        b. Regex parse of temperatures (digit and word forms).
        c. Detect polarity markers within 3 tokens preceding an event.
        d. Resolve ambiguous *-lich forms (nördlich = north /
           northerly) via wind-context window.
    4. Build relation instances within each clause.
    5. Combine per-clause outputs into per-sample atoms + relations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from pgat_length.composition.schema import (
    CLAUSE_BOUNDARY_PUNCT,
    CLAUSE_BOUNDARY_WORDS,
    EVENT_LEXICON,
    GERMAN_NUMBER_WORDS,
    INTENSITY_LEXICON,
    LOCATION_LEXICON,
    POLARITY_MARKERS,
    Relation,
    Slot,
    TIME_LEXICON,
    WIND_CONTEXT_WORDS,
    WIND_DIRECTION_LEXICON,
    temperature_bucket,
)


# ---------------------------------------------------------------------------
# Tokenization.
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"([A-Za-zÄÖÜäöüß]+|-?\d+|[.,;:!?])", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Simple German tokenizer: words, integers, single-char punctuation."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# ---------------------------------------------------------------------------
# Clause splitting.
# ---------------------------------------------------------------------------

def split_clauses(tokens: list[str]) -> list[list[str]]:
    """Split token list on punctuation and coordinating conjunctions."""
    clauses: list[list[str]] = []
    current: list[str] = []
    for tok in tokens:
        if tok in CLAUSE_BOUNDARY_PUNCT or tok in CLAUSE_BOUNDARY_WORDS:
            if current:
                clauses.append(current)
                current = []
            continue
        current.append(tok)
    if current:
        clauses.append(current)
    return clauses


# ---------------------------------------------------------------------------
# Temperature parsing.
# ---------------------------------------------------------------------------

_TEMP_UNIT_TOKENS: frozenset[str] = frozenset({"grad", "°", "celsius"})
_MINUS_TOKENS: frozenset[str] = frozenset({"minus", "-"})


def _parse_number_token(tok: str) -> int | None:
    """Return an integer if `tok` is a digit token or a German number word."""
    if not tok:
        return None
    try:
        return int(tok)
    except ValueError:
        return GERMAN_NUMBER_WORDS.get(tok)


def extract_temperatures(tokens: list[str]) -> list[tuple[int, int]]:
    """Return list of (position, value) for each temperature in the clause.

    Handles:
        - "10 grad", "10 °"
        - "zehn grad"
        - "minus 5 grad", "- 5 grad"
        - "10 bis 15 grad" -> yields both 10 and 15
    Position is the index of the leftmost number token (used for locality).
    """
    results: list[tuple[int, int]] = []
    n = len(tokens)
    for i, tok in enumerate(tokens):
        if tok not in _TEMP_UNIT_TOKENS:
            continue
        # Walk backward to find the number(s) preceding "grad".
        j = i - 1
        collected: list[tuple[int, int]] = []
        while j >= 0 and (i - j) <= 6:  # short lookback
            candidate = _parse_number_token(tokens[j])
            if candidate is None:
                if tokens[j] == "bis" and collected:
                    j -= 1
                    continue
                if tokens[j] in _MINUS_TOKENS and collected:
                    # Apply minus to the leftmost collected value.
                    pos, value = collected[-1]
                    collected[-1] = (pos, -value)
                    j -= 1
                    continue
                break
            # Handle explicit sign token before the number.
            sign = 1
            if j - 1 >= 0 and tokens[j - 1] in _MINUS_TOKENS:
                sign = -1
                collected.append((j - 1, sign * candidate))
                j -= 2
            else:
                collected.append((j, candidate))
                j -= 1
        for pos, value in reversed(collected):
            results.append((pos, value))
    return results


# ---------------------------------------------------------------------------
# Lexicon matching for a single slot.
# ---------------------------------------------------------------------------

def match_lexicon(tokens: list[str], lexicon: dict[str, str]) -> list[tuple[int, str]]:
    """Return list of (position, canonical_atom) for every match in `tokens`."""
    return [(i, lexicon[tok]) for i, tok in enumerate(tokens) if tok in lexicon]


# ---------------------------------------------------------------------------
# Wind direction resolution.
# The *-lich adjectives (nördlich, südlich, ...) may indicate either a
# location or a wind direction. We treat them as wind_direction only when a
# wind-context word appears within 3 tokens; otherwise they map to their
# corresponding location.
# ---------------------------------------------------------------------------

_LICH_TO_LOCATION: dict[str, str] = {
    "nördlich": "nord", "noerdlich": "nord",
    "nördliche": "nord", "noerdliche": "nord",
    "nördlicher": "nord", "noerdlicher": "nord",
    "südlich": "süd", "suedlich": "süd",
    "südliche": "süd", "suedliche": "süd",
    "südlicher": "süd", "suedlicher": "süd",
    "östlich": "ost", "oestlich": "ost",
    "östliche": "ost", "oestliche": "ost",
    "östlicher": "ost", "oestlicher": "ost",
    "westlich": "west",
    "westliche": "west",
    "westlicher": "west",
    "nordöstlich": "nordost", "nordoestlich": "nordost",
    "nordwestlich": "nordwest",
    "südöstlich": "südost", "suedoestlich": "südost",
    "südwestlich": "südwest", "suedwestlich": "südwest",
}


def resolve_wind_directions(
    tokens: list[str],
    raw_locations: list[tuple[int, str]],
    raw_wind_directions: list[tuple[int, str]],
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Disambiguate *-lich forms by context.

    A raw wind-direction hit becomes a real wind direction iff a wind-context
    word appears within +/- 3 tokens; otherwise it is demoted to a location.
    """
    resolved_wind: list[tuple[int, str]] = []
    demoted_to_location: list[tuple[int, str]] = []
    n = len(tokens)
    for pos, canonical in raw_wind_directions:
        window_lo = max(0, pos - 3)
        window_hi = min(n, pos + 4)
        window = tokens[window_lo:window_hi]
        has_wind_context = any(t in WIND_CONTEXT_WORDS for t in window)
        if has_wind_context:
            resolved_wind.append((pos, canonical))
        else:
            location = _LICH_TO_LOCATION.get(tokens[pos])
            if location is not None:
                demoted_to_location.append((pos, location))
    combined_locations = list(raw_locations) + demoted_to_location
    combined_locations.sort()
    return combined_locations, resolved_wind


# ---------------------------------------------------------------------------
# Polarity detection.
# ---------------------------------------------------------------------------

def detect_polarity_for_events(
    tokens: list[str],
    events: list[tuple[int, str]],
) -> list[tuple[str, str]]:
    """For each event, check if a negation marker appears within 3 tokens
    before it. Returns list of (polarity_label, event_atom) tuples for
    negated events only.
    """
    negations: list[tuple[str, str]] = []
    for pos, event_atom in events:
        window_lo = max(0, pos - 3)
        window = tokens[window_lo:pos]
        if any(t in POLARITY_MARKERS for t in window):
            negations.append(("neg", event_atom))
    return negations


# ---------------------------------------------------------------------------
# Public data classes.
# ---------------------------------------------------------------------------

@dataclass
class ExtractedSample:
    uid: str | None
    reference: str
    atoms: dict[str, list[str]] = field(default_factory=dict)
    relations: dict[str, list[list[str]]] = field(default_factory=dict)
    num_propositions: int = 0
    extraction_confidence: str = "high"    # "high" | "medium" | "low"
    extraction_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "uid": self.uid,
            "reference": self.reference,
            "atoms": self.atoms,
            "relations": self.relations,
            "num_propositions": self.num_propositions,
            "extraction_confidence": self.extraction_confidence,
            "extraction_notes": list(self.extraction_notes),
        }


# ---------------------------------------------------------------------------
# Main extraction.
# ---------------------------------------------------------------------------

def _empty_atom_dict() -> dict[str, list[str]]:
    return {slot.value: [] for slot in Slot}


def _empty_relation_dict() -> dict[str, list[list[str]]]:
    return {rel.value: [] for rel in Relation}


def extract(reference: str, uid: str | None = None) -> ExtractedSample:
    """Extract atoms and relations from a single German reference."""
    sample = ExtractedSample(
        uid=uid,
        reference=reference,
        atoms=_empty_atom_dict(),
        relations=_empty_relation_dict(),
    )
    if not reference or not reference.strip():
        sample.extraction_confidence = "low"
        sample.extraction_notes.append("empty reference")
        return sample

    tokens = tokenize(reference)
    if not tokens:
        sample.extraction_confidence = "low"
        sample.extraction_notes.append("no tokens")
        return sample

    clauses = split_clauses(tokens)
    if not clauses:
        clauses = [tokens]

    total_props = 0
    seen_atoms: dict[str, set[str]] = {slot.value: set() for slot in Slot}
    notes: list[str] = []

    for clause in clauses:
        # Slot-level lexicon matches within the clause.
        clause_events = match_lexicon(clause, EVENT_LEXICON)
        clause_locations_raw = match_lexicon(clause, LOCATION_LEXICON)
        clause_wind_raw = match_lexicon(clause, WIND_DIRECTION_LEXICON)
        clause_times = match_lexicon(clause, TIME_LEXICON)
        clause_intensities = match_lexicon(clause, INTENSITY_LEXICON)
        clause_temps = extract_temperatures(clause)

        # Resolve wind-direction ambiguity.
        clause_locations, clause_wind = resolve_wind_directions(
            clause, clause_locations_raw, clause_wind_raw
        )

        # Bucket temperatures.
        clause_temp_atoms = [(pos, temperature_bucket(v)) for pos, v in clause_temps]

        # Detect polarity per event.
        clause_negations = detect_polarity_for_events(clause, clause_events)

        # Register atoms (dedup within slot; keep insertion order across clauses).
        for pos, atom in clause_events:
            if atom not in seen_atoms[Slot.EVENT.value]:
                seen_atoms[Slot.EVENT.value].add(atom)
                sample.atoms[Slot.EVENT.value].append(atom)
        for pos, atom in clause_locations:
            if atom not in seen_atoms[Slot.LOCATION.value]:
                seen_atoms[Slot.LOCATION.value].add(atom)
                sample.atoms[Slot.LOCATION.value].append(atom)
        for pos, atom in clause_times:
            if atom not in seen_atoms[Slot.TIME.value]:
                seen_atoms[Slot.TIME.value].add(atom)
                sample.atoms[Slot.TIME.value].append(atom)
        for pos, atom in clause_temp_atoms:
            if atom not in seen_atoms[Slot.TEMPERATURE.value]:
                seen_atoms[Slot.TEMPERATURE.value].add(atom)
                sample.atoms[Slot.TEMPERATURE.value].append(atom)
        for pos, atom in clause_intensities:
            if atom not in seen_atoms[Slot.INTENSITY.value]:
                seen_atoms[Slot.INTENSITY.value].add(atom)
                sample.atoms[Slot.INTENSITY.value].append(atom)
        for pos, atom in clause_wind:
            if atom not in seen_atoms[Slot.WIND_DIRECTION.value]:
                seen_atoms[Slot.WIND_DIRECTION.value].add(atom)
                sample.atoms[Slot.WIND_DIRECTION.value].append(atom)
        for polarity, event in clause_negations:
            key = f"{polarity}:{event}"
            if key not in seen_atoms[Slot.POLARITY.value]:
                seen_atoms[Slot.POLARITY.value].add(key)
                sample.atoms[Slot.POLARITY.value].append(polarity)

        # Relation binding within clause.
        # Deduplicate atom lists first (locations like "schleswig holstein"
        # would otherwise contribute two identical bindings).
        uniq_events = _dedup_positioned(clause_events)
        uniq_locations = _dedup_positioned(clause_locations)
        uniq_times = _dedup_positioned(clause_times)
        uniq_wind = _dedup_positioned(clause_wind)

        clause_rels: dict[str, set[tuple[str, str]]] = {rel.value: set() for rel in Relation}

        # event-location: cross-product of events and locations in the clause.
        for _, e in uniq_events:
            for _, loc in uniq_locations:
                clause_rels[Relation.EVENT_LOCATION.value].add((e, loc))
        # temperature-location: pair each temp with the *nearest* location
        # by token position within the clause (avoids exploding combinations).
        if clause_temp_atoms and uniq_locations:
            for temp_pos, temp_atom in clause_temp_atoms:
                nearest_loc = min(
                    uniq_locations, key=lambda x: abs(x[0] - temp_pos)
                )
                clause_rels[Relation.TEMPERATURE_LOCATION.value].add(
                    (temp_atom, nearest_loc[1])
                )
        # event-time: cross-product within clause.
        for _, e in uniq_events:
            for _, t in uniq_times:
                clause_rels[Relation.EVENT_TIME.value].add((e, t))
        # wind_direction-location: only if we resolved wind direction here.
        if uniq_wind and uniq_locations:
            for _, wdir in uniq_wind:
                nearest_loc = uniq_locations[0]  # heuristic: first location
                clause_rels[Relation.WIND_DIRECTION_LOCATION.value].add(
                    (wdir, nearest_loc[1])
                )
        # negation-scope
        for polarity, event in clause_negations:
            clause_rels[Relation.NEGATION_SCOPE.value].add((polarity, event))

        # Emit clause relations (still allow the same relation to appear from
        # a different clause -- that reflects true repetition).
        for rel_name, pairs in clause_rels.items():
            for pair in pairs:
                sample.relations[rel_name].append(list(pair))
                total_props += 1

    sample.num_propositions = total_props

    # Confidence heuristics.
    n_atoms = sum(len(v) for v in sample.atoms.values())
    if n_atoms == 0:
        sample.extraction_confidence = "low"
        notes.append("no atoms extracted")
    elif total_props == 0 and n_atoms > 0:
        sample.extraction_confidence = "medium"
        notes.append("atoms present but no relations bound")

    sample.extraction_notes.extend(notes)
    return sample


def extract_many(rows: Iterable[tuple[str, str]]) -> list[ExtractedSample]:
    """Convenience: run `extract` over an iterable of (uid, reference)."""
    return [extract(reference=ref, uid=uid) for uid, ref in rows]


def _dedup_positioned(items: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Return `items` with atoms deduplicated by canonical value, keeping the
    earliest occurrence (its position). Preserves ordering by position.
    """
    seen: set[str] = set()
    out: list[tuple[int, str]] = []
    for pos, atom in sorted(items, key=lambda x: x[0]):
        if atom in seen:
            continue
        seen.add(atom)
        out.append((pos, atom))
    return out
