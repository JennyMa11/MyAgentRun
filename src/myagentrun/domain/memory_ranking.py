"""Deterministic lexical selectors; the v1 path stays frozen for paired comparison.

Both selectors are pure, offline and reproducible: the same records and the same
query must yield the same cards on any platform and in any process.  Nothing here
reads a clock, a random source or a network, and no model participates in ranking.

``memory_lexical_v1`` counts *distinct* query terms per field with fixed weights
(the frozen P2 baseline).  ``memory_lexical_v2`` is BM25F over the same four
fields: per-field term frequency with saturation, per-field document-frequency
IDF and per-field length normalisation, fused with the same field weights.  The
two therefore agree on which field matters and differ only in how a hit counts.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Literal

type SelectorName = Literal["memory_lexical_v1", "memory_lexical_v2"]

SELECTOR_V1: SelectorName = "memory_lexical_v1"
SELECTOR_V2: SelectorName = "memory_lexical_v2"
SELECTORS: tuple[SelectorName, ...] = (SELECTOR_V1, SELECTOR_V2)
DEFAULT_SELECTOR: SelectorName = SELECTOR_V1

_ID = re.compile(r"[a-z][a-z0-9_]{0,127}")

# The v1 field weights.  v2 keeps them as fusion weights, so an ablation isolates
# the counting rule (IDF, saturation, length) instead of also moving the weights.
FIELD_WEIGHTS: dict[str, float] = {"topic": 8.0, "paths": 4.0, "error_type": 4.0, "content": 1.0}
FIELDS: tuple[str, ...] = ("topic", "paths", "error_type", "content")

_K1 = 1.2
_B = 0.5


def _han(character: str) -> bool:
    return "\u3400" <= character <= "\u9fff" or "\U00020000" <= character <= "\U000323af"


def term_counts(text: str) -> Counter[str]:
    """NFKC/casefold terms with multiplicity: ASCII words, Han bigrams, other runs.

    A Han run of length n yields its n-1 adjacent bigrams (a single-character run
    yields that character), so repeated bigrams in one field are counted.
    """
    counts: Counter[str] = Counter()
    run: list[str] = []
    previous = ""

    def flush() -> None:
        word = "".join(run)
        if previous == "han" and len(word) > 1:
            counts.update(word[i : i + 2] for i in range(len(word) - 1))
        elif word:
            counts[word] += 1
        run.clear()

    for character in unicodedata.normalize("NFKC", text).casefold():
        if _han(character):
            category = "han"
        elif (character.isascii() and character.isalnum()) or character.isalpha():
            category = "word"
        else:
            category = ""
        if category != previous or not category:
            flush()
        if category:
            run.append(character)
        previous = category
    flush()
    return counts


def lexical_terms(text: str) -> frozenset[str]:
    """Distinct terms of :func:`term_counts`, preserving the frozen v1 semantics."""
    return frozenset(term_counts(text))


@dataclass(frozen=True, slots=True)
class FieldedText:
    """The four searchable fields of one record, already rendered as text."""

    topic: str
    paths: str
    error_type: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {
            "topic": self.topic,
            "paths": self.paths,
            "error_type": self.error_type,
            "content": self.content,
        }

    def tokens(self) -> frozenset[str]:
        """Every term in any field, used to explain a zero-result query."""
        return frozenset().union(*(term_counts(text).keys() for text in self.as_dict().values()))


def _tokenized(documents: tuple[FieldedText, ...]) -> list[dict[str, Counter[str]]]:
    return [
        {field: term_counts(text) for field, text in document.as_dict().items()}
        for document in documents
    ]


@dataclass(frozen=True, slots=True)
class RankedDocument:
    """One scored record: the integer score plus which fields produced it."""

    score: int
    matched_fields: tuple[str, ...]


def _v1_document(fields: dict[str, Counter[str]], query: frozenset[str]) -> RankedDocument:
    total = 0
    matched: list[str] = []
    for field, counts in fields.items():
        hits = len(query & counts.keys())
        if hits:
            matched.append(field)
            total += hits * int(FIELD_WEIGHTS[field])
    return RankedDocument(total, tuple(matched))


def rank_v1(
    documents: tuple[FieldedText, ...], query: frozenset[str]
) -> tuple[RankedDocument, ...]:
    """Distinct-term counts per field, weighted; the frozen P2 baseline."""
    return tuple(_v1_document(fields, query) for fields in _tokenized(documents))


def _idf(document_frequency: int, total: int) -> float:
    """BM25 IDF with a small floor.

    The floor matters because a project holds at most 100 records: without it a
    term appearing in every record scores zero and two rules that both match the
    query become indistinguishable.
    """
    return max(
        0.05, math.log(1.0 + (total - document_frequency + 0.5) / (document_frequency + 0.5))
    )


def rank_v2(
    documents: tuple[FieldedText, ...], query: frozenset[str]
) -> tuple[RankedDocument, ...]:
    """BM25F over the four fields, scaled to integers so card scores stay ints.

    Scaling by 100 and rounding preserves the ordering; equal scores still break
    on ``entry_id`` downstream, so ranking stays total and deterministic.
    """
    if not documents:
        return ()
    if not query:
        return tuple(RankedDocument(0, ()) for _ in documents)
    tokenized = _tokenized(documents)
    total = len(documents)
    statistics: dict[str, tuple[dict[str, int], float]] = {}
    for field in FIELDS:
        lengths = [sum(fields[field].values()) for fields in tokenized]
        frequencies = {
            term: sum(1 for fields in tokenized if term in fields[field]) for term in sorted(query)
        }
        statistics[field] = (frequencies, sum(lengths) / len(lengths) if lengths else 0.0)
    ranked: list[RankedDocument] = []
    for fields in tokenized:
        fused = 0.0
        matched_fields: list[str] = []
        for field, counts in fields.items():
            matched = query & counts.keys()
            if not matched:
                continue
            matched_fields.append(field)
            frequencies, average = statistics[field]
            length = sum(counts.values())
            normalisation = 1.0 - _B + _B * (length / average if average else 1.0)
            field_score = 0.0
            for term in matched:
                frequency = float(counts[term])
                field_score += _idf(frequencies[term], total) * (
                    frequency * (_K1 + 1.0) / (frequency + _K1 * normalisation)
                )
            fused += FIELD_WEIGHTS[field] * field_score
        ranked.append(RankedDocument(round(fused * 100), tuple(matched_fields)))
    return tuple(ranked)


def rank(
    documents: tuple[FieldedText, ...], query: frozenset[str], selector: SelectorName
) -> tuple[RankedDocument, ...]:
    """Score every document in input order under one named selector."""
    if selector == SELECTOR_V2:
        return rank_v2(documents, query)
    if selector == SELECTOR_V1:
        return rank_v1(documents, query)
    raise ValueError("unknown_memory_selector")
