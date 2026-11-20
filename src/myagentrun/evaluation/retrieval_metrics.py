"""Ranking-quality metrics for project-memory discovery.

This module measures one thing: given a frozen corpus of approved notes and a
frozen set of ``query -> expected entry_ids`` pairs, how well does a selector put
the expected notes in the top five?  It does not run a model, a tool loop or a
task, and it cannot produce a claim about task success.

Evidence rules follow the harness metrics convention: a ratio reports its
numerator, the checked denominator and the declared opportunity count, and an
unreachable denominator is ``None`` rather than 0% or 100%.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from myagentrun.domain import JsonValue
from myagentrun.domain.memory import MemoryEntry, MemorySelection
from myagentrun.domain.memory_discovery import MAX_QUERY_CHARS, discover_memory
from myagentrun.domain.memory_ranking import SelectorName

# The depth that a progressive read can actually reach: search_memory is capped
# at five cards, and the Context projection is capped at five entries as well.
RETRIEVAL_DEPTH = 5


@dataclass(frozen=True, slots=True)
class GoldQuery:
    query_id: str
    query: str
    gold: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.query_id or not self.gold or not self.query.strip():
            raise ValueError("gold_query_invalid")
        if len(self.query) > MAX_QUERY_CHARS or len({*self.gold}) != len(self.gold):
            raise ValueError("gold_query_invalid")


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    query_id: str
    selector: SelectorName
    ranked: tuple[str, ...]
    gold: tuple[str, ...]
    recall: float
    reciprocal_rank: float
    ndcg: float
    term_overlap: bool = True
    """False means the query shared no term with any note: a lexical miss by design."""

    @property
    def hit(self) -> bool:
        return self.recall > 0.0

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "query_id": self.query_id,
            "selector": self.selector,
            "ranked": list(self.ranked),
            "gold": list(self.gold),
            "recall_at_5": self.recall,
            "reciprocal_rank_at_5": self.reciprocal_rank,
            "ndcg_at_5": self.ndcg,
            "term_overlap": self.term_overlap,
        }


@dataclass(frozen=True, slots=True)
class Ratio:
    """A rate that refuses to pretend an empty denominator is zero."""

    numerator: int
    denominator: int

    @property
    def value(self) -> float | None:
        return self.numerator / self.denominator if self.denominator else None

    def to_json(self) -> dict[str, JsonValue]:
        return {"numerator": self.numerator, "denominator": self.denominator, "value": self.value}


@dataclass(frozen=True, slots=True)
class SelectorReport:
    selector: SelectorName
    depth: int
    outcomes: tuple[QueryOutcome, ...]
    corpus_sha256: str

    @property
    def queries(self) -> int:
        return len(self.outcomes)

    @property
    def hit_rate(self) -> Ratio:
        """Queries with at least one gold note inside the top five."""
        return Ratio(sum(1 for item in self.outcomes if item.hit), self.queries)

    @property
    def mean_recall(self) -> float | None:
        """Mean per-query recall@5; differs from hit rate when a query has many golds."""
        return sum(item.recall for item in self.outcomes) / self.queries if self.queries else None

    @property
    def mrr(self) -> float | None:
        return (
            sum(item.reciprocal_rank for item in self.outcomes) / self.queries
            if self.queries
            else None
        )

    @property
    def ndcg(self) -> float | None:
        return sum(item.ndcg for item in self.outcomes) / self.queries if self.queries else None

    @property
    def zero_overlap(self) -> tuple[str, ...]:
        """Queries that no lexical selector can answer; reported, never silently dropped."""
        return tuple(item.query_id for item in self.outcomes if not item.term_overlap)

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "selector": self.selector,
            "depth": self.depth,
            "corpus_sha256": self.corpus_sha256,
            "queries": self.queries,
            "hit_at_5": self.hit_rate.to_json(),
            "mean_recall_at_5": self.mean_recall,
            "mrr_at_5": self.mrr,
            "ndcg_at_5": self.ndcg,
            "zero_overlap_queries": list(self.zero_overlap),
            "outcomes": [item.to_json() for item in self.outcomes],
        }


def corpus_sha256(entries: tuple[MemoryEntry, ...]) -> str:
    material = "\n".join(
        f"{entry.entry_id}\t{entry.revision}\t{entry.content_sha256}"
        for entry in sorted(entries, key=lambda item: item.entry_id)
    )
    return hashlib.sha256(material.encode()).hexdigest()


def _dcg(gains: list[float]) -> float:
    return sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))


def _outcome(
    gold: GoldQuery, ranked: tuple[str, ...], selector: SelectorName, *, term_overlap: bool
) -> QueryOutcome:
    depth = RETRIEVAL_DEPTH
    found = [entry_id for entry_id in ranked[:depth] if entry_id in gold.gold]
    recall = len(set(found)) / len(set(gold.gold))
    reciprocal = 1.0 / (ranked.index(found[0]) + 1) if found else 0.0
    ideal = _dcg([1.0] * min(len(gold.gold), depth))
    actual = _dcg([1.0 if entry_id in gold.gold else 0.0 for entry_id in ranked[:depth]])
    return QueryOutcome(
        gold.query_id,
        selector,
        ranked,
        gold.gold,
        recall,
        reciprocal,
        actual / ideal if ideal else 0.0,
        term_overlap,
    )


def evaluate_selector(
    entries: tuple[MemoryEntry, ...],
    queries: tuple[GoldQuery, ...],
    *,
    selector: SelectorName,
    project_id: str,
    applicable_version: str,
) -> SelectorReport:
    """Score one selector over the frozen corpus; no storage, model or clock access."""
    outcomes: list[QueryOutcome] = []
    for gold in queries:
        result = discover_memory(
            entries,
            project_id=project_id,
            applicable_version=applicable_version,
            query=gold.query,
            selector=selector,
        )
        outcomes.append(
            _outcome(
                gold,
                tuple(card.entry_id for card in result.cards),
                selector,
                term_overlap=result.term_overlap,
            )
        )
    return SelectorReport(selector, RETRIEVAL_DEPTH, tuple(outcomes), corpus_sha256(entries))


def compare(reports: tuple[SelectorReport, ...]) -> dict[str, JsonValue]:
    """Return a paired, per-query comparison; no significance claim is made."""
    if not reports:
        return {"selectors": [], "paired": []}
    baseline = reports[0]
    index = {item.query_id: item for item in baseline.outcomes}
    paired: list[JsonValue] = []
    for report in reports[1:]:
        for outcome in report.outcomes:
            before = index.get(outcome.query_id)
            if before is None:
                continue
            paired.append(
                {
                    "query_id": outcome.query_id,
                    "baseline": before.selector,
                    "candidate": outcome.selector,
                    "recall_delta": outcome.recall - before.recall,
                    "reciprocal_rank_delta": outcome.reciprocal_rank - before.reciprocal_rank,
                    "ndcg_delta": outcome.ndcg - before.ndcg,
                }
            )
    return {
        "selectors": [report.selector for report in reports],
        "paired": paired,
        "note": "Per-query paired differences; this fixed set is not a significance test.",
    }


def selection_is_empty(selection: MemorySelection) -> bool:
    """Guard for callers that must distinguish 'no knowledge' from 'no match'."""
    return not selection.entries
