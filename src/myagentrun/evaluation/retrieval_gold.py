"""Load a frozen retrieval gold set and score selectors against it.

The corpus is built from plain note definitions plus synthetic approval records, so
the measurement needs no private database, no operator and no model.  Approval
metadata is fabricated on purpose and is only used to satisfy the committed-state
invariant of ``MemoryEntry``; it authorizes nothing.

Run it with::

    uv run python scripts/memory_retrieval_eval.py --baseline

The report it writes under ``docs/results/`` is a measurement of ranking quality on
a fixed 12-note corpus.  It is not a task-success result and must not be quoted as
one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from myagentrun.domain.memory import (
    MemoryEntry,
    MemoryEvidence,
    MemoryEvidenceKind,
    MemoryKind,
)
from myagentrun.domain.memory_ranking import SELECTOR_V1, SELECTOR_V2, SelectorName
from myagentrun.evaluation.retrieval_metrics import (
    GoldQuery,
    SelectorReport,
    compare,
    evaluate_selector,
)

GOLD_SET_PATH = Path("benchmarks/memory_v2/retrieval-gold-v1.json")
_KIND = {kind.value: kind for kind in MemoryKind}
_APPROVED_AT = "2026-11-19T00:00:00+00:00"
_SOURCE_RUN = "retrieval_gold_source"
_SYNTHETIC_APPROVER = "fixture_operator"


@dataclass(frozen=True, slots=True)
class GoldSet:
    """A frozen corpus and its frozen queries."""

    case_set_id: str
    project_id: str
    applicable_version: str
    queries: tuple[GoldQuery, ...]
    entries: tuple[MemoryEntry, ...]


def _evidence(proposed: MemoryEntry) -> MemoryEvidence:
    """Synthetic evidence that satisfies the transition chain for this measurement only."""
    if proposed.kind is MemoryKind.ENVIRONMENT:
        return MemoryEvidence(
            "0" * 64,
            proposed.project_id,
            proposed.applicable_version,
            MemoryEvidenceKind.OPERATOR_CONFIGURATION,
            proposed.source_run_id,
            accepted=True,
            source_exists=True,
            confirmed_by=_SYNTHETIC_APPROVER,
            confirmed_at=_APPROVED_AT,
        )
    return MemoryEvidence(
        "0" * 64,
        proposed.project_id,
        proposed.applicable_version,
        MemoryEvidenceKind.INDEPENDENT_ACCEPTANCE,
        proposed.source_run_id,
        accepted=True,
        source_exists=True,
    )


def _committed(note: dict[str, object], project_id: str, version: str) -> MemoryEntry:
    """Build a committed record from a plain note definition.

    Approval metadata is synthetic: it satisfies the committed-state invariant so
    discovery accepts the record, and it authorizes nothing outside this fixture.
    """
    paths = note.get("paths")
    error_type = note.get("error_type")
    proposed = MemoryEntry.propose(
        str(note["entry_id"]),
        project_id,
        _KIND[str(note["kind"])],
        topic=str(note["topic"]),
        content=str(note["content"]),
        source_run_id=_SOURCE_RUN,
        applicable_version=version,
        paths=tuple(str(item) for item in paths) if isinstance(paths, list) else (),
        error_type=None if error_type is None else str(error_type),
    )
    validated = proposed.validate(_evidence(proposed))
    approved = validated.approve(
        expected_revision=1,
        content_sha256=validated.content_sha256,
        approver=_SYNTHETIC_APPROVER,
        approved_at=_APPROVED_AT,
    )
    return approved.commit(expected_revision=1, approval_sha256=approved.approval_sha256)


def load_gold_set(path: Path = GOLD_SET_PATH) -> GoldSet:
    """Read and validate the frozen case set; unknown or missing fields are errors."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("gold_set_schema_unsupported")
    project_id, version = str(payload["project_id"]), str(payload["applicable_version"])
    queries = tuple(
        GoldQuery(str(item["query_id"]), str(item["query"]), tuple(item["gold"]))
        for item in payload["queries"]
    )
    entries = tuple(_committed(note, project_id, version) for note in payload["notes"])
    known = {entry.entry_id for entry in entries}
    unknown = {gold for item in queries for gold in item.gold} - known
    if unknown:
        raise ValueError(f"gold_set_references_unknown_notes: {sorted(unknown)}")
    repeated = len({item.query_id for item in queries}) != len(queries)
    if repeated or len(known) != len(entries):
        raise ValueError("gold_set_duplicate_ids")
    return GoldSet(str(payload["case_set_id"]), project_id, version, queries, entries)


def evaluate(
    gold_set: GoldSet, selectors: tuple[SelectorName, ...] = (SELECTOR_V1, SELECTOR_V2)
) -> tuple[SelectorReport, ...]:
    """Score every requested selector on exactly the same corpus and queries."""
    return tuple(
        evaluate_selector(
            gold_set.entries,
            gold_set.queries,
            selector=selector,
            project_id=gold_set.project_id,
            applicable_version=gold_set.applicable_version,
        )
        for selector in selectors
    )


def report(gold_set: GoldSet, selectors: tuple[SelectorName, ...]) -> dict[str, object]:
    reports = evaluate(gold_set, selectors)
    return {
        "schema_version": "1.0",
        "case_set_id": gold_set.case_set_id,
        "scope": "ranking quality on a fixed synthetic corpus; not a task-success result",
        "corpus_notes": len(gold_set.entries),
        "queries": len(gold_set.queries),
        "reports": [item.to_json() for item in reports],
        "paired": compare(reports),
    }
