"""Retrieval gold-set contracts: metrics, loader, and the paired selector report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from myagentrun.domain.memory import MemoryEntry, MemoryEvidence, MemoryEvidenceKind, MemoryKind
from myagentrun.domain.memory_discovery import discover_memory
from myagentrun.domain.memory_ranking import SELECTOR_V1, SELECTOR_V2
from myagentrun.evaluation.retrieval_gold import GOLD_SET_PATH, evaluate, load_gold_set, report
from myagentrun.evaluation.retrieval_metrics import GoldQuery, Ratio, compare, evaluate_selector


def test_the_frozen_gold_set_loads_and_is_internally_consistent() -> None:
    gold_set = load_gold_set()
    assert gold_set.case_set_id == "memory_retrieval_gold_v1"
    assert len(gold_set.queries) == 12
    assert len(gold_set.entries) == 12
    assert all(entry.status.value == "committed" for entry in gold_set.entries)
    assert {gold for query in gold_set.queries for gold in query.gold} == {
        entry.entry_id for entry in gold_set.entries
    }


def test_the_gold_set_is_frozen_on_disk() -> None:
    """Every note and query is data, so a silent edit changes the corpus hash too."""
    gold_set = load_gold_set()
    report_v1 = evaluate(gold_set, (SELECTOR_V1,))[0]
    payload = json.loads(GOLD_SET_PATH.read_text(encoding="utf-8"))
    assert payload["project_id"] == gold_set.project_id
    assert len(payload["queries"]) == report_v1.queries
    assert report_v1.corpus_sha256 == evaluate(load_gold_set(), (SELECTOR_V1,))[0].corpus_sha256


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data["queries"][0].__setitem__("gold", ["g_missing"]),
        lambda data: data["queries"].append(dict(data["queries"][0])),
        lambda data: data["notes"].append(dict(data["notes"][0])),
        lambda data: data.__setitem__("schema_version", "2.0"),
    ],
)
def test_invalid_gold_sets_are_rejected(tmp_path: Path, mutate: object) -> None:
    payload = json.loads(GOLD_SET_PATH.read_text(encoding="utf-8"))
    assert callable(mutate)
    mutate(payload)
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="gold_set"):
        load_gold_set(path)


def _committed(entry_id: str, topic: str, content: str) -> MemoryEntry:
    proposed = MemoryEntry.propose(
        entry_id,
        "p",
        MemoryKind.DIAGNOSIS,
        topic=topic,
        content=content,
        source_run_id="run_one",
        applicable_version="v1",
    )
    validated = proposed.validate(
        MemoryEvidence(
            "a" * 64,
            "p",
            "v1",
            MemoryEvidenceKind.INDEPENDENT_ACCEPTANCE,
            source_run_id="run_one",
            accepted=True,
            source_exists=True,
        )
    )
    approved = validated.approve(
        expected_revision=1,
        content_sha256=validated.content_sha256,
        approver="fixture_operator",
        approved_at="2026-01-01T00:00:00+00:00",
    )
    return approved.commit(expected_revision=1, approval_sha256=approved.approval_sha256)


def test_rank_position_drives_reciprocal_rank_and_ndcg() -> None:
    """Two notes match; whichever the selector puts first must score higher."""
    strong = _committed("strong", "sqlite_cancel", "sqlite sqlite cancel")
    weak = _committed("weak", "unrelated", "sqlite appears once here")
    gold = GoldQuery("q", "sqlite cancel", ("weak",))

    outcome = evaluate_selector(
        (strong, weak), (gold,), selector=SELECTOR_V1, project_id="p", applicable_version="v1"
    ).outcomes[0]

    assert outcome.ranked[0] == "strong"
    assert outcome.ranked.index("weak") == 1
    assert outcome.reciprocal_rank == 0.5
    assert 0.0 < outcome.ndcg < 1.0
    assert outcome.recall == 1.0


def test_a_query_whose_gold_note_is_absent_scores_zero_not_unknown() -> None:
    corpus = (_committed("only", "sqlite", "sqlite cancel"),)
    outcome = evaluate_selector(
        corpus,
        (
            GoldQuery(
                "q",
                "sqlite",
                ("only",),
            ),
        ),
        selector=SELECTOR_V1,
        project_id="p",
        applicable_version="v1",
    ).outcomes[0]
    assert outcome.reciprocal_rank == 1.0
    absent = evaluate_selector(
        corpus,
        (
            GoldQuery(
                "q2",
                "sqlite",
                ("absent",),
            ),
        ),
        selector=SELECTOR_V1,
        project_id="p",
        applicable_version="v1",
    ).outcomes[0]
    assert absent.reciprocal_rank == 0.0
    assert absent.recall == 0.0


def test_a_ratio_with_no_denominator_is_unknown_not_zero() -> None:
    assert Ratio(0, 0).value is None
    assert Ratio(1, 4).value == 0.25


def test_empty_inputs_do_not_invent_a_score() -> None:
    result = evaluate_selector(
        (), (), selector=SELECTOR_V1, project_id="p", applicable_version="v1"
    )
    assert result.queries == 0
    assert result.hit_rate.value is None
    assert result.mrr is None
    assert result.ndcg is None
    assert compare((result,))["paired"] == []


def test_the_report_keeps_both_selectors_on_identical_input() -> None:
    gold_set = load_gold_set()
    payload = report(gold_set, (SELECTOR_V1, SELECTOR_V2))
    assert payload["corpus_notes"] == 12
    assert payload["queries"] == 12
    reports = payload["reports"]
    assert isinstance(reports, list)
    assert len(reports) == 2
    hashes = {entry["corpus_sha256"] for entry in reports}
    assert len(hashes) == 1, "a paired comparison must read one corpus"
    assert {entry["selector"] for entry in reports} == {SELECTOR_V1, SELECTOR_V2}


def test_a_paraphrase_without_shared_terms_is_reported_as_zero_overlap() -> None:
    """The documented limit must be visible instead of looking like an empty corpus."""
    gold_set = load_gold_set()
    payload = report(gold_set, (SELECTOR_V1, SELECTOR_V2))
    reports = payload["reports"]
    assert isinstance(reports, list)
    zero = {entry["selector"]: entry["zero_overlap_queries"] for entry in reports}
    assert zero[SELECTOR_V1] == zero[SELECTOR_V2] == ["q05"]


def test_the_report_states_what_it_does_not_measure() -> None:
    payload = report(load_gold_set(), (SELECTOR_V1,))
    scope = str(payload["scope"])
    assert "not a task-success result" in scope


def _scoped_record(entry_id: str, *, content: str, paths: tuple[str, ...] = ()) -> MemoryEntry:
    entry = MemoryEntry.propose(
        entry_id,
        "alpha",
        MemoryKind.DIAGNOSIS,
        topic=entry_id,
        content=content,
        source_run_id="run_one",
        applicable_version="v1",
        paths=paths,
    )
    evidence = MemoryEvidence(
        "a" * 64,
        "alpha",
        "v1",
        MemoryEvidenceKind.INDEPENDENT_ACCEPTANCE,
        "run_one",
        accepted=True,
        source_exists=True,
    )
    validated = entry.validate(evidence)
    approved = validated.approve(
        expected_revision=1,
        content_sha256=validated.content_sha256,
        approver="fixture_operator",
        approved_at="2026-01-01T00:00:00+00:00",
    )
    return approved.commit(expected_revision=1, approval_sha256=approved.approval_sha256)


def test_term_overlap_reports_the_discovery_set_not_the_caller_filter() -> None:
    """``term_overlap`` must answer "does this project know the word at all?".

    The field documents a corpus-level limit: false means no eligible record shares
    a term with the query.  Computing it after the caller's path/error filter made a
    narrowed search report a corpus-level miss that does not exist, which then
    misdirects the diagnosis toward "add semantic recall".
    """
    matching = _scoped_record(
        "a_matching", content="The pool write must be retried.", paths=("src/other",)
    )
    other = _scoped_record("b_other", content="Unrelated note about formatting.", paths=("src/db",))

    unfiltered = discover_memory(
        (matching, other), project_id="alpha", applicable_version="v1", query="write"
    )
    narrowed = discover_memory(
        (matching, other),
        project_id="alpha",
        applicable_version="v1",
        query="write",
        paths=("src/db",),
    )

    assert unfiltered.term_overlap is True
    # The explicit path filter removes the only record that shares a term, so the
    # caller sees no cards; the corpus-level diagnostic must still say "this project
    # knows the word", otherwise the miss is misreported as a lexical gap.
    assert [card.entry_id for card in unfiltered.cards] == ["a_matching"]
    assert narrowed.cards == ()
    assert narrowed.term_overlap is True


def test_term_overlap_is_false_only_when_no_eligible_record_shares_a_term() -> None:
    entries = (
        _scoped_record("a_one", content="Sqlite locking requires readback."),
        _scoped_record("b_two", content="Formatting conventions."),
    )
    result = discover_memory(
        entries, project_id="alpha", applicable_version="v1", query="中文编码乱码"
    )
    assert result.cards == ()
    assert result.term_overlap is False
