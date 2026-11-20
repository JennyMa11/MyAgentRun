"""BM25F selector contracts; v1 stays the default and must not move."""

from __future__ import annotations

import random

import pytest

from myagentrun.domain.memory_ranking import (
    FieldedText,
    RankedDocument,
    lexical_terms,
    rank,
    rank_v1,
    rank_v2,
    term_counts,
)


def _document(topic: str, content: str, *, paths: str = "", error: str = "") -> FieldedText:
    return FieldedText(topic, paths, error, content)


def test_ranking_selectors_are_named_and_closed() -> None:
    documents = (_document("sqlite", "sqlite"),)
    with pytest.raises(ValueError, match="unknown_memory_selector"):
        rank(documents, lexical_terms("sqlite"), "memory_lexical_v3")  # type: ignore[arg-type]


def test_v1_and_v2_agree_when_there_is_nothing_to_rank() -> None:
    assert rank((), frozenset(), "memory_lexical_v1") == ()
    assert rank((), frozenset(), "memory_lexical_v2") == ()
    documents = (_document("sqlite", "body"),)
    assert rank(documents, frozenset(), "memory_lexical_v1") == (RankedDocument(0, ()),)
    assert rank(documents, frozenset(), "memory_lexical_v2") == (RankedDocument(0, ()),)


def test_term_counts_keep_multiplicity_while_terms_stay_frozen() -> None:
    counts = term_counts("sqlite sqlite cancel")
    assert counts["sqlite"] == 2
    assert lexical_terms("sqlite sqlite cancel") == frozenset({"sqlite", "cancel"})
    # Overlapping bigrams are counted per occurrence, exactly as v1 extracts them.
    counts = term_counts("取消取消")
    assert counts["取消"] == 2
    assert lexical_terms("取消取消") == frozenset(counts)
    assert term_counts("!?!") == {}


def test_idf_lifts_a_rare_term_over_a_common_one() -> None:
    """At least one record; 'common' appears everywhere, 'rare' appears once."""
    documents = (
        *(_document(f"t{index}", "common body") for index in range(8)),
        _document("t9", "rare body"),
    )

    rare = rank_v2(documents, lexical_terms("rare"))[-1].score
    common = rank_v2(documents, lexical_terms("common"))[0].score

    assert rare > common > 0
    # The frozen v1 baseline cannot tell them apart: both are one distinct term.
    assert rank_v1(documents, lexical_terms("rare"))[-1].score == 1
    assert rank_v1(documents, lexical_terms("common"))[0].score == 1


def test_saturation_keeps_a_repeated_term_bounded() -> None:
    once = _document("t1", "retry the request")
    three = _document("t2", "retry retry retry the request")
    documents = (once, three)

    ranked = rank_v2(documents, lexical_terms("retry"))

    assert ranked[1].score > ranked[0].score
    assert ranked[1].score < ranked[0].score * 3


def test_length_normalisation_favours_the_focused_record() -> None:
    focused = _document("t1", "retry once")
    padded = _document("t2", "retry " + " ".join(f"unrelated{index}" for index in range(60)))
    documents = (focused, padded)

    ranked = rank_v2(documents, lexical_terms("retry"))

    assert ranked[0].score > ranked[1].score


def test_field_weights_survive_the_selector_change() -> None:
    """The same term in ``topic`` must still outrank the same term in ``content``."""
    documents = (
        _document("unrelated_topic", "sqlite"),
        _document("sqlite", "unrelated body"),
    )

    ranked = rank_v2(documents, lexical_terms("sqlite"))

    assert ranked[1].score > ranked[0].score


def test_matched_fields_are_reported_by_both_selectors() -> None:
    documents = (_document("sqlite", "cancel", paths="src/db", error="OperationalError"),)
    terms = lexical_terms("sqlite src/db OperationalError cancel")

    assert rank_v1(documents, terms)[0].matched_fields == (
        "topic",
        "paths",
        "error_type",
        "content",
    )
    assert rank_v2(documents, terms)[0].matched_fields == (
        "topic",
        "paths",
        "error_type",
        "content",
    )


def test_v2_is_independent_of_document_order() -> None:
    documents = tuple(_document(f"t{index}", f"sqlite note {index}") for index in range(6))
    terms = lexical_terms("sqlite note")

    direct = rank_v2(documents, terms)
    shuffled = list(documents)
    random.Random(0).shuffle(shuffled)
    shuffled_scores = {
        document.topic: scored
        for document, scored in zip(shuffled, rank_v2(tuple(shuffled), terms), strict=True)
    }

    assert shuffled_scores == {
        document.topic: scored for document, scored in zip(documents, direct, strict=True)
    }


def test_ranking_is_stable_across_processes_and_runs() -> None:
    """No clock, no hash randomisation, no locale: the same input ranks identically."""
    documents = tuple(
        _document(f"t{index}", f"sqlite note {index % 3} 事务取消") for index in range(12)
    )
    terms = lexical_terms("sqlite 事务 取消 note")
    first = rank_v2(documents, terms)
    second = rank_v2(documents, terms)
    assert first == second
    assert all(isinstance(item.score, int) for item in first)
