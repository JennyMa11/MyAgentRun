"""Fencing one dead reservation must not release money or open a general bypass.

The fence exists because a crash between `reserve` and `settle` otherwise makes a
paid run permanently unresumable. These contracts pin the narrowness: exactly one
named operation, its amount unchanged, its row provably unscored, and the evidence
re-validated on every reopen.
"""

import json
from decimal import Decimal
from pathlib import Path

import pytest
from minidsh_rl.cost_store import CostStore
from minidsh_rl.positive_control_v2_recovery import (
    FENCE,
    FencedResources,
    load_fence,
    outstanding,
    record_fence,
)

LEDGER = "ledger.json"


def _ledger(tmp_path: Path, *, pending: dict[str, str], settled: dict[str, str]) -> Path:
    """A ledger with exactly the named charges, so tests control what blocks."""
    state = {
        "schema_version": 2,
        "limit": "100",
        "historical": "0",
        "uncertainty": "0",
        "legacy_hash": "a" * 64,
        "quote_hash": "b" * 64,
        "charges": {
            **{k: {"amount": v, "status": "pending"} for k, v in pending.items()},
            **{k: {"amount": v, "status": "estimated"} for k, v in settled.items()},
        },
        "bills": {},
    }
    path = tmp_path / LEDGER
    path.write_text(json.dumps(state, indent=2))
    return path


def _row(root: Path, run_id: str, *, scored: bool = False) -> Path:
    row = root / "rollouts" / run_id
    row.mkdir(parents=True)
    if scored:
        (row / "learning-binding").mkdir()
    return row


def test_the_fence_names_one_operation_and_keeps_its_amount(tmp_path: Path) -> None:
    """The money stays owed; only the blocker is lifted."""
    root = tmp_path / "run"
    root.mkdir()
    ledger = _ledger(tmp_path, pending={"sample_dead": "0.009", "sample_live": "0.009"}, settled={})
    _row(root, "row_dead")

    fence = record_fence(
        root, ledger, operation="sample_dead", row_run_id="row_dead", reason="crash"
    )
    assert fence.amount == Decimal("0.009")
    assert fence.operation == "sample_dead"

    loaded = load_fence(root, ledger)
    assert loaded == fence
    # Only the fenced one stops blocking; the other pending charge still does.
    assert outstanding(ledger, loaded) == ["sample_live"]


def test_the_fenced_amount_is_still_counted_as_committed(tmp_path: Path) -> None:
    """A fence must never look like a refund."""
    root = tmp_path / "run"
    root.mkdir()
    ledger = _ledger(tmp_path, pending={"sample_dead": "0.009"}, settled={})
    _row(root, "row_dead")
    record_fence(root, ledger, operation="sample_dead", row_run_id="row_dead", reason="crash")

    class Budget:
        available = Decimal("0")
        committed = Decimal("0.009")
        path = ledger

        @property
        def has_pending(self) -> bool:
            return True

    resources = FencedResources(Budget(), load_fence(root, ledger))
    assert resources.committed == Decimal("0.009"), "fencing must not reduce committed spend"
    assert resources.has_pending is False


def test_a_scored_row_cannot_be_fenced(tmp_path: Path) -> None:
    """If the row was sealed, the sample did resolve; do not write it off."""
    root = tmp_path / "run"
    root.mkdir()
    ledger = _ledger(tmp_path, pending={"sample_dead": "0.009"}, settled={})
    _row(root, "row_done", scored=True)
    with pytest.raises(ValueError, match="sealed result"):
        record_fence(root, ledger, operation="sample_dead", row_run_id="row_done", reason="crash")


def test_a_row_that_sampled_but_failed_to_seal_may_be_fenced(tmp_path: Path) -> None:
    """result.json without a binding means sampling finished and validation failed.

    That is exactly the shape of a crash between reserve and settle: the charge may
    be real, the trajectory was never scored, and refusing to fence it would make the
    run unresumable for no gain.
    """
    root = tmp_path / "run"
    root.mkdir()
    ledger = _ledger(tmp_path, pending={"sample_dead": "0.009"}, settled={})
    row = _row(root, "row_raw")
    (row / "result.json").write_text(json.dumps({"status": "budget_exhausted"}))
    (row / "verification").mkdir()
    (row / "verification/result.json").write_text(json.dumps({"passed": False}))

    fence = record_fence(
        root, ledger, operation="sample_dead", row_run_id="row_raw", reason="crash"
    )
    assert fence.sealed_result_absent is True


def test_an_already_resolved_operation_cannot_be_fenced(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    ledger = _ledger(tmp_path, pending={}, settled={"sample_done": "0.009"})
    _row(root, "row_done")
    with pytest.raises(ValueError, match="already resolved"):
        record_fence(root, ledger, operation="sample_done", row_run_id="row_done", reason="crash")


def test_fences_accumulate_and_all_of_them_stop_blocking(tmp_path: Path) -> None:
    """Each crash writes one entry; the newest must not un-fence the older ones."""
    root = tmp_path / "run"
    root.mkdir()
    ledger = _ledger(tmp_path, pending={"a": "0.009", "b": "0.009"}, settled={})
    _row(root, "row_a")
    _row(root, "row_b")
    record_fence(root, ledger, operation="a", row_run_id="row_a", reason="crash")
    record_fence(root, ledger, operation="b", row_run_id="row_b", reason="crash")

    latest = load_fence(root, ledger)
    assert latest is not None
    assert latest.operation == "b"
    assert outstanding(ledger, latest, root=root) == [], "both fenced ops must stay released"


def test_the_same_operation_cannot_be_fenced_twice(tmp_path: Path) -> None:
    """Re-fencing one operation would inflate the written-off total silently."""
    root = tmp_path / "run"
    root.mkdir()
    ledger = _ledger(tmp_path, pending={"a": "0.009"}, settled={})
    _row(root, "row_a")
    record_fence(root, ledger, operation="a", row_run_id="row_a", reason="crash")
    with pytest.raises(ValueError, match="already recorded"):
        record_fence(root, ledger, operation="a", row_run_id="row_a", reason="again")


def test_editing_the_fenced_amount_invalidates_the_resume(tmp_path: Path) -> None:
    """The evidence is re-read every time; a hand-edited amount must not slip through."""
    root = tmp_path / "run"
    root.mkdir()
    ledger = _ledger(tmp_path, pending={"sample_dead": "0.009"}, settled={})
    _row(root, "row_dead")
    record_fence(root, ledger, operation="sample_dead", row_run_id="row_dead", reason="crash")

    raw = json.loads((root / FENCE).read_text())
    raw["entries"][-1]["amount"] = "0"
    (root / FENCE).write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="changed since it was recorded"):
        load_fence(root, ledger)


def test_a_fence_bound_to_another_ledger_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    first = _ledger(tmp_path, pending={"sample_dead": "0.009"}, settled={})
    _row(root, "row_dead")
    record_fence(root, first, operation="sample_dead", row_run_id="row_dead", reason="crash")

    other = tmp_path / "other.json"
    other.write_text(first.read_text())
    with pytest.raises(ValueError, match="different ledger"):
        load_fence(root, other)


def test_spending_after_the_fence_does_not_invalidate_it(tmp_path: Path) -> None:
    """The ledger changes as the run spends; that must not revoke an existing fence."""
    carry = tmp_path / "carry.json"
    carry.write_text(
        json.dumps({"schema_version": 1, "limit": "10", "spent": "0", "pending": {}, "settled": []})
    )
    quote = tmp_path / "quote.json"
    quote.write_text("{}")
    root = tmp_path / "run"
    root.mkdir()
    _row(root, "row_dead")
    with CostStore(
        root / "budget.json",
        legacy=carry,
        quote=quote,
        limit=Decimal("10"),
        uncertainty=Decimal("0"),
    ) as budget:
        budget.reserve("sample_dead", Decimal("0.009"))
        fence = record_fence(
            root, budget.path, operation="sample_dead", row_run_id="row_dead", reason="crash"
        )
        # A later, unrelated charge rewrites the ledger file entirely.
        budget.reserve("sample_later", Decimal("0.01"))
        budget.settle("sample_later", Decimal("0.004"))
        assert load_fence(root, budget.path) == fence


def test_no_fence_means_every_pending_operation_blocks(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, pending={"a": "0.01", "b": "0.02"}, settled={})
    assert outstanding(ledger, None) == ["a", "b"]


def test_the_fence_opens_a_real_ledger_that_would_otherwise_refuse(tmp_path: Path) -> None:
    """End to end: the same CostStore that blocks admission resumes once fenced."""
    carry = tmp_path / "carry.json"
    carry.write_text(
        json.dumps({"schema_version": 1, "limit": "10", "spent": "0", "pending": {}, "settled": []})
    )
    quote = tmp_path / "quote.json"
    quote.write_text("{}")
    root = tmp_path / "run"
    root.mkdir()
    _row(root, "row_dead")

    with CostStore(
        root / "budget.json",
        legacy=carry,
        quote=quote,
        limit=Decimal("10"),
        uncertainty=Decimal("0"),
    ) as budget:
        budget.reserve("sample_dead", Decimal("0.009"))
        assert FencedResources(budget, None).has_pending is True

        fence = record_fence(
            root, budget.path, operation="sample_dead", row_run_id="row_dead", reason="crash"
        )
        assert FencedResources(budget, fence).has_pending is False
        assert budget.committed == Decimal("0.009")
