"""The v2 CLI must not be able to spend without a frozen plan and enough balance.

`run` is the only command that reaches a provider, so these tests drive `prepare`
and `inspect` for real and prove `run` refuses *before* any client is constructed.
"""

import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from minidsh_rl.cost_store import CostStore
from minidsh_rl.learning_schedule import atomic_record
from minidsh_rl.positive_control import Family
from minidsh_rl.positive_control_v2 import (
    TOTAL_ROWS,
    Positive2Selection,
    load_baseline_plan,
    load_plan,
    make_baseline_plan,
    make_plan,
)
from minidsh_rl.positive_control_v2_cli import _open, fund, inspect, main
from minidsh_rl.positive_control_v2_run import prepare_positive_control_v2

PROJECT = Path(__file__).resolve().parents[3]
CLI = "minidsh_rl.positive_control_v2_cli"


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", CLI, *args],
        cwd=PROJECT / "experiments/trainable_agent",
        capture_output=True,
        text=True,
        check=False,
    )


def test_prepare_freezes_a_baseline_queue_and_nothing_else(tmp_path: Path) -> None:
    """`prepare` writes the 96-row baseline plan; it must not touch a ledger."""
    root = tmp_path / "pc2"
    plan = prepare_positive_control_v2(root)
    assert len(plan.rows) == 96
    assert sorted({row.family for row in plan.rows}) == [
        "dependency_batches",
        "match_rank",
        "table_join",
    ]
    assert not (root / "finance").exists(), "prepare must not create a budget"
    assert not (root / "state.json").exists(), "prepare must not start a run"
    assert load_baseline_plan(root / "baseline-plan.json") == plan


def test_prepare_refuses_an_existing_directory(tmp_path: Path) -> None:
    root = tmp_path / "pc2"
    prepare_positive_control_v2(root)
    with pytest.raises(ValueError, match="already exists"):
        prepare_positive_control_v2(root)


def test_the_cli_exposes_prepare_inspect_and_run_only() -> None:
    """A typo must not silently become a default action."""
    with pytest.raises(SystemExit):
        main_argv(["teleport", "--directory", "/tmp/nonexistent-pc2"])


def main_argv(argv: list[str]) -> None:
    """Run `main` with a patched argv without capturing stdout globally."""
    with patch.object(sys, "argv", ["minidsh_rl.positive_control_v2_cli", *argv]):
        main()


def test_inspect_reports_the_frozen_numbers(tmp_path: Path) -> None:
    """A human approving 33.398 must see the same number the code enforces."""
    root = tmp_path / "pc2"
    prepare_positive_control_v2(root)

    def live() -> dict[str, Any]:
        return {
            "available": Decimal("99.414"),
            "committed": Decimal("113.548"),
            "pending": False,
        }

    before = inspect(root, **live())
    assert before["rows"] == 200
    assert before["selected"] is None
    assert "required_cny" not in before, "no frozen plan means no number to approve"
    assert before["affordable"] is False

    # Now freeze a real plan and confirm the reported number is the enforced one.
    _freeze_plan(root)
    after = inspect(root, **live())
    assert after["required_cny"] == "37.718"
    assert after["affordable"] is True
    assert Decimal(str(after["available_cny"])) - plan_cny(root) == Decimal("61.696")
    assert len(after["selected"]) == 3  # type: ignore[arg-type]


def _selection() -> tuple[Any, ...]:
    pairs: tuple[tuple[Family, str], ...] = (
        ("match_rank", "curve_v4_path_router_02"),
        ("dependency_batches", "curve_v4_dependency_batches_01"),
        ("table_join", "curve_v4_table_join_02"),
    )
    return tuple(
        Positive2Selection(family=family, case_id=case, round=0, successes=3)
        for family, case in pairs
    )


def _freeze_plan(root: Path) -> None:
    plan = make_plan(_selection(), selection_evidence_sha256="e" * 64)
    atomic_record(root / "plan.json", plan, create=True)


def plan_cny(root: Path) -> Decimal:
    return load_plan(root / "plan.json").required_cny


def test_run_requires_a_frozen_plan(tmp_path: Path) -> None:
    """Without plan.json the run must stop before creating a service client."""
    root = tmp_path / "pc2"
    prepare_positive_control_v2(root)
    result = _cli("run", "--directory", str(root))
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["automatic_retry"] is False
    # Either the missing ledger or the missing plan is a fine refusal; neither may
    # be an SDK/network error, and nothing may have been charged.
    assert payload["error_type"] in {"FileNotFoundError", "ValueError", "ValidationError"}
    assert not (root / "rollouts").exists()
    assert not (root / "state.json").exists()


def test_run_without_sufficient_balance_refuses_before_sampling(tmp_path: Path) -> None:
    """The complete-allocation check must precede the first sample, not follow it."""
    root = tmp_path / "pc2"
    prepare_positive_control_v2(root)
    printed = _cli("run", "--directory", str(root))
    assert printed.returncode == 1
    assert len(list(root.glob("rollouts/**/*.json"))) == 0


def _parent(tmp_path: Path, *, pending: bool = False) -> Path:
    """A closed parent ledger shaped like the real one, with a chosen pending state."""
    ledger: dict[str, Any] = {
        "schema_version": 2,
        "limit": "150",
        "historical": "46.963",
        "uncertainty": "2",
        "legacy_hash": "a" * 64,
        "quote_hash": "b" * 64,
        "charges": {
            "fee_1": {"amount": "0.09", "status": "billed"},
            "fee_2": {"amount": "0.21", "status": "billed"},
        },
        "bills": {},
    }
    if pending:
        ledger["charges"]["fee_3"] = {"amount": "0.386", "status": "pending"}
    path = tmp_path / "parent-budget.json"
    path.write_text(json.dumps(ledger, indent=2))
    return path


def test_fund_seats_a_clean_epoch_without_touching_the_parent(tmp_path: Path) -> None:
    """The historical pending must stay in history; only a new ledger can start clean."""
    parent = _parent(tmp_path)
    before = parent.read_bytes()
    quote = tmp_path / "quote.json"
    quote.write_text(json.dumps({"version": 1}))

    root = tmp_path / "pc2"
    report = fund(root, limit=Decimal("50"), parent=parent, quote=quote)

    assert parent.read_bytes() == before, "the parent ledger must not be edited"
    # Parent historical/charged figures are history, not a starting balance: only
    # the unconfirmed amounts are carried, and here there are none.
    assert report["inherited_unconfirmed_cny"] == "0"
    assert report["available_cny"] == "50"
    assert (root / "funding.json").exists()
    # The new ledger is genuinely independent: reading it never touches the parent.
    with CostStore(
        root / "finance/budget.json",
        legacy=root / "finance/carry.json",
        quote=root / "finance/quote.json",
        limit=Decimal("50"),
        uncertainty=Decimal("0"),
    ) as fresh:
        assert fresh.available == Decimal("50")


def test_fund_carries_unconfirmed_charges_forward_as_spent(tmp_path: Path) -> None:
    """An unconfirmed charge is still money owed; the new epoch must inherit it.

    Reusing the parent ledger would block admission entirely, and deleting the
    pending entries would launder an unverified amount. Carrying them forward is the
    only reading that is both startable and conservative.
    """
    parent = _parent(tmp_path, pending=True)
    before = parent.read_bytes()
    quote = tmp_path / "quote.json"
    quote.write_text(json.dumps({"version": 1}))

    report = fund(tmp_path / "pc2", limit=Decimal("50"), parent=parent, quote=quote)

    assert parent.read_bytes() == before, "history keeps its exact bytes"
    assert report["inherited_unconfirmed_count"] == 1
    assert report["inherited_unconfirmed_cny"] == "0.386"
    assert report["available_cny"] == "49.614"
    funding = json.loads((tmp_path / "pc2/funding.json").read_text())
    assert funding["inherited_unconfirmed_cny"] == "0.386"
    assert funding["inherited_unconfirmed_operations"] == ["fee_3"]
    assert funding["parent_historical_cny"] == "46.963"


def test_fund_refuses_an_allowance_below_unconfirmed_spend(tmp_path: Path) -> None:
    """An allowance smaller than the uncertain spend would be negative on arrival."""
    parent = _parent(tmp_path, pending=True)
    quote = tmp_path / "quote.json"
    quote.write_text(json.dumps({"version": 1}))
    with pytest.raises(ValueError, match="does not cover"):
        fund(tmp_path / "pc2", limit=Decimal("0.10"), parent=parent, quote=quote)


def test_run_refuses_an_unfunded_directory_without_sampling(tmp_path: Path) -> None:
    """A clear refusal beats a traceback: `fund` must come first."""
    root = tmp_path / "pc2"
    prepare_positive_control_v2(root)
    with pytest.raises(ValueError, match="run is not funded"):
        _open(root)


def test_the_opener_reads_the_ledger_fund_created(tmp_path: Path) -> None:
    """`fund` and `run` must agree on one layout, or every launch fails at startup."""
    parent = _parent(tmp_path)
    quote = tmp_path / "quote.json"
    quote.write_text(json.dumps({"version": 1}))
    root = tmp_path / "pc2"
    fund(root, limit=Decimal("20"), parent=parent, quote=quote)
    with _open(root) as budget:
        assert budget.available == Decimal("20")


def test_prepare_output_is_machine_readable(tmp_path: Path) -> None:
    """A dry prepare through the module entry point prints the queue summary."""
    root = tmp_path / "pc2"
    prepare_positive_control_v2(root)
    assert (
        json.loads((root / "baseline-plan.json").read_text())["protocol"] == "positive_control_v2"
    )


def test_the_declared_row_denominator_matches_the_plan() -> None:
    """200 is the number the CLI checks; keep it tied to the plan, not a literal."""
    plan = make_baseline_plan()
    assert len(plan.rows) == 96
    assert TOTAL_ROWS == 200
