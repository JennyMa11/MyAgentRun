"""Frozen positive-control v2 queue: denominators, cost and refill binding.

These are the numbers the protocol document quotes, so they are derived here rather
than trusted in prose.  Nothing in this module touches a model, container or SDK.
"""

from decimal import Decimal
from pathlib import Path

import pytest
from minidsh_rl.learning_schedule import (
    EXTRA_COST,
    GROUP_CLOSE_COST,
    GROUP_TRAIN_COST,
    INITIAL_COST,
    RUN_COST,
)
from minidsh_rl.positive_control import (
    CANDIDATES,
    FAMILIES,
    InfeasiblePositiveControlError,
)
from minidsh_rl.positive_control_v2 import (
    BASELINE_ROUNDS,
    BASELINE_ROWS,
    CHECKPOINTS,
    EPOCHS,
    EVAL_ROWS,
    PLANNED_GROUPS,
    REFILL_SLOTS,
    SAMPLED_GROUPS,
    TOTAL_ROWS,
    TRAIN_ROWS,
    Positive2Selection,
    make_baseline_plan,
    make_baseline_rows,
    make_plan,
    plan_sha256,
    required_cny,
    select_tasks,
    verify_plan,
)


def _results(counts: dict[tuple[str, int], int]) -> dict[str, bool]:
    return {
        row.run_id: row.repeat < counts.get((row.family, row.round or 0), 0)
        for row in make_baseline_rows()
    }


def _feasible() -> tuple[Positive2Selection, ...]:
    return select_tasks(
        make_baseline_rows(),
        _results({("match_rank", 0): 3, ("dependency_batches", 0): 4, ("table_join", 0): 2}),
    )


def test_the_frozen_denominators_are_the_ones_the_protocol_quotes() -> None:
    """200 rows over 14 sampled groups; a silent resize would invalidate the quote.

    Widened from 152 to 200 on 2026-11-13: four baseline rounds per family instead of
    two, because both match_rank candidates measured 0/8 under the eval seeds and the
    two remaining candidates had never been sampled. The training and evaluation
    denominators are unchanged; only the baseline window grew.
    """
    assert BASELINE_ROWS == 96
    assert TRAIN_ROWS == 56
    assert EVAL_ROWS == 48
    assert TOTAL_ROWS == 200
    assert PLANNED_GROUPS == 12
    assert SAMPLED_GROUPS == 14
    assert REFILL_SLOTS == 2
    assert CHECKPOINTS == (0, 6, 12)


def test_the_plan_costs_exactly_the_declared_upper_bound() -> None:
    """The budget argument rests on this number, so it is recomputed, not copied."""
    plan = make_plan(_feasible())
    assert plan.required_cny == Decimal("37.718")
    assert plan.required_cny == required_cny(TOTAL_ROWS, PLANNED_GROUPS, SAMPLED_GROUPS)


def test_the_upper_bound_fits_the_live_ledger_and_the_strict_one_is_declared() -> None:
    """Available is 99.414 in the fresh epoch; the widened plan must fit it.

    The new ledger inherited only the three unconfirmed charges (0.586) from the
    previous epoch, whose 35.415 was consumed by the first baseline attempt.
    """
    available = Decimal("99.414")
    ordinary = make_plan(_feasible()).required_cny
    strict = (
        TOTAL_ROWS * RUN_COST
        + SAMPLED_GROUPS * GROUP_TRAIN_COST
        + SAMPLED_GROUPS * GROUP_CLOSE_COST
        + INITIAL_COST
        + EXTRA_COST
    )
    assert ordinary < available
    assert strict > ordinary
    # This time the pessimistic reading also fits: the new epoch is large enough that
    # the worst case is affordable, so the breakers guard against overrun rather than
    # against starting at all.
    assert strict < available
    assert available - ordinary >= Decimal("60")


def test_the_breakers_do_not_fire_in_the_declared_base_case() -> None:
    """A breaker that trips on the expected path is a redesign, not a safeguard.

    At the checkpoint-12 decision every row except the final 24 eval rows is paid,
    leaving 63.856 — far above the 2.50 reserve — so the base case completes. If this
    ever flips, the plan would silently degrade to one measured point.
    """
    available = Decimal("99.414")
    before_final = required_cny(TOTAL_ROWS - 24, PLANNED_GROUPS, SAMPLED_GROUPS)
    assert available - before_final == Decimal("63.856")
    assert available - before_final > Decimal("2.50")
    assert available - required_cny(TOTAL_ROWS, PLANNED_GROUPS, SAMPLED_GROUPS) == Decimal("61.696")


def test_the_optimistic_and_pessimistic_readings_are_both_stated() -> None:
    """40.396 fits 99.414, unlike the first epoch where the worst case overflowed."""
    available = Decimal("99.414")
    pessimistic = (
        TOTAL_ROWS * RUN_COST
        + SAMPLED_GROUPS * GROUP_TRAIN_COST
        + SAMPLED_GROUPS * GROUP_CLOSE_COST
        + INITIAL_COST
        + EXTRA_COST
    )
    assert pessimistic == Decimal("40.396")
    assert available - pessimistic == Decimal("59.018")


def test_refill_reserve_is_priced_whether_or_not_it_is_used() -> None:
    """A reserve that only costs money when needed is not a reserve."""
    plan = make_plan(_feasible())
    assert len(plan.refill_rows) == REFILL_SLOTS * 4
    assert {row.group for row in plan.refill_rows} == {"refill_00", "refill_01"}
    refill = {row.group: row for row in plan.refill_rows}
    # With every candidate now baselined, the reserve falls back to the family's
    # first candidate by frozen order: same family, still chosen without looking at
    # observed difficulty.
    assert refill["refill_00"].case_id == CANDIDATES["match_rank"][0]
    assert refill["refill_01"].case_id == CANDIDATES["dependency_batches"][0]


def test_each_epoch_holds_each_selected_task_exactly_once() -> None:
    plan = make_plan(_feasible())
    for epoch in range(EPOCHS):
        rows = [row for row in plan.planned_rows if row.checkpoint // len(FAMILIES) == epoch]
        assert {row.case_id for row in rows} == set(plan.selected)
        assert len(rows) == len(FAMILIES) * 4


def test_every_group_is_a_complete_group_of_four() -> None:
    plan = make_plan(_feasible())
    counts: dict[str, int] = {}
    for row in plan.planned_rows:
        counts[row.group] = counts.get(row.group, 0) + 1
    assert set(counts.values()) == {4}
    assert len(counts) == PLANNED_GROUPS


def test_selection_outside_the_window_is_infeasible_rather_than_silently_kept() -> None:
    with pytest.raises(InfeasiblePositiveControlError):
        select_tasks(
            make_baseline_rows(),
            _results({("match_rank", 0): 0, ("dependency_batches", 0): 4, ("table_join", 0): 2}),
        )


def test_a_second_baseline_round_can_rescue_a_family() -> None:
    selection = select_tasks(
        make_baseline_rows(),
        _results({("match_rank", 1): 2, ("dependency_batches", 0): 4, ("table_join", 0): 2}),
    )
    assert selection[0].round == 1
    assert selection[0].case_id == CANDIDATES["match_rank"][1]


def test_a_tampered_plan_is_rejected_on_verify() -> None:
    plan = make_plan(_feasible())
    broken = plan.model_copy(update={"required_cny": Decimal("1")})
    with pytest.raises(ValueError, match="declared cost"):
        verify_plan(broken)


def test_the_plan_is_deterministic_across_calls() -> None:
    first, second = make_plan(_feasible()), make_plan(_feasible())
    assert first == second
    assert plan_sha256(first) == plan_sha256(second)


def test_baseline_plan_is_reproducible_and_covers_every_round() -> None:
    plan = make_baseline_plan()
    assert len(plan.rows) == BASELINE_ROWS
    # Every candidate is now measured, so no family can be declared infeasible
    # on the strength of a task that was never sampled.
    assert {row.round for row in plan.rows} == set(range(BASELINE_ROUNDS))
    assert {row.family for row in plan.rows} == set(FAMILIES)
    for family in FAMILIES:
        observed = {row.case_id for row in plan.rows if row.family == family}
        assert observed == set(CANDIDATES[family])


def test_the_protocol_document_quotes_the_generated_numbers() -> None:
    """The prose is the deliverable, so the numbers in it are gated, not remembered.

    A reader cannot re-derive 37.718 from the document alone; if the generator moves
    and the prose does not, the protocol silently promises a budget it will not honour.
    """
    root = Path(__file__).resolve().parents[3]
    text = (root / "docs/trainable-agent-rl/positive-control-v2.md").read_text(encoding="utf-8")
    plan = make_plan(_feasible())

    assert f"{TOTAL_ROWS} 行" in text, f"protocol must quote {TOTAL_ROWS} rows"
    assert str(plan.required_cny) in text, f"protocol must quote {plan.required_cny}"
    assert f"{SAMPLED_GROUPS} 采样" in text, f"protocol must quote {SAMPLED_GROUPS} sampled groups"
    assert f"{PLANNED_GROUPS} 计划" in text, f"protocol must quote {PLANNED_GROUPS} planned groups"
