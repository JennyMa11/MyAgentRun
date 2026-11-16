"""Frozen positive-control v2 queue: three tasks, four epochs, two checkpoints, refills.

v2 differs from v1 in three declared ways: twelve planned groups instead of eighteen,
two checkpoints instead of three, and a reserve of two refill groups so that a
zero-signal group is resampled rather than silently paid for and skipped.  The
reserve is *declared up front*: the plan's cost covers it whether or not the run
needs it, so the budget cannot be discovered to be short after sampling began.

Baseline, feasibility window, judge thresholds and seed derivation are frozen here
so a reload must reproduce the queue exactly.  No model, container or SDK call
happens in this module.
"""

# ruff: noqa: PERF401 - explicit loops follow the frozen scientific queue order

import hashlib
import random
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from minidsh_rl.learning_contracts import LearningCase
from minidsh_rl.learning_rollout import _case_digest
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
    Family,
    InfeasiblePositiveControlError,
    positive_cases,
)

# One fallback round, not two: the reserve is spent on refills instead of wider
# baseline search, because v4 showed the failures were signal-starved, not unsuited.
#
# WIDENED 2026-11-13, after the first real baseline.  With two rounds, both
# match_rank candidates measured 0/8 under the eval seeds and the frozen 1-4/8
# window rejected the whole epoch *after* 48 rows had been paid for.  The two
# remaining match_rank candidates had never been sampled, so "infeasible" was really
# "unmeasured".  Four rounds cover every candidate.  This is an explicit deviation
# from the frozen protocol, recorded in docs/trainable-agent-rl/positive-control-v2.md;
# the row and cost envelope grows with it (BASELINE_ROWS/TOTAL_ROWS/required_cny are
# derived, never hard-coded).
BASELINE_ROUNDS = 4
GROUP_SIZE = 4
EPOCHS = 4
CHECKPOINTS = (0, 6, 12)
EVAL_SEEDS = 8
SHUFFLE_SEED = 29
TRAIN_SEED_BASE = 972_000
EVAL_SEED_BASE = 982_000
REFILL_SLOTS = 2
REFILL_SEED_BASE = 973_000
BASELINE_MIN = 1
BASELINE_MAX = 4
PASS_MIN = 16
PASS_DELTA = 8
PASS_TASK_MIN = 4
FAIL_MAX = 11
FAIL_MIN_UPDATES = 10
RUN_ID = r"^[a-z][a-z0-9_]{0,63}$"

PLANNED_GROUPS = EPOCHS * len(FAMILIES)
TRAIN_ROWS = (PLANNED_GROUPS + REFILL_SLOTS) * GROUP_SIZE
BASELINE_ROWS = len(FAMILIES) * BASELINE_ROUNDS * EVAL_SEEDS
EVAL_ROWS = (len(CHECKPOINTS) - 1) * len(FAMILIES) * EVAL_SEEDS
TOTAL_ROWS = BASELINE_ROWS + TRAIN_ROWS + EVAL_ROWS
SAMPLED_GROUPS = PLANNED_GROUPS + REFILL_SLOTS


class FrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Positive2Row(FrozenRecord):
    run_id: str = Field(pattern=RUN_ID)
    phase: Literal["baseline", "train", "refill", "eval"]
    family: Family
    case_id: str
    seed: int = Field(ge=0, strict=True)
    repeat: int = Field(ge=0, le=7, strict=True)
    round: int | None = Field(default=None, ge=0, le=BASELINE_ROUNDS - 1, strict=True)
    group: str
    checkpoint: int = Field(ge=0, le=12, strict=True)


class Positive2Selection(FrozenRecord):
    family: Family
    case_id: str
    round: int = Field(ge=0, le=BASELINE_ROUNDS - 1, strict=True)
    successes: int = Field(ge=0, le=8, strict=True)


class Baseline2Plan(FrozenRecord):
    schema_version: Literal[1] = 1
    protocol: Literal["positive_control_v2"] = "positive_control_v2"
    rows: tuple[Positive2Row, ...]
    case_sha256: dict[str, str]


class Positive2Plan(FrozenRecord):
    schema_version: Literal[1] = 1
    protocol: Literal["positive_control_v2"] = "positive_control_v2"
    dataset_version: Literal["curve_dataset_v4"] = "curve_dataset_v4"
    families: tuple[str, ...]
    selected: tuple[str, ...]
    selection: tuple[Positive2Selection, ...]
    baseline_rows: tuple[Positive2Row, ...]
    planned_rows: tuple[Positive2Row, ...]
    refill_rows: tuple[Positive2Row, ...]
    eval_rows: tuple[Positive2Row, ...]
    checkpoints: tuple[int, ...]
    group_size: Literal[4] = 4
    eval_seeds: Literal[8] = 8
    shuffle_seed: Literal[29] = 29
    train_seed_base: Literal[972000] = 972000
    eval_seed_base: Literal[982000] = 982000
    refill_seed_base: Literal[973000] = 973000
    refill_slots: Literal[2] = 2
    baseline_min: Literal[1] = 1
    baseline_max: Literal[4] = 4
    pass_min: Literal[16] = 16
    pass_delta: Literal[8] = 8
    pass_task_min: Literal[4] = 4
    fail_max: Literal[11] = 11
    fail_min_updates: Literal[10] = 10
    required_cny: Decimal
    case_sha256: dict[str, str]
    selection_evidence_sha256: str | None = None


def _family(case: LearningCase) -> Family:
    family = case.family_group
    if family not in FAMILIES:
        raise ValueError(f"unknown family group: {family}")
    return family


def _case_sha256(cases: dict[str, LearningCase]) -> dict[str, str]:
    return {
        case_id: _case_digest(cases[case_id])
        for family in FAMILIES
        for case_id in CANDIDATES[family][:BASELINE_ROUNDS]
    }


def _refill_case(family: Family) -> str:
    """A same-family task for the reserve group, chosen by frozen order.

    With a narrow baseline this is "the first candidate the baseline never sampled".
    Once the baseline covers every candidate that phrase has no referent, so the
    reserve falls back to the family's first candidate: still same-family, still
    declared in advance, still chosen without looking at observed difficulty.
    """
    return CANDIDATES[family][BASELINE_ROUNDS % len(CANDIDATES[family])]


def make_baseline_rows() -> tuple[Positive2Row, ...]:
    cases = positive_cases()
    rows = []
    for family_index, family in enumerate(FAMILIES):
        for round_index, case_id in enumerate(CANDIDATES[family][:BASELINE_ROUNDS]):
            for repeat in range(EVAL_SEEDS):
                rows.append(
                    Positive2Row(
                        run_id=f"pc2_base_{family_index}_{round_index}_{repeat}",
                        phase="baseline",
                        family=family,
                        case_id=case_id,
                        seed=EVAL_SEED_BASE + family_index * 100 + repeat,
                        repeat=repeat,
                        round=round_index,
                        group=f"base_{family_index}_{round_index}",
                        checkpoint=0,
                    )
                )
    if any(_family(cases[row.case_id]) != row.family for row in rows):
        raise ValueError("baseline family binding changed")
    return tuple(rows)


def make_baseline_plan() -> Baseline2Plan:
    cases = positive_cases()
    return Baseline2Plan(rows=make_baseline_rows(), case_sha256=_case_sha256(cases))


def _baseline_counts(
    rows: tuple[Positive2Row, ...], results: Mapping[str, bool]
) -> dict[tuple[str, int], int]:
    counts: dict[tuple[str, int], int] = {}
    for row in rows:
        key = (row.family, row.round or 0)
        counts[key] = counts.get(key, 0) + int(bool(results.get(row.run_id)))
    return counts


def select_tasks(
    baseline_rows: tuple[Positive2Row, ...], results: Mapping[str, bool]
) -> tuple[Positive2Selection, ...]:
    """Apply the frozen fallback order and the 1-4/8 feasibility window."""
    counts = _baseline_counts(baseline_rows, results)
    selected = []
    for family in FAMILIES:
        for round_index, case_id in enumerate(CANDIDATES[family][:BASELINE_ROUNDS]):
            successes = counts.get((family, round_index), 0)
            if BASELINE_MIN <= successes <= BASELINE_MAX:
                selected.append(
                    Positive2Selection(
                        family=family, case_id=case_id, round=round_index, successes=successes
                    )
                )
                break
        else:
            observed = [counts.get((family, index), 0) for index in range(BASELINE_ROUNDS)]
            raise InfeasiblePositiveControlError(
                f"no feasible baseline task for {family}: {observed}"
            )
    return tuple(selected)


def _validate_selection(selection: tuple[Positive2Selection, ...]) -> None:
    if tuple(item.family for item in selection) != FAMILIES:
        raise ValueError("selection must contain one task per family in declared order")
    for item in selection:
        if item.case_id != CANDIDATES[item.family][item.round]:
            raise ValueError("selection does not follow the frozen fallback order")
        if not BASELINE_MIN <= item.successes <= BASELINE_MAX:
            raise ValueError("selected task is outside the feasibility window")


def make_planned_rows(
    selected: tuple[str, ...], cases: dict[str, LearningCase]
) -> tuple[Positive2Row, ...]:
    """Twelve planned groups: four epochs, one task per family each epoch."""
    if len(selected) != len(FAMILIES) or len(set(selected)) != len(FAMILIES):
        raise ValueError("exactly three distinct selected tasks required")
    if tuple(_family(cases[case_id]) for case_id in selected) != FAMILIES:
        raise ValueError("selected tasks must be ordered by family")
    rng = random.Random(SHUFFLE_SEED)
    rows = []
    group = 0
    for _ in range(EPOCHS):
        order = list(selected)
        rng.shuffle(order)
        for case_id in order:
            for repeat in range(GROUP_SIZE):
                rows.append(
                    Positive2Row(
                        run_id=f"pc2_train_{group:02}_{repeat}",
                        phase="train",
                        family=_family(cases[case_id]),
                        case_id=case_id,
                        seed=TRAIN_SEED_BASE + group * 100 + repeat,
                        repeat=repeat,
                        group=f"train_{group:02}",
                        checkpoint=group,
                    )
                )
            group += 1
    return tuple(rows)


def make_refill_rows(cases: dict[str, LearningCase]) -> tuple[Positive2Row, ...]:
    """Two declared reserve groups, one per family slot, priced whether used or not.

    Slot 0 is the match_rank reserve and slot 1 the dependency_batches reserve: the
    two families with the lowest v4 training success among the frozen candidates.
    """
    slots: tuple[Family, ...] = ("match_rank", "dependency_batches")
    rows = []
    for index, family in enumerate(slots):
        case_id = _refill_case(family)
        for repeat in range(GROUP_SIZE):
            rows.append(
                Positive2Row(
                    run_id=f"pc2_refill_{index}_{repeat}",
                    phase="refill",
                    family=family,
                    case_id=case_id,
                    seed=REFILL_SEED_BASE + index * 100 + repeat,
                    repeat=repeat,
                    group=f"refill_{index:02}",
                    checkpoint=0,
                )
            )
    if any(_family(cases[row.case_id]) != row.family for row in rows):
        raise ValueError("refill family binding changed")
    return tuple(rows)


def make_eval_rows(
    selected: tuple[str, ...], cases: dict[str, LearningCase]
) -> tuple[Positive2Row, ...]:
    rows = []
    for checkpoint in CHECKPOINTS[1:]:
        for task_index, case_id in enumerate(selected):
            for repeat in range(EVAL_SEEDS):
                rows.append(
                    Positive2Row(
                        run_id=f"pc2_eval_{checkpoint:02}_{task_index}_{repeat}",
                        phase="eval",
                        family=_family(cases[case_id]),
                        case_id=case_id,
                        seed=EVAL_SEED_BASE + task_index * 100 + repeat,
                        repeat=repeat,
                        group=f"eval_{checkpoint:02}",
                        checkpoint=checkpoint,
                    )
                )
    return tuple(rows)


def required_cny(rows: int, planned_groups: int, sampled_groups: int) -> Decimal:
    """The repository's existing quote: rows, mixed groups, sampled groups, session."""
    return (
        rows * RUN_COST
        + planned_groups * GROUP_TRAIN_COST
        + sampled_groups * GROUP_CLOSE_COST
        + INITIAL_COST
        + EXTRA_COST
    )


def make_plan(
    selection: tuple[Positive2Selection, ...],
    *,
    selection_evidence_sha256: str | None = None,
) -> Positive2Plan:
    _validate_selection(selection)
    cases = positive_cases()
    selected = tuple(item.case_id for item in selection)
    baseline_rows = make_baseline_rows()
    planned_rows = make_planned_rows(selected, cases)
    refill_rows = make_refill_rows(cases)
    eval_rows = make_eval_rows(selected, cases)
    rows = len(baseline_rows) + len(planned_rows) + len(refill_rows) + len(eval_rows)
    plan = Positive2Plan(
        families=FAMILIES,
        selected=selected,
        selection=selection,
        baseline_rows=baseline_rows,
        planned_rows=planned_rows,
        refill_rows=refill_rows,
        eval_rows=eval_rows,
        checkpoints=CHECKPOINTS,
        required_cny=required_cny(rows, PLANNED_GROUPS, SAMPLED_GROUPS),
        case_sha256=_case_sha256(cases),
        selection_evidence_sha256=selection_evidence_sha256,
    )
    verify_plan(plan)
    return plan


def verify_plan(plan: Positive2Plan) -> None:  # noqa: PLR0912 - frozen invariants
    """Recompute every frozen denominator; a reload must reproduce the queue."""
    cases = positive_cases()
    if plan.families != FAMILIES or plan.checkpoints != CHECKPOINTS:
        raise ValueError("plan constants changed")
    _validate_selection(plan.selection)
    if plan.selected != tuple(item.case_id for item in plan.selection):
        raise ValueError("selected tasks do not match selection")
    if len(plan.baseline_rows) != BASELINE_ROWS:
        raise ValueError("baseline denominator changed")
    expected_baseline = {
        case_id for family in FAMILIES for case_id in CANDIDATES[family][:BASELINE_ROUNDS]
    }
    if {row.case_id for row in plan.baseline_rows} != expected_baseline:
        raise ValueError("baseline candidate set changed")
    if len(plan.planned_rows) != PLANNED_GROUPS * GROUP_SIZE:
        raise ValueError("planned training denominator changed")
    if len(plan.refill_rows) != REFILL_SLOTS * GROUP_SIZE:
        raise ValueError("refill reserve changed")
    if len(plan.eval_rows) != EVAL_ROWS:
        raise ValueError("evaluation denominator changed")
    groups = {row.group for row in plan.planned_rows}
    if len(groups) != PLANNED_GROUPS:
        raise ValueError("planned group count changed")
    if any(
        sum(1 for row in plan.planned_rows if row.group == group) != GROUP_SIZE for group in groups
    ):
        raise ValueError("planned group size changed")
    for epoch in range(EPOCHS):
        epoch_rows = [row for row in plan.planned_rows if row.checkpoint // len(FAMILIES) == epoch]
        if {row.case_id for row in epoch_rows} != set(plan.selected):
            raise ValueError("an epoch does not hold each selected task exactly once")
    refill_groups = {row.group for row in plan.refill_rows}
    if len(refill_groups) != REFILL_SLOTS:
        raise ValueError("refill group count changed")
    for row in plan.refill_rows:
        if _family(cases[row.case_id]) != row.family:
            raise ValueError("refill family binding changed")
        if row.case_id != _refill_case(row.family):
            raise ValueError("refill does not follow the frozen candidate order")
    if len(plan.eval_rows) != (len(CHECKPOINTS) - 1) * len(FAMILIES) * EVAL_SEEDS:
        raise ValueError("evaluation seeds changed")
    rows = (
        len(plan.baseline_rows)
        + len(plan.planned_rows)
        + len(plan.refill_rows)
        + len(plan.eval_rows)
    )
    if plan.required_cny != required_cny(rows, PLANNED_GROUPS, SAMPLED_GROUPS):
        raise ValueError("declared cost does not match the frozen queue")
    if plan.case_sha256 != _case_sha256(cases):
        raise ValueError("case digests changed")


def load_baseline_plan(path: Path) -> Baseline2Plan:
    plan = Baseline2Plan.model_validate_json(path.read_bytes())
    if plan != make_baseline_plan():
        raise ValueError("baseline plan changed")
    return plan


def load_plan(path: Path) -> Positive2Plan:
    plan = Positive2Plan.model_validate_json(path.read_bytes())
    if plan != make_plan(plan.selection, selection_evidence_sha256=plan.selection_evidence_sha256):
        raise ValueError("positive-control v2 plan changed")
    return plan


def plan_sha256(plan: Positive2Plan) -> str:
    return hashlib.sha256(plan.model_dump_json().encode()).hexdigest()
