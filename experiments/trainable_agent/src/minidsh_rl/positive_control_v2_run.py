"""Positive-control v2 execution controller: twelve planned groups, two refills.

The v1 controller (`positive_control_run.py`) is left frozen so its 18-group queue
stays replayable.  This module adds the three things the v2 protocol declares:

1. a reserve of two refill groups, spent only when a planned group carries no
   reward signal — DAPO's Dynamic Sampling idea, bounded at two refills because the
   filtered groups still cost sampling time and money;
2. two pre-declared budget circuit breakers, checked against a live ledger rather
   than discovered after the fact;
3. explicit `insufficient_updates` and `budget_stop` terminal states, so a thin run
   reports itself as thin instead of being read as a negative result.

No SDK client is created here.  A provider adapter must be injected; it owns
financial leasing and every RPC.
"""

import json
from decimal import Decimal
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from minidsh_rl.learning_contracts import LearningCase
from minidsh_rl.learning_execution import Checkpoint, GroupRecord, group_datum_hash
from minidsh_rl.learning_rollout import LearningGroup, LearningRollout, assemble_group
from minidsh_rl.learning_schedule import ExecutionKind, atomic_record, digest
from minidsh_rl.positive_control import positive_cases
from minidsh_rl.positive_control_v2 import (
    CHECKPOINTS,
    FAIL_MAX,
    FAIL_MIN_UPDATES,
    PASS_DELTA,
    PASS_MIN,
    PASS_TASK_MIN,
    PLANNED_GROUPS,
    REFILL_SLOTS,
    Baseline2Plan,
    Positive2Plan,
    Positive2Row,
    load_baseline_plan,
    load_plan,
    make_baseline_plan,
    make_plan,
    select_tasks,
    verify_plan,
)

# Below this the second checkpoint's 24 evaluation rows are dropped rather than
# risking an unbudgeted call; below the floor the run stops and says why.
CHECKPOINT_RESERVE_CNY = Decimal("2.50")
STOP_FLOOR_CNY = Decimal("1.00")
MIN_UPDATES = FAIL_MIN_UPDATES


class ResourcePort(Protocol):
    """Only the read side of a ledger; the controller never draws on it."""

    @property
    def available(self) -> Decimal: ...
    @property
    def committed(self) -> Decimal: ...
    @property
    def has_pending(self) -> bool: ...


class Positive2Runtime(Protocol):
    @property
    def execution_kind(self) -> ExecutionKind: ...

    async def initialize(self, directory: Path) -> Checkpoint: ...

    async def reconnect(self, before: Checkpoint, directory: Path) -> Checkpoint: ...

    async def sample(
        self,
        case: LearningCase,
        row: Positive2Row,
        before: Checkpoint,
        output: Path,
    ) -> LearningRollout: ...

    async def update(
        self,
        key: str,
        group: LearningGroup,
        before: Checkpoint,
        output: Path,
    ) -> Checkpoint: ...


class Positive2State(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    protocol: Literal["positive_control_v2"] = "positive_control_v2"
    baseline_plan_sha256: str
    phase: Literal["prepared", "baseline", "selected", "training", "finished", "stopped"] = (
        "prepared"
    )
    kind: ExecutionKind
    initial: Checkpoint | None = None
    current: Checkpoint | None = None
    pending: Literal["baseline", "train", "refill", "eval", "update"] | None = None
    pending_key: str = ""
    results: dict[str, bool] = Field(default_factory=dict)
    completed: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)
    refilled: list[str] = Field(default_factory=list)
    recovered: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)
    snapshots: dict[int, Checkpoint] = Field(default_factory=dict)
    skipped_checkpoints: list[int] = Field(default_factory=list)
    selection_evidence_sha256: str | None = None
    plan_sha256: str | None = None
    selected: list[str] = Field(default_factory=list)
    stop_reason: str = ""
    final: dict[str, object] | None = None


def prepare_positive_control_v2(root: Path) -> Baseline2Plan:
    """Freeze the baseline queue; no runtime, client or spend is created."""
    if root.exists():
        raise ValueError("positive-control v2 directory already exists")
    root.mkdir(parents=True)
    plan = make_baseline_plan()
    atomic_record(root / "baseline-plan.json", plan, create=True)
    return plan


def _save(root: Path, state: Positive2State) -> None:
    atomic_record(root / "state.json", state)


def _recover(state: Positive2State, *, recover: bool) -> Positive2State:
    """Drop an in-flight marker only when the caller says so explicitly.

    A crash mid-call leaves `pending` set, and the artifact on disk may or may not
    exist. Rewinding the marker makes the next attempt redo that one operation,
    which is either free (nothing was written) or a duplicated charge (it was).
    That trade is the operator's to make, so it needs `recover=True`; the default
    refuses, and the refusal names the operation that must be inspected first.
    """
    if state.pending is None:
        return state
    if not recover:
        raise ValueError(
            f"in-flight operation; explicit recovery required (pending {state.pending} "
            f"{state.pending_key})"
        )
    recovered = state.model_copy(
        update={
            "pending": None,
            "pending_key": "",
            "recovered": [*state.recovered, state.pending_key],
        }
    )
    return recovered


def _load_state(root: Path, kind: ExecutionKind, *, recover: bool) -> Positive2State:
    path = root / "state.json"
    state = (
        Positive2State.model_validate_json(path.read_bytes())
        if path.exists()
        else Positive2State(baseline_plan_sha256=digest(root / "baseline-plan.json"), kind=kind)
    )
    if state.baseline_plan_sha256 != digest(root / "baseline-plan.json"):
        raise ValueError("baseline plan changed")
    if state.kind != kind:
        raise ValueError("runtime provenance changed")
    state = _recover(state, recover=recover)
    plan_path = root / "plan.json"
    if plan_path.exists():
        plan = load_plan(plan_path)
        if state.plan_sha256 != digest(plan_path) or tuple(state.selected) != plan.selected:
            raise ValueError("frozen plan changed")
    return state


def _check_row(row: Positive2Row, value: LearningRollout, before: Checkpoint) -> None:
    if (
        value.run_id != row.run_id
        or value.seed != row.seed
        or value.case.task.case_id != row.case_id
        or value.policy != before.policy
        or value.budget_violations
        or value.status == "cancelled"
    ):
        raise ValueError("trajectory identity/budget/cancellation failure")


async def _sample_rows(  # noqa: PLR0913, PLR0917 - explicit execution boundary
    root: Path,
    state: Positive2State,
    runtime: Positive2Runtime,
    rows: tuple[Positive2Row, ...],
    before: Checkpoint,
    phase: Literal["baseline", "train", "refill", "eval"],
) -> tuple[LearningRollout, ...]:
    cases = positive_cases()
    values = []
    for row in rows:
        if row.run_id in state.completed:
            continue
        state.pending, state.pending_key = phase, row.run_id
        _save(root, state)
        value = await runtime.sample(
            cases[row.case_id], row, before, root / "rollouts" / row.run_id
        )
        _check_row(row, value, before)
        state.results[row.run_id] = value.success
        state.completed.append(row.run_id)
        state.pending, state.pending_key = None, ""
        _save(root, state)
        values.append(value)
    return tuple(values)


def _group_record(before: Checkpoint, group: LearningGroup) -> GroupRecord:
    return GroupRecord(
        before=before,
        run_ids=group.run_ids,
        datum_sha256=group_datum_hash(group),
        has_signal=group.batch is not None and group.batch.has_signal,
        input_sha256={},
    )


async def _train_one(  # noqa: PLR0913, PLR0917 - explicit execution boundary
    root: Path,
    state: Positive2State,
    runtime: Positive2Runtime,
    rows: tuple[Positive2Row, ...],
    key: str,
    phase: Literal["train", "refill"],
) -> bool:
    """Sample one group and update if it carries signal; returns whether it had signal.

    A group whose rows are all recorded *and* whose record is written is already
    closed, so a resumed run reuses its verdict instead of resampling it — that is
    the only way a resume after an interruption avoids paying twice. A group with
    rows but no record is torn, and is refused rather than replayed: resampling it
    would replace the trajectories that were already paid for.
    """
    folder = root / "updates" / key
    closed = folder / "group.json"
    done = [row.run_id for row in rows if row.run_id in state.completed]
    if closed.exists():
        if len(done) != len(rows):
            raise ValueError("closed group is missing its trajectories")
        record = GroupRecord.model_validate_json(closed.read_bytes())
        if record.run_ids != tuple(row.run_id for row in rows):
            raise ValueError("closed group names other trajectories")
        if key in state.groups:
            return record.has_signal
        if not record.has_signal:
            # A zero-signal group never takes a gradient, so "no confirmed update" is
            # its correct terminal state, not a torn one. Recording it here is what
            # lets a resume continue instead of refusing a group that behaved exactly
            # as designed (the verifier wrote `skipped` and spent nothing).
            state.groups.append(key)
            state.pending, state.pending_key = None, ""
            _save(root, state)
            return False
        # Signal-bearing but unconfirmed: the optimiser may or may not have run, and
        # a resume cannot tell. Refusing is the only honest answer.
        raise ValueError("closed group has no confirmed update; explicit recovery required")
    if done:
        raise ValueError("partially completed training group; do not replay")
    if state.current is None:
        raise ValueError("current checkpoint missing")
    before = state.current
    values = await _sample_rows(root, state, runtime, rows, before, phase)
    if len(values) != len(rows):
        raise ValueError("training group not fully sampled")
    group = assemble_group(values)
    folder.mkdir(parents=True, exist_ok=True)
    record = _group_record(before, group)
    atomic_record(closed, record, create=True)
    state.pending, state.pending_key = "update", key
    _save(root, state)
    after = await runtime.update(key, group, before, folder)
    state.current = after
    state.groups.append(key)
    state.pending, state.pending_key = None, ""
    _save(root, state)
    if record.has_signal:
        state.updated.append(key)
    return record.has_signal


def _refill_for(family: str, used: list[str]) -> str | None:
    """One reserve slot per refillable family, spent at most once.

    The slot is chosen by family, not by which group failed, so a refill cannot be
    quietly selected for the task that happens to look easiest.
    """
    for index in range(REFILL_SLOTS):
        key = f"refill_{index:02}"
        if key in used:
            continue
        if family == ("match_rank", "dependency_batches")[index]:
            return key
    return None


def _should_measure(checkpoint: int, resources: ResourcePort | None) -> bool:
    """Apply the declared breaker: above the reserve everything runs.

    Below it the *later* new checkpoint is dropped so 0 and 6 still form a curve.
    Dropping 6 as well would leave one measured point, which cannot show a trend
    and would spend 2.16 to prove nothing; so the first new checkpoint is kept and
    at most one extra 2.16 is risked.  Both outcomes are recorded in
    `skipped_checkpoints`, never inferred from a missing file.
    """
    if resources is None or resources.available >= CHECKPOINT_RESERVE_CNY:
        return True
    return checkpoint != CHECKPOINTS[-1]


async def _training(
    root: Path,
    state: Positive2State,
    runtime: Positive2Runtime,
    plan: Positive2Plan,
    resources: ResourcePort | None,
) -> None:
    state.phase = "training"
    _save(root, state)
    planned = {row.group for row in plan.planned_rows}
    for group_index in range(PLANNED_GROUPS):
        key = f"train_{group_index:02}"
        if resources is not None and resources.available < STOP_FLOOR_CNY:
            state.phase = "stopped"
            state.stop_reason = f"budget_stop: {resources.available} CNY below floor"
            state.final = final_report(plan, state, root, reason=state.stop_reason)
            (root / "final-report.json").write_text(json.dumps(state.final, indent=2))
            _save(root, state)
            return
        if key not in planned:
            raise ValueError("planned group missing from the frozen queue")
        rows = tuple(row for row in plan.planned_rows if row.group == key)
        signal = await _train_one(root, state, runtime, rows, key, "train")
        if not signal:
            slot = _refill_for(rows[0].family, state.refilled)
            if slot is not None:
                state.refilled.append(slot)
                await _train_one(
                    root,
                    state,
                    runtime,
                    tuple(row for row in plan.refill_rows if row.group == slot),
                    slot,
                    "refill",
                )
        completed = group_index + 1
        if completed in CHECKPOINTS[1:] and not _should_measure(completed, resources):
            state.skipped_checkpoints.append(completed)
            _save(root, state)
            continue
        if completed in CHECKPOINTS[1:]:
            if state.current is None:
                raise ValueError("checkpoint missing for evaluation")
            state.snapshots[completed] = state.current
            _save(root, state)
            await _sample_rows(
                root,
                state,
                runtime,
                tuple(row for row in plan.eval_rows if row.checkpoint == completed),
                state.snapshots[completed],
                "eval",
            )
    state.phase = "finished"
    state.final = final_report(plan, state, root)
    (root / "final-report.json").write_text(json.dumps(state.final, indent=2))
    _save(root, state)


def _no_patch_submission(root: Path, run_id: str) -> bool | None:
    """Derived label: finished without ever submitting a patch.

    v4 had 22 such rollouts.  They are neither successes nor budget failures, and
    folding them into either would hide the behaviour the next stage is meant to
    target.  Absent artifacts yield None rather than a guess.
    """
    path = root / "rollouts" / run_id / "result.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get("status") != "finished":
        return False
    patches = record.get("patches")
    if type(patches) is not int:
        return None
    return patches == 0


def final_report(
    plan: Positive2Plan, state: Positive2State, root: Path, *, reason: str = ""
) -> dict[str, object]:
    baseline_rows = tuple(row for row in plan.baseline_rows if row.case_id in plan.selected)
    final_rows = tuple(row for row in plan.eval_rows if row.checkpoint == CHECKPOINTS[-1])
    baseline_measured = all(row.run_id in state.results for row in baseline_rows)
    final_measured = all(row.run_id in state.results for row in final_rows)
    baseline = sum(state.results.get(row.run_id, False) for row in baseline_rows)
    final = sum(state.results.get(row.run_id, False) for row in final_rows)
    per_task = {
        case_id: {
            "baseline": sum(
                state.results.get(row.run_id, False)
                for row in baseline_rows
                if row.case_id == case_id
            ),
            "final": sum(
                state.results.get(row.run_id, False) for row in final_rows if row.case_id == case_id
            ),
        }
        for case_id in plan.selected
    }
    updates = len(state.updated)
    samples = len(state.groups)
    if not final_measured:
        status = "inconclusive"
        reason = reason or "final checkpoint not fully measured"
    elif (
        final >= PASS_MIN
        and final - baseline >= PASS_DELTA
        and all(item["final"] >= PASS_TASK_MIN for item in per_task.values())
    ):
        status = "pass"
        reason = reason or "criteria met"
    elif final <= FAIL_MAX and updates >= FAIL_MIN_UPDATES:
        status = "fail"
        reason = reason or "final below threshold with enough updates"
    else:
        status = "inconclusive"
        reason = reason or "between pass and fail thresholds"
    if status != "pass" and updates < MIN_UPDATES and "budget_stop" not in reason:
        status = "inconclusive"
        reason = f"insufficient_updates: {updates} of {MIN_UPDATES} required"

    training_rows = (*plan.planned_rows, *plan.refill_rows)
    patched_free = [
        row.run_id for row in training_rows if _no_patch_submission(root, row.run_id) is True
    ]
    unknown = [
        row.run_id for row in training_rows if _no_patch_submission(root, row.run_id) is None
    ]
    return {
        "schema_version": 1,
        "protocol": "positive_control_v2",
        "status": status,
        "reason": reason,
        "stop_reason": state.stop_reason,
        "baseline_measured": baseline_measured,
        "final_measured": final_measured,
        "baseline_successes": baseline if baseline_measured else None,
        "final_successes": final if final_measured else None,
        "delta": final - baseline if baseline_measured and final_measured else None,
        "per_task": per_task,
        "updates": updates,
        "sampled_groups": samples,
        "planned_groups": PLANNED_GROUPS,
        "refill_groups_used": len(state.refilled),
        "mixed_group_ratio": (updates / samples) if samples else None,
        "zero_signal_groups": samples - updates,
        "skipped_checkpoints": list(state.skipped_checkpoints),
        "completed_rows": len(state.completed),
        "declared_rows": (
            len(plan.baseline_rows)
            + len(plan.planned_rows)
            + len(plan.refill_rows)
            + len(plan.eval_rows)
        ),
        "no_patch_submission": patched_free,
        "no_patch_submission_unknown": len(unknown),
        "criterion": {
            "pass_min": PASS_MIN,
            "pass_delta": PASS_DELTA,
            "pass_task_min": PASS_TASK_MIN,
            "fail_max": FAIL_MAX,
            "fail_min_updates": FAIL_MIN_UPDATES,
        },
    }


async def _baseline(
    root: Path,
    state: Positive2State,
    runtime: Positive2Runtime,
    plan: Baseline2Plan,
) -> Positive2Plan:
    if state.initial is None:
        raise ValueError("initial checkpoint missing")
    state.phase = "baseline"
    _save(root, state)
    await _sample_rows(root, state, runtime, plan.rows, state.initial, "baseline")
    results_path = root / "baseline-results.json"
    results = {row.run_id: state.results[row.run_id] for row in plan.rows}
    if results_path.exists():
        if json.loads(results_path.read_text()) != results:
            raise ValueError("baseline results changed")
    else:
        results_path.write_text(json.dumps(results, indent=2))
    evidence = digest(results_path)
    selection = select_tasks(plan.rows, results)
    frozen = make_plan(selection, selection_evidence_sha256=evidence)
    plan_path = root / "plan.json"
    if plan_path.exists():
        if load_plan(plan_path) != frozen:
            raise ValueError("frozen positive-control v2 plan changed")
    else:
        atomic_record(plan_path, frozen, create=True)
    state.selection_evidence_sha256 = evidence
    state.plan_sha256 = digest(plan_path)
    state.selected = list(frozen.selected)
    state.phase = "selected"
    _save(root, state)
    return frozen


async def execute_positive_control_v2(
    root: Path,
    runtime: Positive2Runtime,
    *,
    resources: ResourcePort | None = None,
    resume: bool = False,
    recover: bool = False,
) -> Positive2State:
    """Baseline, freeze the selected plan, then train and evaluate within budget."""
    baseline_plan = load_baseline_plan(root / "baseline-plan.json")
    existed = (root / "state.json").exists()
    state = _load_state(root, runtime.execution_kind, recover=recover)
    if state.phase in {"finished", "stopped"}:
        return state
    if existed and not resume:
        raise ValueError("explicit resume required")
    resume_required = existed
    if state.initial is None:
        if (root / "initial.json").exists():
            raise ValueError("initial checkpoint already exists")
        state.initial = state.current = await runtime.initialize(root)
        recorded = Checkpoint.model_validate_json((root / "initial.json").read_bytes())
        if recorded != state.initial:
            raise ValueError("initial receipt mismatch")
        _save(root, state)
    elif not resume:
        raise ValueError("explicit resume required")
    if resume and state.current is not None and resume_required:
        # A fresh runtime has no client; reconnect to the checkpoint the state names
        # before touching the queue, otherwise the next sample fails on `client is None`.
        state.current = await runtime.reconnect(state.current, root)
        _save(root, state)
    plan = await _baseline(root, state, runtime, baseline_plan)
    verify_plan(plan)
    await _training(root, state, runtime, plan, resources)
    return state
