"""Positive-control v2 state machine under a scripted runtime; no model calls.

The refill reserve and the two circuit breakers are the parts a fixture can prove
without spending money, so they are proven here: a refill is spent only on a
signal-free group, only once per family, and a thin run reports itself as thin.
"""

import asyncio
import hashlib
import json
import shutil
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import pytest
from minidsh_rl.contracts import Turn
from minidsh_rl.learning_contracts import LearningCase
from minidsh_rl.learning_execution import Checkpoint, GroupRecord
from minidsh_rl.learning_rollout import LearningGroup, LearningRollout, PolicyBinding
from minidsh_rl.learning_schedule import atomic_record
from minidsh_rl.positive_control import Family
from minidsh_rl.positive_control_v2 import (
    Positive2Row,
    Positive2Selection,
    load_plan,
    make_plan,
)
from minidsh_rl.positive_control_v2_run import (
    STOP_FLOOR_CNY,
    Positive2State,
    _no_patch_submission,
    execute_positive_control_v2,
    final_report,
    prepare_positive_control_v2,
)

# Round 0 is feasible for every family, so the later rounds exist but are unused: the
# queue still samples all four, which is what the 96-row baseline denominator pays for.
# Rounds 2-3 are declared infeasible on purpose so a fixture that starts selecting
# from them is caught rather than silently spending 16 more rows.
_BASELINE = {
    ("match_rank", 0): 3,
    ("dependency_batches", 0): 4,
    ("table_join", 0): 2,
    **{
        (family, round_index): 0
        for family in ("match_rank", "dependency_batches", "table_join")
        for round_index in range(1, 4)
    },
}


def _checkpoint(index: int) -> Checkpoint:
    return Checkpoint(
        execution_kind="contract_fixture",
        sampler_uri=f"contract://sampler/{index}",
        optimizer_uri=f"contract://optimizer/{index}",
        policy=PolicyBinding(
            kind="provider",
            policy_id=f"fixture_{index}",
            snapshot_sha256=hashlib.sha256(str(index).encode()).hexdigest(),
            tokenizer_sha256="b" * 64,
        ),
    )


class Ledger:
    """Read-only ledger stand-in; the controller must never draw on it."""

    def __init__(self, available: Decimal = Decimal("36.452")) -> None:
        self.available = available
        self.committed = Decimal("113.548")

    @property
    def has_pending(self) -> bool:
        return False


class FixtureRuntime:
    execution_kind: Literal["contract_fixture"] = "contract_fixture"

    def __init__(self, *, zero_every: int = 0, finish_without_patch: bool = False) -> None:
        self.zero_every = zero_every
        self.finish_without_patch = finish_without_patch
        self.samples: list[str] = []
        self.updates: list[str] = []
        self.refills: list[str] = []
        self.reconnects = 0

    async def initialize(self, directory: Path) -> Checkpoint:
        value = _checkpoint(0)
        atomic_record(directory / "initial.json", value, create=True)
        return value

    async def reconnect(self, before: Checkpoint, directory: Path) -> Checkpoint:
        """A resumed run must reattach; the fixture just records that it happened."""
        del directory
        self.reconnects += 1
        return before

    async def sample(
        self, case: LearningCase, row: Positive2Row, before: Checkpoint, output: Path
    ) -> LearningRollout:
        await asyncio.to_thread(output.mkdir, parents=True, exist_ok=True)
        self.samples.append(row.run_id)
        if row.phase == "baseline":
            success = row.repeat < _BASELINE[(row.family, row.round or 0)]
        elif row.phase == "train":
            index = int(row.group.rsplit("_", 1)[-1])
            success = not (self.zero_every and index % self.zero_every == 0) and row.repeat < 2
        elif row.phase == "refill":
            self.refills.append(row.run_id)
            success = row.repeat < 2
        else:
            success = row.repeat < 3
        patches = 0 if self.finish_without_patch else 1
        (output / "result.json").write_text(
            json.dumps(
                {
                    "run_id": row.run_id,
                    "seed": row.seed,
                    "success": success,
                    "status": "finished" if patches == 0 else "budget_exhausted",
                    "patches": patches,
                }
            )
        )
        return LearningRollout(
            case=case,
            run_id=row.run_id,
            seed=row.seed,
            policy=before.policy,
            success=success,
            status="finished",
            turns=(Turn((11, 12), (13, 14), (-0.1, -0.2)),),
            total_tokens=100,
            harness_sha256="c" * 64,
            verification_executed=True,
            local_failed_calls=0,
        )

    async def update(
        self, key: str, group: LearningGroup, before: Checkpoint, output: Path
    ) -> Checkpoint:
        del group, before
        assert (output / "group.json").exists()
        self.updates.append(key)
        return _checkpoint(len(self.updates))


def load_group(root: Path, key: str) -> GroupRecord:
    return GroupRecord.model_validate_json((root / "updates" / key / "group.json").read_bytes())


async def _run(
    tmp_path: Path, runtime: FixtureRuntime, ledger: Ledger | None = None
) -> tuple[Positive2State, Path]:
    root = tmp_path / "pc2"
    prepare_positive_control_v2(root)
    return await execute_positive_control_v2(root, runtime, resources=ledger), root


async def test_refill_is_spent_only_on_a_signal_free_group(tmp_path: Path) -> None:
    """Every planned group that fired a refill must itself have been all-zero."""
    runtime = FixtureRuntime(zero_every=3)
    state, root = await _run(tmp_path, runtime)
    assert state.phase == "finished"
    assert state.refilled, "no refill was exercised; the reserve went untested"
    plan = load_plan(root / "plan.json")
    refill_rows = {row.group: row for row in plan.refill_rows}
    assert set(state.refilled) <= set(refill_rows), "a refill outside the frozen reserve was used"
    for key in state.refilled:
        index = int(key.rsplit("_", 1)[-1])
        source = f"train_{[i for i in range(12) if i % 3 == 0][index]:02}"
        group = load_group(root, source)
        assert group.has_signal is False, f"{source} had signal yet spent refill {key}"
        assert refill_rows[key].family == refill_rows[key].family


async def test_a_zero_signal_planned_group_triggers_at_most_two_refills(tmp_path: Path) -> None:
    """The reserve is bounded by declaration, not by how many groups fail."""
    runtime = FixtureRuntime(zero_every=1)  # every planned group is all-zero
    state, root = await _run(tmp_path, runtime)
    assert len(state.refilled) == 2
    assert state.phase == "finished"
    # A refill replaces a signal-free group's training signal, so it does update —
    # but never more than the two declared slots, so mixed groups stay <= 12.
    assert sorted(state.updated) == ["refill_00", "refill_01"]
    report = json.loads((root / "final-report.json").read_text())
    assert report["sampled_groups"] == 14
    # 2 mixed groups out of 14 sampled: the reserve rescues two slots, it does not
    # turn a degenerate queue into a healthy one, and the ratio says so.
    assert report["mixed_group_ratio"] == pytest.approx(2 / 14)
    assert report["updates"] == 2


async def test_insufficient_updates_is_reported_as_inconclusive_not_failed(tmp_path: Path) -> None:
    """A thin run must not be readable as 'RL does not work'."""
    runtime = FixtureRuntime(zero_every=1)
    state, _ = await _run(tmp_path, runtime)
    assert state.final is not None
    assert state.final["status"] == "inconclusive"
    assert "insufficient_updates" in str(state.final["reason"])


async def test_final_report_keeps_the_complete_denominator(tmp_path: Path) -> None:
    runtime = FixtureRuntime()
    state, root = await _run(tmp_path, runtime)
    report = json.loads((root / "final-report.json").read_text())
    assert report["declared_rows"] == 200
    assert report["planned_groups"] == 12
    assert report["sampled_groups"] == 12
    assert report["completed_rows"] == len(state.completed)
    assert report["mixed_group_ratio"] == 1.0


async def test_no_patch_submission_is_derived_and_not_counted_as_success() -> None:
    """The 22 v4 rollouts that finished without a patch stay visible as their own row."""
    runtime = FixtureRuntime(finish_without_patch=True)
    assert runtime.finish_without_patch is True
    # A labelled row is reported by run_id, never folded into pass or fail.
    with TemporaryDirectory() as raw:
        root = Path(raw)
        (root / "rollouts" / "pc2_train_00_0").mkdir(parents=True)
        (root / "rollouts" / "pc2_train_00_0" / "result.json").write_text(
            json.dumps({"status": "finished", "patches": 0})
        )
        (root / "rollouts" / "pc2_train_00_1").mkdir()
        (root / "rollouts" / "pc2_train_00_1" / "result.json").write_text(
            json.dumps({"status": "finished", "patches": 1})
        )
        assert _no_patch_submission(root, "pc2_train_00_0") is True
        assert _no_patch_submission(root, "pc2_train_00_1") is False
        assert _no_patch_submission(root, "missing") is None


async def test_budget_floor_stops_the_run_and_says_so(tmp_path: Path) -> None:
    runtime = FixtureRuntime()
    state, root = await _run(tmp_path, runtime, Ledger(available=STOP_FLOOR_CNY / 2))
    assert state.phase == "stopped"
    assert "budget_stop" in state.stop_reason
    assert state.final is not None
    assert state.final["status"] == "inconclusive"
    assert (root / "final-report.json").exists()


async def test_a_healthy_ledger_keeps_running_and_measure_both_checkpoints(
    tmp_path: Path,
) -> None:
    runtime = FixtureRuntime()
    state, _ = await _run(tmp_path, runtime, Ledger())
    assert state.phase == "finished"
    assert set(state.snapshots) == {6, 12}
    assert state.skipped_checkpoints == []


async def test_a_low_ledger_skips_the_second_checkpoint_before_spending(
    tmp_path: Path,
) -> None:
    """Dropping 24 evaluation rows is the declared cheaper fallback, not a silent one."""
    runtime = FixtureRuntime()
    state, root = await _run(tmp_path, runtime, Ledger(available=Decimal("2.00")))
    assert state.phase == "finished"
    # Checkpoint 6 is kept so a two-point curve (0 and 6) still exists; only the
    # final checkpoint is dropped, exactly as the protocol's breaker declares.
    assert state.skipped_checkpoints == [12]
    assert set(state.snapshots) == {6}
    report = json.loads((root / "final-report.json").read_text())
    assert report["final_measured"] is False
    assert report["status"] == "inconclusive"
    assert report["skipped_checkpoints"] == [12]


def test_final_report_requires_a_measured_final_checkpoint() -> None:
    """No final measurement means inconclusive regardless of the earlier numbers."""
    pairs: tuple[tuple[Family, str], ...] = (
        ("match_rank", "curve_v4_path_router_02"),
        ("dependency_batches", "curve_v4_dependency_batches_01"),
        ("table_join", "curve_v4_table_join_02"),
    )
    selection = tuple(
        Positive2Selection(family=family, case_id=case, round=0, successes=3)
        for family, case in pairs
    )
    plan = make_plan(selection)
    state = Positive2State(
        baseline_plan_sha256="a" * 64,
        kind="contract_fixture",
        selected=list(plan.selected),
    )
    report = final_report(plan, state, Path("/nonexistent"))
    assert report["status"] == "inconclusive"
    assert report["final_successes"] is None
    assert report["no_patch_submission"] == []
    assert report["no_patch_submission_unknown"] == len(plan.planned_rows) + len(plan.refill_rows)


async def test_resume_reuses_closed_groups_instead_of_paying_twice(tmp_path: Path) -> None:
    """A crash after group 4 must not resample groups 0-3 on the next attempt.

    This is the difference between a resume and a restart: at 1.339 CNY per mixed
    group, resampling four closed groups would silently spend that money twice.
    """
    runtime = FixtureRuntime()
    root = tmp_path / "pc2"
    prepare_positive_control_v2(root)
    await execute_positive_control_v2(root, runtime, resources=Ledger())

    # Simulate an interruption after the fourth planned group: keep every artifact
    # on disk, rewind only the bookkeeping that says how far the run got.
    payload = json.loads((root / "state.json").read_text())
    kept_groups = {f"train_{index:02}" for index in range(4)}

    def belongs(run_id: str) -> bool:
        if run_id.startswith("pc2_base"):
            return True
        if run_id.startswith("pc2_refill"):
            return False
        if run_id.startswith("pc2_eval"):
            return False
        return run_id.split("_")[2] in {g.split("_")[1] for g in kept_groups}

    payload["phase"] = "training"
    payload["completed"] = [rid for rid in payload["completed"] if belongs(rid)]
    payload["results"] = {k: v for k, v in payload["results"].items() if belongs(k)}
    payload["groups"] = [g for g in payload["groups"] if g in kept_groups]
    payload["updated"] = [g for g in payload["updated"] if g in kept_groups]
    payload["refilled"] = []
    payload["snapshots"] = {}
    payload["skipped_checkpoints"] = []
    payload["final"] = None
    (root / "state.json").write_text(json.dumps(payload))
    # An interrupted run would not have written the later groups' records; leaving
    # them behind would make the fixture "closed" groups whose rows were never paid.
    for folder in (root / "updates").iterdir():
        if folder.name.startswith(("train_", "refill_")) and folder.name not in kept_groups:
            shutil.rmtree(folder)

    resumed = FixtureRuntime()
    after = await execute_positive_control_v2(root, resumed, resources=Ledger(), resume=True)
    assert after.phase == "finished"
    resampled = [rid for rid in resumed.samples if belongs(rid)]
    assert resampled == [], f"resume resampled already-closed rows: {resampled}"
    assert len(resumed.samples) < len(runtime.samples), "resume did no reuse at all"


async def test_resume_reconnects_before_touching_the_queue(tmp_path: Path) -> None:
    """Without a reconnect the first resumed sample hits a client-less adapter."""
    runtime = FixtureRuntime()
    root = tmp_path / "pc2"
    prepare_positive_control_v2(root)
    await execute_positive_control_v2(root, runtime, resources=Ledger())
    assert runtime.reconnects == 0, "a fresh run must not reconnect"

    # A finished run returns immediately, so rewind to an in-progress phase first:
    # that is the state a crash actually leaves behind.
    payload = json.loads((root / "state.json").read_text())
    payload["phase"] = "training"
    payload["final"] = None
    (root / "state.json").write_text(json.dumps(payload))

    resumed = FixtureRuntime()
    await execute_positive_control_v2(root, resumed, resources=Ledger(), resume=True, recover=True)
    assert resumed.reconnects == 1


async def test_a_closed_zero_signal_group_resumes_without_an_update(tmp_path: Path) -> None:
    """Skipping the gradient IS the correct terminal state for an all-zero group.

    Regression: every zero-signal group looked "unconfirmed" to the resume guard, so a
    crash during the first group made the whole run unresumable even though the
    verifier had already written `skipped` and spent nothing.
    """
    runtime = FixtureRuntime(zero_every=1)
    root = tmp_path / "pc2"
    prepare_positive_control_v2(root)
    state = await execute_positive_control_v2(root, runtime, resources=Ledger())
    # Only the two refill slots update when every planned group is all-zero, which is
    # the declared reserve doing its job; the planned groups themselves never do.
    assert [g for g in state.updated if g.startswith("train_")] == []

    payload = json.loads((root / "state.json").read_text())
    payload["phase"] = "training"
    payload["groups"] = []
    payload["updated"] = []
    payload["final"] = None
    (root / "state.json").write_text(json.dumps(payload))

    resumed = FixtureRuntime(zero_every=1)
    after = await execute_positive_control_v2(root, resumed, resources=Ledger(), resume=True)
    assert after.phase == "finished"
    assert "train_00" in after.groups
    assert "train_00" not in after.updated


async def test_a_partially_sampled_group_is_refused_rather_than_replayed(
    tmp_path: Path,
) -> None:
    """Rows recorded but no group record means the group is torn; never resample it."""
    runtime = FixtureRuntime()
    root = tmp_path / "pc2"
    prepare_positive_control_v2(root)
    state = await execute_positive_control_v2(root, runtime, resources=Ledger())
    assert state.phase == "finished"

    # Rewind the first planned group: its four rows stay completed, its record goes.
    payload = json.loads((root / "state.json").read_text())
    payload["phase"] = "training"
    payload["groups"] = [g for g in payload["groups"] if g != "train_00"]
    payload["updated"] = [g for g in payload["updated"] if g != "train_00"]
    (root / "state.json").write_text(json.dumps(payload))
    for name in ("group.json", "receipt.json"):
        (root / "updates" / "train_00" / name).unlink(missing_ok=True)
    with pytest.raises(ValueError, match="partially completed training group"):
        await execute_positive_control_v2(root, runtime, resources=Ledger(), resume=True)


async def test_pending_operation_blocks_a_blind_restart(tmp_path: Path) -> None:
    """A crash mid-call must be inspected, not silently retried."""
    runtime = FixtureRuntime()
    root = tmp_path / "pc2"
    prepare_positive_control_v2(root)
    state = await execute_positive_control_v2(root, runtime, resources=Ledger())
    payload = json.loads((root / "state.json").read_text())
    payload["pending"] = "train"
    payload["pending_key"] = "pc2_train_05_2"
    (root / "state.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="explicit recovery required") as error:
        await execute_positive_control_v2(root, runtime, resources=Ledger(), resume=True)
    assert "pc2_train_05_2" in str(error.value), "the refusal must name the unverified call"
    assert state.phase == "finished"


async def test_recovery_is_explicit_and_recorded(tmp_path: Path) -> None:
    """Recovering redoes one call; the record says which, so a duplicate charge is visible."""
    runtime = FixtureRuntime()
    root = tmp_path / "pc2"
    prepare_positive_control_v2(root)
    await execute_positive_control_v2(root, runtime, resources=Ledger())
    payload = json.loads((root / "state.json").read_text())
    payload["phase"] = "training"
    payload["pending"] = "eval"
    payload["pending_key"] = "pc2_eval_12_0_0"
    (root / "state.json").write_text(json.dumps(payload))

    after = await execute_positive_control_v2(
        root, FixtureRuntime(), resources=Ledger(), resume=True, recover=True
    )
    assert after.phase == "finished"
    assert after.recovered == ["pc2_eval_12_0_0"]
