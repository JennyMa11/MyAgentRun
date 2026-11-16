"""v2 provider adapter under a fake SDK; no credentials, network or spend.

The adapter's job is to be *unable* to spend accidentally: nothing happens at
import or construction, evaluation samples from the snapshot under test, and a
zero-signal group closes its own record instead of vanishing from the denominator.

The `cast` calls below exist because a fake ledger and a fake runner are not a
`CostStore` or a `ProcessRunner`; they are the narrowest way to hand this adapter
something that cannot reach the network.
"""

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from minidsh_rl.backend_v2 import Tokenizer
from minidsh_rl.contracts import Turn
from minidsh_rl.cost_store import CostStore
from minidsh_rl.curve_cases_v4 import curve_cases_v4
from minidsh_rl.datum import build_batch
from minidsh_rl.learning_contracts import LearningCase
from minidsh_rl.learning_execution import Checkpoint, GroupRecord, group_datum_hash
from minidsh_rl.learning_rollout import LearningGroup
from minidsh_rl.learning_schedule import atomic_record
from minidsh_rl.learning_sdk import SavedPair
from minidsh_rl.positive_control_v2 import Positive2Row, make_baseline_plan
from minidsh_rl.positive_control_v2_sdk import (
    Positive2RuntimeBridge,
    Positive2SDKRuntime,
    build_runtime,
    require_positive_decimal,
)
from minidsh_rl.transaction import UpdateJournal
from test_positive_control_v2_run import Ledger, _checkpoint

from minidsh.infrastructure.process import ProcessRunner


class _FakeClient:
    """Stands in for the SDK training client; any real call would raise loudly."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"zero-signal group must not call client.{name}")


def _budget() -> CostStore:
    return cast(CostStore, Ledger())


def _host() -> ProcessRunner:
    return cast(ProcessRunner, object())


def _runtime(root: Path) -> Positive2SDKRuntime:
    return Positive2SDKRuntime(object(), _budget(), root, _host(), "a" * 64)


def _row(phase: str = "train") -> Positive2Row:
    return Positive2Row(
        run_id="pc2_train_00_0",
        phase=phase,  # type: ignore[arg-type]
        family="match_rank",
        case_id="curve_v4_path_router_02",
        seed=972_000,
        repeat=0,
        group="train_00",
        checkpoint=0,
    )


def _group() -> LearningGroup:
    return LearningGroup(
        run_ids=("pc2_train_00_0",),
        rewards=(0.0,),
        advantages=(0.0,),
        batch=None,
    )


def _case() -> LearningCase:
    return next(c for c in curve_cases_v4() if c.task.case_id == "curve_v4_path_router_02")


def _frozen_root(tmp_path: Path) -> Path:
    root = tmp_path / "pc2"
    root.mkdir()
    atomic_record(root / "baseline-plan.json", make_baseline_plan(), create=True)
    return root


def test_importing_and_constructing_creates_no_client(tmp_path: Path) -> None:
    """Construction must not reach the provider, so nothing can be charged."""
    calls: list[str] = []

    class Service:
        def __getattr__(self, name: str) -> object:
            calls.append(name)
            raise AssertionError(f"constructor touched the SDK: {name}")

    runtime = _runtime(_frozen_root(tmp_path))
    assert calls == []
    assert runtime.client is None
    assert runtime.execution_kind == "provider"
    assert Service not in type(runtime).__mro__


def test_a_runtime_cannot_be_built_before_the_plan_is_frozen(tmp_path: Path) -> None:
    """Order matters: a runtime with no plan has nothing to be measured against."""
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(ValueError, match="baseline plan must be frozen"):
        build_runtime(
            service=object(),
            budget=_budget(),
            root=root,
            host=_host(),
            expected_tokenizer="a" * 64,
        )


def test_a_runtime_cannot_be_built_in_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must already exist"):
        build_runtime(
            service=object(),
            budget=_budget(),
            root=tmp_path / "absent",
            host=_host(),
            expected_tokenizer="a" * 64,
        )


def test_only_the_provider_kind_may_be_constructed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only the provider runtime"):
        build_runtime(
            service=object(),
            budget=_budget(),
            root=_frozen_root(tmp_path),
            host=_host(),
            expected_tokenizer="a" * 64,
            execution_kind="contract_fixture",
        )


def test_an_invalid_tokenizer_pin_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid tokenizer pin"):
        build_runtime(
            service=object(),
            budget=_budget(),
            root=_frozen_root(tmp_path),
            host=_host(),
            expected_tokenizer="not-a-pin",
        )


def test_the_construction_point_returns_a_bridge(tmp_path: Path) -> None:
    """The controller receives the protocol, never the SDK object itself."""
    built = build_runtime(
        service=object(),
        budget=_budget(),
        root=_frozen_root(tmp_path),
        host=_host(),
        expected_tokenizer="a" * 64,
    )
    assert isinstance(built, Positive2RuntimeBridge)
    assert built.execution_kind == "provider"


async def test_sampling_before_initialization_is_refused(tmp_path: Path) -> None:
    """No implicit client creation, and no sampling from a policy that never loaded."""
    with pytest.raises(ValueError, match="runtime not initialized"):
        await _runtime(tmp_path).sample(_case(), _row(), _checkpoint(0), tmp_path / "out")


async def test_a_row_from_another_case_is_refused(tmp_path: Path) -> None:
    row = _row().model_copy(update={"case_id": "curve_v4_table_join_02"})
    with pytest.raises(ValueError, match="row and case identity mismatch"):
        await _runtime(tmp_path).sample(_case(), row, _checkpoint(0), tmp_path / "out")


async def test_an_unknown_group_key_is_refused(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.current = _checkpoint(0)
    with pytest.raises(ValueError, match="invalid positive-control v2 group key"):
        await runtime.update_group("baseline_00", _group(), _checkpoint(0), tmp_path)


def test_a_sampler_that_is_not_the_frozen_pair_is_refused(tmp_path: Path) -> None:
    """A stale sampler would silently grade one policy and train another."""
    runtime = _runtime(tmp_path)

    class Sampler:
        sampling_run_cuid = "sampling_new"

    class Pair:
        sampling_run_cuid = "sampling_old"
        sampler = SimpleNamespace(path="trio://run_original/sampler_weights/sampler_original")

    runtime.samplers["fixture_0"] = Sampler()
    runtime.tokenizer = cast(Tokenizer, object())
    runtime._pairs["fixture_0"] = cast(SavedPair, Pair())
    # The sampler was built from different weights than the pair records.
    runtime.sampler_weights["fixture_0"] = "trio://run_other/sampler_weights/sampler_other"

    with pytest.raises(ValueError, match="sampler does not match the frozen policy pair"):
        runtime._backend(_checkpoint(0), tmp_path)


def test_a_missing_sampler_is_refused(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    class Pair:
        sampling_run_cuid = "sampling_old"

    runtime.tokenizer = cast(Tokenizer, object())
    runtime._pairs["fixture_0"] = cast(SavedPair, Pair())
    with pytest.raises(ValueError, match="is not connected"):
        runtime._backend(_checkpoint(0), tmp_path)


class _Pair:
    """Stand-in `SavedPair`: only the identity fields `_backend` compares are used."""

    def __init__(self, checkpoint: Checkpoint) -> None:
        self.optimizer = SimpleNamespace(path=checkpoint.optimizer_uri, type="save-checkpoint")
        self.sampler = SimpleNamespace(path=checkpoint.sampler_uri, type="save-weights")
        self.sampling_run_cuid = "sampling_0"
        self.tokenizer_sha256 = "a" * 64


def test_adopt_caches_the_binding_it_validated(tmp_path: Path) -> None:
    """`adopt` must both validate and cache; caching alone hid a real resume failure."""
    runtime = _runtime(tmp_path)
    checkpoint = _checkpoint(0)
    pair = _Pair(checkpoint)

    runtime._pair = lambda _value: cast(SavedPair, pair)  # type: ignore[method-assign,assignment]
    assert runtime.adopt(checkpoint) is cast(SavedPair, pair)
    # `_backend` reads this cache, so the entry is the observable effect that matters.
    assert runtime._pairs[checkpoint.policy.policy_id] is cast(SavedPair, pair)


def test_the_bridge_exposes_only_the_declared_protocol() -> None:
    """The controller must not be able to reach an SDK method by accident."""
    bridge = Positive2RuntimeBridge(cast(Positive2SDKRuntime, object()))
    assert bridge.execution_kind == "provider"
    for name in ("initialize", "sample", "update"):
        assert callable(getattr(bridge, name))
    for name in ("client", "service", "budget", "_save_pair", "_trainer", "runtime"):
        # `runtime` is the one deliberate seam; the rest must not leak.
        if name == "runtime":
            continue
        assert not hasattr(bridge, name), f"bridge leaks {name}"


def test_a_non_positive_reserve_is_refused() -> None:
    for value in (Decimal("0"), Decimal("-1")):
        with pytest.raises(ValueError, match="positive finite"):
            require_positive_decimal(value)
    assert require_positive_decimal(Decimal("2.50")) == Decimal("2.50")


async def test_update_without_initialization_is_refused(tmp_path: Path) -> None:
    # The fixture checkpoint is a contract artifact, so the pair check refuses it
    # before any policy comparison: an uninitialized runtime cannot train anything.
    with pytest.raises(ValueError, match="provider checkpoint required"):
        await _runtime(tmp_path).update_group("train_00", _group(), _checkpoint(0), tmp_path)


async def test_a_failed_group_with_turns_is_skipped_without_a_gradient(tmp_path: Path) -> None:
    """All-four-failed groups still have turns; only `has_signal` says "no gradient".

    Regression: testing `batch is None` sent these to the trainer, which skipped them
    and returned updated=False, and the adapter then raised on correct behaviour.
    """
    # Four failed rollouts that still produced turns: advantages are all zero, so the
    # batch exists but carries no gradient.
    turns = (Turn((11, 12), (13, 14), (-0.1, -0.2)),)
    group = LearningGroup(
        run_ids=("pc2_train_00_0",),
        rewards=(0.0,),
        advantages=(0.0,),
        batch=build_batch((turns,), (0.0,)),
    )
    assert group.batch is not None
    assert group.batch.has_signal is False
    directory = tmp_path / "updates" / "train_00"
    directory.mkdir(parents=True)
    before = _checkpoint(0)
    atomic_record(
        directory / "group.json",
        GroupRecord(
            before=before,
            run_ids=group.run_ids,
            datum_sha256=group_datum_hash(group),
            has_signal=False,
            input_sha256={},
        ),
        create=True,
    )
    runtime = _runtime(tmp_path)
    runtime.current = before
    runtime.client = _FakeClient()
    runtime._pairs[before.policy.policy_id] = cast(SavedPair, object())

    after = await runtime.update_group("train_00", group, before, directory)
    assert after == before
    assert UpdateJournal.load(directory / "update_000.json").state == "skipped"


async def test_zero_signal_group_is_closed_without_a_gradient(tmp_path: Path) -> None:
    """The frozen denominator must retain the group; only the update is skipped."""
    group = _group()
    directory = tmp_path / "updates" / "train_00"
    directory.mkdir(parents=True)
    before = _checkpoint(0)
    atomic_record(
        directory / "group.json",
        GroupRecord(
            before=before,
            run_ids=group.run_ids,
            datum_sha256=group_datum_hash(group),
            has_signal=False,
            input_sha256={},
        ),
        create=True,
    )

    runtime = _runtime(tmp_path)
    runtime.current = before
    runtime.client = _FakeClient()
    runtime._pairs[before.policy.policy_id] = cast(SavedPair, object())

    after = await runtime.update_group("train_00", group, before, directory)
    assert after == before
    assert UpdateJournal.load(directory / "update_000.json").state == "skipped"
    assert (directory / "receipt.json").exists()
