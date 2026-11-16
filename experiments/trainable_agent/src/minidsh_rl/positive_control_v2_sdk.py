"""PyTRIO adapter for positive-control v2; constructed only by an explicit caller.

ADR 0045 froze the runtime contract; this module is the implementation the v2
protocol needs.  Three properties matter more than the plumbing:

* no client is created at import or construction time, so importing this module
  cannot spend money — `initialize` is the first thing that talks to the provider;
* evaluation samples from the *snapshot being measured*, not from the newest
  weights, because grading the final policy at every checkpoint would erase the
  curve the experiment exists to draw;
* a zero-signal group still writes a journal, a receipt and a checkpoint identity,
  so the frozen denominator stays complete even though no gradient is taken.
"""

from __future__ import annotations

import asyncio
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Literal

from minidsh.infrastructure.process import ProcessRunner
from minidsh_rl.cost_store import CostStore
from minidsh_rl.curve_sdk import FinancialBackend
from minidsh_rl.learning_contracts import LearningCase
from minidsh_rl.learning_execution import (
    Checkpoint,
    group_datum_hash,
    write_update_receipt,
)
from minidsh_rl.learning_rollout import LearningGroup, LearningRollout, run_learning_rollout
from minidsh_rl.learning_schedule import ExecutionKind, atomic_record
from minidsh_rl.learning_sdk import LearningSDKRuntime, SavedPair
from minidsh_rl.minirepo_verification import IMAGE
from minidsh_rl.positive_control_v2 import Positive2Row
from minidsh_rl.transaction import UpdateJournal

_SLOT = re.compile(r"^(train|refill)_(\d{2})$")


class Positive2SDKRuntime(LearningSDKRuntime):
    """RuntimePort for v2: snapshot-aware sampling and group-keyed checkpoints."""

    execution_kind: Literal["provider"] = "provider"

    def __init__(  # noqa: PLR0913 - dependencies are explicit at the SDK boundary
        self,
        service: object,
        budget: CostStore,
        root: Path,
        host: ProcessRunner,
        expected_tokenizer: str,
        *,
        image: str = IMAGE,
    ) -> None:
        super().__init__(service, budget, root, host, expected_tokenizer, image=image)
        self._pairs: dict[str, SavedPair] = {}
        self.sampler_weights: dict[str, str] = {}

    def _pair_for(self, checkpoint: Checkpoint) -> SavedPair:
        """Cache the frozen weight pair; a saved snapshot never changes."""
        key = checkpoint.policy.policy_id
        if key not in self._pairs:
            self._pairs[key] = self._pair(checkpoint)
        return self._pairs[key]

    def adopt(self, checkpoint: Checkpoint) -> SavedPair:
        """Validate a checkpoint's binding and cache it for later sampling.

        One method, not two call sites: `_pair` validates but does not cache, and
        seeding only that is what made the first resumed sample fail its
        sampler/pair identity check after a crash.
        """
        pair = self._pair(checkpoint)
        key = checkpoint.policy.policy_id
        self._pairs[key] = pair
        # Record which weights the sampler must have been built from. `_save_pair` in the
        # parent fills `self.samplers` without touching this map, so `_backend` falls back
        # to the pair's own path when the entry is absent.
        self.sampler_weights.setdefault(key, checkpoint.sampler_uri)
        return pair

    def _backend(self, checkpoint: Checkpoint, directory: Path) -> FinancialBackend:
        if self.tokenizer is None:
            raise ValueError("runtime not initialized")
        pair = self._pair_for(checkpoint)
        sampler = self.samplers.get(checkpoint.policy.policy_id)
        if sampler is None:
            raise ValueError("sampler for the checkpoints policy is not connected")
        # Compare the *weights* the sampler was created from, not the sampling run id:
        # reconnecting a saved checkpoint starts a new sampling run, so its cuid always
        # differs from the one recorded when the pair was first saved. The path is the
        # stable identity, and it is what decides which weights get graded.
        if self.sampler_weights.get(checkpoint.policy.policy_id) not in (None, pair.sampler.path):
            raise ValueError("sampler does not match the frozen policy pair")
        backend = FinancialBackend(
            sampler, self.tokenizer, self.budget, checkpoint.policy.policy_id
        )
        backend.directory = directory
        return backend

    async def initialize(self, directory: Path) -> Checkpoint:
        checkpoint = await super().initialize(directory)
        (directory / "execution-kind.json").write_text(
            json.dumps({"execution_kind": self.execution_kind})
        )
        return checkpoint

    async def reconnect(self, before: Checkpoint, directory: Path) -> Checkpoint:
        """Rebuild the client and samplers for a resumed run.

        The parent's `restore` is bound to the v1 `ExecutionState`; v2 keeps its own
        `state.json`, so reconnection has to be expressed here. Both directions of
        the policy pair are re-verified: a resume that silently attached to different
        weights would make every later group incomparable with the ones already paid.
        """
        if directory != self.root or self.client is not None:
            raise ValueError("invalid runtime reconnect")
        initial = Checkpoint.model_validate_json((directory / "initial.json").read_bytes())
        for cached in (before, initial):
            self.adopt(cached)
        folder = (
            directory
            / "resumes"
            / f"attempt_{len(list((directory / 'resumes').glob('attempt_*'))) + 1:03}"
        )
        folder.mkdir(parents=True, exist_ok=False)

        def attach() -> None:
            self.client = self.service.create_training_client_from_state_with_optimizer(
                before.optimizer_uri
            )
            self._check_tokenizer()
            for checkpoint in (before, initial):
                if checkpoint.policy.policy_id not in self.samplers:
                    self.samplers[checkpoint.policy.policy_id] = self._connect(
                        checkpoint.sampler_uri
                    )
                self.sampler_weights[checkpoint.policy.policy_id] = checkpoint.sampler_uri

        await asyncio.to_thread(self._trainer(folder).charge, "reconnect", Decimal(".01"), attach)
        atomic_record(folder / "reconnected.json", before, create=True)
        self.current = before
        return before

    async def sample(
        self,
        case: LearningCase,
        row: Positive2Row,
        before: Checkpoint,
        output: Path,
    ) -> LearningRollout:
        """Sample from the checkpoint under test, not from the newest weights."""
        if row.case_id != case.task.case_id:
            raise ValueError("row and case identity mismatch")
        backend = self._backend(before, output)
        return await run_learning_rollout(
            case,
            backend,
            policy=before.policy,
            run_id=row.run_id,
            seed=row.seed,
            output=output,
            host=self.host,
            image=self.image,
        )

    async def update_group(
        self,
        key: str,
        group: LearningGroup,
        before: Checkpoint,
        output: Path,
    ) -> Checkpoint:
        """Train only from the newest weights; a zero-signal group closes without one.

        Named differently from the parent's `update` on purpose: this one returns
        the confirmed checkpoint, which the v2 controller needs and the older
        controller tracked through mutable state instead.
        """
        match = _SLOT.match(key)
        if match is None:
            raise ValueError("invalid positive-control v2 group key")
        self._pair_for(before)
        if before != self.current or self.client is None:
            raise ValueError("update policy mismatch")
        slot = int(match.group(2))
        journal_path = output / f"update_{slot:03}.json"
        # `batch is None` only when no rollout produced any turn. A group whose four
        # rollouts all failed still has turns, so it carries a batch with zero
        # advantage — `has_signal` is the field that says whether a gradient exists.
        # Testing `batch is None` sent those groups into the trainer, which correctly
        # skipped them and returned updated=False, and this adapter then raised on a
        # group that had behaved exactly as specified.
        if group.batch is None or not group.batch.has_signal:
            journal = UpdateJournal.create(journal_path, group_datum_hash(group))
            journal.advance("skipped")
            write_update_receipt(output, key, group, before, before, updated=False)
            return before
        result = await asyncio.to_thread(self._trainer(output).update, slot, group.batch)
        if result["updated"] is not True:
            raise ValueError("signal-bearing group did not update")
        after = await asyncio.to_thread(self._save_pair, output, key)
        write_update_receipt(output, key, group, before, after, updated=True)
        atomic_record(output / "checkpoint.json", after, create=True)
        self.current = after
        return after


class Positive2RuntimeBridge:
    """Adapt the SDK runtime to the v2 `Positive2Runtime` protocol.

    The bridge exists so the controller never imports `pytrio`: it sees only the
    three methods it declares, and the SDK stays behind one construction point.
    """

    execution_kind: Literal["provider"] = "provider"

    def __init__(self, runtime: Positive2SDKRuntime) -> None:
        self.runtime = runtime

    async def initialize(self, directory: Path) -> Checkpoint:
        return await self.runtime.initialize(directory)

    async def reconnect(self, before: Checkpoint, directory: Path) -> Checkpoint:
        return await self.runtime.reconnect(before, directory)

    async def sample(
        self, case: LearningCase, row: Positive2Row, before: Checkpoint, output: Path
    ) -> LearningRollout:
        return await self.runtime.sample(case, row, before, output)

    async def update(
        self, key: str, group: LearningGroup, before: Checkpoint, output: Path
    ) -> Checkpoint:
        return await self.runtime.update_group(key, group, before, output)


def build_runtime(  # noqa: PLR0913 - explicit construction at the operator boundary
    *,
    service: object,
    budget: CostStore,
    root: Path,
    host: ProcessRunner,
    expected_tokenizer: str,
    image: str = IMAGE,
    execution_kind: ExecutionKind = "provider",
) -> Positive2RuntimeBridge:
    """The single construction point; every argument must come from a frozen plan."""
    if execution_kind != "provider":
        raise ValueError("only the provider runtime may be constructed here")
    if re.fullmatch(r"[a-f0-9]{64}", expected_tokenizer) is None:
        raise ValueError("invalid tokenizer pin")
    if not root.is_dir():
        raise ValueError("positive-control directory must already exist")
    if not (root / "baseline-plan.json").exists():
        raise ValueError("baseline plan must be frozen before a runtime is built")
    return Positive2RuntimeBridge(
        Positive2SDKRuntime(service, budget, root, host, expected_tokenizer, image=image)
    )


def require_positive_decimal(value: Decimal) -> Decimal:
    """Reject a zero or negative reserve before it reaches a charge."""
    if not value.is_finite() or value <= 0:
        raise ValueError("reserve must be a positive finite amount")
    return value
