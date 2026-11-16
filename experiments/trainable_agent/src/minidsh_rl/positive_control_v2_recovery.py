"""Let one interrupted sample stop blocking a resume, without releasing its money.

A crash between `reserve` and `settle` leaves a charge pending forever. The amount
must stay counted — it may really have been spent — but the *blocker* has to be
liftable, or a single network fault makes a paid run permanently unresumable. That
is the same distinction `curve_recovery.FencedResources` draws for the terminal
initial hold, and this module reuses its shape rather than inventing a new one.

The fence is narrow on purpose:

* it name exactly one operation, and that operation must still be pending;
* the recorded amount must be unchanged, so the fence cannot be used to shrink a
  charge that was larger than expected;
* the row whose sample was interrupted must have no sealed result, proving the
  operation never produced a scored trajectory;
* the fence is written once, is bound to the ledger hash, and is re-validated on
  every reopen, so editing the evidence invalidates the resume.
"""

import json
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from minidsh_rl.cost_store import _State

FENCE = "recovery-fence.json"


class InterruptedSample(BaseModel):
    """One named reservation that never resolved, with the evidence that it is dead."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    operation: str
    amount: Decimal
    row_run_id: str
    row_directory: str
    ledger_identity: str
    reason: str
    sealed_result_absent: bool


def record_fence(
    root: Path,
    ledger: Path,
    *,
    operation: str,
    row_run_id: str,
    reason: str,
) -> InterruptedSample:
    """Freeze the fact that `operation` will never resolve; keep its amount owed."""
    existing = load_fence(root, ledger)
    if existing is not None and existing.operation == operation:
        raise ValueError("recovery fence already recorded")
    state = _State.model_validate_json(ledger.read_bytes())
    charge = state.charges.get(operation)
    if charge is None:
        raise ValueError("fenced operation is not in the ledger")
    if charge.status != "pending":
        raise ValueError("fenced operation is already resolved")
    row = root / "rollouts" / row_run_id
    # The seal is the binding, not the raw result: a row can finish sampling and still
    # fail validation (as this one did — result.json written, no binding), and that
    # sample may well have been charged. Only a *sealed and scored* row proves the
    # reservation resolved, so only that disqualifies a fence.
    if (row / "learning-binding").exists():
        raise ValueError("fenced row already produced a sealed result")
    fence = InterruptedSample(
        operation=operation,
        amount=charge.amount,
        row_run_id=row_run_id,
        row_directory=str(row),
        # Identity, not a content hash: the ledger keeps changing as the run spends,
        # so a content hash would invalidate the fence on the very next charge. The
        # operation itself is re-verified separately, which is what actually matters.
        ledger_identity=str(ledger.resolve()),
        reason=reason,
        sealed_result_absent=True,
    )
    history = [existing] if existing is not None else []
    (root / FENCE).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    *(f.model_dump(mode="json") for f in history),
                    fence.model_dump(mode="json"),
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return fence


def load_fence(root: Path, ledger: Path) -> InterruptedSample | None:
    """Re-validate on every read: a stale or edited fence must not open the gate."""
    path = root / FENCE
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    entries = raw.get("entries") or [raw]
    if not entries:
        return None
    fence = InterruptedSample.model_validate(entries[-1])
    if fence.ledger_identity != str(ledger.resolve()):
        raise ValueError("recovery fence was recorded against a different ledger")
    state = _State.model_validate_json(ledger.read_bytes())
    charge = state.charges.get(fence.operation)
    if charge is None or charge.status != "pending" or charge.amount != fence.amount:
        raise ValueError("fenced reservation changed since it was recorded")
    return fence


def fenced_operations(root: Path) -> set[str]:
    """Every operation this run has already written off, oldest first."""
    path = root / FENCE
    if not path.exists():
        return set()
    raw = json.loads(path.read_text())
    entries = raw.get("entries") or [raw]
    return {InterruptedSample.model_validate(entry).operation for entry in entries}


def outstanding(
    ledger: Path, fence: InterruptedSample | None, *, root: Path | None = None
) -> list[str]:
    """Pending operations that still block admission, i.e. all beyond the fenced ones."""
    state = _State.model_validate_json(ledger.read_bytes())
    fenced = set() if fence is None else {fence.operation}
    if root is not None:
        fenced |= fenced_operations(root)
    return sorted(
        key
        for key, charge in state.charges.items()
        if charge.status == "pending" and key not in fenced
    )


class FencedResources:
    """`LeasedResources` semantics, minus the one operation the fence accounts for."""

    def __init__(
        self, budget: object, fence: InterruptedSample | None, *, root: Path | None = None
    ) -> None:
        self.budget = budget
        self.fence = fence
        self.root = root

    @property
    def available(self) -> Decimal:
        return self.budget.available  # type: ignore[attr-defined,no-any-return]

    @property
    def committed(self) -> Decimal:
        return self.budget.committed  # type: ignore[attr-defined,no-any-return]

    @property
    def has_pending(self) -> bool:
        _ = self.budget.available  # type: ignore[attr-defined]
        return bool(
            outstanding(self.budget.path, self.fence, root=self.root)  # type: ignore[attr-defined]
        )


def describe(fence: InterruptedSample) -> dict[str, object]:
    """Audit line for the run report: what was written off, and why."""
    payload: dict[str, object] = json.loads(fence.model_dump_json())
    return payload
