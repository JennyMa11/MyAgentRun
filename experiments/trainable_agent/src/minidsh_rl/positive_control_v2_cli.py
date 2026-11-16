"""Offline prepare/inspect and one explicitly admitted v2 provider run.

Same shape as `curve_run.py`: `prepare` freezes everything, `inspect` prints the
numbers a human approves, and `run` is the only command that can spend.  Nothing
here downloads a model, opens a socket or constructs an SDK client until `run`
has verified the frozen plan against the live ledger.

    uv run --project experiments/trainable_agent python -m minidsh_rl.positive_control_v2_cli \
        prepare --directory benchmark-results/trainable_agent/positive-control-v2/run-01
"""

import argparse
import asyncio
import json
import logging
import sys
from decimal import Decimal
from pathlib import Path

import pytrio as trio  # type: ignore[import-untyped]

from minidsh.infrastructure.process import CommandPolicy, LocalCommandRunner
from minidsh.infrastructure.sandbox import WorkspaceSandbox
from minidsh_rl.billing_calibration import price_card
from minidsh_rl.cost_store import CostStore
from minidsh_rl.curve_run import validate_config
from minidsh_rl.learning_components import verify_components
from minidsh_rl.learning_gates import PROJECT
from minidsh_rl.learning_run import ExecutionConfig
from minidsh_rl.learning_schedule import digest
from minidsh_rl.positive_control_v2 import (
    PLANNED_GROUPS,
    SAMPLED_GROUPS,
    TOTAL_ROWS,
    load_plan,
    plan_sha256,
    required_cny,
    verify_plan,
)
from minidsh_rl.positive_control_v2_recovery import (
    FencedResources,
    load_fence,
    outstanding,
    record_fence,
)
from minidsh_rl.positive_control_v2_run import (
    execute_positive_control_v2,
    prepare_positive_control_v2,
)
from minidsh_rl.positive_control_v2_sdk import build_runtime
from minidsh_rl.preflight import service_evidence


def _open(root: Path) -> CostStore:
    """Open the ledger `fund` created; refuse to invent one.

    Deliberately not `CurveBudget`: that type binds the older 150-CNY migration and
    requires a `migration.json`, which a fresh positive-control epoch does not have.
    """
    finance = root / "finance"
    for name in ("budget.json", "carry.json", "quote.json"):
        if not (finance / name).exists():
            raise ValueError(f"run is not funded: missing finance/{name}; run `fund` first")
    funding = json.loads((root / "funding.json").read_bytes())
    return CostStore(
        finance / "budget.json",
        legacy=finance / "carry.json",
        quote=finance / "quote.json",
        limit=Decimal(str(funding["limit_cny"])),
        uncertainty=Decimal("0"),
    )


def _config(root: Path) -> ExecutionConfig:
    config = ExecutionConfig.model_validate_json((root / "execution-config.json").read_bytes())
    validate_config(config)
    return config


async def _run(root: Path, *, resume: bool, recover: bool) -> dict[str, object]:
    """Admit the whole run, then let the controller do baseline -> plan -> train.

    `plan.json` does not exist yet on a fresh directory: it is frozen from the
    baseline measurements, which are themselves the first thing the controller does.
    So admission is checked against the *declared* 152-row envelope, and the frozen
    plan is re-verified after it appears.
    """
    plan_path = root / "plan.json"
    if plan_path.exists():
        frozen = load_plan(plan_path)
        verify_plan(frozen)
        required = frozen.required_cny
    else:
        # No plan yet: price the worst case the protocol allows, so a run that cannot
        # afford the declared envelope never samples the first row.
        required = required_cny(TOTAL_ROWS, PLANNED_GROUPS, SAMPLED_GROUPS)
    config = _config(root)
    with _open(root) as budget:
        fence = load_fence(root, budget.path)
        resources = FencedResources(budget, fence, root=root)
        if resources.has_pending:
            blocked = outstanding(budget.path, fence, root=root)
            raise ValueError(f"unknown financial operation; never replay: {blocked}")
        if not resume and budget.available < required:
            raise ValueError("complete positive-control v2 allocation unavailable")
        live = root / "live-preflight"
        live.mkdir(exist_ok=True)
        attempt = live / f"attempt_{len(list(live.glob('attempt_*.json'))) + 1:03}.json"
        await service_evidence(attempt)
        price_card(attempt)
        verify_components(config.components.verify())
        host = LocalCommandRunner(WorkspaceSandbox(root), CommandPolicy(frozenset({"docker"})))
        runtime = build_runtime(
            service=trio.ServiceClient(),
            budget=budget,
            root=root,
            host=host,
            expected_tokenizer=config.tokenizer_sha256,
            image=config.image,
        )
        state = await execute_positive_control_v2(
            root, runtime, resources=resources, resume=resume, recover=recover
        )
        if plan_path.exists():
            verify_plan(load_plan(plan_path))
        return {
            "phase": state.phase,
            "completed_rows": len(state.completed),
            "sampled_groups": len(state.groups),
            "updated_groups": len(state.updated),
            "refilled": state.refilled,
            "skipped_checkpoints": state.skipped_checkpoints,
            "status": (state.final or {}).get("status"),
            "reason": (state.final or {}).get("reason"),
            "available_cny": str(budget.available),
        }


def fund(root: Path, *, limit: Decimal, parent: Path, quote: Path) -> dict[str, object]:
    """Open a new, independently funded epoch and carry only *unconfirmed* spend over.

    Two numbers must not be confused. The parent's `historical` + `charges` is what
    has already been spent and billed — it is history, and this epoch does not
    inherit it as a starting balance, or nothing could ever be funded twice. What it
    does inherit is the three operations whose outcome was never confirmed: those may
    still become real charges, so they are carried forward as spent and the new
    allowance must cover them. The parent file is read, hashed and bound, never
    edited, so the historical audit trail keeps its exact bytes.
    """
    if (root / "finance").exists():
        raise ValueError("positive-control v2 directory already funded")
    state = json.loads(parent.read_bytes())
    charges = state.get("charges", {})
    unconfirmed_ops = sorted(k for k, c in charges.items() if c.get("status") == "pending")
    unconfirmed = sum((Decimal(str(charges[key]["amount"])) for key in unconfirmed_ops), Decimal(0))
    if limit <= unconfirmed:
        raise ValueError("new allowance does not cover inherited unconfirmed spend")
    finance = root / "finance"
    finance.mkdir(parents=True)
    legacy = finance / "carry.json"
    legacy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "limit": str(limit),
                "spent": str(unconfirmed),
                "pending": {},
                "settled": unconfirmed_ops,
            },
            indent=2,
        )
        + "\n"
    )
    (finance / "quote.json").write_bytes(quote.read_bytes())
    ledger = CostStore(
        finance / "budget.json",
        legacy=legacy,
        quote=finance / "quote.json",
        limit=limit,
        uncertainty=Decimal("0"),
    )
    try:
        available = ledger.available
    finally:
        ledger.close()
    record = {
        "schema_version": 1,
        "parent": str(parent),
        "parent_sha256": digest(parent),
        "limit_cny": str(limit),
        "inherited_unconfirmed_cny": str(unconfirmed),
        "inherited_unconfirmed_operations": unconfirmed_ops,
        "available_cny": str(available),
        "parent_historical_cny": str(state["historical"]),
        "note": "parent totals are history and are not re-charged to this epoch",
    }
    (root / "funding.json").write_text(json.dumps(record, indent=2) + "\n")
    return {
        "limit_cny": str(limit),
        "inherited_unconfirmed_cny": str(unconfirmed),
        "inherited_unconfirmed_count": len(unconfirmed_ops),
        "available_cny": str(available),
        "parent_sha256": digest(parent),
    }


def inspect(
    root: Path, *, available: Decimal, committed: Decimal, pending: bool
) -> dict[str, object]:
    """The numbers a human approves, computed from the frozen plan and the live lease.

    Pure over its arguments so a test can assert the report without a terminal or a
    lease, and so `run` and `inspect` cannot disagree about which `required_cny`
    they enforce.
    """
    plan_path = root / "plan.json"
    body: dict[str, object] = {"rows": TOTAL_ROWS, "selected": None, "plan_sha256": None}
    if plan_path.exists():
        frozen = load_plan(plan_path)
        verify_plan(frozen)
        body |= {
            "selected": list(frozen.selected),
            "required_cny": str(frozen.required_cny),
            "plan_sha256": plan_sha256(frozen),
        }
    required = body.get("required_cny")
    body |= {
        "available_cny": str(available),
        "committed_cny": str(committed),
        "pending": pending,
        "affordable": required is not None and available >= Decimal(str(required)),
    }
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "fund", "fence", "inspect", "run"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--quote", type=Path)
    parser.add_argument("--limit")
    parser.add_argument("--operation")
    parser.add_argument("--row")
    parser.add_argument("--reason")
    args = parser.parse_args()
    for name in ("httpx", "httpcore", "pytrio"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    try:
        if args.command == "prepare":
            if args.config is None:
                parser.error("prepare requires --config")
            config = ExecutionConfig.model_validate_json(args.config.read_bytes())
            validate_config(config)
            plan = prepare_positive_control_v2(args.directory)
            (args.directory / "execution-config.json").write_bytes(args.config.read_bytes())
            sys.stdout.write(
                json.dumps(
                    {
                        "baseline_rows": len(plan.rows),
                        "families": sorted({row.family for row in plan.rows}),
                        "source_root": str(PROJECT),
                        "config_sha256": config.tokenizer_sha256,
                    }
                )
                + "\n"
            )
        elif args.command == "fund":
            if args.parent is None or args.quote is None or args.limit is None:
                parser.error("fund requires --parent --quote --limit")
            sys.stdout.write(
                json.dumps(
                    fund(
                        args.directory,
                        limit=Decimal(args.limit),
                        parent=args.parent,
                        quote=args.quote,
                    )
                )
                + "\n"
            )
        elif args.command == "fence":
            if args.operation is None or args.row is None or args.reason is None:
                parser.error("fence requires --operation --row --reason")
            with _open(args.directory) as budget:
                fence = record_fence(
                    args.directory,
                    budget.path,
                    operation=args.operation,
                    row_run_id=args.row,
                    reason=args.reason,
                )
            sys.stdout.write(fence.model_dump_json(indent=2) + "\n")
        elif args.command == "inspect":
            with _open(args.directory) as budget:
                report = inspect(
                    args.directory,
                    available=budget.available,
                    committed=budget.committed,
                    pending=FencedResources(
                        budget, load_fence(args.directory, budget.path)
                    ).has_pending,
                )
                sys.stdout.write(json.dumps(report) + "\n")
        else:
            sys.stdout.write(
                json.dumps(
                    asyncio.run(_run(args.directory, resume=args.resume, recover=args.recover))
                )
                + "\n"
            )
    except Exception as error:
        # Exception messages from HTTP clients may include signed URLs; never print them.
        sys.stdout.write(
            json.dumps({"error_type": type(error).__name__, "automatic_retry": False}) + "\n"
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
