"""One real trajectory: prove model + tools + container + grader agree before 152 rows.

This is deliberately *not* part of the 152-row denominator.  It answers one question
the offline contracts cannot: does the provider adapter feed the real Harness well
enough for the model to act and for the independent grader to score it?  Cost is one
sample plus one container run (~0.09 CNY), and the artifact it leaves is a single
`LearningRollout` that a human can read.

Usage:
    PYTHONPATH=experiments/trainable_agent/src uv run --project experiments/trainable_agent \\
        python -m minidsh_rl.positive_control_v2_smoke --directory <fresh dir>
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
from minidsh_rl.cost_store import CostStore
from minidsh_rl.learning_components import verify_components
from minidsh_rl.learning_run import ExecutionConfig
from minidsh_rl.learning_sdk import tokenizer_pin
from minidsh_rl.positive_control import positive_cases
from minidsh_rl.positive_control_v2 import make_baseline_plan
from minidsh_rl.positive_control_v2_sdk import Positive2SDKRuntime
from minidsh_rl.preflight import service_evidence


def _ledger(root: Path, limit: Decimal, quote: Path) -> CostStore:
    """A one-off ledger sized for a single sample, not for the whole experiment."""
    carry = root / "carry.json"
    carry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "limit": str(limit),
                "spent": "0",
                "pending": {},
                "settled": [],
            },
            indent=2,
        )
        + "\n"
    )
    (root / "quote.json").write_bytes(quote.read_bytes())
    return CostStore(
        root / "budget.json",
        legacy=carry,
        quote=root / "quote.json",
        limit=limit,
        uncertainty=Decimal("0"),
    )


async def smoke(root: Path, config_path: Path, quote: Path, limit: Decimal) -> dict[str, object]:
    await asyncio.to_thread(root.mkdir, parents=True, exist_ok=False)
    config = ExecutionConfig.model_validate_json(await asyncio.to_thread(config_path.read_bytes))
    await asyncio.to_thread(verify_components, config.components.verify())
    evidence = root / "service.json"
    await service_evidence(evidence)
    with _ledger(root, limit, quote) as budget:
        host = LocalCommandRunner(WorkspaceSandbox(root), CommandPolicy(frozenset({"docker"})))
        runtime = Positive2SDKRuntime(
            trio.ServiceClient(), budget, root, host, config.tokenizer_sha256, image=config.image
        )
        checkpoint = await runtime.initialize(root)
        if runtime.tokenizer is None:
            raise ValueError("tokenizer missing after initialization")
        # The pin must match the frozen config, or every later row is unrenderable.
        actual = tokenizer_pin(runtime.tokenizer)
        if actual != config.tokenizer_sha256:
            raise ValueError(f"tokenizer pin mismatch: {actual}")
        row = make_baseline_plan().rows[0]
        case = positive_cases()[row.case_id]
        rollout = await runtime.sample(case, row, checkpoint, root / "rollouts" / row.run_id)
        return {
            "run_id": rollout.run_id,
            "case_id": rollout.case.task.case_id,
            "status": rollout.status,
            "success": rollout.success,
            "verification_executed": rollout.verification_executed,
            "turns": len(rollout.turns),
            "total_tokens": rollout.total_tokens,
            "budget_violations": list(rollout.budget_violations),
            "spent_cny": str(budget.committed),
            "available_cny": str(budget.available),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--quote", type=Path, required=True)
    parser.add_argument("--limit", default="1.00")
    args = parser.parse_args()
    for name in ("httpx", "httpcore", "pytrio"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    try:
        sys.stdout.write(
            json.dumps(
                asyncio.run(smoke(args.directory, args.config, args.quote, Decimal(args.limit))),
                indent=2,
            )
            + "\n"
        )
    except Exception as error:
        sys.stdout.write(json.dumps({"error_type": type(error).__name__}) + "\n")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
