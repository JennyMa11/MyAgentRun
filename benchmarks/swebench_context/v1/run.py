# ruff: noqa: T201, RUF001 -- bounded experiment metadata, no prompts or credentials
"""Frozen six-strategy Lite20 sweep; single agent, sequential disposable containers."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import statistics
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

from myagentrun.adapters import JsonlTraceWriter, ScriptedModel
from myagentrun.benchmark.hashes import frozen_tree_digest
from myagentrun.benchmark.public_swe import PublicStudyError, UsageModel, _capture_patch, write_json
from myagentrun.benchmark.swebench_smoke import (
    SmokeSelection,
    configured_profile,
    grade_patch,
    provider_factory,
    record_isolation,
    runtime_options,
)
from myagentrun.composition import RuntimeComponents, RuntimeOptions, build_local_runtime
from myagentrun.domain import AgentBudget, JsonValue, ModelResponse
from myagentrun.infrastructure.containers import ContainerResources, DockerHost, DockerSandbox
from myagentrun.infrastructure.process import CommandPolicy, ProcessRunner
from myagentrun.tools import GitTool, ShellTool
from myagentrun.tools.runtime import ToolError

STRATEGIES = ("budget", "sliding", "summary", "retrieval", "retrieval_summary", "symbol")
TASK_COUNT = 20
RESOURCES = ContainerResources(platform="linux/arm64", memory_mb=1024, cpus=1, pids_limit=128)
COMMANDS = frozenset(
    {"python", "python3", "pytest", "git", "grep", "find", "ls", "cat", "sed", "head", "tail", "wc"}
)


def schedule(identities: list[str]) -> list[dict[str, Any]]:
    """Rotate arm order for each original task without selecting by outcome."""
    return [
        {
            "instance_id": identity,
            "strategy": strategy,
            "status": "not_run",
            "success": None,
            "tokens": 0,
            "unknown_usage_calls": 0,
        }
        for index, identity in enumerate(identities)
        for strategy in (
            *STRATEGIES[index % len(STRATEGIES) :],
            *STRATEGIES[: index % len(STRATEGIES)],
        )
    ]


def strategy_options(selection: SmokeSelection, strategy: str) -> RuntimeOptions:
    """Keep every baseline option except the selected Context implementation."""
    if strategy not in (*STRATEGIES, "summary_pruned", "summary_model"):
        raise ValueError("unknown context strategy")
    return replace(runtime_options(selection), context_strategy=strategy)


def remaining_allowance(rows: list[dict[str, Any]], *, total: int, per_task: int) -> int:
    """Reserve one nominal allowance per unknown task; never call it actual usage."""
    known = sum(row.get("tokens", 0) for row in rows)
    reserve = sum(row.get("unknown_usage_calls", 0) > 0 for row in rows) * per_task
    return max(0, total - known - reserve)


def summarize_arm(rows: list[dict[str, Any]], nominal: int) -> dict[str, Any]:
    """Preserve declared denominator, failed consumption, and missing evidence."""
    known = sum(row.get("tokens", 0) for row in rows)
    unknown = sum(row.get("unknown_usage_calls", 0) for row in rows)
    attempted = sum(row["status"] != "not_run" for row in rows)
    graded = sum(isinstance(row.get("success"), bool) for row in rows)
    passed = sum(row.get("success") is True for row in rows)
    terminal = all(row["status"] not in {"running", "not_run"} for row in rows)
    durations = [row["duration_ms"] for row in rows if "duration_ms" in row]
    percentiles = (
        statistics.quantiles(durations, n=100, method="inclusive")
        if len(durations) > 1
        else durations * 99
    )
    steps = [row["steps"] for row in rows if "steps" in row]
    tools = [row["tool_calls"] for row in rows if "tool_calls" in row]
    return {
        "declared": len(rows),
        "attempted": attempted,
        "graded": graded,
        "passed": passed,
        "not_run": len(rows) - attempted,
        "ungraded_attempts": attempted - graded,
        "confirmed_success_share": passed / len(rows) if rows else None,
        "success_rate": passed / len(rows) if rows and graded == len(rows) else None,
        "known_tokens": known,
        "actual_total_tokens": known if terminal and not unknown else None,
        "unknown_usage_calls": unknown,
        "nominal_tokens": nominal,
        "total_budget_satisfied": False
        if known > nominal
        else True
        if terminal and not unknown
        else None,
        "avg_steps": statistics.mean(steps) if steps else None,
        "steps_sample_count": len(steps),
        "avg_tool_calls": statistics.mean(tools) if tools else None,
        "tool_calls_sample_count": len(tools),
        "latency_p50_ms": percentiles[49] if percentiles else None,
        "latency_p95_ms": percentiles[94] if percentiles else None,
        "latency_sample_count": len(durations),
        "latency_scope": "Agent Loop only; excludes image setup and independent verification",
        "latency_quantile": "linear interpolation at (n-1)*p",
    }


def reader_source() -> str:
    """Reuse the real file worker with a frozen Python3.8 compatibility adapter."""
    worker = files("myagentrun.infrastructure").joinpath("file_worker.py").read_text()
    original = 'dispatch(Path("/workspace"), request["operation"], request["arguments"])'
    if worker.count(original) != 1:
        raise ValueError("file worker entry contract changed")
    compatibility = (
        "from __future__ import annotations\n"
        "from pathlib import Path\n"
        "def _context_is_relative_to(self, other):\n"
        "    try:\n"
        "        self.relative_to(other)\n"
        "        return True\n"
        "    except ValueError:\n"
        "        return False\n"
        "if not hasattr(Path, 'is_relative_to'):\n"
        "    Path.is_relative_to = _context_is_relative_to\n"
    )
    return compatibility + worker.replace(original, original.replace("/workspace", "/testbed"))


class ContainerContextFiles:
    """Context-only reads of the current container, never the host artifact directory."""

    def __init__(self, runner: ProcessRunner) -> None:
        self.runner = runner
        self.worker = reader_source()

    async def invoke(self, operation: str, arguments: dict[str, JsonValue]) -> JsonValue:
        if operation not in {"list_files", "read_file"}:
            raise ToolError("context_read_only", "context backend only permits reads")
        request = json.dumps({"operation": operation, "arguments": arguments})
        observed = await self.runner.run(("python", "-c", self.worker, request))
        if observed.timed_out or observed.exit_code:
            raise ToolError("context_backend_failed", "container context read failed")
        payload = json.loads(observed.stdout)
        if payload.get("error") is not None:
            raise ToolError("context_read_denied", "container context read denied")
        return cast("JsonValue", payload["result"])

    async def prepare(self, operation: str, arguments: dict[str, JsonValue]) -> tuple[str, str]:
        del arguments
        return operation, "read-only context operation"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluator_path(path: Path) -> Path:
    """Keep venv/bin/python identity: resolve() discards its pyvenv.cfg context."""
    return path.absolute()


def verify_protocol(root: Path) -> dict[str, Any]:
    """Fail before credentials or calls if frozen code, gates, or inputs changed."""
    protocol = json.loads((root / "protocol.json").read_text())
    if sha(root / "protocol.json") != (root / "protocol.sha256").read_text().strip():
        raise ValueError("protocol changed after preparation")
    if frozen_tree_digest(root / "reproduction") != protocol["reproduction_hash"]:
        raise ValueError("frozen execution files changed")
    selection = SmokeSelection.model_validate_json((root / "selection.json").read_text())
    if sha(root / "selection.json") != protocol["selection_sha256"]:
        raise ValueError("selection changed")
    for name, field in (
        ("images.json", "images_sha256"),
        ("gate-evidence-index.json", "gate_index_sha256"),
    ):
        if sha(root / name) != protocol[field]:
            raise ValueError("image mapping or gate index changed")
    selection.verify(Path(protocol["sources"]))
    for gate in json.loads((root / "gate-evidence-index.json").read_text())["tasks"]:
        for kind, expected in (("reference", True), ("empty", False)):
            path = Path(gate[kind + "_report"])
            if sha(path) != gate[kind + "_sha256"]:
                raise ValueError("reference/empty evidence changed")
            if json.loads(path.read_text())[gate["instance_id"]]["resolved"] is not expected:
                raise ValueError("reference/empty gate failed")
    return cast("dict[str, Any]", protocol)


def write_report(root: Path, report: dict[str, Any], selection: SmokeSelection) -> None:
    report["arms"] = {
        strategy: summarize_arm(
            [row for row in report["results"] if row["strategy"] == strategy],
            selection.total_tokens,
        )
        for strategy in STRATEGIES
    }
    report["updated_at"] = datetime.now(UTC).isoformat()
    write_json(root / "report.json", report)
    lines = [
        "# Context策略：固定Lite20运行进度",
        "",
        f"状态：{report['status']}。每种20题、一次尝试；尚未评分和失败均保留在声明分母。",
        "",
        "| 策略 | 已尝试/20 | 已评分/20 | 确认通过/20 | 已知Token | 未知用量调用 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy, summary in report["arms"].items():
        lines.append(
            f"| {strategy} | {summary['attempted']} | {summary['graded']} | "
            f"{summary['passed']} | {summary['known_tokens']:,} | "
            f"{summary['unknown_usage_calls']} |"
        )
    lines += [
        "",
        "Full History历史结果11/20；服务时段不同，只有每题一次执行，不能据此宣称通用机制收益。",
        "当前保留原Token估算实现及其reasoning字段遗漏；没有同时修复第二个机制。",
        "检索读取在当前任务容器中经过Tool Runtime，计入工具/墙钟预算与Trace；模型仍只见shell/git。",
        "完整实际消耗与相关预算约束在用量未知时保持未知；未知任务的名义预留不代表已证实上界。",
        "本机证据：`benchmark-results/swebench-context/v1/`。",
    ]
    (root / "progress.md").write_text("\n".join(lines) + "\n")


def trace_usage(path: Path) -> dict[str, int]:
    events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    calls = [
        event["data"]
        for event in events
        if event["name"] in {"model_call_completed", "model_call_failed"}
    ]
    known = [call for call in calls if call.get("usage_known")]
    started = sum(event["name"] == "model_call_started" for event in events)
    return {
        "tokens": sum(call["input_tokens"] + call["output_tokens"] for call in known),
        "input_tokens": sum(call["input_tokens"] for call in known),
        "output_tokens": sum(call["output_tokens"] for call in known),
        "cached_input_tokens": sum(call.get("cached_input_tokens", 0) for call in known),
        "unknown_usage_calls": max(0, started - len(known)),
    }


async def run_case(  # noqa: PLR0913, PLR0917 -- explicit case execution dependencies
    root: Path,
    row: dict[str, Any],
    selection: SmokeSelection,
    protocol: dict[str, Any],
    credentials: Path,
    allowance: int,
) -> None:
    task = next(task for task in selection.tasks if task.instance_id == row["instance_id"])
    directory = root / "runs" / row["strategy"] / task.instance_id
    if "repeat" in row:
        directory /= f"repeat-{row['repeat']}"
    directory.mkdir(parents=True, exist_ok=False)
    row.update(status="running", run_id=uuid.uuid4().hex, evidence_directory=str(directory))
    write_json(directory / "attempt.json", {**row, "protocol_sha256": sha(root / "protocol.json")})
    image = json.loads((root / "images.json").read_text())[task.instance_id]
    budget = AgentBudget(
        max_steps=selection.max_steps,
        max_tokens=min(selection.task_tokens, allowance),
        max_tool_calls=selection.max_tool_calls,
        max_duration_seconds=selection.max_seconds,
    )
    model = UsageModel(provider_factory(credentials, selection)())
    try:
        async with DockerSandbox(DockerHost(), image=image, resources=RESOURCES) as box:
            await record_isolation(box, directory / "isolation.json")
            runner = box.runner("/testbed", CommandPolicy(COMMANDS))
            runtime = build_local_runtime(
                workspace=directory,
                model=model,
                budget=budget,
                options=strategy_options(selection, row["strategy"]),
                components=RuntimeComponents(
                    runner=runner,
                    tools=(ShellTool(runner), GitTool(runner)),
                    files=ContainerContextFiles(runner),
                ),
            )
            runtime.events.subscribe("*", JsonlTraceWriter(directory / "trace.jsonl"))
            result = await runtime.agent.run(
                (
                    Path(protocol["sources"]) / "tasks" / task.instance_id / "problem_statement.md"
                ).read_text(),
                run_id=row["run_id"],
            )
            row.update(
                status=result.status.value,
                steps=result.steps,
                tool_calls=result.tool_calls,
                duration_ms=result.duration_ms,
                failure_reason=result.failure_reason,
                nominal_task_tokens=budget.max_tokens,
            )
            row.update(trace_usage(directory / "trace.jsonl"))
            row["token_overrun"] = max(0, row["tokens"] - budget.max_tokens)
            patch = directory / "model.patch"
            try:
                await _capture_patch(box, "/testbed", task.base_commit, patch)
            except PublicStudyError:
                diagnostic = await box.execute(
                    ("git", "status", "--porcelain"), cwd="/testbed", timeout_seconds=15
                )
                write_json(
                    directory / "capture-diagnostic.json",
                    {"exit_code": diagnostic.exit_code, "stderr": diagnostic.stderr[:4000]},
                )
                raise
        row["success"] = await grade_patch(
            task,
            image=image,
            sources=Path(protocol["sources"]),
            patch=patch,
            output=directory / "verifier",
            evaluator=Path(protocol["evaluator"]),
            timeout_seconds=selection.verification_timeout_seconds,
        )
        row["evaluation_status"] = "graded"
        write_json(
            directory / "sidecar.json",
            {
                "schema_version": "1.0",
                "case_id": task.instance_id,
                "strategy": row["strategy"],
                "run_id": row["run_id"],
                "criterion_source": selection.official_commit,
                "task_digest": task.task_digest,
                "patch_sha256": sha(patch),
                "resolved": row["success"],
                "protocol_sha256": sha(root / "protocol.json"),
            },
        )
    except Exception as error:
        if row["status"] == "running":
            row["status"] = "failed"
        row.update(trace_usage(directory / "trace.jsonl"))
        row.update(
            evaluation_status="infrastructure_ungraded",
            infrastructure_error=type(error).__name__,
        )
    finally:
        write_json(directory / "result.json", row)


async def reader_probe(root: Path, selection: SmokeSelection) -> None:
    """Verify real container reads, path denial and budgeted Snapshot retrieval offline."""
    images = json.loads((root / "images.json").read_text())
    async with DockerSandbox(
        DockerHost(), image=images[selection.tasks[0].instance_id], resources=RESOURCES
    ) as box:
        await record_isolation(box, root / "preflight/reader-isolation.json")
        runner = box.runner("/testbed", CommandPolicy(COMMANDS))
        backend = ContainerContextFiles(runner)
        listing = await backend.invoke("list_files", {"max_results": 5})
        if not isinstance(listing, dict) or not listing.get("files"):
            raise ValueError("container context listing empty")
        paths = cast("list[str]", listing["files"])
        content = await backend.invoke("read_file", {"path": paths[0], "max_chars": 40})
        if not isinstance(content, dict) or not isinstance(content.get("content"), str):
            raise ValueError("container context read failed")
        for path in ("../etc/passwd", "/etc/passwd", ".env"):
            try:
                await backend.invoke("read_file", {"path": path})
            except ToolError:
                pass
            else:
                raise ValueError("context boundary probe failed")
        model = ScriptedModel([ModelResponse("offline probe complete")])
        session = root / "preflight/session"
        await asyncio.to_thread(session.mkdir, parents=True)
        runtime = build_local_runtime(
            workspace=session,
            model=model,
            budget=AgentBudget(max_steps=2, max_tokens=100_000, max_tool_calls=30),
            options=strategy_options(selection, "retrieval"),
            components=RuntimeComponents(
                runner=runner, tools=(ShellTool(runner), GitTool(runner)), files=backend
            ),
        )
        runtime.events.subscribe("*", JsonlTraceWriter(root / "preflight/reader-trace.jsonl"))
        result = await runtime.agent.run("Find django setup configuration", run_id="context_probe")
        if result.tool_calls <= 1 or result.status.value != "finished":
            raise ValueError("snapshot retrieval did not execute through tools")
        if {tool.name for tool in runtime.tools.definitions()} != {"shell", "git"}:
            raise ValueError("context tools leaked into model tool definitions")
        write_json(
            root / "preflight/reader-result.json",
            {"real_container_reads": True, "denial_probes": 3, "tool_calls": result.tool_calls},
        )


async def prepare(root: Path, project: Path, evaluator: Path) -> None:
    baseline_root = project / "benchmark-results/swebench-lite/v1/baseline"
    baseline = json.loads((baseline_root / "study-report.json").read_text())
    source_hash = frozen_tree_digest(project / "src/myagentrun")
    if source_hash != baseline["source_hash"]:
        raise ValueError("runtime differs from historical baseline; declare a new protocol")
    selection_path = project / "benchmarks/swebench_lite/v1/selection.json"
    selection = SmokeSelection.model_validate_json(selection_path.read_text())
    sources = project / "benchmark-results/swebench-lite/v1/sources"
    selection.verify(sources)
    if len(selection.tasks) != TASK_COUNT or not await asyncio.to_thread(evaluator.is_file):
        raise ValueError("fixed20 and independent evaluator required")
    await asyncio.to_thread(root.mkdir, parents=True, exist_ok=False)
    snapshot = root / "reproduction"
    shutil.copytree(project / "src/myagentrun", snapshot / "src/myagentrun")
    relative = Path("benchmarks/swebench_context/v1/run.py")
    (snapshot / relative).parent.mkdir(parents=True)
    shutil.copy2(project / relative, snapshot / relative)
    helper = Path("benchmarks/swebench_smoke/v1/official_bridge.py")
    (snapshot / helper).parent.mkdir(parents=True)
    shutil.copy2(project / helper, snapshot / helper)
    shutil.copy2(project / "myagentrun.toml", snapshot / "myagentrun.toml")
    shutil.copy2(selection_path, root / "selection.json")
    shutil.copy2(baseline_root / "images.json", root / "images.json")
    gates = json.loads((baseline_root / "gate-evidence-index.json").read_text())
    for gate in gates["tasks"]:
        for kind in ("reference", "empty"):
            gate[kind + "_report"] = str((project / gate[kind + "_report"]).resolve())
    write_json(root / "gate-evidence-index.json", gates)
    profile = configured_profile(project / "myagentrun.toml")
    protocol = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "strategies": list(STRATEGIES),
        "declared_runs": len(selection.tasks) * len(STRATEGIES),
        "repeats_per_task_strategy": 1,
        "model": profile.model,
        "endpoint": profile.endpoint,
        "temperature": 0,
        "selection_sha256": sha(root / "selection.json"),
        "images_sha256": sha(root / "images.json"),
        "gate_index_sha256": sha(root / "gate-evidence-index.json"),
        "source_hash": source_hash,
        "reproduction_hash": frozen_tree_digest(snapshot),
        "historical_report": str(baseline_root / "study-report.json"),
        "historical_report_sha256": sha(baseline_root / "study-report.json"),
        "sources": str(sources),
        "evaluator": str(evaluator),
        "project_root": str(project),
        "arm_tokens": selection.total_tokens,
        "experiment_nominal_tokens": selection.total_tokens * len(STRATEGIES),
        "strategy_parameters": {
            "sliding_window": 20,
            "recent_messages": 12,
            "summary_max_chars": 4000,
        },
        "unknown_policy": (
            "retain unknown; reserve one nominal task allowance; continue other tasks; "
            "actual total and compliance unknown"
        ),
        "scheduling": (
            "original task order; arm order rotated left by original task index mod6; "
            "one container at a time"
        ),
        "runtime_changes": "none; keep original reasoning estimate omission; strategy only",
        "retrieval_bridge": (
            "original file worker, read-only operations, root/testbed; "
            "postponed annotations and Path.is_relative_to shim for Python3.8"
        ),
        "retrieval_budget": (
            "native SnapshotRetriever: list<=80 files, read<=24*4000chars per step, top5; "
            "all calls charged"
        ),
        "limits": (
            "single repeat, historical FullHistory, service drift; "
            "not a general mechanism-benefit claim"
        ),
    }
    write_json(root / "protocol.json", protocol)
    (root / "protocol.sha256").write_text(sha(root / "protocol.json") + "\n")
    verify_protocol(root)
    await reader_probe(root, selection)
    report = {
        "status": "prepared",
        "protocol": protocol,
        "results": schedule([t.instance_id for t in selection.tasks]),
    }
    write_report(root, report, selection)
    print(
        {
            "prepared": str(root),
            "declared_runs": protocol["declared_runs"],
            "offline_probe": "passed",
        }
    )


async def run(root: Path, credentials: Path) -> None:
    protocol = verify_protocol(root)
    selection = SmokeSelection.model_validate_json((root / "selection.json").read_text())
    if not (root / "preflight/reader-result.json").is_file():
        raise ValueError("offline real-container bridge preflight required")
    report = json.loads((root / "report.json").read_text())
    with (root / "run_started.json").open("x") as claim:
        json.dump({"pid": os.getpid(), "protocol_sha256": sha(root / "protocol.json")}, claim)
    os.chdir(root / "reproduction")
    provider_factory(credentials, selection)  # Validate private credentials without a model call.
    report["status"] = "running"
    write_report(root, report, selection)
    for row in report["results"]:
        arm = [other for other in report["results"] if other["strategy"] == row["strategy"]]
        allowance = remaining_allowance(
            arm, total=selection.total_tokens, per_task=selection.task_tokens
        )
        if not allowance:
            row["failure_reason"] = "arm_nominal_allowance_exhausted"
            write_report(root, report, selection)
            continue
        verify_protocol(root)
        row["status"] = "running"
        write_report(root, report, selection)
        await run_case(root, row, selection, protocol, credentials, allowance)
        write_report(root, report, selection)
        print(
            {
                k: row.get(k)
                for k in (
                    "strategy",
                    "instance_id",
                    "status",
                    "success",
                    "tokens",
                    "unknown_usage_calls",
                    "infrastructure_error",
                )
            },
            flush=True,
        )
    report["status"] = "execution_complete"
    write_report(root, report, selection)
    print({"completed": True, "arms": report["arms"]}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, default=Path("benchmark-results/swebench-context/v1"))
    parser.add_argument("--evaluator", type=Path)
    parser.add_argument(
        "--credentials-file", type=Path, default=Path(".myagentrun/credentials.toml")
    )
    args = parser.parse_args()
    root, credentials = args.root.resolve(), args.credentials_file.resolve()
    if args.action == "prepare":
        if args.evaluator is None:
            parser.error("prepare requires --evaluator")
        asyncio.run(prepare(root, Path.cwd().resolve(), evaluator_path(args.evaluator)))
    else:
        asyncio.run(run(root, credentials))


if __name__ == "__main__":
    main()
