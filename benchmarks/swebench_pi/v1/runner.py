# ruff: noqa: T201 -- bounded experiment summaries only, never credentials
"""Pi 1.5.0 vs historical MyAgentRun Lite20; same model, tasks, images and budgets."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import shutil
import signal
import tomllib
import uuid
from pathlib import Path
from typing import Any

from myagentrun.benchmark.hashes import frozen_tree_digest
from myagentrun.benchmark.public_swe import _capture_patch, write_json
from myagentrun.benchmark.swebench_smoke import (
    SmokeSelection,
    grade_patch,
    record_isolation,
)
from myagentrun.infrastructure.containers import ContainerResources, DockerHost, DockerSandbox

BASELINE = Path("benchmark-results/swebench-lite/v1/baseline")
SOURCES = BASELINE.parent / "sources"
SCRIPTS = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path("benchmark-results/pi-comparison/v1")
RESOURCES = ContainerResources(platform="linux/arm64", memory_mb=1024, cpus=1, pids_limit=128)
EVALUATOR = Path("/private/tmp/myagentrun-swebench-evaluator-20261116/bin/python")
TASK_COUNT = 20
MAX_REQUEST_BYTES = 80000


def credential() -> str:
    """Only the trusted host reads this key; it goes to a private child stdin pipe."""
    return tomllib.loads(Path(".myagentrun/credentials.toml").read_text())["credentials"][
        "MYAGENTRUN_MOEROUTER_API_KEY"
    ]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep the full denominator, unknown consumption and failures."""
    known = sum(row.get("tokens", 0) for row in rows)
    unknown = sum(row.get("unknown_usage_calls", 0) for row in rows)
    return {
        "declared_tasks": len(rows),
        "graded_tasks": sum(isinstance(r["success"], bool) for r in rows),
        "resolved": sum(r["success"] is True for r in rows),
        "known_tokens": known,
        "actual_total_tokens": None if unknown else known,
        "unknown_usage_calls": unknown,
    }


def trace_usage(events: list[dict[str, Any]]) -> dict[str, int]:
    """Preserve known consumption if an outer worker/transport failure interrupts a case."""
    starts = [e for e in events if e["name"] == "model_call_started"]
    ends = [e["data"] for e in events if e["name"] == "model_call_completed"]
    known = [data for data in ends if data.get("usage_known") is True]
    return {
        "tokens": sum(data["input_tokens"] + data["output_tokens"] for data in known),
        "input_tokens": sum(data["input_tokens"] for data in known),
        "output_tokens": sum(data["output_tokens"] for data in known),
        "cached_input_tokens": sum(data.get("cached_input_tokens", 0) for data in known),
        "unknown_usage_calls": max(0, len(starts) - len(ends)) + len(ends) - len(known),
    }


def prepare(output: Path, pi_root: Path) -> None:
    """Freeze installed third-party runtime and actual pre-existing environment gates."""
    selection_file = Path("benchmarks/swebench_lite/v1/selection.json")
    selection = SmokeSelection.model_validate_json(selection_file.read_text())
    selection.verify(SOURCES)
    images = json.loads((BASELINE / "images.json").read_text())
    gates = json.loads((BASELINE / "gate-evidence-index.json").read_text())
    assert len(gates["tasks"]) == len(selection.tasks) == TASK_COUNT
    for row in gates["tasks"]:
        assert row["image"] == images[row["instance_id"]]
        for kind, expected in (("reference", True), ("empty", False)):
            path = Path(row[kind + "_report"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row[kind + "_sha256"]
            assert json.loads(path.read_text())[row["instance_id"]]["resolved"] is expected
    package = json.loads((pi_root / "package.json").read_text())
    assert package["name"] == "@earendil-works/pi-coding-agent"
    assert package["version"] == "1.5.0"
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "runtime"
    runtime.mkdir()
    shutil.copytree(pi_root, runtime / "pi", symlinks=False)
    shutil.copy2(Path(shutil.which("node") or "missing-node").resolve(), runtime / "node")
    (runtime / "eval").mkdir()
    for name in ("worker.mjs", "policy.mjs", "remote_io.py", "runner.py"):
        shutil.copy2(SCRIPTS / name, runtime / "eval" / name)
    shutil.copy2(selection_file, output / "selection.json")
    shutil.copy2(BASELINE / "images.json", output / "images.json")
    shutil.copy2(BASELINE / "gate-evidence-index.json", output / "gate-evidence-index.json")
    write_json(
        output / "protocol.json",
        {
            "schema_version": "1.0",
            "comparison": "pi_vs_historical_myagentrun_lite20",
            "pi_package": package["name"],
            "pi_version": package["version"],
            "model": "glm-5.3-flash",
            "endpoint": "https://moerouter.com/v1/chat/completions",
            "temperature": 0,
            "thinking_parameter": "not_sent",
            "transport": "sse",
            "task_tokens": selection.task_tokens,
            "total_tokens": selection.total_tokens,
            "max_steps": selection.max_steps,
            "max_tools": selection.max_tool_calls,
            "max_seconds": selection.max_seconds,
            "max_output_tokens": 16384,
            "context_window": 65536,
            "context_policy": "full_history",
            "compaction": False,
            "retry": False,
            "cache_warming": False,
            "tools": ["read", "bash", "edit", "write"],
            "provider_timeout_ms": 300000,
            "resources": {"cpu": 1, "memory_mb": 1024, "pids": 128},
            "pi_runtime_sha256": frozen_tree_digest(runtime / "pi"),
            "node_sha256": hashlib.sha256((runtime / "node").read_bytes()).hexdigest(),
            "eval_sha256": frozen_tree_digest(runtime / "eval"),
            "selection_sha256": hashlib.sha256(selection_file.read_bytes()).hexdigest(),
            "historical_report": str(BASELINE / "study-report.json"),
            "historical_report_sha256": hashlib.sha256(
                (BASELINE / "study-report.json").read_bytes()
            ).hexdigest(),
            "user_scope": "only Pi20; compare historical 11/20; service-state difference retained",
            "limits": [
                "different native toolsets/prompts",
                "historical service outage",
                "single attempt per task",
                "no mechanism attribution",
            ],
        },
    )
    print({"prepared": str(output), "tasks": 20, "pi_version": package["version"]})


async def pi_case(  # noqa: PLR0913, PLR0917, PLR0915 -- explicit evaluator boundary
    box: DockerSandbox,
    output: Path,
    root: Path,
    task: str,
    tokens: int,
    seconds: float,
    run_id: str,
) -> dict[str, Any]:
    """Run unmodified Pi session; bridge native tool operations into an owned container."""
    runtime = root / "runtime"
    for name in ("state", "temp"):
        (output / name).mkdir()
    await box.host.run(
        ("docker", "cp", str(runtime / "eval/remote_io.py"), box.name + ":/opt/pi-eval-io.py")
    )
    process = await asyncio.create_subprocess_exec(
        str((runtime / "node").resolve()),
        str((runtime / "eval/worker.mjs").resolve()),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PI_OFFLINE": "1",
            "PI_TELEMETRY": "0",
            "PI_CODING_AGENT_DIR": str((output / "state").resolve()),
            "TMPDIR": str((output / "temp").resolve()),
        },
        start_new_session=True,
        limit=16 * 1024 * 1024,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    key = credential()
    configuration = {
        "piRoot": str((runtime / "pi").resolve()),
        "stateDir": str((output / "state").resolve()),
        "tempDir": str((output / "temp").resolve()),
        "runId": run_id,
        "apiKey": key,
        "task": task,
        "budget": {"maxTokens": tokens, "maxSteps": 200, "maxTools": 400},
        "maxSeconds": seconds,
    }
    process.stdin.write((json.dumps(configuration) + "\n").encode())
    await process.stdin.drain()
    result: dict[str, Any] | None = None
    bridge_tasks: set[asyncio.Task[None]] = set()
    write_lock = asyncio.Lock()

    async def bridge(request: dict[str, Any]) -> None:
        try:
            body = json.dumps(request).encode()
            if len(body) > MAX_REQUEST_BYTES:
                raise ValueError("tool_request_exceeds_80000_byte_limit")
            payload = base64.b64encode(body).decode()
            timeout = min(float(request.get("timeout") or 120), seconds) + 15
            call = await box.execute(
                ("python", "/opt/pi-eval-io.py", payload), timeout_seconds=timeout
            )
            response = json.loads(call.stdout)
        except Exception as error:
            response = {"ok": False, "error": type(error).__name__ + ": " + str(error)[:200]}
        response["id"] = request["id"]
        async with write_lock:
            if process.returncode is None:
                process.stdin.write((json.dumps(response) + "\n").encode())
                await process.stdin.drain()

    async def stderr() -> None:
        content = bytearray()
        while chunk := await process.stderr.read(65536):
            if len(content) < 1024 * 1024:
                content.extend(chunk[: 1024 * 1024 - len(content)])
        (output / "worker.stderr").write_text(
            content.decode(errors="replace").replace(key, "[REDACTED]")
        )

    error_task = asyncio.create_task(stderr())
    try:
        async with asyncio.timeout(seconds + 30):
            with (output / "trace.jsonl").open("x") as trace:
                while line := await process.stdout.readline():
                    item = json.loads(line)
                    if item["type"] == "io":
                        call = asyncio.create_task(bridge(item))
                        bridge_tasks.add(call)
                        call.add_done_callback(bridge_tasks.discard)
                    elif item["type"] == "event":
                        trace.write(json.dumps(item).replace(key, "[REDACTED]") + "\n")
                        trace.flush()
                    elif item["type"] == "result":
                        result = item
                        process.stdin.close()
                await process.wait()
        await asyncio.gather(*bridge_tasks, return_exceptions=True)
        await error_task
    finally:
        if process.returncode is None:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
        for call in bridge_tasks:
            call.cancel()
        await asyncio.gather(*bridge_tasks, return_exceptions=True)
        if not error_task.done():
            await error_task
    if result is None:
        raise RuntimeError("Pi worker did not produce a terminal result; inspect bounded stderr")
    return result


async def run(output: Path, *, probe: bool = False) -> None:
    selection = SmokeSelection.model_validate_json((output / "selection.json").read_text())
    selection.verify(SOURCES)
    protocol = json.loads((output / "protocol.json").read_text())
    assert frozen_tree_digest(output / "runtime/eval") == protocol["eval_sha256"]
    assert frozen_tree_digest(output / "runtime/pi") == protocol["pi_runtime_sha256"]
    images = json.loads((output / "images.json").read_text())
    if probe:
        number = await asyncio.to_thread(lambda: len(list(output.glob("probe*"))))
        directory = output / ("probe" if not (output / "probe").exists() else f"probe-{number}")
        directory.mkdir(exist_ok=False)
        async with DockerSandbox(
            DockerHost(), image=images[selection.tasks[0].instance_id], resources=RESOURCES
        ) as box:
            await record_isolation(box, directory / "isolation.json")
            result = await pi_case(
                box,
                directory,
                output,
                "In /testbed create pi_probe.txt containing exactly 'pi works' using write. "
                "Then read it using read, change works to ready using edit, "
                "and use bash to print it. "
                "Finish after all four tools succeed.",
                20000,
                300,
                uuid.uuid4().hex,
            )
            check = await box.execute(
                (
                    "python",
                    "-c",
                    "from pathlib import Path; "
                    "assert Path('/testbed/pi_probe.txt').read_text() == 'pi ready'",
                )
            )
            result["independent_check"] = check.exit_code == 0
            write_json(directory / "result.json", result)
            print(
                {
                    k: result.get(k)
                    for k in ("status", "tokens", "unknown_usage_calls", "independent_check")
                }
            )
        return
    with (output / "run_started.json").open("x") as claim:
        json.dump(
            {
                "protocol_sha256": hashlib.sha256(
                    (output / "protocol.json").read_bytes()
                ).hexdigest()
            },
            claim,
        )
    rows = [
        {"instance_id": t.instance_id, "status": "not_run", "success": None, "tokens": 0}
        for t in selection.tasks
    ]
    report = {"schema_version": "1.0", "harness": "pi", "protocol": protocol, "results": rows}
    stopped = False
    for task, row in zip(selection.tasks, rows, strict=True):
        known = aggregate(rows)["known_tokens"]
        if stopped or known >= selection.total_tokens:
            row["failure_reason"] = "batch_stopped"
            continue
        directory = output / "runs" / task.instance_id
        directory.mkdir(parents=True, exist_ok=False)
        row.update({"run_id": uuid.uuid4().hex, "status": "running"})
        report["aggregate"] = aggregate(rows)
        write_json(output / "report.json", report)
        try:
            async with DockerSandbox(
                DockerHost(), image=images[task.instance_id], resources=RESOURCES
            ) as box:
                await record_isolation(box, directory / "isolation.json")
                row.update(
                    await pi_case(
                        box,
                        directory,
                        output,
                        (SOURCES / "tasks" / task.instance_id / "problem_statement.md").read_text(),
                        min(selection.task_tokens, selection.total_tokens - known),
                        selection.max_seconds,
                        row["run_id"],
                    )
                )
                stopped |= row.get("unknown_usage_calls", 0) > 0
                patch = directory / "model.patch"
                await _capture_patch(box, "/testbed", task.base_commit, patch)
            row["success"] = await grade_patch(
                task,
                image=images[task.instance_id],
                sources=SOURCES,
                patch=patch,
                output=directory / "verifier",
                evaluator=EVALUATOR,
                timeout_seconds=selection.verification_timeout_seconds,
            )
            write_json(
                directory / "sidecar.json",
                {
                    "case_id": task.instance_id,
                    "run_id": row["run_id"],
                    "task_digest": task.task_digest,
                    "criterion_source": selection.official_commit,
                    "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
                    "resolved": row["success"],
                },
            )
        except Exception as error:
            trace_path = directory / "trace.jsonl"
            events = (
                [json.loads(line) for line in trace_path.read_text().splitlines()]
                if trace_path.exists()
                else []
            )
            row.update(
                {
                    "status": "failed",
                    **trace_usage(events),
                    "failure_reason": type(error).__name__ + ": " + str(error)[:300],
                    "infrastructure_error": True,
                }
            )
            stopped = True
        report["aggregate"] = aggregate(rows)
        write_json(output / "report.json", report)
    report["aggregate"] = aggregate(rows)
    write_json(output / "report.json", report)
    print(report["aggregate"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "probe", "run"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pi-root",
        type=Path,
        default=Path(
            "/Users/jennyma/.nvm/versions/node/v22.23.2/lib/node_modules/@earendil-works/pi-coding-agent"
        ),
    )
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output, args.pi_root)
    else:
        asyncio.run(run(args.output, probe=args.action == "probe"))


if __name__ == "__main__":
    main()
