"""Small native Docker SWE-bench smoke run; not an official leaderboard score.

Preparation and official grading dependencies live outside the core runtime.
The source directory contains private scoring data and is never mounted into
the agent. Run ``python -m myagentrun.benchmark.swebench_smoke --help``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import tomllib
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from myagentrun.adapters import JsonlTraceWriter
from myagentrun.adapters.models import CredentialRedactingModel, OpenAICompatibleModel
from myagentrun.adapters.profiles import ModelProfile
from myagentrun.adapters.streaming import close_stream
from myagentrun.benchmark.hashes import frozen_tree_digest
from myagentrun.benchmark.public_swe import (
    PublicStudyError,
    UsageModel,
    _capture_patch,
    run_checked,
    write_json,
)
from myagentrun.composition import RuntimeComponents, RuntimeOptions, build_local_runtime
from myagentrun.config import DEFAULT_CONFIG_NAME, resolve_config
from myagentrun.domain import AgentBudget, ModelCallError, ModelRequest, ModelResponse
from myagentrun.infrastructure.containers import ContainerResources, DockerHost, DockerSandbox
from myagentrun.infrastructure.process import CommandPolicy
from myagentrun.ports import Model, StreamingModel
from myagentrun.tools import GitTool, ShellTool

_COMMANDS = frozenset(
    {"python", "python3", "pytest", "git", "grep", "find", "ls", "cat", "sed", "head", "tail", "wc"}
)
_RESOURCES = ContainerResources(platform="linux/arm64", memory_mb=1024, cpus=1, pids_limit=128)
_PROMPT = (
    "You are a coding agent working in /testbed. Resolve the supplied issue. "
    "Use shell with argv arrays; no shell string interpreter is available. "
    "Use python to read/search files and atomically edit using os.replace. "
    "Network is disabled; dependencies are installed. Run relevant existing tests. "
    "Do not commit, alter Git metadata or modify tests to hide a defect. "
    "Finish with a brief explanation."
)
_HELPER = Path(__file__).resolve().parents[3] / "benchmarks/swebench_smoke/v1/official_bridge.py"
_EVAL_LOG_LIMIT = 128 * 1024 * 1024


def configured_profile(config: Path = Path(DEFAULT_CONFIG_NAME)) -> ModelProfile:
    """Read explicit non-secret model settings; never fall back to a paid provider."""
    if not config.is_file():
        raise ValueError("explicit project model configuration is required")
    settings = resolve_config(config_path=config, environ={}).model_options
    if not all(settings.get(name) for name in ("model", "endpoint", "api_key_env")):
        raise ValueError("model configuration requires model, endpoint and api_key_env")
    return ModelProfile(
        name="openai-compatible",
        model=settings["model"],
        endpoint=settings["endpoint"],
        api_key_env=settings["api_key_env"],
    )


class SmokeTask(BaseModel):
    """A frozen public task identity, with no hidden labels or reference patch."""

    model_config = ConfigDict(extra="forbid")
    instance_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+__[a-zA-Z0-9_-]+-\d+$")
    repo: str = Field(pattern=r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")
    base_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    task_digest: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")


class SmokeSelection(BaseModel):
    """Protocol locked before the first model call."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"]
    official_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    task_repo_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    environment: Literal["custom_native_arm64_python39", "custom_native_arm64"]
    context_input_budget: int = Field(default=8192, gt=0)
    model_transport: Literal["json", "sse"] = "json"
    model_timeout_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    max_output_tokens: int | None = Field(default=None, gt=0)
    max_stream_bytes: int = Field(default=1_048_576, gt=0, le=64 * 1024 * 1024)
    verification_timeout_seconds: float = Field(default=180, gt=0, allow_inf_nan=False)
    total_tokens: int = Field(gt=0)
    task_tokens: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    max_tool_calls: int = Field(gt=0)
    max_seconds: float = Field(gt=0, allow_inf_nan=False)
    tasks: list[SmokeTask] = Field(min_length=1)

    def verify(self, sources: Path) -> None:
        """Check complete original task trees before every run and grade."""
        if len({task.instance_id for task in self.tasks}) != len(self.tasks):
            raise PublicStudyError("duplicate smoke tasks")
        for task in self.tasks:
            if frozen_tree_digest(sources / "tasks" / task.instance_id) != task.task_digest:
                raise PublicStudyError("source mismatch: " + task.instance_id)


def runtime_options(selection: SmokeSelection) -> RuntimeOptions:
    """Use the frozen context limit as well as the four execution budgets."""
    return RuntimeOptions(
        builtin_tools_enabled=False,
        allow_shell=True,
        normalize_tool_results=True,
        error_recovery_enabled=False,
        artifact_offload=False,
        tool_max_retries=0,
        system_prompt=_PROMPT,
        context_input_budget=selection.context_input_budget,
    )


class StreamedCompletion:
    """Buffer the existing SSE adapter's final response at the model port.

    SSE keeps the router connection active during long generations. The normal
    Loop still sees exactly one model response and accounts its reported usage.
    No implicit retries or partial tool execution are introduced.
    """

    def __init__(self, model: StreamingModel) -> None:
        self.model = model

    async def complete(self, request: ModelRequest) -> ModelResponse:
        stream = self.model.stream(request)
        final: ModelResponse | None = None
        try:
            async for event in stream:
                if event.response is not None:
                    final = event.response
        finally:
            await close_stream(stream)
        if final is None:
            raise ModelCallError("SSE returned no final response", code="stream_incomplete")
        return final


def evaluation_script(original: str, completion: str) -> str:
    """Adapt only conda activation and offline install; retain official tests.

    The completion token and real test exit code are appended by the trusted
    evaluator, never by the model. Original and derived scripts are retained.
    """
    start = ": '>>>>> Start Test Output'"
    end = ": '>>>>> End Test Output'"
    if original.count(start) != 1 or original.count(end) != 1:
        raise ValueError("expected exactly one pair of official test markers")
    if original.index(start) >= original.index(end):
        raise ValueError("test markers are out of order")
    lines = [
        line
        for line in original.splitlines()
        if line not in {"source /opt/miniconda3/bin/activate", "conda activate testbed", "git show"}
    ]
    script = "\n".join(lines) + "\n"
    script = script.replace(
        "python -m pip install -e .\n",
        "python -m pip install --no-build-isolation --no-deps -e . || exit 86\n",
    )
    script = script.replace(start, "set +x\nprintf '%s\\n' '>>>>> Start Test Output'")
    script = script.replace(
        end,
        "test_exit=$?\nprintf '%s\\n' '>>>>> End Test Output'\n"
        "printf '>>>>> Test Exit Code: %s\\n' \"$test_exit\"",
    )
    return script + f"\nprintf '%s\\n' '{completion}'\n"


def save_evaluation_log(directory: Path, raw: bytes) -> Path:
    """Retain raw test bytes and generate a bounded UTF-8 view for official parsers."""
    if len(raw) > _EVAL_LOG_LIMIT:
        raise PublicStudyError("independent test log exceeded 128 MiB limit")
    try:
        text = raw.decode("utf-8")
        valid = True
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
        valid = False
    # Preserve multiline unittest descriptions and additionally expose their
    # actual canonical test IDs to official parsers. No outcome is inferred
    # without a completed status line; raw bytes remain the fact source.
    lines: list[str] = []
    pending: str | None = None
    in_tests = False
    added = 0
    for line in text.splitlines(keepends=True):
        lines.append(line)
        stripped = line.strip()
        if stripped == ">>>>> Start Test Output":
            in_tests = True
            pending = None
        elif stripped == ">>>>> End Test Output":
            in_tests = False
            pending = None
        elif in_tests:
            if re.fullmatch(r"test\S* \([^\n]+\)", stripped):
                pending = stripped
            elif re.match(r"test\S* \([^\n]+\) \.\.\. ", stripped):
                pending = None
            elif pending and (
                match := re.search(r" \.\.\. (ok|OK|FAIL|ERROR|skipped(?: .*)?)$", stripped)
            ):
                lines.append(pending + " ... " + match.group(1) + "\n")
                added += 1
                pending = None
    text = "".join(lines)
    (directory / "test_output.raw").write_bytes(raw)
    path = directory / "test_output.txt"
    path.write_text(text, encoding="utf-8")
    write_json(
        directory / "log-metadata.json",
        {
            "raw_size_bytes": len(raw),
            "raw_utf8_valid": valid,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "normalized_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "normalization": "utf-8 with replacement of undecodable bytes; raw retained",
            "unittest_multiline_status_lines_added": added,
            "normalization_version": "utf8_unittest_multiline_v1",
        },
    )
    return path


def summarize(rows: list[dict[str, object]], nominal_tokens: int) -> dict[str, object]:
    """Keep missing evidence unknown and retain the entire declared denominator."""
    resolved = sum(row.get("success") is True for row in rows)
    graded = sum(isinstance(row.get("success"), bool) for row in rows)
    tokens = sum(int(str(row.get("tokens", 0))) for row in rows)
    return {
        "declared_tasks": len(rows),
        "graded_tasks": graded,
        "resolved": resolved,
        "task_success_rate": resolved / len(rows) if rows else None,
        "evidence_coverage": graded / len(rows) if rows else None,
        "total_tokens": tokens,
        "tokens_per_success": tokens / resolved if resolved else None,
        "nominal_token_budget": nominal_tokens,
        "token_overrun": max(0, tokens - nominal_tokens),
        "unknown_usage_calls": sum(int(str(row.get("unknown_usage_calls", 0))) for row in rows),
    }


def fingerprint(selection: SmokeSelection, images: dict[str, str]) -> str:
    """Lock core code, helper, protocol and immutable image identities."""
    root = Path(__file__).resolve().parents[3]
    helper = root / "benchmarks/swebench_smoke/v1/official_bridge.py"
    body = selection.model_dump_json() + json.dumps(images, sort_keys=True)
    body += frozen_tree_digest(root / "src/myagentrun") + helper.read_text()
    body += json.dumps(configured_profile().describe(), sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


async def record_isolation(box: DockerSandbox, output: Path) -> None:
    """Inspect actual Docker isolation before any task code or model call."""
    raw = await run_checked(box.host, ("docker", "inspect", box.name))
    info = json.loads(raw)[0]
    config = info["HostConfig"]
    if (
        info["Mounts"]
        or config["Privileged"]
        or config["NetworkMode"] != "none"
        or config["CapDrop"] != ["ALL"]
        or "no-new-privileges" not in config["SecurityOpt"]
        or config["Memory"] != _RESOURCES.memory_mb * 1024 * 1024
        or config["NanoCpus"] != _RESOURCES.cpus * 1_000_000_000
        or config["PidsLimit"] != _RESOURCES.pids_limit
    ):
        raise PublicStudyError("actual container isolation differs from frozen protocol")
    write_json(
        output,
        {
            "container": box.name,
            "image": info["Image"],
            "mounts": info["Mounts"],
            "network": config["NetworkMode"],
            "privileged": config["Privileged"],
            "cap_drop": config["CapDrop"],
            "security_opt": config["SecurityOpt"],
            "memory_bytes": config["Memory"],
            "nano_cpus": config["NanoCpus"],
            "pids_limit": config["PidsLimit"],
        },
    )


async def grade_patch(  # noqa: PLR0913 - explicit independent scoring boundary
    task: SmokeTask,
    *,
    image: str,
    sources: Path,
    patch: Path,
    output: Path,
    evaluator: Path,
    timeout_seconds: float = 180,
) -> bool:
    """Apply captured patch in a fresh offline container and use official grader."""
    await asyncio.to_thread(output.mkdir, parents=True, exist_ok=False)
    directory = sources / "tasks" / task.instance_id
    original = (directory / "eval.sh").read_text()
    completion = "myagentrun_eval_completed_" + uuid.uuid4().hex
    script = output / "eval.sh"
    script.write_text(evaluation_script(original, completion))
    (output / "original_eval.sh").write_text(original)
    host = DockerHost()
    async with DockerSandbox(host, image=image, resources=_RESOURCES) as box:
        await record_isolation(box, output / "isolation.json")
        head = await box.execute(("git", "rev-parse", "HEAD"), cwd="/testbed")
        if head.exit_code or head.stdout.strip() != task.base_commit:
            raise PublicStudyError("initial repository commit mismatch")
        if (await asyncio.to_thread(patch.stat)).st_size:
            await run_checked(host, ("docker", "cp", str(patch), box.name + ":/tmp/model.patch"))
            applied = await box.execute(
                ("git", "apply", "--binary", "/tmp/model.patch"), cwd="/testbed"
            )
            if applied.exit_code or applied.timed_out:
                write_json(output / "report.json", {"failure_type": "patch_apply_failed"})
                return False
        await run_checked(host, ("docker", "cp", str(script), box.name + ":/tmp/eval.sh"))
        result = await box.execute(
            ("bash", "-c", "ulimit -f 131072; bash /tmp/eval.sh > /tmp/test_output.txt 2>&1"),
            cwd="/testbed",
            timeout_seconds=timeout_seconds,
        )
        if result.timed_out or not box.alive:
            raise PublicStudyError("independent evaluation timeout")
        raw_log = output / "test_output.raw"
        await run_checked(host, ("docker", "cp", box.name + ":/tmp/test_output.txt", str(raw_log)))
        raw = await asyncio.to_thread(raw_log.read_bytes)
        log = await asyncio.to_thread(save_evaluation_log, output, raw)
        if completion.encode() not in raw:
            raise PublicStudyError("independent evaluation did not complete")
    process = await asyncio.create_subprocess_exec(
        str(evaluator),
        str(_HELPER),
        str(directory),
        str(log),
        str(patch),
        str(output / "report.json"),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env={key: value for key, value in os.environ.items() if key in {"PATH", "LANG", "HOME"}},
    )
    try:
        await asyncio.wait_for(process.wait(), timeout=60)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise PublicStudyError("official grading helper failed")
    report = json.loads((output / "report.json").read_text())
    if not report[task.instance_id]["patch_successfully_applied"]:
        raise PublicStudyError("official grader found no valid test run")
    return bool(report[task.instance_id]["resolved"])


async def check_environment(
    selection: SmokeSelection, sources: Path, output: Path, evaluator: Path
) -> None:
    """Require every gold patch to resolve and every empty patch to be unresolved."""
    selection.verify(sources)
    await asyncio.to_thread(output.mkdir, parents=True, exist_ok=True)
    images = json.loads((output / "images.json").read_text())
    empty = output / "empty.patch"
    empty.write_text("")
    gates = []
    for task in selection.tasks:
        reference = await grade_patch(
            task,
            image=images[task.instance_id],
            sources=sources,
            patch=sources / "tasks" / task.instance_id / "gold.patch",
            output=output / "gates" / task.instance_id / "reference",
            evaluator=evaluator,
            timeout_seconds=selection.verification_timeout_seconds,
        )
        nop = await grade_patch(
            task,
            image=images[task.instance_id],
            sources=sources,
            patch=empty,
            output=output / "gates" / task.instance_id / "empty",
            evaluator=evaluator,
            timeout_seconds=selection.verification_timeout_seconds,
        )
        gates.append(
            {"instance_id": task.instance_id, "reference_passed": reference, "empty_passed": nop}
        )
        write_json(
            output / "gates.json",
            {
                "fingerprint": fingerprint(selection, images),
                "tasks": gates,
            },
        )
        if not reference or nop:
            raise PublicStudyError("reference/empty gate failed: " + task.instance_id)


async def run_smoke(  # noqa: PLR0915 - single sequential experiment with accounting
    selection: SmokeSelection,
    sources: Path,
    output: Path,
    evaluator: Path,
    model_factory: Callable[[], Model],
) -> dict[str, object]:
    """Run only after complete matching offline gates; preserve all declared cases."""
    selection.verify(sources)
    images = json.loads((output / "images.json").read_text())
    gates = json.loads((output / "gates.json").read_text())
    if gates["fingerprint"] != fingerprint(selection, images) or {
        gate["instance_id"]
        for gate in gates["tasks"]
        if gate["reference_passed"] and not gate["empty_passed"]
    } != {task.instance_id for task in selection.tasks}:
        raise PublicStudyError("complete matching reference/empty gates required")
    # Exclusive claim persists even if interrupted before the first model response.
    with (output / "run_started.json").open("x") as stream:
        json.dump({"fingerprint": gates["fingerprint"], "system_prompt": _PROMPT}, stream)
    rows: list[dict[str, object]] = [
        {"instance_id": t.instance_id, "status": "not_run", "success": None, "tokens": 0}
        for t in selection.tasks
    ]
    report: dict[str, object] = {
        "schema_version": "1.0",
        "configuration": "shell_git_full_history_v1",
        "environment": selection.environment,
        "fingerprint": gates["fingerprint"],
        "tools": ["shell", "git"],
        "skills": False,
        "verification": False,
        "memory": False,
        "model": configured_profile().model,
        "endpoint": configured_profile().endpoint,
        "temperature": 0,
        "thinking": None,
        "context_input_budget": selection.context_input_budget,
        "model_transport": selection.model_transport,
        "model_timeout_seconds": selection.model_timeout_seconds,
        "max_output_tokens": selection.max_output_tokens,
        "max_stream_bytes": selection.max_stream_bytes,
        "system_prompt": _PROMPT,
        "results": rows,
    }
    predictions = output / "predictions.jsonl"
    stopped = False
    for task, row in zip(selection.tasks, rows, strict=True):
        consumed = int(str(summarize(rows, selection.total_tokens)["total_tokens"]))
        if stopped or consumed >= selection.total_tokens:
            row["failure_type"] = "sweep_stopped" if stopped else "aggregate_budget_exhausted"
            continue
        directory = output / "runs" / task.instance_id
        directory.mkdir(parents=True, exist_ok=False)
        row["run_id"] = uuid.uuid4().hex
        row["status"] = "running"
        report["aggregate"] = summarize(rows, selection.total_tokens)
        write_json(output / "report.json", report)
        model = UsageModel(model_factory())
        budget = AgentBudget(
            max_steps=selection.max_steps,
            max_tokens=min(selection.task_tokens, selection.total_tokens - consumed),
            max_tool_calls=selection.max_tool_calls,
            max_duration_seconds=selection.max_seconds,
        )
        try:
            host = DockerHost()
            async with DockerSandbox(
                host, image=images[task.instance_id], resources=_RESOURCES
            ) as box:
                await record_isolation(box, directory / "isolation.json")
                runner = box.runner("/testbed", CommandPolicy(_COMMANDS))
                runtime = build_local_runtime(
                    workspace=directory,
                    model=model,
                    budget=budget,
                    options=runtime_options(selection),
                    components=RuntimeComponents(
                        runner=runner, tools=(ShellTool(runner), GitTool(runner))
                    ),
                )
                runtime.events.subscribe("*", JsonlTraceWriter(directory / "trace.jsonl"))
                result = await runtime.agent.run(
                    (sources / "tasks" / task.instance_id / "problem_statement.md").read_text(),
                    run_id=str(row["run_id"]),
                )
                row.update(
                    {
                        "status": result.status.value,
                        "tokens": result.token_usage.total_tokens,
                        "input_tokens": result.token_usage.input_tokens,
                        "output_tokens": result.token_usage.output_tokens,
                        "cached_input_tokens": result.token_usage.cached_input_tokens,
                        "steps": result.steps,
                        "tool_calls": result.tool_calls,
                        "duration_ms": result.duration_ms,
                        "failure_reason": result.failure_reason,
                        "unknown_usage_calls": model.unknown_calls,
                        "token_overrun": max(
                            0, result.token_usage.total_tokens - budget.max_tokens
                        ),
                    }
                )
                stopped |= model.unknown_calls > 0
                write_json(output / "report.json", report)
                if not box.alive:
                    raise PublicStudyError("agent container stopped before patch capture")
                patch = directory / "model.patch"
                await _capture_patch(box, "/testbed", task.base_commit, patch)
            prediction = {
                "instance_id": task.instance_id,
                "model_name_or_path": "myagentrun_smoke_v1",
                "model_patch": patch.read_text(),
            }
            with predictions.open("a") as stream:
                stream.write(json.dumps(prediction) + "\n")
            row["success"] = await grade_patch(
                task,
                image=images[task.instance_id],
                sources=sources,
                patch=patch,
                output=directory / "verifier",
                evaluator=evaluator,
                timeout_seconds=selection.verification_timeout_seconds,
            )
            write_json(
                directory / "sidecar.json",
                {
                    "schema_version": "1.0",
                    "case_id": task.instance_id,
                    "run_id": row["run_id"],
                    "criterion_source": selection.official_commit,
                    "task_digest": task.task_digest,
                    "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
                    "resolved": row["success"],
                    "evidence": "verifier/report.json",
                },
            )
        except (OSError, PublicStudyError) as exc:
            row.update({"failure_type": "infrastructure_error", "failure_reason": str(exc)})
            stopped = True
        report["aggregate"] = summarize(rows, selection.total_tokens)
        write_json(output / "report.json", report)
    report["aggregate"] = summarize(rows, selection.total_tokens)
    write_json(output / "report.json", report)
    return report


def provider_factory(
    credentials: Path, selection: SmokeSelection | None = None
) -> Callable[[], Model]:
    """Use the frozen project configuration and an explicitly selected private store."""
    profile = configured_profile()
    values = tomllib.loads(credentials.read_text()).get("credentials", {})
    secret = values.get(profile.api_key_env)
    if not isinstance(secret, str) or not secret.strip():
        raise ValueError("selected provider credential is not configured")
    environment = {profile.api_key_env: secret}
    secret = profile.api_key(environment)

    def factory() -> Model:
        provider = OpenAICompatibleModel(
            endpoint=profile.endpoint,
            model=profile.model,
            api_key=secret,
            max_attempts=1,
            require_usage=True,
            timeout_seconds=selection.model_timeout_seconds if selection else 120,
            max_output_tokens=selection.max_output_tokens if selection else None,
            max_stream_bytes=selection.max_stream_bytes if selection else 1_048_576,
        )
        redacted = CredentialRedactingModel(provider, secret)
        return (
            StreamedCompletion(redacted)
            if selection and selection.model_transport == "sse"
            else redacted
        )

    return factory


def main() -> None:
    """Expose explicit preflight and model stages; never overwrite an existing run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument(
        "--credentials-file", type=Path, default=Path(".myagentrun/credentials.toml")
    )
    args = parser.parse_args()
    selection = SmokeSelection.model_validate_json(args.selection.read_text())
    if args.action == "check":
        asyncio.run(check_environment(selection, args.sources, args.output, args.evaluator))
    else:
        asyncio.run(
            run_smoke(
                selection,
                args.sources,
                args.output,
                args.evaluator,
                lambda: provider_factory(args.credentials_file, selection)(),  # noqa: PLW0108 - defer secrets until gates pass
            )
        )


if __name__ == "__main__":
    main()
