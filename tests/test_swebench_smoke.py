"""Contracts for offline SWE-bench smoke grading and immutable local images."""

from __future__ import annotations

import asyncio
import json
import runpy
import subprocess
from dataclasses import asdict
from pathlib import Path

import httpx
import pytest

from myagentrun.adapters import ScriptedModel
from myagentrun.adapters.models import OpenAICompatibleModel
from myagentrun.benchmark.public_swe import UsageModel
from myagentrun.benchmark.swebench_smoke import (
    SmokeSelection,
    StreamedCompletion,
    configured_profile,
    evaluation_script,
    runtime_options,
    save_evaluation_log,
    summarize,
)
from myagentrun.composition import build_local_runtime
from myagentrun.domain import ModelCallError, ModelRequest, ModelResponse, RunStatus
from myagentrun.infrastructure.containers import DockerHost, DockerSandbox


def test_evaluator_python_path_preserves_the_virtual_environment_symlink(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "benchmarks/swebench_context/v1/run.py"
    api = runpy.run_path(str(script))
    binary = tmp_path / "base_python"
    binary.write_text("test interpreter placeholder")
    link = tmp_path / "venv/bin/python"
    link.parent.mkdir(parents=True)
    link.symlink_to(binary)
    assert api["evaluator_path"](link) == link.absolute()
    assert api["evaluator_path"](link) != binary


def test_context_sweep_rotates_all_six_arms_without_task_replacement() -> None:
    script = Path(__file__).resolve().parents[1] / "benchmarks/swebench_context/v1/run.py"
    api = runpy.run_path(str(script))
    schedule = api["schedule"](["first", "second", "third"])
    strategies = api["STRATEGIES"]
    assert len(schedule) == 18
    assert len({(r["strategy"], r["instance_id"]) for r in schedule}) == 18
    for identity in ("first", "second", "third"):
        assert {r["strategy"] for r in schedule if r["instance_id"] == identity} == set(strategies)
    assert [r["strategy"] for r in schedule[:6]] != [r["strategy"] for r in schedule[6:12]]


def test_context_sweep_changes_only_strategy_in_runtime_options() -> None:
    script = Path(__file__).resolve().parents[1] / "benchmarks/swebench_context/v1/run.py"
    api = runpy.run_path(str(script))
    selection = SmokeSelection.model_validate_json(
        (script.parents[2] / "swebench_lite/v1/selection.json").read_text()
    )
    baseline = asdict(runtime_options(selection))
    for strategy in api["STRATEGIES"]:
        actual = asdict(api["strategy_options"](selection, strategy))
        assert actual.pop("context_strategy") == strategy
        assert actual == {k: v for k, v in baseline.items() if k != "context_strategy"}
    with pytest.raises(ValueError, match="unknown context strategy"):
        api["strategy_options"](selection, "surprise")


def test_context_container_reader_keeps_real_path_and_secret_boundaries(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "benchmarks/swebench_context/v1/run.py"
    api = runpy.run_path(str(script))
    namespace: dict[str, object] = {"__name__": "offline_worker_contract"}
    exec(compile(api["reader_source"](), "frozen_context_reader", "exec"), namespace)
    dispatch = namespace["dispatch"]
    assert callable(dispatch)
    root = tmp_path / "repo"
    root.mkdir()
    (root / "module.py").write_text("def repaired():\n    return 42\n")
    (root / ".env").write_text("test secret")
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    assert dispatch(root, "read_file", {"path": "module.py", "max_chars": 12}) == {
        "path": "module.py",
        "start_line": 1,
        "end_line": 2,
        "content": "def repaired",
        "truncated": True,
    }
    for path in ("../outside", "escape/outside", ".env"):
        with pytest.raises(ValueError, match=r"escapes workspace|sensitive file"):
            dispatch(root, "read_file", {"path": path})


def test_context_summary_keeps_unknown_ungraded_and_unrun_cases() -> None:
    script = Path(__file__).resolve().parents[1] / "benchmarks/swebench_context/v1/run.py"
    summarize_context = runpy.run_path(str(script))["summarize_arm"]
    result = summarize_context(
        [
            {"status": "finished", "success": True, "tokens": 100, "unknown_usage_calls": 0},
            {"status": "failed", "success": False, "tokens": 30, "unknown_usage_calls": 1},
            {"status": "failed", "success": None, "tokens": 20, "unknown_usage_calls": 0},
            {"status": "not_run", "success": None, "tokens": 0},
        ],
        1000,
    )
    assert result["declared"] == 4
    assert result["attempted"] == 3
    assert result["graded"] == 2
    assert result["passed"] == 1
    assert result["known_tokens"] == 150
    assert result["actual_total_tokens"] is None
    assert result["unknown_usage_calls"] == 1
    assert result["success_rate"] is None
    assert result["confirmed_success_share"] == 0.25


def test_context_allowance_charges_failure_and_reserves_unknown_task_once() -> None:
    script = Path(__file__).resolve().parents[1] / "benchmarks/swebench_context/v1/run.py"
    allowance = runpy.run_path(str(script))["remaining_allowance"]
    rows = [
        {"tokens": 100, "unknown_usage_calls": 3, "status": "failed"},
        {"tokens": 30, "unknown_usage_calls": 0, "status": "failed"},
    ]
    assert allowance(rows, total=2000, per_task=1000) == 870
    assert allowance(rows, total=1000, per_task=1000) == 0


def test_authorized_unknown_usage_is_limited_to_the_exact_interrupted_call() -> None:
    script = Path(__file__).resolve().parents[1] / "benchmarks/swebench_lite/v1/report.py"
    allowed = runpy.run_path(str(script))["authorized_unknown"]
    authorization = {"accepted_unknown_calls": [{"run_id": "interrupted", "step_id": 30}]}
    assert allowed("interrupted", {30}, authorization)
    assert not allowed("other-run", {30}, authorization)
    assert not allowed("interrupted", {30, 31}, authorization)
    assert not allowed("interrupted", {31}, authorization)
    assert not allowed("interrupted", set(), authorization)
    assert not allowed("interrupted", {30}, None)


def test_study_usage_summary_preserves_unknowns_and_all_known_failure_consumption() -> None:
    script = Path(__file__).resolve().parents[1] / "benchmarks/swebench_lite/v1/report.py"
    summarize_usage = runpy.run_path(str(script))["usage_summary"]
    rows = [
        {"known_tokens": 100, "remaining_unknown_usage_calls": 0, "success": True},
        {"known_tokens": 30, "remaining_unknown_usage_calls": 1, "success": False},
    ]
    unknown = summarize_usage(rows, 200)
    assert unknown["known_total_tokens"] == 130
    assert unknown["unknown_usage_remaining"] == 1
    assert unknown["actual_total_tokens"] is None
    assert unknown["aggregate_token_overrun"] is None
    assert unknown["tokens_per_success_including_failures"] is None
    rows[1]["remaining_unknown_usage_calls"] = 0
    known = summarize_usage(rows, 200)
    assert known["actual_total_tokens"] == 130
    assert known["tokens_per_success_including_failures"] == 130


@pytest.mark.parametrize(
    ("log_type", "prompt_tokens", "expected_unknown"),
    [(5, 0, 1), (5, 100, 1), (2, 0, 1), (2, 100, 0)],
)
def test_usage_reconciliation_requires_consumption_evidence(
    tmp_path: Path, log_type: int, prompt_tokens: int, expected_unknown: int
) -> None:
    """Error-log zero placeholders must not turn unknown model usage into zero."""
    directory = tmp_path / "runs/example__repo-1"
    directory.mkdir(parents=True)
    (tmp_path / "report.json").write_text(json.dumps({"model": "test-model"}))
    records = [
        {
            "name": "model_call_started",
            "step_id": 1,
            "run_id": "run-1",
            "occurred_at": "2026-11-16T12:00:00+00:00",
        },
        {
            "name": "model_call_failed",
            "step_id": 1,
            "run_id": "run-1",
            "occurred_at": "2026-11-16T12:01:00+00:00",
            "data": {"usage_known": False},
        },
        {"name": "run_finished"},
    ]
    trace_text = "\n".join(json.dumps(row) for row in records) + "\n"
    (directory / "trace.jsonl").write_text(trace_text)
    provider = tmp_path / "provider.json"
    provider.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "request_id": "20261116120002request",
                        "model_name": "test-model",
                        "type": log_type,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": 10 if prompt_tokens else 0,
                    }
                ]
            }
        )
    )
    script = Path(__file__).resolve().parents[1] / "benchmarks/swebench_lite/v1/reconcile_usage.py"
    reconcile = runpy.run_path(str(script))["reconcile"]
    result = reconcile(tmp_path, provider)
    assert result["unknown_usage_remaining"] == expected_unknown
    assert result["actual_total_tokens"] == (None if expected_unknown else 110)
    assert result["calls"][0]["tokens"] == (None if expected_unknown else 110)
    assert (directory / "trace.jsonl").read_text() == trace_text


def test_invalid_utf8_test_output_retains_raw_evidence_and_parseable_markers(
    tmp_path: Path,
) -> None:
    raw = b"diagnostic\xff\n>>>>> Start Test Output\nPASSED test_example\n>>>>> End Test Output\n"
    path = save_evaluation_log(tmp_path, raw)
    assert (tmp_path / "test_output.raw").read_bytes() == raw
    text = path.read_text(encoding="utf-8")
    assert "PASSED test_example" in text
    assert ">>>>> Start Test Output" in text
    assert ">>>>> End Test Output" in text
    metadata = json.loads((tmp_path / "log-metadata.json").read_text())
    assert metadata["raw_utf8_valid"] is False
    assert metadata["raw_size_bytes"] == len(raw)


def test_multiline_unittest_descriptions_keep_real_status_and_original_label(
    tmp_path: Path,
) -> None:
    for status in ("ok", "FAIL", "ERROR", "skipped 'optional dependency'"):
        raw = (
            ">>>>> Start Test Output\n"
            "test_example (package.TestCase)\n"
            f"A multiline description ... {status}\n"
            "test_incomplete (package.TestCase)\n"
            ">>>>> End Test Output\n"
        ).encode()
        text = save_evaluation_log(tmp_path, raw).read_text()
        assert f"test_example (package.TestCase) ... {status}" in text
        assert f"A multiline description ... {status}" in text
        assert "test_incomplete (package.TestCase) ..." not in text
        assert (tmp_path / "test_output.raw").read_bytes() == raw


@pytest.mark.asyncio
async def test_configurable_sse_limit_accepts_large_framing_but_still_bounds_it() -> None:
    frames = [
        {
            "choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}],
            "transport_metadata": "x" * 1_100_000,
        },
        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3}},
    ]
    body = "".join("data: " + json.dumps(frame) + "\n\n" for frame in frames)

    def respond(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, text=body + "data: [DONE]\n\n")

    for limit in (1_048_576, 2_097_152):
        provider = OpenAICompatibleModel(
            endpoint="https://example.invalid/v1/chat/completions",
            model="test",
            require_usage=True,
            max_stream_bytes=limit,
            transport=httpx.MockTransport(respond),
        )
        model = StreamedCompletion(provider)
        if limit == 1_048_576:
            with pytest.raises(ModelCallError, match="output limit"):
                await model.complete(ModelRequest(()))
        else:
            assert (await model.complete(ModelRequest(()))).content == "done"


@pytest.mark.asyncio
async def test_sse_completion_preserves_reported_usage_and_counts_missing_usage() -> None:
    include_usage = True
    expected_output_limit = 256

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["max_tokens"] == expected_output_limit
        assert payload["stream_options"] == {"include_usage": True}
        frames: list[dict[str, object]] = [
            {"choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]},
        ]
        if include_usage:
            frames.append({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3}})
        body = "".join("data: " + json.dumps(frame) + "\n\n" for frame in frames)
        return httpx.Response(200, text=body + "data: [DONE]\n\n")

    provider = OpenAICompatibleModel(
        endpoint="https://example.invalid/v1/chat/completions",
        model="test",
        require_usage=True,
        max_attempts=1,
        max_output_tokens=256,
        transport=httpx.MockTransport(respond),
    )
    model = UsageModel(StreamedCompletion(provider))
    response = await model.complete(ModelRequest(()))
    assert response.content == "done"
    assert response.usage.total_tokens == 15
    assert model.unknown_calls == 0
    expected_output_limit = 128
    summary = await model.complete(
        ModelRequest((), max_output_tokens=128, purpose="context_summary")
    )
    assert summary.usage.total_tokens == 15
    expected_output_limit = 256

    include_usage = False
    with pytest.raises(ModelCallError, match="omitted usage"):
        await model.complete(ModelRequest(()))
    assert model.unknown_calls == 1


@pytest.mark.asyncio
async def test_frozen_large_context_budget_reaches_model_instead_of_default_8192(
    tmp_path: Path,
) -> None:
    selection = SmokeSelection.model_validate_json(
        await asyncio.to_thread(Path("benchmarks/swebench_smoke/v2/selection.json").read_text)
    )
    large = selection.model_copy(update={"context_input_budget": 65_536})
    for protocol, expected in [(selection, RunStatus.FAILED), (large, RunStatus.FINISHED)]:
        runtime = build_local_runtime(
            workspace=tmp_path,
            model=ScriptedModel([ModelResponse("done")]),
            options=runtime_options(protocol),
        )
        result = await runtime.agent.run("x" * 40_000)
        assert result.status is expected
        if expected is RunStatus.FAILED:
            assert "context input budget" in str(result.failure_reason)


def test_model_configuration_selects_custom_endpoint_without_provider_fallback(
    tmp_path: Path,
) -> None:
    config = tmp_path / "myagentrun.toml"
    with pytest.raises(ValueError, match="model configuration"):
        configured_profile(config)
    config.write_text(
        '[myagentrun]\nmodel_profile = "openai"\nmodel = "glm-5.3-flash"\n'
        'endpoint = "https://moerouter.com/v1/chat/completions"\n'
        'api_key_env = "MYAGENTRUN_MOEROUTER_API_KEY"\n'
    )
    profile = configured_profile(config)
    assert profile.model == "glm-5.3-flash"
    assert profile.endpoint == "https://moerouter.com/v1/chat/completions"
    assert profile.api_key_env == "MYAGENTRUN_MOEROUTER_API_KEY"
    assert not profile.supports_thinking


def test_locally_built_image_must_use_an_immutable_config_id() -> None:
    DockerSandbox(DockerHost(), image="sha256:" + "a" * 64)
    with pytest.raises(ValueError, match="pinned"):
        DockerSandbox(DockerHost(), image="my-smoke:latest")


def test_offline_grading_preserves_tests_and_records_real_exit_status() -> None:
    original = (
        "#!/bin/bash\nset -uxo pipefail\n"
        "source /opt/miniconda3/bin/activate\nconda activate testbed\n"
        "python -m pip install -e .\n"
        ": '>>>>> Start Test Output'\npytest -rA tests/test_example.py\n"
        ": '>>>>> End Test Output'\ngit checkout HEAD tests/test_example.py\n"
    )
    script = evaluation_script(original, "completion-token")
    assert "pytest -rA tests/test_example.py\n" in script
    assert "--no-build-isolation --no-deps" in script
    assert "conda activate" not in script
    assert "test_exit=$?" in script
    assert script.index("test_exit=$?") < script.index("git checkout HEAD")
    assert "completion-token" in script
    with pytest.raises(ValueError, match="markers"):
        evaluation_script("echo no tests", "completion-token")


def test_ungraded_tasks_stay_unknown_and_in_the_declared_denominator() -> None:
    rows: list[dict[str, object]] = [
        {"success": True, "tokens": 12},
        {"success": False, "tokens": 9},
        {"success": None, "status": "not_run"},
    ]
    report = summarize(rows, 20)
    assert report["resolved"] == 1
    assert report["declared_tasks"] == 3
    assert report["graded_tasks"] == 2
    assert report["task_success_rate"] == 1 / 3
    assert report["evidence_coverage"] == 2 / 3
    assert report["tokens_per_success"] == 21
    assert report["token_overrun"] == 1
    assert summarize([], 20)["task_success_rate"] is None


@pytest.mark.integration
@pytest.mark.parametrize("test_exit", [0, 1])
def test_real_evaluation_script_records_failing_test_before_successful_cleanup(
    tmp_path: Path, test_exit: int
) -> None:
    original = (
        "#!/bin/bash\nset -uxo pipefail\n"
        ": '>>>>> Start Test Output'\n"
        f"(exit {test_exit})\n"
        ": '>>>>> End Test Output'\ntrue\n"
    )
    script = tmp_path / "eval.sh"
    script.write_text(evaluation_script(original, "test_completion_token"))
    result = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0
    assert f">>>>> Test Exit Code: {test_exit}" in result.stdout
    assert "test_completion_token" in result.stdout


def test_summary_study_pairs_rotate_and_keep_all_missing_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project))
    script = project / "benchmarks/swebench_context/v2/run.py"
    api = runpy.run_path(str(script))
    tasks = [f"task-{index}" for index in range(20)]
    rows = api["schedule"](tasks)
    assert len(rows) == 120
    for task in tasks:
        first = [r["strategy"] for r in rows if r["instance_id"] == task and r["repeat"] == 1]
        second = [r["strategy"] for r in rows if r["instance_id"] == task and r["repeat"] == 2]
        assert set(first) == {"summary", "summary_pruned", "summary_model"}
        assert first == list(reversed(second))

    pair = [r for r in rows if r["instance_id"] == tasks[0] and r["repeat"] == 1]
    indexed = {r["strategy"]: r for r in pair}
    indexed["summary"].update(status="failed", success=False, tokens=40, usage_audited=True)
    indexed["summary_pruned"].update(status="finished", success=True, tokens=20, usage_audited=True)
    result = api["compare"](rows, "summary", "summary_pruned")
    assert result["declared_pairs"] == 40
    assert result["outcomes"] == {"b_only": 1, "unscored": 39}
    assert result["pairs_with_complete_usage"] == 1
    assert result["pairs"][0]["actual_token_delta_b_minus_a"] == -20

    indexed["summary"]["unknown_usage_calls"] = 1
    result = api["compare"](rows, "summary", "summary_pruned")
    assert result["pairs_with_complete_usage"] == 0
    assert result["pairs"][0]["actual_token_delta_b_minus_a"] is None
