"""All trimming views preserve current goals and explicitly tagged facts."""

import json
from pathlib import Path

import pytest

from myagentrun.adapters import ScriptedModel
from myagentrun.composition import RuntimeOptions, build_local_runtime
from myagentrun.context import SlidingWindowContext, SummaryCompactionContext
from myagentrun.domain import AgentBudget, Message, ModelResponse, Role, TokenUsage, ToolCall


@pytest.mark.parametrize("strategy", [SlidingWindowContext(2), SummaryCompactionContext(2)])
def test_old_critical_state_and_current_task_survive(
    strategy: SlidingWindowContext | SummaryCompactionContext,
) -> None:
    history = (
        Message(Role.SYSTEM, "rules"),
        Message(Role.USER, "initial"),
        Message(Role.SYSTEM, "keep this decision", name="critical_decision"),
        Message(Role.SYSTEM, "pending error", name="open_error"),
        Message(Role.USER, "latest goal"),
        *(Message(Role.ASSISTANT, f"step {index}") for index in range(20)),
    )
    view = strategy.build(history, "latest goal")
    contents = {message.content for message in view.messages}
    assert {"rules", "initial", "keep this decision", "pending error", "latest goal"} <= contents
    assert history[2].content == "keep this decision"


def test_pruned_summary_retains_late_failure_and_pending_tool_round() -> None:
    def request(call_id: str) -> Message:
        return Message(Role.ASSISTANT, "", tool_calls=(ToolCall(call_id, "shell", {}),))

    def result(call_id: str, stdout: str, exit_code: int) -> Message:
        return Message(
            Role.TOOL,
            json.dumps(
                {"status": "succeeded", "result": {"stdout": stdout, "exit_code": exit_code}}
            ),
            name="shell",
            tool_call_id=call_id,
        )

    early = result("old", "old build output\n" * 600, 0)
    failed = result("failed", "test noise\n" * 600 + "FAILED tests/test_math.py::test_empty", 1)
    pending = result("pending", "new result not yet processed\n" * 600, 0)
    history = (
        Message(Role.SYSTEM, "rules"),
        Message(Role.USER, "fix tests"),
        request("old"),
        early,
        request("failed"),
        failed,
        *(Message(Role.ASSISTANT, f"step {i}") for i in range(5)),
        request("pending"),
        pending,
        Message(Role.USER, "look at the latest output"),
    )
    strategy = SummaryCompactionContext(recent_messages=1, prune_tool_results=True)
    view = strategy.build(history, "fix tests")
    summary = next(m for m in view.messages if m.name == "context_summary")
    assert "FAILED tests/test_math.py::test_empty" in summary.content
    assert '"exit_code":1' in summary.content
    assert '"tool_call_id":"failed"' in summary.content
    assert "test noise\n" * 100 not in summary.content
    assert view.messages[-3] is history[-3]
    assert view.messages[-2] is pending
    assert history[5] is failed
    assert view.metadata["pruned_tool_result_indices"] == [3, 5]


@pytest.mark.asyncio
async def test_pruned_summary_only_shortens_answered_results_in_model_requests(
    tmp_path: Path,
) -> None:
    source = "line with useful code\n" * 220
    (tmp_path / "example.py").write_text(source)
    model = ScriptedModel(
        [
            ModelResponse(
                "", (ToolCall("first", "read_file", {"path": "example.py"}),), "tool_calls"
            ),
            ModelResponse(
                "", (ToolCall("second", "read_file", {"path": "example.py"}),), "tool_calls"
            ),
            ModelResponse("done"),
        ]
    )
    runtime = build_local_runtime(
        workspace=tmp_path, model=model, options=RuntimeOptions(context_strategy="summary_pruned")
    )
    run = await runtime.agent.run("Inspect example.py")
    first_original = next(m for m in model.requests[1].messages if m.tool_call_id == "first")
    latest = {m.tool_call_id: m for m in model.requests[2].messages if m.role is Role.TOOL}
    assert len(latest["first"].content) < len(first_original.content) / 2
    assert "example.py" in latest["first"].content
    assert len(latest["second"].content) == len(first_original.content)
    assert next(m for m in run.history if m.tool_call_id == "first") == first_original


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["below_threshold", "valid", "invalid", "bad_source", "unknown", "no_budget", "overrun"]
)
async def test_model_summary_is_budgeted_and_never_rewrites_history(
    tmp_path: Path, mode: str
) -> None:
    long_history = mode != "below_threshold"
    attempts_summary = long_history and mode != "no_budget"
    history = (
        Message(Role.USER, "fix calc"),
        Message(
            Role.ASSISTANT,
            "Archived failure: empty input. " + "noise " * (1800 if long_history else 1),
        ),
        *(Message(Role.ASSISTANT, f"step {i}") for i in range(20)),
    )
    (tmp_path / "calc.py").write_text("answer = 42")
    payload = json.dumps(
        {
            "goal": [
                {"text": "Fix calc", "sources": [999 if mode == "bad_source" else len(history) + 1]}
            ],
            "constraints": [],
            "progress": [],
            "open_issues": [],
            "next_steps": [],
        }
    )
    summary = ModelResponse(
        payload if mode in {"valid", "bad_source", "overrun"} else "unusable",
        usage=TokenUsage(500_000 if mode == "overrun" else 80, 20),
        usage_known=mode != "unknown",
    )
    ordinary = [
        ModelResponse(
            "",
            (ToolCall("read", "read_file", {"path": "calc.py"}),),
            "tool_calls",
            usage=TokenUsage(10, 5),
        ),
        ModelResponse("done", usage=TokenUsage(10, 5)),
    ]
    if attempts_summary:
        ordinary.insert(0, summary)
    model = ScriptedModel(ordinary)
    runtime = build_local_runtime(
        workspace=tmp_path,
        model=model,
        budget=AgentBudget(max_tokens=400) if mode == "no_budget" else None,
        options=RuntimeOptions(context_strategy="summary_model", context_input_budget=4096),
    )
    result = await runtime.agent.run("Fix calc", prior_history=history)
    assert history[1] in result.history
    assert not any(m.name == "context_model_summary" for m in result.history)
    if mode in {"unknown", "overrun"}:
        assert len(model.requests) == 1
        assert result.unknown_usage_calls == (1 if mode == "unknown" else 0)
        if mode == "overrun":
            assert result.token_usage.total_tokens == 500_020
        return
    assert result.token_usage.total_tokens == (130 if attempts_summary else 30)
    assert len(model.requests) == (3 if attempts_summary else 2)
    if attempts_summary:
        assert model.requests[0].tools == ()
        assert model.requests[0].max_output_tokens == 512
    for request in model.requests[-2:]:
        derived = [m for m in request.messages if m.name == "context_model_summary"]
        if mode == "valid":
            assert len(derived) == 1
            assert derived[0].role is Role.USER
            assert "Fix calc" in derived[0].content
        else:
            assert not derived
            assert any(m.name == "context_summary" for m in request.messages)
