"""Contracts for evidence-backed trace diagnostics."""

import json
from pathlib import Path

import pytest

from myagentrun.evaluation.diagnostics import analyze_trace
from myagentrun.evaluation.harness_metrics import ScoringBudget, ScoringEvidence
from myagentrun.evaluation.tracefile import TraceRecord, TraceRun


def test_diagnosis_links_repeated_failures_and_independent_outcome() -> None:
    trace = TraceRun.from_records(
        (
            TraceRecord("context_built", "r", 1, {"estimated_tokens": 100}),
            TraceRecord(
                "tool_call_started",
                "r",
                1,
                {"call_id": "a", "arguments": {"path": "private-path"}},
            ),
            TraceRecord(
                "tool_call_completed",
                "r",
                1,
                {"call_id": "a", "tool": "read_file", "status": "timed_out"},
            ),
            TraceRecord("context_built", "r", 2, {"estimated_tokens": 150, "omitted_messages": 1}),
            TraceRecord(
                "tool_call_started",
                "r",
                2,
                {"call_id": "b", "arguments": {"path": "private-path"}},
            ),
            TraceRecord(
                "tool_call_completed",
                "r",
                2,
                {"call_id": "b", "tool": "read_file", "status": "timed_out", "attempt": 2},
            ),
            TraceRecord(
                "model_call_completed",
                "r",
                3,
                {"input_tokens": 120, "output_tokens": 5, "content": "private-answer"},
            ),
            TraceRecord("run_finished", "r", 3, {"status": "finished"}),
            TraceRecord("tool_call_completed", "r", 3, {"status": "failed"}),
        ),
        path=Path("run.jsonl"),
    )
    evidence = ScoringEvidence(
        run_id="r",
        source="held-out tests",
        task_success=False,
        budget=ScoringBudget(max_tokens=100),
    )
    report = json.loads(json.dumps(analyze_trace(trace, evidence=evidence)))
    assert report["outcome"]["verified_success"] is False
    assert report["context"]["estimated_token_growth_ratio"] == 0.5
    assert report["metrics"]["measurements"]["tool_calls"] == 2
    findings = {finding["code"]: finding for finding in report["findings"]}
    assert findings["repeated_failed_action"]["evidence"]["step_id"] == 2
    assert findings["tool_timeout"]["count"] == 2
    assert findings["automatic_tool_retry"]["count"] == 1
    assert "finished_verification_failed" in findings
    assert "token_budget_overrun" in findings
    assert "private-" not in str(report)
    with pytest.raises(ValueError, match="does not match"):
        analyze_trace(trace, evidence=ScoringEvidence(run_id="other", source="tests"))


def test_incomplete_trace_preserves_unknowns_and_successful_retests() -> None:
    trace = TraceRun.from_records(
        (
            TraceRecord("model_call_started", "r", 1),
            TraceRecord("tool_call_started", "r", 1, {"call_id": "a", "arguments": {}}),
            TraceRecord("tool_call_completed", "r", 1, {"call_id": "a", "status": "succeeded"}),
            TraceRecord("tool_call_started", "r", 2, {"call_id": "b", "arguments": {}}),
            TraceRecord("tool_call_completed", "r", 2, {"call_id": "b", "status": "succeeded"}),
        ),
    )
    report = json.loads(json.dumps(analyze_trace(trace)))
    assert report["outcome"]["verified_success"] is None
    assert report["context"]["estimated_token_growth_ratio"] is None
    assert {finding["code"] for finding in report["findings"]} == {
        "incomplete_trace",
        "unknown_token_usage",
    }


def test_repetition_requires_visible_arguments_for_each_step() -> None:
    records: list[TraceRecord] = []
    for step_id, value in enumerate(("one", "two", "[REDACTED]", "[REDACTED]"), start=1):
        records.extend(
            (
                TraceRecord(
                    "tool_call_started",
                    "r",
                    step_id,
                    {"call_id": "reused-id", "arguments": {"value": value}},
                ),
                TraceRecord(
                    "tool_call_completed",
                    "r",
                    step_id,
                    {"call_id": "reused-id", "tool": "read_file", "status": "failed"},
                ),
            )
        )
    report = json.loads(json.dumps(analyze_trace(TraceRun.from_records(tuple(records)))))
    rate = report["metrics"]["rates"]["repeated_failed_action_rate"]
    assert rate == {"numerator": 0, "denominator": 2, "eligible": 4, "coverage": 0.5, "value": 0.0}
