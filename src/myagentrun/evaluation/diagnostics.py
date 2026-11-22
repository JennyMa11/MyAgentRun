"""Read-only, deterministic diagnosis of a selected runtime trace."""

from __future__ import annotations

import math
from collections.abc import Callable

from myagentrun.domain import JsonValue
from myagentrun.evaluation.harness_metrics import (
    ScoringEvidence,
    is_tool_execution_error,
    repeated_failed_actions,
    score_harness,
)
from myagentrun.evaluation.latency import latency_from_trace
from myagentrun.evaluation.tracefile import TraceRecord, TraceRun

_TOOL_RULES: tuple[tuple[str, Callable[[TraceRecord], bool], str], ...] = (
    ("tool_execution_error", is_tool_execution_error, "Inspect the failed tool's observation."),
    (
        "tool_timeout",
        lambda r: r.data.get("status") == "timed_out",
        "Check the operation's scope and configured timeout.",
    ),
    (
        "invalid_tool_call",
        lambda r: r.data.get("status") == "invalid",
        "Compare the invocation with the registered tool and its schema.",
    ),
    (
        "permission_denied",
        lambda r: r.data.get("status") == "denied",
        "Review the permission decision and whether the task has an authorized path.",
    ),
    (
        "automatic_tool_retry",
        lambda r: _count(r.data.get("attempt"), default=1) > 1,
        "Check whether automatic retries recovered and inspect their elapsed time.",
    ),
)
_MIN_CONTEXT_ESTIMATES = 2
_MODEL_EVENTS = {"model_call_completed", "model_call_failed"}


def _count(value: JsonValue, *, default: int = 0) -> int:
    return value if type(value) is int and value >= 0 else default


def _context(trace: TraceRun) -> dict[str, JsonValue]:
    records = trace.by_name("context_built")
    estimates = [
        estimate
        for record in records
        if (estimate := _count(record.data.get("estimated_tokens"), default=-1)) >= 0
    ]
    growth = (
        (estimates[-1] - estimates[0]) / estimates[0]
        if (len(estimates) >= _MIN_CONTEXT_ESTIMATES and estimates[0] > 0)
        else None
    )
    return {
        "observed_builds": len(records),
        "token_estimates": len(estimates),
        "first_estimated_tokens": estimates[0] if estimates else None,
        "last_estimated_tokens": estimates[-1] if estimates else None,
        "peak_estimated_tokens": max(estimates, default=None),
        "estimated_token_growth_ratio": growth,
        "builds_with_omission_counts": sum(
            _count(r.data.get("omitted_messages"), default=-1) >= 0 for r in records
        ),
        "builds_with_omissions": sum(_count(r.data.get("omitted_messages")) > 0 for r in records),
    }


def _duration(record: TraceRecord) -> float:
    value = record.data.get("duration_ms")
    if type(value) in (int, float) and isinstance(value, (int, float)) and math.isfinite(value):
        return float(value) if value >= 0 else -1.0
    return -1.0


def analyze_trace(
    trace: TraceRun, *, evidence: ScoringEvidence | None = None
) -> dict[str, JsonValue]:
    """Derive findings for one run, with event references and independent outcome evidence."""
    final = trace.final()
    if final is not None:
        # CLI activity after the Loop (such as the final Git diff) has its own events.
        trace = TraceRun.from_records(
            trace.records[: trace.records.index(final) + 1], path=trace.path
        )
    metrics = score_harness(trace, evidence=evidence)
    positions = {id(record): index for index, record in enumerate(trace.records, start=1)}
    calls = trace.by_name("tool_call_completed")
    model_calls = tuple(record for record in trace.records if record.name in _MODEL_EVENTS)
    findings: list[JsonValue] = []

    def reference(record: TraceRecord) -> dict[str, JsonValue]:
        return {
            "event": record.name,
            "step_id": record.step_id,
            "record_index": positions[id(record)],
        }

    def add(code: str, suggestion: str, count: int, record: TraceRecord | None) -> None:
        if count:
            findings.append(
                {
                    "code": code,
                    "count": count,
                    "evidence": reference(record) if record is not None else None,
                    "suggestion": suggestion,
                }
            )

    def add_matches(code: str, suggestion: str, records: tuple[TraceRecord, ...]) -> None:
        if records:
            add(code, suggestion, len(records), records[-1])

    def slowest(records: tuple[TraceRecord, ...]) -> JsonValue:
        record = max(records, key=_duration, default=None)
        if record is None or _duration(record) < 0:
            return None
        return {"duration_ms": _duration(record), "evidence": reference(record)}

    for code, matches, suggestion in _TOOL_RULES:
        add_matches(code, suggestion, tuple(call for call in calls if matches(call)))
    _, repeated = repeated_failed_actions(trace)
    add_matches(
        "repeated_failed_action", "Review the prior failure before repeating the action.", repeated
    )
    add_matches(
        "model_call_failed",
        "Inspect the provider failure at the referenced step.",
        trace.by_name("model_call_failed"),
    )
    add_matches(
        "context_summary_rejected",
        "Inspect the recorded summary rejection reason.",
        tuple(
            record
            for record in trace.by_name("context_summary_resolved")
            if record.data.get("accepted") is False
        ),
    )

    status = trace.status()
    verified = evidence.task_success if evidence is not None else None
    add(
        "incomplete_trace",
        "Check why the terminal event is missing.",
        int(final is None),
        trace.records[-1],
    )
    add(
        "run_failed",
        "Inspect the terminal failure reason and preceding events.",
        int(status == "failed"),
        final,
    )
    add(
        "budget_exhausted",
        "Inspect the terminal budget reason and the preceding calls.",
        int(status == "budget_exhausted"),
        final,
    )
    add(
        "unknown_token_usage",
        "Check whether the model returned usage for every call.",
        _count(metrics.measurements["unknown_usage_calls"]),
        None,
    )
    if _count(metrics.measurements.get("token_overrun")) > 0:
        add(
            "token_budget_overrun",
            "Review the declared limit and the recorded model usage.",
            1,
            final or (model_calls[-1] if model_calls else None),
        )
    add(
        "finished_verification_failed",
        "Inspect the independent verifier evidence.",
        int(status == "finished" and verified is False),
        final,
    )

    return {
        "schema_version": "1.0",
        "derived": True,
        "analyzer_version": "1.0",
        "run_id": metrics.run_id,
        "trace": str(trace.path),
        "outcome": {"loop_status": status, "verified_success": verified},
        "metrics": metrics.to_json(),
        "context": _context(trace),
        "latency": {
            "breakdown": latency_from_trace(trace).to_json() if final is not None else None,
            "slowest_model_call": slowest(model_calls),
            "slowest_tool_call": slowest(calls),
        },
        "findings": findings,
    }
