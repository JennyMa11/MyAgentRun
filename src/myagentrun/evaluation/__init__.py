"""Read-only evaluation views derived from runtime traces.

Nothing in this package mutates state or calls a model: every function takes a trace
(or an aggregate mapping) and returns a derived view that can be regenerated at any
time. That is what keeps the raw JSONL trace the single fact source.
"""

from myagentrun.evaluation.diff import StepComparison, TraceDiff, diff_traces
from myagentrun.evaluation.harness_metrics import (
    HarnessMetrics,
    MetricRate,
    ScoringEvidence,
    aggregate_harness_metrics,
    score_harness,
)
from myagentrun.evaluation.latency import (
    ToolLatency,
    latency_from_trace,
    retried_tool_calls,
    tool_latencies,
)
from myagentrun.evaluation.pareto import (
    X_METRICS,
    Dominance,
    MetricPoint,
    ParetoAnalysis,
    analyze,
    pareto_analysis,
)
from myagentrun.evaluation.replay import Replay, ReplayStep, ReplayToolCall, build_replay
from myagentrun.evaluation.taxonomy import FailureDiagnosis, classify_failure, summarize_layers
from myagentrun.evaluation.tracefile import TraceError, TraceRecord, TraceRun, TraceStep

__all__ = [
    "X_METRICS",
    "Dominance",
    "FailureDiagnosis",
    "HarnessMetrics",
    "MetricPoint",
    "MetricRate",
    "ParetoAnalysis",
    "Replay",
    "ReplayStep",
    "ReplayToolCall",
    "ScoringEvidence",
    "StepComparison",
    "ToolLatency",
    "TraceDiff",
    "TraceError",
    "TraceRecord",
    "TraceRun",
    "TraceStep",
    "aggregate_harness_metrics",
    "analyze",
    "build_replay",
    "classify_failure",
    "diff_traces",
    "latency_from_trace",
    "pareto_analysis",
    "retried_tool_calls",
    "score_harness",
    "summarize_layers",
    "tool_latencies",
]
