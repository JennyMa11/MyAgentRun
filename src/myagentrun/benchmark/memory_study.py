"""Fixed Memory-only arms and boundary validation for the P5 paired protocol."""

import json
from dataclasses import replace
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from myagentrun.benchmark.runner import BenchmarkManifest
from myagentrun.composition import RuntimeOptions
from myagentrun.domain.memory import MemoryConfig
from myagentrun.evaluation.extension_comparison import CASE_COUNT, GROUPS, REPEATS, RESAMPLES, SEED


class MemoryArm(StrEnum):
    OFF = "off"
    V1 = "v1"
    INDEX = "index"
    ACTIVE = "active"
    CONSOLIDATED = "consolidated"


ARM_GROUPS = dict(zip(GROUPS, MemoryArm, strict=True))
COMMON_TOOLS = frozenset({"read_file", "write_file", "patch_file", "test"})
ACTIVE_TOOLS = frozenset({"search_memory", "read_memory"})
INPUT_BUDGET = 8192
OUTPUT_LIMIT = 8000
COHORT_TOKEN_LIMIT = 1080000
STUDY_ROOT = Path(__file__).resolve().parents[3]


def memory_study_options(arm: MemoryArm, memory: MemoryConfig | None) -> RuntimeOptions:
    if arm is MemoryArm.OFF:
        if memory is not None:
            raise ValueError("off_arm_cannot_receive_memory")
    elif memory is None:
        raise ValueError("approved_memory_seed_required")
    else:
        memory = replace(
            memory,
            retrieval_mode="deterministic_v1" if arm is MemoryArm.V1 else "progressive_v2",
        )
    return RuntimeOptions(
        context_strategy="full",
        context_input_budget=INPUT_BUDGET,
        workspace_instructions_enabled=True,
        allow_write=True,
        allow_shell=True,
        allowed_commands=frozenset({"python", "python3", "node"}),
        enabled_tools=COMMON_TOOLS | ACTIVE_TOOLS
        if arm in {MemoryArm.ACTIVE, MemoryArm.CONSOLIDATED}
        else COMMON_TOOLS,
        memory=memory,
        tool_output_limit=OUTPUT_LIMIT,
        artifact_offload=False,
        normalize_tool_results=True,
        error_recovery_enabled=False,
    )


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)


class _Slot(_Strict):
    case_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,127}$")
    repeat: int = Field(ge=0, lt=REPEATS)
    group: Literal["Baseline", "A", "B", "C", "D"]


class _Budget(_Strict):
    max_steps: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    max_tool_calls: int = Field(gt=0)
    max_duration_seconds: float = Field(gt=0)


# A v2 protocol swaps only the model route, because the credential issued for this
# run does not authorize the v1 route.  The values live here rather than being read
# from the protocol file: a protocol that could name its own endpoint and model would
# be able to widen its own scope, so the file is checked against these constants and
# never trusted as the source of them.
GLM_ROUTE_MODEL: dict[str, Any] = {
    "adapter": "existing OpenAICompatibleModel",
    "id": "glm-5.3-flash",
    "endpoint": "https://moerouter.com/v1/chat/completions",
    "temperature": 0,
    "thinking": False,
    "max_output_tokens": 1024,
    "response_version": "record if supplied; unavailable before request",
    "max_attempts": 1,
    "require_usage": True,
}
GLM_ROUTE_CREDENTIAL_ENV = "MOEROUTER_API_KEY"
# Measured on 2026-11-19: one GLM completion took 103 s (others 6 s), so the v1
# 30-second consolidation deadline cannot hold on this route.  Only the wall clock
# changes; steps, tools and tokens stay identical to v1.
GLM_ROUTE_CONSOLIDATION_SECONDS = 240.0
GLM_ROUTE_PROTOCOL_ID = "memory_v2_p5_v2"
GLM_ROUTE_PROTOCOL_PATH = "benchmarks/memory_v2/p5-protocol-v2.json"
GLM_ROUTE_MATRIX_PATH = "docs/memory-v2-p5-matrix-v2.json"
PROTOCOLS: dict[str, dict[str, Any]] = {
    "memory_v2_p5_v1": {
        "path": "benchmarks/memory_v2/p5-protocol.json",
        "matrix": "docs/memory-v2-p5-matrix.json",
        "credential_env": "DEEPSEEK_API_KEY",
        "provider": "deepseek",
        "model": None,
    },
    GLM_ROUTE_PROTOCOL_ID: {
        "path": GLM_ROUTE_PROTOCOL_PATH,
        "matrix": GLM_ROUTE_MATRIX_PATH,
        "credential_env": GLM_ROUTE_CREDENTIAL_ENV,
        "provider": "glm",
        "model": GLM_ROUTE_MODEL,
    },
}


class _Protocol(_Strict):
    schema_version: Literal["1.0"]
    protocol_id: Literal["memory_v2_p5_v1", "memory_v2_p5_v2"]
    source_suite: Literal["benchmarks/extension_v1/offline.json"]
    model: dict[str, Any]
    isolation: dict[str, Any]
    budget: _Budget
    real_model_runs: int
    nominal_task_tokens: int
    total_token_budget: int
    consolidation_budget: _Budget
    consolidation_attempts: int
    groups: dict[str, str]
    comparisons: list[list[str]]
    secondary_comparison: list[str]
    common_tools: list[str]
    active_tools: list[str]
    context: Literal["full"]
    context_input_budget: int
    tool_output_limit: int
    queue: list[_Slot]
    analysis: dict[str, Any]
    corpus_policy: dict[str, str]
    stop_policy: list[str]
    notes: list[str]

    @model_validator(mode="after")
    def fixed_scope(self) -> Self:
        runs = CASE_COUNT * REPEATS * len(GROUPS)
        actual = {(r.case_id, r.repeat, r.group) for r in self.queue}
        cases = {r.case_id for r in self.queue}
        expected = {(c, r, g) for c in cases for r in range(REPEATS) for g in GROUPS}
        if (
            self.real_model_runs != runs
            or len(self.queue) != runs
            or len(cases) != CASE_COUNT
            or actual != expected
            or self.groups != {g: arm.value for g, arm in ARM_GROUPS.items()}
            or set(self.common_tools) != COMMON_TOOLS
            or set(self.active_tools) != ACTIVE_TOOLS
            or self.context_input_budget != INPUT_BUDGET
            or self.tool_output_limit != OUTPUT_LIMIT
            or self.budget.model_dump()
            != {
                "max_steps": 12,
                "max_tokens": 6000,
                "max_tool_calls": 24,
                "max_duration_seconds": 120.0,
            }
            or self.nominal_task_tokens != runs * self.budget.max_tokens
            or self.total_token_budget != COHORT_TOKEN_LIMIT
            or self.consolidation_attempts != 1
            or self.consolidation_budget.model_dump()
            != {
                "max_steps": 12,
                "max_tokens": 8000,
                "max_tool_calls": 16,
                "max_duration_seconds": (
                    GLM_ROUTE_CONSOLIDATION_SECONDS
                    if self.protocol_id == GLM_ROUTE_PROTOCOL_ID
                    else 30.0
                ),
            }
            or self.analysis.get("seed") != SEED
            or self.analysis.get("resamples") != RESAMPLES
            or self.analysis.get("retain_repeats") != REPEATS
            or self.comparisons != [[a, b] for a, b in pairwise(GROUPS)]
            or self.secondary_comparison != [GROUPS[0], GROUPS[-1]]
        ):
            raise ValueError("memory_study_protocol_scope_changed")
        return self


def study_protocol(path: Path | None = None) -> dict[str, Any]:
    """Read a strictly checked protocol; no provider, secret or execution side effects."""
    selected = path or STUDY_ROOT / PROTOCOLS["memory_v2_p5_v1"]["path"]
    protocol = _Protocol.model_validate_json(selected.read_bytes())
    route = PROTOCOLS[protocol.protocol_id]
    baseline = json.loads((STUDY_ROOT / "benchmarks/extension_v1/manifest.json").read_bytes())
    expected_model = route["model"] or (
        baseline["model"] | {"max_attempts": 1, "require_usage": True}
    )
    if json.dumps(protocol.model, sort_keys=True) != json.dumps(
        expected_model, sort_keys=True
    ) or json.dumps(protocol.isolation, sort_keys=True) != json.dumps(
        baseline["isolation"], sort_keys=True
    ):
        raise ValueError("memory_study_model_or_isolation_changed")
    tasks = BenchmarkManifest.from_file(STUDY_ROOT / protocol.source_suite).tasks
    if {r.case_id for r in protocol.queue} != {t.task_id for t in tasks}:
        raise ValueError("memory_study_tasks_changed")
    return protocol.model_dump(mode="json")


def protocol_route(protocol: dict[str, Any]) -> dict[str, Any]:
    """Return the frozen path, matrix and credential name for a checked protocol."""
    try:
        return PROTOCOLS[str(protocol["protocol_id"])]
    except KeyError:
        raise ValueError("memory_study_protocol_route_unknown") from None
