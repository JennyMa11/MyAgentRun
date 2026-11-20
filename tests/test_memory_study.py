"""Memory ablation arms vary one declared capability at a time."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from myagentrun.benchmark.memory_study import (
    GLM_ROUTE_CREDENTIAL_ENV,
    GLM_ROUTE_MATRIX_PATH,
    GLM_ROUTE_MODEL,
    GLM_ROUTE_PROTOCOL_ID,
    GLM_ROUTE_PROTOCOL_PATH,
    MemoryArm,
    memory_study_options,
    protocol_route,
    study_protocol,
)
from myagentrun.domain.memory import MemoryConfig

ROOT = Path(__file__).resolve().parents[1]


def _memory(database: str = "/private/memory.sqlite3") -> MemoryConfig:
    return MemoryConfig("project", "v1", database, "/private")


def test_p501_protocol_has_ninety_unique_slots_and_three_repeats() -> None:
    protocol = study_protocol()
    slots = [(r["case_id"], r["repeat"], r["group"]) for r in protocol["queue"]]
    assert len(slots) == 90
    assert len(set(slots)) == 90
    assert {r[1] for r in slots} == {0, 1, 2}
    assert len({r[0] for r in slots}) == 6


def test_p502_v1_to_index_changes_only_representation_mode() -> None:
    a = asdict(memory_study_options(MemoryArm.V1, _memory()))
    b = asdict(memory_study_options(MemoryArm.INDEX, _memory()))
    assert a.pop("memory") != b.pop("memory")
    assert a == b
    memory_a = memory_study_options(MemoryArm.V1, _memory()).memory
    memory_b = memory_study_options(MemoryArm.INDEX, _memory()).memory
    assert memory_a is not None
    assert memory_b is not None
    ma, mb = asdict(memory_a), asdict(memory_b)
    assert ma.pop("retrieval_mode") == "deterministic_v1"
    assert mb.pop("retrieval_mode") == "progressive_v2"
    assert ma == mb


def test_p502_index_to_active_changes_only_search_and_read_tools() -> None:
    a = asdict(memory_study_options(MemoryArm.INDEX, _memory()))
    b = asdict(memory_study_options(MemoryArm.ACTIVE, _memory()))
    tools_a, tools_b = set(a.pop("enabled_tools")), set(b.pop("enabled_tools"))
    assert tools_b - tools_a == {"search_memory", "read_memory"}
    assert tools_a <= tools_b
    assert a == b


def test_p502_active_to_consolidated_changes_only_corpus() -> None:
    a = asdict(memory_study_options(MemoryArm.ACTIVE, _memory()))
    b = asdict(memory_study_options(MemoryArm.CONSOLIDATED, _memory("/private/new.sqlite3")))
    ma, mb = a.pop("memory"), b.pop("memory")
    assert ma.pop("database") != mb.pop("database")
    assert ma == mb
    assert a == b


@pytest.mark.parametrize("arm", list(MemoryArm))
def test_p502_no_arm_has_a_memory_write_or_maintenance_tool(arm: MemoryArm) -> None:
    options = memory_study_options(arm, None if arm is MemoryArm.OFF else _memory())
    assert options.enabled_tools is not None
    assert options.enabled_tools <= {
        "read_file",
        "write_file",
        "patch_file",
        "test",
        "search_memory",
        "read_memory",
    }
    assert options.skill_roots == ()
    assert options.verification is None


@pytest.mark.parametrize(
    "arm", [MemoryArm.V1, MemoryArm.INDEX, MemoryArm.ACTIVE, MemoryArm.CONSOLIDATED]
)
def test_p502_enabled_memory_requires_explicit_seed(arm: MemoryArm) -> None:
    with pytest.raises(ValueError, match="approved_memory_seed_required"):
        memory_study_options(arm, None)


def test_p501_v2_protocol_changes_only_the_model_route() -> None:
    """A new route is a declared scope change, not a silent edit of the old one.

    The v1 protocol stays byte-identical because its snapshot and report are
    already signed; v2 must differ only in the route it declares.
    """
    v1 = study_protocol()
    v2 = study_protocol(ROOT / GLM_ROUTE_PROTOCOL_PATH)
    assert v1["protocol_id"] == "memory_v2_p5_v1"
    assert v2["protocol_id"] == GLM_ROUTE_PROTOCOL_ID
    assert v1["model"]["id"] == "deepseek-flash"
    assert v2["model"] == GLM_ROUTE_MODEL
    changed = {key for key in v1 if v1[key] != v2[key]}
    assert changed == {"protocol_id", "model", "consolidation_budget", "notes"}
    # Only the consolidation wall clock may differ; every other budget is identical.
    assert v2["budget"] == v1["budget"]
    assert v2["total_token_budget"] == v1["total_token_budget"]
    assert v2["consolidation_budget"] | {"max_duration_seconds": 0} == (
        v1["consolidation_budget"] | {"max_duration_seconds": 0}
    )
    assert v2["consolidation_budget"]["max_duration_seconds"] > 100
    assert {r["case_id"] for r in v1["queue"]} == {r["case_id"] for r in v2["queue"]}


def test_p501_v2_cannot_widen_its_own_route_from_the_protocol_file(tmp_path: Path) -> None:
    """The route is pinned in code; a protocol naming a different model is refused."""
    protocol = study_protocol(ROOT / GLM_ROUTE_PROTOCOL_PATH)
    protocol["model"]["endpoint"] = "https://attacker.example/v1/chat/completions"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="memory_study_model_or_isolation_changed"):
        study_protocol(path)

    protocol = study_protocol(ROOT / GLM_ROUTE_PROTOCOL_PATH)
    protocol["model"]["id"] = "glm-5.3-pro"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="memory_study_model_or_isolation_changed"):
        study_protocol(path)


def test_p501_v2_route_is_pinned_for_credential_name_and_matrix() -> None:
    """Each protocol owns its credential name and matrix; neither is inferred."""
    v2 = study_protocol(ROOT / GLM_ROUTE_PROTOCOL_PATH)
    route = protocol_route(v2)
    assert route["credential_env"] == GLM_ROUTE_CREDENTIAL_ENV
    assert route["matrix"] == GLM_ROUTE_MATRIX_PATH
    assert route["provider"] == "glm"
    v1_route = protocol_route(study_protocol())
    assert v1_route["credential_env"] == "DEEPSEEK_API_KEY"
    assert v1_route["provider"] == "deepseek"
    assert v1_route["matrix"] == "docs/memory-v2-p5-matrix.json"


def test_p501_v2_matrix_keeps_the_v1_cases_and_selection() -> None:
    v1 = json.loads((ROOT / "docs/memory-v2-p5-matrix.json").read_text())
    v2 = json.loads((ROOT / GLM_ROUTE_MATRIX_PATH).read_text())
    assert v2["matrix_id"] == "memory_v2_p5_v2"
    assert v2["protocol"] == GLM_ROUTE_PROTOCOL_PATH
    assert v2["cases"] == v1["cases"]
    assert v2["component_selection"] == v1["component_selection"]
    assert v2["status"] == v1["status"] == "not_run"
