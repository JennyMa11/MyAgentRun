"""Offline contracts for the external Pi comparison, with no model or home-directory access."""

from __future__ import annotations

import base64
import json
import runpy
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "benchmarks/swebench_pi/v1"


def test_pi_remote_file_operations_reject_traversal_and_symlink_escape(tmp_path: Path) -> None:
    api = runpy.run_path(str(SCRIPTS / "remote_io.py"))
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    for path in ("../secret", "escape/secret", str(tmp_path / "secret")):
        with pytest.raises(ValueError, match="path_outside_workspace"):
            api["checked_path"](path, root)


def test_pi_remote_write_and_read_preserve_content_in_the_workspace(tmp_path: Path) -> None:
    execute = runpy.run_path(str(SCRIPTS / "remote_io.py"))["execute"]
    path = tmp_path / "nested/code.py"
    execute({"op": "write", "path": str(path), "content": "print('hello')\n"}, root=tmp_path)
    result = execute({"op": "read", "path": str(path)}, root=tmp_path)
    assert base64.b64decode(result["data"]) == b"print('hello')\n"
    assert list(path.parent.iterdir()) == [path]


def test_pi_aggregation_keeps_full_denominator_unknown_usage_and_failure_tokens() -> None:
    aggregate = runpy.run_path(str(SCRIPTS / "runner.py"))["aggregate"]
    rows = [
        {"success": True, "tokens": 100, "unknown_usage_calls": 0},
        {"success": False, "tokens": 30, "unknown_usage_calls": 1},
        {"success": None, "tokens": 0},
    ]
    result = aggregate(rows)
    assert result["declared_tasks"] == 3
    assert result["graded_tasks"] == 2
    assert result["resolved"] == 1
    assert result["known_tokens"] == 130
    assert result["actual_total_tokens"] is None
    assert result["unknown_usage_calls"] == 1


def test_pi_interrupted_worker_preserves_known_tokens_and_unsettled_calls() -> None:
    usage = runpy.run_path(str(SCRIPTS / "runner.py"))["trace_usage"]
    events: list[dict[str, object]] = [{"name": "model_call_started"} for _ in range(3)]
    events.extend(
        [
            {
                "name": "model_call_completed",
                "data": {
                    "usage_known": True,
                    "input_tokens": 80,
                    "output_tokens": 20,
                    "cached_input_tokens": 40,
                },
            },
            {"name": "model_call_completed", "data": {"usage_known": False}},
        ]
    )
    assert usage(events) == {
        "tokens": 100,
        "input_tokens": 80,
        "output_tokens": 20,
        "cached_input_tokens": 40,
        "unknown_usage_calls": 2,
    }


def test_pi_pairing_matches_ids_and_rejects_ungraded_or_missing_cases() -> None:
    pair = runpy.run_path(str(SCRIPTS / "report.py"))["paired_counts"]
    ids = ["a", "b", "c", "d"]
    baseline = [{"instance_id": k, "success": k in {"a", "b"}} for k in ids]
    candidate: list[dict[str, object]] = [
        {"instance_id": k, "success": k in {"a", "c"}} for k in reversed(ids)
    ]
    assert pair(ids, baseline, candidate) == {
        "both": 1,
        "pi_only": 1,
        "myagentrun_only": 1,
        "neither": 1,
        "denominator": 4,
    }
    with pytest.raises(ValueError, match="complete independent"):
        pair(ids, baseline, candidate[:-1])
    candidate[0]["success"] = None
    with pytest.raises(ValueError, match="complete independent"):
        pair(ids, baseline, candidate)
    candidate[0].update(infrastructure_error=True, status="failed")
    partial = pair(ids, baseline, candidate, allow_ungraded_infrastructure=True)
    assert partial == {
        "both": 1,
        "pi_only": 1,
        "myagentrun_only": 1,
        "neither": 0,
        "denominator": 4,
        "paired_checked": 3,
        "ungraded": 1,
    }


def test_pi_usage_recovery_requires_real_consumption_not_error_or_zero(tmp_path: Path) -> None:
    reconcile = runpy.run_path(str(SCRIPTS / "reconcile.py"))["reconcile"]
    trace = tmp_path / "runs/case/trace.jsonl"
    trace.parent.mkdir(parents=True)
    events = [
        {
            "name": "model_call_started",
            "step_id": 1,
            "run_id": "r",
            "occurred_at": "2026-11-17T12:00:00Z",
        },
        {
            "name": "model_call_completed",
            "step_id": 1,
            "run_id": "r",
            "occurred_at": "2026-11-17T12:01:00Z",
            "data": {"usage_known": False},
        },
        {"name": "run_finished"},
    ]
    text = "\n".join(json.dumps(e) for e in events) + "\n"
    trace.write_text(text)
    (tmp_path / "protocol.json").write_text(json.dumps({"model": "test"}))
    provider = tmp_path / "usage.json"
    for kind, prompt, expected in [(2, 100, 110), (5, 100, None), (2, 0, None)]:
        provider.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "type": kind,
                            "prompt_tokens": prompt,
                            "completion_tokens": 10,
                            "model_name": "test",
                            "request_id": "20261117120002request",
                        }
                    ]
                }
            )
        )
        result = reconcile(tmp_path, provider)
        assert result["actual_total_tokens"] == expected
        assert result["unknown_usage_remaining"] == (1 if expected is None else 0)
        assert trace.read_text() == text
