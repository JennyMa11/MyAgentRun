# ruff: noqa: T201, RUF001 -- bounded Chinese audit summary only
"""Read-only, complete-denominator Pi vs historical MyAgentRun comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import statistics
import subprocess
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

from myagentrun.benchmark.hashes import frozen_tree_digest
from myagentrun.benchmark.public_swe import write_json
from myagentrun.benchmark.swebench_smoke import SmokeSelection


def paired_counts(
    ids: list[str],
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    allow_ungraded_infrastructure: bool = False,
) -> dict[str, int]:
    """Pair by immutable task ID; never silently remove incomplete cases."""
    if allow_ungraded_infrastructure:
        ungraded = [r for r in candidate if r.get("success") is None]
        if any(not r.get("infrastructure_error") or r.get("status") != "failed" for r in ungraded):
            raise ValueError("complete independent results required for every declared case")
        missing = {r["instance_id"] for r in ungraded}
        if len(candidate) != len(ids) or {r["instance_id"] for r in candidate} != set(ids):
            raise ValueError("complete independent results required for every declared case")
        result = paired_counts(
            [k for k in ids if k not in missing],
            [r for r in baseline if r["instance_id"] not in missing],
            [r for r in candidate if r["instance_id"] not in missing],
        )
        return {
            **result,
            "paired_checked": len(ids) - len(missing),
            "ungraded": len(missing),
            "denominator": len(ids),
        }
    maps = [{row["instance_id"]: row for row in rows} for rows in (baseline, candidate)]
    if any(len(rows) != len(ids) for rows in (baseline, candidate)) or any(
        set(rows) != set(ids)
        or any(not isinstance(row.get("success"), bool) for row in rows.values())
        for rows in maps
    ):
        raise ValueError("complete independent results required for every declared case")
    result = {"both": 0, "pi_only": 0, "myagentrun_only": 0, "neither": 0, "denominator": len(ids)}
    for identity in ids:
        old, new = (rows[identity]["success"] for rows in maps)
        label = (
            "both" if old and new else "pi_only" if new else "myagentrun_only" if old else "neither"
        )
        result[label] += 1
    return result


def audit(root: Path) -> dict[str, Any]:  # noqa: PLR0912, PLR0915 -- explicit evidence checks
    protocol = json.loads((root / "protocol.json").read_text())
    started = json.loads((root / "run_started.json").read_text())
    assert (
        started["protocol_sha256"]
        == hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest()
    )
    assert frozen_tree_digest(root / "runtime/eval") == protocol["eval_sha256"]
    assert frozen_tree_digest(root / "runtime/pi") == protocol["pi_runtime_sha256"]
    assert (
        hashlib.sha256((root / "runtime/node").read_bytes()).hexdigest() == protocol["node_sha256"]
    )
    assert (
        hashlib.sha256((root / "selection.json").read_bytes()).hexdigest()
        == protocol["selection_sha256"]
    )
    selection = SmokeSelection.model_validate_json((root / "selection.json").read_text())
    selection.verify(Path("benchmark-results/swebench-lite/v1/sources"))
    baseline_path = Path(protocol["historical_report"])
    assert (
        hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        == protocol["historical_report_sha256"]
    )
    old = json.loads(baseline_path.read_text())
    gate_file = root / "gate-evidence-index.json"
    assert hashlib.sha256(gate_file.read_bytes()).hexdigest() == old["gate_evidence_index_sha256"]
    gates = json.loads(gate_file.read_text())["tasks"]
    gate_images = {row["instance_id"]: row["image"] for row in gates}
    for gate in gates:
        for kind, expected in (("reference", True), ("empty", False)):
            path = Path(gate[kind + "_report"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == gate[kind + "_sha256"]
            assert json.loads(path.read_text())[gate["instance_id"]]["resolved"] is expected
    roots = [root]
    if (root / "continuations.json").exists():
        roots += [Path(p) for p in json.loads((root / "continuations.json").read_text())["roots"]]
    active = {}
    prior_consumption = 0
    prior_reserve = 0
    for directory in roots:
        local_protocol = json.loads((directory / "protocol.json").read_text())
        local_started = json.loads((directory / "run_started.json").read_text())
        assert (
            local_started["protocol_sha256"]
            == hashlib.sha256((directory / "protocol.json").read_bytes()).hexdigest()
        )
        if directory != root:
            for field in [
                "model",
                "endpoint",
                "temperature",
                "task_tokens",
                "max_steps",
                "max_tools",
                "max_seconds",
                "context_policy",
                "compaction",
                "retry",
                "tools",
                "resources",
                "pi_runtime_sha256",
                "eval_sha256",
                "node_sha256",
            ]:
                assert local_protocol[field] == protocol[field]
            assert frozen_tree_digest(directory / "runtime/eval") == protocol["eval_sha256"]
            assert frozen_tree_digest(directory / "runtime/pi") == protocol["pi_runtime_sha256"]
            assert (
                hashlib.sha256((directory / "runtime/node").read_bytes()).hexdigest()
                == protocol["node_sha256"]
            )
            subset = SmokeSelection.model_validate_json((directory / "selection.json").read_text())
            original_tasks = {task.instance_id: task for task in selection.tasks}
            assert all(task == original_tasks[task.instance_id] for task in subset.tasks)
            assert (
                subset.model_copy(
                    update={"tasks": selection.tasks, "total_tokens": selection.total_tokens}
                )
                == selection
            )
            assert subset.total_tokens == selection.total_tokens - prior_consumption - prior_reserve
            assert local_protocol.get("unknown_usage_nominal_reserve", 0) == prior_reserve
            parent = Path(local_protocol["parent_root"])
            assert (
                hashlib.sha256((parent / "report.json").read_bytes()).hexdigest()
                == local_protocol["parent_report_sha256"]
            )
            assert (
                hashlib.sha256((parent / "usage-reconciliation.json").read_bytes()).hexdigest()
                == local_protocol["parent_reconciliation_sha256"]
            )
            assert {t.instance_id for t in subset.tasks} == set(
                local_protocol["remaining_case_ids"]
            )
            assert (
                hashlib.sha256((directory / "selection.json").read_bytes()).hexdigest()
                == local_protocol["selection_sha256"]
            )
        local_report = json.loads((directory / "report.json").read_text())
        usage_file = directory / "usage-reconciliation.json"
        if usage_file.exists():
            usage = json.loads(usage_file.read_text())
            prior_reserve += selection.task_tokens * len(
                {call["case_id"] for call in usage["calls"] if call["tokens"] is None}
            )
        prior_consumption += (
            json.loads(usage_file.read_text())["known_tokens"]
            if usage_file.exists()
            else local_report["aggregate"]["known_tokens"]
        )
        for row in local_report["results"]:
            if row["status"] == "not_run":
                continue
            if row["instance_id"] in active:
                raise ValueError("duplicate Pi attempts are not a single-shot comparison")
            active[row["instance_id"]] = (directory, row)
    ids = [task.instance_id for task in selection.tasks]
    pairs = paired_counts(
        ids, old["results"], [row for _, row in active.values()], allow_ungraded_infrastructure=True
    )
    images = json.loads((root / "images.json").read_text())
    assert images == gate_images
    secret = tomllib.loads(Path(".myagentrun/credentials.toml").read_text())["credentials"][
        "MYAGENTRUN_MOEROUTER_API_KEY"
    ].encode()
    rows = []
    tokens = 0
    unknown = 0
    for task in selection.tasks:
        source_root, row = active[task.instance_id]
        directory = source_root / "runs" / task.instance_id
        events = [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()]
        assert any(e["name"] == "model_call_started" for e in events)
        ends = [e for e in events if e["name"] == "run_finished"]
        assert len(ends) == 1
        assert ends[0]["data"]["status"] == row["status"] or row.get("infrastructure_error") is True
        assert {e["run_id"] for e in events} == {row["run_id"]}
        assert [e["step_id"] for e in events] == sorted(e["step_id"] for e in events)
        calls = [e["data"] for e in events if e["name"] == "model_call_completed"]
        known = [call for call in calls if call["usage_known"]]
        measured = sum(call["input_tokens"] + call["output_tokens"] for call in known)
        assert measured == row["tokens"]
        measured_unknown = len(calls) - len(known)
        assert measured_unknown == row["unknown_usage_calls"]
        assert len(calls) == len([e for e in events if e["name"] == "model_call_started"])
        remaining_unknown = measured_unknown
        if measured_unknown:
            reconciliation = json.loads((source_root / "usage-reconciliation.json").read_text())
            reconcile = runpy.run_path(str(Path(__file__).with_name("reconcile.py")))["reconcile"]
            verified = reconcile(source_root, Path(reconciliation["source"]))
            assert verified["source_sha256"] == reconciliation["source_sha256"]
            assert verified["calls"] == reconciliation["calls"]
            recovered = [r for r in verified["calls"] if r["run_id"] == row["run_id"]]
            assert len(recovered) == len(calls)
            remaining_unknown = sum(r["tokens"] is None for r in recovered)
            measured = sum(r["tokens"] or 0 for r in recovered)
        if not row.get("infrastructure_error"):
            sidecar = json.loads((directory / "sidecar.json").read_text())
            assert sidecar["case_id"] == task.instance_id
            assert sidecar["run_id"] == row["run_id"]
            assert sidecar["task_digest"] == task.task_digest
            assert sidecar["criterion_source"] == selection.official_commit
            assert (
                sidecar["patch_sha256"]
                == hashlib.sha256((directory / "model.patch").read_bytes()).hexdigest()
            )
            grader = json.loads((directory / "verifier/report.json").read_text())[task.instance_id]
            assert sidecar["resolved"] is row["success"] is grader["resolved"]
            log_meta = json.loads((directory / "verifier/log-metadata.json").read_text())
            assert (
                log_meta["raw_sha256"]
                == hashlib.sha256((directory / "verifier/test_output.raw").read_bytes()).hexdigest()
            )
            assert (
                log_meta["normalized_sha256"]
                == hashlib.sha256((directory / "verifier/test_output.txt").read_bytes()).hexdigest()
            )
        assert all(secret not in p.read_bytes() for p in directory.rglob("*") if p.is_file())
        for where in [directory / "isolation.json"] + (
            [] if row.get("infrastructure_error") else [directory / "verifier/isolation.json"]
        ):
            isolation = json.loads(where.read_text())
            required = {
                "image": images[task.instance_id],
                "mounts": [],
                "network": "none",
                "privileged": False,
                "memory_bytes": 1073741824,
                "nano_cpus": 1000000000,
                "pids_limit": 128,
            }
            assert all(isolation.get(name) == value for name, value in required.items())
            assert "ALL" in isolation["cap_drop"]
            assert "no-new-privileges" in isolation["security_opt"]
        tokens += measured
        unknown += remaining_unknown
        rows.append(
            {
                **row,
                "independent_score_status": "ungraded_infrastructure_error"
                if row.get("infrastructure_error")
                else "graded",
                "native_loop_status": ends[0]["data"]["status"],
                "last_model_error": next(
                    (call.get("error") for call in reversed(calls) if call.get("error")), None
                ),
                "known_tokens": measured,
                "actual_tokens": None if remaining_unknown else measured,
                "remaining_unknown_usage_calls": remaining_unknown,
                "evidence_directory": str(directory),
            }
        )
    removed = []
    isolation_paths = [
        path
        for directory in roots
        for pattern in ["runs/*/isolation.json", "runs/*/verifier/isolation.json"]
        for path in directory.glob(pattern)
    ]
    isolation_paths += [*root.glob("probe*/isolation.json"), root / "bridge-probes/isolation.json"]
    for where in isolation_paths:
        name = json.loads(where.read_text())["container"]
        result = subprocess.run(
            ["docker", "inspect", name], capture_output=True, text=True, check=False, timeout=15
        )
        assert result.returncode != 0
        assert "no such object" in result.stderr.lower()
        removed.append(name)
    setup = [json.loads(p.read_text()) for p in root.glob("probe*/result.json")]
    assert all(r["independent_check"] and r["unknown_usage_calls"] == 0 for r in setup)
    successes = sum(r["success"] is True for r in rows)
    latency = statistics.quantiles([r["duration_ms"] for r in rows], n=100, method="inclusive")
    return {
        "schema_version": "1.0",
        "status": "completed_with_ungraded_cases" if pairs["ungraded"] else "complete",
        "experiment": "pi_vs_historical_myagentrun_lite20",
        "declared_tasks": len(ids),
        "attempted_tasks": len(rows),
        "independently_graded_tasks": len(rows) - pairs["ungraded"],
        "ungraded_tasks": pairs["ungraded"],
        "pi_successes": successes,
        "myagentrun_successes": old["successes"],
        "paired_results": pairs,
        "pi_success_rate": None if pairs["ungraded"] else successes / len(ids),
        "pi_confirmed_success_share": successes / len(ids),
        "pi_success_rate_bounds": [
            successes / len(ids),
            (successes + pairs["ungraded"]) / len(ids),
        ],
        "myagentrun_success_rate": old["successes"] / len(ids),
        "difference_percentage_points": (successes - old["successes"]) * 100 / len(ids),
        "pi_known_tokens": tokens,
        "pi_actual_total_tokens": None if unknown else tokens,
        "pi_unknown_usage_calls": unknown,
        "pi_unknown_usage_nominal_reserve": prior_reserve,
        "pi_total_budget_satisfied": None if unknown else tokens <= selection.total_tokens,
        "protocol_deviations": [
            "ADR0036 amendment: continue unattempted tasks across batches with unknown usage; "
            "reserve nominal task allowance without claiming actual consumption or compliance"
        ]
        if prior_reserve
        else [],
        "pi_raw_unknown_usage_calls": sum(row.get("unknown_usage_calls", 0) for row in rows),
        "execution_roots": [str(p) for p in roots],
        "setup_tokens": sum(r["tokens"] for r in setup),
        "myagentrun_known_tokens": old["known_total_tokens"],
        "myagentrun_actual_total_tokens": old["actual_total_tokens"],
        "myagentrun_unknown_usage_calls": old["unknown_usage_remaining"],
        "pi_loop_states": dict(Counter(r["status"] for r in rows)),
        "pi_failed_model_error_cases": sum(
            not r["success"] and bool(r["last_model_error"]) for r in rows
        ),
        "pi_independent_test_failure_cases": sum(
            not r["success"] and r["status"] == "finished" for r in rows
        ),
        "pi_token_overrun_cases": sum(r["known_tokens"] > selection.task_tokens for r in rows),
        "pi_known_token_overrun": sum(
            max(0, r["known_tokens"] - selection.task_tokens) for r in rows
        ),
        "pi_avg_steps": sum(r["steps"] for r in rows) / len(rows),
        "pi_avg_tool_calls": sum(r["tool_calls"] for r in rows) / len(rows),
        "pi_total_loop_ms": sum(r["duration_ms"] for r in rows),
        "pi_latency_p50_ms": latency[49],
        "pi_latency_p95_ms": latency[94],
        "pi_latency_sample_count": len(rows),
        "latency_scope": "agent loop including model and tools; excludes independent verification",
        "latency_quantile": "linear interpolation at (n-1)*p (inclusive)",
        "myagentrun_measurements": old["harness_metrics"]["measurements"],
        "removed_containers": removed,
        "configured_secret_scan_cases": len(rows),
        "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": started["protocol_sha256"],
        "baseline_report_sha256": protocol["historical_report_sha256"],
        "limits": protocol["limits"],
        "results": rows,
        "baseline_results": old["results"],
    }


def markdown(result: dict[str, Any]) -> str:
    measurements = result["myagentrun_measurements"]
    pairs = result["paired_results"]
    lines = [
        "# Pi 与 MyAgentRun：固定 Lite20 历史对照",
        "",
        f"Pi 已尝试20题、独立评分{result['independently_graded_tasks']}题，"
        f"确认通过 **{result['pi_successes']}/20**，"
        f"MyAgentRun 历史结果 **{result['myagentrun_successes']}/20**；"
        f"已确认通过比例的差值为 {result['difference_percentage_points']:+.1f} 个百分点。",
        "未取得评分的基础设施错误保持未知；通过数/20是已确认通过比例，不能充当评分完整的成功率。",
        "这是同题、同模型、同 Token 额度的两套配置对照，不是单变量机制消融。",
        "历史 MyAgentRun 运行经历了服务端故障，Pi 运行时间不同，工具/提示也不同；"
        "不能把差值完全归因于 Harness。",
        "",
        "| 指标 | Pi 1.5.0 | MyAgentRun 历史基线 |",
        "| --- | ---: | ---: |",
        f"| 已独立评分 | {result['independently_graded_tasks']}/20 | 20/20 |",
        f"| 修复通过 | {result['pi_successes']}/20 | {result['myagentrun_successes']}/20 |",
        f"| 已知 Token | {result['pi_known_tokens']:,} | {result['myagentrun_known_tokens']:,} |",
        f"| 未知用量调用 | {result['pi_unknown_usage_calls']} | "
        f"{result['myagentrun_unknown_usage_calls']} |",
        f"| 平均模型步骤 | {result['pi_avg_steps']:.2f} | {measurements['avg_steps']:.2f} |",
        f"| 平均工具调用 | {result['pi_avg_tool_calls']:.2f} | "
        f"{measurements['avg_tool_calls']:.2f} |",
        f"| Loop 延迟 P50（秒） | {result['pi_latency_p50_ms'] / 1000:.1f} | "
        f"{measurements['latency_p50_ms'] / 1000:.1f} |",
        f"| Loop 延迟 P95（秒） | {result['pi_latency_p95_ms'] / 1000:.1f} | "
        f"{measurements['latency_p95_ms'] / 1000:.1f} |",
        "",
        "延迟包含模型与工具执行，不含独立验证；双方样本数均为20，分位数按(n-1)×p线性插值。",
        f"Pi 接入探针另计 {result['setup_tokens']:,} Token，不混入正式20题；失败与在途超量均保留。",
        "有未知用量的一侧，完整实际消耗和相关预算约束保持未知，不能把已知小计当作完整成本。",
        "协议偏离：第11题连接错误的用量无法核对后，按ADR0036修订跨批次继续未尝试题；"
        "原始停止规则、Trace及失败评分保留，每个含未知用量的历史任务另预留100万Token用于调度。"
        "该预留不代表实际消费或已证明的消费上界。",
        f"Pi 已知 Token 超额 {result['pi_token_overrun_cases']} 题，"
        f"超额合计 {result['pi_known_token_overrun']:,}。",
        f"Pi 未通过任务中，伴随模型调用错误 {result['pi_failed_model_error_cases']} 题；"
        f"Loop正常结束但独立测试未通过 {result['pi_independent_test_failure_cases']} 题。",
        "模型调用错误只能定位到模型服务/流式适配链路，本轮未独立验证具体根因，不能全部归因于服务端或Harness。",
        "",
        f"逐题配对（已检查{pairs['paired_checked']}/20，未评分{pairs['ungraded']}）："
        f"双方通过 {pairs['both']}；仅Pi通过 {pairs['pi_only']}；"
        f"仅MyAgentRun通过 {pairs['myagentrun_only']}；双方未通过 {pairs['neither']}。",
        "",
        "| 固定任务 | Pi 独立评分 | MyAgentRun 历史独立评分 | Pi Loop终态 |",
        "| --- | --- | --- | --- |",
    ]
    baseline = {r["instance_id"]: r for r in result["baseline_results"]}
    for row in result["results"]:
        old = baseline[row["instance_id"]]
        score = (
            "未评分（基础设施错误）"
            if row["success"] is None
            else "通过"
            if row["success"]
            else "未通过"
        )
        lines.append(
            f"| `{row['instance_id']}` | {score} | "
            f"{'通过' if old['success'] else '未通过'} | {row['status']} |"
        )
    lines += [
        "",
        "协议、原生工具差异和限制见 [Pi评测协议](../benchmarks/swebench_pi/v1/README.md) 与 "
        "[ADR0036](adr/0036-pi-historical-comparison.md)。",
        "本机原始证据和机器报告位于 `benchmark-results/pi-comparison/v1/`，不提交敏感产物。",
        "本轮只覆盖固定纯Python子集、自定义ARM64容器和基础配置，不宣称完整官方成绩或未测机制收益。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark-results/pi-comparison/v1"))
    args = parser.parse_args()
    result = audit(args.root)
    write_json(args.root / "comparison-report.json", result)
    Path("docs/pi-vs-myagentrun-lite20.md").write_text(markdown(result))
    print(
        {
            k: result[k]
            for k in [
                "declared_tasks",
                "pi_successes",
                "myagentrun_successes",
                "pi_known_tokens",
                "pi_unknown_usage_calls",
            ]
        }
    )


if __name__ == "__main__":
    main()
