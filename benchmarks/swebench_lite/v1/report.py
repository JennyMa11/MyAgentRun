# ruff: noqa: T201, RUF001 -- bounded Chinese audit summary; never credentials.
"""Audit all 20 independent results before producing a complete study report.

This is a read-only derivative of raw runs. Partial batches are rejected, not
described as complete. Additional roots may contain previously unattempted
cases continued at a task boundary; duplicate attempts are never merged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
import subprocess
import tomllib
from collections import Counter
from pathlib import Path

from myagentrun.benchmark.hashes import frozen_tree_digest
from myagentrun.benchmark.public_swe import write_json
from myagentrun.benchmark.swebench_smoke import SmokeSelection
from myagentrun.evaluation.harness_metrics import (
    ScoringBudget,
    ScoringEvidence,
    aggregate_harness_metrics,
    score_harness,
)
from myagentrun.evaluation.latency import latency_from_trace
from myagentrun.evaluation.tracefile import TraceRun

MIN_SECRET_BYTES = 8
EXPECTED_TASKS = 20


def authorized_unknown(run_id: str, steps: set[int], authorization: dict | None) -> bool:
    """An explicit continuation exception covers exact calls, never a whole batch."""
    if not authorization or not steps:
        return False
    allowed = {
        call["step_id"]
        for call in authorization["accepted_unknown_calls"]
        if call["run_id"] == run_id
    }
    return steps == allowed


def usage_summary(rows: list[dict], nominal_budget: int) -> dict:
    """Known consumption is a lower bound when an explicitly retained unknown exists."""
    known = sum(row["known_tokens"] for row in rows)
    unknown = sum(row["remaining_unknown_usage_calls"] for row in rows)
    successes = sum(row["success"] for row in rows)
    overrun = max(0, known - nominal_budget)
    return {
        "known_total_tokens": known,
        "actual_total_tokens": None if unknown else known,
        "unknown_usage_remaining": unknown,
        "aggregate_token_overrun": None if unknown and not overrun else overrun,
        "known_aggregate_token_overrun_lower_bound": overrun,
        "tokens_per_success_including_failures": known / successes
        if successes and not unknown
        else None,
        "known_tokens_per_success_lower_bound": known / successes if successes else None,
    }


def audit(  # noqa: PLR0912, PLR0915 - explicit final evidence checks
    selection: SmokeSelection, roots: list[Path], credentials: Path
) -> dict:
    expected = {task.instance_id: task for task in selection.tasks}
    if len(expected) != EXPECTED_TASKS or len(selection.tasks) != EXPECTED_TASKS:
        raise ValueError("this report requires exactly 20 unique frozen tasks")
    selection.verify(roots[0].parent / "sources")
    first = json.loads((roots[0] / "report.json").read_text())
    authorization_path = roots[0] / "continuation-authorization.json"
    authorization = (
        json.loads(authorization_path.read_text()) if authorization_path.exists() else None
    )
    if (
        authorization
        and hashlib.sha256((roots[0] / "report.json").read_bytes()).hexdigest()
        != authorization["original_report_sha256"]
    ):
        raise ValueError("original report changed after continuation authorization")
    manifest = json.loads((roots[0] / "source-manifest.json").read_text())
    frozen_selection = roots[0] / "reproduction/benchmarks/swebench_lite/v1/selection.json"
    if hashlib.sha256(frozen_selection.read_bytes()).hexdigest() != manifest["selection_sha256"]:
        raise ValueError("frozen selection file hash mismatch")
    if SmokeSelection.model_validate_json(frozen_selection.read_text()) != selection:
        raise ValueError("selection changed after model execution started")
    if frozen_tree_digest(roots[0] / "reproduction/src/myagentrun") != manifest["source_hash"]:
        raise ValueError("frozen source snapshot hash mismatch")
    gate_index = json.loads((roots[0] / "gate-evidence-index.json").read_text())
    started = json.loads((roots[0] / "run_started.json").read_text())
    if gate_index["source_gate_fingerprint"] != started["fingerprint"]:
        raise ValueError("environment gates do not match the model execution fingerprint")
    expected_images = {row["instance_id"]: row["image"] for row in gate_index["tasks"]}
    if {row["instance_id"] for row in gate_index["tasks"]} != set(expected):
        raise ValueError("environment gate evidence does not cover all 20 cases")
    for gate in gate_index["tasks"]:
        for kind, must_resolve in (("reference", True), ("empty", False)):
            path = Path(gate[kind + "_report"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != gate[kind + "_sha256"]:
                raise ValueError("environment gate report hash mismatch")
            if json.loads(path.read_text())[gate["instance_id"]]["resolved"] is not must_resolve:
                raise ValueError("environment gate result does not match required behavior")
    selected = {}
    for root in roots:
        report = json.loads((root / "report.json").read_text())
        source = json.loads((root / "source-manifest.json").read_text())
        if source["source_hash"] != manifest["source_hash"]:
            raise ValueError("cannot merge different runtime source versions")
        if root != roots[0]:
            if (
                not authorization
                or root.resolve() != Path(authorization["continuation_root"]).resolve()
            ):
                raise ValueError("additional root is not the authorized continuation")
            subset_path = root / source["selection_path"]
            if hashlib.sha256(subset_path.read_bytes()).hexdigest() != source["selection_sha256"]:
                raise ValueError("continuation selection hash mismatch")
            subset = SmokeSelection.model_validate_json(subset_path.read_text())
            subset.verify(roots[0].parent / "sources")
            if {t.instance_id for t in subset.tasks} != set(authorization["unattempted_case_ids"]):
                raise ValueError("continuation changed the authorized unattempted tasks")
            if any(t != expected[t.instance_id] for t in subset.tasks):
                raise ValueError("continuation changed task inputs")
            if (
                subset.model_copy(
                    update={"tasks": selection.tasks, "total_tokens": selection.total_tokens}
                )
                != selection
            ):
                raise ValueError("continuation changed frozen execution budgets or environment")
            if (
                subset.total_tokens
                != selection.total_tokens - authorization["known_tokens_before_continuation"]
            ):
                raise ValueError("continuation nominal aggregate allowance is not accounted")
            if frozen_tree_digest(root / "reproduction/src/myagentrun") != manifest["source_hash"]:
                raise ValueError("continuation source snapshot mismatch")
        for key in (
            "model",
            "endpoint",
            "temperature",
            "thinking",
            "context_input_budget",
            "model_transport",
            "max_output_tokens",
            "max_stream_bytes",
            "model_timeout_seconds",
            "system_prompt",
            "tools",
            "skills",
            "verification",
            "memory",
        ):
            if report.get(key) != first.get(key):
                raise ValueError("cannot merge different execution configurations")
        for row in report["results"]:
            if row["instance_id"] not in expected:
                raise ValueError("unexpected case outside frozen manifest")
            if row["status"] == "not_run":
                continue
            if not isinstance(row.get("success"), bool):
                raise ValueError("not all attempted cases have independent evidence")
            if row["instance_id"] in selected:
                raise ValueError("duplicate attempts are not a single-shot result")
            selected[row["instance_id"]] = (root, row)
    if set(selected) != set(expected):
        raise ValueError("not all 20 frozen cases have been attempted and graded")
    key = tomllib.loads(credentials.read_text())["credentials"][
        "MYAGENTRUN_MOEROUTER_API_KEY"
    ].encode()
    if len(key) < MIN_SECRET_BYTES:
        raise ValueError("configured secret required for exact-byte audit")
    scores = []
    results = []
    secret_checks = []
    observed_paths = {
        "model_call_started": [],
        "tool_call_completed": [],
        "permission_decided": [],
        "context_built": [],
    }
    total = 0
    for task in selection.tasks:
        root, row = selected[task.instance_id]
        directory = root / "runs" / task.instance_id
        for isolation_path in (directory / "isolation.json", directory / "verifier/isolation.json"):
            isolation = json.loads(isolation_path.read_text())
            required_isolation = {
                "image": expected_images[task.instance_id],
                "mounts": [],
                "network": "none",
                "privileged": False,
                "memory_bytes": 1073741824,
                "nano_cpus": 1000000000,
                "pids_limit": 128,
            }
            if any(isolation.get(k) != v for k, v in required_isolation.items()):
                raise ValueError("case container differs from frozen isolation requirements")
            if (
                "ALL" not in isolation["cap_drop"]
                or "no-new-privileges" not in isolation["security_opt"]
            ):
                raise ValueError("case container lacks required privilege restrictions")
        trace = TraceRun.from_file(directory / "trace.jsonl")
        final = trace.final()
        if final is None or final.run_id != row["run_id"] or trace.status() != row["status"]:
            raise ValueError("row and finalized trace do not match")
        if not trace.by_name("model_call_started"):
            raise ValueError("case did not reach an actual model call")
        for event, cases in observed_paths.items():
            if trace.by_name(event):
                cases.append(task.instance_id)
        steps = [record.step_id for record in trace.records]
        if steps != sorted(steps):
            raise ValueError("non-monotonic trace step IDs")
        sidecar = json.loads((directory / "sidecar.json").read_text())
        if (
            sidecar["run_id"] != row["run_id"]
            or sidecar["task_digest"] != task.task_digest
            or sidecar["case_id"] != task.instance_id
            or sidecar["criterion_source"] != selection.official_commit
        ):
            raise ValueError("independent evidence is not bound to the frozen task and run")
        patch = directory / "model.patch"
        if sidecar["patch_sha256"] != hashlib.sha256(patch.read_bytes()).hexdigest():
            raise ValueError("captured patch hash mismatch")
        verifier = json.loads((directory / "verifier/report.json").read_text())[task.instance_id]
        if verifier["resolved"] is not row["success"] or sidecar["resolved"] is not row["success"]:
            raise ValueError("task success is not supported by the independent grader")
        log_meta = json.loads((directory / "verifier/log-metadata.json").read_text())
        raw = directory / "verifier/test_output.raw"
        if log_meta["raw_sha256"] != hashlib.sha256(raw.read_bytes()).hexdigest():
            raise ValueError("independent raw log hash mismatch")
        normalized = directory / "verifier/test_output.txt"
        if log_meta["normalized_sha256"] != hashlib.sha256(normalized.read_bytes()).hexdigest():
            raise ValueError("independent normalized log hash mismatch")
        clean = all(key not in p.read_bytes() for p in directory.rglob("*") if p.is_file())
        if not clean:
            raise ValueError("configured secret detected in run artifacts")
        secret_checks.append({"case_id": task.instance_id, "exact_secret_absent": clean})
        evidence = ScoringEvidence(
            run_id=row["run_id"],
            case_id=task.instance_id,
            source="official criterion report, captured patch hash and exact-secret byte scan",
            task_success=row["success"],
            budget=ScoringBudget(
                max_tokens=min(selection.task_tokens, selection.total_tokens - total),
                max_steps=selection.max_steps,
                max_tool_calls=selection.max_tool_calls,
                max_duration_seconds=selection.max_seconds,
            ),
            secret_checks={"exact_configured_secret_absent": clean},
        )
        metrics = score_harness(trace, evidence=evidence)
        if metrics.measurements["total_tokens"] != row["tokens"]:
            raise ValueError("Token report and raw Trace do not match")
        scores.append(metrics)
        write_json(directory / "scoring-evidence.json", evidence.model_dump(mode="json"))
        write_json(directory / "harness-metrics.json", metrics.to_json())
        tokens = row["tokens"]
        remaining_unknown = row.get("unknown_usage_calls", 0)
        unknown_steps = {
            event.step_id
            for event in trace.by_name("model_call_failed")
            if event.data.get("usage_known") is False
        }
        retained_unknown = bool(
            remaining_unknown
            and len(unknown_steps) == remaining_unknown
            and authorized_unknown(row["run_id"], unknown_steps, authorization)
        )
        if remaining_unknown and not retained_unknown:
            reconciled = json.loads((root / "usage-reconciliation.json").read_text())
            if reconciled["unknown_usage_remaining"]:
                raise ValueError("unreconciled usage remains unknown")
            reconcile = runpy.run_path(str(Path(__file__).with_name("reconcile_usage.py")))[
                "reconcile"
            ]
            verified_usage = reconcile(root, Path(reconciled["source"]))
            if (
                verified_usage["unknown_usage_remaining"]
                or verified_usage["source_sha256"] != reconciled["source_sha256"]
                or verified_usage["calls"] != reconciled["calls"]
            ):
                raise ValueError(
                    "usage sidecar is not supported by valid provider consumption records"
                )
            recovered = [call for call in reconciled["calls"] if call["run_id"] == row["run_id"]]
            expected_steps = {event.step_id for event in trace.by_name("model_call_started")}
            if (
                len(recovered) != len(expected_steps)
                or {c["step_id"] for c in recovered} != expected_steps
            ):
                raise ValueError("usage reconciliation does not cover every model call")
            if any(not isinstance(c["tokens"], int) or c["tokens"] < 0 for c in recovered):
                raise ValueError("usage reconciliation contains invalid token counts")
            tokens = sum(call["tokens"] for call in recovered)
            if tokens < row["tokens"]:
                raise ValueError("reconciliation discarded known failure consumption")
            remaining_unknown = 0
        total += tokens
        results.append(
            {
                **row,
                "latency": latency_from_trace(trace).to_json(),
                "known_tokens": tokens,
                "actual_tokens": None if remaining_unknown else tokens,
                "unknown_usage_retained_by_user_authorization": retained_unknown,
                "remaining_unknown_usage_calls": remaining_unknown,
                "source_root": str(root),
                "evidence_directory": str(directory),
                "independent_raw_log_sha256": log_meta["raw_sha256"],
            }
        )
    containers = {}
    for root in roots:
        for path in root.rglob("isolation.json"):
            name = json.loads(path.read_text())["container"]
            if name in containers:
                continue
            result = subprocess.run(
                ["docker", "inspect", name], capture_output=True, text=True, timeout=15, check=False
            )
            removed = result.returncode != 0 and "no such object" in result.stderr.lower()
            if not removed:
                raise ValueError("owned container cleanup is unverified")
            containers[name] = {"removed": True, "isolation_evidence": str(path)}
    successes = sum(row["success"] for row in results)
    states = Counter(row["status"] for row in results)
    failure_types = Counter()
    model_error_types = Counter()
    for row in results:
        reason = str(row.get("failure_reason") or "")
        matched = re.search(r"ModelAdapterError\[([a-z_]+)\]", reason)
        if matched:
            model_error_types[matched.group(1)] += 1
        if row["success"]:
            continue
        category = (
            matched.group(1)
            if matched
            else ("independent_tests_failed" if row["status"] == "finished" else row["status"])
        )
        failure_types[category] += 1
    return {
        "schema_version": "1.0",
        "dataset": "SWE-bench/SWE-bench_Lite",
        "protocol": "lite20_pipeline_v1",
        "status": "complete_with_unknown_usage"
        if any(r["remaining_unknown_usage_calls"] for r in results)
        else "complete",
        "protocol_amendment": authorization,
        "scope": "fixed 20 tasks from Django/Flask/pytest/SymPy, custom native ARM64 environment",
        "declared_tasks": len(expected),
        "attempted_tasks": len(results),
        "independently_graded_tasks": len(results),
        "successes": successes,
        "task_success_rate": {
            "value": successes / len(expected),
            "numerator": successes,
            "denominator": len(results),
            "eligible": len(expected),
            "coverage": 1.0,
        },
        "model": first["model"],
        "endpoint": first["endpoint"],
        "configuration": {
            k: first.get(k)
            for k in (
                "configuration",
                "context_input_budget",
                "model_transport",
                "max_output_tokens",
                "max_stream_bytes",
                "model_timeout_seconds",
                "tools",
                "skills",
                "verification",
                "memory",
            )
        },
        "source_hash": manifest["source_hash"],
        "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gate_evidence_index_sha256": hashlib.sha256(
            (roots[0] / "gate-evidence-index.json").read_bytes()
        ).hexdigest(),
        "loop_states": dict(states),
        "failure_types": dict(failure_types),
        "model_error_types_including_successful_patches": dict(model_error_types),
        "tasks_with_token_overrun": sum(
            row["known_tokens"] > selection.task_tokens for row in results
        ),
        "sum_task_token_overruns": sum(
            max(0, row["known_tokens"] - selection.task_tokens) for row in results
        ),
        "nominal_token_budget": selection.total_tokens,
        **usage_summary(results, selection.total_tokens),
        "unknown_usage_calls_in_raw_traces": sum(r.get("unknown_usage_calls", 0) for r in results),
        "harness_metrics": aggregate_harness_metrics(scores),
        "latency_totals_ms": {
            name: sum(row["latency"][name] for row in results)
            for name in ("total_ms", "model_ms", "tool_ms", "harness_ms")
        },
        "coverage_matrix": {
            "observed_event_paths": {
                name: {
                    "case_ids": cases,
                    "observed_runs": len(cases),
                    "declared_runs": len(expected),
                    "evidence": "runs/<case_id>/trace.jsonl",
                    "status": "observed; not full contract or safety acceptance",
                }
                for name, cases in observed_paths.items()
            },
            "independent_patch_grading": {
                "case_ids": list(expected),
                "checked_runs": len(results),
                "evidence": "runs/<case_id>/sidecar.json and verifier/report.json",
            },
            "unmeasured": [
                "skills",
                "verification extension",
                "memory",
                "recovery checkpoints",
                "context compaction",
                "unsafe and safe action probes",
                "process cleanup after cancellation",
            ],
        },
        "safety_audit": {
            "exact_configured_secret_checks": secret_checks,
            "owned_containers_removed": containers,
            "unmeasured": ["unknown secrets", "escape resistance", "unsafe action probes"],
        },
        "results": results,
    }


def markdown(report: dict) -> str:
    measurements = report["harness_metrics"]["measurements"]
    rates = report["harness_metrics"]["rates"]
    latency = report["latency_totals_ms"]
    model_errors = json.dumps(
        report["model_error_types_including_successful_patches"], ensure_ascii=False
    )
    unknown = report["unknown_usage_remaining"]
    usage_line = (
        f"已知消耗至少 {report['known_total_tokens']:,} Token，包含失败消耗；"
        f"仍有 {unknown} 次调用用量未知，完整实际消耗未测。"
        if unknown
        else (
            f"实际总消耗 {report['actual_total_tokens']:,} Token，包含失败消耗；未知用量剩余 0 次。"
        )
    )
    overrun = report["aggregate_token_overrun"]
    overrun_label = "未知，不能判定预算满足" if overrun is None else f"{overrun:,}"
    lines = [
        "# SWE-bench Lite 固定 20 题评测结果",
        "",
        (
            f"独立评分完成 {report['independently_graded_tasks']}/20，"
            f"修复成功 {report['successes']}/20。"
        ),
        usage_line,
        (
            "用户确认服务端故障后，按 [ADR 0035](adr/0035-lite20-provider-failure-continuation.md) "
            "续跑原清单未尝试的六题。此前 14 题未重跑；故障调用的未知用量保留，"
            "不能据此宣称原协议无偏离或完整 Token 预算满足。"
            if report.get("protocol_amendment")
            else ""
        ),
        "",
        "模型为 MoeRouter `glm-5.3-flash`；四个纯 Python 仓库、自定义原生 ARM64 环境，",
        "使用冻结官方测试判据和独立评分，保留原始日志与归一化记录。这不是官方全量成绩。",
        "Loop 终态与独立任务成功分别统计，不宣称未启用扩展或机制收益已经验收。",
        "",
        "## 执行与资源",
        "",
        f"Loop 终态分布：`{json.dumps(report['loop_states'], ensure_ascii=False)}`。",
        f"未修复任务的失败分类：`{json.dumps(report['failure_types'], ensure_ascii=False)}`。",
        (f"全部尝试中的模型调用错误（包含留下正确补丁的运行）：`{model_errors}`。"),
        (
            f"平均步骤 {measurements['avg_steps']:.2f}、"
            f"平均工具调用 {measurements['avg_tool_calls']:.2f}，"
            f"对应样本数分别为 {measurements['steps_sample_count']}、"
            f"{measurements['tool_calls_sample_count']}。"
        ),
        (
            f"Agent Loop 耗时 P50 {measurements['latency_p50_ms'] / 1000:.2f} 秒、"
            f"P95 {measurements['latency_p95_ms'] / 1000:.2f} 秒，"
            f"样本数 {measurements['latency_sample_count']}；采用排序后线性插值，"
            "不包含镜像准备和独立验证耗时。"
        ),
        (
            f"累计 Loop 耗时 {latency['total_ms'] / 1000:.2f} 秒，"
            f"其中模型调用 {latency['model_ms'] / 1000:.2f} 秒、"
            f"工具调用 {latency['tool_ms'] / 1000:.2f} 秒、"
            f"Harness 剩余开销 {latency['harness_ms'] / 1000:.2f} 秒。"
            "模型调用计时包含网络传输和服务等待，不等于纯模型计算时间。"
        ),
        (
            f"整轮名义 Token 预算 {report['nominal_token_budget']:,}，"
            f"实际整轮超量：{overrun_label}。"
            f"已知单题 Token 超限至少 {report['tasks_with_token_overrun']} 题，"
            f"已知单题超量合计 {report['sum_task_token_overruns']:,}。"
            "单题预算是否满足另见逐题 `harness-metrics.json` 中的 `constraint_checks`。"
        ),
        "",
        "| 指标 | 分子 / 已检查分母 | 声明机会数 | 证据覆盖率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in (
        "constraint_satisfaction",
        "constraint_satisfied_run_rate",
        "tool_error_rate",
        "invalid_tool_call_rate",
        "permission_denied_rate",
        "repeated_failed_action_rate",
    ):
        rate = rates[name]
        lines.append(
            f"| `{name}` | {rate['numerator']} / {rate['denominator']} | "
            f"{rate['eligible']} | {rate['coverage']} |"
        )
    lines.extend(
        [
            "",
            "工具错误含命令非零退出，不能直接归因为 Harness 缺陷；权限拒绝率不代表安全拦截率。",
            "上述约束比例仅覆盖四类执行预算；配置密钥另行做精确字节扫描，不代表全部安全约束已验收。",
            "",
            "## 逐题结果",
            "",
            "| 任务 | Loop 终态 | 独立评分 | Token（未知单列） |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for row in report["results"]:
        label = "通过" if row["success"] else "未通过"
        tokens_label = (
            f"至少 {row['known_tokens']:,}，另有未知"
            if row["actual_tokens"] is None
            else f"{row['actual_tokens']:,}"
        )
        lines.append(f"| `{row['instance_id']}` | {row['status']} | {label} | {tokens_label} |")
    lines.extend(
        [
            "",
            "## 路径覆盖与未测范围",
            "",
            "| 路径 | 观察到的任务数 / 声明任务数 | 证据 |",
            "| --- | ---: | --- |",
        ]
    )
    for name, item in report["coverage_matrix"]["observed_event_paths"].items():
        lines.append(
            f"| `{name}` | {item['observed_runs']} / {item['declared_runs']} | 原始 Trace |"
        )
    lines.extend(
        [
            "",
            "此表只证明路径被执行，不等同于模块契约、安全边界或机制收益验收。",
            "Skills、Verification 扩展、Memory、恢复检查点、Context 压缩、安全动作探针、",
            "取消后的进程清理均未测；逐项触发 case 与证据路径保存在机器报告的覆盖矩阵。",
            "",
            "配置、抽样与边界见 [Lite 20 题协议](../benchmarks/swebench_lite/v1/README.md)。",
            (
                "完整本机证据见 `benchmark-results/swebench-lite/v1/baseline/` 与续跑目录；"
                "机器报告为前者目录中的 `study-report.json`。"
            ),
            (
                "原始 Trace、Patch、独立评分、sidecar、源码快照和容器审计均留档；"
                "不提交密钥或原始 Trace。"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("benchmark-results/swebench-lite/v1/baseline")
    )
    parser.add_argument("--additional-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--selection", type=Path, default=Path("benchmarks/swebench_lite/v1/selection.json")
    )
    parser.add_argument(
        "--credentials-file", type=Path, default=Path(".myagentrun/credentials.toml")
    )
    parser.add_argument("--markdown", type=Path, default=Path("docs/swe-bench-lite20-results.md"))
    args = parser.parse_args()
    selection = SmokeSelection.model_validate_json(args.selection.read_text())
    report = audit(selection, [args.root, *args.additional_root], args.credentials_file)
    write_json(args.root / "study-report.json", report)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown(report))
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "declared_tasks",
                    "independently_graded_tasks",
                    "successes",
                    "actual_total_tokens",
                    "unknown_usage_remaining",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
