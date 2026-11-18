# ruff: noqa: T201 -- experiment status only
"""Run the repeated, paired Lite20 study of tool pruning and model handoffs."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import random
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.swebench_context.v1 import run as sweep

from myagentrun.benchmark.hashes import frozen_tree_digest
from myagentrun.benchmark.public_swe import write_json
from myagentrun.benchmark.swebench_smoke import (
    SmokeSelection,
    configured_profile,
    grade_patch,
    provider_factory,
)

STRATEGIES = ("summary", "summary_pruned", "summary_model")
REPEATS = 2
BOOTSTRAP_SAMPLES = 2_000
PAIRS = (
    ("tool_result_pruning", "summary", "summary_pruned"),
    ("model_handoff", "summary_pruned", "summary_model"),
)


def schedule(identities: list[str]) -> list[dict[str, Any]]:
    """Rotate all six execution orders, with task identity and repeat as pair keys."""
    orders = tuple(itertools.permutations(STRATEGIES))
    return [
        {
            "strategy": strategy,
            "instance_id": identity,
            "repeat": repeat,
            "status": "not_run",
            "success": None,
            "tokens": 0,
            "unknown_usage_calls": 0,
        }
        for repeat in range(1, REPEATS + 1)
        for index, identity in enumerate(identities)
        for strategy in orders[index % len(orders)][:: (1 if repeat % 2 else -1)]
    ]


def _paired_interval(task_deltas: dict[str, list[int]]) -> list[float] | None:
    if len(task_deltas) < 2:  # noqa: PLR2004 -- at least two clusters
        return None
    means = [sum(values) / len(values) for values in task_deltas.values()]
    random_source = random.Random(20261118)
    dist = sorted(
        sum(random_source.choices(means, k=len(means))) / len(means)
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return [dist[int(0.025 * BOOTSTRAP_SAMPLES)], dist[int(0.975 * BOOTSTRAP_SAMPLES) - 1]]


def compare(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    """Keep missing scores and unknown usage distinct on the fixed pair denominator."""
    index = {(r["strategy"], r["instance_id"], r["repeat"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate attempt in pair manifest")
    outcomes: Counter[str] = Counter()
    deltas: dict[str, list[int]] = {}
    pairs = []
    for a in (r for r in rows if r["strategy"] == left):
        b = index[(right, a["instance_id"], a["repeat"])]
        first, second = a["success"], b["success"]
        outcome = (
            "unscored"
            if not isinstance(first, bool) or not isinstance(second, bool)
            else {
                (True, True): "both_pass",
                (True, False): "a_only",
                (False, True): "b_only",
                (False, False): "neither",
            }[(first, second)]
        )
        outcomes[outcome] += 1
        usage_complete = all(
            r["status"] not in {"not_run", "running"}
            and r.get("usage_audited", False)
            and not r["unknown_usage_calls"]
            for r in (a, b)
        )
        delta = b["tokens"] - a["tokens"] if usage_complete else None
        if delta is not None:
            deltas.setdefault(a["instance_id"], []).append(delta)
        pairs.append(
            {
                "task": a["instance_id"],
                "repeat": a["repeat"],
                "a_run_id": a.get("run_id"),
                "b_run_id": b.get("run_id"),
                "score_outcome": outcome,
                "actual_token_delta_b_minus_a": delta,
                "a_known_tokens": a["tokens"],
                "b_known_tokens": b["tokens"],
            }
        )
    full_clusters = {k: v for k, v in deltas.items() if len(v) == REPEATS}
    return {
        "a": left,
        "b": right,
        "declared_pairs": len(pairs),
        "outcomes": dict(outcomes),
        "pairs_with_complete_usage": sum(map(len, deltas.values())),
        "complete_task_clusters": len(full_clusters),
        "mean_token_delta_complete_clusters": (
            sum(sum(values) / len(values) for values in full_clusters.values()) / len(full_clusters)
            if full_clusters
            else None
        ),
        "descriptive_95pct_token_delta_complete_clusters": _paired_interval(full_clusters),
        "pairs": pairs,
    }


def audit_attempt(row: dict[str, Any], protocol: dict[str, Any], root: Path) -> None:
    """Bind trace and verification evidence, keeping any incompleteness explicit."""
    directory = Path(row["evidence_directory"])
    trace = directory / "trace.jsonl"
    events = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    counts = Counter(x["name"] for x in events if x["data"].get("purpose") == "context_summary")
    row["summary_calls"] = counts["model_call_started"]
    row["summary_known_tokens"] = sum(
        x["data"]["input_tokens"] + x["data"]["output_tokens"]
        for x in events
        if x["name"] in {"model_call_completed", "model_call_failed"}
        and x["data"].get("purpose") == "context_summary"
        and x["data"].get("usage_known")
    )
    row["valid_summaries"] = sum(
        x["name"] == "context_summary_resolved" and x["data"].get("accepted") is True
        for x in events
    )
    row["summary_use_steps"] = sum(
        x["name"] == "context_built"
        and x["data"].get("metadata", {}).get("model_summary_active") is True
        for x in events
    )
    row["pruned_result_steps"] = sum(
        bool(x["data"].get("metadata", {}).get("pruned_tool_result_indices"))
        for x in events
        if x["name"] == "context_built"
    )
    ends = [e for e in events if e["name"] == "run_finished"]
    if trace.exists():
        row["trace_sha256"] = sweep.sha(trace)
    row["usage_audited"] = bool(
        len(ends) == 1
        and ends[0]["data"]["tokens"] == row["tokens"]
        and ends[0]["data"].get("unknown_usage_calls") == row["unknown_usage_calls"]
    )
    if isinstance(row["success"], bool):
        verifier = directory / "verifier"
        payload = json.loads((verifier / "report.json").read_text())
        sidecar = json.loads((directory / "sidecar.json").read_text())
        if (
            sidecar["run_id"] != row["run_id"]
            or sidecar["protocol_sha256"] != sweep.sha(root / "protocol.json")
            or sidecar["criterion_source"] != protocol["official_commit"]
            or sidecar["patch_sha256"] != sweep.sha(directory / "model.patch")
        ):
            raise ValueError("scoring evidence mismatch")
        independent = (
            False
            if payload.get("failure_type") == "patch_apply_failed"
            else payload[row["instance_id"]]["resolved"]
        )
        if independent is not row["success"]:
            raise ValueError("independent grader mismatch")
        row["verifier_sha256"] = sweep.sha(verifier / "report.json")


def publish(root: Path, protocol: dict[str, Any], rows: list[dict[str, Any]], state: str) -> None:
    summaries = {}
    for strategy in STRATEGIES:
        subset = [r for r in rows if r["strategy"] == strategy]
        summary = sweep.summarize_arm(subset, protocol["nominal_tokens_per_strategy"])
        for key in (
            "summary_calls",
            "summary_known_tokens",
            "valid_summaries",
            "summary_use_steps",
            "pruned_result_steps",
        ):
            summary[key] = sum(r.get(key, 0) for r in subset)
        if any(
            r["status"] not in {"not_run", "running"} and not r.get("usage_audited") for r in subset
        ):
            summary["actual_total_tokens"] = None
            summary["total_budget_satisfied"] = (
                False if summary["known_tokens"] > summary["nominal_tokens"] else None
            )
        summaries[strategy] = summary
    comparisons = {name: compare(rows, left, right) for name, left, right in PAIRS}
    write_json(
        root / "report.json",
        {
            "status": state,
            "updated_at": datetime.now(UTC).isoformat(),
            "protocol_sha256": sweep.sha(root / "protocol.json"),
            "arms": summaries,
            "comparisons": comparisons,
            "results": rows,
        },
    )
    lines = [
        "# Summary Lite20 配对评测",
        "",
        f"状态 {state}",
        "",
        "| 策略 | 已尝试 | 已评分 | 通过 | 已知 Token | 未知调用 | 摘要调用 / 有效 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy, a in summaries.items():
        lines.append(
            f"| {strategy} | {a['attempted']}/{a['declared']} | {a['graded']} | "
            f"{a['passed']} | {a['known_tokens']:,} | {a['unknown_usage_calls']} | "
            f"{a['summary_calls']} / {a['valid_summaries']} |"
        )
    lines += [
        "",
        "每个策略使用固定20题、每题两次。Token包含摘要与失败调用的已知用量。",
        "完整配对、审核状态、结果与区间见同目录 report.json。",
    ]
    temp = root / "progress.tmp"
    temp.write_text("\n".join(lines) + "\n")
    temp.replace(root / "progress.md")


async def prepare(root: Path, project: Path, evaluator: Path) -> None:
    """Freeze the new comparison and exercise the same independent grader offline."""
    base = project / "benchmark-results/swebench-lite/v1/baseline"
    await asyncio.to_thread(root.mkdir, parents=True, exist_ok=False)
    snapshot = root / "reproduction"
    shutil.copytree(project / "src/myagentrun", snapshot / "src/myagentrun")
    for path in (
        "benchmarks/swebench_context/v1/run.py",
        "benchmarks/swebench_context/v2/run.py",
        "benchmarks/swebench_smoke/v1/official_bridge.py",
        "myagentrun.toml",
        "uv.lock",
    ):
        destination = snapshot / path
        destination.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(project / path, destination)
    shutil.copy2(project / "benchmarks/swebench_lite/v1/selection.json", root / "selection.json")
    shutil.copy2(base / "images.json", root / "images.json")
    gates = json.loads((base / "gate-evidence-index.json").read_text())
    for gate in gates["tasks"]:
        for key in ("reference_report", "empty_report"):
            gate[key] = str(project / gate[key])
    write_json(root / "gate-evidence-index.json", gates)
    selection = SmokeSelection.model_validate_json((root / "selection.json").read_text())
    if len(selection.tasks) != sweep.TASK_COUNT:
        raise ValueError("study requires the fixed Lite20 set")
    sources = project / "benchmark-results/swebench-lite/v1/sources"
    profile = configured_profile(snapshot / "myagentrun.toml")
    protocol = {
        "schema_version": "2.0",
        "frozen_at": datetime.now(UTC).isoformat(),
        "strategies": STRATEGIES,
        "repeats": REPEATS,
        "comparisons": PAIRS,
        "nominal_tokens_per_strategy": REPEATS * selection.total_tokens,
        "selection_sha256": sweep.sha(root / "selection.json"),
        "images_sha256": sweep.sha(root / "images.json"),
        "gate_index_sha256": sweep.sha(root / "gate-evidence-index.json"),
        "reproduction_hash": frozen_tree_digest(snapshot),
        "sources": str(sources),
        "evaluator": str(evaluator),
        "official_commit": selection.official_commit,
        "model": profile.model,
        "endpoint": profile.endpoint,
        "temperature": 0,
        "unknown_policy": "retain unknown, reserve nominal task allowance, continue other attempts",
        "scheduling": "adjacent A/B/C per task and repeat, six rotated orders, one container",
        "history": "natural trajectory without injected history",
        "interval": "2000 task-cluster resamples using task means, only complete usage clusters",
    }
    write_json(root / "protocol.json", protocol)
    (root / "protocol.sha256").write_text(sweep.sha(root / "protocol.json") + "\n")
    sweep.verify_protocol(root)
    images = json.loads((root / "images.json").read_text())
    if {gate["instance_id"]: gate["image"] for gate in gates["tasks"]} != images:
        raise ValueError("reused gates do not cover the pinned image mapping")
    task = selection.tasks[0]
    image = images[task.instance_id]
    for name, patch, expected in (
        ("reference", sources / "tasks" / task.instance_id / "gold.patch", True),
        ("empty", base / "empty.patch", False),
    ):
        observed = await grade_patch(
            task,
            image=image,
            sources=sources,
            patch=patch,
            output=root / "preflight" / name,
            evaluator=evaluator,
            timeout_seconds=selection.verification_timeout_seconds,
        )
        if observed is not expected:
            raise ValueError("independent grader rehearsal produced unexpected score")
    write_json(root / "preflight/passed.json", {"reference": True, "empty": False})
    publish(root, protocol, schedule([t.instance_id for t in selection.tasks]), "prepared")
    print("Prepared 120 frozen attempts")


async def run(root: Path, credentials: Path) -> None:
    protocol = sweep.verify_protocol(root)
    if not (root / "preflight/passed.json").is_file():
        raise ValueError("offline grader rehearsal required")
    selection = SmokeSelection.model_validate_json((root / "selection.json").read_text())
    rows = schedule([t.instance_id for t in selection.tasks])
    with (root / "run_started.json").open("x") as claim:
        json.dump({"pid": os.getpid(), "protocol_sha256": sweep.sha(root / "protocol.json")}, claim)
    os.chdir(root / "reproduction")
    provider_factory(credentials, selection)
    for row in rows:
        arm = [r for r in rows if r["strategy"] == row["strategy"]]
        allowance = sweep.remaining_allowance(
            arm,
            total=protocol["nominal_tokens_per_strategy"],
            per_task=selection.task_tokens,
        )
        if allowance:
            sweep.verify_protocol(root)
            row["status"] = "running"
            publish(root, protocol, rows, "running")
            await sweep.run_case(root, row, selection, protocol, credentials, allowance)
            audit_attempt(row, protocol, root)
        else:
            row["failure_reason"] = "arm_nominal_allowance_exhausted"
        publish(root, protocol, rows, "running")
        print(
            {
                k: row.get(k)
                for k in (
                    "instance_id",
                    "strategy",
                    "repeat",
                    "status",
                    "success",
                    "tokens",
                    "summary_calls",
                )
            },
            flush=True,
        )
    publish(root, protocol, rows, "execution_complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, default=Path("benchmark-results/swebench-context/v2"))
    parser.add_argument("--evaluator", type=Path)
    parser.add_argument(
        "--credentials-file", type=Path, default=Path(".myagentrun/credentials.toml")
    )
    args = parser.parse_args()
    root, credential = args.root.resolve(), args.credentials_file.resolve()
    if args.action == "prepare":
        if args.evaluator is None:
            parser.error("prepare requires the independent evaluator")
        asyncio.run(prepare(root, Path.cwd().resolve(), sweep.evaluator_path(args.evaluator)))
    else:
        if Path(__file__).resolve() != root / "reproduction/benchmarks/swebench_context/v2/run.py":
            parser.error("run from the frozen source snapshot")
        asyncio.run(run(root, credential))


if __name__ == "__main__":
    main()
