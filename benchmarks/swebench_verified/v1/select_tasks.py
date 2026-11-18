"""Freeze 50 Verified tasks by repository, before builds or model outcomes.

Run in the separate official SWE-bench tool environment (PyYAML required).
The original smoke task tree digest helper is reused without changing v1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import subprocess
from collections import defaultdict
from pathlib import Path

import yaml

SEED = "myagentrun_verified_50_v1_20261116"
COUNT = 50
POPULATION = 500
OFFICIAL_COMMIT = "02e7a74ffd0b707aab73d203fe87bdc7c76afc8e"
TASK_COMMIT = "3d07b464b7b311a0cbfb5ed5b2d8a3b96f84a33d"
EXCLUDED = {"pallets__flask-5014", "pytest-dev__pytest-10051", "sympy__sympy-20590"}


def rank(instance_id: str) -> str:
    return hashlib.sha256((SEED + ":" + instance_id).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "-C", str(args.task_repo), "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()
    if commit != TASK_COMMIT:
        raise ValueError("task repo commit mismatch")
    groups: dict[str, list[dict]] = defaultdict(list)
    total = 0
    for path in sorted((args.task_repo / "tasks").glob("*/task.yaml")):
        task = yaml.safe_load(path.read_text())
        if "SWE-bench/SWE-bench_Verified" in task.get("datasets", []) and task["split"] == "test":
            total += 1
            if task["instance_id"] not in EXCLUDED:
                groups[task["repo"]].append(task)
    if total != POPULATION:
        raise ValueError("expected exactly 500 frozen Verified test instances")
    # Minimum one per eligible repo; allocate the rest by available population,
    # using exact integer largest remainders and lexical ties. No result labels.
    remaining = COUNT - len(groups)
    capacity = {repo: len(rows) - 1 for repo, rows in groups.items()}
    denominator = sum(capacity.values())
    quotas = {repo: 1 + remaining * size // denominator for repo, size in capacity.items()}
    ties = sorted(capacity, key=lambda repo: (-(remaining * capacity[repo] % denominator), repo))
    for repo in ties[: COUNT - sum(quotas.values())]:
        quotas[repo] += 1
    digest = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "swebench_smoke/v1/prepare.py")
    )["tree_digest"]
    tasks = []
    for repo in sorted(groups):
        tasks.extend(
            {
                "instance_id": task["instance_id"],
                "repo": repo,
                "base_commit": task["base_commit"],
                "task_digest": digest(args.task_repo / "tasks" / task["instance_id"]),
            }
            for task in sorted(groups[repo], key=lambda row: rank(row["instance_id"]))[
                : quotas[repo]
            ]
        )
    assert len(tasks) == COUNT
    selection = {
        "schema_version": "1.0",
        "official_commit": OFFICIAL_COMMIT,
        "task_repo_commit": TASK_COMMIT,
        "environment": "custom_native_arm64",
        "context_input_budget": 65_536,
        "model_transport": "sse",
        "model_timeout_seconds": 300,
        "max_output_tokens": 16_384,
        "total_tokens": 50_000_000,
        "task_tokens": 1_000_000,
        "max_steps": 200,
        "max_tool_calls": 400,
        "max_seconds": 3600,
        "tasks": tasks,
    }
    sampling = {
        "schema_version": "1.0",
        "status": "selected_not_run",
        "seed": SEED,
        "method": "minimum_one_then_capacity_proportional_largest_remainder; SHA256 rank per repo",
        "verified_population": total,
        "eligible_population": total - len(EXCLUDED),
        "excluded_smoke_ids": sorted(EXCLUDED),
        "quotas": quotas,
        "eligible_repo_counts": {repo: len(rows) for repo, rows in sorted(groups.items())},
        "missing_repo": {"pallets/flask": "only Verified task already used for smoke debugging"},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (("selection.json", selection), ("sampling.json", sampling)):
        with (args.output / name).open("x") as stream:
            stream.write(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
