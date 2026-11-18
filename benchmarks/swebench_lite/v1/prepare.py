"""Freeze and prepare 20 Lite repository tasks before any model results.

Run in the independent official SWE-bench environment (PyYAML required).
Repository quotas target pure Python environment coverage, not all Lite repos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
import shutil
import subprocess
from pathlib import Path

import yaml

TASK_COMMIT = "3d07b464b7b311a0cbfb5ed5b2d8a3b96f84a33d"
OFFICIAL_COMMIT = "02e7a74ffd0b707aab73d203fe87bdc7c76afc8e"
SEED = "myagentrun_lite20_pipeline_v1_20261116"
QUOTAS = {"django/django": 8, "pallets/flask": 2, "pytest-dev/pytest": 4, "sympy/sympy": 6}
EXCLUDED = {"pallets__flask-5014", "pytest-dev__pytest-10051", "sympy__sympy-20590"}
POPULATION = 300


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "-C", str(args.task_repo), "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()
    if commit != TASK_COMMIT:
        raise ValueError("task repo commit mismatch")
    groups: dict[str, list[dict]] = {repo: [] for repo in QUOTAS}
    population = 0
    for path in sorted((args.task_repo / "tasks").glob("*/task.yaml")):
        task = yaml.safe_load(path.read_text())
        if "SWE-bench/SWE-bench_Lite" not in task.get("datasets", []) or task["split"] != "test":
            continue
        population += 1
        if task["repo"] in groups and task["instance_id"] not in EXCLUDED:
            groups[task["repo"]].append(task)
    if population != POPULATION:
        raise ValueError("expected 300 Lite test tasks")
    helper = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "swebench_smoke/v1/prepare.py")
    )
    digest = helper["tree_digest"]
    tasks = []
    inventory = []
    for repo, quota in QUOTAS.items():
        ranked = sorted(
            groups[repo],
            key=lambda task: hashlib.sha256(
                (SEED + ":" + task["instance_id"]).encode()
            ).hexdigest(),
        )
        if len(ranked) < quota:
            raise ValueError("insufficient tasks for frozen quota")
        for task in ranked[:quota]:
            identity = task["instance_id"]
            original = args.task_repo / "tasks" / identity
            copied = args.artifacts / "sources/tasks" / identity
            shutil.copytree(original, copied)
            tasks.append(
                {
                    "instance_id": identity,
                    "repo": repo,
                    "base_commit": task["base_commit"],
                    "task_digest": digest(copied),
                }
            )
            docker = (original / "Dockerfile").read_text()
            start = docker.index("name: testbed\n")
            end = docker.index("\nEOF_", start)
            upstream = yaml.safe_load(docker[start:end])
            py = next(
                entry.split("=")[1]
                for entry in upstream["dependencies"]
                if isinstance(entry, str) and entry.startswith("python=")
            )
            inventory.append(
                {
                    "instance_id": identity,
                    "repo": repo,
                    "version": task["version"],
                    "upstream_python": py,
                    "dependency_spec": upstream,
                    "dockerfile_sha256": hashlib.sha256(docker.encode()).hexdigest(),
                    "test_command": re.split(
                        r": '>>>>> (?:Start|End) Test Output'", (original / "eval.sh").read_text()
                    )[1].strip(),
                }
            )
    selection = {
        "schema_version": "1.0",
        "official_commit": OFFICIAL_COMMIT,
        "task_repo_commit": TASK_COMMIT,
        "environment": "custom_native_arm64",
        "context_input_budget": 65_536,
        "model_transport": "sse",
        "model_timeout_seconds": 300,
        "max_output_tokens": 16_384,
        "max_stream_bytes": 32 * 1024 * 1024,
        "verification_timeout_seconds": 900,
        "total_tokens": 20_000_000,
        "task_tokens": 1_000_000,
        "max_steps": 200,
        "max_tool_calls": 400,
        "max_seconds": 3600,
        "tasks": tasks,
    }
    sampling = {
        "schema_version": "1.0",
        "seed": SEED,
        "lite_test_population": population,
        "scope": "20 tasks from four pure-Python repos for end-to-end pipeline validation",
        "method": "fixed repository quotas; SHA256(seed:instance_id) rank per repository",
        "quotas": QUOTAS,
        "excluded_prior_smoke_ids": sorted(EXCLUDED),
        "model_outcomes_used_for_selection": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, body in (("selection.json", selection), ("sampling.json", sampling)):
        with (args.output / name).open("x") as stream:
            stream.write(json.dumps(body, indent=2) + "\n")
    (args.artifacts / "environment-inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
