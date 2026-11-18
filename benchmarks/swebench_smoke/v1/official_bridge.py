"""Run with the separately installed, pinned official SWE-bench environment.

This helper never runs inside the agent container. No project dependency on
SWE-bench is introduced; imports below belong to the independent scoring tool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swebench.harness.grading import get_eval_report
from swebench.task.repo import load_task
from swebench.types import TestSpec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    parser.add_argument("log", type=Path)
    parser.add_argument("patch", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    task = load_task(args.task)
    spec = TestSpec(
        instance_id=task["instance_id"],
        image=task["image"],
        eval_script_list=task["eval_script"].splitlines(),
        repo=task["repo"],
        version=task["version"],
        FAIL_TO_PASS=task["FAIL_TO_PASS"],
        PASS_TO_PASS=task["PASS_TO_PASS"],
        log_parser=task["log_parser"],
        eval_type=task["eval_type"],
    )
    prediction = {
        "instance_id": task["instance_id"],
        "model_name_or_path": "myagentrun_smoke_v1",
        "model_patch": args.patch.read_text(),
    }
    report = get_eval_report(spec, prediction, str(args.log), include_tests_status=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
