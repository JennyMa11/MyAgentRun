# ruff: noqa: T201 -- bounded offline scoring metadata only
"""Parse preserved completed test logs with the pinned evaluator venv; no model/test reruns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from myagentrun.benchmark.public_swe import write_json
from myagentrun.benchmark.swebench_smoke import SmokeSelection


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repair(  # noqa: PLR0912, PLR0915 -- explicit audited score derivation
    root: Path, evaluator: Path
) -> dict[str, Any]:
    """Write derived scores and receipts without changing original execution artifacts."""
    original_report = (root / "report.json").read_bytes()
    original_hash = hashlib.sha256(original_report).hexdigest()
    raw = json.loads(original_report)
    protocol = json.loads((root / "protocol.json").read_text())
    if sha(root / "protocol.json") != (root / "protocol.sha256").read_text().strip():
        raise ValueError("frozen protocol changed")
    selection = SmokeSelection.model_validate_json((root / "selection.json").read_text())
    selection.verify(Path(protocol["sources"]))
    api = runpy.run_path(str(Path(__file__).with_name("run.py")))
    helper = root / "reproduction/benchmarks/swebench_smoke/v1/official_bridge.py"
    target = root / "scoring-repair/v2"
    target.mkdir(parents=True, exist_ok=True)
    snapshots = target / "source-reports"
    snapshots.mkdir(exist_ok=True)
    snapshot = snapshots / (original_hash + ".json")
    if not snapshot.exists():
        snapshot.write_bytes(original_report)
    scorers = target / "scorers"
    scorers.mkdir(exist_ok=True)
    for source in (Path(__file__), Path(__file__).with_name("run.py")):
        frozen = scorers / (sha(source) + ".py")
        if not frozen.exists():
            shutil.copy2(source, frozen)
    for row in raw["results"]:
        if row["status"] in {"not_run", "running"} or isinstance(row["success"], bool):
            continue
        directory = Path(row["evidence_directory"])
        patch = directory / "model.patch"
        original = directory / "verifier"
        required = [
            original / name
            for name in ("test_output.raw", "test_output.txt", "eval.sh", "log-metadata.json")
        ]
        if not patch.is_file() or not all(p.is_file() for p in required):
            continue
        raw_log, log, script, metadata_file = required
        metadata = json.loads(metadata_file.read_text())
        if sha(raw_log) != metadata["raw_sha256"] or sha(log) != metadata["normalized_sha256"]:
            raise ValueError("preserved log hash mismatch")
        markers = re.findall(r"myagentrun_eval_completed_[a-f0-9]{32}", script.read_text())
        if not markers or not all(marker.encode() in raw_log.read_bytes() for marker in markers):
            continue
        output = target / "runs" / row["strategy"] / row["instance_id"]
        output.mkdir(parents=True, exist_ok=True)
        receipt_file = output / "sidecar.json"
        binding = {
            "case_id": row["instance_id"],
            "run_id": row["run_id"],
            "strategy": row["strategy"],
            "criterion_source": selection.official_commit,
            "task_digest": next(
                t.task_digest for t in selection.tasks if t.instance_id == row["instance_id"]
            ),
            "patch_sha256": sha(patch),
            "raw_log_sha256": sha(raw_log),
            "normalized_log_sha256": sha(log),
            "metadata_sha256": sha(metadata_file),
            "original_result_sha256": sha(directory / "result.json"),
            "protocol_sha256": sha(root / "protocol.json"),
            "official_helper_sha256": sha(helper),
            "evaluator_entry": str(evaluator),
        }
        if receipt_file.exists():
            receipt = json.loads(receipt_file.read_text())
            if any(receipt[k] != v for k, v in binding.items()):
                raise ValueError("scoring repair source changed")
        else:
            report_path = output / "report.json"
            process = subprocess.run(
                [
                    str(evaluator),
                    str(helper),
                    str(Path(protocol["sources"]) / "tasks" / row["instance_id"]),
                    str(log),
                    str(patch),
                    str(report_path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                env={k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "LANG"}},
            )
            if process.returncode:
                raise RuntimeError("pinned offline grading helper failed")
            result = json.loads(report_path.read_text())[row["instance_id"]]
            if not result["patch_successfully_applied"] or not isinstance(result["resolved"], bool):
                continue
            receipt = {
                **binding,
                "resolved": result["resolved"],
                "scorer_sha256": sha(Path(__file__)),
                "method": "parse existing completed test log; no model call or test execution",
                "report_sha256": sha(report_path),
            }
            write_json(receipt_file, receipt)
        if sha(output / "report.json") != receipt["report_sha256"]:
            raise ValueError("derived grading report changed")
        row.update(
            success=receipt["resolved"],
            evaluation_status="graded_from_preserved_log",
            original_evaluation_status=row.get("evaluation_status"),
            scoring_repair_sidecar=str(receipt_file),
        )
    raw["original_report_sha256"] = original_hash
    raw["original_report_path"] = str(snapshot)
    raw["scoring_repair"] = {
        "reason": "evaluator.resolve() followed venv python symlink outside pyvenv.cfg context",
        "correct_evaluator_entry": str(evaluator),
        "model_calls": 0,
        "test_reruns": 0,
        "scorer_sha256": sha(Path(__file__)),
        "summary_code_sha256": sha(Path(__file__).with_name("run.py")),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    api["write_report"](target, raw, selection)
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark-results/swebench-context/v1"))
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    root, evaluator = args.root.resolve(), args.evaluator.absolute()
    last = None
    while True:
        report = repair(root, evaluator)
        summary = {
            s: {k: v[k] for k in ("attempted", "graded", "passed", "known_tokens")}
            for s, v in report["arms"].items()
        }
        if summary != last:
            print(json.dumps(summary), flush=True)
            last = summary
        if not args.watch or report["status"] == "execution_complete":
            break
        time.sleep(10)


if __name__ == "__main__":
    main()
