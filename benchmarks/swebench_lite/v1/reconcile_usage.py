# ruff: noqa: T201 -- bounded audit summary only; never print credentials or raw logs.
"""Recover missing response usage from the API key's own read-only router logs.

Original traces and reports stay immutable. Matching is conservative: unique
model/request timestamp inside the traced call, and exact token pairs whenever
the trace already knows usage. Unmatched calls remain unknown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from myagentrun.benchmark.public_swe import write_json

URL = "https://moerouter.com/api/log/token"
REQUEST_TIME_LENGTH = 14
CONSUMPTION_LOG_TYPE = 2


def reconcile(root: Path, provider_file: Path) -> dict:
    provider = json.loads(provider_file.read_text())["rows"]
    report = json.loads((root / "report.json").read_text())
    claimed = set()
    matches = []
    unknown = 0
    total = 0
    recovered = 0
    for trace in sorted(root.glob("runs/*/trace.jsonl")):
        records = [json.loads(line) for line in trace.read_text().splitlines()]
        if not any(row["name"] == "run_finished" for row in records):
            raise ValueError("cannot reconcile a nonterminal trace")
        starts = {r["step_id"]: r for r in records if r["name"] == "model_call_started"}
        finals = {
            r["step_id"]: r
            for r in records
            if r["name"] in {"model_call_completed", "model_call_failed"}
        }
        for step, start in starts.items():
            end = finals[step]
            known = end["data"].get("usage_known", True)
            low = datetime.fromisoformat(start["occurred_at"]) - timedelta(seconds=1)
            high = datetime.fromisoformat(end["occurred_at"]) + timedelta(seconds=1)
            candidates = []
            for row in provider:
                # New API error logs use zero placeholders, not measured usage.
                # Only consumption records with real token fields can settle
                # unknown usage for these nonempty coding requests.
                prompt = row.get("prompt_tokens")
                completion = row.get("completion_tokens")
                if (
                    row.get("type") != CONSUMPTION_LOG_TYPE
                    or type(prompt) is not int
                    or prompt <= 0
                    or type(completion) is not int
                    or completion < 0
                ):
                    continue
                request_id = row.get("request_id") or ""
                if (
                    len(request_id) < REQUEST_TIME_LENGTH
                    or request_id in claimed
                    or row["model_name"] != report["model"]
                ):
                    continue
                when = datetime.strptime(request_id[:REQUEST_TIME_LENGTH], "%Y%m%d%H%M%S").replace(
                    tzinfo=UTC
                )
                if not low <= when <= high:
                    continue
                if known and (end["data"]["input_tokens"], end["data"]["output_tokens"]) != (
                    row["prompt_tokens"],
                    row["completion_tokens"],
                ):
                    continue
                candidates.append(row)
            if len(candidates) > 1:
                raise ValueError("ambiguous provider usage records")
            matched = candidates[0] if candidates else None
            if matched:
                claimed.add(matched["request_id"])
                tokens = matched["prompt_tokens"] + matched["completion_tokens"]
                total += tokens
                recovered += 0 if known else tokens
            elif known:
                tokens = end["data"]["input_tokens"] + end["data"]["output_tokens"]
                total += tokens
            else:
                tokens = None
                unknown += 1
            matches.append(
                {
                    "case_id": trace.parent.name,
                    "run_id": start["run_id"],
                    "step_id": step,
                    "trace_usage_known": known,
                    "provider_match": matched,
                    "tokens": tokens,
                }
            )
    return {
        "schema_version": "1.0",
        "source": str(provider_file),
        "source_sha256": hashlib.sha256(provider_file.read_bytes()).hexdigest(),
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "method": (
            "single-agent sequential call intervals +/-1 second; exact known token pair check; "
            "only type=2 consumption records with positive prompt tokens and nonnegative output"
        ),
        "known_tokens": total,
        "actual_total_tokens": total if not unknown else None,
        "recovered_tokens": recovered,
        "unknown_usage_remaining": unknown,
        "original_trace_unchanged": True,
        "calls": matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--provider-log", type=Path)
    parser.add_argument(
        "--credentials-file", type=Path, default=Path(".myagentrun/credentials.toml")
    )
    args = parser.parse_args()
    output = args.root / "usage-reconciliation.json"
    if output.exists():
        raise ValueError("preserve existing reconciliation; use a new output revision")
    source = args.provider_log
    if source is None:
        source = args.root / "provider-log-usage.json"
        if source.exists():
            raise ValueError("preserve existing provider logs; pass --provider-log to use them")
        key = tomllib.loads(args.credentials_file.read_text())["credentials"][
            "MYAGENTRUN_MOEROUTER_API_KEY"
        ]
        with httpx.Client(timeout=30) as client:
            response = client.get(URL, headers={"Authorization": "Bearer " + key})
            response.raise_for_status()
            data = response.json()["data"]
        rows = [
            {
                k: row.get(k)
                for k in (
                    "request_id",
                    "model_name",
                    "prompt_tokens",
                    "completion_tokens",
                    "created_at",
                    "type",
                    "use_time",
                    "is_stream",
                )
            }
            for row in data
        ]
        write_json(source, {"source": URL, "rows": rows})
    result = reconcile(args.root, source)
    (args.root / "reconciliation-scorer.py").write_bytes(Path(__file__).read_bytes())
    write_json(output, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "known_tokens",
                    "actual_total_tokens",
                    "recovered_tokens",
                    "unknown_usage_remaining",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
