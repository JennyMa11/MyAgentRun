"""Conservative native Pi usage reconciliation from private, sanitized router logs."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

CONSUMPTION_TYPE = 2
REQUEST_TIMESTAMP_LENGTH = 14


def reconcile(root: Path, provider_file: Path) -> dict[str, Any]:
    """Use only unique valid consumption records, never error-log placeholders."""
    provider = json.loads(provider_file.read_text())["rows"]
    model = json.loads((root / "protocol.json").read_text())["model"]
    calls = []
    claimed = set()
    for trace in sorted(root.glob("runs/*/trace.jsonl")):
        events = [json.loads(line) for line in trace.read_text().splitlines()]
        if not any(e["name"] == "run_finished" for e in events):
            raise ValueError("nonterminal Pi trace")
        starts = {e["step_id"]: e for e in events if e["name"] == "model_call_started"}
        ends = {e["step_id"]: e for e in events if e["name"] == "model_call_completed"}
        for step, start in starts.items():
            end = ends[step]
            known = end["data"]["usage_known"]
            low = datetime.fromisoformat(start["occurred_at"]) - timedelta(seconds=1)
            high = datetime.fromisoformat(end["occurred_at"]) + timedelta(seconds=1)
            candidates = []
            for row in provider:
                prompt, completion = row.get("prompt_tokens"), row.get("completion_tokens")
                if (
                    row.get("type") != CONSUMPTION_TYPE
                    or type(prompt) is not int
                    or prompt <= 0
                    or type(completion) is not int
                    or completion < 0
                    or row.get("model_name") != model
                ):
                    continue
                request = row.get("request_id") or ""
                if len(request) < REQUEST_TIMESTAMP_LENGTH or request in claimed:
                    continue
                when = datetime.strptime(request[:14], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
                if not low <= when <= high:
                    continue
                if known and (end["data"]["input_tokens"], end["data"]["output_tokens"]) != (
                    prompt,
                    completion,
                ):
                    continue
                candidates.append(row)
            if len(candidates) > 1:
                raise ValueError("ambiguous provider consumption records")
            match = candidates[0] if candidates else None
            if match:
                claimed.add(match["request_id"])
            tokens = (
                (end["data"]["input_tokens"] + end["data"]["output_tokens"])
                if known
                else (match["prompt_tokens"] + match["completion_tokens"] if match else None)
            )
            calls.append(
                {
                    "case_id": trace.parent.name,
                    "run_id": start["run_id"],
                    "step_id": step,
                    "trace_usage_known": known,
                    "tokens": tokens,
                    "provider_match": match,
                }
            )
    unknown = sum(call["tokens"] is None for call in calls)
    known_tokens = sum(call["tokens"] or 0 for call in calls)
    return {
        "schema_version": "1.0",
        "source": str(provider_file),
        "source_sha256": hashlib.sha256(provider_file.read_bytes()).hexdigest(),
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "method": (
            "unique matching model/request start inside call +/-1s; "
            "type2 positive-input usage; exact known token pair"
        ),
        "calls": calls,
        "known_tokens": known_tokens,
        "actual_total_tokens": None if unknown else known_tokens,
        "unknown_usage_remaining": unknown,
        "original_trace_unchanged": True,
    }
