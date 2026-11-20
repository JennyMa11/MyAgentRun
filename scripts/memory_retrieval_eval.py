"""Score the frozen retrieval gold set under both selectors and archive the report.

This is a ranking measurement, not a task run: it starts no model, no tool loop and
no container, and it therefore cannot support a claim about task success.

    uv run python scripts/memory_retrieval_eval.py
    uv run python scripts/memory_retrieval_eval.py --baseline   # v1 only, for comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from myagentrun.domain.memory_ranking import SELECTOR_V1, SELECTOR_V2
from myagentrun.evaluation.retrieval_gold import GOLD_SET_PATH, load_gold_set, report

DESTINATION = Path("docs/results/memory-retrieval-2026-11-19.json")


def main() -> None:
    """Write the paired selector comparison next to its frozen input."""
    parser = argparse.ArgumentParser(description="Ranking metrics for project-memory discovery")
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--baseline", action="store_true", help="score the frozen v1 selector only")
    parser.add_argument("--gold", type=Path, default=GOLD_SET_PATH)
    arguments = parser.parse_args()
    gold_set = load_gold_set(arguments.gold)
    selectors = (SELECTOR_V1,) if arguments.baseline else (SELECTOR_V1, SELECTOR_V2)
    payload = report(gold_set, selectors)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    reports = payload["reports"]
    assert isinstance(reports, list)
    for entry in reports:
        assert isinstance(entry, dict)
        hit = entry["hit_at_5"]
        assert isinstance(hit, dict)
        sys.stdout.write(
            f"{entry['selector']}: hit@5={hit['numerator']}/{hit['denominator']} "
            f"mrr@5={entry['mrr_at_5']} ndcg@5={entry['ndcg_at_5']}\n"
        )
    sys.stdout.write(f"wrote {arguments.output}\n")


if __name__ == "__main__":
    main()
