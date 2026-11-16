"""One curve candidate policy, so feedback and grading cannot disagree.

v4 shipped with `run_tests` under `legacy` while final verification used
`curve_pure_v1`; 28 of 192 trajectories collected 561 `candidate_policy_rejected`
observations from a rule the grader never applied, and 16 of 17 `issubset` tasks
were refused mid-run while the specification allowed them.  These contracts make
the single-policy property checkable instead of promised.
"""

import hashlib
import json
import re
from pathlib import Path

import pytest
from minidsh_rl.contracts import Turn
from minidsh_rl.curve_cases_v2 import curve_cases_v2
from minidsh_rl.learning_rollout import PolicyBinding, load_rollout, seal_rollout, turn_digest
from minidsh_rl.minirepo_prompts import instruction
from minidsh_rl.minirepo_verification import (
    ARCHIVED_CURVE_POLICIES,
    CURVE_POLICY,
)

PROJECT = Path(__file__).resolve().parents[3]
SOURCE = PROJECT / "experiments/trainable_agent/src/minidsh_rl"


def _policy() -> PolicyBinding:
    return PolicyBinding(
        kind="scripted_control",
        policy_id="policy_consistency",
        snapshot_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
    )


def _curve_rollout(directory: Path, *, candidate_policy: str, sidecar: str | None) -> None:
    """Minimal curve artifact; only the policy identity is under test."""
    case = curve_cases_v2()[0]
    directory.mkdir(parents=True)
    (directory / "tokens").mkdir()
    (directory / "tokens/turn_000.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": directory.name,
                "call_index": 0,
                "policy_id": "policy_consistency",
                "prompt_token_ids": [11, 12],
                "completion_token_ids": [13, 14],
                "old_logprobs": [-0.1, -0.2],
                "seed": 17,
                "temperature": 1.0,
                "top_p": 1.0,
                "raw_text": "<tool_call>done",
                "stop_reason": "stop",
            }
        )
    )
    for phase in ("sample-started", "sample-returned"):
        (directory / phase).mkdir()
        receipt = {
            "schema_version": 1,
            "call_index": 0,
            "seed": 17,
            "prompt_sha256": turn_digest((11, 12)),
        }
        if phase == "sample-returned":
            receipt["turn_sha256"] = turn_digest(Turn((11, 12), (13, 14), (-0.1, -0.2)))
        (directory / phase / "turn_000.json").write_text(json.dumps(receipt))
    candidate = dict(case.task.reference)
    (directory / "candidate.json").write_text(json.dumps(candidate))
    (directory / "verification").mkdir()
    (directory / "verification/result.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_id": case.task.case_id,
                "criterion_version": "minirepo_dev_v1",
                "passed": True,
                "exit_code": 0,
                "timed_out": False,
                "candidate_sha256": turn_digest(candidate),
            }
        )
    )
    if sidecar is not None:
        (directory / "verification/policy.json").write_text(json.dumps({"policy": sidecar}))
    (directory / "result.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": directory.name,
                "case_id": case.task.case_id,
                "family": case.task.family,
                "policy_id": "policy_consistency",
                "candidate_policy": candidate_policy,
                "mode": "interactive",
                "prompt_variant": "tool_example",
                "read_variant": "batch_read_eight",
                "instruction_sha256": hashlib.sha256(
                    instruction(case.task, "interactive", "tool_example").encode()
                ).hexdigest(),
                "seed": 17,
                "split": "development",
                "criterion_version": "minirepo_dev_v1",
                "success": True,
                "status": "finished",
                "sampled_calls": 1,
                "synthetic_stop": False,
                "total_tokens": 4,
                "generated_tokens": 2,
                "tool_calls": 0,
                "steps": 1,
            }
        )
    )
    events = [
        {"name": "run_started", "step_id": 0, "data": {}},
        {"name": "model_call_started", "step_id": 1, "data": {}},
        {
            "name": "model_call_completed",
            "step_id": 1,
            "data": {"input_tokens": 2, "output_tokens": 2, "usage_known": True},
        },
        {
            "name": "run_finished",
            "step_id": 1,
            "data": {
                "status": "finished",
                "duration_ms": 1.0,
                "steps": 1,
                "tokens": 4,
                "tool_calls": 0,
            },
        },
    ]
    (directory / "trace.jsonl").write_text(
        "\n".join(
            json.dumps(
                e
                | {
                    "schema_version": "1.0",
                    "run_id": directory.name,
                    "occurred_at": "2026-01-01T00:00:00Z",
                }
            )
            for e in events
        )
        + "\n"
    )


def test_one_constant_names_the_curve_policy_and_it_is_v2() -> None:
    assert CURVE_POLICY == "curve_pure_v2"
    assert CURVE_POLICY in ARCHIVED_CURVE_POLICIES
    assert frozenset({"curve_pure_v1", "curve_pure_v2"}) == ARCHIVED_CURVE_POLICIES


def test_the_two_call_sites_that_disagreed_now_read_one_name() -> None:
    """Asserted on source: the v4 regression was a literal, not a behaviour.

    Both policies accept the reference solution, so no run would have failed while
    the split existed — only the model's observations showed it.
    """
    rollout = (SOURCE / "learning_rollout.py").read_text()
    feedback = (SOURCE / "learning_feedback.py").read_text()
    for name, text in (("learning_rollout", rollout), ("learning_feedback", feedback)):
        assert re.search(r"candidate_policy\s*=\s*CURVE_POLICY|policy\s*=\s*CURVE_POLICY", text), (
            f"{name} does not take its curve policy from the shared constant"
        )
        assert not re.search(r"^\s*(candidate_)?policy\s*=\s*\"curve_pure_v", text, re.MULTILINE), (
            f"{name} still pins a curve policy literal at a call site"
        )


def test_the_rollout_passes_one_policy_to_session_and_verifier() -> None:
    """`run_minirepo` forwards a single argument, so the two sides cannot split."""
    source = (SOURCE / "minirepo_rollout.py").read_text()
    assert re.search(r"candidate_policy\s*=\s*candidate_policy\b", source)
    assert re.search(r"policy\s*=\s*candidate_policy\b", source)


def test_no_module_outside_the_verifier_chooses_a_curve_policy() -> None:
    """One owner for the choice.

    Archived records may still *name* either policy in a `Literal[...]` schema —
    that is how v4 evidence stays readable — but no call site may pick one.
    """
    choice = re.compile(
        r"(?:candidate_)?policy\s*=\s*[\"']curve_pure_v[12][\"']|policy\s*==\s*[\"']curve_pure_v[12][\"']"
    )
    offenders = [
        path.name
        for path in sorted(SOURCE.glob("*.py"))
        if path.name != "minirepo_verification.py" and choice.search(path.read_text())
    ]
    assert offenders == [], f"policy choices outside the verifier: {offenders}"


async def test_a_verifier_sidecar_naming_another_policy_is_rejected(tmp_path: Path) -> None:
    """This is the v4 failure expressed as an artifact the loader can refuse."""
    directory = tmp_path / "mismatch"
    _curve_rollout(directory, candidate_policy="curve_pure_v2", sidecar="curve_pure_v1")
    case = curve_cases_v2()[0]
    with pytest.raises(ValueError, match="policy disagree"):
        await seal_rollout(case, directory, _policy())


async def test_a_matching_sidecar_is_accepted(tmp_path: Path) -> None:
    directory = tmp_path / "match"
    _curve_rollout(directory, candidate_policy="curve_pure_v2", sidecar="curve_pure_v2")
    case = curve_cases_v2()[0]
    await seal_rollout(case, directory, _policy())
    assert load_rollout(case, directory, _policy()).success is True


async def test_an_archived_run_without_a_sidecar_still_loads(tmp_path: Path) -> None:
    """v4 and GSPO predate the sidecar; refusing them would erase the audit trail."""
    directory = tmp_path / "archived"
    _curve_rollout(directory, candidate_policy="curve_pure_v1", sidecar=None)
    case = curve_cases_v2()[0]
    await seal_rollout(case, directory, _policy())
    assert load_rollout(case, directory, _policy()).success is True
