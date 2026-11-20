"""Search hits are the only thing that may establish a Memory path fact."""

from __future__ import annotations

import json

import pytest

from myagentrun.domain import JsonValue, Message, Role, ToolCall
from myagentrun.domain.memory import MAX_MEMORY_PATHS
from myagentrun.domain.memory_query import MemoryMatchState, memory_query


def _call(name: str, arguments: dict[str, JsonValue], call_id: str = "c1") -> ToolCall:
    return ToolCall(call_id, name, arguments)


def _payload(status: str, result: JsonValue = None) -> dict[str, JsonValue]:
    return {"status": status, "result": result, "error": None}


def _paired(call: ToolCall, payload: dict[str, JsonValue]) -> tuple[Message, ...]:
    return (
        Message(Role.ASSISTANT, "", tool_calls=(call,)),
        Message(Role.TOOL, json.dumps(payload), name=call.name, tool_call_id=call.call_id),
    )


def test_a_search_hit_establishes_the_matched_path() -> None:
    """The search root is a scope, not evidence; the hit file is the evidence."""
    call = _call("search", {"query": "pool", "path": "src"})
    payload = _payload(
        "succeeded",
        {
            "matches": [{"path": "src/db/pool.py", "line": 12, "text": "pool"}],
        },
    )

    paths, error = memory_query(_paired(call, payload))

    assert "src/db/pool.py" in paths
    assert error is None


def test_search_hits_are_ordered_most_recent_last_and_deduplicated() -> None:
    first = _call("search", {"query": "a"}, "c1")
    second = _call("search", {"query": "b"}, "c2")
    history = (
        *_paired(
            first,
            _payload(
                "succeeded",
                {"matches": [{"path": "src/b.py"}, {"path": "src/a.py"}, {"path": "src/a.py"}]},
            ),
        ),
        *_paired(second, _payload("succeeded", {"matches": [{"path": "src/a.py"}]})),
    )

    paths, _ = memory_query(history)

    assert paths.count("src/a.py") == 1
    state = MemoryMatchState()
    for call, payload in ((first, history[1]), (second, history[3])):
        state = state.observe(call, json.loads(payload.content))
    assert state.paths[-1] == "src/a.py"
    assert set(state.paths) == {"src/a.py", "src/b.py"}


@pytest.mark.parametrize("status", ["failed", "timed_out", "denied"])
def test_a_search_that_did_not_succeed_grounds_nothing(status: str) -> None:
    call = _call("search", {"query": "pool"}, "c1")
    payload = _payload(status, {"matches": [{"path": "src/db/pool.py"}]})

    paths, _ = memory_query(_paired(call, payload))

    assert paths == ()
    assert MemoryMatchState().observe(call, payload).paths == ()


@pytest.mark.parametrize(
    "value",
    [
        "/etc/passwd",
        "../outside.py",
        "a\\b.py",
        "src/./a.py",
        "src/x\x00.py",
        "",
        "x" * 513,
        7,
    ],
)
def test_only_canonical_relative_hit_paths_are_accepted(value: JsonValue) -> None:
    call = _call("search", {"query": "q"}, "c1")
    payload = _payload("succeeded", {"matches": [{"path": value}]})

    paths, _ = memory_query(_paired(call, payload))

    assert paths == ()


def test_search_hits_are_capped_without_losing_the_newest() -> None:
    rows: list[JsonValue] = [
        {"path": f"src/module_{index}.py"} for index in range(MAX_MEMORY_PATHS + 5)
    ]
    call = _call("search", {"query": "q"}, "c1")

    paths, _ = memory_query(_paired(call, _payload("succeeded", {"matches": rows})))

    assert len(paths) == MAX_MEMORY_PATHS
    assert f"src/module_{MAX_MEMORY_PATHS + 4}.py" in paths
    assert "src/module_0.py" not in paths


def test_code_search_symbol_search_and_grep_hits_all_count() -> None:
    cases: tuple[tuple[str, dict[str, JsonValue], str], ...] = (
        ("search_code", {"candidates": [{"path": "src/c.py", "matches": 2}]}, "src/c.py"),
        ("find_symbol", {"symbols": [{"path": "src/d.py", "start_line": 1}]}, "src/d.py"),
        ("grep", {"matches": [{"path": "src/e.py", "line": 3, "text": "x"}]}, "src/e.py"),
        ("search", {"matches": [{"path": "src/f.py", "line": 3, "text": "x"}]}, "src/f.py"),
    )
    for name, result, expected in cases:
        call = _call(name, {"query": "q", "path": "."}, f"c_{name}")
        paths, _ = memory_query(_paired(call, _payload("succeeded", result)))
        assert paths == (expected,), name
        assert MemoryMatchState().observe(call, _payload("succeeded", result)).paths == (expected,)


def test_a_call_that_was_never_executed_grounds_nothing() -> None:
    """Merely announcing a search must not make its would-be hits true."""
    call = _call("search", {"query": "pool"}, "c1")
    history = (
        Message(Role.USER, "task", tool_calls=(call,)),
        Message(
            Role.TOOL,
            json.dumps(_payload("succeeded", {"matches": [{"path": "src/db/pool.py"}]})),
            name="search",
            tool_call_id="c1",
        ),
    )

    paths, _ = memory_query(history)

    assert paths == ()


@pytest.mark.parametrize("name", ["read_file", "write_file", "patch_file", "list_files"])
def test_a_file_call_that_was_never_executed_grounds_nothing(name: str) -> None:
    """A declared file action is a request; only its observation is evidence."""
    call = _call(name, {"path": "src/db/a.py"}, "c1")
    announced = (Message(Role.ASSISTANT, "", tool_calls=(call,)),)
    unpaired = (
        Message(Role.ASSISTANT, "", tool_calls=(call,)),
        Message(
            Role.TOOL,
            json.dumps(_payload("succeeded", {"path": "src/db/a.py"})),
            name="read_file",
            tool_call_id="other",
        ),
    )

    assert memory_query(announced)[0] == ()
    assert memory_query(unpaired)[0] == ()


@pytest.mark.parametrize("name", ["read_file", "write_file", "patch_file", "list_files"])
@pytest.mark.parametrize("status", ["failed", "timed_out", "denied", "invalid"])
def test_a_file_call_that_did_not_succeed_grounds_nothing(name: str, status: str) -> None:
    """A failed, timed-out or rejected call never proves the file was visited."""
    call = _call(name, {"path": "src/db/a.py"}, "c1")
    payload = _payload(status, {"path": "src/db/a.py"})
    history = _paired(call, payload)

    paths, _ = memory_query(history)

    assert paths == ()
    state = MemoryMatchState().observe(call, payload)
    assert state.paths == ()


@pytest.mark.parametrize("name", ["read_file", "write_file", "patch_file"])
def test_a_successful_file_call_grounds_its_path(name: str) -> None:
    call = _call(name, {"path": "src/db/a.py"}, "c1")
    payload = _payload("succeeded", {"path": "src/db/a.py"})

    paths, _ = memory_query(_paired(call, payload))
    state = MemoryMatchState().observe(call, payload)

    assert paths == ("src/db/a.py",)
    assert state.paths == ("src/db/a.py",)


def test_replayed_history_and_live_observations_agree_on_grounding() -> None:
    """The v1 replay and the v2 live path must derive the same facts."""
    failed = _call("read_file", {"path": "src/db/a.py"}, "c1")
    succeeded = _call("read_file", {"path": "src/db/b.py"}, "c2")
    payloads = (
        (failed, _payload("failed", None)),
        (succeeded, _payload("succeeded", {"path": "src/db/b.py"})),
    )

    history = tuple(message for call, payload in payloads for message in _paired(call, payload))
    live = MemoryMatchState()
    for call, payload in payloads:
        live = live.observe(call, payload)

    assert memory_query(history)[0] == live.paths == ("src/db/b.py",)


def test_a_directory_listing_scope_is_not_evidence_either() -> None:
    """``list_files`` names a directory to look inside, exactly like a search root.

    The same defect ADR 0042 fixed for ``search`` applied here: one empty listing
    of ``src/db`` was enough to unlock every conditional record scoped to it.
    """
    call = _call("list_files", {"path": "src/db", "glob": "zzz*"}, "c1")
    empty = _payload("succeeded", {"files": [], "truncated": False})
    listed = _payload("succeeded", {"files": ["src/db/pool.py"], "truncated": False})

    assert memory_query(_paired(call, empty))[0] == ()
    assert MemoryMatchState().observe(call, empty).paths == ()
    assert memory_query(_paired(call, listed))[0] == ("src/db/pool.py",)
    assert MemoryMatchState().observe(call, listed).paths == ("src/db/pool.py",)


@pytest.mark.parametrize("name", ["read_file", "write_file", "patch_file"])
def test_a_repeated_call_id_cannot_lend_its_facts_to_an_unexecuted_declaration(
    name: str,
) -> None:
    """A reused id must not let an unexecuted call claim another call's result.

    Providers normally mint unique ids, but the fact layer must not depend on that:
    with a dict keyed by id, the last declaration won and its ``path`` argument
    became evidence while the call that really ran was discarded.
    """
    executed = _call(name, {"path": "src/db/real.py"}, "dup")
    unexecuted = _call(name, {"path": "src/db/ghost.py"}, "dup")
    first = _payload("succeeded", {"path": "src/db/real.py"})
    second = _payload("invalid", None)

    history = (
        Message(Role.ASSISTANT, "", tool_calls=(executed, unexecuted)),
        Message(Role.TOOL, json.dumps(first), name=name, tool_call_id="dup"),
        Message(Role.TOOL, json.dumps(second), name=name, tool_call_id="dup"),
    )

    paths, _ = memory_query(history)

    assert "src/db/ghost.py" not in paths
