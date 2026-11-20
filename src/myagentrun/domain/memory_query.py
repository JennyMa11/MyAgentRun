"""Bounded Memory matching hints from causal tool history, without task-text guessing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from myagentrun.domain import JsonValue, Message, Role, ToolCall
from myagentrun.domain.memory import (
    MAX_MEMORY_ERROR_CHARS,
    MAX_MEMORY_PATH_CHARS,
    MAX_MEMORY_PATHS,
)

_FILES = frozenset({"read_file", "write_file", "patch_file"})
# A scope is not evidence: a search root or a listed directory is what the model
# asked to look inside, so only the paths a successful result actually named become
# hints.  ``list_files`` belongs here, not in ``_FILES``: one empty listing used to
# unlock every conditional record scoped to that directory.
_HIT_KEYS: dict[str, str] = {
    "search": "matches",
    "grep": "matches",
    "search_code": "candidates",
    "find_symbol": "symbols",
    "list_files": "files",
}
_FILES_HITS = frozenset({"list_files"})
_ERROR = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*Error):")
_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}")


def _path(value: JsonValue) -> str | None:
    if not isinstance(value, str) or not value or len(value) > MAX_MEMORY_PATH_CHARS:
        return None
    path = PurePosixPath(value)
    if (
        value == "."
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or "\x00" in value
        or path.as_posix() != value
    ):
        return None
    return value


def _failure(payload: dict[str, JsonValue]) -> str | None:
    error = payload.get("error")
    code = error.get("code") if isinstance(error, dict) else None
    if isinstance(code, str) and _CODE.fullmatch(code) is not None:
        return code
    result = payload.get("result")
    if (
        isinstance(result, dict)
        and type(result.get("exit_code")) is int
        and result["exit_code"] != 0
    ):
        for key in ("stderr", "diagnostics"):
            detail = result.get(key)
            found = _ERROR.search(detail) if isinstance(detail, str) else None
            if found is not None and len(found.group(1)) <= MAX_MEMORY_ERROR_CHARS:
                return found.group(1)
        return "command_failed"
    status = payload.get("status")
    if isinstance(status, str) and status in {"failed", "denied", "timed_out", "invalid"}:
        return status
    return None


def _hits(call: ToolCall, payload: dict[str, JsonValue]) -> tuple[str, ...]:
    """Return every canonical path a completed search or listing named."""
    key = _HIT_KEYS.get(call.name)
    if key is None or payload.get("status") != "succeeded":
        return ()
    result = payload.get("result")
    rows = result.get(key) if isinstance(result, dict) else None
    if not isinstance(rows, list):
        return ()
    found: list[str] = []
    for row in rows:
        # ``list_files`` reports bare paths; the search tools report row objects.
        raw = (
            row if call.name in _FILES_HITS else row.get("path") if isinstance(row, dict) else None
        )
        path = _path(raw)
        if path is not None and path not in found:
            found.append(path)
    return tuple(found)


def _observed_paths(call: ToolCall, payload: dict[str, JsonValue]) -> tuple[str, ...]:
    """A read grounds its file; a search grounds only the files it named.

    ``search``/``grep``/``search_code``/``find_symbol`` take a *scope* argument, so
    their ``path`` is what the model chose to look inside, never something it read.
    A file tool is the same in one respect that matters here: its ``path`` argument
    is a request.  Only a call that actually succeeded proves the file was reached,
    so a failed, timed-out, denied or invalid call grounds nothing.
    """
    if call.name in _HIT_KEYS:
        return _hits(call, payload)
    if payload.get("status") != "succeeded":
        return ()
    path = _path(call.arguments.get("path")) if call.name in _FILES else None
    return (path,) if path is not None else ()


def memory_query(history: tuple[Message, ...]) -> tuple[tuple[str, ...], str | None]:
    """Use the latest 32 canonical accessed paths and the last unresolved tool failure.

    Facts come from paired, executed tool results only.  An assistant message merely
    announces a call, so pairing consumes declarations in order: a reused ``call_id``
    cannot let a later unexecuted declaration borrow an earlier call's result.
    """
    paths: list[str] = []
    pending: list[ToolCall] = []
    error_type: str | None = None
    error_tool: str | None = None
    for message in history:
        if message.role is Role.ASSISTANT:
            pending.extend(message.tool_calls)
            continue
        if message.role is not Role.TOOL:
            continue
        index = next(
            (
                position
                for position, call in enumerate(pending)
                if call.call_id == message.tool_call_id and call.name == message.name
            ),
            None,
        )
        if index is None:
            continue
        issued = pending.pop(index)
        try:
            payload: JsonValue = json.loads(message.content)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        for hit in _observed_paths(issued, payload):
            if hit in paths:
                paths.remove(hit)
            paths.append(hit)
            paths = paths[-MAX_MEMORY_PATHS:]
        failed = _failure(payload)
        if failed is not None:
            error_type, error_tool = failed, issued.name
        elif payload.get("status") == "succeeded" and issued.name == error_tool:
            error_type, error_tool = None, None
    return tuple(sorted(paths)), error_type


@dataclass(frozen=True, slots=True)
class MemoryMatchState:
    """Bounded hints from completed calls, retaining no tool output bodies."""

    paths: tuple[str, ...] = ()
    error_type: str | None = None
    error_tool: str | None = None
    failure_observed: bool = False
    succeeded_tools: frozenset[str] = frozenset()

    def observe(self, call: ToolCall, payload: dict[str, JsonValue]) -> MemoryMatchState:
        if payload.get("status") not in {"succeeded", "failed", "timed_out"}:
            return self
        paths = list(self.paths)
        for path in _observed_paths(call, payload):
            if path in paths:
                paths.remove(path)
            paths.append(path)
        error, tool = self.error_type, self.error_tool
        failure = _failure(payload)
        if failure is not None:
            error, tool = failure, call.name
        elif payload.get("status") == "succeeded" and call.name == tool:
            error, tool = None, None
        successes = self.succeeded_tools
        if failure is None and payload.get("status") == "succeeded":
            successes = successes | {call.name}
        return MemoryMatchState(
            tuple(paths[-MAX_MEMORY_PATHS:]),
            error,
            tool,
            self.failure_observed or failure is not None,
            successes,
        )

    def merge_observed(self, observed: MemoryMatchState) -> MemoryMatchState:
        """Current-run executed facts follow seeded history, including hidden retrieval."""
        paths = list(self.paths)
        for path in observed.paths:
            if path in paths:
                paths.remove(path)
            paths.append(path)
        error, tool = self.error_type, self.error_tool
        if observed.failure_observed:
            error, tool = observed.error_type, observed.error_tool
        elif tool in observed.succeeded_tools:
            error, tool = None, None
        return MemoryMatchState(
            tuple(paths[-MAX_MEMORY_PATHS:]),
            error,
            tool,
            self.failure_observed or observed.failure_observed,
            self.succeeded_tools | observed.succeeded_tools,
        )

    def query(self) -> tuple[tuple[str, ...], str | None]:
        return tuple(sorted(self.paths)), self.error_type
