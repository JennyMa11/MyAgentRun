"""Deterministic model-context views for reproducible experiments."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol, cast

from myagentrun.context.retrieval import RetrievedChunk
from myagentrun.context.snapshots import SnapshotRetriever
from myagentrun.domain import ContextView, JsonValue, Message, Role, estimate_tokens
from myagentrun.domain.harness import PreparedContext
from myagentrun.ports.protocols import ContextManager
from myagentrun.tools.results import diagnostics

_MIN_SUMMARY_CHARS = 100
_PINNED_PREFIX_LENGTH = 2


def _pinned_indices(history: tuple[Message, ...]) -> set[int]:
    pinned = set(range(min(2, len(history))))
    latest: dict[str, int] = {}
    for index, message in enumerate(history):
        if message.role is Role.SYSTEM and message.name is None:
            pinned.add(index)
        if message.role is Role.USER:
            latest["current_task"] = index
        if message.role is Role.SYSTEM and message.name == "critical_decision":
            pinned.add(index)
        if message.role is Role.SYSTEM and message.name in {
            "plan_state",
            "modified_files",
            "open_error",
            "workspace_instructions",
            "repository_map",
            "verification_status",
        }:
            latest[message.name] = index
    pinned.update(latest.values())
    return pinned


def _window_start(history: tuple[Message, ...], start: int, floor: int) -> int:
    """Move ``start`` back so a window never begins inside a tool-call round.

    An OpenAI-compatible provider rejects a request whose ``tool`` message has no
    preceding assistant message declaring the matching ``tool_call_id``, so a
    strategy that keeps only the last N messages must not cut a round in half.
    The window therefore grows backwards to the assistant message that opened the
    round — never past ``floor``, which is the pinned prefix.

    This is invisible under a scripted model, which does not validate the request
    it is handed, and surfaces against a real provider as an opaque HTTP 400 that
    looks exactly like a model failure.
    """
    index = start
    while index > floor and history[index].role is Role.TOOL:
        index -= 1
    return index


class FullHistoryContext:
    """Retain every authoritative message."""

    name: str = "full_history"
    version: str = "1.0"

    def build(self, history: tuple[Message, ...], task: str) -> ContextView:
        del task
        return ContextView(history, self.name, self.version, estimate_tokens(history))


@dataclass(frozen=True, slots=True)
class BudgetContext:
    """Keep pinned facts and newest complete turns inside a measured input budget."""

    input_budget: int = 8_192
    count_tokens: Callable[[tuple[Message, ...]], int] = estimate_tokens
    name: str = "budget"
    version: str = "1.1"

    def __post_init__(self) -> None:
        if self.input_budget < 1:
            raise ValueError("input_budget must be positive")

    def with_counter(self, counter: Callable[[tuple[Message, ...]], int]) -> ContextManager:
        return replace(self, count_tokens=counter)

    def build(  # noqa: PLR0912 - selection rules keep pinned state and tool rounds intact
        self, history: tuple[Message, ...], task: str
    ) -> ContextView:
        del task
        if len(history) < len({0, 1}):
            raise ValueError("budget context requires system and task messages")
        first_task = next(
            (index for index, message in enumerate(history) if message.role is Role.USER), 1
        )
        pinned_indices = {0, first_task}
        latest: dict[str, int] = {}
        for index, message in enumerate(history[1:], 1):
            if message.role is Role.SYSTEM and message.name is None:
                pinned_indices.add(index)
            if message.role is Role.USER:
                latest["current_task"] = index
            if message.role is Role.SYSTEM and message.name == "critical_decision":
                pinned_indices.add(index)
            elif (
                message.role is Role.SYSTEM
                and message.name
                in {
                    "plan_state",
                    "modified_files",
                    "open_error",
                    "workspace_instructions",
                    "repository_map",
                    "verification_status",
                }
            ) or (message.name == "model_feedback" and index == len(history) - 1):
                latest[message.name] = index
        pinned_indices.update(latest.values())
        # A tool reply and its declaring assistant message are one indivisible unit.
        groups: list[tuple[int, ...]] = []
        index = 1
        while index < len(history):
            name = history[index].name if history[index].role is Role.SYSTEM else None
            if name == "model_feedback" and index != len(history) - 1:
                index += 1
                continue
            if name is not None and name in latest and index != latest[name]:
                index += 1
                continue
            end = index + 1
            if history[index].tool_calls:
                call_ids = {call.call_id for call in history[index].tool_calls}
                while end < len(history) and history[end].role is Role.TOOL:
                    call_ids.discard(history[end].tool_call_id)
                    end += 1
                if call_ids:
                    raise ValueError("history contains an incomplete tool round")
            groups.append(tuple(range(index, end)))
            index = end
        selected = set(pinned_indices)
        pinned = tuple(history[index] for index in sorted(selected))
        if self.count_tokens(pinned) > self.input_budget:
            raise ValueError("pinned context exceeds input budget")
        for group in reversed(groups):
            candidate = selected | set(group)
            messages = tuple(history[index] for index in sorted(candidate))
            if self.count_tokens(messages) <= self.input_budget:
                selected = candidate
        messages = tuple(history[index] for index in sorted(selected))
        omitted = tuple(index for index in range(len(history)) if index not in selected)
        return ContextView(
            messages,
            self.name,
            self.version,
            self.count_tokens(messages),
            omitted_messages=len(omitted),
            metadata={
                "input_budget": self.input_budget,
                "omitted_indices": list(omitted),
                "omitted_estimated_tokens": estimate_tokens(
                    tuple(history[index] for index in omitted)
                ),
                "omission_reason": "budget_or_superseded_state" if omitted else "none",
            },
        )


@dataclass(frozen=True, slots=True)
class SlidingWindowContext:
    """Pin system/task messages and keep the most recent remaining messages."""

    window_size: int = 20
    name: str = "sliding_window"
    version: str = "1.0"

    def __post_init__(self) -> None:
        if self.window_size < 1:
            msg = "window_size must be positive"
            raise ValueError(msg)

    def build(self, history: tuple[Message, ...], task: str) -> ContextView:
        del task
        if len(history) <= self.window_size + 2:
            selected = history
            start = 0
            omitted_messages: tuple[Message, ...] = ()
        else:
            pinned = history[:2]
            start = _window_start(history, len(history) - self.window_size, floor=len(pinned))
            indices = _pinned_indices(history) | set(range(start, len(history)))
            selected = tuple(history[index] for index in sorted(indices))
            omitted_messages = tuple(
                message for index, message in enumerate(history) if index not in indices
            )
        return ContextView(
            selected,
            self.name,
            self.version,
            estimate_tokens(selected),
            omitted_messages=len(omitted_messages),
            metadata={
                "window_size": self.window_size,
                # Recorded so the adjustment is observable rather than a silent
                # widening of the window.  Zero for the common case.
                "window_extended_for_tool_round": len(history) - start - self.window_size,
                "omitted_estimated_tokens": estimate_tokens(omitted_messages),
            },
        )


_TOOL_RESULT_PRUNE_THRESHOLD = 1_200
_DIAGNOSTIC_TAIL_CHARS = 2_000


def _tool_result_preview(message: Message) -> tuple[Message, bool]:
    """Shorten a tool observation without changing its protocol identity."""
    if message.role is not Role.TOOL or len(message.content) <= _TOOL_RESULT_PRUNE_THRESHOLD:
        return message, False
    try:
        body = json.loads(message.content)
    except ValueError:
        return message, False
    if not isinstance(body, dict):
        return message, False
    result = body.get("result")
    result = result if isinstance(result, dict) else {}
    status = body.get("status")
    failed = status != "succeeded" or result.get("exit_code") not in (0, None)
    preview = {
        "tool_call_id": message.tool_call_id,
        "status": status,
        "context_pruned": True,
        **{
            k: result[k]
            for k in ("exit_code", "timed_out", "start_line", "end_line")
            if k in result
        },
    }
    if isinstance(result.get("path"), str):
        preview["path"] = result["path"][:240]
    if isinstance(result.get("argv"), list):
        preview["command"] = " ".join(map(str, result["argv"]))[:240]
    if isinstance(body.get("summary"), str):
        preview["summary"] = body["summary"][:240]
    if isinstance(body.get("artifacts"), list):
        preview["artifacts"] = [ref for ref in body["artifacts"][:2] if isinstance(ref, str)]
    if failed:
        error = body.get("error")
        if isinstance(error, dict):
            for key in ("code", "message"):
                if isinstance(error.get(key), str):
                    preview[key] = error[key][:240]
        tails = {
            key: result[key][-_DIAGNOSTIC_TAIL_CHARS:]
            for key in ("stdout", "stderr")
            if isinstance(result.get(key), str)
        }
        prior = result.get("diagnostics")
        existing = (
            [line[:240] for line in prior[-2:] if isinstance(line, str)]
            if isinstance(prior, list)
            else []
        )
        newest_first: dict[str, JsonValue] = {
            key: "\n".join(reversed(text.splitlines())) for key, text in tails.items()
        }
        evidence = diagnostics(newest_first, limit=2) or existing
        if not evidence:
            evidence = [text.strip()[-240:] for text in tails.values() if text.strip()]
        preview["diagnostics"] = evidence
    content = json.dumps(preview, ensure_ascii=False, separators=(",", ":"))
    return (
        (replace(message, content=content), failed)
        if len(content) < len(message.content)
        else (message, False)
    )


def _prune_answered_tool_results(
    history: tuple[Message, ...], pinned: set[int], last_assistant: int
) -> tuple[tuple[Message, ...], dict[int, bool]]:
    pruned: dict[int, bool] = {}
    messages = list(history)
    for index in range(last_assistant):
        if index in pinned:
            continue
        original = history[index]
        preview, failed = _tool_result_preview(original)
        if preview is not original:
            messages[index] = preview
            pruned[index] = failed
    return tuple(messages), pruned


@dataclass(frozen=True, slots=True)
class SummaryCompactionContext:
    """Compact older messages into a bounded, deterministic summary."""

    recent_messages: int = 12
    summary_max_chars: int = 4_000
    name: str = "summary_compaction"
    version: str = "1.0"
    prune_tool_results: bool = False
    model_summary: bool = False

    def __post_init__(self) -> None:
        if self.recent_messages < 1 or self.summary_max_chars < _MIN_SUMMARY_CHARS:
            msg = "recent_messages must be positive and summary_max_chars must be at least 100"
            raise ValueError(msg)

    def build(self, history: tuple[Message, ...], task: str) -> ContextView:
        del task
        pinned_indices = _pinned_indices(history)
        pruned: dict[int, bool] = {}
        last_assistant = -1
        if self.prune_tool_results:
            last_assistant = next(
                (i for i in reversed(range(len(history))) if history[i].role is Role.ASSISTANT),
                -1,
            )
            history, pruned = _prune_answered_tool_results(history, pinned_indices, last_assistant)
        metadata: dict[str, JsonValue] = {}
        if self.prune_tool_results:
            metadata["pruned_tool_result_indices"] = cast("list[JsonValue]", list(pruned))
        if len(history) <= self.recent_messages + 2:
            return ContextView(
                history, self.name, self.version, estimate_tokens(history), metadata=metadata
            )

        start = _window_start(history, len(history) - self.recent_messages, floor=2)
        if (
            self.prune_tool_results
            and last_assistant >= _PINNED_PREFIX_LENGTH
            and any(message.role is Role.TOOL for message in history[last_assistant + 1 :])
        ):
            start = min(start, last_assistant)
        omitted_indices = [index for index in range(start) if index not in pinned_indices]
        if self.model_summary:
            metadata["model_summary_source_indices"] = cast(
                "list[JsonValue]", omitted_indices.copy()
            )
            reasoning_chars = sum(len(message.reasoning_content or "") for message in history)
            metadata["model_summary_pressure_estimate"] = (
                estimate_tokens(history) + (reasoning_chars + 3) // 4
            )
        # Recent known failures get space before routine old output fills the summary.
        if pruned:
            omitted_indices.sort(key=lambda index: (0, -index) if pruned.get(index) else (1, index))
        omitted = tuple(history[index] for index in omitted_indices)
        fragments: list[str] = []
        remaining = self.summary_max_chars
        for message in omitted:
            normalized = " ".join(message.content.split())
            fragment = f"[{message.role.value}] {normalized}\n"
            if len(fragment) > remaining:
                fragment = fragment[:remaining]
            fragments.append(fragment)
            remaining -= len(fragment)
            if remaining <= 0:
                break
        summary = Message(
            Role.SYSTEM,
            "Compacted earlier trace (derived, not authoritative):\n" + "".join(fragments),
            name="context_summary",
        )
        # Nothing safely compactable means no summary message at all, rather than
        # an empty one the model has to spend a turn reading.
        prefix = tuple(history[index] for index in sorted(pinned_indices) if index < start)
        head = (*prefix, summary) if omitted else prefix
        selected = (*head, *history[start:])
        return ContextView(
            selected,
            self.name,
            self.version,
            estimate_tokens(selected),
            omitted_messages=len(omitted),
            metadata={
                **metadata,
                "recent_messages": self.recent_messages,
                "summary_source_messages": len(omitted),
                "window_extended_for_tool_round": len(history) - start - self.recent_messages,
                "omitted_estimated_tokens": estimate_tokens(omitted),
                "summary_estimated_tokens": estimate_tokens((summary,)) if omitted else 0,
            },
        )


class Retriever(Protocol):
    """Retrieve task-relevant repository chunks."""

    def search(self, query: str, *, limit: int) -> tuple[RetrievedChunk, ...]:
        """Return ranked chunks."""


@dataclass(frozen=True, slots=True)
class RetrievalContext:
    """Augment any base strategy with repository or symbol retrieval."""

    base: FullHistoryContext | SlidingWindowContext | SummaryCompactionContext
    retriever: Retriever
    limit: int = 5
    name: str = "repo_retrieval"
    version: str = "1.0"

    async def prepare(  # noqa: PLR0913, PLR0917 - explicit preparation and accounting inputs
        self,
        history: tuple[Message, ...],
        task: str,
        run_id: str,
        step_id: int,
        max_calls: int,
        account: Callable[[int, float], None] | None = None,
    ) -> PreparedContext:
        if isinstance(self.retriever, SnapshotRetriever):
            calls, duration = await self.retriever.prepare(
                task, run_id, step_id, max_calls, account
            )
            return PreparedContext(self.build(history, task), calls, duration)
        return PreparedContext(self.build(history, task))

    def build(self, history: tuple[Message, ...], task: str) -> ContextView:
        base_view = self.base.build(history, task)
        chunks = self.retriever.search(task, limit=self.limit)
        if not chunks:
            return ContextView(
                base_view.messages,
                f"{self.name}+{base_view.strategy}",
                self.version,
                base_view.estimated_tokens,
                base_view.omitted_messages,
                {**base_view.metadata, "retrieved_chunks": 0},
            )
        rendered = "\n\n".join(
            f"--- {chunk.path} (score={chunk.score:.2f}) ---\n{chunk.content}" for chunk in chunks
        )
        retrieval_message = Message(
            Role.SYSTEM,
            "Task-relevant repository context (read-only derived view):\n" + rendered,
            name="repo_retrieval",
        )
        messages = (*base_view.messages, retrieval_message)
        return ContextView(
            messages,
            f"{self.name}+{base_view.strategy}",
            self.version,
            estimate_tokens(messages),
            base_view.omitted_messages,
            {
                **base_view.metadata,
                "retrieved_chunks": len(chunks),
                "retrieved_paths": [chunk.path for chunk in chunks],
            },
        )
