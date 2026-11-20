"""Bounded regex grep and hit-centred retrieval windows."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from myagentrun.context.retrieval import RepositoryRetriever, window_around
from myagentrun.domain import JsonValue
from myagentrun.infrastructure import WorkspaceSandbox
from myagentrun.tools.builtin import GrepInput, GrepTool
from myagentrun.tools.runtime import ToolError


async def _grep(tmp_path: Path, **arguments: object) -> dict[str, JsonValue]:
    tool = GrepTool(WorkspaceSandbox(tmp_path))
    parsed = GrepInput.model_validate(arguments)
    result = await tool.execute(parsed)
    assert isinstance(result, dict)
    return result


def test_a_window_is_centred_on_the_match_not_the_file_head() -> None:
    lines = [f"line {number}" for number in range(1, 101)]

    content, start, end = window_around(lines, 79, context=3, max_chars=10_000)

    assert "line 80" in content.splitlines()
    assert start == 77
    assert end == 83


def test_a_window_shrinks_towards_the_match_when_it_exceeds_the_budget() -> None:
    lines = [f"line {number} " + "x" * 40 for number in range(1, 201)]

    content, start, end = window_around(lines, 100, context=40, max_chars=300)

    lines_returned = content.splitlines()
    assert sum(len(line) + 1 for line in lines_returned) <= 301
    assert any(line.startswith("line 101 ") for line in lines_returned)
    assert start > 61  # the head of the naive window was dropped
    assert end < 141


def test_a_window_at_the_end_of_the_file_still_contains_the_match() -> None:
    lines = ["a", "b", "c"]
    content, start, end = window_around(lines, 2, context=5, max_chars=10_000)
    assert content == "a\nb\nc"
    assert (start, end) == (1, 3)


def test_retrieval_returns_the_hit_region_not_the_first_chars(tmp_path: Path) -> None:
    """The regression this change exists for: a match at line 500 was invisible."""
    body = [f"# filler line {number}" for number in range(500)]
    body.append("def target_function():")
    body.append("    return 'needle_in_haystack'")
    (tmp_path / "big.py").write_text("\n".join(body), encoding="utf-8")

    chunks = RepositoryRetriever(tmp_path, max_chunk_chars=600).search(
        "needle_in_haystack", limit=1
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert "needle_in_haystack" in chunk.content
    assert "filler line 0" not in chunk.content
    assert chunk.match_line == 502
    assert chunk.start_line <= chunk.match_line <= chunk.end_line


@pytest.mark.asyncio
async def test_grep_returns_matching_lines_with_context(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text(
        "import os\n\n\ndef load(value):\n    return value + 1\n", encoding="utf-8"
    )

    result = await _grep(tmp_path, pattern=r"return value", context_lines=1)

    matches = cast("list[dict[str, JsonValue]]", result["matches"])
    assert len(matches) == 1
    match = matches[0]
    assert match["path"] == "module.py"
    assert match["line"] == 5
    assert match["after"] == []
    assert match["before"] == ["def load(value):"]


@pytest.mark.asyncio
async def test_grep_supports_regular_expressions_and_case_folding(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("TimeoutError\nValueError\n", encoding="utf-8")

    pattern = r"^[A-Z]\w+Error$"
    sensitive = await _grep(tmp_path, pattern=pattern)
    result = await _grep(tmp_path, pattern=pattern.lower(), case_sensitive=False)

    assert len(cast("list[JsonValue]", sensitive["matches"])) == 2
    assert len(cast("list[JsonValue]", result["matches"])) == 2


@pytest.mark.asyncio
async def test_an_invalid_pattern_is_a_structured_error_not_a_crash(tmp_path: Path) -> None:
    with pytest.raises(ToolError) as raised:
        await _grep(tmp_path, pattern=r"(unclosed")
    assert raised.value.code == "invalid_pattern"


@pytest.mark.asyncio
async def test_grep_is_bounded_by_max_matches(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("\n".join("hit" for _ in range(50)), encoding="utf-8")

    result = await _grep(tmp_path, pattern="hit", max_matches=5)

    assert len(cast("list[JsonValue]", result["matches"])) == 5
    assert result["truncated"] is True


@pytest.mark.asyncio
async def test_grep_truncates_long_lines_and_skips_noise_directories(tmp_path: Path) -> None:
    (tmp_path / "long.txt").write_text("x" * 900 + " needle\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("needle\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "index.js").write_text("needle\n", encoding="utf-8")

    result = await _grep(tmp_path, pattern="needle", context_lines=0)

    matches = cast("list[dict[str, JsonValue]]", result["matches"])
    assert len(matches) == 1
    assert matches[0]["path"] == "long.txt"
    assert len(str(matches[0]["text"])) <= 500


@pytest.mark.asyncio
async def test_grep_cannot_escape_the_workspace_or_read_secrets(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("needle=super-secret\n", encoding="utf-8")
    (tmp_path.parent / "outside.txt").write_text("needle\n", encoding="utf-8")

    leaked = await _grep(tmp_path, pattern="needle")
    assert leaked["matches"] == []

    with pytest.raises(ToolError) as raised:
        await _grep(tmp_path, pattern="needle", path="../outside.txt")
    assert raised.value.code == "sandbox_violation"


@pytest.mark.asyncio
async def test_grep_only_requires_read_permission(tmp_path: Path) -> None:
    definition = GrepTool(WorkspaceSandbox(tmp_path)).definition
    assert definition.permissions == ("fs.read",)
    assert definition.side_effects is False
    assert definition.timeout_seconds > 0
