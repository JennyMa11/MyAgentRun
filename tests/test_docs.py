# ruff: noqa: RUF001
"""Keep the project overview and current technical documentation in sync."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

import myagentrun
from myagentrun.benchmark.ablation import CONTEXT_VARIANTS
from myagentrun.composition import RuntimeOptions, _build_context

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"
DOCS = REPO_ROOT / "docs"

# The long-form documentation is kept with the project rather than published
# here: it is written for the author's own working notes, changes on almost every
# commit, and nothing in `src/` reads it.  The repository still ships the README,
# so the gates that only need the README always run; the ones below that read a
# document out of `docs/` are skipped where that tree is absent instead of
# failing on a file this checkout is not expected to have.
_WHEN_DOCS_PRESENT = pytest.mark.skipif(
    not DOCS.is_dir(), reason="the long-form documentation is not part of this repository"
)

# Benchmark manifests checked by this module.
COUNTED_SUITES = ("core", "repo", "hard")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _declared_version_in_readme() -> str:
    match = re.search(r"^-\s*版本：`([^`]+)`", _read(README), flags=re.MULTILINE)
    assert match is not None, "README no longer declares a version on its bullet line"
    return match.group(1)


def _changelog_versions() -> list[str]:
    return re.findall(r"^##\s*\[([^\]]+)\]", _read(CHANGELOG), flags=re.MULTILINE)


def _manifest_task_count(suite: str) -> int:
    manifest = json.loads(_read(REPO_ROOT / "benchmarks" / suite / "manifest.json"))
    return len(manifest["tasks"])


def test_the_version_is_declared_in_four_places_and_they_agree() -> None:
    """One release, one number.

    ``pyproject.toml`` drives the wheel, ``__init__`` drives ``--version``, and
    README/CHANGELOG tell the reader what they are looking at.  If any one of
    them moves alone the published artifact and the documentation disagree, so
    the gate checks all four rather than trusting the bump script.
    """
    in_pyproject = tomllib.loads(_read(PYPROJECT))["project"]["version"]

    assert in_pyproject == myagentrun.__version__, (
        f"pyproject says {in_pyproject}, package says {myagentrun.__version__}"
    )
    assert _declared_version_in_readme() == in_pyproject, (
        f"README says {_declared_version_in_readme()}, project says {in_pyproject}"
    )

    versions = _changelog_versions()
    assert versions, "CHANGELOG has no version headings"
    assert versions[0] == in_pyproject, (
        f"newest CHANGELOG entry is {versions[0]}, project says {in_pyproject}; "
        "a release without a changelog entry is a release nobody can audit"
    )


def test_readme_keeps_archived_result_tables_off_the_overview() -> None:
    overview = _read(README)
    for name in ("real-model-results", "real-model-ablation-results", "hard-suite-results"):
        assert f"{name}:start" not in overview


@pytest.mark.parametrize("suite", COUNTED_SUITES)
def test_counted_suites_exist_and_are_non_empty(suite: str) -> None:
    assert _manifest_task_count(suite) > 0, f"{suite} suite is empty"


def _the_whole_suite_was_collected(request: pytest.FixtureRequest) -> bool:
    """True only when every test module is in this session.

    The collected-item count is exact and free, but it is only *the* number when
    the full suite ran.  Running a single file must not report a false alarm, so
    the check is skipped rather than failed when the session is filtered.
    """
    collected = {item.path.resolve() for item in request.session.items}
    modules = sorted((REPO_ROOT / "tests").glob("test_*.py"))
    return bool(modules) and all(module.resolve() in collected for module in modules)


# A test-count claim, in the two forms the documentation actually uses.
# "36 个攻击测试" is deliberately *not* matched: it counts something else.
_TEST_COUNT_CLAIM = re.compile(r"(\d+)\s*个测试(?![\u4e00-\u9fff])|(\d+)\s*tests\b")

# The type-check scope, always written right after the command it describes.
_MYPY_FILE_COUNT_CLAIM = re.compile(r"mypy --strict`[^。\n]*?(\d+)\s*个文件")

# The overall coverage figure, always written as "覆盖率 [**]NN%".  Runtime-only
# figures read "覆盖率为/达 100%" and are deliberately not matched: they are a
# different claim about a different scope.
_COVERAGE_CLAIM = re.compile(r"覆盖率[\s*]*(\d+)\s*%")

# CHANGELOG and results/ record what was true for a particular release or run.
# Rewriting measured history to match today's tree would destroy the audit trail.
_COUNT_GATED_DOCS = (
    README,
    *(path for path in DOCS.rglob("*.md") if not path.is_relative_to(DOCS / "results")),
)


@_WHEN_DOCS_PRESENT
def test_historical_results_are_not_gated_against_current_tree() -> None:
    historical = DOCS / "results/offline-regression-2026-09-27.md"
    assert historical.is_file()
    assert historical not in _COUNT_GATED_DOCS
    assert DOCS / "resume.md" in _COUNT_GATED_DOCS


def test_every_document_quotes_the_same_test_count(
    request: pytest.FixtureRequest,
) -> None:
    """One number, everywhere.

    A stale count in a *current* document is the same defect as a stale count on
    the front page — it just hides one page deeper.  Historical records are the
    one legitimate exception, which is why the changelog is excluded.
    """
    if not _the_whole_suite_was_collected(request):
        pytest.skip("only a subset of the suite was collected")

    expected = len(request.session.items)
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{number}"
        for path in _COUNT_GATED_DOCS
        for number, line in enumerate(_read(path).splitlines(), start=1)
        for _match in _TEST_COUNT_CLAIM.finditer(line)
        if int(_match.group(1) or _match.group(2)) != expected
    ]

    assert not offenders, f"these documents quote a test count other than {expected}: " + ", ".join(
        offenders
    )


def test_mypy_file_count_matches_the_type_checked_tree() -> None:
    """The type-check scope is a directory listing, so it is listed.

    Quoting a stale scope is worse than quoting no scope: it tells the reader a
    file is type-checked when it is not.
    """
    checked = len(list((REPO_ROOT / "src").rglob("*.py"))) + len(
        list((REPO_ROOT / "tests").rglob("*.py"))
    )
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{number}"
        for path in _COUNT_GATED_DOCS
        for number, line in enumerate(_read(path).splitlines(), start=1)
        if (match := _MYPY_FILE_COUNT_CLAIM.search(line)) and int(match.group(1)) != checked
    ]

    assert not offenders, (
        f"these documents quote a mypy scope other than {checked} files: " + ", ".join(offenders)
    )


@_WHEN_DOCS_PRESENT
def test_coverage_is_quoted_consistently_across_documents() -> None:
    """Consistency, not truth.

    The suite cannot measure its own coverage, so this checks that every
    document quotes the *same* percentage.  That catches the realistic failure —
    one file updated and the others left behind — without pretending to verify a
    number only the coverage command can produce.
    """
    claims = [
        (path, number, match.group(1))
        for path in _COUNT_GATED_DOCS
        for number, line in enumerate(_read(path).splitlines(), start=1)
        for match in _COVERAGE_CLAIM.finditer(line)
    ]
    assert claims, "no document states a coverage figure"

    quoted = {value for _, _, value in claims}
    assert len(quoted) == 1, "documents disagree on coverage: " + ", ".join(
        f"{path.relative_to(REPO_ROOT)}:{number}={value}" for path, number, value in claims
    )


_VARIANT_NAME = re.compile(r"`([^`]+)`")


# The Context strategy table in `docs/context-engineering.md` is read as a
# specification: it is how a reader decides what a strategy keeps.  It is also the
# place where "injects code" and "keeps history" get conflated, so it is checked
# against the factory rather than trusted.
_CONTEXT_DOC = DOCS / "context-engineering.md"


def _documented_context_classes() -> dict[str, str]:
    """Map each documented strategy name to the class cell of its table row.

    Only tables carrying a `类` column contribute, so the selection and results
    tables further down the same document cannot be mistaken for the strategy table.
    """
    rows: dict[str, str] = {}
    header: list[str] | None = None
    for line in _read(_CONTEXT_DOC).splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            header = None
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if "类" not in header or len(cells) != len(header):
            continue
        if all(set(cell) <= {"-"} for cell in cells):
            continue
        match = _VARIANT_NAME.match(cells[0])
        if match is None:
            continue
        rows[match.group(1)] = cells[header.index("类")]
    return rows


@_WHEN_DOCS_PRESENT
def test_the_context_strategy_table_names_the_base_class_the_factory_builds(
    tmp_path: Path,
) -> None:
    """The table's `类` column must match what `_build_context` actually returns.

    `symbol_retrieval` was documented as `RetrievalContext(FullHistoryContext, ...)`
    while the factory gave it a `summary_compaction` base.  That is not a cosmetic
    error: it is the difference between a prefix-stable strategy and a truncating
    one, which is what decides whether it trips the tool-call-round truncation bug
    and how well it hits the prompt cache.  The mapping is taken from
    `CONTEXT_VARIANTS`, so renaming a strategy fails here instead of silently
    dropping its row from the check.
    """
    documented = _documented_context_classes()
    assert documented, "the context-engineering strategy table could not be found"

    drift: list[str] = []
    for variant in CONTEXT_VARIANTS:
        cell = documented.get(variant.name)
        if cell is None:
            drift.append(f"{variant.name}: documented nowhere in the strategy table")
            continue
        built = _build_context(variant.context_strategy, tmp_path, RuntimeOptions())
        # A retrieval strategy wraps a base, and the base is the part that decides
        # whether history survives — so that is the name the row must carry.
        base = getattr(built, "base", built)
        drift.extend(
            f"{variant.name}: the row omits {name} (row says {cell!r})"
            for name in sorted({type(built).__name__, type(base).__name__})
            if name not in cell
        )

    assert not drift, (
        "docs/context-engineering.md disagrees with composition._build_context: " + "; ".join(drift)
    )
