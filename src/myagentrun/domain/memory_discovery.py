"""Deterministic, body-free project knowledge discovery; no I/O or authority promotion."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Literal

from myagentrun.domain.memory import (
    MAX_RETRIEVED_ENTRIES,
    MemoryEntry,
    MemoryKind,
    MemoryOmission,
    MemoryStatus,
    MemoryValidationError,
    select_memory,
)
from myagentrun.domain.memory_ranking import (
    DEFAULT_SELECTOR,
    FieldedText,
    RankedDocument,
    SelectorName,
    rank,
)
from myagentrun.domain.memory_ranking import (
    lexical_terms as lexical_terms,  # noqa: PLC0414 - frozen public re-export
)
from myagentrun.domain.models import JsonValue
from myagentrun.domain.redaction import redact_text

SELECTOR_VERSION = DEFAULT_SELECTOR
MAX_QUERY_CHARS = 256
MAX_QUERY_BYTES = 1024
_ID = re.compile(r"[a-z][a-z0-9_]{0,127}")
_HASH = re.compile(r"[a-f0-9]{64}")


@dataclass(frozen=True, slots=True)
class MemoryReference:
    entry_id: str
    expected_revision: int
    expected_content_sha256: str

    def __post_init__(self) -> None:
        if (
            _ID.fullmatch(self.entry_id) is None
            or type(self.expected_revision) is not int
            or self.expected_revision < 1
            or _HASH.fullmatch(self.expected_content_sha256) is None
        ):
            raise MemoryValidationError("memory_reference_invalid")


@dataclass(frozen=True, slots=True)
class MemoryCard:
    entry_id: str
    topic: str
    kind: MemoryKind
    paths: tuple[str, ...]
    error_type: str | None
    revision: int
    content_sha256: str
    evidence_sha256: str | None
    approval_sha256: str | None
    applicability: Literal["conditional", "unconditional"]
    score: int
    matched_fields: tuple[str, ...]

    def as_json(self) -> dict[str, JsonValue]:
        return {
            "entry_id": self.entry_id,
            "topic": self.topic,
            "kind": self.kind.value,
            "paths": list(self.paths),
            "error_type": self.error_type,
            "revision": self.revision,
            "content_sha256": self.content_sha256,
            "evidence_sha256": self.evidence_sha256,
            "approval_sha256": self.approval_sha256,
            "applicability": self.applicability,
            "score": self.score,
            "matched_fields": list(self.matched_fields),
        }


@dataclass(frozen=True, slots=True)
class MemorySearchResult:
    cards: tuple[MemoryCard, ...]
    omissions: tuple[MemoryOmission, ...]
    snapshot_sha256: str
    query_sha256: str
    selector_version: str = SELECTOR_VERSION
    term_overlap: bool = True
    """False when no eligible record shares a single term with a non-empty query.

    This is the documented limit of lexical retrieval: a paraphrase with no common
    token is a miss by construction, and callers must be able to tell that apart
    from "the corpus is empty" or "everything was filtered out".
    """

    @property
    def truncated(self) -> bool:
        return any(item.reason in {"selection_limit", "token_limit"} for item in self.omissions)


def snapshot_sha256(entries: tuple[MemoryEntry, ...]) -> str:
    material = sorted(
        (asdict(entry) for entry in entries),
        key=lambda entry: (str(entry["project_id"]), str(entry["entry_id"])),
    )
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _query_terms(query: str | None) -> frozenset[str]:
    if query is None:
        return frozenset()
    try:
        valid = bool(query.strip()) and len(query) <= MAX_QUERY_CHARS
        valid = valid and len(query.encode("utf-8")) <= MAX_QUERY_BYTES
    except UnicodeError:
        valid = False
    if not valid:
        raise MemoryValidationError("memory_query_invalid")
    if redact_text(query) != query:
        raise MemoryValidationError("memory_sensitive_content")
    return lexical_terms(query)


def fielded(entry: MemoryEntry) -> FieldedText:
    """Render one record's four searchable fields; the single place that decides them."""
    return FieldedText(entry.topic, " ".join(entry.paths), entry.error_type or "", entry.content)


def _card(entry: MemoryEntry, ranked: RankedDocument) -> MemoryCard:
    return MemoryCard(
        entry.entry_id,
        entry.topic,
        entry.kind,
        entry.paths,
        entry.error_type,
        entry.revision,
        entry.content_sha256,
        entry.evidence_sha256,
        entry.approval_sha256,
        "conditional" if entry.paths or entry.error_type is not None else "unconditional",
        ranked.score,
        ranked.matched_fields,
    )


def discover_memory(  # noqa: PLR0913 - explicit, bounded query contract
    entries: tuple[MemoryEntry, ...],
    *,
    project_id: str,
    applicable_version: str,
    query: str | None = None,
    paths: tuple[str, ...] = (),
    error_type: str | None = None,
    limit: int = MAX_RETRIEVED_ENTRIES,
    selector: SelectorName = SELECTOR_VERSION,
) -> MemorySearchResult:
    """Discover topics even before their path/error applicability is established."""
    select_memory((), project_id=project_id, applicable_version=applicable_version, paths=paths)
    if type(limit) is not int or not 1 <= limit <= MAX_RETRIEVED_ENTRIES:
        raise MemoryValidationError("memory_limit_invalid")
    terms = _query_terms(query)
    # Never expose another project's IDs, even in diagnostics or snapshot identity.
    scoped = tuple(entry for entry in entries if entry.project_id == project_id)
    candidates = tuple(
        entry
        for entry in sorted(scoped, key=lambda item: item.entry_id)
        if entry.status is MemoryStatus.COMMITTED and entry.applicable_version == applicable_version
    )
    contents: dict[str, set[str]] = {}
    for entry in candidates:
        contents.setdefault(entry.topic, set()).add(entry.content_sha256)
    omissions: list[MemoryOmission] = []
    eligible: list[MemoryEntry] = []
    seen: set[str] = set()
    for entry in candidates:
        if len(contents[entry.topic]) > 1:
            omissions.append(MemoryOmission(entry.entry_id, "topic_conflict"))
            continue
        if entry.topic in seen:
            omissions.append(MemoryOmission(entry.entry_id, "duplicate_topic"))
            continue
        seen.add(entry.topic)
        if (
            paths
            and entry.paths
            and not any(
                path == scope or path.startswith(scope + "/")
                for path in paths
                for scope in entry.paths
            )
        ):
            continue
        if error_type is not None and entry.error_type not in (None, error_type):
            continue
        eligible.append(entry)
    # Score the whole eligible set at once: IDF is a corpus statistic, so ranking
    # one record in isolation would silently change every score.
    fielded_entries = tuple(fielded(entry) for entry in eligible)
    scored = rank(fielded_entries, terms, selector)
    cards = [
        _card(entry, ranked)
        for entry, ranked in zip(eligible, scored, strict=True)
        if query is None or ranked.score > 0
    ]
    cards.sort(key=lambda item: (-item.score, item.entry_id))
    omissions.extend(MemoryOmission(card.entry_id, "selection_limit") for card in cards[limit:])
    # The caller's path/error filter narrows what may be returned, not what the
    # project knows.  Measuring overlap on the filtered set reported a lexical miss
    # that does not exist, which misdirects the diagnosis toward semantic recall.
    overlap = not terms or any(
        not terms.isdisjoint(fielded(entry).tokens()) for entry in candidates
    )
    return MemorySearchResult(
        tuple(cards[:limit]),
        tuple(sorted(omissions, key=lambda item: (item.entry_id, item.reason))),
        snapshot_sha256(scoped),
        hashlib.sha256((query or "").encode()).hexdigest(),
        selector,
        overlap,
    )
