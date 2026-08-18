"""Data contracts shared by the future three-layer memory system.

These types do not read or write chat history, diaries, caches, or models.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .current_conversation import CurrentConversationState


def utc_now() -> datetime:
    """Return an aware UTC timestamp for dataclass default factories."""

    return datetime.now(timezone.utc)


def ensure_aware_datetime(
    value: datetime | None,
    field_name: str,
) -> None:
    """Reject naive datetimes at memory-system boundaries."""

    if value is not None and (
        value.tzinfo is None or value.utcoffset() is None
    ):
        raise ValueError(f"{field_name} must be timezone-aware")


def parse_aware_datetime(value: datetime | str, field_name: str) -> datetime:
    """Restore an aware datetime previously emitted by ``to_dict``."""

    parsed = value
    if isinstance(value, str):
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
    if not isinstance(parsed, datetime):
        raise TypeError(f"{field_name} must be a datetime or ISO datetime string")
    ensure_aware_datetime(parsed, field_name)
    return parsed


def _to_serializable(value: Any) -> Any:
    """Convert memory dataclasses into JSON-compatible Python values."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {
            item.name: _to_serializable(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(_to_serializable(key)): _to_serializable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(item) for item in value]
    return value


class SerializableMemoryModel:
    """Mixin exposing a stable dictionary conversion for memory contracts."""

    def to_dict(self) -> dict[str, Any]:
        serialized = _to_serializable(self)
        if not isinstance(serialized, dict):
            raise TypeError("Memory model serialization must produce a dictionary")
        return serialized


class MemorySource(str, Enum):
    """Logical layer that supplied a fact or memory item."""

    CURRENT_CONVERSATION = "current_conversation"
    RUNTIME_OVERRIDE = "runtime_override"
    USER_BACKGROUND = "user_background"
    LONG_TERM_MEMORY = "long_term_memory"


class MemoryItemStatus(str, Enum):
    """Validity state retained with facts and long-term memories."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    HISTORICAL = "historical"
    UNCERTAIN = "uncertain"


class LongTermMemoryResultStatus(str, Enum):
    """Externally meaningful outcome of a future long-term retrieval."""

    CONFIRMED = "confirmed"
    AMBIGUOUS = "ambiguous"
    NO_MATCH = "no_match"


class MemoryDecisionAction(str, Enum):
    """Actions available to the future Memory Router."""

    CONTEXT_ONLY = "context_only"
    RETRIEVE_LONG_TERM = "retrieve_long_term"
    CLARIFY = "clarify"
    BYPASS = "bypass"


@dataclass
class ConversationTurn(SerializableMemoryModel):
    """One bounded turn in the current conversation layer."""

    turn_id: str
    role: str
    content: str
    timestamp: datetime = field(default_factory=utc_now)
    source_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ensure_aware_datetime(self.timestamp, "ConversationTurn.timestamp")


@dataclass
class NearbyReference(SerializableMemoryModel):
    """A near-distance expression and its currently resolved referent."""

    expression: str
    resolved_to: str | None = None
    source_turn_id: str | None = None
    observed_at: datetime = field(default_factory=utc_now)
    confidence: float = 0.0

    def __post_init__(self) -> None:
        ensure_aware_datetime(
            self.observed_at,
            "NearbyReference.observed_at",
        )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass
class MemoryFact(SerializableMemoryModel):
    """A source-aware fact that can participate in precedence decisions."""

    fact_id: str
    key: str
    value: Any
    source: MemorySource
    observed_at: datetime = field(default_factory=utc_now)
    source_ref: str | None = None
    effective_from: datetime | None = None
    effective_to: datetime | None = None
    status: MemoryItemStatus = MemoryItemStatus.ACTIVE
    confidence: float = 1.0
    explicit: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ensure_aware_datetime(self.observed_at, "MemoryFact.observed_at")
        ensure_aware_datetime(
            self.effective_from,
            "MemoryFact.effective_from",
        )
        ensure_aware_datetime(self.effective_to, "MemoryFact.effective_to")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass
class UserBackgroundState(SerializableMemoryModel):
    """Startup snapshot of the user's currently valid background state."""

    user_id: str
    version: str = "unversioned"
    generated_at: datetime = field(default_factory=utc_now)
    sections: dict[str, Any] = field(default_factory=dict)
    facts: list[MemoryFact] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    is_valid: bool = True
    valid_until: datetime | None = None

    def __post_init__(self) -> None:
        ensure_aware_datetime(
            self.generated_at,
            "UserBackgroundState.generated_at",
        )
        ensure_aware_datetime(
            self.valid_until,
            "UserBackgroundState.valid_until",
        )


@dataclass
class RuntimeStateOverride(SerializableMemoryModel):
    """One runtime fact and the startup-background facts it supersedes."""

    override_id: str
    fact: MemoryFact
    overrides_background_fact_ids: list[str] = field(default_factory=list)
    source_message_id: str | None = None
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        ensure_aware_datetime(
            self.updated_at,
            "RuntimeStateOverride.updated_at",
        )


@dataclass
class RuntimeStateOverrides(SerializableMemoryModel):
    """Process-run collection of state changes newer than the background."""

    run_id: str
    user_id: str
    items: list[RuntimeStateOverride] = field(default_factory=list)
    started_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        ensure_aware_datetime(
            self.started_at,
            "RuntimeStateOverrides.started_at",
        )
        ensure_aware_datetime(
            self.updated_at,
            "RuntimeStateOverrides.updated_at",
        )


@dataclass
class LongTermMemoryItem(SerializableMemoryModel):
    """One detailed memory returned by a future long-term memory source."""

    memory_id: str
    content: str
    source_file: str
    memory_time: datetime | None = None
    confidence: float = 0.0
    status: MemoryItemStatus = MemoryItemStatus.UNCERTAIN
    source_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ensure_aware_datetime(
            self.memory_time,
            "LongTermMemoryItem.memory_time",
        )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass
class LongTermMemoryResult(SerializableMemoryModel):
    """Strictly separated confirmed, ambiguous, and no-match outcomes."""

    status: LongTermMemoryResultStatus = LongTermMemoryResultStatus.NO_MATCH
    confirmed_memories: list[LongTermMemoryItem] = field(default_factory=list)
    unconfirmed_candidates: list[LongTermMemoryItem] = field(default_factory=list)
    no_match_reason: str | None = None
    retrieval_health: str = "not_run"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status is LongTermMemoryResultStatus.CONFIRMED:
            if not self.confirmed_memories:
                raise ValueError("confirmed results require confirmed_memories")
            if self.unconfirmed_candidates:
                raise ValueError(
                    "confirmed results cannot contain unconfirmed_candidates"
                )
            if any(
                item.status is MemoryItemStatus.UNCERTAIN
                for item in self.confirmed_memories
            ):
                raise ValueError(
                    "confirmed memories cannot have uncertain status"
                )
        elif self.status is LongTermMemoryResultStatus.AMBIGUOUS:
            if not self.unconfirmed_candidates:
                raise ValueError(
                    "ambiguous results require unconfirmed_candidates"
                )
            if self.confirmed_memories:
                raise ValueError(
                    "ambiguous results cannot contain confirmed_memories"
                )
        elif self.confirmed_memories or self.unconfirmed_candidates:
            raise ValueError("no_match results cannot contain memory items")

    @property
    def no_match(self) -> bool:
        return self.status is LongTermMemoryResultStatus.NO_MATCH

    def to_dict(self) -> dict[str, Any]:
        serialized = super().to_dict()
        serialized["no_match"] = self.no_match
        return serialized


@dataclass
class MemoryDecision(SerializableMemoryModel):
    """Decision made before any optional long-term retrieval."""

    action: MemoryDecisionAction = MemoryDecisionAction.CONTEXT_ONLY
    reason_codes: list[str] = field(default_factory=list)
    retrieval_query: str | None = None
    missing_clues: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def should_retrieve_long_term(self) -> bool:
        return self.action is MemoryDecisionAction.RETRIEVE_LONG_TERM


@dataclass
class MemoryContextPackage(SerializableMemoryModel):
    """Future handoff object for the LLM message-construction layer."""

    decision: MemoryDecision = field(default_factory=MemoryDecision)
    current_conversation: CurrentConversationState | None = None
    user_background: UserBackgroundState | None = None
    runtime_overrides: RuntimeStateOverrides | None = None
    long_term_result: LongTermMemoryResult = field(
        default_factory=LongTermMemoryResult
    )
    dynamic_system_context: str = ""
    llm_messages: list[dict[str, Any]] = field(default_factory=list)
    response_guidance: str | None = None
    budget_profile_name: str | None = None
    internal_diagnostics: dict[str, Any] = field(default_factory=dict)
