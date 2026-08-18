"""Independent current-conversation state prototype.

This module does not replace or interact with ``BasicMemoryAgent._memory``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from .types import (
    ConversationTurn,
    MemoryFact,
    MemoryItemStatus,
    MemorySource,
    NearbyReference,
    SerializableMemoryModel,
    ensure_aware_datetime,
    parse_aware_datetime,
    utc_now,
)


class ConversationBudgetPolicy(Protocol):
    """Pluggable trim contract for future count- or token-based policies."""

    def apply(
        self,
        turns: Sequence[ConversationTurn],
    ) -> list[ConversationTurn]:
        """Return the turns that should remain in the state."""


@dataclass(frozen=True)
class TurnCountBudget:
    """Stage-2 policy retaining only the newest ``max_turns`` turns."""

    max_turns: int

    def __post_init__(self) -> None:
        if self.max_turns < 0:
            raise ValueError("max_turns cannot be negative")

    def apply(
        self,
        turns: Sequence[ConversationTurn],
    ) -> list[ConversationTurn]:
        ordered = sorted(turns, key=_turn_sort_key)
        if self.max_turns == 0:
            return []
        return ordered[-self.max_turns :]


def _turn_sort_key(turn: ConversationTurn) -> tuple[datetime, str, str]:
    return (
        turn.timestamp.astimezone(timezone.utc),
        turn.source_ref or "",
        turn.turn_id,
    )


def _turn_fingerprint(turn: ConversationTurn) -> tuple[str, str, datetime]:
    return (
        turn.role.casefold(),
        turn.content,
        turn.timestamp.astimezone(timezone.utc),
    )


def _memory_fact_from_dict(data: dict[str, Any]) -> MemoryFact:
    return MemoryFact(
        fact_id=data["fact_id"],
        key=data["key"],
        value=data.get("value"),
        source=MemorySource(data["source"]),
        observed_at=parse_aware_datetime(
            data["observed_at"],
            "MemoryFact.observed_at",
        ),
        source_ref=data.get("source_ref"),
        effective_from=(
            parse_aware_datetime(
                data["effective_from"],
                "MemoryFact.effective_from",
            )
            if data.get("effective_from") is not None
            else None
        ),
        effective_to=(
            parse_aware_datetime(
                data["effective_to"],
                "MemoryFact.effective_to",
            )
            if data.get("effective_to") is not None
            else None
        ),
        status=MemoryItemStatus(data.get("status", MemoryItemStatus.ACTIVE.value)),
        confidence=float(data.get("confidence", 1.0)),
        explicit=bool(data.get("explicit", True)),
        metadata=dict(data.get("metadata", {})),
    )


@dataclass
class CurrentConversationState(SerializableMemoryModel):
    """Per-session, bounded, serializable current-conversation state."""

    session_id: str
    recent_turns: list[ConversationTurn] = field(default_factory=list)
    current_topics: list[str] = field(default_factory=list)
    nearby_references: list[NearbyReference] = field(default_factory=list)
    new_facts: list[MemoryFact] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    max_turns: int | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        ensure_aware_datetime(
            self.created_at,
            "CurrentConversationState.created_at",
        )
        ensure_aware_datetime(
            self.updated_at,
            "CurrentConversationState.updated_at",
        )
        if self.max_turns is not None and self.max_turns < 0:
            raise ValueError("max_turns cannot be negative")
        self.sort_turns()
        self.deduplicate_turns()
        self.trim()

    @staticmethod
    def _is_duplicate(
        existing: ConversationTurn,
        candidate: ConversationTurn,
    ) -> bool:
        if (
            existing.turn_id
            and candidate.turn_id
            and existing.turn_id == candidate.turn_id
        ):
            return True
        return _turn_fingerprint(existing) == _turn_fingerprint(candidate)

    def add_turn(
        self,
        turn: ConversationTurn,
        *,
        replace_existing: bool = False,
        budget_policy: ConversationBudgetPolicy | None = None,
    ) -> bool:
        """Add one turn; return ``False`` when it was a duplicate."""

        for index, existing in enumerate(self.recent_turns):
            if self._is_duplicate(existing, turn):
                if replace_existing:
                    self.recent_turns[index] = turn
                    self.sort_turns()
                    self.trim(budget_policy)
                    self.updated_at = max(self.updated_at, turn.timestamp)
                return False

        self.recent_turns.append(turn)
        self.sort_turns()
        self.trim(budget_policy)
        self.updated_at = max(self.updated_at, turn.timestamp)
        return True

    def add_turns(
        self,
        turns: Iterable[ConversationTurn],
        *,
        replace_existing: bool = False,
        budget_policy: ConversationBudgetPolicy | None = None,
    ) -> int:
        """Add many turns and return the number of non-duplicates."""

        added = 0
        for turn in turns:
            if self.add_turn(
                turn,
                replace_existing=replace_existing,
                budget_policy=budget_policy,
            ):
                added += 1
        return added

    def sort_turns(self) -> None:
        self.recent_turns.sort(key=_turn_sort_key)

    def deduplicate_turns(self, *, prefer_last: bool = False) -> int:
        """Remove duplicates and return the number removed."""

        source = list(reversed(self.recent_turns)) if prefer_last else self.recent_turns
        unique: list[ConversationTurn] = []
        for turn in source:
            if not any(self._is_duplicate(item, turn) for item in unique):
                unique.append(turn)
        if prefer_last:
            unique.reverse()
        removed = len(self.recent_turns) - len(unique)
        self.recent_turns = unique
        self.sort_turns()
        return removed

    def trim(
        self,
        budget_policy: ConversationBudgetPolicy | None = None,
    ) -> int:
        """Apply a pluggable policy; stage 2 defaults to a turn-count limit."""

        policy = budget_policy
        if policy is None and self.max_turns is not None:
            policy = TurnCountBudget(self.max_turns)
        if policy is None:
            return 0
        before = len(self.recent_turns)
        self.recent_turns = list(policy.apply(self.recent_turns))
        self.sort_turns()
        return before - len(self.recent_turns)

    def clear(self) -> None:
        """Clear all session-scoped conversation state."""

        self.recent_turns.clear()
        self.current_topics.clear()
        self.nearby_references.clear()
        self.new_facts.clear()
        self.source_refs.clear()
        self.updated_at = utc_now()

    def clone(self) -> CurrentConversationState:
        return self.from_dict(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CurrentConversationState:
        """Restore a state previously emitted by ``to_dict``."""

        turns = [
            ConversationTurn(
                turn_id=item["turn_id"],
                role=item["role"],
                content=item["content"],
                timestamp=parse_aware_datetime(
                    item["timestamp"],
                    "ConversationTurn.timestamp",
                ),
                source_ref=item.get("source_ref"),
                metadata=dict(item.get("metadata", {})),
            )
            for item in data.get("recent_turns", [])
        ]
        references = [
            NearbyReference(
                expression=item["expression"],
                resolved_to=item.get("resolved_to"),
                source_turn_id=item.get("source_turn_id"),
                observed_at=parse_aware_datetime(
                    item["observed_at"],
                    "NearbyReference.observed_at",
                ),
                confidence=float(item.get("confidence", 0.0)),
            )
            for item in data.get("nearby_references", [])
        ]
        return cls(
            session_id=data["session_id"],
            recent_turns=turns,
            current_topics=list(data.get("current_topics", [])),
            nearby_references=references,
            new_facts=[
                _memory_fact_from_dict(item)
                for item in data.get("new_facts", [])
            ],
            source_refs=list(data.get("source_refs", [])),
            max_turns=data.get("max_turns"),
            created_at=parse_aware_datetime(
                data["created_at"],
                "CurrentConversationState.created_at",
            ),
            updated_at=parse_aware_datetime(
                data["updated_at"],
                "CurrentConversationState.updated_at",
            ),
        )
