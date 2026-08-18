"""Simple, testable precedence rules for memory facts.

Stage 1 compares precedence only. It does not merge or mutate conflicting facts.
"""

from __future__ import annotations

from datetime import datetime
from enum import IntEnum

from .types import MemoryItemStatus, MemorySource


class MemoryPrecedence(IntEnum):
    """Higher values represent more authoritative current information."""

    STALE_OR_HISTORICAL = 10
    UNCERTAIN_EVIDENCE = 20
    CONFIRMED_LONG_TERM = 30
    USER_BACKGROUND = 40
    RUNTIME_OVERRIDE = 50
    CURRENT_CONVERSATION = 60


_ACTIVE_SOURCE_PRECEDENCE = {
    MemorySource.CURRENT_CONVERSATION: MemoryPrecedence.CURRENT_CONVERSATION,
    MemorySource.RUNTIME_OVERRIDE: MemoryPrecedence.RUNTIME_OVERRIDE,
    MemorySource.USER_BACKGROUND: MemoryPrecedence.USER_BACKGROUND,
    MemorySource.LONG_TERM_MEMORY: MemoryPrecedence.CONFIRMED_LONG_TERM,
}


def precedence_rank(
    source: MemorySource,
    status: MemoryItemStatus = MemoryItemStatus.ACTIVE,
    *,
    explicit: bool = True,
) -> MemoryPrecedence:
    """Return the general rank for one source/status combination."""

    if status in {MemoryItemStatus.SUPERSEDED, MemoryItemStatus.HISTORICAL}:
        return MemoryPrecedence.STALE_OR_HISTORICAL
    if status is MemoryItemStatus.UNCERTAIN or not explicit:
        return MemoryPrecedence.UNCERTAIN_EVIDENCE
    return _ACTIVE_SOURCE_PRECEDENCE[source]


def compare_precedence(
    left_source: MemorySource,
    right_source: MemorySource,
    *,
    left_status: MemoryItemStatus = MemoryItemStatus.ACTIVE,
    right_status: MemoryItemStatus = MemoryItemStatus.ACTIVE,
    left_explicit: bool = True,
    right_explicit: bool = True,
    left_observed_at: datetime | None = None,
    right_observed_at: datetime | None = None,
) -> int:
    """Compare two facts without merging them.

    Returns 1 when the left fact wins, -1 when the right fact wins, and 0 when
    the general rule cannot distinguish them. A newer observation breaks ties
    within the same precedence level when both timestamps are available.
    """

    left_rank = precedence_rank(
        left_source, left_status, explicit=left_explicit
    )
    right_rank = precedence_rank(
        right_source, right_status, explicit=right_explicit
    )

    if left_rank > right_rank:
        return 1
    if left_rank < right_rank:
        return -1
    if left_observed_at is not None and right_observed_at is not None:
        if left_observed_at > right_observed_at:
            return 1
        if left_observed_at < right_observed_at:
            return -1
    return 0
