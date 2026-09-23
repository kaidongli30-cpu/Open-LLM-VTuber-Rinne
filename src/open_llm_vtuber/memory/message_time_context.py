"""Compact timestamp labels for chat messages shown to language models."""

from __future__ import annotations

from datetime import date, datetime, timedelta, tzinfo
from typing import Any


MEMORY_DAY_BOUNDARY_HOUR = 3


def parse_message_timestamp(
    value: Any,
    *,
    local_timezone: tzinfo | None = None,
) -> datetime | None:
    """Parse one persisted timestamp and normalize it to the local timezone."""

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
    else:
        return None

    timezone_value = local_timezone or datetime.now().astimezone().tzinfo
    if timezone_value is None:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone_value)
    return parsed.astimezone(timezone_value)


def memory_day_for_timestamp(
    timestamp: datetime,
    *,
    boundary_hour: int = MEMORY_DAY_BOUNDARY_HOUR,
) -> date:
    """Return the logical day containing ``timestamp`` at a 03:00 boundary."""

    if not 0 <= boundary_hour <= 23:
        raise ValueError("boundary_hour must be between 0 and 23")
    local_timestamp = timestamp
    if local_timestamp.hour < boundary_hour:
        local_timestamp -= timedelta(days=1)
    return local_timestamp.date()


def format_message_time_label(
    timestamp: datetime | str,
    *,
    local_timezone: tzinfo | None = None,
    boundary_hour: int = MEMORY_DAY_BOUNDARY_HOUR,
) -> str | None:
    """Format a compact, model-facing timestamp and logical-memory-day label."""

    parsed = parse_message_timestamp(timestamp, local_timezone=local_timezone)
    if parsed is None:
        return None
    memory_day = memory_day_for_timestamp(parsed, boundary_hour=boundary_hour)
    return (
        f"【消息时间：{parsed.isoformat(timespec='seconds')}｜"
        f"记忆日：{memory_day.isoformat()}（{boundary_hour:02d}:00分界）】"
    )


def add_message_time_context(
    content: str,
    timestamp: datetime | str,
    *,
    local_timezone: tzinfo | None = None,
    boundary_hour: int = MEMORY_DAY_BOUNDARY_HOUR,
) -> str:
    """Prefix message text without changing the persisted original text."""

    label = format_message_time_label(
        timestamp,
        local_timezone=local_timezone,
        boundary_hour=boundary_hour,
    )
    if not label:
        return content
    return f"{label}\n{content}"


__all__ = [
    "MEMORY_DAY_BOUNDARY_HOUR",
    "add_message_time_context",
    "format_message_time_label",
    "memory_day_for_timestamp",
    "parse_message_timestamp",
]
