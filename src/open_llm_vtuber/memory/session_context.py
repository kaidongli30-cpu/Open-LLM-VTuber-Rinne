"""Pure helpers for hydrating a cloud conversation's local memory context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .recent_memory_context import load_recent_memory_context
from .today_history_loader import TodayHistoryLoader


@dataclass(frozen=True)
class CloudSessionMemory:
    """The initial conversation turns and separate recent-diary context."""

    messages: tuple[dict[str, str], ...]
    recent_context: str
    diagnostics: dict[str, Any]
    diary_count: int
    weekly_count: int
    today_raw_count: int


def load_cloud_session_memory(
    history_dir: Path,
    *,
    recent_memory_days: int,
    include_recent_context: bool,
) -> CloudSessionMemory:
    """Prepare recent reviewed context alongside current memory-day chat.

    This function is intentionally free of WebSocket, TTS, LLM, and history-write
    side effects.  Recent diaries are returned separately so the agent can place
    them in its single request-level system prompt, while today's raw turns stay
    as ordinary conversation messages.
    """

    recent_result = load_recent_memory_context(
        history_dir,
        days=recent_memory_days,
    )
    messages: list[dict[str, str]] = []

    today_loader = TodayHistoryLoader(history_dir)
    today_result = today_loader.load()
    messages.extend(
        {
            "role": turn.role,
            "content": turn.content,
            "timestamp": turn.timestamp.isoformat(timespec="seconds"),
        }
        for turn in today_result.turns
    )
    diagnostics = {
        "memory_day": recent_result.memory_day.isoformat(),
        "window_start": recent_result.window_start.isoformat(),
        "window_end": recent_result.window_end.isoformat(),
        "diary_files": [item.source_file for item in recent_result.diary_entries],
        "weekly_files": [item.source_file for item in recent_result.weekly_entries],
        "covered_diary_files": list(recent_result.covered_diary_files),
        "missing_diary_dates": list(recent_result.missing_diary_dates),
        "character_count": recent_result.character_count,
        "warnings": list(recent_result.warnings),
    }
    return CloudSessionMemory(
        messages=tuple(messages),
        recent_context=(recent_result.context if include_recent_context else ""),
        diagnostics=diagnostics,
        diary_count=len(recent_result.diary_entries),
        weekly_count=len(recent_result.weekly_entries),
        today_raw_count=len(today_result.turns),
    )


__all__ = ["CloudSessionMemory", "load_cloud_session_memory"]
