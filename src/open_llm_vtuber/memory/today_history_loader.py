"""Read-only loader for the current logical memory day's raw chat turns."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any

from .current_conversation import CurrentConversationState
from .types import (
    ConversationTurn,
    SerializableMemoryModel,
    ensure_aware_datetime,
)


@dataclass
class TodayHistoryDiagnostics(SerializableMemoryModel):
    """Content-free statistics from one root-directory scan."""

    scanned_files: int = 0
    parsed_files: int = 0
    corrupted_files: int = 0
    parsed_records: int = 0
    metadata_records: int = 0
    missing_timestamp_records: int = 0
    invalid_timestamp_records: int = 0
    invalid_content_records: int = 0
    unsupported_role_records: int = 0
    outside_memory_day_records: int = 0
    eligible_records: int = 0
    duplicate_records: int = 0
    truncated_records: int = 0
    returned_turns: int = 0
    mixed_memory_day_files: int = 0
    history_root_missing: bool = False
    corrupted_file_names: list[str] = field(default_factory=list)
    mixed_file_names: list[str] = field(default_factory=list)


@dataclass
class TodayHistoryLoadResult(SerializableMemoryModel):
    """Sorted, deduplicated turns and content-free loader diagnostics."""

    memory_day: date
    window_start: datetime
    window_end: datetime
    turns: list[ConversationTurn] = field(default_factory=list)
    diagnostics: TodayHistoryDiagnostics = field(
        default_factory=TodayHistoryDiagnostics
    )

    def __post_init__(self) -> None:
        ensure_aware_datetime(self.window_start, "TodayHistoryLoadResult.window_start")
        ensure_aware_datetime(self.window_end, "TodayHistoryLoadResult.window_end")


class TodayHistoryLoader:
    """Load direct ``*.json`` children for one local 03:00 memory day."""

    _ROLE_MAP = {
        "human": "user",
        "user": "user",
        "ai": "assistant",
        "assistant": "assistant",
    }

    def __init__(
        self,
        history_root: str | Path = Path("chat_history/rinne_01"),
        *,
        local_timezone: tzinfo | None = None,
        boundary_hour: int = 3,
    ) -> None:
        if not 0 <= boundary_hour <= 23:
            raise ValueError("boundary_hour must be between 0 and 23")
        self.history_root = Path(history_root)
        self.local_timezone = local_timezone or datetime.now().astimezone().tzinfo
        if self.local_timezone is None:
            raise ValueError("A local timezone is required")
        self.boundary_hour = boundary_hour

    def memory_day_window(
        self,
        reference_time: datetime | None = None,
    ) -> tuple[date, datetime, datetime]:
        """Return the local half-open interval ``[03:00, next 03:00)``."""

        if reference_time is None:
            reference_time = datetime.now(self.local_timezone)
        ensure_aware_datetime(reference_time, "reference_time")
        local_reference = reference_time.astimezone(self.local_timezone)
        boundary = datetime.combine(
            local_reference.date(),
            time(hour=self.boundary_hour),
            tzinfo=self.local_timezone,
        )
        if local_reference < boundary:
            boundary -= timedelta(days=1)
        return boundary.date(), boundary, boundary + timedelta(days=1)

    def memory_day_for(self, timestamp: datetime) -> date:
        ensure_aware_datetime(timestamp, "timestamp")
        local_timestamp = timestamp.astimezone(self.local_timezone)
        if local_timestamp.hour < self.boundary_hour:
            local_timestamp -= timedelta(days=1)
        return local_timestamp.date()

    def _parse_timestamp(self, raw_timestamp: Any) -> datetime:
        if not isinstance(raw_timestamp, str):
            raise TypeError("timestamp must be an ISO string")
        normalized = (
            raw_timestamp[:-1] + "+00:00"
            if raw_timestamp.endswith("Z")
            else raw_timestamp
        )
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            parsed = parsed.replace(tzinfo=self.local_timezone)
        return parsed.astimezone(self.local_timezone)

    def _empty_result(
        self,
        reference_time: datetime | None,
        diagnostics: TodayHistoryDiagnostics,
    ) -> TodayHistoryLoadResult:
        memory_day, start, end = self.memory_day_window(reference_time)
        return TodayHistoryLoadResult(
            memory_day=memory_day,
            window_start=start,
            window_end=end,
            diagnostics=diagnostics,
        )

    def load(
        self,
        *,
        reference_time: datetime | None = None,
        limit: int | None = None,
    ) -> TodayHistoryLoadResult:
        """Scan direct JSON children, merge, sort, deduplicate, then limit."""

        if limit is not None and limit < 0:
            raise ValueError("limit cannot be negative")

        diagnostics = TodayHistoryDiagnostics()
        result = self._empty_result(reference_time, diagnostics)
        if not self.history_root.is_dir():
            diagnostics.history_root_missing = True
            return result

        json_files = sorted(
            path
            for path in self.history_root.glob("*.json")
            if path.is_file()
        )
        diagnostics.scanned_files = len(json_files)
        state = CurrentConversationState(
            session_id=f"today-history:{result.memory_day.isoformat()}"
        )

        for path in json_files:
            file_days: set[date] = set()
            try:
                with path.open("r", encoding="utf-8") as file:
                    records = json.load(file)
                if not isinstance(records, list):
                    raise TypeError("history file must contain a JSON array")
            except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
                diagnostics.corrupted_files += 1
                diagnostics.corrupted_file_names.append(path.name)
                continue

            diagnostics.parsed_files += 1
            for index, record in enumerate(records):
                diagnostics.parsed_records += 1
                if not isinstance(record, dict):
                    diagnostics.invalid_content_records += 1
                    continue

                raw_role = str(record.get("role", "")).casefold()
                if raw_role == "metadata":
                    diagnostics.metadata_records += 1
                    continue
                role = self._ROLE_MAP.get(raw_role)
                if role is None:
                    diagnostics.unsupported_role_records += 1
                    continue

                raw_timestamp = record.get("timestamp")
                if raw_timestamp in {None, ""}:
                    diagnostics.missing_timestamp_records += 1
                    continue
                try:
                    timestamp = self._parse_timestamp(raw_timestamp)
                except (TypeError, ValueError, OverflowError):
                    diagnostics.invalid_timestamp_records += 1
                    continue

                file_days.add(self.memory_day_for(timestamp))
                content = record.get("content")
                if not isinstance(content, str):
                    diagnostics.invalid_content_records += 1
                    continue
                if not result.window_start <= timestamp < result.window_end:
                    diagnostics.outside_memory_day_records += 1
                    continue

                diagnostics.eligible_records += 1
                turn = ConversationTurn(
                    turn_id=f"history:{path.name}:{index}",
                    role=role,
                    content=content,
                    timestamp=timestamp,
                    source_ref=f"{path.name}#{index}",
                    metadata={
                        "origin": "today_history",
                        "history_file": path.name,
                        "record_index": index,
                        "original_role": raw_role,
                    },
                )
                if not state.add_turn(turn):
                    diagnostics.duplicate_records += 1

            if len(file_days) > 1:
                diagnostics.mixed_memory_day_files += 1
                diagnostics.mixed_file_names.append(path.name)

        if limit is not None:
            diagnostics.truncated_records = state.trim(
                budget_policy=_NewestTurnLimit(limit)
            )
        diagnostics.returned_turns = len(state.recent_turns)
        result.turns = list(state.recent_turns)
        return result


@dataclass(frozen=True)
class _NewestTurnLimit:
    """Private adapter keeping loader limits independent of state defaults."""

    limit: int

    def apply(
        self,
        turns: list[ConversationTurn] | tuple[ConversationTurn, ...],
    ) -> list[ConversationTurn]:
        ordered = sorted(
            turns,
            key=lambda turn: turn.timestamp.astimezone(timezone.utc),
        )
        if self.limit == 0:
            return []
        return ordered[-self.limit :]
