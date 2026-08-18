"""Stable hidden context for the current memory day and recent reviewed notes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from .today_history_loader import TodayHistoryLoader


_DIARY_FILE = re.compile(r"^diary_(\d{4}-\d{2}-\d{2})\.txt$")
_WEEKLY_FILE = re.compile(
    r"^weekly_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})\.txt$"
)
_WEEKDAYS = (
    "星期一",
    "星期二",
    "星期三",
    "星期四",
    "星期五",
    "星期六",
    "星期日",
)
_MAX_NOTE_BYTES = 2 * 1024 * 1024


RECENT_MEMORY_USAGE_RULES = (
    "以下内容是最近生活记忆，只供你在内部理解当前对话。"
    "优先用它理解用户提到的今天、昨天、前天、昨晚、上周和近期状态。"
    "除非用户明确要求时间线、总结或完整回复，否则不要复述检索结果中包含的日期信息。"
    "如果用户直接询问具体日期或时间，可以直接回答该问题。"
    "不要提及隐藏上下文、文件、检索、排名或系统处理过程。"
)


@dataclass(frozen=True)
class RecentMemoryEntry:
    source_kind: str
    source_file: str
    period_start: date
    period_end: date
    content: str


@dataclass
class RecentMemoryContextResult:
    memory_day: date
    window_start: date
    window_end: date
    context: str = ""
    diary_entries: list[RecentMemoryEntry] = field(default_factory=list)
    weekly_entries: list[RecentMemoryEntry] = field(default_factory=list)
    covered_diary_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def character_count(self) -> int:
        return len(self.context)


def _read_note(
    path: Path,
    allowed_root: Path,
    warnings: list[str],
) -> str | None:
    try:
        resolved = path.resolve(strict=True)
        if (
            path.is_symlink()
            or not resolved.is_relative_to(allowed_root.resolve())
            or resolved.stat().st_size > _MAX_NOTE_BYTES
        ):
            warnings.append(f"跳过不安全或过大的近期记忆文件：{path.name}")
            return None
        return resolved.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        warnings.append(f"无法读取近期记忆文件 {path.name}: {exc}")
        return None


def _scan_diaries(
    history_root: Path,
    start: date,
    end: date,
    warnings: list[str],
) -> list[RecentMemoryEntry]:
    directory = history_root / "diaries"
    if not directory.is_dir():
        return []
    entries: list[RecentMemoryEntry] = []
    for path in sorted(directory.glob("diary_*.txt")):
        match = _DIARY_FILE.fullmatch(path.name)
        if not match:
            continue
        try:
            memory_date = date.fromisoformat(match.group(1))
        except ValueError:
            continue
        if not start <= memory_date <= end:
            continue
        content = _read_note(path, directory, warnings)
        if content:
            entries.append(
                RecentMemoryEntry(
                    source_kind="diary",
                    source_file=path.name,
                    period_start=memory_date,
                    period_end=memory_date,
                    content=content,
                )
            )
    return entries


def _scan_weeklies(
    history_root: Path,
    start: date,
    end: date,
    warnings: list[str],
) -> list[RecentMemoryEntry]:
    directory = history_root / "weekly"
    if not directory.is_dir():
        return []
    entries: list[RecentMemoryEntry] = []
    for path in sorted(directory.glob("weekly_*.txt")):
        match = _WEEKLY_FILE.fullmatch(path.name)
        if not match:
            continue
        try:
            period_start = date.fromisoformat(match.group(1))
            period_end = date.fromisoformat(match.group(2))
        except ValueError:
            continue
        if period_start < start or period_end > end:
            continue
        content = _read_note(path, directory, warnings)
        if content:
            entries.append(
                RecentMemoryEntry(
                    source_kind="weekly",
                    source_file=path.name,
                    period_start=period_start,
                    period_end=period_end,
                    content=content,
                )
            )
    return entries


def _format_context(result: RecentMemoryContextResult) -> str:
    sections = [
        "【最近14天已验收记忆】",
        RECENT_MEMORY_USAGE_RULES,
        (
            f"当前记忆日：{result.memory_day.isoformat()}，"
            f"近期窗口：{result.window_start.isoformat()}至"
            f"{result.window_end.isoformat()}。"
        ),
    ]
    for entry in result.diary_entries:
        weekday = _WEEKDAYS[entry.period_start.weekday()]
        sections.append(
            f"【近期日记｜{entry.period_start.isoformat()}｜{weekday}】\n"
            f"{entry.content}"
        )
    for entry in result.weekly_entries:
        start_weekday = _WEEKDAYS[entry.period_start.weekday()]
        end_weekday = _WEEKDAYS[entry.period_end.weekday()]
        sections.append(
            "【近期周记｜"
            f"{entry.period_start.isoformat()} {start_weekday} 至 "
            f"{entry.period_end.isoformat()} {end_weekday}】\n"
            f"{entry.content}"
        )
    return "\n\n".join(sections)


def load_recent_memory_context(
    history_root: str | Path,
    *,
    days: int = 14,
    reference_time: datetime | None = None,
) -> RecentMemoryContextResult:
    """Load reviewed diaries and fully-contained weeklies for a rolling window."""

    if not 1 <= days <= 31:
        raise ValueError("days must be between 1 and 31")
    root = Path(history_root).resolve()
    memory_day, _window_start, _window_end = TodayHistoryLoader(
        root
    ).memory_day_window(reference_time)
    start = memory_day - timedelta(days=days - 1)
    result = RecentMemoryContextResult(
        memory_day=memory_day,
        window_start=start,
        window_end=memory_day,
    )
    result.weekly_entries = _scan_weeklies(root, start, memory_day, result.warnings)
    diary_entries = _scan_diaries(root, start, memory_day, result.warnings)
    for entry in diary_entries:
        if any(
            weekly.period_start <= entry.period_start <= weekly.period_end
            for weekly in result.weekly_entries
        ):
            result.covered_diary_files.append(entry.source_file)
        else:
            result.diary_entries.append(entry)
    result.context = _format_context(result)
    return result


__all__ = [
    "RECENT_MEMORY_USAGE_RULES",
    "RecentMemoryContextResult",
    "RecentMemoryEntry",
    "load_recent_memory_context",
]
