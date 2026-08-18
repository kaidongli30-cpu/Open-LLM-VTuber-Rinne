"""Load non-overlapping diary, weekly, and monthly memories.

This module is deliberately read-only.  It never creates, edits, moves, or
deletes memory files.  Higher-level summaries take precedence over their
source material so the same date is not sent to an LLM more than once.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path


_DIARY_PATTERN = re.compile(r"^diary_(\d{4}-\d{2}-\d{2})\.txt$")
_WEEKLY_PATTERN = re.compile(
    r"^weekly_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})\.txt$"
)
_MONTHLY_PATTERN = re.compile(r"^monthly_(\d{4}-\d{2})\.txt$")


@dataclass(frozen=True)
class LongTermArchiveEntry:
    """One validated long-term memory file and the dates it covers."""

    kind: str
    path: Path
    start_date: date
    end_date: date
    content: str

    def covers(self, target_date: date) -> bool:
        return self.start_date <= target_date <= self.end_date

    def overlaps(self, other: "LongTermArchiveEntry") -> bool:
        return self.start_date <= other.end_date and other.start_date <= self.end_date


@dataclass
class LongTermArchiveDiagnostics:
    invalid_filenames: int = 0
    unreadable_files: int = 0
    empty_files: int = 0
    skipped_overlapping_weeklies: int = 0
    skipped_covered_diaries: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class LongTermMemorySelection:
    """The exact non-overlapping memory set that may be sent to the LLM."""

    monthly_entries: list[LongTermArchiveEntry] = field(default_factory=list)
    weekly_entries: list[LongTermArchiveEntry] = field(default_factory=list)
    diary_entries: list[LongTermArchiveEntry] = field(default_factory=list)
    diagnostics: LongTermArchiveDiagnostics = field(
        default_factory=LongTermArchiveDiagnostics
    )

    @property
    def total_entries(self) -> int:
        return (
            len(self.monthly_entries)
            + len(self.weekly_entries)
            + len(self.diary_entries)
        )

    def ordered_entries(self) -> list[LongTermArchiveEntry]:
        return sorted(
            self.monthly_entries + self.weekly_entries + self.diary_entries,
            key=lambda item: (item.start_date, item.end_date, item.kind),
        )

    def to_llm_text(self) -> str:
        """Format selected memories without exposing internal diagnostics."""

        if not self.total_entries:
            return ""

        kind_labels = {"monthly": "月记", "weekly": "周记", "diary": "日记"}
        sections = [
            "以下是你亲手写下的长期记忆。系统已经按日期去除重复内容："
            "月记覆盖的日期不再重复提供周记或日记，周记覆盖的日期不再重复提供日记。"
            "请把它们作为过去经历理解；若与用户刚刚明确说出的新情况冲突，"
            "以最新对话为准。"
        ]
        for entry in self.ordered_entries():
            label = kind_labels[entry.kind]
            if entry.start_date == entry.end_date:
                period = entry.start_date.isoformat()
            else:
                period = (
                    f"{entry.start_date.isoformat()} 至 {entry.end_date.isoformat()}"
                )
            sections.append(f"【{label}：{period}】\n{entry.content}")
        return "\n\n".join(sections)


def _month_bounds(month_label: str) -> tuple[date, date]:
    start = datetime.strptime(f"{month_label}-01", "%Y-%m-%d").date()
    if start.month == 12:
        next_month = date(start.year + 1, 1, 1)
    else:
        next_month = date(start.year, start.month + 1, 1)
    return start, next_month - timedelta(days=1)


def _read_entry(
    path: Path,
    kind: str,
    start_date: date,
    end_date: date,
    diagnostics: LongTermArchiveDiagnostics,
) -> LongTermArchiveEntry | None:
    try:
        content = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        diagnostics.unreadable_files += 1
        diagnostics.warnings.append(f"无法读取 {path.name}: {exc}")
        return None
    if not content:
        diagnostics.empty_files += 1
        diagnostics.warnings.append(f"跳过空文件 {path.name}")
        return None
    return LongTermArchiveEntry(kind, path, start_date, end_date, content)


def _scan_monthlies(
    directory: Path, diagnostics: LongTermArchiveDiagnostics
) -> list[LongTermArchiveEntry]:
    entries: list[LongTermArchiveEntry] = []
    if not directory.exists():
        return entries
    for path in sorted(directory.glob("monthly_*.txt")):
        match = _MONTHLY_PATTERN.fullmatch(path.name)
        if not match:
            diagnostics.invalid_filenames += 1
            continue
        try:
            start, end = _month_bounds(match.group(1))
        except ValueError:
            diagnostics.invalid_filenames += 1
            continue
        entry = _read_entry(path, "monthly", start, end, diagnostics)
        if entry:
            entries.append(entry)
    return entries


def _scan_weeklies(
    directory: Path, diagnostics: LongTermArchiveDiagnostics
) -> list[LongTermArchiveEntry]:
    entries: list[LongTermArchiveEntry] = []
    if not directory.exists():
        return entries
    for path in sorted(directory.glob("weekly_*.txt")):
        match = _WEEKLY_PATTERN.fullmatch(path.name)
        if not match:
            diagnostics.invalid_filenames += 1
            continue
        try:
            start = date.fromisoformat(match.group(1))
            end = date.fromisoformat(match.group(2))
        except ValueError:
            diagnostics.invalid_filenames += 1
            continue
        if (
            start.weekday() != 0
            or end.weekday() != 6
            or end - start != timedelta(days=6)
        ):
            diagnostics.invalid_filenames += 1
            continue
        entry = _read_entry(path, "weekly", start, end, diagnostics)
        if entry:
            entries.append(entry)
    return entries


def _scan_diaries(
    directory: Path, diagnostics: LongTermArchiveDiagnostics
) -> list[LongTermArchiveEntry]:
    entries: list[LongTermArchiveEntry] = []
    if not directory.exists():
        return entries
    for path in sorted(directory.glob("diary_*.txt")):
        match = _DIARY_PATTERN.fullmatch(path.name)
        if not match:
            diagnostics.invalid_filenames += 1
            continue
        try:
            diary_date = date.fromisoformat(match.group(1))
        except ValueError:
            diagnostics.invalid_filenames += 1
            continue
        entry = _read_entry(path, "diary", diary_date, diary_date, diagnostics)
        if entry:
            entries.append(entry)
    return entries


def select_long_term_memories(
    history_root: str | Path = Path("chat_history/rinne_01"),
) -> LongTermMemorySelection:
    """Select all available memories without overlapping date coverage.

    Monthly memories are selected first.  A weekly memory is selected only
    when none of its seven dates overlap a selected monthly memory.  A diary
    is selected only when its date is not covered by a selected monthly or
    weekly memory.
    """

    root = Path(history_root)
    diagnostics = LongTermArchiveDiagnostics()
    monthlies = _scan_monthlies(root / "monthly", diagnostics)
    all_weeklies = _scan_weeklies(root / "weekly", diagnostics)
    all_diaries = _scan_diaries(root / "diaries", diagnostics)

    weeklies: list[LongTermArchiveEntry] = []
    for weekly in all_weeklies:
        if any(weekly.overlaps(monthly) for monthly in monthlies):
            diagnostics.skipped_overlapping_weeklies += 1
        else:
            weeklies.append(weekly)

    diaries: list[LongTermArchiveEntry] = []
    for diary in all_diaries:
        if any(
            container.covers(diary.start_date) for container in monthlies + weeklies
        ):
            diagnostics.skipped_covered_diaries += 1
        else:
            diaries.append(diary)

    return LongTermMemorySelection(
        monthly_entries=monthlies,
        weekly_entries=weeklies,
        diary_entries=diaries,
        diagnostics=diagnostics,
    )


def load_today_messages(
    history_root: str | Path,
    reference_time: datetime | None = None,
) -> list[dict[str, str]]:
    """Read today's root-level chat files using the existing 03:00 boundary.

    This intentionally mirrors the current production rule: a chat file is
    included when the timestamp in its filename is on or after today's 03:00
    boundary.  Subdirectories are never scanned.
    """

    root = Path(history_root)
    now = reference_time or datetime.now()
    today_start = now.replace(hour=3, minute=0, second=0, microsecond=0)
    if now.hour < 3:
        today_start -= timedelta(days=1)

    messages: list[dict[str, str]] = []
    if not root.exists():
        return messages

    for json_file in sorted(root.glob("*.json"), key=lambda path: path.name):
        try:
            file_time = datetime.strptime(json_file.stem[:19], "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            continue
        if file_time < today_start:
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if (
                role not in {"human", "ai"}
                or not isinstance(content, str)
                or not content
            ):
                continue
            messages.append(
                {"role": "user" if role == "human" else "assistant", "content": content}
            )
    return messages
