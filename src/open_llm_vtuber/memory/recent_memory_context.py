"""Stable hidden context for reviewed compactions of recent memory days."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from .diary_review import diary_approval_status
from .recent_diary_compaction import (
    load_matching_compaction,
    publication_paths,
)
from .today_history_loader import TodayHistoryLoader


_DIARY_FILE = re.compile(r"^diary_(\d{4}-\d{2}-\d{2})\.txt$")
_MAX_NOTE_BYTES = 2 * 1024 * 1024
_RELATIVE_LABELS = {1: "昨天", 2: "前天", 3: "大前天"}


RECENT_MEMORY_USAGE_RULES = (
    "以下内容是昨天、前天和大前天的已验收日记压缩版，只供你在内部理解当前对话。"
    "标题中的相对日期均以当前03:00分界的记忆日为基准。今天只能依据当前对话"
    "与今天的原始聊天；本上下文不包含今天的日记，绝不能把这三天的事情说成今天发生。"
    "如果用户询问这三天内的具体细节，而压缩版没有直接给出答案，先静默调用"
    "search_recent_diary_detail核对对应日记原文，不要调用长期记忆。"
    "除非用户明确要求时间线、总结或完整回复，否则不要复述检索结果中包含的日期信息。"
    "如果用户直接询问具体日期或时间，可以直接回答该问题。"
    "不要提及隐藏上下文、文件、检索、排名或系统处理过程。"
    "除非用户明确询问记忆系统或项目实现，否则不要主动提及记忆层级、提示词、"
    "工具、模型、API或后台等幕后实现。近期记忆里的这些表述只用于理解当时处境，"
    "不能自动当作当前系统状态复述。"
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
    # Kept as an explicit empty field for diagnostics/API compatibility.  The
    # always-on recent context intentionally never loads weekly summaries.
    weekly_entries: list[RecentMemoryEntry] = field(default_factory=list)
    covered_diary_files: list[str] = field(default_factory=list)
    compacted_diary_files: list[str] = field(default_factory=list)
    full_diary_files: list[str] = field(default_factory=list)
    missing_diary_dates: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_diary_character_count: int = 0
    delivered_diary_character_count: int = 0

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
    compacted_diary_files: list[str],
    full_diary_files: list[str],
) -> tuple[list[RecentMemoryEntry], int]:
    directory = history_root / "diaries"
    if not directory.is_dir():
        return [], 0
    entries: list[RecentMemoryEntry] = []
    source_character_count = 0
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
        approval_status = diary_approval_status(
            history_root,
            memory_date.isoformat(),
            path,
        )
        if approval_status not in {"approved", "changed"}:
            warnings.append(
                f"跳过未验收日记 {path.name}（状态：{approval_status}）"
            )
            continue
        if approval_status == "changed":
            warnings.append(
                f"日记 {path.name} 在验收后发生修改，按当前内容读取；请谨慎复核"
            )
        content = _read_note(path, directory, warnings)
        if content:
            source_character_count += len(content)
            compaction = load_matching_compaction(
                history_root,
                path,
                memory_date.isoformat(),
            )
            source_kind = "diary"
            if compaction is not None:
                _title, separator, compact_body = compaction.summary.partition("\n")
                content = compact_body if separator else compaction.summary
                source_kind = "recent_diary_context"
                compacted_diary_files.append(path.name)
            else:
                summary_path, manifest_path = publication_paths(
                    history_root, memory_date.isoformat()
                )
                if summary_path.exists() or manifest_path.exists():
                    warnings.append(
                        f"近期压缩版与当前日记不匹配，跳过常驻注入：{path.name}"
                    )
                else:
                    warnings.append(
                        f"近期日记没有已发布压缩版，跳过常驻注入：{path.name}"
                    )
                continue
            entries.append(
                RecentMemoryEntry(
                    source_kind=source_kind,
                    source_file=path.name,
                    period_start=memory_date,
                    period_end=memory_date,
                    content=content,
                )
            )
    return entries, source_character_count


def _format_context(result: RecentMemoryContextResult) -> str:
    sections = [
        "【近三日已验收日记压缩版】",
        RECENT_MEMORY_USAGE_RULES,
        f"当前记忆日：{result.memory_day.isoformat()}（本地03:00分界）。",
    ]
    for entry in result.diary_entries:
        relative_day = (result.memory_day - entry.period_start).days
        label = _RELATIVE_LABELS.get(relative_day, f"{relative_day}天前")
        sections.append(
            f"【{label}的日记压缩版】\n{entry.content}"
        )
    return "\n\n".join(sections)


def load_recent_memory_context(
    history_root: str | Path,
    *,
    days: int = 3,
    reference_time: datetime | None = None,
) -> RecentMemoryContextResult:
    """Load hash-bound diary compactions for prior complete memory days.

    The current memory day is deliberately excluded.  Weekly and monthly
    summaries remain available to the on-demand long-term retrieval service.
    A missing or stale compaction is skipped rather than silently replaced by
    a full diary; the original is available only through the silent recent
    detail tool.
    """

    if not 1 <= days <= 31:
        raise ValueError("days must be between 1 and 31")
    root = Path(history_root).resolve()
    memory_day, _window_start, _window_end = TodayHistoryLoader(
        root
    ).memory_day_window(reference_time)
    start = memory_day - timedelta(days=days)
    end = memory_day - timedelta(days=1)
    result = RecentMemoryContextResult(
        memory_day=memory_day,
        window_start=start,
        window_end=end,
    )
    (
        result.diary_entries,
        result.source_diary_character_count,
    ) = _scan_diaries(
        root,
        start,
        end,
        result.warnings,
        result.compacted_diary_files,
        result.full_diary_files,
    )
    result.delivered_diary_character_count = sum(
        len(entry.content) for entry in result.diary_entries
    )
    loaded_dates = {entry.period_start for entry in result.diary_entries}
    result.missing_diary_dates = [
        (start + timedelta(days=offset)).isoformat()
        for offset in range(days)
        if start + timedelta(days=offset) not in loaded_dates
    ]
    result.context = _format_context(result)
    return result


__all__ = [
    "RECENT_MEMORY_USAGE_RULES",
    "RecentMemoryContextResult",
    "RecentMemoryEntry",
    "load_recent_memory_context",
]
