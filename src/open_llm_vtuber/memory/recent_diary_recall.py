"""Silent, on-demand access to the previous three approved diary originals."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable

from .cloud_memory_tool import CloudMemoryToolHandlerResult
from .diary_review import diary_approval_status
from .today_history_loader import TodayHistoryLoader


TOOL_NAME = "search_recent_diary_detail"
RECENT_DIARY_DAYS = 3
_MAX_NOTE_BYTES = 2 * 1024 * 1024
_DIARY_TITLE = re.compile(
    r"^(?:\ufeff)?(?:\d{4}[年./-]\d{1,2}[月./-]\d{1,2}日?|"
    r"\d{4}-\d{2}-\d{2})(?:\s*[（(][^\n）)]*[）)])?\s*$"
)
_ABSOLUTE_DATE = re.compile(r"(?<!\d)(\d{4})[年./-](\d{1,2})[月./-](\d{1,2})日?")
_RELATIVE_LABELS = {1: "昨天", 2: "前天", 3: "大前天"}

_TOOL_DESCRIPTION = (
    "仅用于静默核对昨天、前天或大前天的已验收日记原文。"
    "当常驻的近三日压缩版已经表明事情属于这三天之一，但没有用户所问的精确答案，"
    "或用户直接追问这三天的具体细节时调用。不要为今天发生的事、四天前及更早的旧事、"
    "或压缩版已经足够回答的内容调用。调用和阅读过程绝不能告诉用户。"
)
_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "要在近三日日记原文中核对的具体事实或细节。",
        }
    },
    "required": ["query"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class RecentDiaryOriginal:
    memory_day: date
    relative_day: int
    source_file: str
    content: str


@dataclass
class RecentDiaryRecallResult:
    memory_day: date
    entries: list[RecentDiaryOriginal] = field(default_factory=list)
    context: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "status": "completed",
            "memory_day": self.memory_day.isoformat(),
            "protected_days": RECENT_DIARY_DAYS,
            "source_files": [entry.source_file for entry in self.entries],
            "source_memory_days": [
                entry.memory_day.isoformat() for entry in self.entries
            ],
            "character_count": len(self.context),
            "warnings": list(self.warnings),
        }


def _strip_diary_title(content: str) -> str:
    lines = content.strip().splitlines()
    if lines and _DIARY_TITLE.fullmatch(lines[0].strip()):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines).strip()


def _requested_relative_days(query: str, memory_day: date) -> set[int]:
    requested: set[int] = set()
    normalized = " ".join(str(query or "").split())
    if "大前天" in normalized:
        requested.add(3)
        normalized = normalized.replace("大前天", "")
    if "前天" in normalized:
        requested.add(2)
    if "昨天" in normalized or "昨晚" in normalized:
        requested.add(1)
    for match in _ABSOLUTE_DATE.finditer(normalized):
        try:
            mentioned = date(
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )
        except ValueError:
            continue
        relative_day = (memory_day - mentioned).days
        if 1 <= relative_day <= RECENT_DIARY_DAYS:
            requested.add(relative_day)
    return requested or set(range(1, RECENT_DIARY_DAYS + 1))


def load_recent_diary_originals(
    history_root: str | Path,
    *,
    query: str = "",
    reference_time: datetime | None = None,
) -> RecentDiaryRecallResult:
    """Load only approved originals from the previous three memory days."""

    root = Path(history_root).resolve()
    memory_day, _window_start, _window_end = TodayHistoryLoader(
        root
    ).memory_day_window(reference_time)
    result = RecentDiaryRecallResult(memory_day=memory_day)
    diary_root = root / "diaries"
    if not diary_root.is_dir():
        result.warnings.append("近期日记目录不存在")
        result.context = _format_context(result)
        return result

    allowed_root = diary_root.resolve()
    for relative_day in sorted(_requested_relative_days(query, memory_day), reverse=True):
        day = memory_day - timedelta(days=relative_day)
        path = diary_root / f"diary_{day.isoformat()}.txt"
        if not path.is_file():
            result.warnings.append(f"缺少{_RELATIVE_LABELS[relative_day]}的日记")
            continue
        approval = diary_approval_status(root, day.isoformat(), path)
        if approval not in {"approved", "changed"}:
            result.warnings.append(
                f"跳过未验收的{_RELATIVE_LABELS[relative_day]}日记（{approval}）"
            )
            continue
        try:
            resolved = path.resolve(strict=True)
            if (
                path.is_symlink()
                or not resolved.is_relative_to(allowed_root)
                or resolved.stat().st_size > _MAX_NOTE_BYTES
            ):
                result.warnings.append(f"跳过不安全或过大的近期日记：{path.name}")
                continue
            content = _strip_diary_title(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeError) as exc:
            result.warnings.append(f"无法读取近期日记 {path.name}: {exc}")
            continue
        if content:
            result.entries.append(
                RecentDiaryOriginal(
                    memory_day=day,
                    relative_day=relative_day,
                    source_file=path.name,
                    content=content,
                )
            )
    result.context = _format_context(result)
    return result


def _format_context(result: RecentDiaryRecallResult) -> str:
    sections = [
        "【近三日已验收日记原文｜仅供静默核对】",
        "只把原文中明确支持的内容用于回答；不能从空缺处推断或补写。"
        "如果原文仍没有答案，就像自然记不起一样坦率表达，不要提及日记、检索、工具或后台。",
    ]
    for entry in result.entries:
        sections.append(
            f"【{_RELATIVE_LABELS[entry.relative_day]}的日记原文】\n{entry.content}"
        )
    if not result.entries:
        sections.append(
            "这三天的已验收日记原文没有提供可核对的内容。不要猜测或编造答案。"
        )
    return "\n\n".join(sections)


def apply_long_term_cutoff(
    arguments: dict[str, Any],
    *,
    memory_day: date,
) -> tuple[dict[str, Any] | None, str | None]:
    """Hard-limit Layer 3 to D-4 and older.

    ``None`` means the requested range lies entirely in the protected recent
    window and should be served by ``search_recent_diary_detail`` instead.
    """

    cutoff = memory_day - timedelta(days=RECENT_DIARY_DAYS + 1)
    normalized = dict(arguments)
    start_text = normalized.get("start_date")
    end_text = normalized.get("end_date")
    try:
        start = date.fromisoformat(start_text) if isinstance(start_text, str) else None
        end = date.fromisoformat(end_text) if isinstance(end_text, str) else None
    except ValueError:
        start = None
        end = None
    if start is not None and end is not None and start > cutoff:
        return None, "requested_range_within_recent_three_days"
    if start is None or end is None:
        start = date(1900, 1, 1)
        end = cutoff
    else:
        end = min(end, cutoff)
    normalized.update(
        {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "time_range_confidence": "high",
            "time_range_basis": (
                "后台边界：第三层只允许检索大前天之前的记忆日"
            ),
        }
    )
    return normalized, None


class RecentDiaryRecallTool:
    """A one-shot internal tool whose progress is never shown or spoken."""

    name = TOOL_NAME

    def __init__(
        self,
        handler: Callable[[dict[str, Any]], Awaitable[CloudMemoryToolHandlerResult]],
        *,
        turn_started_perf: float,
    ) -> None:
        self._handler = handler
        self._turn_started_perf = turn_started_perf
        self._called = False
        self._exhausted = False
        self._arguments: dict[str, Any] | None = None
        self._result: CloudMemoryToolHandlerResult | None = None
        self._requested_after_seconds: float | None = None
        self._elapsed_seconds = 0.0

    @property
    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": _TOOL_DESCRIPTION,
                "parameters": _PARAMETERS,
            },
        }

    @property
    def claude_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": _TOOL_DESCRIPTION,
            "input_schema": _PARAMETERS,
        }

    @property
    def prompt_instructions(self) -> str:
        return (
            f"Internal recent diary tool: {self.name}. Input: query (string). "
            "It silently reads only yesterday, the day before yesterday, and three "
            "days ago when their compressed summaries lack an exact answer. "
            "In prompt-tool mode output only: "
            '[{"mcp_server":"internal_memory","tool":'
            '"search_recent_diary_detail","arguments":'
            '"{\\"query\\":\\"...\\"}"}]'
        )

    @property
    def decision_instructions(self) -> str:
        return (
            "【近三日日记原文核对规则】\n"
            "常驻上下文只提供昨天、前天、大前天的压缩版。如果用户问的事情属于这三天，"
            "而压缩版没有直接给出所需的具体答案，必须先静默调用"
            f"{self.name}核对原文；不要改用长期记忆。若压缩版已经足够，直接回答，不调用。"
            "今天的事只看今日对话；四天前及更早才使用长期记忆。绝不能向用户说"
            "‘我仔细想想’或提及这次核对。原文仍无证据时，自然承认记不起，禁止编造。"
        )

    @property
    def followup_instructions(self) -> str:
        return (
            "近三日日记原文核对已经结束。只依据明确证据自然回答；若没有答案就自然说"
            "记不起。不要再次调用，也不要提及日记、检索、工具或内部过程。"
        )

    @property
    def called(self) -> bool:
        return self._called

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    async def execute(
        self, arguments: dict[str, Any] | None
    ) -> CloudMemoryToolHandlerResult:
        if self._exhausted and self._result is not None:
            return self._result
        self._called = True
        self._exhausted = True
        self._requested_after_seconds = round(
            time.perf_counter() - self._turn_started_perf, 3
        )
        query = str((arguments or {}).get("query") or "").strip()
        self._arguments = {"query": " ".join(query.split())}
        if not query:
            self._result = CloudMemoryToolHandlerResult(
                content="近期日记原文核对缺少明确目标。请不要猜测。",
                diagnostics={"status": "failed", "failure": "empty_query"},
                is_error=True,
            )
            return self._result
        started = time.perf_counter()
        try:
            self._result = await self._handler(self._arguments)
        except Exception as exc:  # pragma: no cover - final runtime shield
            self._result = CloudMemoryToolHandlerResult(
                content="近期日记原文本轮无法读取。请不要猜测。",
                diagnostics={
                    "status": "failed",
                    "failure": "handler_exception",
                    "error_type": type(exc).__name__,
                },
                is_error=True,
            )
        self._elapsed_seconds = round(time.perf_counter() - started, 3)
        return self._result

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": "completed" if self._called else "not_requested",
            "called": self._called,
            "arguments": self._arguments,
            "requested_after_seconds": self._requested_after_seconds,
            "elapsed_seconds": self._elapsed_seconds,
            "is_error": self._result.is_error if self._result else False,
            "diagnostics": dict(self._result.diagnostics) if self._result else {},
        }


__all__ = [
    "RECENT_DIARY_DAYS",
    "RecentDiaryRecallTool",
    "TOOL_NAME",
    "apply_long_term_cutoff",
    "load_recent_diary_originals",
]
