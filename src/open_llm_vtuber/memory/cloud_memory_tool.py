"""One-turn hidden tool that lets the cloud model request long-term recall."""

from __future__ import annotations

import asyncio
import copy
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

from .time_range import ValidatedTimeRange, normalize_time_range


TOOL_NAME = "search_long_term_memory"
_MEMORY_SOURCE = r"(?:长期记忆|记忆库|日记|聊天记录|历史记录|记忆)"
_LOOKUP_VERB = r"(?:查|查询|搜索|检索|翻查|翻看|调取|读取|找|找找|看看)"
_EXPLICIT_MEMORY_LOOKUP = re.compile(
    rf"(?:请|麻烦|帮我|你)?{_LOOKUP_VERB}.{{0,12}}{_MEMORY_SOURCE}|"
    rf"{_MEMORY_SOURCE}.{{0,12}}{_LOOKUP_VERB}"
)
_EXPLICIT_MEMORY_OPT_OUT = re.compile(
    rf"(?:不用|不要|别|无需|不必).{{0,10}}(?:{_LOOKUP_VERB}.{{0,6}})?{_MEMORY_SOURCE}|"
    rf"{_MEMORY_SOURCE}.{{0,10}}(?:不用|不要|别查|无需|不必)"
)
_EXPLICIT_RECALL_REQUEST = re.compile(
    r"(?:你|凛祢)(?:再)?(?:仔细|好好)?想想|(?:再)?(?:仔细|好好)想想|"
    r"(?:帮我)?回忆(?:一下|下|看看)?|"
    r"还记得|记不记得"
)
_CONCRETE_RECALL_TARGET = re.compile(
    r"第一次|第[一二三四五六七八九十百\d]+次|那次|那天|那件|那段|"
    r"某次|某天|哪天|哪一年|什么时候|多久以前|几年前|去年|上个月|"
    r"原话|具体(?:日期|时间|地点|名字|内容|原因|细节)|"
    r"(?:说|提|告诉|答应|发生|做|见面|去|写|聊|经历)过(?:什么|哪|的)|"
    r"20\d{2}[年./-]\d{1,2}|[一二三四五六七八九十\d]{1,2}月|"
    r"上旬|中旬|下旬|年初|年中|年底|暑假|寒假|生日那周"
)
_TOOL_DESCRIPTION = '回想四天前或更早、你和用户真实经历过的事。用户提起一段具体的旧经历——不管是问句，还是用陈述句、感叹句唤起回忆——而当前对话、近三日日记和背景里没有你需要的细节时，先调用本工具再回答；只有零碎线索时也要查，不要按常理补全。眼前已经有答案时直接回答，不要为了确认再查。今天的事看当前对话；昨天、前天、大前天的细节用search_recent_diary_detail。只是聊此刻的心情，或用户已经把新的事情说完整时，不用查。你以前说过“想不起来”，不代表现在查不到，用户再问就再查。query写要找的经历，不要猜用户没说的年份或日期；用户给了时间说法（如“五月中下旬”“生日前一天”），能可靠换算就写成包住它的start_date/end_date，拿不准就用low并放宽，换算不了就不填。第一次的结果里有detail_navigation_options、但仍缺精确细节时，可以原样复制其中一个seed_event_id再调用一次；每轮最多两次。区分事情发生的时间和告诉你、写进记忆的时间：一般用time_range_type=occurrence（事情发生的时间）；只有问“某天告诉过你什么”时才用recorded。'
_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "需要从更早长期记忆中寻找的经历或事实。",
        },
        "question_granularity": {
            "type": "string",
            "enum": ["specific_event", "exact_detail", "overview"],
            "description": "具体事件、精确细节或跨多日概括。",
        },
        "seed_event_id": {
            "type": "string",
            "description": (
                "仅用于第二次细节回源；必须从初次工具结果的"
                "detail_navigation_options中原样复制，不得自行编造。"
            ),
        },
        "time_range_type": {
            "type": "string",
            "enum": ["occurrence", "recorded"],
            "description": "occurrence表示事情发生时间（默认）；recorded仅用于用户明确给出的讲述或记录日期。两种日期不得互换。",
        },
        "start_date": {
            "type": "string",
            "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
            "description": "可选，包含起点的日期；含义由time_range_type决定，格式YYYY-MM-DD。",
        },
        "end_date": {
            "type": "string",
            "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
            "description": "可选，包含终点的日期；含义由time_range_type决定，格式YYYY-MM-DD。",
        },
        "time_range_confidence": {
            "type": "string",
            "enum": ["high", "medium", "low", "none"],
            "description": (
                "时间范围的可靠程度：high/medium为可硬过滤，low为软提示，"
                "none为没有可靠范围。"
            ),
        },
        "time_range_basis": {
            "type": "string",
            "description": "可选，简短说明范围来自用户哪种时间说法。",
        },
    },
    "required": ["query", "question_granularity"],
    "additionalProperties": False,
}


class LongTermRecallPolicy(str, Enum):
    """How strongly the current user turn requests long-term recall."""

    AUTO = "auto"
    FORCE = "force"
    OFF = "off"


def classify_long_term_recall_policy(text: str) -> LongTermRecallPolicy:
    """Classify explicit memory control without guessing from past-tense words."""

    if not isinstance(text, str):
        return LongTermRecallPolicy.AUTO
    normalized = " ".join(text.split())
    if not normalized:
        return LongTermRecallPolicy.AUTO
    if _EXPLICIT_MEMORY_OPT_OUT.search(normalized):
        return LongTermRecallPolicy.OFF
    concrete_target = _CONCRETE_RECALL_TARGET.search(normalized)
    if concrete_target and (
        _EXPLICIT_MEMORY_LOOKUP.search(normalized)
        or _EXPLICIT_RECALL_REQUEST.search(normalized)
    ):
        return LongTermRecallPolicy.FORCE
    return LongTermRecallPolicy.AUTO


def should_force_long_term_recall(text: str) -> bool:
    """Return whether the user explicitly requested one concrete recollection.

    This compatibility helper deliberately does not force on generic wording
    such as ``以前`` or ``当时``. Those turns remain a semantic model decision.
    """

    return classify_long_term_recall_policy(text) is LongTermRecallPolicy.FORCE


@dataclass(frozen=True)
class CloudMemoryToolHandlerResult:
    content: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False


class CloudLongTermMemoryTool:
    """Per-turn callable state shared by the agent loop and trial logger."""

    name = TOOL_NAME

    def __init__(
        self,
        handler: Callable[[dict[str, Any]], Awaitable[CloudMemoryToolHandlerResult]],
        *,
        turn_started_perf: float,
        recall_policy: LongTermRecallPolicy | str | None = None,
        force_first_call: bool = False,
        max_retrieval_seconds: float = 30.0,
    ) -> None:
        self._handler = handler
        self._turn_started_perf = turn_started_perf
        self._called = False
        self._arguments: dict[str, Any] | None = None
        self._result: CloudMemoryToolHandlerResult | None = None
        self._requested_after_seconds: float | None = None
        self._elapsed_seconds = 0.0
        self._call_records: list[dict[str, Any]] = []
        self._exhausted = False
        self._allowed_detail_seed_ids: set[str] = set()
        if recall_policy is None:
            recall_policy = (
                LongTermRecallPolicy.FORCE
                if force_first_call
                else LongTermRecallPolicy.AUTO
            )
        self._recall_policy = LongTermRecallPolicy(recall_policy)
        self._force_first_call = self._recall_policy is LongTermRecallPolicy.FORCE
        self._max_retrieval_seconds = max(0.001, float(max_retrieval_seconds))
        self._retrieval_started_perf: float | None = None
        self._time_range = normalize_time_range()

    @property
    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": _TOOL_DESCRIPTION,
                "parameters": copy.deepcopy(_PARAMETERS),
            },
        }

    @property
    def claude_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": _TOOL_DESCRIPTION,
            "input_schema": copy.deepcopy(_PARAMETERS),
        }

    @property
    def prompt_instructions(self) -> str:
        return (
            "Internal memory tool:\n"
            f"Tool: {self.name}\n"
            f"Description: {_TOOL_DESCRIPTION}\n"
            "Parameters: query (string), question_granularity "
            "(specific_event|exact_detail|overview), seed_event_id "
            "(optional, second detail lookup only), start_date/end_date "
            "(optional inclusive YYYY-MM-DD range), time_range_confidence "
            "(high|medium|low|none), time_range_basis (optional).\n"
            "When this tool is required in prompt-tool mode, output only a JSON array "
            "using this structure: "
            '[{"mcp_server":"internal_memory","tool":"search_long_term_memory",'
            '"arguments":"{\\"query\\":\\"...\\",'
            '\\"question_granularity\\":\\"specific_event\\",'
            '\\"start_date\\":\\"2026-05-01\\",'
            '\\"end_date\\":\\"2026-05-31\\",'
            '\\"time_range_confidence\\":\\"low\\"}"}]'
        )

    @property
    def decision_instructions(self) -> str:
        if self.force_first_call:
            return '本轮是明确的旧经历或精确细节问题：先调用search_long_term_memory，不要直接回答。query、粒度和时间范围按用户的原话填写。'.replace("search_long_term_memory", self.name)
        return ""

    @property
    def followup_instructions(self) -> str:
        if not self._exhausted:
            return '长期记忆的初步结果已返回。证据够就直接自然地回答。如果用户问的是日期、原话、具体画面、原因等精确细节，而证据里没有直接答案（否则你只能说“想不起来”），先不要回答：从detail_navigation_options里挑最可能的一条，原样复制它的seed_event_id再调用一次。导航选项本身不算答案，概括性的问题不用再展开。用户问原话时，完整保留证据里的原句和重复的字词；用户说有两次、三件这样的少数几项时，证据支持的每一项都要说到。健康、硬件、法律或安全方面的事，证据里只是担心或不确定，就保留这份不确定，不要说“应该没事”。'
        return '长期记忆本轮已经查完，不要再调用。直接依据结果和当前对话自然回答。用户问原话时，完整保留证据里的原句和重复的字词；用户说有两次、三件这样的少数几项时，证据支持的每一项都要说到。健康、硬件、法律或安全方面的事，证据里只是担心或不确定，就保留这份不确定，不要说“应该没事”。'

    @property
    def called(self) -> bool:
        return self._called

    @property
    def force_first_call(self) -> bool:
        """Whether the first model request should select this tool."""

        return self._force_first_call and not self._called

    @property
    def time_range(self) -> ValidatedTimeRange:
        return self._time_range

    @staticmethod
    def _range_arguments(time_range: ValidatedTimeRange) -> dict[str, Any]:
        if not time_range.is_bounded:
            return {}
        values: dict[str, Any] = {
            "start_date": time_range.start_date,
            "end_date": time_range.end_date,
            "time_range_confidence": time_range.confidence,
            "time_range_type": time_range.time_range_type,
        }
        if time_range.basis:
            values["time_range_basis"] = time_range.basis
        return values

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    def reject_due_to_recent_context(
        self,
        arguments: dict[str, Any] | None,
        *,
        reason: str,
    ) -> CloudMemoryToolHandlerResult:
        """Record a model-requested lookup that was safely skipped.

        This is not a failed retrieval: the requested period or subject is
        already present in the always-on recent context, so calling Layer 3
        would only add latency and competing evidence.
        """

        self._called = True
        self._force_first_call = False
        self._exhausted = True
        self._requested_after_seconds = round(
            time.perf_counter() - self._turn_started_perf,
            3,
        )
        self._elapsed_seconds = 0.0
        self._arguments = dict(arguments or {})
        self._result = CloudMemoryToolHandlerResult(
            content=(
                "第三层检索未执行：当前已提供的近期记忆已经覆盖这项问题。"
                "请直接依据现有上下文回答，不要再次调用长期记忆工具，也不要向用户"
                "提及日记、上下文、检索、工具或这次拦截。只输出自然的最终回答。"
            ),
            diagnostics={
                "status": "skipped_recent_context",
                "retrieval_skipped": True,
                "reason": reason,
            },
            is_error=False,
        )
        self._call_records.append(
            {
                "arguments": dict(self._arguments),
                "elapsed_seconds": 0.0,
                "is_error": False,
                "tool_result_sent_to_cloud": self._result.content,
                "diagnostics": dict(self._result.diagnostics),
            }
        )
        return self._result

    @property
    def call_count(self) -> int:
        return len(self._call_records)

    @property
    def active_claude_schema(self) -> dict[str, Any]:
        schema = self.claude_schema
        if self._called and not self._exhausted:
            schema["description"] = (
                "初步证据不足以回答精确细节。必须从刚才返回的"
                "detail_navigation_options中选择最可能的一项，并原样复制其"
                "seed_event_id以打开唯一一个相邻日记窗口。"
            )
            schema["input_schema"]["required"] = [
                "query",
                "question_granularity",
                "seed_event_id",
            ]
        return schema

    @property
    def active_openai_schema(self) -> dict[str, Any]:
        schema = self.openai_schema
        if self._called and not self._exhausted:
            schema["function"]["description"] = (
                "初步证据不足以回答精确细节。必须从刚才返回的"
                "detail_navigation_options中选择最可能的一项，并原样复制其"
                "seed_event_id以打开唯一一个相邻日记窗口。"
            )
            schema["function"]["parameters"]["required"] = [
                "query",
                "question_granularity",
                "seed_event_id",
            ]
        return schema

    async def execute(
        self,
        arguments: dict[str, Any] | None,
    ) -> CloudMemoryToolHandlerResult:
        if self._exhausted and self._result is not None:
            return self._result
        was_called = self._called
        self._called = True
        self._force_first_call = False
        self._requested_after_seconds = round(
            time.perf_counter() - self._turn_started_perf,
            3,
        )
        normalized = dict(arguments or {})
        query = normalized.get("query")
        granularity = normalized.get("question_granularity")
        seed_event_id = normalized.get("seed_event_id")
        range_fields_present = any(
            key in normalized
            for key in (
                "start_date",
                "end_date",
                "time_range_confidence",
                "time_range_basis",
                "time_range_type",
            )
        )
        try:
            requested_range = (
                normalize_time_range(
                    normalized.get("start_date"),
                    normalized.get("end_date"),
                    normalized.get("time_range_confidence", "none"),
                    normalized.get("time_range_basis", ""),
                    normalized.get(
                        "time_range_type",
                        self._time_range.time_range_type if was_called else "occurrence",
                    ),
                )
                if range_fields_present
                else self._time_range
            )
        except ValueError as exc:
            self._exhausted = True
            self._result = CloudMemoryToolHandlerResult(
                content=(
                    "长期记忆工具收到的时间范围无效。请不要猜测日期；"
                    "如无法确定时间，请省略时间范围后再回答。"
                ),
                diagnostics={"failure": "invalid_time_range", "error": str(exc)},
                is_error=True,
            )
            return self._result
        if was_called and range_fields_present:
            if (
                requested_range.start_date != self._time_range.start_date
                or requested_range.end_date != self._time_range.end_date
                or requested_range.time_range_type != self._time_range.time_range_type
            ):
                self._exhausted = True
                self._result = CloudMemoryToolHandlerResult(
                    content=(
                        "第二次长期记忆调用不能改变初次检索的时间范围。"
                        "请只复制导航选项中的seed_event_id，不要猜测新的日期。"
                    ),
                    diagnostics={"failure": "detail_range_changed"},
                    is_error=True,
                )
                return self._result
            # The second call is a detail navigation, not a new date
            # interpretation.  Preserve the first call's confidence and basis
            # even if the model only copied the dates or omitted them entirely.
            requested_range = self._time_range
        self._time_range = requested_range
        if was_called and not isinstance(seed_event_id, str):
            self._exhausted = True
            self._result = CloudMemoryToolHandlerResult(
                content=(
                    "第二次长期记忆调用缺少初次结果中的seed_event_id。"
                    "请不要继续检索或猜测旧经历。"
                ),
                diagnostics={"failure": "missing_detail_seed"},
                is_error=True,
            )
            return self._result
        if (
            was_called
            and isinstance(seed_event_id, str)
            and self._allowed_detail_seed_ids
            and seed_event_id not in self._allowed_detail_seed_ids
        ):
            self._exhausted = True
            self._arguments = {
                "query": " ".join(str(query or "").split()),
                "question_granularity": str(granularity or "auto"),
                "seed_event_id": seed_event_id,
            }
            self._arguments.update(self._range_arguments(self._time_range))
            self._result = CloudMemoryToolHandlerResult(
                content=(
                    "第二次长期记忆调用选择了导航白名单之外的事件。"
                    "该线索不能作为事实，请不要继续检索或猜测旧经历。"
                ),
                diagnostics={"failure": "detail_seed_not_allowed"},
                is_error=True,
            )
            self._elapsed_seconds = 0.0
            self._call_records.append(
                {
                    "arguments": dict(self._arguments),
                    "elapsed_seconds": 0.0,
                    "is_error": True,
                    "tool_result_sent_to_cloud": self._result.content,
                    "diagnostics": dict(self._result.diagnostics),
                }
            )
            return self._result
        if not isinstance(query, str) or not query.strip():
            self._result = CloudMemoryToolHandlerResult(
                content="长期记忆工具未收到有效检索目标。请不要猜测旧经历。",
                diagnostics={"failure": "empty_query"},
                is_error=True,
            )
            return self._result
        if granularity not in {"specific_event", "exact_detail", "overview"}:
            granularity = "auto"
        self._arguments = {
            "query": " ".join(query.split()),
            "question_granularity": granularity,
        }
        self._arguments.update(self._range_arguments(self._time_range))
        if isinstance(seed_event_id, str) and seed_event_id.strip():
            self._arguments["seed_event_id"] = seed_event_id.strip()
        started = time.perf_counter()
        if self._retrieval_started_perf is None:
            self._retrieval_started_perf = started
        try:
            elapsed_budget = started - self._retrieval_started_perf
            remaining_seconds = self._max_retrieval_seconds - elapsed_budget
            if remaining_seconds <= 0:
                raise asyncio.TimeoutError
            self._result = await asyncio.wait_for(
                self._handler(self._arguments),
                timeout=remaining_seconds,
            )
        except asyncio.TimeoutError:
            self._exhausted = True
            self._result = CloudMemoryToolHandlerResult(
                content=(
                    "长期记忆检索超过本轮等待上限，未取得可靠证据。"
                    "请基于当前对话、近期记忆和用户背景自然回应，不要猜测旧经历，"
                    "也不要再次调用长期记忆工具。"
                ),
                diagnostics={
                    "failure": "retrieval_timeout",
                    "timeout_seconds": self._max_retrieval_seconds,
                },
                is_error=True,
            )
        except Exception as exc:  # pragma: no cover - final runtime shield
            self._result = CloudMemoryToolHandlerResult(
                content="长期记忆工具本轮不可用。请不要根据无关内容猜测用户经历。",
                diagnostics={
                    "failure": "handler_exception",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                is_error=True,
            )
        finally:
            self._elapsed_seconds = round(time.perf_counter() - started, 3)
        detail_available = bool(
            self._result
            and not self._result.is_error
            and self._result.diagnostics.get("detail_followup_available")
        )
        if not was_called and self._result is not None:
            self._allowed_detail_seed_ids = {
                str(item.get("candidate_id"))
                for item in self._result.diagnostics.get(
                    "detail_navigation_options",
                    [],
                )
                if isinstance(item, dict) and item.get("candidate_id")
            }
        self._exhausted = bool(seed_event_id) or not detail_available
        self._call_records.append(
            {
                "arguments": dict(self._arguments),
                "elapsed_seconds": self._elapsed_seconds,
                "is_error": self._result.is_error if self._result else True,
                "tool_result_sent_to_cloud": (
                    self._result.content if self._result else None
                ),
                "diagnostics": (dict(self._result.diagnostics) if self._result else {}),
            }
        )
        return self._result

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": "completed" if self._called else "not_requested",
            "called": self._called,
            "recall_policy": self._recall_policy.value,
            "call_count": len(self._call_records),
            "exhausted": self._exhausted,
            "arguments": self._arguments,
            "time_range": self._time_range.to_dict(),
            "requested_after_seconds": self._requested_after_seconds,
            "elapsed_seconds": self._elapsed_seconds,
            "is_error": self._result.is_error if self._result else False,
            "tool_result_sent_to_cloud": (
                self._result.content if self._result else None
            ),
            "diagnostics": (dict(self._result.diagnostics) if self._result else {}),
            "calls": [dict(item) for item in self._call_records],
        }


__all__ = [
    "CloudLongTermMemoryTool",
    "CloudMemoryToolHandlerResult",
    "LongTermRecallPolicy",
    "TOOL_NAME",
    "classify_long_term_recall_policy",
    "should_force_long_term_recall",
]
