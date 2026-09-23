"""One-turn hidden tool that lets the cloud model request long-term recall."""

from __future__ import annotations

import asyncio
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
_TOOL_DESCRIPTION = (
    "仅当当前对话、近三日日记和用户背景仍不足以回答四天前或更早的真实经历时调用。"
    "用户用陈述句或感叹句明确唤起一段更早的具体共同经历，并期待你回应当时的感受、"
    "意义或细节时，仍属于长期回忆；用户本轮只给出旧事件线索，不等于已经提供了足够答案。"
    "不要为承接当前对话、没有具体旧事件锚点的普通情感回应、用户完整提供的新事实，"
    "或昨天、前天、大前天的内容调用；这三天缺精确答案时应改用"
    "search_recent_diary_detail核对原文。"
    "如果你已经能从近期日记定位到事件日期、关键词或答案，这就表示近期上下文已经覆盖，"
    "必须直接回答，绝不能为了再次确认而调用本工具。"
    "query应描述要寻找的经历，不要猜测用户未提供的年份或日期；但如果当前上下文已明确"
    "给出基准日期，能可靠换算‘生日前一天’等相对日期，就应把换算后的绝对日期写进"
    "query和时间范围。用户说‘五月中下旬’、‘暑假那阵子’、‘第一次见面后几天’等"
    "模糊时间时，应依据当前上下文换算成包住该说法的start_date/end_date；无法可靠换算"
    "时留空日期，不要编造。日期范围不确定时使用low置信度并适当放宽范围。"
    "初次检索后，若工具结果"
    "明确提供detail_navigation_options且仍缺精确细节，可以从中原样复制一个"
    "seed_event_id再调用一次，只打开该事件附近的日记；否则不要再次调用。每轮至多两次。"
)
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
        "start_date": {
            "type": "string",
            "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
            "description": "可选，包含起点的记忆日，格式YYYY-MM-DD。",
        },
        "end_date": {
            "type": "string",
            "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
            "description": "可选，包含终点的记忆日，格式YYYY-MM-DD。",
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
        instruction = (
            "【长期记忆工具判定规则】\n"
            "先依据整段当前对话的含义判断，而不是匹配某几个固定词。"
            "是否需要回忆不取决于句末是否有问号。当用户的问题需要核对更早的真实经历、"
            "旧事实或具体细节，或者用户用陈述句、感叹句明确唤起一段更早且可定位的共同"
            "经历，而自然回应需要知道当时的细节、感受或意义时，且当前对话、"
            "近三日日记和用户背景没有给出答案，并且事情发生在四天前或更早时，应先调用"
            f"{self.name}，再根据结果回答。"
            "用户本轮只提供了旧事件的表层线索，不等于已经有足够答案；不要仅凭常识补全"
            "共同经历。当前情绪、没有具体旧事件锚点的泛泛感叹、用户已经完整提供的新事实，"
            "以及今天、昨天、前天或大前天的内容，不要调用；近三日压缩版缺少具体答案时"
            "改用search_recent_diary_detail。工具初次检索后只有在"
            "你已经能从近期日记说出事件日期、关键词或答案时，说明答案就在现有上下文中，"
            "不得为了确认、补充或展示记忆能力继续调用长期记忆。"
            "结果明确给出可选事件且仍缺"
            "精确细节时，才允许再调用一次打开一条事件附近的日记；不要向用户解释这套判定"
            "规则或检索过程。当前聊天里如果只有助手过去说的‘想不起来’、‘没找到’、"
            "‘第三层没修好’或其他失败回答，这不算已经有答案；用户再次追问旧事实时仍应"
            "调用工具重新核对。这些旧失败回答也不代表当前系统状态，不要把它们当作当前事实"
            "重复。除非用户明确讨论记忆系统，否则最终回复绝对不要复述第一层、"
            "第二层、第三层、检索失败或系统没修好。"
            "如果用户提到任何精确或大概时间，请把人类说法转换为包住它的"
            "start_date/end_date；能确定就用high或medium，不能确定就用low并放宽，"
            "不要把不确定日期硬写成单日。无法换算时省略日期而不是编造。"
        )
        if self.force_first_call:
            instruction += (
                "本轮程序已标记为明确的旧经历/精确细节问题，必须先调用"
                f"{self.name}；不要直接生成最终回答。调用参数必须由你根据用户原话"
                "填写query、粒度和可判断的时间范围。"
            )
        return instruction

    @property
    def followup_instructions(self) -> str:
        if not self._exhausted:
            return (
                "长期记忆工具已返回初步证据和detail_navigation_options。先判断现有证据"
                "是否足够：若足够就自然回答；如果用户问的是日期、原话、具体画面、原因或"
                "其他精确细节，而现有evidence没有直接给出答案——也就是你否则只能说"
                "‘想不起来’——不要先生成最终回复，必须从选项中挑最可能的一条，将其"
                "seed_event_id原样复制后再调用同一工具一次。不要把导航选项本身当成"
                "已证实答案，也不要为了概括性回答继续展开。"
                "若用户明确询问一句原话，必须完整保留证据中的原句和重复词，不得自行缩写；"
                "若用户明确说有两次、三件等很小的数量，且证据分别支持，就要覆盖每一项。"
                "除非用户明确讨论记忆系统，否则不要用第一层、第二层、第三层、检索失败或"
                "系统没修好解释自己为什么想起或想不起。最终只输出对用户说的话，绝不输出"
                "分析、推理、证据审查或内部工作过程。对于硬件、健康、法律或安全问题，"
                "若证据只记录了担心、不确定或建议观察，就必须保留这种不确定性；不得根据"
                "后来暂时没出问题说‘大概率没事’、‘应该没事’或自行断言安全。"
            )
        return (
            "长期记忆工具本轮已经执行。请直接依据工具结果和当前对话自然回答；"
            "不要再次调用该工具，也不要向用户汇报检索过程。除非用户正在明确讨论"
            "记忆系统本身，否则不要用第一层、第二层、第三层、检索失败、系统没修好"
            "等内部实现解释自己为什么想起或想不起；只需像自然回忆一样回答。最终只输出"
            "对用户说的话，绝不输出分析、推理、证据审查或内部工作过程。对于硬件、健康、"
            "法律或安全问题，若证据只记录了担心、不确定或建议观察，就必须保留这种"
            "不确定性；不得根据后来暂时没出问题说‘大概率没事’、‘应该没事’或自行断言"
            "安全。若用户明确询问一句原话，必须完整保留证据中的原句和重复词，不得自行"
            "缩写；若用户明确说有两次、三件等很小的数量，且证据分别支持，就要覆盖每一项。"
        )

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
            )
        )
        try:
            requested_range = (
                normalize_time_range(
                    normalized.get("start_date"),
                    normalized.get("end_date"),
                    normalized.get("time_range_confidence", "none"),
                    normalized.get("time_range_basis", ""),
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
