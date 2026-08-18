"""One-turn hidden tool that lets the cloud model request long-term recall."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


TOOL_NAME = "search_long_term_memory"
_TOOL_DESCRIPTION = (
    "仅当当前对话、最近14天记忆和用户背景仍不足以回答用户关于更早真实经历的内容时调用。"
    "不要为承接当前对话、普通情感回应、用户已经提供的事实，或可由今天与近期记忆回答的"
    "昨天、前天、昨晚、上周等问题调用。每轮最多调用一次。"
    "query应描述要寻找的经历，不要猜测用户未提供的年份或日期。"
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
    },
    "required": ["query", "question_granularity"],
    "additionalProperties": False,
}


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
        handler: Callable[
            [dict[str, Any]], Awaitable[CloudMemoryToolHandlerResult]
        ],
        *,
        turn_started_perf: float,
    ) -> None:
        self._handler = handler
        self._turn_started_perf = turn_started_perf
        self._called = False
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
            "Internal memory tool:\n"
            f"Tool: {self.name}\n"
            f"Description: {_TOOL_DESCRIPTION}\n"
            "Parameters: query (string), question_granularity "
            "(specific_event|exact_detail|overview).\n"
            "When this tool is required in prompt-tool mode, output only a JSON array "
            "using this structure: "
            '[{"mcp_server":"internal_memory","tool":"search_long_term_memory",'
            '"arguments":"{\\"query\\":\\"...\\",'
            '\\"question_granularity\\":\\"specific_event\\"}"}]'
        )

    @property
    def decision_instructions(self) -> str:
        return (
            "【长期记忆工具判定规则】\n"
            "先依据整段当前对话的含义判断，而不是匹配某几个固定词。"
            "当用户的问题需要核对更早的真实经历、旧事实或具体细节，且当前对话、"
            "最近14天记忆和用户背景没有给出答案时，应先调用"
            f"{self.name}，再根据结果回答。"
            "普通聊天、情感回应、用户刚刚提供的事实，以及可由今天或近期记忆回答的"
            "相对时间内容，不要调用。工具每轮最多调用一次；不要向用户解释这套判定"
            "规则或检索过程。"
        )

    @property
    def followup_instructions(self) -> str:
        return (
            "长期记忆工具本轮已经执行。请直接依据工具结果和当前对话自然回答；"
            "不要再次调用该工具，也不要向用户汇报检索过程。"
        )

    @property
    def called(self) -> bool:
        return self._called

    async def execute(
        self,
        arguments: dict[str, Any] | None,
    ) -> CloudMemoryToolHandlerResult:
        if self._called and self._result is not None:
            return self._result
        self._called = True
        self._requested_after_seconds = round(
            time.perf_counter() - self._turn_started_perf,
            3,
        )
        normalized = dict(arguments or {})
        query = normalized.get("query")
        granularity = normalized.get("question_granularity")
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
        started = time.perf_counter()
        try:
            self._result = await self._handler(self._arguments)
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
        return self._result

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": "completed" if self._called else "not_requested",
            "called": self._called,
            "arguments": self._arguments,
            "requested_after_seconds": self._requested_after_seconds,
            "elapsed_seconds": self._elapsed_seconds,
            "is_error": self._result.is_error if self._result else False,
            "tool_result_sent_to_cloud": (
                self._result.content if self._result else None
            ),
            "diagnostics": (
                dict(self._result.diagnostics) if self._result else {}
            ),
        }


__all__ = [
    "CloudLongTermMemoryTool",
    "CloudMemoryToolHandlerResult",
    "TOOL_NAME",
]
