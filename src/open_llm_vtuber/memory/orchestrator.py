"""Context-only MemoryOrchestrator prototype for architecture stage 2."""

from __future__ import annotations

from typing import Any

from .current_conversation import CurrentConversationState
from .today_history_loader import TodayHistoryLoadResult
from .types import (
    LongTermMemoryResult,
    LongTermMemoryResultStatus,
    MemoryContextPackage,
    MemoryDecision,
    MemoryDecisionAction,
    RuntimeStateOverrides,
    UserBackgroundState,
)


class MemoryOrchestrator:
    """Minimal future-facing orchestration interface.

    The stage-2 implementation only merges caller-supplied current-session and
    today-history turns. It never retrieves long-term memory, calls a model,
    modifies supplied state, or prepares live-agent messages/system text.
    """

    async def prepare_turn(
        self,
        user_message: str,
        *,
        current_conversation: CurrentConversationState | None = None,
        today_history: TodayHistoryLoadResult | None = None,
        user_background: UserBackgroundState | None = None,
        runtime_overrides: RuntimeStateOverrides | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryContextPackage:
        """Return a context-only package without performing any I/O."""

        del user_message, metadata
        merged_conversation = self._merge_current_and_today(
            current_conversation,
            today_history,
        )
        today_diagnostics = (
            today_history.diagnostics.to_dict()
            if today_history is not None
            else None
        )
        return MemoryContextPackage(
            decision=MemoryDecision(
                action=MemoryDecisionAction.CONTEXT_ONLY,
                reason_codes=["stage2_context_only_default"],
            ),
            current_conversation=merged_conversation,
            user_background=user_background,
            runtime_overrides=runtime_overrides,
            long_term_result=LongTermMemoryResult(
                status=LongTermMemoryResultStatus.NO_MATCH,
                no_match_reason="Long-term retrieval is disabled in stage 2.",
                retrieval_health="not_run",
            ),
            dynamic_system_context="",
            llm_messages=[],
            internal_diagnostics={
                "stage": 2,
                "long_term_retrieval_called": False,
                "live_pipeline_connected": False,
                "today_history": today_diagnostics,
            },
        )

    @staticmethod
    def _merge_current_and_today(
        current: CurrentConversationState | None,
        today_history: TodayHistoryLoadResult | None,
    ) -> CurrentConversationState | None:
        if current is None and today_history is None:
            return None

        if current is None:
            merged = CurrentConversationState(
                session_id=(
                    "today-history:"
                    f"{today_history.memory_day.isoformat()}"
                )
            )
        else:
            merged = current.clone()
            merged.recent_turns = []

        if today_history is not None:
            merged.add_turns(today_history.turns)
        if current is not None:
            merged.add_turns(
                current.recent_turns,
                replace_existing=True,
            )
            merged.trim()
        return merged

    async def commit_turn(
        self,
        package: MemoryContextPackage,
        *,
        assistant_response: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Stage-2 no-op placeholder for future post-turn state updates."""

        del package, assistant_response, metadata
        return None
