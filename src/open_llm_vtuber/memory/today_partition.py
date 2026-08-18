"""Read-only partitioning for one memory day's messages at a chosen time."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from .today_history_loader import TodayHistoryLoadResult
from .types import (
    ConversationTurn,
    SerializableMemoryModel,
    ensure_aware_datetime,
)


def _message_sort_key(message: ConversationTurn) -> tuple[datetime, str, str]:
    return (
        message.timestamp.astimezone(timezone.utc),
        message.source_ref or "",
        message.turn_id,
    )


def _count_complete_role_pairs(
    messages: list[ConversationTurn],
) -> tuple[int, int]:
    """Count adjacent user/assistant pairs and unpaired message records."""

    complete_pairs = 0
    unpaired_messages = 0
    index = 0
    while index < len(messages):
        if (
            index + 1 < len(messages)
            and messages[index].role == "user"
            and messages[index + 1].role == "assistant"
        ):
            complete_pairs += 1
            index += 2
            continue
        unpaired_messages += 1
        index += 1
    return complete_pairs, unpaired_messages


@dataclass
class TodayMessagePartitionDiagnostics(SerializableMemoryModel):
    """Content-free statistics for one chosen delivery time."""

    loaded_memory_day_messages: int = 0
    messages_strictly_before_as_of: int = 0
    messages_at_or_after_as_of: int = 0
    requested_recent_messages: int = 20
    recent_messages: int = 0
    older_today_messages: int = 0
    complete_user_assistant_pairs_in_recent: int = 0
    unpaired_messages_in_recent: int = 0
    recent_window_filled: bool = False
    chronological: bool = True


@dataclass
class TodayMessagePartition(SerializableMemoryModel):
    """Complete internal partition; no field is connected to a live LLM."""

    memory_day: date
    as_of_exclusive: datetime
    window_start: datetime
    window_end: datetime
    all_messages_before_as_of: list[ConversationTurn] = field(default_factory=list)
    older_today_messages: list[ConversationTurn] = field(default_factory=list)
    recent_messages: list[ConversationTurn] = field(default_factory=list)
    diagnostics: TodayMessagePartitionDiagnostics = field(
        default_factory=TodayMessagePartitionDiagnostics
    )

    def __post_init__(self) -> None:
        ensure_aware_datetime(
            self.as_of_exclusive,
            "TodayMessagePartition.as_of_exclusive",
        )
        ensure_aware_datetime(
            self.window_start,
            "TodayMessagePartition.window_start",
        )
        ensure_aware_datetime(
            self.window_end,
            "TodayMessagePartition.window_end",
        )


@dataclass
class TodayLlmDeliveryPreview(SerializableMemoryModel):
    """Exact simulated turn payload plus diagnostics kept outside the LLM."""

    memory_day: date
    as_of_exclusive: datetime
    simulated_messages_for_llm: list[dict[str, str]] = field(default_factory=list)
    internal_delivery_manifest_not_sent_to_llm: list[dict[str, Any]] = field(
        default_factory=list
    )
    older_today_message_count_not_sent: int = 0
    messages_at_or_after_as_of_not_sent: int = 0
    history_message_count: int = 0
    current_user_input_included: bool = True
    dynamic_system_context: str = ""
    live_pipeline_connected: bool = False
    selection_rule: str = (
        "Keep the newest N normalized messages with timestamp strictly before "
        "as_of; order them chronologically; append the current user input once; "
        "send only role and content."
    )
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ensure_aware_datetime(
            self.as_of_exclusive,
            "TodayLlmDeliveryPreview.as_of_exclusive",
        )
        if self.live_pipeline_connected:
            raise ValueError("This preview must never claim a live connection")
        if not self.current_user_input_included:
            raise ValueError("This stage requires one current user input")
        if len(self.simulated_messages_for_llm) != self.history_message_count + 1:
            raise ValueError(
                "simulated messages must contain history plus one current input"
            )
        if not self.simulated_messages_for_llm:
            raise ValueError("simulated messages cannot be empty")
        if self.simulated_messages_for_llm[-1].get("role") != "user":
            raise ValueError("the current input must be the final user message")

    def to_dict(self) -> dict[str, Any]:
        """Make the LLM payload visibly separate from internal diagnostics."""

        return {
            "preview_metadata": {
                "memory_day": self.memory_day.isoformat(),
                "as_of_exclusive": self.as_of_exclusive.isoformat(),
                "history_message_count": self.history_message_count,
                "current_user_input_included": self.current_user_input_included,
                "total_message_count": len(self.simulated_messages_for_llm),
                "dynamic_system_context": self.dynamic_system_context,
                "live_pipeline_connected": self.live_pipeline_connected,
                "selection_rule": self.selection_rule,
            },
            "would_send_to_llm": {
                "messages": [
                    dict(message) for message in self.simulated_messages_for_llm
                ]
            },
            "internal_diagnostics_not_sent_to_llm": {
                "delivery_manifest": [
                    dict(item)
                    for item in self.internal_delivery_manifest_not_sent_to_llm
                ],
                "older_today_message_count": (self.older_today_message_count_not_sent),
                "messages_at_or_after_as_of": (
                    self.messages_at_or_after_as_of_not_sent
                ),
                "partition": dict(self.diagnostics),
            },
        }


class TodayMessagePartitioner:
    """Split loaded messages into older-today and recent raw-message regions."""

    def __init__(self, recent_message_count: int = 20) -> None:
        if recent_message_count <= 0:
            raise ValueError("recent_message_count must be positive")
        self.recent_message_count = recent_message_count

    def partition(
        self,
        history: TodayHistoryLoadResult,
        *,
        as_of: datetime,
    ) -> TodayMessagePartition:
        """Partition without mutating the supplied loader result.

        ``as_of`` is exclusive because it represents the instant immediately
        before a new user input would be added to conversation history.
        """

        ensure_aware_datetime(as_of, "as_of")
        local_as_of = as_of.astimezone(history.window_start.tzinfo)
        if not history.window_start <= local_as_of < history.window_end:
            raise ValueError("as_of must fall inside the loaded memory-day window")

        ordered = sorted(history.turns, key=_message_sort_key)
        before_as_of = [
            message for message in ordered if message.timestamp < local_as_of
        ]
        recent = before_as_of[-self.recent_message_count :]
        older = before_as_of[: -len(recent)] if recent else list(before_as_of)
        complete_pairs, unpaired = _count_complete_role_pairs(recent)
        chronological = all(
            left.timestamp <= right.timestamp
            for left, right in zip(before_as_of, before_as_of[1:])
        )
        diagnostics = TodayMessagePartitionDiagnostics(
            loaded_memory_day_messages=len(ordered),
            messages_strictly_before_as_of=len(before_as_of),
            messages_at_or_after_as_of=len(ordered) - len(before_as_of),
            requested_recent_messages=self.recent_message_count,
            recent_messages=len(recent),
            older_today_messages=len(older),
            complete_user_assistant_pairs_in_recent=complete_pairs,
            unpaired_messages_in_recent=unpaired,
            recent_window_filled=(len(recent) == self.recent_message_count),
            chronological=chronological,
        )
        return TodayMessagePartition(
            memory_day=history.memory_day,
            as_of_exclusive=local_as_of,
            window_start=history.window_start,
            window_end=history.window_end,
            all_messages_before_as_of=before_as_of,
            older_today_messages=older,
            recent_messages=recent,
            diagnostics=diagnostics,
        )


def build_today_llm_delivery_preview(
    partition: TodayMessagePartition,
    *,
    current_user_input: str,
) -> TodayLlmDeliveryPreview:
    """Build ``recent history + current input`` without calling a live service."""

    if not isinstance(current_user_input, str):
        raise TypeError("current_user_input must be a string")

    simulated_messages = [
        {"role": message.role, "content": message.content}
        for message in partition.recent_messages
    ]
    simulated_messages.append({"role": "user", "content": current_user_input})
    manifest = [
        {
            "message_number": index,
            "role": message.role,
            "timestamp": message.timestamp.isoformat(),
            "source_ref": message.source_ref,
            "turn_id": message.turn_id,
        }
        for index, message in enumerate(partition.recent_messages, start=1)
    ]
    manifest.append(
        {
            "message_number": len(manifest) + 1,
            "role": "user",
            "timestamp": partition.as_of_exclusive.isoformat(),
            "source_ref": "current_user_input",
            "turn_id": None,
        }
    )
    return TodayLlmDeliveryPreview(
        memory_day=partition.memory_day,
        as_of_exclusive=partition.as_of_exclusive,
        simulated_messages_for_llm=simulated_messages,
        internal_delivery_manifest_not_sent_to_llm=manifest,
        older_today_message_count_not_sent=(partition.diagnostics.older_today_messages),
        messages_at_or_after_as_of_not_sent=(
            partition.diagnostics.messages_at_or_after_as_of
        ),
        history_message_count=len(partition.recent_messages),
        diagnostics=partition.diagnostics.to_dict(),
    )
