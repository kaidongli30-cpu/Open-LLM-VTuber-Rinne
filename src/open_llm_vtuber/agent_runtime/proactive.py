"""In-memory, policy-gated proactive messaging for Rinne's remote channels."""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from enum import Enum
from typing import Awaitable, Callable

from .message import AgentChannel


CHINA_STANDARD_TIME = timezone(timedelta(hours=8), name="Asia/Shanghai")


class ProactiveCategory(str, Enum):
    REMINDER = "reminder"
    TASK_COMPLETION = "task_completion"
    COMPANION = "companion"


class ProactiveBlockReason(str, Enum):
    PAUSED = "paused"
    CATEGORY_DISABLED = "category_disabled"
    QUIET_HOURS = "quiet_hours"
    DAILY_LIMIT = "daily_limit"
    COOLDOWN = "cooldown"
    DUPLICATE = "duplicate"


class ProactiveDispatchStatus(str, Enum):
    BLOCKED = "blocked"
    DUPLICATE = "duplicate"
    SUBMISSION_FAILED = "submission_failed"
    PLATFORM_REJECTED = "platform_rejected"
    SUBMITTED = "submitted"
    DELIVERY_FAILED = "delivery_failed"
    DELIVERED = "delivered"


class ProactiveAuditStage(str, Enum):
    GENERATED = "generated"
    BLOCKED = "blocked"
    SUBMITTED = "submitted"
    SUBMISSION_FAILED = "submission_failed"
    PLATFORM_REJECTED = "platform_rejected"
    PLATFORM_ACCEPTED = "platform_accepted"
    DELIVERY_FAILED = "delivery_failed"
    DELIVERY_CONFIRMED = "delivery_confirmed"


@dataclass(frozen=True, slots=True)
class ProactiveMessagePolicy:
    timezone: tzinfo = CHINA_STANDARD_TIME
    quiet_start: time = time(23, 0)
    quiet_end: time = time(8, 0)
    daily_limit: int = 3
    cooldown: timedelta = timedelta(hours=2)
    paused: bool = True
    enabled_categories: frozenset[ProactiveCategory] = field(
        default_factory=lambda: frozenset(
            {
                ProactiveCategory.REMINDER,
                ProactiveCategory.TASK_COMPLETION,
            }
        )
    )

    def __post_init__(self) -> None:
        if self.quiet_start == self.quiet_end:
            raise ValueError("quiet_start and quiet_end must be different")
        if self.daily_limit < 1:
            raise ValueError("daily_limit must be positive")
        if self.cooldown < timedelta(0):
            raise ValueError("cooldown must not be negative")


@dataclass(frozen=True, slots=True)
class ProactiveMessageRequest:
    request_id: str
    channel: AgentChannel
    recipient_id: str = field(repr=False)
    category: ProactiveCategory
    text: str = field(repr=False)

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be blank")
        if not self.recipient_id.strip():
            raise ValueError("recipient_id must not be blank")
        if not self.text.strip():
            raise ValueError("text must not be blank")
        if self.channel is AgentChannel.DESKTOP:
            raise ValueError("proactive dispatcher is for remote channels")


@dataclass(frozen=True, slots=True)
class ProactiveGateDecision:
    allowed: bool
    reason: ProactiveBlockReason | None = None


@dataclass(frozen=True, slots=True)
class ProactiveDeliveryReceipt:
    accepted: bool
    delivery_confirmed: bool | None = None


@dataclass(frozen=True, slots=True)
class ProactiveDispatchResult:
    status: ProactiveDispatchStatus
    reason: ProactiveBlockReason | None = None


@dataclass(frozen=True, slots=True)
class ProactiveAuditEntry:
    occurred_at: datetime
    request_id: str
    channel: AgentChannel
    category: ProactiveCategory
    stage: ProactiveAuditStage
    error_type: str | None = None


ProactiveDeliverer = Callable[
    [ProactiveMessageRequest],
    Awaitable[ProactiveDeliveryReceipt],
]


class ProactiveMessageGate:
    """Enforce pause, category, quiet hours, cap, cooldown and deduplication."""

    def __init__(
        self,
        policy: ProactiveMessagePolicy | None = None,
        *,
        duplicate_cache_size: int = 2048,
    ) -> None:
        if duplicate_cache_size < 1:
            raise ValueError("duplicate_cache_size must be positive")
        self._policy = policy or ProactiveMessagePolicy()
        self._paused = self._policy.paused
        self._enabled_categories = set(self._policy.enabled_categories)
        self._accepted_by_day: dict[date, int] = {}
        self._last_accepted_at: datetime | None = None
        self._accepted_ids: OrderedDict[str, None] = OrderedDict()
        self._duplicate_cache_size = duplicate_cache_size

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def set_category_enabled(
        self,
        category: ProactiveCategory,
        enabled: bool,
    ) -> None:
        if enabled:
            self._enabled_categories.add(category)
        else:
            self._enabled_categories.discard(category)

    def evaluate(
        self,
        request: ProactiveMessageRequest,
        *,
        now: datetime,
    ) -> ProactiveGateDecision:
        local_now = self._local_time(now)
        if self._paused:
            return ProactiveGateDecision(False, ProactiveBlockReason.PAUSED)
        if request.category not in self._enabled_categories:
            return ProactiveGateDecision(
                False,
                ProactiveBlockReason.CATEGORY_DISABLED,
            )
        if self._in_quiet_hours(local_now.timetz().replace(tzinfo=None)):
            return ProactiveGateDecision(False, ProactiveBlockReason.QUIET_HOURS)
        if request.request_id in self._accepted_ids:
            return ProactiveGateDecision(False, ProactiveBlockReason.DUPLICATE)
        if self._accepted_by_day.get(local_now.date(), 0) >= self._policy.daily_limit:
            return ProactiveGateDecision(False, ProactiveBlockReason.DAILY_LIMIT)
        if (
            self._last_accepted_at is not None
            and local_now - self._last_accepted_at < self._policy.cooldown
        ):
            return ProactiveGateDecision(False, ProactiveBlockReason.COOLDOWN)
        return ProactiveGateDecision(True)

    def record_accepted(
        self,
        request: ProactiveMessageRequest,
        *,
        now: datetime,
    ) -> None:
        local_now = self._local_time(now)
        self._accepted_by_day[local_now.date()] = (
            self._accepted_by_day.get(local_now.date(), 0) + 1
        )
        self._last_accepted_at = local_now
        self._accepted_ids[request.request_id] = None
        self._accepted_ids.move_to_end(request.request_id)
        while len(self._accepted_ids) > self._duplicate_cache_size:
            self._accepted_ids.popitem(last=False)
        self._prune_old_daily_counts(local_now.date())

    def _local_time(self, now: datetime) -> datetime:
        if now.tzinfo is None:
            raise ValueError("now must include a timezone")
        return now.astimezone(self._policy.timezone)

    def _in_quiet_hours(self, current: time) -> bool:
        start = self._policy.quiet_start
        end = self._policy.quiet_end
        if start < end:
            return start <= current < end
        return current >= start or current < end

    def _prune_old_daily_counts(self, today: date) -> None:
        for stored_day in list(self._accepted_by_day):
            if stored_day != today:
                del self._accepted_by_day[stored_day]


class ProactiveMessageDispatcher:
    """Dispatch policy-approved messages and retain a bounded in-memory audit."""

    def __init__(
        self,
        *,
        gate: ProactiveMessageGate,
        deliverer: ProactiveDeliverer,
        audit_size: int = 512,
    ) -> None:
        if audit_size < 1:
            raise ValueError("audit_size must be positive")
        self._gate = gate
        self._deliverer = deliverer
        self._audit: deque[ProactiveAuditEntry] = deque(maxlen=audit_size)
        self._dispatch_lock = asyncio.Lock()

    @property
    def audit_entries(self) -> tuple[ProactiveAuditEntry, ...]:
        return tuple(self._audit)

    async def dispatch(
        self,
        request: ProactiveMessageRequest,
        *,
        now: datetime,
    ) -> ProactiveDispatchResult:
        async with self._dispatch_lock:
            return await self._dispatch_serialized(request, now=now)

    async def _dispatch_serialized(
        self,
        request: ProactiveMessageRequest,
        *,
        now: datetime,
    ) -> ProactiveDispatchResult:
        self._record(request, now, ProactiveAuditStage.GENERATED)
        decision = self._gate.evaluate(request, now=now)
        if not decision.allowed:
            self._record(request, now, ProactiveAuditStage.BLOCKED)
            status = (
                ProactiveDispatchStatus.DUPLICATE
                if decision.reason is ProactiveBlockReason.DUPLICATE
                else ProactiveDispatchStatus.BLOCKED
            )
            return ProactiveDispatchResult(status=status, reason=decision.reason)

        self._record(request, now, ProactiveAuditStage.SUBMITTED)
        try:
            receipt = await self._deliverer(request)
        except Exception as exc:
            self._record(
                request,
                now,
                ProactiveAuditStage.SUBMISSION_FAILED,
                error_type=type(exc).__name__,
            )
            return ProactiveDispatchResult(ProactiveDispatchStatus.SUBMISSION_FAILED)

        if not receipt.accepted:
            self._record(request, now, ProactiveAuditStage.PLATFORM_REJECTED)
            return ProactiveDispatchResult(ProactiveDispatchStatus.PLATFORM_REJECTED)

        self._gate.record_accepted(request, now=now)
        self._record(request, now, ProactiveAuditStage.PLATFORM_ACCEPTED)
        if receipt.delivery_confirmed is True:
            self._record(request, now, ProactiveAuditStage.DELIVERY_CONFIRMED)
            return ProactiveDispatchResult(ProactiveDispatchStatus.DELIVERED)
        if receipt.delivery_confirmed is False:
            self._record(request, now, ProactiveAuditStage.DELIVERY_FAILED)
            return ProactiveDispatchResult(ProactiveDispatchStatus.DELIVERY_FAILED)
        return ProactiveDispatchResult(ProactiveDispatchStatus.SUBMITTED)

    def _record(
        self,
        request: ProactiveMessageRequest,
        occurred_at: datetime,
        stage: ProactiveAuditStage,
        *,
        error_type: str | None = None,
    ) -> None:
        if occurred_at.tzinfo is None:
            raise ValueError("occurred_at must include a timezone")
        self._audit.append(
            ProactiveAuditEntry(
                occurred_at=occurred_at,
                request_id=request.request_id,
                channel=request.channel,
                category=request.category,
                stage=stage,
                error_type=error_type,
            )
        )


@dataclass(frozen=True, slots=True)
class ScheduledReminder:
    reminder_id: str
    due_at: datetime
    channel: AgentChannel
    recipient_id: str = field(repr=False)
    text: str

    def __post_init__(self) -> None:
        if not self.reminder_id.strip():
            raise ValueError("reminder_id must not be blank")
        if self.due_at.tzinfo is None:
            raise ValueError("due_at must include a timezone")

    def to_request(self) -> ProactiveMessageRequest:
        return ProactiveMessageRequest(
            request_id=f"reminder:{self.reminder_id}",
            channel=self.channel,
            recipient_id=self.recipient_id,
            category=ProactiveCategory.REMINDER,
            text=self.text,
        )


class InMemoryReminderScheduler:
    """Hold explicit reminders in memory; never creates or modifies files."""

    def __init__(self, *, max_reminders: int = 1000) -> None:
        if max_reminders < 1:
            raise ValueError("max_reminders must be positive")
        self._reminders: dict[str, ScheduledReminder] = {}
        self._max_reminders = max_reminders

    @property
    def pending(self) -> tuple[ScheduledReminder, ...]:
        return tuple(sorted(self._reminders.values(), key=lambda item: item.due_at))

    def schedule(self, reminder: ScheduledReminder) -> None:
        if (
            reminder.reminder_id not in self._reminders
            and len(self._reminders) >= self._max_reminders
        ):
            raise ValueError("reminder capacity reached")
        self._reminders[reminder.reminder_id] = reminder

    def cancel(self, reminder_id: str) -> bool:
        return self._reminders.pop(reminder_id, None) is not None

    async def dispatch_due(
        self,
        dispatcher: ProactiveMessageDispatcher,
        *,
        now: datetime,
    ) -> list[ProactiveDispatchResult]:
        if now.tzinfo is None:
            raise ValueError("now must include a timezone")
        results: list[ProactiveDispatchResult] = []
        due = [item for item in self.pending if item.due_at <= now]
        for reminder in due:
            result = await dispatcher.dispatch(reminder.to_request(), now=now)
            results.append(result)
            if result.status in {
                ProactiveDispatchStatus.DUPLICATE,
                ProactiveDispatchStatus.SUBMITTED,
                ProactiveDispatchStatus.DELIVERY_FAILED,
                ProactiveDispatchStatus.DELIVERED,
            }:
                self._reminders.pop(reminder.reminder_id, None)
        return results


def task_completion_request(
    *,
    task_id: str,
    channel: AgentChannel,
    recipient_id: str,
    text: str,
) -> ProactiveMessageRequest:
    return ProactiveMessageRequest(
        request_id=f"task:{task_id}",
        channel=channel,
        recipient_id=recipient_id,
        category=ProactiveCategory.TASK_COMPLETION,
        text=text,
    )


__all__ = [
    "InMemoryReminderScheduler",
    "ProactiveAuditEntry",
    "ProactiveAuditStage",
    "ProactiveBlockReason",
    "ProactiveCategory",
    "ProactiveDeliveryReceipt",
    "ProactiveDispatchResult",
    "ProactiveDispatchStatus",
    "ProactiveMessageDispatcher",
    "ProactiveMessageGate",
    "ProactiveMessagePolicy",
    "ProactiveMessageRequest",
    "ScheduledReminder",
    "task_completion_request",
]
