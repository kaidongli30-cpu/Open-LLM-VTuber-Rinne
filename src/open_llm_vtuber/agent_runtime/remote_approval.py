"""One-time approvals for Rinne's strictly read-only remote file actions."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Callable, Mapping

from .access_policy import ChannelIdentityPolicy
from .message import AgentChannel


REMOTE_APPROVAL_PREFIX = "rinne-approval:v1:"
REMOTE_APPROVAL_DEFAULT_TTL = timedelta(minutes=2)

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
_BUTTON_DATA_RE = re.compile(
    r"^rinne-approval:v1:([A-Za-z0-9_-]{16,128}):(allow-once|deny)$"
)


class ReadOnlyApprovalAction(str, Enum):
    FILE_ROOTS = "computer_file_roots"
    FILE_STAT = "computer_file_stat"
    LIST_DIRECTORY = "computer_list_directory"
    FIND_BY_NAME = "computer_find_by_name"
    READ_TEXT = "computer_read_text"


class RemoteApprovalDecision(str, Enum):
    ALLOW_ONCE = "allow-once"
    DENY = "deny"


class RemoteApprovalState(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    CONSUMED = "consumed"
    INVALIDATED = "invalidated"


class RemoteApprovalResolution(str, Enum):
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    ALREADY_RESOLVED = "already_resolved"
    NOT_FOUND = "not_found"
    INVALID_PAYLOAD = "invalid_payload"


class RemoteApprovalAuditStage(str, Enum):
    REQUESTED = "requested"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    CONSUMED = "consumed"
    INVALIDATED = "invalidated"


class RemoteApprovalCapacityError(RuntimeError):
    """Raised when all bounded approval slots are still pending."""


@dataclass(frozen=True, slots=True)
class RemoteApprovalRequest:
    token: str = field(repr=False)
    channel: AgentChannel
    requester_id: str = field(repr=False)
    action: ReadOnlyApprovalAction
    action_fingerprint: str = field(repr=False)
    summary: str = field(repr=False)
    created_at: datetime
    expires_at: datetime

    def button_data(self, decision: RemoteApprovalDecision) -> str:
        return f"{REMOTE_APPROVAL_PREFIX}{self.token}:{decision.value}"


@dataclass(frozen=True, slots=True)
class RemoteApprovalAuditEntry:
    occurred_at: datetime
    token_digest: str
    action: ReadOnlyApprovalAction
    stage: RemoteApprovalAuditStage


@dataclass(slots=True)
class _RemoteApprovalRecord:
    request: RemoteApprovalRequest
    state: RemoteApprovalState = RemoteApprovalState.PENDING


TokenFactory = Callable[[], str]


class ReadOnlyRemoteApprovalBroker:
    """Bind one allowlisted remote decision to one exact read-only action."""

    def __init__(
        self,
        *,
        identity_policy: ChannelIdentityPolicy,
        ttl: timedelta = REMOTE_APPROVAL_DEFAULT_TTL,
        capacity: int = 128,
        audit_size: int = 512,
        token_factory: TokenFactory | None = None,
    ) -> None:
        if ttl <= timedelta(0) or ttl > timedelta(minutes=10):
            raise ValueError("remote approval TTL must be within 10 minutes")
        if capacity < 1:
            raise ValueError("remote approval capacity must be positive")
        if audit_size < 1:
            raise ValueError("remote approval audit size must be positive")
        self._identity_policy = identity_policy
        self._ttl = ttl
        self._capacity = capacity
        self._token_factory = token_factory or (lambda: secrets.token_urlsafe(24))
        self._records: OrderedDict[str, _RemoteApprovalRecord] = OrderedDict()
        self._audit: deque[RemoteApprovalAuditEntry] = deque(maxlen=audit_size)
        self._lock = threading.RLock()

    @property
    def audit_entries(self) -> tuple[RemoteApprovalAuditEntry, ...]:
        with self._lock:
            return tuple(self._audit)

    def is_pending(
        self,
        request: RemoteApprovalRequest,
        *,
        now: datetime | None = None,
    ) -> bool:
        current = _aware_now(now)
        with self._lock:
            record = self._records.get(request.token)
            if record is None or record.request != request:
                return False
            if self._expire_record(record, current):
                return False
            return record.state is RemoteApprovalState.PENDING

    def invalidate_pending(
        self,
        request: RemoteApprovalRequest,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Invalidate one exact pending request, for example after send failure."""

        current = _aware_now(now)
        with self._lock:
            record = self._records.get(request.token)
            if record is None or record.request != request:
                return False
            if self._expire_record(record, current):
                return False
            if record.state is not RemoteApprovalState.PENDING:
                return False
            record.state = RemoteApprovalState.INVALIDATED
            self._record(
                request,
                current,
                RemoteApprovalAuditStage.INVALIDATED,
            )
            return True

    def request(
        self,
        *,
        channel: AgentChannel,
        requester_id: str,
        action: ReadOnlyApprovalAction,
        parameters: Mapping[str, object],
        summary: str,
        now: datetime | None = None,
    ) -> RemoteApprovalRequest:
        current = _aware_now(now)
        requester = requester_id.strip()
        if channel is AgentChannel.DESKTOP:
            raise ValueError("remote approvals cannot originate from desktop")
        self._identity_policy.require_allowed(channel, requester)
        if not isinstance(action, ReadOnlyApprovalAction):
            raise TypeError("remote approvals accept read-only actions only")
        display_summary = summary.strip()
        if not display_summary or len(display_summary) > 500:
            raise ValueError("approval summary must contain 1 to 500 characters")
        fingerprint = fingerprint_read_only_action(action, parameters)

        with self._lock:
            self._expire(current)
            self._make_capacity()
            token = self._new_unique_token()
            request = RemoteApprovalRequest(
                token=token,
                channel=channel,
                requester_id=requester,
                action=action,
                action_fingerprint=fingerprint,
                summary=display_summary,
                created_at=current,
                expires_at=current + self._ttl,
            )
            self._records[token] = _RemoteApprovalRecord(request=request)
            self._record(request, current, RemoteApprovalAuditStage.REQUESTED)
            return request

    def resolve_button(
        self,
        *,
        button_data: str,
        channel: AgentChannel,
        requester_id: str,
        now: datetime | None = None,
    ) -> RemoteApprovalResolution:
        parsed = parse_remote_approval_button_data(button_data)
        if parsed is None:
            return RemoteApprovalResolution.INVALID_PAYLOAD
        token, decision = parsed
        current = _aware_now(now)
        requester = requester_id.strip()
        self._identity_policy.require_allowed(channel, requester)

        with self._lock:
            record = self._records.get(token)
            if record is None:
                return RemoteApprovalResolution.NOT_FOUND
            request = record.request
            if not _same_requester(request, channel, requester):
                return RemoteApprovalResolution.NOT_FOUND
            if self._expire_record(record, current):
                return RemoteApprovalResolution.EXPIRED
            if record.state is not RemoteApprovalState.PENDING:
                return RemoteApprovalResolution.ALREADY_RESOLVED
            if decision is RemoteApprovalDecision.DENY:
                record.state = RemoteApprovalState.DENIED
                self._record(
                    request,
                    current,
                    RemoteApprovalAuditStage.DENIED,
                )
                return RemoteApprovalResolution.DENIED
            record.state = RemoteApprovalState.APPROVED
            self._record(
                request,
                current,
                RemoteApprovalAuditStage.APPROVED,
            )
            return RemoteApprovalResolution.APPROVED

    def consume(
        self,
        *,
        token: str,
        channel: AgentChannel,
        requester_id: str,
        action: ReadOnlyApprovalAction,
        parameters: Mapping[str, object],
        now: datetime | None = None,
    ) -> bool:
        current = _aware_now(now)
        requester = requester_id.strip()
        self._identity_policy.require_allowed(channel, requester)
        if not isinstance(action, ReadOnlyApprovalAction):
            return False
        supplied_fingerprint = fingerprint_read_only_action(action, parameters)

        with self._lock:
            record = self._records.get(token.strip())
            if record is None:
                return False
            request = record.request
            if not _same_requester(request, channel, requester):
                return False
            if self._expire_record(record, current):
                return False
            if record.state is not RemoteApprovalState.APPROVED:
                return False
            if not hmac.compare_digest(
                supplied_fingerprint,
                request.action_fingerprint,
            ):
                record.state = RemoteApprovalState.INVALIDATED
                self._record(
                    request,
                    current,
                    RemoteApprovalAuditStage.INVALIDATED,
                )
                return False
            record.state = RemoteApprovalState.CONSUMED
            self._record(
                request,
                current,
                RemoteApprovalAuditStage.CONSUMED,
            )
            return True

    def _new_unique_token(self) -> str:
        for _ in range(8):
            token = self._token_factory().strip()
            if not _TOKEN_RE.fullmatch(token):
                raise ValueError("approval token factory returned an invalid token")
            if token not in self._records:
                return token
        raise RuntimeError("approval token factory returned repeated tokens")

    def _make_capacity(self) -> None:
        while len(self._records) >= self._capacity:
            removable = next(
                (
                    token
                    for token, record in self._records.items()
                    if record.state is not RemoteApprovalState.PENDING
                ),
                None,
            )
            if removable is None:
                raise RemoteApprovalCapacityError(
                    "all remote approval slots are pending"
                )
            del self._records[removable]

    def _expire(self, now: datetime) -> None:
        for record in self._records.values():
            self._expire_record(record, now)

    def _expire_record(
        self,
        record: _RemoteApprovalRecord,
        now: datetime,
    ) -> bool:
        if (
            record.state
            in {
                RemoteApprovalState.PENDING,
                RemoteApprovalState.APPROVED,
            }
            and now >= record.request.expires_at
        ):
            record.state = RemoteApprovalState.EXPIRED
            self._record(
                record.request,
                now,
                RemoteApprovalAuditStage.EXPIRED,
            )
        return record.state is RemoteApprovalState.EXPIRED

    def _record(
        self,
        request: RemoteApprovalRequest,
        occurred_at: datetime,
        stage: RemoteApprovalAuditStage,
    ) -> None:
        self._audit.append(
            RemoteApprovalAuditEntry(
                occurred_at=occurred_at,
                token_digest=hashlib.sha256(request.token.encode("ascii")).hexdigest()[
                    :16
                ],
                action=request.action,
                stage=stage,
            )
        )


def fingerprint_read_only_action(
    action: ReadOnlyApprovalAction,
    parameters: Mapping[str, object],
) -> str:
    if not isinstance(action, ReadOnlyApprovalAction):
        raise TypeError("action must be a ReadOnlyApprovalAction")
    try:
        canonical = json.dumps(
            {
                "action": action.value,
                "parameters": dict(parameters),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("approval parameters must be JSON serializable") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_remote_approval_button_data(
    button_data: str,
) -> tuple[str, RemoteApprovalDecision] | None:
    match = _BUTTON_DATA_RE.fullmatch(button_data.strip())
    if match is None:
        return None
    return match.group(1), RemoteApprovalDecision(match.group(2))


def _same_requester(
    request: RemoteApprovalRequest,
    channel: AgentChannel,
    requester_id: str,
) -> bool:
    return request.channel is channel and hmac.compare_digest(
        request.requester_id,
        requester_id,
    )


def _aware_now(now: datetime | None) -> datetime:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("remote approval clock must include a timezone")
    return current


__all__ = [
    "REMOTE_APPROVAL_DEFAULT_TTL",
    "REMOTE_APPROVAL_PREFIX",
    "ReadOnlyApprovalAction",
    "ReadOnlyRemoteApprovalBroker",
    "RemoteApprovalAuditEntry",
    "RemoteApprovalAuditStage",
    "RemoteApprovalCapacityError",
    "RemoteApprovalDecision",
    "RemoteApprovalRequest",
    "RemoteApprovalResolution",
    "RemoteApprovalState",
    "fingerprint_read_only_action",
    "parse_remote_approval_button_data",
]
