"""Default-disabled runtime configuration for proactive remote messages."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Mapping

from .proactive import ProactiveCategory, ProactiveMessagePolicy


PROACTIVE_ENABLED_ENV = "RINNE_PROACTIVE_ENABLED"
PROACTIVE_QQ_RECIPIENT_ENV = "RINNE_QQ_PROACTIVE_RECIPIENT_OPENID"
PROACTIVE_QUIET_START_ENV = "RINNE_PROACTIVE_QUIET_START"
PROACTIVE_QUIET_END_ENV = "RINNE_PROACTIVE_QUIET_END"
PROACTIVE_DAILY_LIMIT_ENV = "RINNE_PROACTIVE_DAILY_LIMIT"
PROACTIVE_COOLDOWN_MINUTES_ENV = "RINNE_PROACTIVE_COOLDOWN_MINUTES"
PROACTIVE_COMPANION_ENABLED_ENV = "RINNE_PROACTIVE_COMPANION_ENABLED"
PROACTIVE_REMINDER_POLL_SECONDS_ENV = "RINNE_PROACTIVE_REMINDER_POLL_SECONDS"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})


class ProactiveRuntimeConfigurationError(ValueError):
    """Raised when proactive messaging cannot fail closed safely."""


@dataclass(frozen=True, slots=True)
class ProactiveRuntimeConfig:
    enabled: bool = False
    qq_recipient_openid: str = field(default="", repr=False)
    quiet_start: time = time(23, 0)
    quiet_end: time = time(8, 0)
    daily_limit: int = 3
    cooldown: timedelta = timedelta(hours=2)
    companion_enabled: bool = False
    reminder_poll_seconds: float = 15.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "qq_recipient_openid",
            self.qq_recipient_openid.strip(),
        )
        if self.quiet_start == self.quiet_end:
            raise ProactiveRuntimeConfigurationError(
                "proactive quiet hours must not cover the entire day"
            )
        if self.daily_limit < 1 or self.daily_limit > 20:
            raise ProactiveRuntimeConfigurationError(
                "proactive daily limit must be between 1 and 20"
            )
        if self.cooldown < timedelta(0) or self.cooldown > timedelta(days=1):
            raise ProactiveRuntimeConfigurationError(
                "proactive cooldown must be between 0 and 1440 minutes"
            )
        if not 1.0 <= self.reminder_poll_seconds <= 300.0:
            raise ProactiveRuntimeConfigurationError(
                "reminder poll interval must be between 1 and 300 seconds"
            )

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "ProactiveRuntimeConfig":
        values = os.environ if environ is None else environ
        return cls(
            enabled=_parse_bool(
                values.get(PROACTIVE_ENABLED_ENV, ""),
                PROACTIVE_ENABLED_ENV,
            ),
            qq_recipient_openid=values.get(PROACTIVE_QQ_RECIPIENT_ENV, ""),
            quiet_start=_parse_time(
                values.get(PROACTIVE_QUIET_START_ENV, "23:00"),
                PROACTIVE_QUIET_START_ENV,
            ),
            quiet_end=_parse_time(
                values.get(PROACTIVE_QUIET_END_ENV, "08:00"),
                PROACTIVE_QUIET_END_ENV,
            ),
            daily_limit=_parse_int(
                values.get(PROACTIVE_DAILY_LIMIT_ENV, "3"),
                PROACTIVE_DAILY_LIMIT_ENV,
            ),
            cooldown=timedelta(
                minutes=_parse_int(
                    values.get(PROACTIVE_COOLDOWN_MINUTES_ENV, "120"),
                    PROACTIVE_COOLDOWN_MINUTES_ENV,
                )
            ),
            companion_enabled=_parse_bool(
                values.get(PROACTIVE_COMPANION_ENABLED_ENV, ""),
                PROACTIVE_COMPANION_ENABLED_ENV,
            ),
            reminder_poll_seconds=_parse_float(
                values.get(PROACTIVE_REMINDER_POLL_SECONDS_ENV, "15"),
                PROACTIVE_REMINDER_POLL_SECONDS_ENV,
            ),
        )

    def require_qq_recipient(self, allowed_openids: frozenset[str]) -> str:
        if not self.enabled:
            raise ProactiveRuntimeConfigurationError("proactive messaging is disabled")
        recipient = self.qq_recipient_openid
        if not recipient:
            raise ProactiveRuntimeConfigurationError(
                f"enabled proactive messaging requires {PROACTIVE_QQ_RECIPIENT_ENV}"
            )
        if recipient not in allowed_openids:
            raise ProactiveRuntimeConfigurationError(
                "proactive QQ recipient must exactly match the QQ allowlist"
            )
        return recipient

    def to_policy(self) -> ProactiveMessagePolicy:
        categories = {
            ProactiveCategory.REMINDER,
            ProactiveCategory.TASK_COMPLETION,
        }
        if self.companion_enabled:
            categories.add(ProactiveCategory.COMPANION)
        return ProactiveMessagePolicy(
            quiet_start=self.quiet_start,
            quiet_end=self.quiet_end,
            daily_limit=self.daily_limit,
            cooldown=self.cooldown,
            paused=not self.enabled,
            enabled_categories=frozenset(categories),
        )


def _parse_bool(raw_value: str, field_name: str) -> bool:
    normalized = raw_value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ProactiveRuntimeConfigurationError(
        f"{field_name} must be one of: 1, 0, true, false, yes, no, on, off"
    )


def _parse_time(raw_value: str, field_name: str) -> time:
    try:
        return datetime.strptime(raw_value.strip(), "%H:%M").time()
    except ValueError as exc:
        raise ProactiveRuntimeConfigurationError(
            f"{field_name} must use 24-hour HH:MM format"
        ) from exc


def _parse_int(raw_value: str, field_name: str) -> int:
    try:
        return int(raw_value.strip())
    except ValueError as exc:
        raise ProactiveRuntimeConfigurationError(
            f"{field_name} must be an integer"
        ) from exc


def _parse_float(raw_value: str, field_name: str) -> float:
    try:
        return float(raw_value.strip())
    except ValueError as exc:
        raise ProactiveRuntimeConfigurationError(
            f"{field_name} must be a number"
        ) from exc


__all__ = [
    "PROACTIVE_COMPANION_ENABLED_ENV",
    "PROACTIVE_COOLDOWN_MINUTES_ENV",
    "PROACTIVE_DAILY_LIMIT_ENV",
    "PROACTIVE_ENABLED_ENV",
    "PROACTIVE_QQ_RECIPIENT_ENV",
    "PROACTIVE_QUIET_END_ENV",
    "PROACTIVE_QUIET_START_ENV",
    "PROACTIVE_REMINDER_POLL_SECONDS_ENV",
    "ProactiveRuntimeConfig",
    "ProactiveRuntimeConfigurationError",
]
