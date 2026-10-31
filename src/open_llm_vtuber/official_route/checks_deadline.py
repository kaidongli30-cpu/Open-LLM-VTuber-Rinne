"""One shared foreground deadline across both checks and the final wire gate."""

from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic

_DEADLINE = ContextVar("official_check_deadline", default=None)
_TIMEOUT = ContextVar("official_check_timeout", default=None)
_FOREGROUND = ContextVar("official_check_foreground", default=False)
_AUDIT = ContextVar("official_check_audit", default=None)
_SCOPE = ContextVar("official_check_scope", default=("chat_content", "background"))
_DAY = ContextVar("official_check_day", default=None)


@contextmanager
def check_deadline(seconds=20, *, deadline=None, foreground=None):
    existing = _DEADLINE.get()
    chosen = monotonic() + seconds if deadline is None else deadline
    if existing is not None:
        chosen = min(existing, chosen)
    deadline_token = _DEADLINE.set(chosen)
    timeout_token = _TIMEOUT.set(seconds)
    foreground_token = _FOREGROUND.set(
        seconds <= 20 if foreground is None else foreground
    )
    try:
        yield chosen
    finally:
        _TIMEOUT.reset(timeout_token)
        _DEADLINE.reset(deadline_token)
        _FOREGROUND.reset(foreground_token)


@contextmanager
def check_scope(category="chat_content", phase=None, *, audit=None, day=None):
    scope_token = _SCOPE.set((category, _SCOPE.get()[1] if phase is None else phase))
    audit_token = _AUDIT.set(_AUDIT.get() if audit is None else audit)
    day_token = _DAY.set(_DAY.get() if day is None else str(day))
    try:
        yield
    finally:
        _DAY.reset(day_token)
        _AUDIT.reset(audit_token)
        _SCOPE.reset(scope_token)


def is_foreground():
    return _FOREGROUND.get()


def check_day():
    from datetime import datetime

    return _DAY.get() or datetime.now().astimezone().date().isoformat()


def audit_call(seconds, wait_seconds=0):
    target = _AUDIT.get()
    if target is not None:
        category, phase = _SCOPE.get()
        target.append(
            dict(
                category=category,
                phase=phase,
                seconds=seconds,
                wait_seconds=wait_seconds,
            )
        )


def remaining_timeout(default):
    value = _TIMEOUT.get()
    value = default if value is None else value
    deadline = _DEADLINE.get()
    return (
        max(0.0, min(value, deadline - monotonic())) if deadline is not None else value
    )


def is_pending(record):
    return record is None or record.source == "check_failure"
