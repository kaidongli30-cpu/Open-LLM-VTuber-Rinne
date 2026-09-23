"""Helpers for useful runtime logs that never include conversation content."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


_KNOWN_LOG_KEYS = frozenset(
    {
        "args",
        "content",
        "data",
        "error",
        "id",
        "input",
        "message",
        "name",
        "role",
        "status",
        "text",
        "tool_id",
        "tool_name",
        "type",
    }
)
_KNOWN_MESSAGE_ROLES = frozenset({"assistant", "system", "tool", "user"})


def text_log_fields(value: Any) -> dict[str, int | bool]:
    """Describe text-like data without returning any part of its content."""

    if value is None:
        return {"present": False, "chars": 0}
    if isinstance(value, str):
        return {"present": bool(value), "chars": len(value)}
    return {"present": True, "chars": 0}


def message_batch_log_fields(messages: Iterable[Any]) -> dict[str, Any]:
    """Summarize an LLM message batch without serializing message values."""

    items = list(messages)
    roles: Counter[str] = Counter()
    content_kinds: Counter[str] = Counter()
    for item in items:
        if not isinstance(item, Mapping):
            roles["invalid"] += 1
            content_kinds[type(item).__name__] += 1
            continue
        role = item.get("role")
        roles[role if role in _KNOWN_MESSAGE_ROLES else "unknown"] += 1
        content = item.get("content")
        if isinstance(content, str):
            content_kinds["text"] += 1
        elif isinstance(content, list):
            content_kinds["multimodal"] += 1
        elif content is None:
            content_kinds["none"] += 1
        else:
            content_kinds[type(content).__name__] += 1
    return {
        "count": len(items),
        "roles": dict(sorted(roles.items())),
        "content_kinds": dict(sorted(content_kinds.items())),
    }


def mapping_log_fields(value: Any) -> dict[str, Any]:
    """Describe a mapping while deliberately omitting all values."""

    if not isinstance(value, Mapping):
        return {"type": type(value).__name__, "key_count": 0, "known_keys": []}
    known_keys = sorted(
        key for key in value.keys() if isinstance(key, str) and key in _KNOWN_LOG_KEYS
    )
    return {
        "type": type(value).__name__,
        "key_count": len(value),
        "known_keys": known_keys,
    }
