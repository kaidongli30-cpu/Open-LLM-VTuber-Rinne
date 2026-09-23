"""Load user-authored response rules without involving Layer-2 generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..data_paths import resolve_character_history_root
from .layer2_context import LAYER2_DIRECTORY_NAME


MANDATORY_RESPONSE_RULES_NAME = "user_mandatory_response_rules.md"
LEGACY_RESPONSE_RULES_NAME = "darling_mandatory_response_rules.md"
MAX_MANDATORY_RESPONSE_RULES_BYTES = 32 * 1024


@dataclass(frozen=True)
class MandatoryResponseRulesResult:
    context: str
    diagnostics: dict[str, Any]
    path: Path


def load_mandatory_response_rules(
    history_root: str | Path | None = None,
) -> MandatoryResponseRulesResult:
    """Load the user-authored rules stored beside, but outside, Layer 2 output."""

    history = resolve_character_history_root(history_root).resolve()
    directory = history / LAYER2_DIRECTORY_NAME
    path = directory / MANDATORY_RESPONSE_RULES_NAME
    if not path.is_file():
        legacy_path = directory / LEGACY_RESPONSE_RULES_NAME
        if legacy_path.is_file():
            path = legacy_path
    diagnostics: dict[str, Any] = {
        "status": "missing",
        "path": str(path),
    }
    if not path.is_file():
        return MandatoryResponseRulesResult("", diagnostics, path)

    try:
        byte_count = path.stat().st_size
        if byte_count <= 0:
            raise ValueError("mandatory_response_rules_empty")
        if byte_count > MAX_MANDATORY_RESPONSE_RULES_BYTES:
            raise ValueError(
                "mandatory_response_rules_too_large:"
                f"{byte_count}>{MAX_MANDATORY_RESPONSE_RULES_BYTES}"
            )
        raw = path.read_bytes()
        context = raw.decode("utf-8").strip()
        if not context:
            raise ValueError("mandatory_response_rules_empty")
    except (OSError, UnicodeError, ValueError) as exc:
        return MandatoryResponseRulesResult(
            "",
            {
                **diagnostics,
                "status": "invalid",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            path,
        )

    return MandatoryResponseRulesResult(
        context,
        {
            **diagnostics,
            "status": "loaded",
            "character_count": len(context),
            "byte_count": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        path,
    )


__all__ = [
    "MANDATORY_RESPONSE_RULES_NAME",
    "MAX_MANDATORY_RESPONSE_RULES_BYTES",
    "MandatoryResponseRulesResult",
    "load_mandatory_response_rules",
]
