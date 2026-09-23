"""Validated date windows requested by the long-term-memory tool.

The cloud model is responsible for interpreting a person's wording (for
example, "五月中下旬" or "生日那周").  The local runtime deliberately does
not try to reproduce that language understanding.  It only accepts the
model's canonical inclusive ISO dates and decides whether the window is a
hard filter or a soft ranking hint based on the model's stated confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Any, Literal


TimeRangeConfidence = Literal["high", "medium", "low", "none"]

_CONFIDENCES = frozenset({"high", "medium", "low", "none"})
_MAX_BASIS_CHARS = 240
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class ValidatedTimeRange:
    """A safe, inclusive memory-day window supplied by the cloud model."""

    start_date: str | None = None
    end_date: str | None = None
    confidence: TimeRangeConfidence = "none"
    basis: str = ""

    @property
    def mode(self) -> Literal["hard", "soft", "none"]:
        if self.start_date is None:
            return "none"
        return "soft" if self.confidence == "low" else "hard"

    @property
    def is_bounded(self) -> bool:
        return self.start_date is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "confidence": self.confidence,
            "basis": self.basis,
            "mode": self.mode,
        }


def _parse_iso_date(value: Any, field_name: str) -> date | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO date string")
    normalized = value.strip()
    if not normalized:
        return None
    if _ISO_DATE.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} must use YYYY-MM-DD")
    try:
        return date.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must use YYYY-MM-DD"
        ) from exc


def normalize_time_range(
    start_date: Any = None,
    end_date: Any = None,
    confidence: Any = "none",
    basis: Any = "",
) -> ValidatedTimeRange:
    """Validate and normalize model-produced date-range arguments.

    A single date is treated as a one-day exact window.  A supplied range
    without an explicit confidence is deliberately downgraded to ``low`` so
    it cannot silently become a hard exclusion.  Reversed or malformed dates
    are rejected instead of being guessed or silently swapped.
    """

    start = _parse_iso_date(start_date, "start_date")
    end = _parse_iso_date(end_date, "end_date")
    if (start is None) != (end is None):
        if start is not None:
            end = start
        else:
            start = end
    if start is not None and end is not None and end < start:
        raise ValueError("end_date cannot be before start_date")

    normalized_confidence = str(confidence or "none").strip().casefold()
    if normalized_confidence not in _CONFIDENCES:
        raise ValueError(
            "time_range_confidence must be high, medium, low, or none"
        )
    if start is None:
        normalized_confidence = "none"
    elif normalized_confidence == "none":
        normalized_confidence = "low"

    if basis is None:
        normalized_basis = ""
    elif isinstance(basis, str):
        normalized_basis = " ".join(basis.split())[:_MAX_BASIS_CHARS]
    else:
        raise ValueError("time_range_basis must be a string")

    return ValidatedTimeRange(
        start_date=start.isoformat() if start is not None else None,
        end_date=end.isoformat() if end is not None else None,
        confidence=normalized_confidence,  # type: ignore[arg-type]
        basis=normalized_basis,
    )


__all__ = [
    "TimeRangeConfidence",
    "ValidatedTimeRange",
    "normalize_time_range",
]
