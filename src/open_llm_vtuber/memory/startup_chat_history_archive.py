"""Archive completed chat-history JSON files during backend startup.

The live conversation loader reads JSON files directly below the character
history root, while older raw sessions are exposed through ``past_history``.
This module performs the one mutating startup step that keeps those two areas
separate. A memory day starts at 03:00, matching the existing diary and
today-history rules.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

from ..data_paths import resolve_character_history_root
from .diary_review import diary_approval_status
from .diary_sources import history_days


_HISTORY_TIMESTAMP_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(?:_|$)"
)
_ARCHIVE_MARKER_FILENAME = ".last_archived_memory_boundary"


@dataclass(frozen=True)
class ChatHistoryArchiveResult:
    """Summary of one startup archive attempt."""

    boundary: datetime
    moved_count: int = 0
    failed_count: int = 0
    skipped_due_to_marker: bool = False
    marker_written: bool = False


def _latest_memory_boundary(reference_time: datetime) -> datetime:
    """Return the most recent local 03:00 boundary.

    Between midnight and 02:59 the current logical memory day is still the
    previous calendar day, so the previous day's 03:00 boundary is used. This
    prevents an unfinished logical day from being archived too early.
    """

    boundary = reference_time.replace(hour=3, minute=0, second=0, microsecond=0)
    if reference_time < boundary:
        boundary -= timedelta(days=1)
    return boundary


def _history_timestamp(path: Path) -> datetime | None:
    """Parse the timestamp prefix used by chat-history filenames."""

    match = _HISTORY_TIMESTAMP_PATTERN.match(path.stem)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group("timestamp"), "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def _marker_value(boundary: datetime) -> str:
    return boundary.isoformat(timespec="seconds")


def archive_old_chat_history_on_startup(
    history_root: str | Path | None = None,
    *,
    reference_time: datetime | None = None,
) -> ChatHistoryArchiveResult:
    """Move completed-memory-day JSON files to ``past_history`` once per day.

    Only files whose message days have ended and have reviewed diaries are
    eligible. Unknown, empty, current-day and unreviewed sources stay in place.
    A small marker in ``past_history``
    prevents repeated work when the backend is restarted during the same
    logical day.
    """

    root = resolve_character_history_root(history_root)
    now = reference_time or datetime.now()
    boundary = _latest_memory_boundary(now)
    archive_dir = root / "past_history"
    marker = archive_dir / _ARCHIVE_MARKER_FILENAME
    marker_text = _marker_value(boundary)

    eligible_sources: list[Path] = []
    for source in sorted(root.glob("*.json"), key=lambda path: path.name):
        if not source.is_file():
            continue

        timestamp = _history_timestamp(source)
        if timestamp is None:
            logger.warning(
                "[chat archive] unrecognised filename timestamp; keeping {}",
                source.name,
            )
            continue
        if timestamp >= boundary:
            continue
        try:
            days = history_days(source)
            if not days or any(day >= boundary.date() for day in days):
                continue
            if any(
                diary_approval_status(
                    root, day.isoformat(), root / "diaries" / f"diary_{day}.txt"
                )
                not in {"approved", "changed"}
                for day in days
            ):
                logger.info(
                    "[chat archive] keeping source with unreviewed days: {}",
                    source.name,
                )
                continue
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            logger.warning(
                "[chat archive] unreadable message dates; keeping {}", source.name
            )
            continue
        eligible_sources.append(source)

    try:
        if (
            marker.read_text(encoding="utf-8").strip() == marker_text
            and not eligible_sources
        ):
            logger.info(
                "[chat archive] already completed for memory day {}; skipping",
                boundary.date().isoformat(),
            )
            return ChatHistoryArchiveResult(
                boundary=boundary,
                skipped_due_to_marker=True,
            )
    except FileNotFoundError:
        pass
    except (OSError, UnicodeError) as exc:
        logger.warning(
            "[chat archive] could not read daily marker; checking files anyway: {}",
            exc,
        )

    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("[chat archive] could not create past_history: {}", exc)
        return ChatHistoryArchiveResult(boundary=boundary, failed_count=1)

    moved_count = 0
    failed_count = 0
    for source in eligible_sources:
        target = archive_dir / source.name
        if target.exists():
            logger.error(
                "[chat archive] destination exists; keeping source to avoid overwrite: {}",
                source.name,
            )
            failed_count += 1
            continue

        try:
            shutil.move(str(source), str(target))
        except (OSError, shutil.Error) as exc:
            logger.error(
                "[chat archive] move failed; keeping source {}: {}",
                source.name,
                exc,
            )
            failed_count += 1
        else:
            moved_count += 1

    if failed_count:
        logger.error(
            "[chat archive] {} old JSON file(s) could not be moved; completion marker not written",
            failed_count,
        )
        return ChatHistoryArchiveResult(
            boundary=boundary,
            moved_count=moved_count,
            failed_count=failed_count,
        )

    try:
        marker.write_text(marker_text + "\n", encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        logger.error(
            "[chat archive] files moved, but daily marker could not be written: {}",
            exc,
        )
        return ChatHistoryArchiveResult(
            boundary=boundary,
            moved_count=moved_count,
        )

    logger.info(
        "[chat archive] archived JSON files before {}: moved {} file(s)",
        boundary.isoformat(timespec="minutes"),
        moved_count,
    )
    return ChatHistoryArchiveResult(
        boundary=boundary,
        moved_count=moved_count,
        marker_written=True,
    )


__all__ = [
    "ChatHistoryArchiveResult",
    "archive_old_chat_history_on_startup",
]
