"""Read-only discovery of real chat days, including previously archived sources."""

import json
from datetime import date, datetime, timedelta
from pathlib import Path


def history_files(root: Path):
    """Prefer the live copy if an archive filename also exists."""
    live = {path.name: path for path in root.glob("*.json")}
    archived = {path.name: path for path in (root / "past_history").glob("*.json")}
    return sorted((archived | live).values(), key=lambda path: path.name)


def history_days(path: Path) -> set[date]:
    """Reject ambiguous records so archival never guesses their memory day."""
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("history_not_list")
    days = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("history_record_not_object")
        if record.get("role") == "metadata":
            continue
        stamp = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
        if stamp.tzinfo is not None:
            stamp = stamp.astimezone().replace(tzinfo=None)
        days.add((stamp - timedelta(hours=3)).date())
    return days


def pending_chat_diary_days(root: Path, latest: date) -> list[date]:
    """Backfill real older chat days, never invent intervening empty days."""
    from .diary_review import diary_approval_status

    pending = set()
    for path in history_files(root):
        try:
            days = history_days(path)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue  # The archive guard independently keeps unreadable sources.
        for day in days:
            if day >= latest:
                continue
            diary = root / "diaries" / f"diary_{day.isoformat()}.txt"
            status = diary_approval_status(root, day.isoformat(), diary)
            if status == "missing" or (
                path.parent == root and status not in {"approved", "changed"}
            ):
                pending.add(day)
    return sorted(pending)
