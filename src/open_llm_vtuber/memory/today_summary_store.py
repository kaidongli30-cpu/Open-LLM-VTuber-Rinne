"""Private, versioned summary snapshots and exact source-coverage checks."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .diary_sources import history_files
from .message_time_context import memory_day_for_timestamp, parse_message_timestamp
from .today_life_summary import (
    SUMMARY_PROMPT_VERSION,
    SUMMARY_SCHEMA_VERSION,
    SummaryValidationError,
    canonical_messages,
    source_digest,
    validate_summary_data,
)


def load_day_messages(root: Path, memory_day: str) -> list[dict[str, str]]:
    target = date.fromisoformat(memory_day)
    ordered = []
    for path in history_files(Path(root)):
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SummaryValidationError("unreadable_history_source") from exc
        if not isinstance(records, list):
            raise SummaryValidationError("invalid_history_source")
        for record in records:
            if not isinstance(record, dict) or record.get("role") not in {
                "human",
                "ai",
                "user",
                "assistant",
            }:
                continue
            stamp = parse_message_timestamp(record.get("timestamp"))
            if stamp is None:
                raise SummaryValidationError("undated_history_source")
            if memory_day_for_timestamp(stamp) == target:
                ordered.append((stamp, len(ordered), record))
    ordered.sort(key=lambda item: (item[0], item[1]))
    return canonical_messages([record for _, _, record in ordered])


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".summary-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@contextmanager
def summary_lock(root: Path, memory_day: str):
    """OS-owned lock; a crashed worker does not leave a stale ownership lock."""
    date.fromisoformat(memory_day)
    path = Path(root) / "today_summaries" / "locks" / f"{memory_day}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class TodaySummaryStore:
    def __init__(self, history_root: Path | str):
        self.root = Path(history_root) / "today_summaries"

    def latest_path(self, memory_day: str) -> Path:
        date.fromisoformat(memory_day)
        return self.root / f"{memory_day}.latest.json"

    def save(
        self,
        day: str,
        messages: list[dict[str, str]],
        summary: dict[str, Any],
        *,
        turns: int,
        model: str,
        seconds: float,
    ) -> dict[str, Any]:
        date.fromisoformat(day)
        checked = validate_summary_data(summary, messages)
        # A fresh snapshot protects previous generations and user inspection edits.
        stem = f"summary-{day}-{turns:06d}-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)
        text_path = self.root / (stem + ".txt")
        text_path.write_text(checked["text"], encoding="utf-8")
        snapshot = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "prompt_version": SUMMARY_PROMPT_VERSION,
            "memory_day": day,
            "covered_messages": len(messages),
            "covered_turns": turns,
            "source_sha256": source_digest(messages),
            "summary": checked,
            "text_file": text_path.name,
            "model": model,
            "seconds": seconds,
            "created_at": datetime.now().astimezone().isoformat(),
        }
        atomic_json(self.root / (stem + ".json"), snapshot)
        # Publish only after both immutable files have been completed.
        atomic_json(self.latest_path(day), {"snapshot": stem + ".json"})
        return snapshot

    def load(self, day: str, messages: list[dict[str, str]]) -> dict[str, Any] | None:
        try:
            pointer = json.loads(self.latest_path(day).read_text(encoding="utf-8"))
            name = pointer["snapshot"]
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not name.startswith(f"summary-{day}-")
            ):
                return None
            snapshot = json.loads((self.root / name).read_text(encoding="utf-8"))
            count = snapshot["covered_messages"]
            if (
                snapshot["schema_version"] != SUMMARY_SCHEMA_VERSION
                or snapshot["prompt_version"] != SUMMARY_PROMPT_VERSION
                or snapshot["memory_day"] != day
                or type(count) is not int
                or not 0 < count <= len(messages)
                or snapshot["source_sha256"] != source_digest(messages[:count])
            ):
                return None
            checked = validate_summary_data(snapshot["summary"], messages[:count])
            text_name = snapshot["text_file"]
            if Path(text_name).name != text_name or text_name != name[:-5] + ".txt":
                return None
            if (self.root / text_name).read_text(encoding="utf-8") != checked["text"]:
                return None
            if snapshot["summary"]["text"] != checked["text"]:
                return None
            return snapshot
        except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError):
            return None

    def previous_attempt(self, day: str) -> dict[str, Any]:
        try:
            result = json.loads(
                (self.root / f"{day}.attempt.json").read_text(encoding="utf-8")
            )
            return result if isinstance(result, dict) else {}
        except (OSError, ValueError):
            return {}

    def record_attempt(self, day: str, value: dict[str, Any]) -> None:
        date.fromisoformat(day)
        atomic_json(self.root / f"{day}.attempt.json", value)


def diary_summary_material(
    root: Path, day: str, records: list[dict[str, Any]], *, max_tail_turns: int = 10
) -> dict[str, Any] | None:
    """Return same-day summary plus only uncovered originals, or fall back."""
    messages = canonical_messages(records)
    snapshot = TodaySummaryStore(root).load(day, messages)
    if snapshot is None:
        return None
    tail = messages[snapshot["covered_messages"] :]
    if sum(message["role"] == "user" for message in tail) > max_tail_turns:
        return None
    if len(tail) > max_tail_turns * 4:
        return None
    return {
        "memory_day": day,
        "covered_messages": snapshot["covered_messages"],
        "summary": snapshot["summary"]["text"],
        "tail": tail,
    }
