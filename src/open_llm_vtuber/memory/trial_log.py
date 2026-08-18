"""Private JSONL diagnostics for the live memory-system trial period."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


_WRITE_LOCK = threading.Lock()


def write_memory_trial_record(
    history_root: str | Path,
    record: dict[str, Any],
) -> Path:
    """Append one complete turn without writing private content to Git paths."""

    root = Path(history_root).resolve()
    log_dir = (root / "memory_trial_logs").resolve()
    if not log_dir.is_relative_to(root):
        raise ValueError("memory trial log directory escapes history root")
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"memory_trial_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(line)
    return path


__all__ = ["write_memory_trial_record"]
