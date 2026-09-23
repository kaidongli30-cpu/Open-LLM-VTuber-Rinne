"""Privacy-safe evidence of recent QQ transport activity."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path


class TransportHealthStore:
    """Persist timestamps only; never copy message text or account identifiers."""

    def __init__(self, data_root: Path) -> None:
        self.path = data_root / "runtime" / "transport_health.json"
        self._lock = threading.RLock()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load(self) -> dict[str, object]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return {"version": 1}
        if not isinstance(payload, dict) or payload.get("version") != 1:
            return {"version": 1}
        return payload

    def _write(self, payload: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def record_inbound(self) -> None:
        with self._lock:
            payload = self._load()
            now = self._now()
            payload.update({"updated_at": now, "last_inbound_at": now})
            self._write(payload)

    def record_reply_submitted(self) -> None:
        with self._lock:
            payload = self._load()
            now = self._now()
            payload.update({"updated_at": now, "last_reply_submitted_at": now})
            self._write(payload)

    def record_remote_probe_success(self) -> None:
        """Record a silent OneBot request that reached the QQ service."""

        with self._lock:
            payload = self._load()
            now = self._now()
            payload.update(
                {
                    "updated_at": now,
                    "last_probe_at": now,
                    "last_probe_ok_at": now,
                    "consecutive_probe_failures": 0,
                    "probe_state": "ok",
                    "probe_error": "",
                }
            )
            self._write(payload)

    def record_remote_probe_failure(self, error_code: str) -> None:
        """Record a probe failure without persisting exceptions or QQ data."""

        safe_error = (
            error_code
            if error_code
            in {
                "adapter_unavailable",
                "timeout",
                "connection",
                "action_failed",
                "invalid_response",
                "cancelled",
            }
            else "action_failed"
        )
        with self._lock:
            payload = self._load()
            failures = payload.get("consecutive_probe_failures", 0)
            if not isinstance(failures, int) or isinstance(failures, bool):
                failures = 0
            now = self._now()
            payload.update(
                {
                    "updated_at": now,
                    "last_probe_at": now,
                    "last_probe_failed_at": now,
                    "consecutive_probe_failures": min(failures + 1, 1_000_000),
                    "probe_state": "failed",
                    "probe_error": safe_error,
                }
            )
            self._write(payload)
