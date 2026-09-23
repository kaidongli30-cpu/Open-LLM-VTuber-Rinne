"""Logical QQ chat sessions backed by Rinne's canonical history JSON files."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from ..chat_history_manager import create_new_history, update_metadate
from ..data_paths import character_history_root


PRIVATE_HISTORY_CHANNEL = "qq_private"
PRIVATE_HISTORY_VERSION = 1


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class PrivateHistorySessionManager:
    """Open, resume, and close one QQ-only history file for a text context."""

    def __init__(self, context: Any) -> None:
        self._context = context

    @property
    def conf_uid(self) -> str:
        character_config = getattr(self._context, "character_config", None)
        return str(getattr(character_config, "conf_uid", "") or "").strip()

    @property
    def history_uid(self) -> str:
        return str(getattr(self._context, "history_uid", "") or "").strip()

    def resume_latest_open(self) -> str:
        """Resume the newest unclosed QQ session after a backend restart."""

        conf_uid = self.conf_uid
        if not conf_uid:
            return ""
        root = character_history_root(conf_uid)
        if not root.is_dir():
            return ""
        try:
            candidates = sorted(
                root.glob("*.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return ""
        for path in candidates:
            metadata = self._read_metadata(path)
            if (
                metadata.get("source_channel") == PRIVATE_HISTORY_CHANNEL
                and metadata.get("session_status") == "open"
            ):
                self._context.history_uid = path.stem
                logger.info("[私人QQ桥接] 已恢复未结束的 QQ 聊天记录")
                return path.stem
        return ""

    def ensure_open(self) -> str:
        """Create a QQ history lazily when the first `over` is submitted."""

        current = self.history_uid
        if current:
            return current
        conf_uid = self.conf_uid
        if not conf_uid:
            raise RuntimeError("private QQ history has no character conf_uid")
        history_uid = create_new_history(conf_uid)
        if not history_uid:
            raise RuntimeError("private QQ history could not be created")
        if not update_metadate(
            conf_uid,
            history_uid,
            {
                "source_channel": PRIVATE_HISTORY_CHANNEL,
                "session_version": PRIVATE_HISTORY_VERSION,
                "session_status": "open",
                "started_at": _now_iso(),
            },
        ):
            raise RuntimeError("private QQ history metadata could not be written")
        self._context.history_uid = history_uid
        logger.info("[私人QQ桥接] 已开始新的 QQ 聊天记录")
        return history_uid

    def close(self, *, reason: str) -> bool:
        """Seal the current JSON; the next submitted turn opens a new one."""

        history_uid = self.history_uid
        conf_uid = self.conf_uid
        if not history_uid or not conf_uid:
            return False
        if not update_metadate(
            conf_uid,
            history_uid,
            {
                "session_status": "closed",
                "ended_at": _now_iso(),
                "end_reason": reason,
            },
        ):
            raise RuntimeError("private QQ history could not be closed")
        self._context.history_uid = ""
        logger.info("[私人QQ桥接] 当前 QQ 聊天记录已封存")
        return True

    @staticmethod
    def _read_metadata(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, list) or not payload:
            return {}
        first = payload[0]
        if not isinstance(first, dict) or first.get("role") != "metadata":
            return {}
        return first


__all__ = [
    "PRIVATE_HISTORY_CHANNEL",
    "PRIVATE_HISTORY_VERSION",
    "PrivateHistorySessionManager",
]
