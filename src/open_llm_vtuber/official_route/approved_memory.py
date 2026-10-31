"""An exact, locally reviewed memory view used only by the official route."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

from loguru import logger

_SCOPES = {"layer2", "recent", "rules"}
_RECENT_HEADER = re.compile(r"【(?:昨天|前天|大前天)的日记压缩版】\n")


def _normalize(text):
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _sha(text):
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


class ApprovedMemoryView:
    """Keep approvals separate from classifier caches and original memories.

    Only exact reviewed text and its exact lines are approved. Edited or new
    memory text follows the ordinary checks. No approvals are copied into the
    persistent model cache, so reloading without this file removes the exceptions.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.approved_at = ""
        self.status = "missing"
        self._entries = {}
        self._approved = set()
        self._recent = set()
        try:
            if not self.path.exists():
                return
            if self.path.is_symlink() or self.path.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("invalid_memory_view_file")
            raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
            if raw.get("schema_version") != 1 or not isinstance(
                raw.get("entries"), list
            ):
                raise ValueError("invalid_memory_view_schema")
            datetime.fromisoformat(raw["approved_at"])
            entries, approved, recent = {}, set(), set()
            for row in raw["entries"]:
                scope, source, text = row["scope"], row["source_sha256"], row["text"]
                if (
                    scope not in _SCOPES
                    or not isinstance(source, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", source)
                    or not isinstance(text, str)
                    or not text.strip()
                    or row["text_sha256"] != _sha(text)
                ):
                    raise ValueError("invalid_memory_view_entry")
                target = _normalize(text)
                key = (scope, source)
                if key in entries and entries[key] != target:
                    raise ValueError("conflicting_memory_view_entries")
                entries[key] = target
                approved.add(_sha(target))
                approved.update(
                    _sha(line) for line in target.splitlines() if line.strip()
                )
                if scope == "recent":
                    recent.add(_sha(target))
            if recent:
                approved.update(
                    _sha(f"【{label}的日记压缩版】")
                    for label in ("昨天", "前天", "大前天")
                )
            self._entries, self._approved, self._recent = entries, approved, recent
            self.approved_at = raw["approved_at"]
            self.status = "loaded"
        except (OSError, UnicodeError, ValueError, KeyError, TypeError):
            self.status = "invalid"
            logger.warning("官方专用记忆版本无效，继续使用原检查；未改原始记忆")

    @property
    def enabled(self):
        return self.status == "loaded" and bool(self._entries)

    def approved(self, text):
        if not self.enabled:
            return False
        text = _normalize(text)
        if _sha(text) in self._approved:
            return True
        header = _RECENT_HEADER.match(text)
        return bool(header and _sha(text[header.end() :]) in self._recent)

    def project(self, text, scope, parts):
        """Substitute only the exact source units that were reviewed."""
        if not self.enabled:
            return text
        if "".join(part["text"] for part in parts) != text:
            raise ValueError("memory_view_parts_mismatch")
        output = []
        for part in parts:
            value = part["text"]
            target = (
                None if part["protected"] else self._entries.get((scope, _sha(value)))
            )
            if target is not None:
                leading = value[: len(value) - len(value.lstrip())]
                trailing = value[len(value.rstrip()) :]
                output.append(leading + target + trailing)
            else:
                output.append(value)
        return "".join(output)
