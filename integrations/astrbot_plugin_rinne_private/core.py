"""Persistent strict-over input batching independent from AstrBot internals."""

from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable


STATE_VERSION = 2
SEEN_MESSAGE_LIMIT = 512


class PendingStoreError(RuntimeError):
    """Raised when private pending state cannot be loaded or persisted safely."""


class InputOutcomeKind(str, Enum):
    BUFFERED = "buffered"
    CANCELLED = "cancelled"
    DUPLICATE = "duplicate"
    EMPTY = "empty"
    STATUS = "status"
    SUBMITTED = "submitted"
    SESSION_ENDED = "session_ended"


@dataclass(frozen=True, slots=True)
class InputOutcome:
    kind: InputOutcomeKind
    reply_text: str = ""
    pending_count: int = 0


@dataclass(slots=True)
class _SenderState:
    items: list[dict[str, Any]]
    seen_message_ids: deque[str]


class AtomicPendingStore:
    """Atomically persist pending private turns without logging their content."""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root).expanduser()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.path = self.data_root / "pending_turns.json"
        self._senders: dict[str, _SenderState] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PendingStoreError("pending state is unreadable") from exc
        if not isinstance(payload, dict) or payload.get("version") not in {1, STATE_VERSION}:
            raise PendingStoreError("pending state version is invalid")
        state_version = int(payload["version"])
        raw_senders = payload.get("senders")
        if not isinstance(raw_senders, dict):
            raise PendingStoreError("pending sender state is invalid")
        loaded: dict[str, _SenderState] = {}
        for sender_id, raw in raw_senders.items():
            if not isinstance(sender_id, str) or not isinstance(raw, dict):
                raise PendingStoreError("pending sender entry is invalid")
            items = raw.get("items", [])
            seen = raw.get("seen_message_ids", [])
            if not isinstance(items, list):
                raise PendingStoreError("pending items are invalid")
            if state_version == 1:
                if not all(isinstance(item, str) for item in items):
                    raise PendingStoreError("pending legacy text items are invalid")
                items = [
                    {"parts": [{"type": "text", "text": item}]}
                    for item in items
                ]
            if not all(_valid_message_item(item) for item in items):
                raise PendingStoreError("pending message items are invalid")
            if not isinstance(seen, list) or not all(
                isinstance(item, str) for item in seen
            ):
                raise PendingStoreError("pending message IDs are invalid")
            loaded[sender_id] = _SenderState(
                items=list(items),
                seen_message_ids=deque(seen[-SEEN_MESSAGE_LIMIT:], maxlen=SEEN_MESSAGE_LIMIT),
            )
        self._senders = loaded

    def _state_for(self, sender_id: str) -> _SenderState:
        return self._senders.setdefault(
            sender_id,
            _SenderState(items=[], seen_message_ids=deque(maxlen=SEEN_MESSAGE_LIMIT)),
        )

    def snapshot(self, sender_id: str) -> tuple[dict[str, Any], ...]:
        return tuple(self._state_for(sender_id).items)

    def has_seen(self, sender_id: str, message_id: str) -> bool:
        return message_id in self._state_for(sender_id).seen_message_ids

    def append(
        self,
        sender_id: str,
        message_id: str,
        parts: list[dict[str, Any]] | str,
    ) -> None:
        state = self._state_for(sender_id)
        if isinstance(parts, str):
            parts = [{"type": "text", "text": parts}]
        state.items.append({"parts": parts})
        state.seen_message_ids.append(message_id)
        self._save()

    def mark_seen(self, sender_id: str, message_id: str) -> None:
        state = self._state_for(sender_id)
        state.seen_message_ids.append(message_id)
        self._save()

    def clear_items(self, sender_id: str) -> None:
        self._state_for(sender_id).items.clear()
        self._save()

    def _save(self) -> None:
        payload = {
            "version": STATE_VERSION,
            "senders": {
                sender_id: {
                    "items": list(state.items),
                    "seen_message_ids": list(state.seen_message_ids),
                }
                for sender_id, state in self._senders.items()
            },
        }
        temp_path = self.path.with_suffix(".json.tmp")
        try:
            with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        except OSError as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise PendingStoreError("pending state could not be persisted") from exc


def _valid_message_item(item: object) -> bool:
    if not isinstance(item, dict) or set(item) != {"parts"}:
        return False
    parts = item.get("parts")
    if not isinstance(parts, list) or not parts:
        return False
    for part in parts:
        if not isinstance(part, dict):
            return False
        part_type = part.get("type")
        if part_type in {"text", "audio_text"}:
            if set(part) != {"type", "text"} or not isinstance(part.get("text"), str):
                return False
        elif part_type in {"image", "document", "video"}:
            required = {"type", "name", "relative_path", "size", "sha256"}
            if set(part) != required:
                return False
            if not all(isinstance(part.get(key), str) for key in required - {"size"}):
                return False
            if not isinstance(part.get("size"), int):
                return False
        else:
            return False
    return True


def build_submission(
    items: tuple[dict[str, Any], ...],
) -> tuple[str, tuple[dict[str, Any], ...]]:
    """Build one channel-neutral user message without treating `over` as input."""

    messages: list[str] = []
    media: list[dict[str, Any]] = []
    for item in items:
        parts: list[str] = []
        for part in item["parts"]:
            if part["type"] == "text":
                parts.append(part["text"])
            elif part["type"] == "audio_text":
                parts.append(part["text"])
            else:
                label = {
                    "image": "图片",
                    "document": "文件",
                    "video": "视频",
                }[part["type"]]
                parts.append(f"[{label}：{part['name']}]")
                media.append(
                    {
                        "kind": part["type"],
                        "name": part["name"],
                        "relative_path": part["relative_path"],
                        "size": part["size"],
                        "sha256": part["sha256"],
                    }
                )
        messages.append("".join(parts))
    return "".join(messages), tuple(media)


def build_submission_text(items: tuple[str, ...]) -> str:
    """Compatibility helper used by existing callers and tests."""

    structured = tuple(
        {"parts": [{"type": "text", "text": item}]} for item in items
    )
    return build_submission(structured)[0]


Submitter = Callable[[str, str, str, tuple[dict[str, Any], ...]], AsyncIterator[str]]
SentenceDeliverer = Callable[[str], Awaitable[None]]
SessionEnder = Callable[[str, str], Awaitable[bool]]
StatusProvider = Callable[[str], Awaitable[str]]


class StrictOverController:
    """Buffer every private message until a standalone `over` is received."""

    def __init__(
        self,
        *,
        store: AtomicPendingStore,
        max_pending_messages: int = 64,
        max_pending_chars: int = 45_000,
    ) -> None:
        if max_pending_messages < 1 or max_pending_chars < 1:
            raise ValueError("pending limits must be positive")
        self._store = store
        self._max_pending_messages = max_pending_messages
        self._max_pending_chars = max_pending_chars
        self._locks: dict[str, asyncio.Lock] = {}

    async def handle_text(
        self,
        *,
        sender_id: str,
        message_id: str,
        text: str,
        submitter: Submitter,
        deliver_sentence: SentenceDeliverer,
        end_session: SessionEnder | None = None,
        status_provider: StatusProvider | None = None,
    ) -> InputOutcome:
        normalized = text.strip()
        parts = [{"type": "text", "text": normalized}] if normalized else []

        async def legacy_submitter(sender, message, submission, _media):
            async for sentence in submitter(sender, message, submission):  # type: ignore[call-arg]
                yield sentence

        return await self.handle_message(
            sender_id=sender_id,
            message_id=message_id,
            parts=parts,
            submitter=legacy_submitter,
            deliver_sentence=deliver_sentence,
            end_session=end_session,
            status_provider=status_provider,
        )

    async def handle_message(
        self,
        *,
        sender_id: str,
        message_id: str,
        parts: list[dict[str, Any]],
        submitter: Submitter,
        deliver_sentence: SentenceDeliverer,
        end_session: SessionEnder | None = None,
        status_provider: StatusProvider | None = None,
    ) -> InputOutcome:
        lock = self._locks.setdefault(sender_id, asyncio.Lock())
        async with lock:
            if self._store.has_seen(sender_id, message_id):
                return InputOutcome(InputOutcomeKind.DUPLICATE)

            if parts and not _valid_message_item({"parts": parts}):
                raise ValueError("message parts are invalid")
            normalized = ""
            if len(parts) == 1 and parts[0].get("type") == "text":
                normalized = str(parts[0].get("text") or "").strip()
            command = normalized.casefold()
            if command == "/cancel":
                self._store.mark_seen(sender_id, message_id)
                self._store.clear_items(sender_id)
                return InputOutcome(
                    InputOutcomeKind.CANCELLED,
                    reply_text="已经清空刚才还没提交的内容。",
                )
            if command == "/status":
                self._store.mark_seen(sender_id, message_id)
                pending = self._store.snapshot(sender_id)
                health_text = (
                    await status_provider(sender_id)
                    if status_provider is not None
                    else "凛祢本地健康检查尚未配置。"
                )
                return InputOutcome(
                    InputOutcomeKind.STATUS,
                    reply_text=(
                        f"{health_text}\n"
                        f"当前缓存：{len(pending)} 条，等待你发送 over。"
                    ),
                    pending_count=len(pending),
                )
            if command == "/end":
                pending = self._store.snapshot(sender_id)
                if pending:
                    self._store.mark_seen(sender_id, message_id)
                    return InputOutcome(
                        InputOutcomeKind.STATUS,
                        reply_text=(
                            f"还有 {len(pending)} 条内容没有提交。请先发送 over "
                            "或 /cancel，再发送 /end。"
                        ),
                        pending_count=len(pending),
                    )
                if end_session is None:
                    raise RuntimeError("QQ session end callback is not configured")
                closed = await end_session(sender_id, message_id)
                self._store.mark_seen(sender_id, message_id)
                if closed:
                    return InputOutcome(
                        InputOutcomeKind.SESSION_ENDED,
                        reply_text=(
                            "这次聊天已经保存好了。下一次发送 over 时会开始一份新记录。"
                        ),
                    )
                return InputOutcome(
                    InputOutcomeKind.SESSION_ENDED,
                    reply_text=(
                        "现在没有正在记录的聊天。下一次发送 over 时会开始一份新记录。"
                    ),
                )
            if command == "over":
                self._store.mark_seen(sender_id, message_id)
                pending = self._store.snapshot(sender_id)
                if not pending:
                    return InputOutcome(
                        InputOutcomeKind.EMPTY,
                        reply_text="还没有收到要提交的内容。",
                    )
                submission, media = build_submission(pending)
                async for sentence in submitter(
                    sender_id, message_id, submission, media
                ):
                    if sentence.strip():
                        await deliver_sentence(sentence.strip())
                self._store.clear_items(sender_id)
                return InputOutcome(
                    InputOutcomeKind.SUBMITTED,
                    pending_count=len(pending),
                )

            if not parts:
                self._store.mark_seen(sender_id, message_id)
                return InputOutcome(InputOutcomeKind.EMPTY)

            pending = self._store.snapshot(sender_id)
            pending_chars = sum(
                len(str(part.get("text") or ""))
                for item in pending
                for part in item["parts"]
                if part.get("type") in {"text", "audio_text"}
            )
            if len(pending) >= self._max_pending_messages:
                self._store.mark_seen(sender_id, message_id)
                return InputOutcome(
                    InputOutcomeKind.STATUS,
                    reply_text="缓存的消息已经到上限，请发送 over 或 /cancel。",
                    pending_count=len(pending),
                )
            incoming_chars = sum(
                len(str(part.get("text") or ""))
                for part in parts
                if part.get("type") in {"text", "audio_text"}
            )
            if pending_chars + incoming_chars > self._max_pending_chars:
                self._store.mark_seen(sender_id, message_id)
                return InputOutcome(
                    InputOutcomeKind.STATUS,
                    reply_text="缓存的文字已经到上限，请发送 over 或 /cancel。",
                    pending_count=len(pending),
                )
            self._store.append(sender_id, message_id, parts)
            return InputOutcome(
                InputOutcomeKind.BUFFERED,
                pending_count=len(pending) + 1,
            )
