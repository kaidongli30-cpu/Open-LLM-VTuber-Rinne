"""Check the final wire payload before any official request is dispatched."""

from collections import Counter
from dataclasses import dataclass
import json
import re

from ..memory.message_time_context import format_message_time_label
from .clean_view import (
    BACKGROUND_PLACEHOLDER,
    HISTORY_NOTICE,
    BACKGROUND_PENDING,
    HISTORY_PENDING,
    fixed_protected_line,
)
from .checks_deadline import check_deadline, check_scope, is_pending
from .content_check import ContentChecker, fingerprint, normalized_lines
from .program_text import fixed_current_time_line


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    checked_lines: int
    blocked: dict[str, int]
    pending: bool = False
    missing_lines: int = 0


class OfficialContentBlocked(RuntimeError):
    def __init__(self, result: GateResult):
        self.result = result
        super().__init__("official_request_blocked")


WRAPPER_LABELS = (
    "较早已经说过：",
    "草稿中的重复句：",
    "草稿中重复的问题：",
    "旧问题之后的对话节选：",
    "重复原因：",
    "整段草稿：",
    "同一聊天记录里已经问过：",
    "随后对话（用于核对是否已有回答）：",
)
WRAPPER_EVIDENCE_PREFIX = "已存在的信息依据（消息"
WRAPPER_ROLES = ("user：", "assistant：")


def split_wrapper(line):
    """Recognize only one exact leading label and one exact role prefix."""
    rest = line.strip()
    label = next((p for p in WRAPPER_LABELS if rest.startswith(p)), "")
    if not label:
        match = re.match(r"已存在的信息依据（消息[0-9]+）：", rest)
        label = match[0] if match else ""
    if label:
        rest = rest[len(label) :].lstrip()
    role = next((p for p in WRAPPER_ROLES if rest.startswith(p)), "")
    if role:
        rest = rest[len(role) :].lstrip()
    return label, role, rest.strip()


def _serialized(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _fixed(line):
    if fixed_current_time_line(line):
        return True
    if line in (
        BACKGROUND_PLACEHOLDER,
        HISTORY_NOTICE,
        BACKGROUND_PENDING,
        HISTORY_PENDING,
    ) or fixed_protected_line(line):
        return True
    match = re.fullmatch(
        r"【消息时间：([^｜]+)｜记忆日：\d{4}-\d{2}-\d{2}（(\d{2}):00分界）】", line
    )
    if not match:
        return False
    try:
        return format_message_time_label(match[1], boundary_hour=int(match[2])) == line
    except ValueError:
        return False


class RequestGate:
    def __init__(self, checker: ContentChecker, *, model=None):
        self.checker = checker
        self._official_thinking: set[str] = set()
        self.model = model
        self.cache_only = False
        self._tool_specs = {}

    def _metadata(self, value):
        unknown = {}
        for key, item in value.items():
            if key == "model" and item == self.model:
                from .model_profiles import configuration_issue

                if configuration_issue({"model": item}) is None:
                    continue
            if key == "max_tokens" and type(item) is int and item > 0:
                continue
            if key in ("thinking", "extra_body"):
                from .model_profiles import request_options

                options = [
                    request_options(self.model, flag)
                    .get("extra_body", {})
                    .get("thinking")
                    for flag in (False, True)
                ]
                options = [value for value in options if value is not None]
                if (
                    item in options
                    if key == "thinking"
                    else item in [{"thinking": x} for x in options]
                ):
                    continue
            if key == "tool_choice" and isinstance(item, dict):
                if item == {"type": "auto"}:
                    continue
                if (
                    set(item) == {"type", "name"}
                    and item["type"] == "tool"
                    and item["name"] in self._tool_specs
                ):
                    continue
            unknown[key] = item
        return unknown

    def _block_metadata(self, block, kind):
        unknown = {}
        spec = self._tool_specs.get(block.get("name")) or {}
        properties = spec.get("input_schema", {}).get("properties", {})
        for key, item in block.items():
            if key in ("text", "content"):
                continue
            if (
                key == "type"
                and item == kind
                and kind in ("text", "tool_use", "tool_result")
            ):
                continue
            if key == "cache_control" and item == {"type": "ephemeral"}:
                continue
            if (
                key in ("id", "tool_use_id")
                and isinstance(item, str)
                and re.fullmatch(r"toolu_[A-Za-z0-9_-]{1,128}", item)
            ):
                continue
            if kind == "tool_use" and key == "name" and item in self._tool_specs:
                continue
            if kind == "tool_use" and key == "input" and isinstance(item, dict):
                # Parameter names come from our checked schema. Every textual
                # value still enters the classifier, including nested values.
                unknown[key] = {
                    name: value
                    for name, value in item.items()
                    if name not in properties
                }
                if not unknown[key]:
                    unknown.pop(key)
                for name, value in item.items():
                    if name in properties:
                        unknown["argument:" + name] = value
                continue
            unknown[key] = item
        return unknown

    def _structured(self, value, category):
        if getattr(self.checker, "runtime_two_pass", False):
            return [(category, text) for text in self.checker.schema_texts(value)]
        return [(category, _serialized(value))]

    def trust_official_thinking(self, block: dict) -> None:
        """Called only on blocks emitted by this trial's official client."""
        if block.get("type") not in ("thinking", "redacted_thinking"):
            raise ValueError("not_thinking_block")
        self._official_thinking.add(fingerprint(_serialized(block)))

    def _content(self, content, category, role=""):
        if isinstance(content, str):
            return [(category, content)]
        if not isinstance(content, list):
            return self._structured(content, category)
        result = []
        for block in content:
            if not isinstance(block, dict):
                result.extend(self._structured(block, category))
                continue
            kind = block.get("type")
            if kind in ("thinking", "redacted_thinking"):
                if (
                    role == "assistant"
                    and fingerprint(_serialized(block)) in self._official_thinking
                ):
                    continue
                result.append(("untrusted_thinking", ""))
            elif kind in ("image", "image_url", "video", "document"):
                result.append(("unsupported_media", ""))
            elif kind == "text":
                result.append((category, str(block.get("text", ""))))
                extra = self._block_metadata(block, kind)
                if extra:
                    result.extend(self._structured(extra, category))
            elif kind == "tool_result":
                result.extend(
                    self._content(block.get("content", ""), "tool_result", role)
                )
                result.extend(
                    self._structured(
                        self._block_metadata(block, kind),
                        "tool_result",
                    )
                )
            elif kind == "tool_use":
                for key, value in self._block_metadata(block, kind).items():
                    if key.startswith("argument:"):
                        result.extend(self._structured(value, "chat_content"))
                    else:
                        result.extend(self._structured({key: value}, "protocol_fields"))
            else:
                result.extend(self._structured(block, category))
        return result

    async def check(
        self, request: dict, *, timeout_seconds=None, deadline=None, cached_only=False
    ) -> GateResult:
        if timeout_seconds is not None or deadline is not None:
            with check_deadline(
                20 if timeout_seconds is None else timeout_seconds, deadline=deadline
            ):
                return await self.check(request, cached_only=cached_only)
        chunks = []
        if request.get("tools"):
            if getattr(self.checker, "runtime_two_pass", False):
                ready = self.checker.schema_ready(request["tools"])
                if ready:
                    self._tool_specs = {t["name"]: t for t in request["tools"]}
                else:
                    records = [
                        self.checker.cached(text, "line")
                        for text in self.checker.schema_texts(request["tools"])
                    ]
                    blocked = any(not is_pending(r) and r.label != "A" for r in records)
                    return GateResult(
                        False,
                        0,
                        {"tools_blocked" if blocked else "tools_unrecorded": 1},
                        not blocked,
                        0 if blocked else 1,
                    )
            else:
                chunks.append(("tools", _serialized(request["tools"])))
        chunks.extend(self._content(request.get("system", ""), "system"))
        for message in request.get("messages", []):
            if message.get("role") not in ("user", "assistant"):
                chunks.extend(
                    self._structured({"role": message.get("role")}, "protocol_fields")
                )
            chunks.extend(
                self._content(
                    message.get("content", ""), "messages", message.get("role", "")
                )
            )
            extra = {k: v for k, v in message.items() if k not in ("role", "content")}
            if extra:
                chunks.extend(self._structured(extra, "messages"))
        if getattr(self.checker, "runtime_two_pass", False):
            extras = self._metadata(
                {
                    key: value
                    for key, value in request.items()
                    if key not in {"system", "messages", "tools"}
                }
            )
            chunks.extend(self._structured(extras, "request_metadata"))
        blocked = Counter()
        lines = []
        for category, text in chunks:
            if category in ("untrusted_thinking", "unsupported_media"):
                blocked[category] += 1
                continue
            for line in normalized_lines(text):
                lines.append((category, line))
                if self.checker.keyword_hit(line):
                    blocked[category] += 1
        # Scan all locations first; a keyword hit must not initiate a model call.
        if blocked:
            return GateResult(False, len(lines), dict(blocked))
        unique = list(dict.fromkeys(line for _, line in lines))
        missing = []
        for line in unique:
            if self.checker.cached(line, "line") is None:
                if _fixed(line):
                    self.checker.record_fixed_line(line)
                else:
                    label, role, rest = split_wrapper(line)
                    record = self.checker.cached(rest, "line") if rest else None
                    if (
                        (label or role)
                        and rest
                        and (
                            record.label == "A" if record is not None else _fixed(rest)
                        )
                    ):
                        self.checker._remember(
                            line,
                            "line",
                            (),
                            "A",
                            100,
                            "wrapper_composed",
                            "固定标签＋已知A内容",
                        )
                    else:
                        missing.append(line)
        if missing:
            if not cached_only:
                # Keep origins separate for metadata-only replay auditing.
                for origin in dict.fromkeys(
                    category for category, line in lines if line in missing
                ):
                    texts = list(
                        dict.fromkeys(
                            line
                            for category, line in lines
                            if category == origin and line in missing
                        )
                    )
                    category = (
                        "protocol_fields"
                        if origin in ("request_metadata", "protocol_fields")
                        else "tool_results"
                        if origin == "tool_result"
                        else "chat_content"
                    )
                    with check_scope(category):
                        await self.checker.check_many(texts, kind="line")
        pending = 0
        for category, line in lines:
            record = self.checker.cached(line, "line")
            if is_pending(record):
                pending += 1
            if is_pending(record) or record.label != "A":
                blocked[category] += 1
        return GateResult(
            not blocked, len(lines), dict(blocked), bool(pending), pending
        )

    async def require(self, request: dict) -> GateResult:
        result = await self.check(request, cached_only=self.cache_only)
        if not result.allowed:
            raise OfficialContentBlocked(result)
        return result
