"""Check the final wire payload before any official request is dispatched."""

from collections import Counter
from dataclasses import dataclass
import json
import re

from ..memory.message_time_context import format_message_time_label
from .clean_view import BACKGROUND_PLACEHOLDER, HISTORY_NOTICE, fixed_protected_line
from .content_check import ContentChecker, fingerprint, normalized_lines


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    checked_lines: int
    blocked: dict[str, int]


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
    if line in (BACKGROUND_PLACEHOLDER, HISTORY_NOTICE) or fixed_protected_line(line):
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
    def __init__(self, checker: ContentChecker):
        self.checker = checker
        self._official_thinking: set[str] = set()

    def trust_official_thinking(self, block: dict) -> None:
        """Called only on blocks emitted by this trial's official client."""
        if block.get("type") not in ("thinking", "redacted_thinking"):
            raise ValueError("not_thinking_block")
        self._official_thinking.add(fingerprint(_serialized(block)))

    def _content(self, content, category, role=""):
        if isinstance(content, str):
            return [(category, content)]
        if not isinstance(content, list):
            return [(category, _serialized(content))]
        result = []
        for block in content:
            if not isinstance(block, dict):
                result.append((category, _serialized(block)))
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
                extra = {
                    k: v
                    for k, v in block.items()
                    if k not in ("type", "text", "cache_control")
                }
                if extra:
                    result.append((category, _serialized(extra)))
            elif kind == "tool_result":
                result.extend(
                    self._content(block.get("content", ""), "tool_result", role)
                )
                result.append(
                    (
                        "tool_result",
                        _serialized({k: v for k, v in block.items() if k != "content"}),
                    )
                )
            else:
                result.append((category, _serialized(block)))
        return result

    async def check(self, request: dict) -> GateResult:
        chunks = []
        if request.get("tools"):
            chunks.append(("tools", _serialized(request["tools"])))
        chunks.extend(self._content(request.get("system", ""), "system"))
        for message in request.get("messages", []):
            chunks.extend(
                self._content(
                    message.get("content", ""), "messages", message.get("role", "")
                )
            )
            extra = {k: v for k, v in message.items() if k not in ("role", "content")}
            if extra:
                chunks.append(("messages", _serialized(extra)))
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
            await self.checker.check_many(missing, kind="line")
        for category, line in lines:
            record = self.checker.cached(line, "line")
            if record is None or record.label != "A":
                blocked[category] += 1
        return GateResult(not blocked, len(lines), dict(blocked))

    async def require(self, request: dict) -> GateResult:
        result = await self.check(request)
        if not result.allowed:
            raise OfficialContentBlocked(result)
        return result
