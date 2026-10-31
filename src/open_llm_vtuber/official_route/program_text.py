"""Recognize exact program literals while retaining every unknown variable."""

import ast
import json
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
import re

from ..memory.message_time_context import (
    format_message_time_label,
    memory_day_for_timestamp,
)

PROGRAM_TEXT_VERSION = "official-program-templates-v3"

_NAMES = {
    "QQ_TEXTING_SYSTEM_INSTRUCTION",
    "SCENE_RESPONSE_INSTRUCTIONS",
    "TEMPORAL_SOURCE_INSTRUCTIONS",
    "FACTUAL_PRIORITY_INSTRUCTIONS",
    "FINAL_REPLY_ONLY_INSTRUCTION",
    "EXTERNAL_TOOL_BUDGET_EXHAUSTED_INSTRUCTION",
    "ORDINARY_EXTERNAL_TOOL_BUDGET_EXHAUSTED_INSTRUCTION",
    "SCENE_CONTINUITY_SYSTEM_INSTRUCTION",
    "SCENE_CONTEXT_HEADER",
    "BACKGROUND_PLACEHOLDER",
    "HISTORY_NOTICE",
    "BACKGROUND_PENDING",
    "HISTORY_PENDING",
    "_MP_GUIDANCE_START",
    "_MP_LATER_RULE",
    "RECENT_MEMORY_USAGE_RULES",
}
_HEADINGS = {
    "【Layer 2当前背景与场景状态】",
    "【历史系统信号】",
    "【当前场景记忆｜本轮必须先核对】",
    "<current_user_background>",
    "</current_user_background>",
    "【近三日已验收日记压缩版】",
    "【昨天的日记压缩版】",
    "【前天的日记压缩版】",
    "【大前天的日记压缩版】",
}
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


def _memory_tool_literals(root):
    """Read only explicitly selected, statically complete property returns.

    No tool instance is constructed and no return involving user/tool data is
    evaluated. The one supported substitution uses the module's exact default
    tool name; an altered instance name remains unknown text for classification.
    """
    values = set()
    selected = (
        (
            "memory/cloud_memory_tool.py",
            "CloudLongTermMemoryTool",
            {"decision_instructions", "followup_instructions"},
        ),
        (
            "memory/recent_diary_recall.py",
            "RecentDiaryRecallTool",
            {"followup_instructions"},
        ),
    )
    for relative, class_name, methods in selected:
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        tool_name = next(
            (
                node.value.value
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "TOOL_NAME"
                    for target in node.targets
                )
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ),
            None,
        )
        for cls in tree.body:
            if not isinstance(cls, ast.ClassDef) or cls.name != class_name:
                continue
            for method in cls.body:
                if (
                    not isinstance(method, ast.FunctionDef)
                    or method.name not in methods
                ):
                    continue
                for node in ast.walk(method):
                    if not isinstance(node, ast.Return):
                        continue
                    result = node.value
                    if isinstance(result, ast.Constant) and isinstance(
                        result.value, str
                    ):
                        if result.value:
                            values.add(result.value)
                    elif (
                        isinstance(result, ast.Call)
                        and not result.keywords
                        and len(result.args) == 2
                        and isinstance(result.func, ast.Attribute)
                        and result.func.attr == "replace"
                        and isinstance(result.func.value, ast.Constant)
                        and isinstance(result.func.value.value, str)
                        and isinstance(result.args[0], ast.Constant)
                        and isinstance(result.args[0].value, str)
                        and isinstance(result.args[1], ast.Attribute)
                        and result.args[1].attr == "name"
                        and isinstance(result.args[1].value, ast.Name)
                        and result.args[1].value.id == "self"
                        and tool_name is not None
                    ):
                        values.add(
                            result.func.value.value.replace(
                                result.args[0].value, tool_name
                            )
                        )
    return values


@lru_cache(maxsize=1)
def fixed_literals():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "agent/agents/basic_memory_agent.py",
        root / "agent_runtime/text_turn.py",
        root / "conversations/scene_continuity.py",
        root / "memory/scene_state.py",
        root / "rinne_renderer_profile.py",
        root / "memory/recent_memory_context.py",
        Path(__file__).with_name("clean_view.py"),
    ]
    values = set(_HEADINGS) | _memory_tool_literals(root)
    bindings = {
        "SCENE_ENVELOPE_OPEN": "<rinne_scene>",
        "SCENE_ENVELOPE_CLOSE": "</rinne_scene>",
    }
    for path in paths:
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if not isinstance(node, ast.Assign):
                continue
            names = [
                target.id for target in node.targets if isinstance(target, ast.Name)
            ]
            if not any(name in _NAMES for name in names):
                continue
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                values.add(value.value)
            elif isinstance(value, ast.JoinedStr):
                pieces = []
                for part in value.values:
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        pieces.append(part.value)
                    elif (
                        isinstance(part, ast.FormattedValue)
                        and isinstance(part.value, ast.Name)
                        and part.value.id in bindings
                    ):
                        pieces.append(bindings[part.value.id])
                    else:
                        break
                else:
                    values.add("".join(pieces))
    # An exact constant may appear one line at a time in the final request gate.
    return frozenset(
        values
        | {
            line.strip()
            for value in values
            for line in value.splitlines()
            if line.strip()
        }
    )


def fixed_time_line(text):
    if fixed_current_time_line(text):
        return True
    match = re.fullmatch(r"当前记忆日：([0-9]{4}-[0-9]{2}-[0-9]{2})（本地03:00分界）。", text)
    if match:
        try:
            return date.fromisoformat(match[1]).isoformat() == match[1]
        except ValueError:
            return False
    match = re.fullmatch(
        r"【消息时间：([^｜]+)｜记忆日：\d{4}-\d{2}-\d{2}（(\d{2}):00分界）】", text
    )
    if match:
        try:
            return (
                format_message_time_label(match[1], boundary_hour=int(match[2])) == text
            )
        except ValueError:
            return False
    return False


def fixed_current_time_line(text):
    """Validate the complete notification actually rendered by _to_messages.

    These variables contain only a valid date/time, weekday and derived memory
    day. No arbitrary text or leading/trailing suffix is covered by the rule.
    """
    match = re.fullmatch(
        r"\[当前时间：([0-9]{4})年([0-9]{2})月([0-9]{2})日 "
        r"([0-9]{2}):([0-9]{2})，(星期[一二三四五六日])；"
        r"当前记忆日：([0-9]{4}-[0-9]{2}-[0-9]{2})（凌晨03:00分界）。"
        r"这是当前的准确时间与记忆日，如果回复中涉及信息请务必遵守\]",
        text,
    )
    if match is None:
        return False
    try:
        parsed = datetime(*(int(value) for value in match.groups()[:5]))
    except ValueError:
        return False
    weekday = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")[
        parsed.weekday()
    ]
    try:
        memory_day = memory_day_for_timestamp(parsed).isoformat()
    except (ValueError, OverflowError):
        return False
    return match[6] == weekday and match[7] == memory_day


def fixed_line(text):
    text = text.strip()
    if text in fixed_literals() or fixed_time_line(text):
        return True
    # Import lazily: clean_view itself uses this module through the checker.
    from .clean_view import fixed_protected_line

    return fixed_protected_line(text)


def strip_program_text(text):
    """Remove exact literal spans, not arbitrary text starting with a heading."""
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        value = None
    if isinstance(value, dict) and set(value).issubset({"user", "assistant"}):
        cleaned = {
            key: strip_program_text(item) if isinstance(item, str) else item
            for key, item in value.items()
        }
        if all(isinstance(item, str) and not item.strip() for item in cleaned.values()):
            return ""
        return json.dumps(cleaned, ensure_ascii=False)
    result = text
    for literal in sorted(fixed_literals(), key=len, reverse=True):
        result = result.replace(literal, "")
    lines = []
    for line in result.splitlines():
        stripped = line.strip()
        if fixed_line(stripped):
            continue
        # Remove a single exact wrapper prefix. The remainder is always checked.
        prefix = next(
            (label for label in WRAPPER_LABELS if stripped.startswith(label)), ""
        )
        if not prefix:
            match = re.match(r"已存在的信息依据（消息[0-9]+）：", stripped)
            prefix = match[0] if match else ""
        if prefix:
            stripped = stripped[len(prefix) :].lstrip()
        role = next(
            (role for role in ("user：", "assistant：") if stripped.startswith(role)),
            "",
        )
        if role:
            stripped = stripped[len(role) :].lstrip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


def schema_texts(value):
    """Return *all* textual keys and values; JSON syntax alone needs no model.

    Used for tool schemas, arguments and result metadata. Unknown field names,
    enum/default/description strings and ids are included, never prefix-trusted.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in schema_texts(item)]
    if isinstance(value, dict):
        return [
            text
            for key, item in value.items()
            for text in [str(key), *schema_texts(item)]
        ]
    if value is None or type(value) in (bool, int, float):
        return []
    raise ValueError("unsupported_schema_value")
