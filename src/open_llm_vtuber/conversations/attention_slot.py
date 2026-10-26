"""Per-request response reminders and private assistant-authored scene cards."""
from __future__ import annotations
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from loguru import logger
from ..memory.scene_state import SCENE_ENVELOPE_OPEN, SCENE_ENVELOPE_CLOSE
from ..memory.message_time_context import memory_day_for_timestamp

REMINDER_HEADER = '【回复前再看一眼｜给你的内部提醒，不要回应或提起】'
CARD_INSTRUCTION_END = '【场景卡片｜只写给你自己看】\n回复用户的话说完以后，另起一行，在整条回复的最末尾写一张场景卡片，记下这一刻的情况（用户这一轮说的和你刚回复的都算进去）。卡片第一行写<rinne_scene>，最后一行写</rinne_scene>，中间五行，每行一句完整的话：\n在哪：你和用户此刻在什么地方；不在一起时分别写。\n我们：你们两个是不是在一起，是什么状态。\n在做：此刻你们在做什么；各做各的就分别写。\n刚才：刚刚发生了什么。\n打算：接下来说好或打算做什么；没有就写“没有”。\n写卡片时，核对每件事是谁说的、谁做的、对谁做的；容易混淆时直接写“凛祢”和“用户”，不要把双方的动作、话语或经历写反。只写今天的聊天和用户这一轮的话里有依据的；拿不准就写“不确定”，不要猜。卡片只给你自己看，用户看不到也听不到；正文照常说话，不要提起卡片。'
CARD_INSTRUCTION_START = '【场景卡片｜只写给你自己看】\n开口回复用户之前，先在整条回复的最开头写一张场景卡片，记下这一刻的情况（用户这一轮说的话也算进去）。卡片第一行写<rinne_scene>，最后一行写</rinne_scene>，中间五行，每行一句完整的话：\n在哪：你和用户此刻在什么地方；不在一起时分别写。\n我们：你们两个是不是在一起，是什么状态。\n在做：此刻你们在做什么；各做各的就分别写。\n刚才：刚刚发生了什么。\n打算：接下来说好或打算做什么；没有就写“没有”。\n写完卡片另起一行，再照常回复用户。写卡片时，核对每件事是谁说的、谁做的、对谁做的；容易混淆时直接写“凛祢”和“用户”，不要把双方的动作、话语或经历写反。只写今天的聊天和用户这一轮的话里有依据的；拿不准就写“不确定”，不要猜。卡片只给你自己看，用户看不到也听不到；正文照常说话，不要提起卡片。'
CARD_INJECTION_TEMPLATE = '【你上一轮记下的场景卡片｜可能有错】\n这是你在{written_hhmm}自己记的，只给你自己看。和用户这一轮说的话或今天的聊天对不上时，以用户和聊天为准，这一轮的新卡片照实改过来。\n{card_lines}'

CARD_LABELS = ("在哪", "我们", "在做", "刚才", "打算")
_LABEL = re.compile(r"^\s*(在哪|我们|在做|刚才|打算)\s*[：:]\s*(.*?)\s*$")


def build_response_reminder(rules_text: str) -> str | None:
    lines = rules_text.replace("\r\n", "\n").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("7. "):
            selected = [line[3:]]
            for following in lines[index + 1:]:
                if not following.startswith("- "):
                    break
                selected.append(following)
            return "\n".join([REMINDER_HEADER, *selected])
    return None


def _normalize_card(block: str) -> str | None:
    values = {}
    for line in block.splitlines():
        match = _LABEL.match(line)
        if match:
            values[match[1]] = match[2].strip()
    if len(values) < 3:
        return None
    return "\n".join(f"{label}：{values[label]}" for label in CARD_LABELS if label in values)


def extract_scene_card(text: str) -> tuple[str, str | None, str]:
    full = re.compile(re.escape(SCENE_ENVELOPE_OPEN) + r"(.*?)" + re.escape(SCENE_ENVELOPE_CLOSE), re.S)
    blocks = list(full.finditer(text))
    candidate = blocks[-1][1] if blocks else None
    visible = full.sub("", text)
    status = "multiple" if len(blocks) > 1 else "none"
    had_unclosed = SCENE_ENVELOPE_OPEN in visible
    if had_unclosed:
        # Remove only the marker and immediately following label lines. The
        # rest can be visible prose when an opening card missed its close tag.
        while SCENE_ENVELOPE_OPEN in visible:
            prefix, tail = visible.split(SCENE_ENVELOPE_OPEN, 1)
            lines = tail.lstrip("\r\n").splitlines(keepends=True)
            consumed = []
            while lines and _LABEL.match(lines[0].rstrip("\r\n")):
                consumed.append(lines.pop(0).rstrip("\r\n"))
            if not blocks:
                candidate = "\n".join(consumed)
            visible = prefix + "".join(lines)
        if not blocks:
            status = "unclosed"
    if not blocks and not had_unclosed:
        lines = visible.splitlines()
        for at_start in (True, False):
            selected = []
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()
            indexes = range(len(lines)) if at_start else range(len(lines) - 1, -1, -1)
            for index in indexes:
                if not _LABEL.match(lines[index]):
                    break
                selected.append(index)
            if len(selected) >= 3:
                candidate = "\n".join(lines[index] for index in sorted(selected))
                lines = [line for index, line in enumerate(lines) if index not in selected]
                status = "untagged"
        visible = "\n".join(lines)
    card = _normalize_card(candidate) if candidate is not None else None
    if blocks and len(blocks) == 1:
        status = "ok" if card and len(card.splitlines()) == 5 else "partial" if card else "malformed"
    visible = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", visible).strip()
    return visible, card, status


def render_previous_card(card_lines: str, written_at: datetime | str) -> str:
    stamp = datetime.fromisoformat(written_at) if isinstance(written_at, str) else written_at
    return CARD_INJECTION_TEMPLATE.format(written_hhmm=stamp.astimezone().strftime("%H:%M"), card_lines=card_lines)


class SceneCardStore:
    def __init__(self, history_root):
        self.path = Path(history_root) / "scene_card" / "current_card.json"

    def save(self, card_lines: str, written_at: datetime, source_channel: str = "desktop") -> None:
        temporary = None
        try:
            stamp = written_at.astimezone()
            payload = dict(version=1, card_lines=card_lines, written_at=stamp.isoformat(), memory_day=memory_day_for_timestamp(stamp).isoformat(), source_channel=source_channel)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temporary.replace(self.path)
        except Exception as exc:
            logger.warning("[场景卡片] 保存失败：{}", type(exc).__name__)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def load_fresh(self, now: datetime) -> dict | None:
        try:
            if not self.path.exists():
                return None
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            written = datetime.fromisoformat(payload["written_at"]).astimezone()
            current = now.astimezone()
            if payload.get("version") != 1 or not isinstance(payload.get("card_lines"), str):
                raise ValueError("Invalid scene card format")
            age = (current - written).total_seconds()
            if not 0 <= age <= 7200 or payload["memory_day"] != memory_day_for_timestamp(current).isoformat():
                return None
            return payload
        except Exception as exc:
            logger.warning("[场景卡片] 读取失败：{}", type(exc).__name__)
            return None
