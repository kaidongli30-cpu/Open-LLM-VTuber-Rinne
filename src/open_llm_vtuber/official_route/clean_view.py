"""Build a disposable checked view without editing source memories or history."""

from dataclasses import dataclass
from datetime import date
import json
import re

from ..memory.layer2_context import format_layer2_context
from ..rinne_renderer_profile import _MP_GUIDANCE_START, _MP_LATER_RULE
from .content_check import ContentChecker

BACKGROUND_PLACEHOLDER = "（这里是一段两人亲密时光的记录，细节略。）"
HISTORY_NOTICE = "[系统提示：这之前有一段两人亲密相处的时光，细节略。]"
OPEN_BACKGROUND = "<current_user_background>"
CLOSE_BACKGROUND = "</current_user_background>"
FIXED_ANCHORS = (_MP_GUIDANCE_START, "【核心原则】", _MP_LATER_RULE)


def fixed_wrapper_parts(as_of_date: str):
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", as_of_date):
        return None
    try:
        date.fromisoformat(as_of_date)
    except ValueError:
        return None
    rendered = format_layer2_context(as_of_date, "FIXED_BODY_SENTINEL")
    return rendered.split("FIXED_BODY_SENTINEL")


def fixed_protected_fragment(text: str, name: str) -> bool:
    if name.startswith("anchor_"):
        return text in FIXED_ANCHORS
    if name == "wrapper_suffix":
        return text == fixed_wrapper_parts("2026-01-01")[1]
    if name == "wrapper_prefix":
        match = re.match(r"【第二层记忆：截至 ([0-9]{4}-[0-9]{2}-[0-9]{2}) ", text)
        parts = fixed_wrapper_parts(match[1]) if match else None
        return parts is not None and text == parts[0]
    return False


def fixed_protected_line(line: str) -> bool:
    parts = fixed_wrapper_parts("2026-01-01")
    known = {
        x.strip()
        for text in (*parts, *FIXED_ANCHORS)
        for x in text.splitlines()
        if x.strip()
    }
    if line in known:
        return True
    match = re.fullmatch(
        r"【第二层记忆：截至 ([0-9]{4}-[0-9]{2}-[0-9]{2}) 的用户当前个人背景】", line
    )
    parts = fixed_wrapper_parts(match[1]) if match else None
    return parts is not None and line == parts[0].splitlines()[0]


def _check_protected(part, checker):
    text = part["text"]
    hit = checker.keyword_hit(text)
    valid = not hit and fixed_protected_fragment(text, part["name"])
    return checker._remember(
        text,
        "protected",
        (),
        "B" if hit else ("A" if valid else "U"),
        100 if hit or valid else 0,
        "keyword" if hit else "fixed_text",
        "关键词命中" if hit else ("固定文字核对" if valid else "固定文字不符"),
    )


class ProtectedContentBlocked(RuntimeError):
    def __init__(self, blocked):
        self.blocked = blocked
        super().__init__("protected_fragments_not_A")


@dataclass
class CleanBackground:
    text: str
    manifest: list[dict]


@dataclass
class CleanHistory:
    turns: list[dict]
    manifest: list[dict]


def _paragraphs(text):
    return re.split(r"(\r?\n[ \t]*(?:\r?\n)+)", text)


def _items(text):
    # A list item's continuation remains with its item. With no list markers,
    # retain the paragraph boundaries rather than inventing a new structure.
    starts = [m.start() for m in re.finditer(r"(?m)^[ \t]*(?:[-*]|\d+[.)、])\s+", text)]
    if not starts:
        return _paragraphs(text)
    points = sorted(set([0, *starts, len(text)]))
    result = []
    for a, b in zip(points, points[1:]):
        value = text[a:b]
        trailing = re.search(r"(?:\r?\n[ \t]*)+$", value)
        if trailing:
            result.extend([value[: trailing.start()], value[trailing.start() :]])
        else:
            result.append(value)
    return result


def background_parts(text: str, anchors=(), *, layer2=False):
    spans = []
    if layer2:
        opening = text.find(OPEN_BACKGROUND)
        closing = text.rfind(CLOSE_BACKGROUND)
        if opening < 0 or closing < opening:
            raise ValueError("background_wrapper_missing")
        content_start = opening + len(OPEN_BACKGROUND)
        if text[content_start : content_start + 1] == "\n":
            content_start += 1
        suffix_start = closing - 1 if text[closing - 1 : closing] == "\n" else closing
        spans.extend(
            [
                (0, content_start, "wrapper_prefix"),
                (suffix_start, len(text), "wrapper_suffix"),
            ]
        )
    for index, anchor in enumerate(anchors):
        if not anchor:
            raise ValueError("empty_anchor")
        positions = [m.start() for m in re.finditer(re.escape(anchor), text)]
        if not positions:
            raise ValueError("protected_anchor_missing")
        spans.extend(
            (start, start + len(anchor), f"anchor_{index}") for start in positions
        )
    spans.sort()
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("overlapping_protected_spans")
    parts = []
    position = 0
    for start, end, name in spans + [(len(text), len(text), "")]:
        if start > position:
            source = text[position:start]
            values = (
                _items(source)
                if layer2 and position >= content_start
                else _paragraphs(source)
            )
            parts.extend(
                dict(
                    text=value, protected=False, name="item" if layer2 else "paragraph"
                )
                for value in values
                if value
            )
        if end > start:
            parts.append(dict(text=text[start:end], protected=True, name=name))
        position = end
    assert "".join(x["text"] for x in parts) == text
    return parts


async def clean_background(
    text: str, checker: ContentChecker, *, anchors=(), layer2=False
) -> CleanBackground:
    parts = background_parts(text, anchors, layer2=layer2)
    protected = [x for x in parts if x["protected"]]
    protected_records = [_check_protected(part, checker) for part in protected]
    blocked = [
        dict(kind=part["name"], sha256=record.sha256, label=record.label)
        for part, record in zip(protected, protected_records)
        if record.label != "A"
    ]
    if blocked:
        raise ProtectedContentBlocked(blocked)
    by_text = {
        part["text"]: record for part, record in zip(protected, protected_records)
    }
    ordinary = [x["text"] for x in parts if not x["protected"] and x["text"].strip()]
    ordinary_records = iter(await checker.check_many(ordinary))
    output = []
    manifest = []
    blocked_run = False
    gap = ""
    newline = "\r\n" if "\r\n" in text else "\n"
    for part in parts:
        value = part["text"]
        if not value.strip():
            if blocked_run:
                gap += value
            else:
                output.append(value)
            continue
        record = by_text[value] if part["protected"] else next(ordinary_records)
        replaced = record.label != "A"
        manifest.append(
            dict(
                sha256=record.sha256,
                label=record.label,
                confidence=record.confidence,
                source=record.source,
                protected=part["protected"],
                kind=part["name"],
                replaced=replaced,
            )
        )
        if replaced:
            if not blocked_run:
                if "".join(output).rsplit("\n", 1)[-1].strip():
                    output.append(newline)
                output.append(BACKGROUND_PLACEHOLDER)
            blocked_run = True
            gap = ""
        else:
            if blocked_run:
                if (gap + value).split("\n", 1)[0].strip():
                    output.append(newline)
                output.append(gap)
            output.append(value)
            blocked_run = False
            gap = ""
            checker.record_lines(value, record)
    if gap:
        output.append(gap)
    return CleanBackground("".join(output), manifest)


async def clean_history(turns: list[dict], checker: ContentChecker) -> CleanHistory:
    texts = [
        json.dumps(
            {"user": turn["user"], "assistant": turn.get("assistant", "")},
            ensure_ascii=False,
        )
        for turn in turns
    ]
    contexts = [tuple(texts[max(0, i - 3) : i]) for i in range(len(turns))]
    records = await checker.check_many(texts, kind="turn", contexts=contexts)
    output = []
    manifest = []
    pending_notice = False
    for index, (turn, record) in enumerate(zip(turns, records)):
        keep = record.label == "A"
        manifest.append(
            dict(
                index=index,
                sha256=record.sha256,
                label=record.label,
                confidence=record.confidence,
                source=record.source,
                removed=not keep,
            )
        )
        if not keep:
            pending_notice = True
            continue
        current = dict(turn)
        checker.record_lines(turn["user"], record)
        checker.record_lines(turn.get("assistant", ""), record)
        if pending_notice:
            current["user"] = HISTORY_NOTICE + "\n" + current["user"]
            checker.record_fixed_line(HISTORY_NOTICE)
            pending_notice = False
        output.append(current)
    return CleanHistory(output, manifest)
