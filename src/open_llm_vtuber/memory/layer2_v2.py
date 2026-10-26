"""Versioned, evidence-checked daily background and shared-memory pipeline.

The pure day processor accepts an injected caller. It never opens a history
directory; only the runtime entry point and publisher do so.
"""

from __future__ import annotations

import calendar
import copy
import hashlib
import json
import re
import shutil
import time
import uuid
from collections import Counter
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path

from jsonschema import Draft202012Validator
from loguru import logger

from . import layer2_runtime as runtime
from .layer2_context import (
    format_layer2_context,
    load_layer2_publication,
    load_current_layer2_context,
    sha256_file,
)

LEGACY_BLOCK_TITLES = {1: "他是什么样的人", 2: "他最近的生活", 3: "他和凛祢现在的关系"}
BLOCK_TITLES = {
    1: "用户是什么样的人",
    2: "用户最近的生活",
    3: "用户和凛祢现在的关系",
    4: "两人重要的回忆",
    5: "以后的计划和约定",
    6: "不要假设的事",
}
FACT_BLOCKS = (1, 2, 3, 5, 6)
LEDGER_VERSION = "2.0-layer2-v2-ledger"
PROJECTION_VERSION = "2.0-layer2-v2-blocks"
PUBLICATION_VERSION = "2.0-layer2-v2-publication"
TOOL_NAMES = {
    "ledger_update": "submit_layer2_v2_ledger_update",
    "memory_lines": "submit_layer2_v2_memory_lines",
    "block_render": "submit_layer2_v2_block",
}
TOOL_DESCRIPTIONS = {
    "ledger_update": "提交底账的修改。",
    "memory_lines": "提交回忆短句。",
    "block_render": "提交这一块的阅读版。",
}
EXAMPLE_HEADINGS = {
    "ledger_update": "给底账更新",
    "memory_lines": "给回忆短句",
    "block_render": "给分块重写",
}
ROOT = Path(__file__).resolve().parent
OBSOLETE_WORDS = (
    "前几天",
    "前两天",
    "这几天",
    "这两天",
    "这阵子",
    "上周",
    "这周",
    "本周",
    "下周",
    "上个月",
    "这个月",
    "本月",
    "下个月",
    "最近",
    "近期",
    "近日",
    "刚刚",
    "刚才",
    "昨天",
    "今天",
    "明天",
    "前天",
    "后天",
    "今晚",
    "昨晚",
    "明晚",
    "今年",
    "去年",
    "明年",
)
QUOTED = re.compile(r'“[^”]*”|「[^」]*」|『[^』]*』|"[^"]*"|《[^》]*》', re.S)
COUNT_PATTERN = re.compile(r"(?<!第)[一二两三四五六七八九十百\d]+次")
DATE_PREFIX = re.compile(
    r"^\s*(\d{4}年)?\d{1,2}月\d{1,2}日([到至、～~-]\d{1,2}(月\d{1,2})?日?)?[：:，,\s]*"
)
MANUAL_PREFIX = re.compile(
    r"^((?:\d{4}年)?\d{1,2}月\d{1,2}(?:、\d{1,2})?日(?:到(?:(?:\d{4}年)?\d{1,2}月)?\d{1,2}日)?)[：:](.*)$"
)
OP_KEYS = (
    "added_facts",
    "updated_facts",
    "retired_facts",
    "reviewed_facts",
    "added_memories",
    "updated_memories",
)
HE_WORD = re.compile(r"(?<![其吉利排])他(?![们人俩乡日处者])")
FIRST_TIME_WORDS = ("第一次", "头一次", "头一回", "第一回", "首次")
MAX_NEW_MEMORIES = 2
FIRST_TIME_PROBLEM = "写了‘第一次’，但证据里没有写着‘第一次’的日记原文；日记没明说就去掉‘第一次’，明说了就附上那段原文"
MEMORY_FIRST_TIME_PROBLEM = "输入里没有‘第一次’，句子里不要写‘第一次’"


def first_time_count(text):
    return sum(text.count(word) for word in FIRST_TIME_WORDS)


def day(value):
    return value if isinstance(value, date) else date.fromisoformat(value)


def iso(value):
    return day(value).isoformat()


def clean(value):
    return str(value).replace("\r\n", "\n").replace("\n", " ").strip()


def save_json(path, value):
    runtime._write_json(Path(path), value)


def _attempt_count(caller):
    value = getattr(caller, "attempt_count", None)
    return value if isinstance(value, int) else None


def obsolete_words(text):
    text = QUOTED.sub("", text)
    found = [w for w in OBSOLETE_WORDS if w in text]
    if re.search(r"(?<![\d几])日前", text):
        found.append("日前")
    return found


def months_before(value, months=3):
    d = day(value)
    index = d.year * 12 + d.month - 1 - months
    year, month = divmod(index, 12)
    return date(year, month + 1, min(d.day, calendar.monthrange(year, month + 1)[1]))


def ten_day(value, current):
    d, now = day(value), day(current)
    return (
        (f"{d.year}年" if d.year != now.year else "")
        + f"{d.month}月"
        + ("上旬" if d.day <= 10 else "中旬" if d.day <= 20 else "下旬")
    )


def date_label(start, end, current):
    a, b, now = day(start), day(end), day(current)
    prefix = f"{a.year}年" if a.year != now.year else ""
    left = prefix + f"{a.month}月{a.day}"
    if a == b:
        return left + "日"
    if (a.year, a.month) == (b.year, b.month):
        return left + (f"、{b.day}日" if (b - a).days == 1 else f"日到{b.day}日")
    right = (f"{b.year}年" if a.year != b.year else "") + f"{b.month}月{b.day}日"
    return left + "日到" + right


def review_date(kind, value, expected=None):
    d = day(value)
    if kind == "persistent":
        return None
    if kind == "scheduled" and expected:
        return (day(expected) + timedelta(days=1)).isoformat()
    return (d + timedelta(days=30 if kind == "ongoing" else 14)).isoformat()


def last_mentioned(fact):
    return max([fact["last_updated_on"], *fact.get("source_dates", [])])


def next_id(prefix, value, items, key):
    used = {x[key] for x in items}
    base = prefix + "-" + iso(value).replace("-", "") + "-"
    i = 1
    while base + f"{i:03d}" in used:
        i += 1
    return base + f"{i:03d}"


def review_queue(state, current):
    today = iso(current)
    candidates = []
    for f in state["facts"]:
        if f["status"] != "current" or f["temporal_class"] == "persistent":
            continue
        due, end = f.get("review_after"), f.get("expected_end_on")
        if (due and due <= today) or (end and end < today):
            candidates.append(f)
    return sorted(
        candidates,
        key=lambda f: (
            min(x for x in [f.get("review_after"), f.get("expected_end_on")] if x),
            f["fact_id"],
        ),
    )[:5]


def fact_hints(fact, current):
    d = day(current)
    hints = []
    mentioned = day(last_mentioned(fact))
    if (
        fact["block"] != 3
        and fact["temporal_class"] != "persistent"
        and mentioned < d - timedelta(days=30)
    ):
        hints.append(ten_day(mentioned, d) + "以后没再提")
    if (
        fact["temporal_class"] == "scheduled"
        and fact.get("expected_end_on")
        and day(fact["expected_end_on"]) < d
    ):
        hints.append(
            "原定"
            + date_label(fact["expected_end_on"], fact["expected_end_on"], d)
            + "，日子已过"
        )
    return "；".join(hints)


def prompt(kind, examples_text=""):
    text = (
        (ROOT / "prompts" / f"layer2_v2_{kind}.txt")
        .read_text(encoding="utf-8")
        .replace("\r\n", "\n")
    )
    title = EXAMPLE_HEADINGS[kind]
    match = re.search(
        r"^## " + re.escape(title) + r"\s*\n(.*?)(?=^## |\Z)",
        examples_text.replace("\r\n", "\n"),
        re.M | re.S,
    )
    return text + (
        "\n\n## 例子（用户确认过的写法）\n\n" + match[1].strip() if match else ""
    )


def schema(kind):
    return json.loads(
        (ROOT / "layer2_v2_schemas" / (kind + ".json")).read_text(encoding="utf-8")
    )


def validate_schema(kind, result):
    errors = sorted(
        Draft202012Validator(schema(kind)).iter_errors(result),
        key=lambda e: str(e.path),
    )
    if errors:
        raise runtime.Layer2RuntimeError(
            "v2_schema_invalid:" + ";".join(str(list(e.path)) for e in errors)
        )


def evidence(excerpts, diary_text):
    norm = runtime.cloud_runner._normalize_evidence_text
    source = norm(diary_text)
    return [x for x in excerpts if len(norm(x)) >= 5 and norm(x) in source]


def _blank_fact(
    fid,
    block,
    topic,
    statement,
    current,
    *,
    origin="ledger",
    temporal_class="persistent",
    expected=None,
):
    d = iso(current)
    return dict(
        fact_id=fid,
        block=int(block),
        topic=topic,
        statement=statement,
        status="current",
        temporal_class=temporal_class,
        origin=origin,
        first_observed_on=d,
        last_updated_on=d,
        source_dates=[d],
        review_after=review_date(temporal_class, d, expected),
        expected_end_on=expected or None,
        evidence_history=[],
        previous_statements=[],
        supersedes_fact_ids=[],
        retired=None,
        notes=[],
    )


def _blank_memory(
    mid, start, end, title, happened, why, later, current, *, origin="daily"
):
    d = iso(current)
    return dict(
        memory_id=mid,
        label=None,
        status="current",
        date_start=iso(start),
        date_end=iso(end),
        title=title,
        happened=happened,
        why=why,
        later=later,
        origin=origin,
        source_fact_ids=[],
        source_dates=[d],
        evidence_history=[],
        reading_line=None,
        line_mode=None,
        line_written_on=None,
        faded_on=None,
        line_stale=False,
        previous_versions=[],
        created_on=d,
        updated_on=d,
    )


def _retire(fact, current, basis, reason, into=None):
    fact["status"] = "retired"
    fact["retired"] = dict(on=iso(current), basis=basis, reason=reason, into=into)


def _evidence_entry(current, excerpts, kind="diary"):
    return dict(
        source_date=iso(current), evidence_type=kind, evidence_excerpts=excerpts
    )


def parse_overview(text):
    blocks = {n: [] for n in BLOCK_TITLES}
    titles = {v: k for k, v in {**BLOCK_TITLES, **LEGACY_BLOCK_TITLES}.items()}
    titles.update({v: k for k, v in BLOCK_TITLES.items()})
    current = None
    for line in text.replace("\r\n", "\n").splitlines():
        if not line.strip():
            continue
        if line.startswith("### "):
            title = line[4:].strip()
            if title not in titles:
                raise runtime.Layer2RuntimeError("v2_manual_edit_unparseable")
            current = titles[title]
        elif current is None:
            raise runtime.Layer2RuntimeError("v2_manual_edit_unparseable")
        elif line.startswith("- "):
            blocks[current].append(line[2:].strip())
        elif blocks[current]:
            blocks[current][-1] += " " + line.strip()
        else:
            raise runtime.Layer2RuntimeError("v2_manual_edit_unparseable")
    return blocks


def _projection_map(projection):
    return {int(b["block"]): b for b in (projection or {}).get("blocks", [])}


def apply_manual_edits(state, projection, edited_overview, current):
    metrics = dict(
        removed_lines=0,
        added_lines=0,
        retired_facts=0,
        new_manual_facts=0,
        memory_changes=0,
    )
    if edited_overview is None:
        return copy.deepcopy(projection), set(), [], metrics
    edited = parse_overview(edited_overview)
    result = copy.deepcopy(projection)
    original = _projection_map(projection)
    output = _projection_map(result)
    facts = {f["fact_id"]: f for f in state["facts"]}
    memories = {m["memory_id"]: m for m in state["memories"]}
    changed, changes = set(), []
    for b in BLOCK_TITLES:
        old = original.get(b, {}).get("lines", [])
        remaining = Counter(x.strip() for x in edited[b])
        kept, removed = [], []
        for line in old:
            text = line["text"].strip()
            if remaining[text]:
                kept.append(copy.deepcopy(line))
                remaining[text] -= 1
            else:
                removed.append(line)
        old_counts = Counter(x["text"].strip() for x in old)
        added = []
        for text in edited[b]:
            if old_counts[text]:
                old_counts[text] -= 1
            else:
                added.append(text)
        if not added and not removed:
            continue
        changed.add(b)
        metrics["removed_lines"] += len(removed)
        metrics["added_lines"] += len(added)
        new_lines = kept
        if b != 4:
            kept_ids = {fid for line in kept for fid in line["source_ids"]}
            for line in removed:
                retired = []
                for fid in line["source_ids"]:
                    if (
                        fid in facts
                        and fid not in kept_ids
                        and facts[fid]["status"] == "current"
                    ):
                        _retire(
                            facts[fid],
                            current,
                            "manual_removed",
                            "用户在阅读版里删掉了这一行",
                        )
                        facts[fid]["evidence_history"].append(
                            _evidence_entry(current, [line["text"]], "manual_edit")
                        )
                        retired.append(fid)
                        metrics["retired_facts"] += 1
                changes.append(
                    f"块{b} 删掉了一行：“{line['text']}”（程序已退役："
                    + "、".join(retired)
                    + "）"
                )
            for text in added:
                fid = (
                    "manual-" + hashlib.sha256(f"{b}\n{text}".encode()).hexdigest()[:16]
                )
                f = _blank_fact(fid, b, "用户亲手写", text, current, origin="manual")
                f["evidence_history"].append(
                    _evidence_entry(current, [text], "manual_edit")
                )
                if fid in facts:
                    facts[fid].update(f)
                else:
                    state["facts"].append(f)
                    facts[fid] = f
                new_lines.append(dict(text=text, source_ids=[fid]))
                metrics["new_manual_facts"] += 1
                changes.append(f"块{b} 加了一行：“{text}”（程序已记成新事实 {fid}）")
        else:
            edited_ids = set()
            for text in added:
                if text.startswith("纪念日："):
                    state["anniversaries"].setdefault("previous", []).append(
                        dict(line=state["anniversaries"]["line"], until=iso(current))
                    )
                    state["anniversaries"].update(
                        line=text[len("纪念日：") :],
                        origin="manual",
                        updated_on=iso(current),
                    )
                    new_lines.append(dict(text=text, source_ids=["anniversaries"]))
                    continue
                match = MANUAL_PREFIX.match(text)
                matching = [
                    line
                    for line in removed
                    if match
                    and (oldmatch := MANUAL_PREFIX.match(line["text"]))
                    and oldmatch[1] == match[1]
                    and line["source_ids"]
                    and line["source_ids"][0] in memories
                ]
                if len(matching) == 1:
                    mid = matching[0]["source_ids"][0]
                    m = memories[mid]
                    m["previous_versions"].append(
                        dict(
                            field="reading_line",
                            text=m.get("reading_line") or "",
                            until=iso(current),
                        )
                    )
                    m.update(
                        reading_line=match[2].strip(),
                        line_mode="manual",
                        status="current",
                        line_stale=False,
                        updated_on=iso(current),
                        line_written_on=iso(current),
                    )
                    edited_ids.add(mid)
                    changes.append(f"块4 改了一件回忆的句子：{mid}“{match[2].strip()}”")
                else:
                    stamp = day(current)
                    if match:
                        dm = re.match(r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})", match[1])
                        try:
                            stamp = date(
                                int(dm[1] or stamp.year), int(dm[2]), int(dm[3])
                            )
                        except ValueError as exc:
                            raise runtime.Layer2RuntimeError(
                                "v2_manual_edit_unparseable"
                            ) from exc
                    body = match[2].strip() if match else text
                    mid = next_id("mem", stamp, state["memories"], "memory_id")
                    m = _blank_memory(
                        mid,
                        stamp,
                        stamp,
                        body[:20],
                        body,
                        "",
                        "",
                        current,
                        origin="manual",
                    )
                    m.update(
                        reading_line=body,
                        line_mode="manual",
                        line_written_on=iso(current),
                    )
                    m["evidence_history"].append(
                        _evidence_entry(current, [text], "manual_edit")
                    )
                    state["memories"].append(m)
                    memories[mid] = m
                    changes.append(f"块4 加了一件回忆：{mid}“{body}”")
                new_lines.append(dict(text=text, source_ids=[mid]))
                metrics["memory_changes"] += 1
            for line in removed:
                for mid in line["source_ids"]:
                    if mid == "anniversaries":
                        if not any(x.startswith("纪念日：") for x in added):
                            state["anniversaries"].setdefault("previous", []).append(
                                dict(
                                    line=state["anniversaries"]["line"],
                                    until=iso(current),
                                )
                            )
                            state["anniversaries"].update(
                                line="", origin="manual", updated_on=iso(current)
                            )
                    elif mid in memories and mid not in edited_ids:
                        memories[mid]["status"] = "hidden"
                        metrics["memory_changes"] += 1
        output[b] = {
            **output.get(b, {}),
            "block": b,
            "title": BLOCK_TITLES[b],
            "lines": new_lines,
        }
    result["blocks"] = [output[b] for b in sorted(output)]
    return result, changed, changes, metrics


def ledger_task(state, current, diary_text, queue, manual_changes):
    facts = sorted(
        (f for f in state["facts"] if f["status"] == "current"),
        key=lambda f: (f["block"], f["topic"], f["fact_id"]),
    )
    memories = sorted(
        (m for m in state["memories"] if m["status"] == "current"),
        key=lambda m: (m["date_start"], m["memory_id"]),
    )
    catalog = [
        "｜".join(
            [
                f["fact_id"],
                str(f["block"]),
                f["topic"],
                last_mentioned(f),
                clean(f["statement"])
                + ("〔用户亲手写〕" if f["origin"] == "manual" else ""),
            ]
        )
        for f in facts
    ]
    rows = [
        "｜".join(
            [
                m["memory_id"],
                m["date_start"]
                + ("～" + m["date_end"] if m["date_start"] != m["date_end"] else ""),
                *[
                    clean(m.get(k, "")) or "（无）"
                    for k in ("title", "happened", "why", "later")
                ],
            ]
        )
        for m in memories
    ]
    due = []
    for f in queue:
        reasons = []
        if f.get("review_after") and f["review_after"] <= iso(current):
            reasons.append(f"复查的日子 {f['review_after']} 到了")
        if f.get("expected_end_on") and f["expected_end_on"] < iso(current):
            reasons.append(f"原定 {f['expected_end_on']} 结束，日子已过")
        due.append(f["fact_id"] + "｜" + "；".join(reasons))
    return "\n\n".join(
        [
            f"【今天】{iso(current)}",
            f"【底账目录】共 {len(facts)} 条。每行：编号｜块｜主题｜最后提到｜说法\n"
            + "\n".join(catalog),
            f"【回忆】共 {len(memories)} 件。每行：编号｜日期｜标题｜那天｜为什么记｜后来\n"
            + "\n".join(rows),
            "【纪念日】" + (state["anniversaries"]["line"] or "（无）"),
            f"【该复查的旧事实】共 {len(queue)} 条。每行：编号｜原因\n"
            + ("\n".join(due) or "（无）"),
            "【用户亲手改过的地方】\n" + ("\n".join(manual_changes) or "（无）"),
            "【今天的日记】\n" + diary_text,
        ]
    )


def validate_ledger_update(
    result,
    state,
    current,
    diary_text,
    queue,
    *,
    accept_duplicate=False,
    accept_long_statement=False,
):
    result = copy.deepcopy(result)
    validate_schema("ledger_update", result)
    facts = {f["fact_id"]: f for f in state["facts"] if f["status"] == "current"}
    memories = {
        m["memory_id"]: m for m in state["memories"] if m["status"] == "current"
    }
    due = {f["fact_id"] for f in queue}
    problems, warnings = {}, []
    updated = [x["fact_id"] for x in result["updated_facts"]]
    retired = [x["fact_id"] for x in result["retired_facts"]]
    new_ids = [x["temp_id"] for x in result["added_facts"]]
    for key in OP_KEYS:
        for i, op in enumerate(result[key]):
            errors = []
            if result["source_date"] != iso(current):
                errors.append("source_date_mismatch")
            if "evidence_excerpts" in op:
                op["evidence_excerpts"] = evidence(op["evidence_excerpts"], diary_text)
            required = key in (
                "added_facts",
                "updated_facts",
                "added_memories",
                "updated_memories",
            )
            if key in ("added_facts", "updated_facts"):
                if not op["statement"].startswith("用户"):
                    errors.append("statement_must_start_user")
                if obsolete_words(op["statement"]):
                    errors.append("obsolete_words")
                size = len(op["statement"].strip())
                if size > 150:
                    if accept_long_statement:
                        warnings.append("fact_statement_long")
                    else:
                        errors.append(
                            f"说法有 {size} 个字，超过 150 个字；删掉过去的经过和不再要紧的细节"
                        )
                for k in ["expected_end_on"]:
                    if op.get(k):
                        try:
                            day(op[k])
                        except ValueError:
                            errors.append("invalid_date")
                if op["temporal_class"] == "scheduled" and not op["expected_end_on"]:
                    warnings.append("scheduled_without_end")
                if int(op["block"]) == 3 and COUNT_PATTERN.search(op["statement"]):
                    warnings.append("counts_in_relationship")
                if key == "added_facts":
                    if new_ids.count(op["temp_id"]) != 1:
                        errors.append("duplicate_temp_id")
                elif op["fact_id"] not in facts or updated.count(op["fact_id"]) != 1:
                    errors.append("invalid_update_id")
            if key == "retired_facts":
                fid = op["fact_id"]
                basis = op["basis"]
                target = op["replaced_by"]
                if fid not in facts or retired.count(fid) != 1:
                    errors.append("invalid_retired_id")
                if fid in updated:
                    errors.append("update_and_retire")
                if basis in ("changed_today", "replaced") or (
                    basis == "plan_done" and fid not in due
                ):
                    required = True
                if facts.get(fid, {}).get("block") == 5:
                    if basis == "stale":
                        errors.append("promise_stale")
                    expired = (
                        facts[fid]["temporal_class"] == "scheduled"
                        and facts[fid].get("expected_end_on")
                        and day(facts[fid]["expected_end_on"]) < day(current)
                    )
                    if basis == "plan_done":
                        required = not expired
                        if required and not op.get("evidence_excerpts"):
                            errors.append("promise_done_no_evidence")
                if basis == "replaced" and (
                    target not in facts
                    and target not in new_ids
                    or target == fid
                    or target in retired
                ):
                    errors.append("invalid_replacement")
                if basis == "duplicate" and (
                    target not in facts or target == fid or target in retired
                ):
                    errors.append("invalid_duplicate_target")
                if basis == "stale" and (
                    fid not in due
                    or facts.get(fid, {}).get("temporal_class") == "persistent"
                ):
                    errors.append("invalid_stale_retirement")
            if key == "reviewed_facts":
                if op["fact_id"] not in due or op["fact_id"] in updated + retired:
                    errors.append("invalid_review_id")
                try:
                    day(op["next_review_after"])
                except ValueError:
                    errors.append("invalid_review_date")
            if key == "added_memories":
                try:
                    if (
                        not date(2025, 1, 1)
                        <= day(op["date_start"])
                        <= day(op["date_end"])
                        <= day(current)
                    ):
                        errors.append("invalid_memory_date")
                except ValueError:
                    errors.append("invalid_memory_date")
                if any(
                    obsolete_words(op[k]) for k in ("title", "happened", "why", "later")
                ):
                    errors.append("obsolete_words")
                duplicate = any(
                    m["date_start"] == op["date_start"]
                    and SequenceMatcher(None, m["title"], op["title"]).ratio() >= 0.6
                    for m in memories.values()
                )
                if duplicate:
                    if accept_duplicate:
                        warnings.append("possible_duplicate_memory")
                    else:
                        errors.append("possible_duplicate_memory")
                if any(
                    COUNT_PATTERN.search(op[k])
                    for k in ("title", "happened", "why", "later")
                ):
                    warnings.append("counts_in_memory")
            if key == "updated_memories":
                if op["memory_id"] not in memories:
                    errors.append("invalid_memory_id")
                if obsolete_words(op["later"]):
                    errors.append("obsolete_words")
            current_firsts = old_firsts = 0
            if key in ("added_facts", "updated_facts"):
                current_firsts = first_time_count(op["statement"])
                if key == "updated_facts":
                    old_firsts = first_time_count(
                        facts.get(op["fact_id"], {}).get("statement", "")
                    )
            elif key == "added_memories":
                current_firsts = sum(
                    first_time_count(op[k])
                    for k in ("title", "happened", "why", "later")
                )
            elif key == "updated_memories":
                current_firsts = first_time_count(op["later"])
                old_firsts = first_time_count(
                    memories.get(op["memory_id"], {}).get("later", "")
                )
            if current_firsts > old_firsts and not any(
                first_time_count(x) for x in op.get("evidence_excerpts", [])
            ):
                errors.append(FIRST_TIME_PROBLEM)
            if required and not op.get("evidence_excerpts"):
                errors.append("evidence_not_in_diary")
            if errors:
                problems[f"{key}[{i}]"] = errors
    if result["source_date"] != iso(current):
        problems["$"] = ["source_date_mismatch"]
    if len(result["added_memories"]) > MAX_NEW_MEMORIES:
        problems["added_memories"] = [
            f"added_memories：一天最多新加 2 件回忆，这次交了 {len(result['added_memories'])} 件；相关的合成一件，只留最值得记的"
        ]
    if result["anniversaries_line"] and obsolete_words(result["anniversaries_line"]):
        problems["anniversaries_line"] = ["obsolete_words"]
    # An invalid addition must not become a dangling replacement target.
    invalid_new = {
        x["temp_id"]
        for i, x in enumerate(result["added_facts"])
        if f"added_facts[{i}]" in problems
    }
    for i, op in enumerate(result["retired_facts"]):
        if op["replaced_by"] in invalid_new:
            problems.setdefault(f"retired_facts[{i}]", []).append("invalid_replacement")
    return result, problems, warnings


def problem_messages(
    kind, result, problems, *, state=None, current=None, facts=(), memories=()
):
    """Keep diagnostic codes local; give the model actionable Chinese feedback."""
    messages = []
    known = {f["fact_id"]: f for f in (state or {}).get("facts", [])}
    memory_map = {m["memory_id"]: m for m in memories}
    for key, codes in problems.items():
        match = re.fullmatch(r"(\w+)\[(\d+)\]", key)
        op = (
            result.get(match[1], [])[int(match[2])]
            if match and kind == "ledger_update"
            else {}
        )
        fid = op.get("fact_id", op.get("memory_id", op.get("temp_id", key)))
        label = f"{key}（{fid}）" if op else key
        for code in codes:
            text = None
            if code == "obsolete_words":
                if kind == "ledger_update":
                    fields = (
                        [
                            op[k]
                            for k in ("statement", "title", "happened", "why", "later")
                            if k in op
                        ]
                        if op
                        else [result.get(key, "")]
                    )
                elif kind == "memory_lines":
                    fields = [
                        x["text"] for x in result["lines"] if x["memory_id"] == key
                    ]
                else:
                    fields = [result["lines"][int(match[2])]["text"]]
                words = list(
                    dict.fromkeys(w for field in fields for w in obsolete_words(field))
                )
                text = (
                    f"{label}里有会过时的词"
                    + "、".join("“" + w + "”" for w in words)
                    + "；改成具体的日子或几月几旬（引号里的原话除外）"
                )
            elif code == "evidence_not_in_diary":
                excerpts = op.get("evidence_excerpts", [])
                text = (
                    f"证据“{excerpts[0][:20]}”在日记里找不到原文；照日记一字不改地复制"
                    if excerpts
                    else "没有可核对的证据；附上至少五个字、与日记原文一致的证据"
                )
            elif code == "statement_must_start_user":
                text = "说法要以“用户”开头"
            elif code in ("invalid_update_id", "invalid_retired_id"):
                exists = fid in known and known[fid]["status"] == "current"
                text = (
                    f"{fid} 这次出现了不止一次；一条事实一天只能更新一次，也不能又更新又退役"
                    if exists
                    else f"{fid} 不在【底账目录】里；只能更新、退役或复查目录里现有的条目"
                )
            elif code == "invalid_memory_id":
                text = f"{fid} 不在【回忆】里；只能更新【回忆】里现有的回忆"
            elif code in ("update_and_retire", "duplicate_temp_id"):
                text = f"{fid} 这次出现了不止一次；一条事实一天只能更新一次，也不能又更新又退役"
            elif code in ("invalid_replacement", "invalid_duplicate_target"):
                text = f"{fid} 的 replaced_by 要写取代它的那一条：现有条目的编号，或这次新加的 temp_id（如 new1），而且那一条这次没有被退役"
            elif code == "invalid_stale_retirement":
                text = (
                    f"{fid} 是长期有效的事实，不能用 stale；没有新的变化就保留"
                    if known.get(fid, {}).get("temporal_class") == "persistent"
                    else f"{fid} 不在【该复查的旧事实】里，不能用 stale"
                )
            elif code == "promise_stale":
                text = f"{fid} 是块 5 的计划或约定，不能因为很久没提就退役；做到了用 plan_done 并附日记原文，取消了用 changed_today，都没有就保留"
            elif code == "promise_done_no_evidence":
                text = f"{fid} 是块 5 的计划或约定，用 plan_done 退役要附写着做到了的日记原文；只做到一部分就更新成只剩没做到的部分"
            elif code == "invalid_review_id":
                handled = fid in {
                    x["fact_id"]
                    for k in ("updated_facts", "retired_facts")
                    for x in result[k]
                }
                text = (
                    f"{fid} 这次已经更新或退役了，不要再写进 reviewed_facts"
                    if handled
                    else f"{fid} 不在【该复查的旧事实】里；只能复查队列里的条目"
                )
            elif code in ("invalid_date", "invalid_review_date"):
                text = f"{fid} 的日子不是有效日期；按 YYYY-MM-DD 填写真实存在的日子"
            elif code == "source_date_mismatch":
                text = f"source_date 应该是 {iso(current)}，与【今天】保持一致"
            elif code == "invalid_memory_date":
                text = "回忆的日子不对：date_start 不能晚于 date_end，date_end 不能晚于【今天】，也不能早于 2025-01-01"
            elif code == "possible_duplicate_memory":
                old = next(
                    (
                        m
                        for m in (state or {}).get("memories", [])
                        if m["status"] == "current"
                        and m["date_start"] == op["date_start"]
                        and SequenceMatcher(None, m["title"], op["title"]).ratio()
                        >= 0.6
                    ),
                    None,
                )
                text = (
                    f"和已有的回忆 {old['memory_id']}（{old['title']}）像是同一件事；同一件事不要新加，有后续照第五节处理"
                    if old
                    else "像是已有的回忆；同一件事不要新加，有后续照第五节处理"
                )
            elif code == "line_length":
                row = next(x for x in result["lines"] if x["memory_id"] == key)
                size = len(DATE_PREFIX.sub("", row["text"].strip()))
                limit = 80 if memory_mode(memory_map[key], current) == "久远" else 150
                text = (
                    f"{key}：句子有 {size} 个字，超过 {limit} 个字；删掉次要的细节，核心和关键的原话留着"
                    if size
                    else f"{key}：句子是空的；写出这件回忆的核心和关键的原话"
                )
            elif code == "memory_id_missing_or_duplicate":
                count = sum(x["memory_id"] == key for x in result["lines"])
                text = f"少了 {key}" if count == 0 else f"{key} 交了不止一次"
            elif code == "wrong_mode":
                text = f"{key} 的模式应该是“{memory_mode(memory_map[key], current)}”"
            elif code == "unexpected_memory_id":
                invalid = sorted(
                    {x["memory_id"] for x in result["lines"]} - memory_map.keys()
                )
                text = (
                    "、".join(invalid)
                    + " 不在【要写的回忆】里；只交要求的编号，每件一次"
                )
            elif code == "fact_coverage_mismatch":
                ids = {f["fact_id"] for f in facts}
                used = {
                    fid for row in result["lines"] for fid in row["source_fact_ids"]
                }
                omitted = {row["fact_id"] for row in result["omitted"]}
                messages.extend(
                    f"{fid} 没有被任何一行引用，也不在 omitted 里；写进某一行，或者放进 omitted 并写明理由"
                    for fid in sorted(ids - used - omitted)
                )
                messages.extend(
                    f"{fid} 不是这一块的现役事实，不能引用，也不能放进 omitted"
                    for fid in sorted((used | omitted) - ids)
                )
                continue
            elif code == "id_in_text":
                text = f"第 {int(match[2]) + 1} 行里写了编号；编号只写在依据里"
            elif code == "duplicate_lines":
                seen = {}
                for i, row in enumerate(result["lines"], 1):
                    value = clean(row["text"])
                    if value in seen:
                        messages.append(
                            f"第 {i} 行和第 {seen[value]} 行完全一样；合成一行"
                        )
                    else:
                        seen[value] = i
                continue
            elif code == "block_mismatch":
                text = f"block 应该是 {facts[0]['block']}"
            # Existing spec-defined Chinese messages already contain details.
            if text is None:
                text = (
                    code
                    if re.search(r"[\u4e00-\u9fff]", code)
                    else f"{label} 不符合这一项的要求；按任务说明修正后重新提交"
                )
            messages.append(text if text.startswith(label) else f"{label}：{text}")
    return messages


def _repair_task(task, result, problems, *, kind="ledger_update", **context):
    lines = problem_messages(kind, result, problems, **context)
    return (
        task
        + "\n\n【上一次提交的内容】\n"
        + json.dumps(result, ensure_ascii=False)
        + "\n\n【上一次提交的问题】\n"
        + "\n".join(f"{i}. {text}" for i, text in enumerate(lines, 1))
        + "\n\n请重新提交完整的结果：改正这些问题，没有问题的部分照旧。"
    )


def call_ledger_update(
    state, current, diary_text, caller, examples_text, run_dir, manual_changes=()
):
    run_dir = Path(run_dir)
    queue = review_queue(state, current)
    if hasattr(caller, "set_evidence_source"):
        caller.set_evidence_source(diary_text)
    attempts_before = _attempt_count(caller)
    save_json(run_dir / "review_due.json", [f["fact_id"] for f in queue])
    task = ledger_task(state, current, diary_text, queue, manual_changes)
    started = time.monotonic()
    result = caller.call(
        "ledger_update",
        prompt("ledger_update", examples_text),
        task,
        run_dir / "ledger_update/attempts",
    )
    prepared, issues, warnings = validate_ledger_update(
        result, state, current, diary_text, queue
    )
    new_check_hits = dict(
        first_time=sum(FIRST_TIME_PROBLEM in x for x in issues.values()),
        extra_memories=max(0, len(prepared["added_memories"]) - MAX_NEW_MEMORIES),
        long_statement=sum(
            any(x.startswith("说法有 ") for x in errors) for errors in issues.values()
        ),
        promise_stale=sum("promise_stale" in x for x in issues.values()),
        promise_done_no_evidence=sum(
            "promise_done_no_evidence" in x for x in issues.values()
        ),
    )
    initial_issues = copy.deepcopy(issues)
    retried = bool(issues)
    if issues:
        result = caller.call(
            "ledger_update",
            prompt("ledger_update", examples_text),
            _repair_task(task, result, issues, state=state, current=current),
            run_dir / "ledger_update/content_retry/attempts",
        )
        prepared, issues, warnings = validate_ledger_update(
            result,
            state,
            current,
            diary_text,
            queue,
            accept_duplicate=True,
            accept_long_statement=True,
        )
    filtered = copy.deepcopy(prepared)
    for key in OP_KEYS:
        filtered[key] = [
            op
            for i, op in enumerate(prepared[key])
            if f"{key}[{i}]" not in issues and "$" not in issues
        ]
    if len(filtered["added_memories"]) > MAX_NEW_MEMORIES:
        retained_indices = [
            i
            for i in range(len(prepared["added_memories"]))
            if f"added_memories[{i}]" not in issues
        ]
        for i in retained_indices[MAX_NEW_MEMORIES:]:
            issues[f"added_memories[{i}]"] = ["extra_memories_dropped"]
        filtered["added_memories"] = filtered["added_memories"][:MAX_NEW_MEMORIES]
        warnings.append("extra_memories_dropped")
    if "anniversaries_line" in issues or "$" in issues:
        filtered["anniversaries_line"] = ""
    save_json(
        run_dir / "validation.json",
        dict(
            initial_issues=initial_issues,
            dropped=issues,
            warnings=warnings,
            content_retry=retried,
        ),
    )
    save_json(run_dir / "ledger_update/applied.json", filtered)
    metrics = dict(
        api_attempt_count=1 + int(retried),
        content_retry=retried,
        model_ops={k: len(prepared[k]) for k in OP_KEYS},
        applied_ops={k: len(filtered[k]) for k in OP_KEYS},
        dropped_ops=sum(k not in {"$", "added_memories"} for k in issues),
        warnings=len(warnings),
        new_check_hits=new_check_hits,
        elapsed_wall_seconds=time.monotonic() - started,
    )
    apply_ledger_update(state, filtered, current, queue)
    if attempts_before is not None:
        metrics["api_attempt_count"] = _attempt_count(caller) - attempts_before
    clamped = state["last_update"]["counts"].get("review_clamped", 0)
    if clamped:
        metrics["warnings"] += clamped
        save_json(
            run_dir / "validation.json",
            dict(
                initial_issues=initial_issues,
                dropped=issues,
                warnings=warnings + ["review_date_clamped"] * clamped,
                content_retry=retried,
            ),
        )
    return filtered, metrics


def apply_ledger_update(state, result, current, queue):
    d = iso(current)
    facts = {f["fact_id"]: f for f in state["facts"]}
    memories = {m["memory_id"]: m for m in state["memories"]}
    temps = {}
    for op in result["added_facts"]:
        fid = next_id("fct", d, state["facts"], "fact_id")
        temps[op["temp_id"]] = fid
        f = _blank_fact(
            fid,
            op["block"],
            op["topic"],
            op["statement"],
            d,
            temporal_class=op["temporal_class"],
            expected=op["expected_end_on"],
        )
        f["evidence_history"].append(_evidence_entry(d, op["evidence_excerpts"]))
        state["facts"].append(f)
        facts[fid] = f
    handled = set()
    for op in result["updated_facts"]:
        f = facts[op["fact_id"]]
        handled.add(f["fact_id"])
        f["previous_statements"].append(
            dict(statement=f["statement"], until=d, reason=op["reason"])
        )
        for key in ("statement", "topic", "temporal_class"):
            f[key] = op[key]
        f.update(
            block=int(op["block"]),
            expected_end_on=op["expected_end_on"] or None,
            last_updated_on=d,
        )
        f["source_dates"] = sorted(set([*f["source_dates"], d]))
        f["review_after"] = review_date(f["temporal_class"], d, f["expected_end_on"])
        f["evidence_history"].append(_evidence_entry(d, op["evidence_excerpts"]))
    for op in result["retired_facts"]:
        f = facts[op["fact_id"]]
        handled.add(f["fact_id"])
        target = temps.get(op["replaced_by"], op["replaced_by"]) or None
        _retire(f, d, op["basis"], op["reason"], target)
        if op["evidence_excerpts"]:
            f["evidence_history"].append(_evidence_entry(d, op["evidence_excerpts"]))
        if target in facts and f["fact_id"] not in facts[target]["supersedes_fact_ids"]:
            facts[target]["supersedes_fact_ids"].append(f["fact_id"])
    clamped = 0
    for op in result["reviewed_facts"]:
        f = facts[op["fact_id"]]
        handled.add(f["fact_id"])
        requested = day(op["next_review_after"])
        bounded = max(
            day(d) + timedelta(days=7), min(day(d) + timedelta(days=120), requested)
        )
        clamped += int(requested != bounded)
        f["review_after"] = bounded.isoformat()
        f["notes"].append(f"{d} 复查保留：{op['reason']}")
    for old in queue:
        if old["fact_id"] not in handled:
            f = facts[old["fact_id"]]
            f["review_after"] = (day(d) + timedelta(days=14)).isoformat()
            f["notes"].append(d + " 复查没有处理")
    for op in result["added_memories"]:
        mid = next_id("mem", op["date_start"], state["memories"], "memory_id")
        m = _blank_memory(
            mid,
            op["date_start"],
            op["date_end"],
            op["title"],
            op["happened"],
            op["why"],
            op["later"],
            d,
        )
        m["evidence_history"].append(_evidence_entry(d, op["evidence_excerpts"]))
        state["memories"].append(m)
    for op in result["updated_memories"]:
        m = memories[op["memory_id"]]
        m["previous_versions"].append(dict(field="later", text=m["later"], until=d))
        m.update(later=op["later"], updated_on=d, line_stale=True)
        m["source_dates"] = sorted(set([*m["source_dates"], d]))
        m["evidence_history"].append(_evidence_entry(d, op["evidence_excerpts"]))
    if (
        result["anniversaries_line"]
        and result["anniversaries_line"] != state["anniversaries"]["line"]
    ):
        a = state["anniversaries"]
        a["previous"].append(dict(line=a["line"], until=d))
        a.update(line=result["anniversaries_line"], origin="daily", updated_on=d)
    state["last_update"] = dict(
        source_date=d,
        run_kind="daily",
        counts={**{k: len(result[k]) for k in OP_KEYS}, "review_clamped": clamped},
    )


def memory_mode(memory, current):
    return "久远" if day(memory["date_end"]) <= months_before(current) else "近"


def memories_to_write(state, current):
    return sorted(
        [
            m["memory_id"]
            for m in state["memories"]
            if m["status"] == "current"
            and m["line_mode"] != "manual"
            and (
                m["reading_line"] is None
                or m.get("line_stale")
                or m["line_mode"] == "近"
                and memory_mode(m, current) == "久远"
            )
        ],
        key=lambda mid: next(
            (m["date_start"], mid) for m in state["memories"] if m["memory_id"] == mid
        ),
    )


def memory_task(memories, current):
    parts = [f"【今天】{iso(current)}", f"【要写的回忆】共 {len(memories)} 件。"]
    for m in memories:
        parts.append(
            "\n".join(
                [
                    "编号：" + m["memory_id"],
                    "模式：" + memory_mode(m, current),
                    "日期："
                    + m["date_start"]
                    + (
                        "～" + m["date_end"] if m["date_start"] != m["date_end"] else ""
                    ),
                    "标题：" + clean(m["title"]),
                    "那天：" + clean(m["happened"]),
                    "为什么记：" + clean(m["why"]),
                    "后来：" + (clean(m["later"]) or "（无）"),
                    "现在的句子："
                    + (clean(m["reading_line"]) if m["reading_line"] else "（无）"),
                ]
            )
        )
    return "\n\n".join(parts)


def check_memory_lines(result, memories, current):
    validate_schema("memory_lines", result)
    expected = {m["memory_id"]: m for m in memories}
    actual = Counter(x["memory_id"] for x in result["lines"])
    issues = {}
    valid = {}
    warnings = []
    for mid, m in expected.items():
        rows = [x for x in result["lines"] if x["memory_id"] == mid]
        errors = []
        if actual[mid] != 1:
            errors.append("memory_id_missing_or_duplicate")
        else:
            x = rows[0]
            text = x["text"].strip()
            stripped = DATE_PREFIX.sub("", text)
            if text != stripped:
                warnings.append("date_prefix_removed")
            text = stripped
            if x["mode"] != memory_mode(m, current):
                errors.append("wrong_mode")
            if not text or len(text) > (
                80 if memory_mode(m, current) == "久远" else 150
            ):
                errors.append("line_length")
            if obsolete_words(text):
                errors.append("obsolete_words")
            if first_time_count(text) and not any(
                first_time_count(m.get(k) or "")
                for k in ("title", "happened", "why", "later", "reading_line")
            ):
                errors.append(MEMORY_FIRST_TIME_PROBLEM)
            if "用户" in text:
                warnings.append("user_word")
            if HE_WORD.search(text):
                warnings.append("he_word")
            if COUNT_PATTERN.search(text):
                warnings.append("counts_in_memory")
            if not errors:
                valid[mid] = text
        if errors:
            issues[mid] = errors
    if any(mid not in expected for mid in actual):
        issues = {mid: ["unexpected_memory_id"] for mid in expected}
        valid = {}
    return valid, issues, warnings


def call_memory_lines(state, current, memory_ids, caller, examples_text, run_dir):
    attempts_before = _attempt_count(caller)
    by_id = {m["memory_id"]: m for m in state["memories"]}
    selected = sorted(
        [
            by_id[mid]
            for mid in memory_ids
            if by_id[mid]["status"] == "current" and by_id[mid]["line_mode"] != "manual"
        ],
        key=lambda m: (m["date_start"], m["memory_id"]),
    )
    metrics = dict(
        requested=len(selected),
        written=0,
        faded=0,
        fallback=0,
        api_attempt_count=0,
        warnings=[],
        first_time_hits=0,
    )
    for i in range(0, len(selected), 10):
        batch = selected[i : i + 10]
        pending = batch
        accepted = {}
        issues = {}
        result = None
        for retry in range(2):
            if not pending:
                break
            task = memory_task(pending, current)
            if retry:
                task += "\n\n【上一次的问题】\n" + "\n".join(
                    problem_messages(
                        "memory_lines",
                        result,
                        issues,
                        memories=pending,
                        current=current,
                    )
                )
            try:
                metrics["api_attempt_count"] += 1
                result = caller.call(
                    "memory_lines",
                    prompt("memory_lines", examples_text),
                    task,
                    Path(run_dir)
                    / "memory_lines"
                    / f"batch_{i // 10 + 1:02d}"
                    / f"pass_{retry + 1}"
                    / "attempts",
                )
                valid, issues, warnings = check_memory_lines(result, pending, current)
                accepted.update(valid)
                metrics["warnings"].extend(warnings)
                save_json(
                    Path(run_dir)
                    / "memory_lines"
                    / f"batch_{i // 10 + 1:02d}"
                    / f"pass_{retry + 1}"
                    / "validation.json",
                    dict(issues=issues, warnings=warnings),
                )
                if not retry:
                    metrics["first_time_hits"] += sum(
                        MEMORY_FIRST_TIME_PROBLEM in x for x in issues.values()
                    )
                pending = [m for m in pending if m["memory_id"] not in valid]
            except Exception as exc:
                if getattr(exc, "fatal_trial", False):
                    raise
                metrics["warnings"].append("memory_call_failed")
                break
        for m in batch:
            mid = m["memory_id"]
            if mid in accepted:
                previous = m.get("reading_line")
                mode = memory_mode(m, current)
                if previous:
                    m["previous_versions"].append(
                        dict(field="reading_line", text=previous, until=iso(current))
                    )
                if mode == "久远" and m.get("faded_on") is None:
                    m["faded_on"] = iso(current)
                    metrics["faded"] += 1
                m.update(
                    reading_line=accepted[mid],
                    line_mode=mode,
                    line_written_on=iso(current),
                    line_stale=False,
                )
                metrics["written"] += 1
            else:
                if m["reading_line"] is None:
                    m.update(reading_line=m["title"], line_mode="近")
                m["line_stale"] = True
                metrics["fallback"] += 1
                metrics["warnings"].append("memory_line_fallback")
    if attempts_before is not None:
        metrics["api_attempt_count"] = _attempt_count(caller) - attempts_before
    return metrics


RETIRE_LABELS = {
    "changed_today": "结束了",
    "replaced": "被取代",
    "plan_done": "做完了或日子已过",
    "duplicate": "重复",
    "stale": "过了时效",
    "manual_removed": "用户亲手删掉",
}


def block_changes(before, state, block, current, old_hints):
    previous = {f["fact_id"]: f for f in before["facts"]}
    now = {f["fact_id"]: f for f in state["facts"]}
    changes = []
    hints = {}
    for fid, f in now.items():
        old = previous.get(fid)
        here = f["status"] == "current" and f["block"] == block
        was = old and old["status"] == "current" and old["block"] == block
        if here:
            hints[fid] = fact_hints(f, current)
            if not old:
                changes.append(f"新加：{fid}（{f['topic']}）")
            elif old["block"] != block:
                changes.append(f"移进这一块：{fid}（{f['topic']}，从块{old['block']}）")
            elif old != f and any(
                old.get(k) != f.get(k)
                for k in (
                    "statement",
                    "topic",
                    "temporal_class",
                    "status",
                    "expected_end_on",
                )
            ):
                changes.append(
                    f"更新：{fid}（{f['topic']}）；原来的说法：{old['statement']}"
                )
            if hints[fid] != old_hints.get(fid, "") and hints[fid]:
                changes.append(f"提示有变化：{fid}（{f['topic']}）：{hints[fid]}")
        if was and not here:
            if f["status"] == "current":
                changes.append(f"移出这一块：{fid}（{f['topic']}，移到块{f['block']}）")
            else:
                r = f["retired"] or {}
                changes.append(
                    f"退役：{fid}（{f['topic']}）；理由：{RETIRE_LABELS.get(r.get('basis'), r.get('basis', ''))}：{r.get('reason', '')}"
                )
    return changes, hints


def block_task(block, facts, old_lines, changes, hints, current, full):
    rows = [
        "｜".join(
            [
                f["fact_id"],
                f["topic"],
                clean(f["statement"]),
                hints.get(f["fact_id"], ""),
            ]
        )
        for f in sorted(facts, key=lambda f: (f["topic"], f["fact_id"]))
    ]
    old = (
        "无，从头写这一块。"
        if full
        else f"共 {len(old_lines)} 行。每行：行号｜内容〔依据〕\n"
        + "\n".join(
            f"L{i}｜{clean(line['text'])}〔" + "、".join(line["source_ids"]) + "〕"
            for i, line in enumerate(old_lines, 1)
        )
    )
    return "\n\n".join(
        [
            f"【今天】{iso(current)}",
            f"【这一块】块{block}：{BLOCK_TITLES[block]}",
            f"【这一块的事实】共 {len(facts)} 条。每行：编号｜主题｜说法｜提示\n"
            + "\n".join(rows),
            "【旧版】" + old,
            "【这次的变化】\n" + ("\n".join(changes) or "（无）"),
        ]
    )


def check_block(result, block, facts):
    validate_schema("block_render", result)
    errors = {}
    warnings = []
    ids = {f["fact_id"] for f in facts}
    if result["block"] != str(block):
        errors["block"] = ["block_mismatch"]
    used = set()
    texts = []
    for i, line in enumerate(result["lines"]):
        text = clean(line["text"])
        texts.append(text)
        used.update(line["source_fact_ids"])
        if obsolete_words(text):
            errors[f"lines[{i}]"] = ["obsolete_words"]
        if re.search(r"[a-z]{2,6}-\d{8}-\d{3}|manual-[0-9a-f]{16}", text):
            errors.setdefault(f"lines[{i}]", []).append("id_in_text")
        if "用户" in text:
            warnings.append("user_word")
        if HE_WORD.search(text):
            warnings.append("he_word")
        if block == 3 and COUNT_PATTERN.search(text):
            warnings.append("counts_in_relationship")
        if len(text) > 140:
            warnings.append("long_line")
    omitted = {x["fact_id"] for x in result["omitted"]}
    if used | omitted != ids or not used <= ids or not omitted <= ids:
        errors["coverage"] = ["fact_coverage_mismatch"]
    if len(texts) != len(set(texts)):
        errors["lines"] = ["duplicate_lines"]
    if len(omitted) > 2:
        warnings.append("many_omitted")
    if used & omitted:
        warnings.append("used_and_omitted")
    return errors, warnings


def fallback_block(facts, old_lines, changed_ids):
    ids = {f["fact_id"] for f in facts}
    lines = [
        copy.deepcopy(x)
        for x in old_lines
        if set(x["source_ids"]) <= ids and not set(x["source_ids"]) & changed_ids
    ]
    referenced = {fid for x in lines for fid in x["source_ids"]}
    for f in facts:
        if f["fact_id"] not in referenced:
            lines.append(
                dict(
                    text=re.sub(r"^用户", "用户", clean(f["statement"])),
                    source_ids=[f["fact_id"]],
                )
            )
    return lines


def call_block_render(
    state,
    before,
    projection,
    block,
    current,
    caller,
    examples_text,
    run_dir,
    manual_changed=False,
    force_full=False,
):
    attempts_before = _attempt_count(caller)
    old = _projection_map(projection).get(block, {})
    facts = [
        f for f in state["facts"] if f["status"] == "current" and f["block"] == block
    ]
    changes, hints = block_changes(
        before, state, block, current, old.get("hint_snapshot", {})
    )
    full = (
        force_full
        or not old.get("lines")
        or projection.get("schema_version") != PROJECTION_VERSION
    )
    metadata = dict(
        block=block,
        title=BLOCK_TITLES[block],
        lines=[],
        omitted=[],
        rendered_on=iso(current),
        render_mode="full" if full else "rewrite",
        hint_snapshot=hints,
        needs_rerender=False,
    )
    if not facts:
        return metadata, dict(
            mode="unchanged", lines=0, omitted=0, warnings=[], api_attempt_count=0
        )
    old_lines = old.get("lines", [])
    if (
        not full
        and not changes
        and not manual_changed
        and not old.get("needs_rerender")
        and hints == old.get("hint_snapshot", {})
    ):
        output = copy.deepcopy(old)
        output["render_mode"] = "unchanged"
        return output, dict(
            mode="unchanged",
            lines=len(output["lines"]),
            omitted=len(output.get("omitted", [])),
            warnings=[],
            api_attempt_count=0,
        )
    task = block_task(block, facts, old_lines, changes, hints, current, full)
    warnings = []
    attempts = 0
    for retry in range(2):
        try:
            attempts += 1
            value = caller.call(
                "block_render",
                prompt("block_render", examples_text),
                task,
                Path(run_dir)
                / "blocks"
                / str(block)
                / f"pass_{retry + 1}"
                / "attempts",
            )
            problems, warnings = check_block(value, block, facts)
            save_json(
                Path(run_dir)
                / "blocks"
                / str(block)
                / f"pass_{retry + 1}"
                / "validation.json",
                dict(issues=problems, warnings=warnings),
            )
            if not problems:
                metadata["lines"] = [
                    dict(text=clean(x["text"]), source_ids=x["source_fact_ids"])
                    for x in value["lines"]
                ]
                metadata["omitted"] = value["omitted"]
                new_by_id = {f["fact_id"]: f for f in facts}
                oldfacts = {f["fact_id"]: f for f in before["facts"]}
                for line in old_lines:
                    sources = line["source_ids"]
                    if all(
                        fid in new_by_id
                        and fid in oldfacts
                        and new_by_id[fid]["statement"] == oldfacts[fid]["statement"]
                        and hints.get(fid, "")
                        == old.get("hint_snapshot", {}).get(fid, "")
                        for fid in sources
                    ) and line["text"] not in [x["text"] for x in metadata["lines"]]:
                        warnings.append("unchanged_old_line_reworded")
                break
            if retry == 0:
                task = _repair_task(
                    task, value, problems, kind="block_render", facts=facts
                )
        except Exception as exc:
            if getattr(exc, "fatal_trial", False):
                raise
            warnings = ["block_call_failed"]
            break
    else:
        problems = True
    if not metadata["lines"]:
        prev = {f["fact_id"]: f for f in before["facts"]}
        changed = {
            f["fact_id"]
            for f in facts
            if f["fact_id"] in prev
            and f["statement"] != prev[f["fact_id"]]["statement"]
        }
        metadata.update(
            lines=fallback_block(facts, old_lines, changed),
            omitted=[],
            render_mode="fallback",
            needs_rerender=True,
        )
        warnings.append("block_render_fallback")
    if attempts_before is not None:
        attempts = _attempt_count(caller) - attempts_before
    return metadata, dict(
        mode=metadata["render_mode"],
        lines=len(metadata["lines"]),
        omitted=len(metadata["omitted"]),
        warnings=warnings,
        api_attempt_count=attempts,
    )


def assemble_memories(state, current):
    lines = []
    if state["anniversaries"]["line"]:
        lines.append(
            dict(
                text="纪念日：" + state["anniversaries"]["line"],
                source_ids=["anniversaries"],
            )
        )
    for m in sorted(state["memories"], key=lambda m: (m["date_start"], m["memory_id"])):
        if m["status"] == "current":
            lines.append(
                dict(
                    text=date_label(m["date_start"], m["date_end"], current)
                    + "："
                    + (m["reading_line"] or m["title"]),
                    source_ids=[m["memory_id"]],
                )
            )
    return dict(
        block=4,
        title=BLOCK_TITLES[4],
        lines=lines,
        omitted=[],
        rendered_on=iso(current),
        render_mode="assembled",
        hint_snapshot={},
        needs_rerender=False,
    )


def render_overview(projection):
    return "\n\n".join(
        "### "
        + BLOCK_TITLES[int(b["block"])]
        + "\n"
        + "\n".join("- " + clean(x["text"]) for x in b["lines"])
        for b in sorted(projection["blocks"], key=lambda b: int(b["block"]))
        if b["lines"]
    )


def _finish(state, projection, metrics, current):
    state["as_of_date"] = iso(current)
    projection["as_of_date"] = iso(current)
    metrics.update(
        current_fact_count=sum(f["status"] == "current" for f in state["facts"]),
        memory_count=sum(m["status"] == "current" for m in state["memories"]),
        overview_characters=len(render_overview(projection)),
    )
    metrics["he_word_count"] = metrics.get("memory_lines", {}).get(
        "warnings", []
    ).count("he_word") + sum(
        b.get("warnings", []).count("he_word")
        for b in metrics.get("blocks", {}).values()
    )
    return state, projection, metrics


def initial_render(state, as_of, caller, examples_text, run_dir):
    state = copy.deepcopy(state)
    projection = dict(
        schema_version=PROJECTION_VERSION, as_of_date=iso(as_of), blocks=[], warnings=[]
    )
    metrics = dict(
        pipeline="v2",
        manual_edits={},
        ledger_update={},
        review_due={},
        blocks={},
        warnings=[],
    )
    metrics["memory_lines"] = call_memory_lines(
        state, as_of, memories_to_write(state, as_of), caller, examples_text, run_dir
    )
    for b in FACT_BLOCKS:
        block, m = call_block_render(
            state,
            state,
            projection,
            b,
            as_of,
            caller,
            examples_text,
            run_dir,
            force_full=True,
        )
        projection["blocks"].append(block)
        metrics["blocks"][str(b)] = m
    projection["blocks"].append(assemble_memories(state, as_of))
    projection["blocks"].sort(key=lambda b: b["block"])
    return _finish(state, projection, metrics, as_of)


def process_day(
    state,
    base_projection,
    current_day,
    diary_text,
    edited_overview,
    caller,
    examples_text,
    run_dir,
    *,
    allow_backdated=False,
):
    if not allow_backdated and day(current_day) <= day(state["as_of_date"]):
        raise runtime.Layer2RuntimeError("v2_day_must_advance")
    state = copy.deepcopy(state)
    before = copy.deepcopy(state)
    run_dir = Path(run_dir)
    projection, changed, manual, manual_metrics = apply_manual_edits(
        state, base_projection, edited_overview, current_day
    )
    save_json(run_dir / "manual_edits.json", manual)
    queue = review_queue(state, current_day)
    update, ledger_metrics = call_ledger_update(
        state, current_day, diary_text, caller, examples_text, run_dir, manual
    )
    metrics = dict(
        pipeline="v2",
        manual_edits=manual_metrics,
        ledger_update=ledger_metrics,
        blocks={},
        warnings=[],
    )
    metrics["memory_lines"] = call_memory_lines(
        state,
        current_day,
        memories_to_write(state, current_day),
        caller,
        examples_text,
        run_dir,
    )
    output = dict(
        schema_version=PROJECTION_VERSION,
        as_of_date=iso(current_day),
        blocks=[],
        warnings=[],
    )
    for b in FACT_BLOCKS:
        block, m = call_block_render(
            state,
            before,
            projection,
            b,
            current_day,
            caller,
            examples_text,
            run_dir,
            manual_changed=b in changed,
        )
        output["blocks"].append(block)
        metrics["blocks"][str(b)] = m
    output["blocks"].append(assemble_memories(state, current_day))
    output["blocks"].sort(key=lambda b: b["block"])
    due = {f["fact_id"] for f in queue}
    retired = {x["fact_id"] for x in update["retired_facts"]}
    reviewed = {x["fact_id"] for x in update["reviewed_facts"]}
    updated = {x["fact_id"] for x in update["updated_facts"]}
    metrics["review_due"] = dict(
        selected=len(due),
        retired=len(due & retired),
        reviewed=len(due & reviewed),
        updated=len(due & updated),
        unhandled=len(due - retired - reviewed - updated),
    )
    return _finish(state, output, metrics, current_day)


def deepseek_request_schema(source):
    """Preserve data properties named title through the shared converter.

    The legacy converter strips schema metadata recursively, including a
    property literally named title. Protect that property name only while
    converting; the request and local validation retain the frozen schema.
    """
    marker = "__layer2_title_property__"

    def rename(node, before, after):
        if isinstance(node, dict):
            result = {key: rename(value, before, after) for key, value in node.items()}
            properties = result.get("properties")
            if isinstance(properties, dict) and before in properties:
                if after in properties:
                    raise runtime.Layer2RuntimeError("v2_schema_property_collision")
                properties[after] = properties.pop(before)
            return result
        if isinstance(node, list):
            return [rename(value, before, after) for value in node]
        return copy.deepcopy(node)

    protected = rename(source, "title", marker)
    converted = runtime.cloud_runner._deepseek_strict_schema(protected)
    return rename(converted, marker, "title")


def _format_issues(definition, value):
    return list(Draft202012Validator(definition).iter_errors(value))


def _format_preservation_rules(definition, value, issues, evidence_source=None):
    """Hold valid known data fixed while the model repairs invalid syntax."""
    bad = [
        tuple(e.absolute_path)
        for e in issues
        if e.validator
        not in {"additionalProperties", "required", "minItems", "maxItems"}
    ]
    rules = []

    def walk(node, current, path=()):
        if any(path[: len(p)] == p for p in bad):
            return
        if "$ref" in node:
            node = definition["$defs"][node["$ref"].rsplit("/", 1)[-1]]
        if node.get("type") == "object" and isinstance(current, dict):
            known = node.get("properties", {})
            rules.append(("members", path, [key for key in current if key in known]))
            for key in current:
                if key in known:
                    walk(known[key], current[key], (*path, key))
        elif node.get("type") == "array" and isinstance(current, list):
            size = min(
                max(len(current), node.get("minItems", 0)),
                node.get("maxItems", len(current)),
            )
            rules.append(("length", path, size))
            if size == len(current):
                for index, item in enumerate(current):
                    walk(node.get("items", {}), item, (*path, index))
        else:
            if (
                evidence_source is not None
                and len(path) >= 2
                and path[-2] == "evidence_excerpts"
                and isinstance(current, str)
                and not evidence([current], evidence_source)
            ):
                # An unsupported quotation is not valid content to pin. Its
                # changed replacement must pass the diary-evidence check.
                # An unchanged quote is not a formatting content change;
                # the regular semantic validator still checks its evidence.
                rules.append(("evidence", path, (current, evidence_source)))
            else:
                rules.append(("value", path, copy.deepcopy(current)))

    walk(definition, value)
    return rules


def _format_changed_paths(value, rules):
    changed = []
    for kind, path, expected in rules:
        try:
            actual = value
            for part in path:
                actual = actual[part]
            valid = (
                all(key in actual for key in expected)
                if kind == "members"
                else len(actual) == expected
                if kind == "length"
                else (actual == expected[0] or bool(evidence([actual], expected[1])))
                if kind == "evidence"
                else actual == expected
            )
        except (KeyError, IndexError, TypeError):
            valid = False
        if not valid:
            changed.append(list(path))
    return changed


def _format_repair_task(original, anchor, issues):
    messages = []
    for item in issues:
        path = item["path"]
        value = anchor
        try:
            for part in path:
                value = value[part]
        except (KeyError, IndexError, TypeError):
            value = None
        location = "工具参数" + "".join(
            f"[{p}]" if isinstance(p, int) else "." + p for p in path
        )
        rule = item["rule"]
        if rule == "invalid_evidence_quote":
            text = f"{location}：证据“{str(value)[:20]}”在日记里找不到原文；照日记一字不改地复制，其他合法内容照旧"
        elif rule == "restore_original_valid_content":
            text = (
                location
                + "：原先合法的内容被改动了；恢复为原值 "
                + json.dumps(item["expected"], ensure_ascii=False)
            )
        elif rule == "additionalProperties":
            names = re.findall(r"'([^']+)'", item.get("message", ""))
            text = (
                location
                + "：多了格式不允许的字段 "
                + "、".join(names)
                + "；只删除这些多余字段，原有合法内容照旧"
            )
        elif rule == "required":
            text = location + "：缺少格式要求的字段；补齐字段，原有合法内容照旧"
        elif rule in ("minItems", "maxItems"):
            text = (
                location
                + f"：列表交了 {len(value) if isinstance(value, list) else 0} 项，数量不符合工具格式；按格式要求的数量提交"
            )
        elif rule in ("minLength", "maxLength"):
            text = (
                location
                + f"：文字有 {len(value) if isinstance(value, str) else 0} 个字，不符合该字段的字数要求；按格式规定调整"
            )
        elif rule == "enum":
            text = location + "：填写的值不在格式允许的选项里；从该字段的规定选项中选择"
        elif rule == "type":
            text = location + "：值的类型不对；按格式要求提交文字、数字、列表或对象"
        elif rule in ("pattern", "format"):
            text = location + "：填写方式不符合格式；按该字段规定的写法提交"
        else:
            text = location + "：不符合工具格式；只修正这一项的格式，其他合法内容照旧"
        messages.append(text)
    return (
        original
        + "\n\n【格式修正，不重新判断内容】\n"
        + (
            "上一份候选因格式不合要求而未被接收。只修正下面列出的格式错误；"
            "保留已经选择的事实和回忆、合法字段的值及列表项目，不重新决定是否值得记，"
            "不要用空列表替代已有项目。修正后的工具参数仍须完整符合原格式。\n"
            "只有下面明确指出原文不匹配的摘录可以改成日记中确实存在的原句，其他合法内容照旧。\n"
            "【需要修正的问题】\n"
            + "\n".join(f"{i}. {text}" for i, text in enumerate(messages, 1))
            + "\n【待修正的原候选】\n"
            + json.dumps(anchor, ensure_ascii=False)
        )
    )


class ModelCaller:
    def __init__(self, settings, reasoning_effort="high", before_request=None):
        if runtime.cloud_runner.claude_transport.enabled(settings):
            raise runtime.Layer2RuntimeError("v2_provider_not_supported")
        self.settings = settings
        self.reasoning_effort = reasoning_effort
        self.before_request = before_request
        self.attempt_count = 0
        self._evidence_source = None

    def set_evidence_source(self, diary_text):
        self._evidence_source = diary_text

    def call(self, kind, system_prompt, task_prompt, attempts_dir):
        settings = self.settings
        definition = schema(kind)
        url = runtime.cloud_runner._request_url_for_settings(settings)
        if settings.llm_provider == "deepseek_llm":
            function = dict(
                name=TOOL_NAMES[kind],
                description=TOOL_DESCRIPTIONS[kind],
                strict=True,
                parameters=deepseek_request_schema(definition),
            )
        else:
            function = dict(
                name=TOOL_NAMES[kind],
                description=TOOL_DESCRIPTIONS[kind],
                parameters=dict(
                    type="object",
                    additionalProperties=False,
                    required=["result_json"],
                    properties={
                        "result_json": dict(
                            type="string",
                            description="严格符合任务格式的完整 JSON 对象文本",
                        )
                    },
                ),
            )
            task_prompt += "\n\n【输出格式】\n" + json.dumps(
                definition, ensure_ascii=False
            )
        payload = dict(
            model=settings.model,
            messages=[
                dict(role="system", content=system_prompt),
                dict(role="user", content=task_prompt),
            ],
            max_tokens=settings.max_output_tokens,
            tools=[dict(type="function", function=function)],
        )
        if settings.llm_provider == "deepseek_llm":
            payload.update(
                thinking=dict(type="enabled"), reasoning_effort=self.reasoning_effort
            )
        else:
            payload.update(
                temperature=settings.temperature,
                tool_choice=dict(type="function", function=dict(name=TOOL_NAMES[kind])),
            )
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + settings.llm_api_key,
        }
        for key, header in [
            ("organization_id", "OpenAI-Organization"),
            ("project_id", "OpenAI-Project"),
        ]:
            if getattr(settings, key, None):
                headers[header] = getattr(settings, key)
        anchor = None
        anchor_attempt = None
        anchor_rules = []
        anchor_issues = []
        repair_issues = []
        for attempt in range(1, 4):
            attempt_payload = copy.deepcopy(payload)
            if anchor_attempt is not None:
                attempt_payload["messages"][1]["content"] = _format_repair_task(
                    task_prompt, anchor, repair_issues
                )
            attempt_dir = Path(attempts_dir) / f"attempt_{attempt:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
            if repair_issues:
                save_json(
                    attempt_dir / "validation.json", dict(format_issues=repair_issues)
                )
            if self.before_request:
                self.before_request(kind, attempt_payload, attempt_dir)
            self.attempt_count += 1
            save_json(
                attempt_dir / "request_metadata.json",
                dict(
                    attempt=attempt,
                    model=settings.model,
                    provider=settings.llm_provider,
                    base_url=runtime.cloud_runner._safe_base_url(settings.base_url),
                    request_endpoint=runtime.cloud_runner._safe_base_url(url),
                    request_payload=attempt_payload,
                    credential_written=False,
                    timeout_seconds=settings.timeout_seconds,
                    format_anchor_attempt=anchor_attempt,
                ),
            )
            try:
                response = runtime.provider_adapter._post_json(
                    url,
                    attempt_payload,
                    headers,
                    settings.timeout_seconds,
                    attempt_dir / "response.json",
                    proxy_url=settings.proxy_url,
                )
                choice = response["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise runtime.Layer2RuntimeError("v2_output_truncated")
                calls = choice["message"].get("tool_calls", [])
                if len(calls) != 1 or calls[0]["function"]["name"] != TOOL_NAMES[kind]:
                    raise runtime.Layer2RuntimeError("v2_expected_tool_missing")
                value = json.loads(calls[0]["function"]["arguments"])
                if settings.llm_provider != "deepseek_llm":
                    value = json.loads(value["result_json"])
                issues = _format_issues(definition, value)
                if issues:
                    if anchor_attempt is None:
                        anchor = copy.deepcopy(value)
                        anchor_attempt = attempt
                        anchor_rules = _format_preservation_rules(
                            definition,
                            anchor,
                            issues,
                            self._evidence_source if kind == "ledger_update" else None,
                        )
                        anchor_issues = [
                            dict(
                                path=list(e.absolute_path),
                                rule=e.validator,
                                message=e.message
                                if e.validator == "additionalProperties"
                                else e.validator,
                            )
                            for e in issues[:20]
                        ]
                        anchor_issues += [
                            dict(path=list(path), rule="invalid_evidence_quote")
                            for rule, path, _ in anchor_rules
                            if rule == "evidence"
                        ]
                    current_issues = [
                        dict(
                            path=list(e.absolute_path),
                            rule=e.validator,
                            message=e.message
                            if e.validator == "additionalProperties"
                            else e.validator,
                        )
                        for e in issues[:20]
                    ]
                    repair_issues = anchor_issues + [
                        item for item in current_issues if item not in anchor_issues
                    ]
                    raise runtime.Layer2RuntimeError(
                        "v2_schema_invalid:" + str(list(issues[0].absolute_path))
                    )
                validate_schema(kind, value)
                changed = _format_changed_paths(value, anchor_rules)
                if changed:
                    # Each retry still repairs the original anchor. Retain its
                    # schema errors when also reporting unwanted content edits.
                    repair_issues = list(anchor_issues)
                    for path in changed[:20]:
                        expected = anchor
                        for part in path:
                            expected = expected[part]
                        repair_issues.append(
                            dict(
                                path=path,
                                rule="restore_original_valid_content",
                                expected=expected,
                            )
                        )
                    raise runtime.Layer2RuntimeError(
                        "v2_format_repair_changed_content:" + str(changed[:20])
                    )
                save_json(attempt_dir / "tool_arguments.json", value)
                return value
            except Exception as exc:
                if getattr(exc, "fatal_trial", False):
                    raise
                save_json(
                    attempt_dir / "error.json",
                    dict(error_type=type(exc).__name__, reason=str(exc)),
                )
                if attempt < 3:
                    time.sleep(min(attempt * 2, 5))
        raise runtime.Layer2RuntimeError("v2_call_failed_three_attempts:" + kind)


def publish_v2(
    layer2_root, state, projection, as_of, diary_sha256, run_id, metrics, provider_safe
):
    root = Path(layer2_root)
    publications = root / "publications"
    publications.mkdir(parents=True, exist_ok=True)
    staging = publications / (".staging-" + run_id)
    target = publications / (iso(as_of) + "_" + run_id)
    pointer = root / "current.json"
    previous = pointer.read_bytes() if pointer.exists() else None
    if staging.exists() or target.exists():
        raise runtime.Layer2RuntimeError("publication_run_id_conflict")
    staging.mkdir()
    names = {
        "background": "background.json",
        "projection_result": "projection_result.json",
        "model_facing_overview": "model_facing_overview.md",
        "model_context_preview": "model_context_preview.md",
        "publication_metrics": "publication_metrics.json",
    }
    try:
        save_json(staging / "background.json", state)
        save_json(staging / "projection_result.json", projection)
        save_json(staging / "publication_metrics.json", metrics)
        overview = render_overview(projection)
        (staging / "model_facing_overview.md").write_text(
            overview, encoding="utf-8", newline="\n"
        )
        (staging / "model_context_preview.md").write_text(
            format_layer2_context(iso(as_of), overview) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        examples = root / "v2_examples.md"
        prompt_hash = {
            k: hashlib.sha256(prompt(k).encode()).hexdigest() for k in TOOL_NAMES
        }
        prompt_hash["examples"] = (
            hashlib.sha256(
                examples.read_text(encoding="utf-8").replace("\r\n", "\n").encode()
            ).hexdigest()
            if examples.exists()
            else None
        )
        manifest = dict(
            schema_version=PUBLICATION_VERSION,
            pipeline="v2",
            status="published",
            published_at=runtime._iso_now(),
            as_of_date=iso(as_of),
            diary_sha256=diary_sha256,
            provider=provider_safe,
            prompt_sha256=prompt_hash,
            schema_sha256={
                k: hashlib.sha256(
                    (ROOT / "layer2_v2_schemas" / (k + ".json"))
                    .read_text(encoding="utf-8")
                    .replace("\r\n", "\n")
                    .encode()
                ).hexdigest()
                for k in TOOL_NAMES
            },
            files={
                k: dict(name=name, sha256=sha256_file(staging / name))
                for k, name in names.items()
            },
        )
        save_json(staging / "manifest.json", manifest)
        staging.rename(target)
        runtime._write_json_atomic(
            pointer,
            dict(
                schema_version=runtime.POINTER_SCHEMA_VERSION,
                updated_at=runtime._iso_now(),
                as_of_date=iso(as_of),
                publication=target.relative_to(root).as_posix(),
                manifest_sha256=sha256_file(target / "manifest.json"),
            ),
        )
        loaded = load_layer2_publication(root.parent)
        if loaded.diagnostics["status"] != "loaded":
            raise runtime.Layer2RuntimeError("post_publication_verification_failed")
        runtime._write_bytes_atomic(
            root / "current_layer2.md", (loaded.context.rstrip() + "\n").encode()
        )
    except Exception:
        if previous is None:
            pointer.unlink(missing_ok=True)
        else:
            runtime._write_bytes_atomic(pointer, previous)
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target


def run_daily_layer2_update_v2(memory_day, history_root, config_path, settings):
    history = Path(history_root).resolve()
    root = history / "layer2"
    current = day(memory_day)
    status = runtime.layer2_publication_status(memory_day, history)
    if status == "current":
        return dict(status="already_current", memory_day=memory_day)
    if status == "stale":
        raise runtime.Layer2RuntimeError(
            "same_day_diary_changed_manual_review_required"
        )
    if status == "invalid":
        raise runtime.Layer2RuntimeError("current_layer2_publication_invalid")
    loaded = load_layer2_publication(history)
    if loaded.diagnostics["status"] == "missing":
        raise runtime.Layer2RuntimeError("v2_bootstrap_not_supported")
    manifest = runtime._read_json(loaded.background_path.parent / "manifest.json")
    if manifest.get("pipeline") != "v2":
        raise runtime.Layer2RuntimeError("v2_requires_v2_base_publication")
    state = runtime._read_json(loaded.background_path)
    projection = runtime._read_json(
        loaded.background_path.parent / "projection_result.json"
    )
    if current <= day(state["as_of_date"]):
        raise runtime.Layer2RuntimeError("v2_day_must_advance")
    diary = history / "diaries" / ("diary_" + memory_day + ".txt")
    if not diary.is_file() or not diary.read_text(encoding="utf-8").strip():
        raise runtime.Layer2RuntimeError("diary_missing_or_empty")
    text = diary.read_text(encoding="utf-8")
    diary_hash = sha256_file(diary)
    pointer = root / "current.json"
    pointer_before = pointer.read_bytes()
    cloud_settings, safe = runtime._settings_for_cloud(Path(config_path), settings)
    examples_path = root / "v2_examples.md"
    if not examples_path.exists():
        logger.warning("Layer 2 v2 example file is missing")
    examples = (
        examples_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if examples_path.exists()
        else ""
    )
    with runtime._daily_lock(root, memory_day):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:10]
        run_dir = root / "runs" / memory_day / run_id
        save_json(
            run_dir / "run.json",
            dict(
                status="running",
                memory_day=memory_day,
                provider=safe,
                diary_sha256=diary_hash,
            ),
        )
        try:
            caller = ModelCaller(cloud_settings, settings.reasoning_effort)
            edited = runtime._extract_model_facing_overview(
                load_current_layer2_context(history).context
            )
            state, projection, metrics = process_day(
                state, projection, current, text, edited, caller, examples, run_dir
            )
            if (
                pointer.read_bytes() != pointer_before
                or sha256_file(diary) != diary_hash
            ):
                raise runtime.Layer2RuntimeError("layer2_source_changed_before_publish")
            metrics["api_attempt_count"] = caller.attempt_count
            save_json(run_dir / "metrics.json", metrics)
            target = publish_v2(
                root, state, projection, current, diary_hash, run_id, metrics, safe
            )
            pruned = runtime._prune_retained_runtime(
                root, current, settings.audit_retention_days
            )
            result = dict(
                status="published",
                memory_day=memory_day,
                publication=str(target),
                pipeline="v2",
                pruned=pruned,
            )
            save_json(run_dir / "run.json", result)
            return result
        except Exception as exc:
            save_json(
                run_dir / "run.json",
                dict(
                    status="failed",
                    memory_day=memory_day,
                    error_type=type(exc).__name__,
                    error=str(exc),
                ),
            )
            raise
