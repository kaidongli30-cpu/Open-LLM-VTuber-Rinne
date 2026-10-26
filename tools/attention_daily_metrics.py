"""Read-only attention metrics for Rinne's daily trial and debug logs."""

import argparse
import json
import re
from datetime import date, timedelta
from pathlib import Path


# Frozen echo_values, is_echo, compact, common_substring are inserted here.
def compact(s):
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", s))


def common_substring(a, b):
    best = 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        current = [0] * (len(b) + 1)
        for i, cb in enumerate(b, 1):
            if ca == cb:
                current[i] = prev[i - 1] + 1
                best = max(best, current[i])
        prev = current
    return best


def echo_values(user, reply):
    cleaned = re.sub(r"\[[A-Za-z_]+\]", "", reply)
    while True:
        updated = re.sub(r"（[^（）]*）|\([^()]*\)", "", cleaned)
        if updated == cleaned:
            break
        cleaned = updated
    cleaned = cleaned.lstrip()
    m = re.search(r"[。！？!?～~…\n]", cleaned)
    first = cleaned[: m.start()] if m else cleaned[:40]
    a, b = compact(user), compact(first)
    length = common_substring(a, b)
    return dict(
        common_length=length, first_length=len(b), fraction=length / len(b) if b else 0
    )


def is_echo(values, min_length=4, min_ratio=0.35):
    return values["common_length"] >= min_length and values["fraction"] >= min_ratio


THRESHOLDS = {"min_length": 3, "min_ratio": 0.45}


def collect(root: Path, start: date, end: date):
    metrics = {}
    day = start
    while day <= end:
        metrics[day.isoformat()] = dict(
            day=day.isoformat(),
            turns=0,
            rewrite_events=0,
            rewrite_event_types={},
            opening_echoes=0,
            fallbacks=0,
            card_injections=0,
            card_statuses={},
            trial_files=0,
            debug_lines=0,
            malformed_trial_lines=0,
        )
        day += timedelta(days=1)
    for p in sorted((root / "chat_history/rinne_01/memory_trial_logs").glob("*.jsonl")):
        match = re.search(r"\d{4}-\d{2}-\d{2}", p.name)
        if not match or match[0] not in metrics:
            continue
        row = metrics[match[0]]
        row["trial_files"] += 1
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    row["malformed_trial_lines"] += 1
                    continue
                row["turns"] += 1
                reply = record.get("assistant_response") or ""
                row["opening_echoes"] += int(
                    is_echo(
                        echo_values(record.get("user_input") or "", reply), **THRESHOLDS
                    )
                )
                row["fallbacks"] += int("抱歉，刚才云端没有返回有效内容" in reply)
                attention = record.get("attention_slot") or {}
                row["card_injections"] += int(bool(attention.get("card_injected")))
                status = attention.get("card_status")
                if status is not None:
                    row["card_statuses"][status] = (
                        row["card_statuses"].get(status, 0) + 1
                    )
    needles = [
        "检测到明显重复句，已让模型整段重写一次",
        "重复回复整段重写等待超过",
        "重复回复整段重写失败",
        "重写结果未通过语言与身份检查",
        "重复内容已整段重写一次",
        "重复回复重写等待超过",
        "重复回复重写失败",
        "重写结果未通过检查",
        "重复问句已重写或移除",
    ]
    for p in sorted((root / "logs").glob("debug_*.log")):
        with p.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                match = re.search(r"\d{4}-\d{2}-\d{2}", line[:40])
                if not match or match[0] not in metrics:
                    continue
                row = metrics[match[0]]
                row["debug_lines"] += 1
                for needle in needles:
                    if needle in line:
                        row["rewrite_events"] += 1
                        row["rewrite_event_types"][needle] = (
                            row["rewrite_event_types"].get(needle, 0) + 1
                        )
    return list(metrics.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.end < args.start:
        parser.error("--end must not precede --start")
    data = collect(args.project, args.start, args.end)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            dict(
                start=args.start.isoformat(),
                end=args.end.isoformat(),
                echo_thresholds=THRESHOLDS,
                rewrite_log_notice="2026-10-21 起去重日志写法变化，之前与之后的计数不能直接比较。",
                days=data,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(dict(days=len(data), out=str(args.out))))


if __name__ == "__main__":
    main()
