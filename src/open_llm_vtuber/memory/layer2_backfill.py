"""Incrementally build Layer-2 background from approved local diaries.

This command never copies or uploads diary files outside the configured model
request. It refuses to overwrite a newer or invalid publication.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from ..data_paths import resolve_character_history_root
from .diary_review import load_matching_approval, wait_for_diary_approval
from .layer2_context import load_layer2_publication
from .layer2_runtime import run_daily_layer2_update


DIARY_NAME = re.compile(r"diary_(\d{4}-\d{2}-\d{2})\.txt\Z")


class Layer2BackfillError(RuntimeError):
    """A migration precondition failed without changing the publication."""


def _last_complete_day(now: datetime) -> date:
    return (now - timedelta(days=2 if now.hour < 3 else 1)).date()


def plan_backfill(
    history_root: str | Path | None = None,
    *,
    reference_time: datetime | None = None,
    approve_interactively: bool = False,
) -> dict[str, Any]:
    history = resolve_character_history_root(history_root).resolve()
    publication = load_layer2_publication(history)
    publication_status = publication.diagnostics.get("status")
    if publication_status == "invalid":
        raise Layer2BackfillError("当前第二层发布未通过校验；请先人工检查")
    published_day = (
        date.fromisoformat(str(publication.diagnostics["as_of_date"]))
        if publication_status == "loaded"
        else None
    )
    cutoff = _last_complete_day(reference_time or datetime.now())
    selected: list[str] = []
    needs_review: list[str] = []
    for diary in sorted((history / "diaries").glob("diary_*.txt")):
        match = DIARY_NAME.fullmatch(diary.name)
        if match is None or not diary.is_file() or diary.stat().st_size == 0:
            continue
        try:
            day = date.fromisoformat(match.group(1))
        except ValueError:
            continue
        if day > cutoff or (published_day is not None and day <= published_day):
            continue
        day_text = day.isoformat()
        approval = load_matching_approval(history, day_text, diary)
        if approval is None and approve_interactively:
            wait_for_diary_approval(history, day_text, diary)
            approval = load_matching_approval(history, day_text, diary)
        if approval is None:
            needs_review.append(day_text)
            continue
        selected.append(day_text)
    if needs_review and selected:
        first_unreviewed = min(needs_review)
        selected = [day for day in selected if day < first_unreviewed]
    return {
        "history_root": str(history),
        "published_through": published_day.isoformat() if published_day else None,
        "eligible_days": selected,
        "needs_review": needs_review,
        "cutoff": cutoff.isoformat(),
    }


def run_backfill(
    history_root: str | Path | None = None,
    *,
    config_path: str | Path = "conf.yaml",
    reference_time: datetime | None = None,
    approve_interactively: bool = False,
) -> dict[str, Any]:
    plan = plan_backfill(
        history_root,
        reference_time=reference_time,
        approve_interactively=approve_interactively,
    )
    published: list[str] = []
    for day in plan["eligible_days"]:
        outcome = run_daily_layer2_update(
            day,
            history_root=plan["history_root"],
            config_path=config_path,
        )
        if outcome.get("status") not in {"published", "already_current"}:
            raise Layer2BackfillError(f"{day} 未完成发布：{outcome.get('status')}")
        published.append(day)
    return {**plan, "published_days": published}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从已验收日记补齐第二层用户背景")
    parser.add_argument("--history-root", type=Path)
    parser.add_argument("--config-path", type=Path, default=Path("conf.yaml"))
    parser.add_argument("--approve-interactively", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        if arguments.dry_run:
            result = plan_backfill(
                arguments.history_root,
                approve_interactively=False,
            )
        else:
            result = run_backfill(
                arguments.history_root,
                config_path=arguments.config_path,
                approve_interactively=arguments.approve_interactively,
            )
    except (Layer2BackfillError, OSError, ValueError) as exc:
        print(f"第二层背景补齐失败：{exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
