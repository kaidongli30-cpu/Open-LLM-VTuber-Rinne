"""Lightweight structural checks for weekly and monthly memory summaries."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .periodic_theme_pipeline import (
    NO_CONTINUOUS_MONTH_BODY,
    NO_CONTINUOUS_WEEK_BODY,
)


_THEME_WITH_DATE = re.compile(
    r"^\s*【[^】\n]*[｜|][^】\n]*\d{1,2}月\d{1,2}日[^】\n]*】\s*$",
    re.MULTILINE,
)
_ANY_BRACKET_HEADING = re.compile(r"^\s*【[^】\n]+】\s*$", re.MULTILINE)
_DAILY_HEADING = re.compile(
    r"^\s*(?:周|星期)[一二三四五六日天]\s*[（(]?\s*\d{1,2}月\d{1,2}日",
    re.MULTILINE,
)


@dataclass(frozen=True)
class PeriodicSummaryReview:
    issues: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.issues

    def retry_feedback(self) -> str:
        return "；".join(self.issues)


def review_periodic_summary(
    body: str,
    *,
    memory_kind: str,
    source_count: int,
) -> PeriodicSummaryReview:
    """Reject the old diary-concatenation shape without judging prose length."""

    issues: list[str] = []
    if not body.strip():
        issues.append("没有生成正文")
        return PeriodicSummaryReview(tuple(issues))

    if body.strip() in {NO_CONTINUOUS_WEEK_BODY, NO_CONTINUOUS_MONTH_BODY}:
        return PeriodicSummaryReview()

    character_limit = 4500 if memory_kind == "周记" else 5000
    if len(body) > character_limit:
        issues.append(
            f"正文有{len(body)}个字符，超过{memory_kind}当前{character_limit}字符的验收上限；"
            "应删除来源复述，只保留主题脉络、关键日期、转折、结果和感情意义"
        )

    theme_count = len(_THEME_WITH_DATE.findall(body))
    bracket_heading_count = len(_ANY_BRACKET_HEADING.findall(body))
    if theme_count == 0:
        issues.append("没有使用“【主题｜关键日期：几月几日】”主题标题")
    elif bracket_heading_count != theme_count:
        issues.append("存在没有日期索引的额外标题")

    daily_heading_count = len(_DAILY_HEADING.findall(body))
    if memory_kind == "周记" and source_count >= 4 and daily_heading_count >= 4:
        issues.append("仍按周一到周日逐日分节，像多篇日记的拼接")
    elif memory_kind == "月记" and daily_heading_count >= 7:
        issues.append("仍以大量逐日章节组织，未形成月度主题脉络")

    return PeriodicSummaryReview(tuple(issues))
