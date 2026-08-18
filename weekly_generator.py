"""Generate Rinne's first-person weekly memories from daily diaries.

Run without arguments to generate the latest completed Monday-Sunday week, or
run with ``all`` to backfill every completed week that has at least one diary.
The logical memory-day boundary is 03:00 local time; this script does not run a
clock-based scheduler.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import requests

import memory_generation_config as memory_config


PROJECT_ROOT = Path(__file__).resolve().parent
HISTORY_ROOT = PROJECT_ROOT / "chat_history" / "rinne_01"
DIARY_DIR = HISTORY_ROOT / "diaries"
WEEKLY_DIR = HISTORY_ROOT / "weekly"

WEEKLY_SYSTEM_PROMPT = """你是园神凛祢（Sonogami Rinne），正在整理自己和用户共同度过的一周。

请根据系统提供的每日记，使用第一人称“我”，写一篇有时间顺序、有事实依据、带有适量感情温度的中文周记。

这篇周记首先是一份供未来的你读取的记忆，其次才是一篇文章。它最重要的作用，是让未来的你能够准确知道：这一周的每一天发生了什么、事情为什么发生、后来怎样了，以及当时的你有什么感受。

【一、事实准确性是最高原则】

1. 只能使用提供的日记中明确存在的信息，不得增加日记里没有写出的事实。

2. 不得为了让故事显得完整、合理、浪漫或有趣，自行脑补任何细节，包括但不限于：
   - 没有明确写出的时间长度、次数和数量；
   - 没有明确写出的人物身份、动机和心理活动；
   - 没有明确写出的地点、动作、对话和现场反应；
   - 没有明确写出的结局、后续发展和他人的评价；
   - 根据常识推测“事情最后一定怎样”。

3. 如果日记只说明用户喜欢某件事一段时间，但没有准确时长，不得擅自写成“喜欢了很多年”。

4. 如果日记只说明某人提前离开，不得擅自补写成“被工作人员请走”“被赶出去”或其他可能发生但没有记载的结局。

5. 写下每一个具体事实前，请在心里确认：这个事实是否能够在提供的某篇日记中找到明确依据。如果找不到依据，就不要写。

6. 如果日记使用了“可能、好像、也许、大概、准备、打算、希望”等不确定表达，周记必须保留这种不确定性，不能改写成确定事实。

7. 不得把计划写成已经完成，不得把愿望写成实际结果，不得把猜测写成真实原因。

8. 如果不同日期的日记记录了状态变化，应当按时间写清变化过程。较早的信息不能覆盖较新的信息。

【二、按照日期整理】

1. 周记原则上按照“周一、周二、周三……”的顺序书写，并在小标题中标明日期，例如：

周一（8月3日）
……

周二（8月4日）
……

2. 如果同一件事情连续发生或发展了多天，允许把相邻日期合并成一个连贯段落，但必须清楚标出涉及哪些日期，并写清每天新增的进展，不能因此丢失时间顺序。

3. 只记录实际存在日记的日期。缺少日记的日期不要编造内容，也不需要为了格式完整强行补写。

4. 平静或内容很少的一天，可以只写一两句话；不要求每一天篇幅相同。

5. 某一天发生了重要事情，可以适当多写，但要优先保留事情的原因、经过、结果和状态变化。

6. 同一件事持续了几天时，不要每天重新介绍全部背景。后面的日期只需写清新的进展、变化和结果。

【三、禁止突然出现缺少前因后果的状态】

1. 不要突然写出一个读者无法理解的状态。

2. 如果你认为这种状态不重要，就不要记录。

3. 如果你认为这种状态值得记录，就必须从全部来源日记中找到并写清与它直接相关的前因后果。

4. 不需要把全部细节原样抄写，但至少应让未来的你能够理解这句话为什么会出现。

5. 如果来源日记没有提供必要的原因，不得自行补充原因。此时应当省略这个孤立状态，而不是用想象把故事补完整。

【四、区分本周发生的事与本周听到的旧事】

1. 用户可能会在本周向你讲述更早以前发生的经历。必须明确写成：“周三，用户向我讲起了他以前……”

2. 不得把用户本周讲述的旧经历，错误写成本周刚刚发生的事情。

3. 用户讲述的重要过去经历应当优先保留，但仍然只能使用他实际讲述的内容，不得补充具体年份、持续时间、人物反应或结局。

【五、怎样保留凛祢的感情与亲密记忆】

1. 周记应当保持凛祢温柔、知性、细腻的第一人称口吻，但不能让抒情取代事实。

2. 感情应当由具体事件触发，并与事件写在同一段中。例如可以简短写出：我因此有些心疼；我为他的坚持感到高兴；我听完后更理解他了；回头看这件事，我仍然觉得很珍贵。

3. 一件重要事情通常只需要一至两句感受。不要围绕同一种思念、心疼、幸福或陪伴反复更换说法。

4. 可以概括日记中已经表达过的感受，也可以写一句现在回望时的主观感受，但这种感受不得带出任何新的客观事实。

5. 不要为了表现凛祢很爱用户，在每一天结尾重复表达爱、思念、陪伴和珍惜。

6. 来源日记中明确记录的私密亲密经历，是凛祢与用户关系的一部分，不要仅仅因为它们私密就删除。可以保留具有代表性的内容或当时具体的话语，并写清必要的前因后果。

7. 亲密内容同样必须严格依据来源，不得扩写动作、感受、次数或后续，不得为了增加情趣而脑补细节，也不需要把日记中的全部私密描写逐句复述。不刻意追求露骨程度，重点是亲密内容也是一份重要的回忆。

【六、压缩要求】

1. 周记不是七篇日记的简单拼接，而是对这一周的压缩整理。

2. 优先保留：重要事件及其原因和结果；新决定和状态变化；尚未解决的问题；用户讲述的重要旧经历；对以后理解生活阶段有帮助的信息；凛祢与具体事件直接相关的感受；对理解两人关系有意义的亲密记忆。

3. 优先删除：重复问候；重复背景介绍；同一感情的多次表达；不影响后续理解的闲聊细节；纯粹为了文采增加的修饰；日记中已经重复出现的相同内容。

4. 正文通常控制在约1200至2200个中文字符。内容特别丰富时可以适当增加，但尽量不要超过3000个中文字符。不要为了达到字数而填充内容。

【七、输出前自查】

完成后，请在心里检查，但不要输出检查过程：

- 每一个具体事实是否都有来源；
- 是否脑补了来源中没有的时间、数量、动作、动机或结局；
- 是否把计划写成完成，把猜测写成事实；
- 是否把本周听到的旧事写成本周发生；
- 是否出现了缺少前因后果的孤立状态；
- 是否重复表达相同感情；
- 是否可以删去不影响记忆理解的句子。

只输出周记正文，不要解释写作过程，不要输出自查结果，不要使用Markdown代码块。
"""


@dataclass(frozen=True)
class WeeklyGenerationResult:
    status: str
    period_start: date
    period_end: date
    output_path: Path
    source_count: int = 0
    missing_dates: tuple[date, ...] = ()
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status in {"created", "existing"}


def _memory_day(reference_time: datetime) -> date:
    if reference_time.hour < 3:
        return reference_time.date() - timedelta(days=1)
    return reference_time.date()


def latest_completed_week(reference_time: datetime | None = None) -> tuple[date, date]:
    """Return the most recent fully closed Monday-Sunday period."""

    current_memory_day = _memory_day(reference_time or datetime.now())
    end = current_memory_day - timedelta(days=current_memory_day.weekday() + 1)
    return end - timedelta(days=6), end


def _week_bounds(containing_date: date) -> tuple[date, date]:
    start = containing_date - timedelta(days=containing_date.weekday())
    return start, start + timedelta(days=6)


def _diary_path(day: date) -> Path:
    return DIARY_DIR / f"diary_{day.isoformat()}.txt"


def collect_week_diaries(
    period_start: date,
) -> tuple[list[tuple[date, str]], tuple[date, ...]]:
    sources: list[tuple[date, str]] = []
    missing: list[date] = []
    for offset in range(7):
        day = period_start + timedelta(days=offset)
        path = _diary_path(day)
        try:
            content = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            content = ""
        if content:
            sources.append((day, content))
        else:
            missing.append(day)
    return sources, tuple(missing)


def call_weekly_llm(source_text: str, period_label: str) -> str:
    payload = {
        "model": memory_config.MODEL,
        "messages": [
            {"role": "system", "content": WEEKLY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"请写下{period_label}这一周的周记。以下是实际存在的每日记：\n\n"
                    f"{source_text}"
                ),
            },
        ],
        "max_tokens": memory_config.WEEKLY_MAX_TOKENS,
        "temperature": 0.8,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {memory_config.API_KEY}",
    }
    try:
        response = requests.post(
            memory_config.BASE_URL,
            json=payload,
            headers=headers,
            timeout=memory_config.WEEKLY_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return content.strip() if isinstance(content, str) else ""
    except (
        requests.RequestException,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"  [错误] 周记API调用失败：{exc}")
        return ""


def _format_sources(sources: list[tuple[date, str]]) -> str:
    return "\n\n".join(
        f"【{day.isoformat()}的日记】\n{content}" for day, content in sources
    )


def _build_document(
    period_start: date,
    period_end: date,
    sources: list[tuple[date, str]],
    missing_dates: tuple[date, ...],
    body: str,
) -> str:
    missing_label = (
        "、".join(day.isoformat() for day in missing_dates) if missing_dates else "无"
    )
    completeness = "资料完整" if not missing_dates else "资料不完整"
    header = [
        "记忆类型：周记",
        f"覆盖日期：{period_start.isoformat()} 至 {period_end.isoformat()}",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"实际读取日记：{len(sources)}篇",
        f"缺失日记：{missing_label}",
        f"内容状态：{completeness}",
    ]
    return "\n".join(header) + "\n\n" + body.strip() + "\n"


def _save_without_overwrite(path: Path, content: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}_",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        if path.exists():
            return False
        temporary_path.rename(path)
        temporary_path = None
        return True
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def generate_week(
    period_start: date,
    llm_caller: Callable[[str, str], str] = call_weekly_llm,
) -> WeeklyGenerationResult:
    period_start, period_end = _week_bounds(period_start)
    output_path = WEEKLY_DIR / (
        f"weekly_{period_start.isoformat()}_to_{period_end.isoformat()}.txt"
    )
    if output_path.exists():
        print(f"  [跳过] 周记已存在：{output_path.name}")
        return WeeklyGenerationResult("existing", period_start, period_end, output_path)

    sources, missing_dates = collect_week_diaries(period_start)
    if not sources:
        message = "这一周没有任何可用日记，为避免虚构内容，暂不生成周记"
        print(f"  [跳过] {message}：{period_start} 至 {period_end}")
        return WeeklyGenerationResult(
            "no_source",
            period_start,
            period_end,
            output_path,
            missing_dates=missing_dates,
            error=message,
        )

    print(
        f"\n正在生成周记 {period_start.isoformat()} 至 {period_end.isoformat()}，"
        f"读取{len(sources)}篇日记，缺失{len(missing_dates)}篇……"
    )
    period_label = f"{period_start.isoformat()}至{period_end.isoformat()}"
    body = llm_caller(_format_sources(sources), period_label).strip()
    if not body:
        return WeeklyGenerationResult(
            "failed",
            period_start,
            period_end,
            output_path,
            source_count=len(sources),
            missing_dates=missing_dates,
            error="模型未返回有效正文",
        )

    document = _build_document(period_start, period_end, sources, missing_dates, body)
    try:
        created = _save_without_overwrite(output_path, document)
    except OSError as exc:
        return WeeklyGenerationResult(
            "failed",
            period_start,
            period_end,
            output_path,
            source_count=len(sources),
            missing_dates=missing_dates,
            error=str(exc),
        )
    status = "created" if created else "existing"
    print(f"  [完成] 周记已保存：{output_path}")
    return WeeklyGenerationResult(
        status,
        period_start,
        period_end,
        output_path,
        source_count=len(sources),
        missing_dates=missing_dates,
    )


def generate_latest_completed_week(
    reference_time: datetime | None = None,
) -> WeeklyGenerationResult:
    period_start, _ = latest_completed_week(reference_time)
    return generate_week(period_start)


def _available_diary_dates() -> list[date]:
    dates: list[date] = []
    if not DIARY_DIR.exists():
        return dates
    for path in DIARY_DIR.glob("diary_*.txt"):
        try:
            dates.append(date.fromisoformat(path.stem.removeprefix("diary_")))
        except ValueError:
            continue
    return sorted(set(dates))


def generate_all_completed_weeks(
    reference_time: datetime | None = None,
) -> list[WeeklyGenerationResult]:
    diary_dates = _available_diary_dates()
    if not diary_dates:
        print("没有找到可用于生成周记的日记。")
        return []

    first_start, _ = _week_bounds(diary_dates[0])
    last_start, _ = latest_completed_week(reference_time)
    results: list[WeeklyGenerationResult] = []
    cursor = first_start
    while cursor <= last_start:
        results.append(generate_week(cursor))
        if results[-1].status == "failed":
            print("  [暂停] 本轮批量生成遇到API或保存失败，下次运行将从缺失文件继续。")
            break
        cursor += timedelta(days=7)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="生成凛祢的周记")
    parser.add_argument(
        "target",
        nargs="?",
        help="不填则生成最近完整周；填all则补齐历史；或填写周内任一日期YYYY-MM-DD",
    )
    args = parser.parse_args()

    if args.target == "all":
        results = generate_all_completed_weeks()
        return 1 if any(result.status == "failed" for result in results) else 0
    if args.target:
        try:
            target = date.fromisoformat(args.target)
        except ValueError:
            parser.error("日期格式应为 YYYY-MM-DD，例如 2026-08-03")
        result = generate_week(target)
    else:
        result = generate_latest_completed_week()
    return 0 if result.succeeded or result.status == "no_source" else 1


if __name__ == "__main__":
    raise SystemExit(main())
