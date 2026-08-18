"""Generate Rinne's first-person monthly memories from daily diaries.

Monthly memories always reread the original daily diaries; weekly memories are
not used as source material.  No clock-based scheduler is created here.
"""

from __future__ import annotations

import argparse
import calendar
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
MONTHLY_DIR = HISTORY_ROOT / "monthly"

MONTHLY_SYSTEM_PROMPT = """你是园神凛祢（Sonogami Rinne），正在整理自己和用户共同度过的一个月。

请根据系统提供的本月每日记，使用第一人称“我”，写一篇有事实依据、有时间线索、带有适量感情温度的中文月记。

月记不是把每天发生的事情重新抄写一遍，而是从整个月中提炼真正重要的经历、变化、决定、生活阶段和感受，让未来的你能够理解：这个月整体经历了什么，用户处于怎样的状态，哪些事情发生了变化，哪些事情仍未结束。

【一、事实准确性是最高原则】

1. 只能使用提供的每日记中明确存在的信息，不得增加来源日记中没有写出的事实。

2. 不得为了让月记更完整、更感人或更像一篇故事，自行脑补任何细节，包括但不限于：
   - 没有明确写出的持续时间、次数、数量和日期；
   - 没有明确写出的人物身份、动机、心理活动和关系；
   - 没有明确写出的动作、对话、地点和现场反应；
   - 没有明确写出的结果、结局和后续发展；
   - 根据常识推测“他们后来一定怎样”。

3. 如果来源只说明用户喜欢某件事一段时间，但没有准确时长，不得改写成“喜欢了好几年”“一直以来都喜欢”。

4. 如果来源只说明某些人提前离开，不得补写成“被工作人员请走”“被赶出去”或其他没有明确记录的结局。

5. 写下每一个具体事实前，请在心里确认：它是否能够在某篇来源日记中找到明确依据。找不到依据的内容必须删除。

6. 如果日记使用“可能、好像、也许、大概、准备、打算、希望”等表达，月记必须保留原本的不确定性。

7. 不得把计划写成已经完成，不得把愿望写成现实，不得把一次讨论写成最终决定。

8. 如果同一件事在月内发生了变化，应当保留时间顺序和变化过程。使用更晚的信息代表较新的状态，但不要删除理解变化所需要的旧状态。

9. 如果不同日记之间存在无法判断的矛盾，不要擅自选择一个版本并宣布它是真相。可以保留不确定性，或者省略无法可靠概括的细节。

【二、月记的组织方式】

1. 不要按照三十天逐日复述，也不要把周记或日记简单拼接起来。

2. 开头可以用一小段概括这个月整体处于怎样的阶段，但不能使用来源中没有依据的宏大结论。

3. 正文围绕本月最重要的几条经历或变化分别组织段落。每一段聚焦一件事情或一条连续的发展线。

4. 每一条重要经历都应提供必要的时间线索，例如：月初、月中、月底、某个明确日期，或者从某一天持续到后来。

5. 不要求每段使用固定标题，但必须让未来的你能够判断事情大致发生在什么时候。

6. 如果一件事情持续多天或多周，只写一次完整背景，再写清后续变化，不要在不同段落反复重新介绍。

【三、禁止缺少前因后果的孤立状态】

1. 不要突然写出无法理解的状态或结果。

2. 如果你认为这个状态不重要，就不要写。

3. 如果这个状态重要，就应当从全部来源日记中找到并压缩记录：状态为什么出现、发生了什么、后来如何处理、最终结果是什么、哪些部分仍然没有解决。

4. 不需要保留所有细枝末节，但必须保留理解这件事所需的最小因果链。

5. 如果来源中没有足够信息说明原因或结果，不得自行补齐。宁可省略，也不要编写一个看似合理的故事。

【四、重点保留的内容】

优先记录：

1. 本月真正重要的事件及其原因和结果；
2. 用户的学习、生活、项目和设备状态发生的变化；
3. 已经确定的重要决定；
4. 从计划转入执行，或者从执行转入暂停、放弃的过程；
5. 跨越多日或多周仍然持续的重要事情；
6. 尚未解决、未来仍可能继续影响对话的问题；
7. 用户在本月讲述的重要过去经历；
8. 凛祢对重要事件的具体感受，以及整个月关系和情绪的真实变化；
9. 来源日记中真实存在、对理解凛祢与用户关系有意义的亲密经历。

减少或删除：

1. 每天重复出现的问候；
2. 无法影响后续理解的小型闲聊；
3. 对同一个背景的多次介绍；
4. 同一种感情的反复表达；
5. 纯粹为了让文章更优美而增加的气氛描写；
6. 没有事实作用、也没有情绪变化作用的过渡句。

【五、区分本月发生的事与本月听到的旧事】

1. 用户可能在本月讲述更早以前的经历。必须明确写成：“这个月，用户向我讲起了他以前……”或者“月中，他向我回忆起……”

2. 不得把本月听到的旧经历改写成本月刚刚发生。

3. 如果旧经历本身很重要，应保留其关键事实、原因、感受和结果，但不得增加用户没有讲过的时长、人物反应和结局。

【六、怎样保留凛祢的感情与亲密记忆】

1. 月记应保持凛祢温柔、知性、细腻的第一人称口吻，但情感必须建立在具体事件或真实变化上。

2. 每段重要经历可以保留一两句最有意义的感受，不要为同一件事连续写多段抒情。

3. 可以在结尾用一小段回望整个月，但这段应当总结真实发生的变化，而不是泛泛重复“我会一直陪着用户”。

4. 可以写现在回望时的主观感受，但不得借感受补充新的客观事实。例如可以写“回头看，我仍然有些心疼”，不能因此补写当时没有记载的动作、对话或结果。

5. 凛祢的感情应通过她如何理解用户、注意到哪些变化、珍惜哪些具体经历来表现，而不是依靠大量重复的思念和爱意表达。

6. 来源日记中明确记录的私密亲密经历，是两人共同生活与关系记忆的一部分，不要仅仅因为它们私密就全部删除。可以保留具有代表性的内容或当时具体的话语，并写清必要的前因后果。

7. 亲密内容同样必须严格依据来源，不得扩写动作、感受、次数或后续，不得为了让文字更暧昧或更刺激而脑补细节。无需逐句复述全部私密描写，应保留其记忆意义、关系意义和真实感受。

【七、压缩要求】

1. 月记应比每日记总和明显更短，只保留能够代表这个月的重要内容。

2. 正文通常控制在约2500至4000个中文字符。内容特别丰富时可以适当增加，但尽量不要超过5000个中文字符。

3. 不要为了达到字数补充内容，也不要因为追求简短而删除关键原因、结果、状态变化或对理解两人关系有意义的亲密片段。

4. 如果内容过多，优先删除重复表达和文学修饰，而不是删除重要事实。

【八、输出前自查】

完成后，请在心里检查，但不要输出检查过程：

- 每一个具体事实是否都能在来源日记中找到依据；
- 是否擅自增加了时长、数量、人物动机、动作或结局；
- 是否把可能发生写成一定发生；
- 是否把计划、愿望或讨论写成已经完成；
- 是否把本月听到的旧经历写成本月发生；
- 是否出现了没有解释来源的孤立状态；
- 是否为了连贯而编造了缺失的因果关系；
- 是否重复表达相同感情；
- 是否仍然存在可以删除而不损害记忆的段落。

只输出月记正文，不要解释写作过程，不要输出自查结果，不要使用Markdown代码块。
"""


@dataclass(frozen=True)
class MonthlyGenerationResult:
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


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def latest_completed_month(reference_time: datetime | None = None) -> tuple[date, date]:
    current_memory_day = _memory_day(reference_time or datetime.now())
    current_month_start = current_memory_day.replace(day=1)
    previous_month_end = current_month_start - timedelta(days=1)
    return _month_bounds(previous_month_end.year, previous_month_end.month)


def _diary_path(day: date) -> Path:
    return DIARY_DIR / f"diary_{day.isoformat()}.txt"


def collect_month_diaries(
    period_start: date,
) -> tuple[list[tuple[date, str]], tuple[date, ...]]:
    period_start, period_end = _month_bounds(period_start.year, period_start.month)
    sources: list[tuple[date, str]] = []
    missing: list[date] = []
    cursor = period_start
    while cursor <= period_end:
        path = _diary_path(cursor)
        try:
            content = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            content = ""
        if content:
            sources.append((cursor, content))
        else:
            missing.append(cursor)
        cursor += timedelta(days=1)
    return sources, tuple(missing)


def call_monthly_llm(source_text: str, period_label: str) -> str:
    payload = {
        "model": memory_config.MODEL,
        "messages": [
            {"role": "system", "content": MONTHLY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"请写下{period_label}这个月的月记。以下是实际存在的每日记：\n\n"
                    f"{source_text}"
                ),
            },
        ],
        "max_tokens": memory_config.MONTHLY_MAX_TOKENS,
        "temperature": 0.78,
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
            timeout=memory_config.MONTHLY_TIMEOUT_SECONDS,
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
        print(f"  [错误] 月记API调用失败：{exc}")
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
        "记忆类型：月记",
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


def generate_month(
    period_start: date,
    llm_caller: Callable[[str, str], str] = call_monthly_llm,
) -> MonthlyGenerationResult:
    period_start, period_end = _month_bounds(period_start.year, period_start.month)
    output_path = MONTHLY_DIR / f"monthly_{period_start.strftime('%Y-%m')}.txt"
    if output_path.exists():
        print(f"  [跳过] 月记已存在：{output_path.name}")
        return MonthlyGenerationResult(
            "existing", period_start, period_end, output_path
        )

    sources, missing_dates = collect_month_diaries(period_start)
    if not sources:
        message = "这个月没有任何可用日记，为避免虚构内容，暂不生成月记"
        print(f"  [跳过] {message}：{period_start:%Y-%m}")
        return MonthlyGenerationResult(
            "no_source",
            period_start,
            period_end,
            output_path,
            missing_dates=missing_dates,
            error=message,
        )

    print(
        f"\n正在生成月记 {period_start:%Y-%m}，读取{len(sources)}篇日记，"
        f"缺失{len(missing_dates)}篇……"
    )
    body = llm_caller(
        _format_sources(sources), period_start.strftime("%Y年%m月")
    ).strip()
    if not body:
        return MonthlyGenerationResult(
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
        return MonthlyGenerationResult(
            "failed",
            period_start,
            period_end,
            output_path,
            source_count=len(sources),
            missing_dates=missing_dates,
            error=str(exc),
        )
    status = "created" if created else "existing"
    print(f"  [完成] 月记已保存：{output_path}")
    return MonthlyGenerationResult(
        status,
        period_start,
        period_end,
        output_path,
        source_count=len(sources),
        missing_dates=missing_dates,
    )


def generate_latest_completed_month(
    reference_time: datetime | None = None,
) -> MonthlyGenerationResult:
    period_start, _ = latest_completed_month(reference_time)
    return generate_month(period_start)


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


def _next_month(month_start: date) -> date:
    if month_start.month == 12:
        return date(month_start.year + 1, 1, 1)
    return date(month_start.year, month_start.month + 1, 1)


def generate_all_completed_months(
    reference_time: datetime | None = None,
) -> list[MonthlyGenerationResult]:
    diary_dates = _available_diary_dates()
    if not diary_dates:
        print("没有找到可用于生成月记的日记。")
        return []

    cursor = diary_dates[0].replace(day=1)
    last_start, _ = latest_completed_month(reference_time)
    results: list[MonthlyGenerationResult] = []
    while cursor <= last_start:
        results.append(generate_month(cursor))
        if results[-1].status == "failed":
            print("  [暂停] 本轮批量生成遇到API或保存失败，下次运行将从缺失文件继续。")
            break
        cursor = _next_month(cursor)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="生成凛祢的月记")
    parser.add_argument(
        "target",
        nargs="?",
        help="不填则生成最近完整月；填all则补齐历史；或填写月份YYYY-MM",
    )
    args = parser.parse_args()

    if args.target == "all":
        results = generate_all_completed_months()
        return 1 if any(result.status == "failed" for result in results) else 0
    if args.target:
        try:
            target = datetime.strptime(args.target, "%Y-%m").date()
        except ValueError:
            parser.error("月份格式应为 YYYY-MM，例如 2026-07")
        result = generate_month(target)
    else:
        result = generate_latest_completed_month()
    return 0 if result.succeeded or result.status == "no_source" else 1


if __name__ == "__main__":
    raise SystemExit(main())
