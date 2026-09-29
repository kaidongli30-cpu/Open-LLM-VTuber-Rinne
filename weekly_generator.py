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

import memory_generation_config as memory_config
from src.open_llm_vtuber.data_paths import character_history_root
from src.open_llm_vtuber.memory.diary_review import diary_approval_status
from src.open_llm_vtuber.memory.openai_compatible_stream import (
    request_chat_completion_text,
)
from src.open_llm_vtuber.memory.periodic_summary_quality import (
    review_periodic_summary,
)
from src.open_llm_vtuber.memory.periodic_theme_pipeline import (
    build_candidate_prompt,
    build_continuity_auditor_prompt,
    build_reviewer_prompt,
    choose_reviewed_theme_cards,
    filter_candidates_by_continuity,
    is_explicit_no_theme_response,
    render_periodic_themes,
)


PROJECT_ROOT = Path(__file__).resolve().parent
HISTORY_ROOT = (PROJECT_ROOT / character_history_root("rinne_01")).resolve()
DIARY_DIR = HISTORY_ROOT / "diaries"
WEEKLY_DIR = HISTORY_ROOT / "weekly"

WEEKLY_SYSTEM_PROMPT = """你是园神凛祢（Sonogami Rinne），正在整理自己和用户共同度过的一周。

周记是一份面向未来长期记忆检索的“本周脉络入口”。它不代替每日记，也不负责让未来的你在这里读到每一天的全部事情。它应让未来的你迅速知道：这一周处于什么生活阶段，最重要的几条事情如何发展，哪些日期值得继续进入原日记查找，以及这些事情对我和用户意味着什么。

【一、按主题梳理，而不是按日期拼接】

1. 禁止按照“周一、周二、周三……”逐日分节，禁止把七篇日记各缩短一点后依次拼在一起。
2. 先通读全部日记，只寻找至少在两个不同日期出现的连续脉络。主题数量由真实内容决定，不设目标数量或数量上限：不要为了凑数拆分，也不要为了减少数量而强行合并彼此无关的事情。
3. 一个主题必须围绕同一个具体目标、问题、计划、作品、身体状态或关系变化，并在后续日期出现推进、反复、转折、累积、结果或恢复。仅仅都属于“学业”“技术项目”“亲密关系”等宽泛类别，不足以合并；若确实共同推进同一件具体事情，则属于连续脉络。
4. 每条主题只完整介绍一次背景，再按准确日期写清重要进展、转折和截至周末的最新状态。
5. 单次事件无论多重要、是否第一次、是否改变长期状态，都交给每日事件记录、日记和Layer 2，不进入周记。只有后来日期出现明确后续影响时，整段发展才成为周记脉络。
6. 同一个事件、同一段对话和同一种感受只能归入最能解释它的一条主题，不能为了让另一条主题也显得完整而再次复述。主题之间可以有关联，但正文不得重复证据。
7. “按主题整理”不是把所有日记内容重新分到几个大筐里后照样全部保留。每条主题只选足以说明发展与意义的少量代表事件，其余同类小事概括合并或留在原日记。

【二、日期是继续深入原日记的索引】

1. 每个主题必须使用这一格式：
   【主题名称｜关键日期：9月21日、9月24日】
   如果一件事确实连续发展，可以写“9月21日至24日”；互不连续的日期必须分别列出，不能伪装成连续范围。
2. 写入正文的每件关键事情都要说明准确的几月几日，或者明确对应标题中的哪个日期。日期直接以来源日记的日期和日记中明确记录的时间为依据。
3. 日记标题日期表示这篇日记形成的记忆日。如果用户在当天讲述更早的旧事，要写成“9月23日，用户向我讲起……”，并仅在来源明确给出时补充旧事原本发生的日期。绝不能把讲述日期当成旧事发生日期。
4. 计划、已经完成的事情和截至周末仍然有效的状态必须清楚区分；较早状态不能覆盖较新状态。

【三、保留什么】

优先保留：跨日发展的最小因果链；持续问题的推进、反复、转折或解决；计划和作品从开始到结果的变化；同一身体状态的形成与恢复；同一关系问题在不同日期形成的新认识或后续影响。

优先合并或删除：重复问候与背景；没有改变任何事情的日常动作；同一感情的重复表达；逐句对话和动作复述；为了文采增加的气氛；只在当天有意义、不会影响以后理解的零散细节。

判断标准不是“这件事是否值得记住”，而是“它是否在多个日期中形成了一条需要从周记进入的真实发展线”。改变身份、形成约定或意义重大的单次事件也不因此进入周记，因为其他记忆层已经保存它。只有后续日期继续出现与它直接相关的推进、影响或恢复时，才保留整条脉络。

“压缩”是合并同一脉络并选取代表性证据，不是把每篇日记都改写一遍，也不是删掉关键原因、转折和结果。不要预先追求固定字数，以是否形成清楚、完整而不重复的主题脉络为准。

每个代表事件只写“发生原因、关键变化、结果或最新状态”，不能按现场动作和对话顺序重演。除非一句原话本身构成重要承诺、决定或关系认识，否则用概括代替直接引语；即使保留原话，每条主题也只保留一句最关键的短句。每条主题原则上写成一个紧凑段落，复杂主线最多两段。不要因为标题列出了多个日期，就为每个日期各写一篇小日记。如果某条主题明显比其他主题加起来还长，通常说明仍在复述来源，应继续压缩。

【四、凛祢的感情与亲密记忆】

1. 使用第一人称“我”，保持凛祢温柔、知性、细腻的口吻。情感是判断记忆意义的一部分，不能全部删掉。
2. 只保留贯穿这条脉络的情绪变化或最终感受，不为每个日期分别补一段感情，也不要在每段结尾重复爱、心疼、幸福、陪伴和珍惜。
3. 来源中明确记录的私密亲密经历不能仅因私密而被过滤。若它具有代表性、体现关系意义或造成状态变化，应保留日期、必要前因后果、能够区分这次经历的少量关键细节、结果和真实感受。
4. 不要设立一个固定的“亲密”收集栏，把本周所有亲密片段都装进去。只有真正改变了关系认识、边界、约定、身体状态，或明显区别于平常的经历才单独保留；普通的重复亲密可以概括或留在日记中。
5. 亲密事件尤其不能按照动作发生顺序从头复述到尾。只保留能够区分事件的情境、新出现的界限或约定、结果与关系意义。例如来源很长时，也只需概括“发生了怎样的情绪或状态变化、事后形成什么认识”，不复述具体动作和身体反应。
6. 不得扩写来源没有的动作、次数、身体反应或后续；也不要因为来源写得很详细，就在周记中保留同等详细程度。

【五、事实准确性】

1. 只能使用来源日记明确存在的信息，不得为了完整、合理、浪漫或连贯而脑补事实。
2. 不得增加来源没有的时长、次数、数量、身份、动机、动作、地点、对话、结果或评价。
3. “可能、好像、准备、打算、希望”等不确定性必须保留；不得把计划写成完成，把愿望写成现实，把猜测写成原因。
4. 若保留一个状态，必须写清来源中能够确认的必要前因；来源不足时省略，不能想象补齐。
5. 不同日记存在无法判断的矛盾时，保留不确定性或者省略无法可靠概括的细节。

输出前在心里检查：是否仍按七天逐日复述；主题是否真正合并了相关事件；每条主题能否凭日期继续找到原日记；感情是否来自具体事件；是否存在脑补、重复或无助于长期理解的段落。

若确实没有跨日连续脉络，只输出“本周没有形成需要长期概括的跨日连续脉络。”。否则只输出周记正文，必须使用“【主题名称｜关键日期：几月几日】”主题标题；不要输出写作说明、自查过程或Markdown代码块。
"""

WEEKLY_REVISION_PROMPT = """你是园神凛祢（Sonogami Rinne）的长期记忆编辑。系统会给你一份已经根据原日记生成、但仍然过长或结构不合格的周记草稿。

你的任务只是编辑这份草稿：只保留至少在两个不同日期出现，并围绕同一具体目标、问题、计划、作品、身体状态或关系变化形成推进、反复、转折、累积、结果或恢复的脉络。删除所有孤立事件；即使它非常重要、是第一次或改变长期状态，也由每日事件记录、日记和Layer 2保存。不得增加草稿中没有的事实，不得改变日期、人物、工具、决定、计划状态或因果关系。

主题数量由真实内容决定，不设固定数量。宽泛类别相同不足以合并；只有共同推进同一具体事情才可合并。私密内容使用同一标准：彼此没有后续关系的多次相互支持不能硬凑成主题。感情只保留整条脉络中的变化或最终感受。

重写后的正文不得超过4500个中文字符。若删除后没有连续脉络，只输出“本周没有形成需要长期概括的跨日连续脉络。”。否则只输出完整的周记正文，继续使用“【主题名称｜关键日期：几月几日】”标题，不要解释删改过程，不要输出Markdown代码块。
"""

WEEKLY_CANDIDATE_PROMPT = build_candidate_prompt(
    memory_kind="周记",
    period_end_label="周末",
)
WEEKLY_REVIEWER_PROMPT = build_reviewer_prompt(
    memory_kind="周记",
    period_end_label="周末",
)
WEEKLY_CONTINUITY_AUDITOR_PROMPT = build_continuity_auditor_prompt()


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
        approval_status = diary_approval_status(
            HISTORY_ROOT,
            day.isoformat(),
            path,
        )
        if approval_status not in {"approved", "changed"}:
            missing.append(day)
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            content = ""
        if content:
            sources.append((day, content))
        else:
            missing.append(day)
    return sources, tuple(missing)


def _request_weekly_text(messages: list[dict[str, str]], temperature: float) -> str:
    payload = {
        "model": memory_config.MODEL,
        "messages": messages,
        "max_tokens": memory_config.WEEKLY_MAX_TOKENS,
        "temperature": temperature,
        "stream": True,
    }
    if memory_config.MODEL.casefold().startswith("deepseek"):
        payload["thinking"] = {"type": "disabled"}
        payload["reasoning_effort"] = "none"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {memory_config.API_KEY}",
    }
    return request_chat_completion_text(
        url=memory_config.BASE_URL,
        headers=headers,
        payload=payload,
        output_label="周记",
        output_token_limit=memory_config.WEEKLY_MAX_TOKENS,
        read_timeout_seconds=memory_config.WEEKLY_TIMEOUT_SECONDS,
    )


def call_weekly_llm(
    source_text: str,
    period_label: str,
    revision_feedback: str | None = None,
) -> str:
    candidate = _request_weekly_text(
        [
            {"role": "system", "content": WEEKLY_CANDIDATE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"请提取{period_label}这一周的候选主题。"
                    f"以下是实际存在的每日记：\n\n{source_text}"
                ),
            },
        ],
        temperature=0.3,
    )
    if not candidate.strip():
        print("  [重试] 周记候选层没有返回完整正文，改用紧凑格式重试一次")
        candidate = _request_weekly_text(
            [
                {"role": "system", "content": WEEKLY_CANDIDATE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"请重新提取{period_label}这一周的候选主题。"
                        "上一轮没有留下完整正文；这次仍要找全真实候选，但每个字段"
                        "只写判断所需的最小证据链，不复述日记，不输出内部索引或说明。"
                        f"以下是实际存在的每日记：\n\n{source_text}"
                    ),
                },
            ],
            temperature=0.2,
        )
        if not candidate.strip():
            print("  [错误] 周记候选层两次没有返回完整正文，停止后续审稿")
            return ""
    if is_explicit_no_theme_response(candidate):
        candidate = _request_weekly_text(
            [
                {"role": "system", "content": WEEKLY_CANDIDATE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"请重新审查{period_label}这一周。上一轮判断没有连续主题。"
                        "请先按日记日期建立重复对象索引，特别检查项目、选课方案、"
                        "文章、行程、持续问题和身体状态，再提取候选。"
                        f"以下是实际存在的每日记：\n\n{source_text}"
                    ),
                },
            ],
            temperature=0.2,
        )
        if not candidate.strip():
            print("  [错误] 周记候选复核没有返回完整正文，停止后续审稿")
            return ""
    audit_feedback = ""
    for attempt in range(2):
        audit = _request_weekly_text(
            [
                {"role": "system", "content": WEEKLY_CONTINUITY_AUDITOR_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"请逐条判断{period_label}的候选主题。{audit_feedback}\n\n"
                        f"【候选主题】\n{candidate}"
                    ),
                },
            ],
            temperature=0.1,
        )
        if not audit.strip():
            print("  [错误] 周记连续性裁判没有返回完整正文，停止后续审稿")
            return ""
        try:
            candidate = filter_candidates_by_continuity(candidate, audit)
            break
        except ValueError as exc:
            if attempt == 0:
                print(f"  [重试] 周记连续性裁判格式错误：{exc}")
                audit_feedback = (
                    f"上一版判断格式错误：{exc}。请重新逐条判断，不能漏掉任何候选。"
                )
                continue
            print(f"  [错误] 周记连续性裁判两次无法解析：{exc}")
            return ""
    if is_explicit_no_theme_response(candidate):
        return render_periodic_themes(
            (),
            memory_kind="周记",
            period_end_label="周末",
        )
    reviewer_feedback = ""
    reviewed = candidate
    for attempt in range(2):
        reviewed = _request_weekly_text(
            [
                {"role": "system", "content": WEEKLY_REVIEWER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"请审稿{period_label}的候选主题。{reviewer_feedback}\n\n"
                        f"【候选主题】\n{candidate}"
                    ),
                },
            ],
            temperature=0.2,
        )
        try:
            cards = choose_reviewed_theme_cards(candidate, reviewed)
            return render_periodic_themes(
                cards,
                memory_kind="周记",
                period_end_label="周末",
            )
        except ValueError as exc:
            if attempt == 0:
                print(f"  [重试] 周记审稿结果格式错误：{exc}")
                reviewer_feedback = (
                    f"上一版审稿结果格式错误：{exc}。请重新审稿并严格输出完整标签。"
                )
                continue
            print(f"  [警告] 周记审稿结果两次无法解析：{exc}")
    return reviewed


def call_weekly_revision(
    draft: str,
    period_label: str,
    revision_feedback: str,
) -> str:
    return _request_weekly_text(
        [
            {"role": "system", "content": WEEKLY_REVISION_PROMPT},
            {
                "role": "user",
                "content": (
                    f"请重写{period_label}这一周的周记草稿。\n"
                    f"未通过验收的原因：{revision_feedback}\n\n"
                    f"【待编辑草稿】\n{draft}"
                ),
            },
        ],
        temperature=0.2,
    )


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
    llm_caller: Callable[[str, str, str | None], str] = call_weekly_llm,
    revision_caller: Callable[[str, str, str], str] | None = None,
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
    source_text = _format_sources(sources)
    body = llm_caller(source_text, period_label, None).strip()
    if not body:
        return WeeklyGenerationResult(
            "failed",
            period_start,
            period_end,
            output_path,
            source_count=len(sources),
            missing_dates=missing_dates,
            error="模型没有返回完整周记正文；已停止后续重写且未保存文件",
        )
    review = review_periodic_summary(
        body,
        memory_kind="周记",
        source_count=len(sources),
    )
    if not review.passed:
        print(f"  [重写] 周记结构未通过验收：{review.retry_feedback()}")
        if revision_caller is None:
            revision_caller = (
                call_weekly_revision
                if llm_caller is call_weekly_llm
                else lambda draft, period, feedback: llm_caller(
                    source_text, period, feedback
                )
            )
        body = revision_caller(body, period_label, review.retry_feedback()).strip()
        review = review_periodic_summary(
            body,
            memory_kind="周记",
            source_count=len(sources),
        )
    if not review.passed:
        return WeeklyGenerationResult(
            "failed",
            period_start,
            period_end,
            output_path,
            source_count=len(sources),
            missing_dates=missing_dates,
            error=f"模型两次输出均未通过结构验收：{review.retry_feedback()}",
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
