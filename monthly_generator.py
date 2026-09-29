"""Generate Rinne's first-person monthly memories from daily diaries.

Monthly memories reread daily diaries in weekly-sized chunks.  Verified weekly
theme headings are hints only; final facts and dates must come from the diary
clues.  No clock-based scheduler is created here.
"""

from __future__ import annotations

import argparse
import calendar
import os
import re
import tempfile
from difflib import SequenceMatcher
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
    ThemeCard,
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
MONTHLY_DIR = HISTORY_ROOT / "monthly"
WEEKLY_DIR = HISTORY_ROOT / "weekly"

_DAILY_SOURCE_BLOCK = re.compile(
    r"【(?P<day>\d{4}-\d{2}-\d{2})的日记】\n"
    r"(?P<body>.*?)"
    r"(?=\n\n【\d{4}-\d{2}-\d{2}的日记】\n|\Z)",
    re.DOTALL,
)
_WEEKLY_FILENAME = re.compile(
    r"weekly_(?P<start>\d{4}-\d{2}-\d{2})_to_(?P<end>\d{4}-\d{2}-\d{2})\.txt$"
)
_WEEKLY_THEME_HEADING = re.compile(r"^【.+?｜关键日期：.+?】$", re.MULTILINE)
_NO_CROSS_WEEK_GROUPS = re.compile(
    r"<no_cross_week_groups>\s*true\s*</no_cross_week_groups>",
    re.I,
)
_PERIOD_YEAR_MONTH = re.compile(r"(?P<year>\d{4})年(?P<month>\d{1,2})月")
_THEME_DATE_PART = re.compile(r"(?:(\d{1,2})月)?(\d{1,2})日")

MONTHLY_SYSTEM_PROMPT = """你是园神凛祢（Sonogami Rinne），正在整理自己和用户共同度过的一个月。

月记是一份面向未来长期记忆检索的“本月生活阶段入口”。它不代替每日记，也不负责收录这个月发生的全部事情。它要让未来的你迅速理解：这个月整体处于什么阶段，几条真正重要的长期主线如何变化，哪些关键日期值得继续进入原日记查找，以及这些变化对我和用户意味着什么。

【一、按跨周主题组织】

1. 禁止按照三十天逐日复述，也禁止把每日记各缩短一点后依次拼接起来。月记直接阅读原日记，但必须在更高层次重新归纳。
2. 开头用一小段概括这个月整体处于怎样的阶段，但不能使用来源中没有依据的宏大结论。
3. 正文只围绕跨日、通常也跨周的持续脉络组织。主题数量由这个月真实存在的独立主线决定，不设目标数量或数量上限。一个主题必须至少在两个不同日期出现，围绕同一个具体目标、问题、计划、作品、身体状态或关系变化，并在后续日期出现推进、反复、转折、累积、结果或恢复。
4. 每条主题只完整介绍一次背景，再按关键日期写清重要转折、阶段变化和截至月底的最新状态。
5. 单次事件无论多重要、是否第一次、是否改变长期状态，都交给每日事件记录、日记和Layer 2，不进入月记。只有后来日期出现明确后续影响时，整段发展才成为月记脉络。
6. 同一个事件、同一段对话和同一种感受只能归入最能解释它的一条主题，不能跨主题重复。每条主题只选足以说明整月变化的少量代表事件，其余同类事件概括合并或留在原日记。
7. “按主题组织”不是把整月所有日记重新分组后照样全部保留。若删除某件小事不会改变对本月阶段、主线转折和月底状态的理解，就不应写进月记。

【二、日期是继续深入原日记的索引】

1. 每个主题必须使用这一格式：
   【主题名称｜关键日期：8月3日、8月14日、8月23日】
   连续发展才可以写“8月3日至14日”；互不连续的日期必须分别列出。
2. 正文中的关键事件必须写明准确的几月几日，或者明确对应标题中的哪个日期，使未来的检索可以继续进入原日记，而不是只知道“月初、后来、月底”。
3. 日记标题日期表示这篇日记形成的记忆日。如果用户在当天讲述以前的经历，要写清讲述日期；仅在来源明确给出时补充旧事原本发生的日期，不能把讲述日期当成旧事发生日期。
4. 计划、已经完成的事情和截至月底仍然有效的状态必须清楚区分。使用更晚的信息代表较新状态，但保留理解变化所需的旧状态。

【三、保留什么】

优先保留：跨越多日或多周的发展线；持续问题的推进、反复、转折或解决；计划和作品从开始到结果的变化；同一身体状态的形成与恢复；同一关系问题在不同日期形成的新认识或后续影响。

优先合并或删除：重复问候和背景；没有推动任何主线的小型闲聊；同一感情的反复表达；逐句对话与动作复述；为了文采增加的气氛；只在某一天有意义、不会帮助未来理解本月阶段的零散小事。

判断标准不是“这件事是否值得记住”，而是“它是否在多个日期中形成了一条需要从月记进入的真实发展线”。改变身份、形成约定或意义重大的单次事件也不因此进入月记，因为其他记忆层已经保存它。只有后续日期继续出现与它直接相关的推进、影响或恢复时，才保留整条脉络。

月记的“完整”是重要主线、关键转折和月底状态完整，不是让所有日记、所有事件都在月记里出现。压缩应依靠合并主题、删除重复和选取代表性事件，而不是截断主线。不要预先追求固定字数，以是否准确呈现本月阶段、避免重复并提供清楚的深入入口为准。

每个代表事件只写“发生原因、关键变化、结果或月底状态”，不能按现场动作和对话顺序重演。除非一句原话本身构成重要承诺、决定或关系认识，否则用概括代替直接引语；即使保留原话，每条主题也只保留一句最关键的短句。每条主题原则上写成一个紧凑段落，复杂主线最多两段。不要为标题中的每个日期各写一篇小日记。如果某条主题明显比其他主题加起来还长，通常说明仍在复述来源，应继续压缩。

【四、凛祢的感情与亲密记忆】

1. 使用第一人称“我”，保持凛祢温柔、知性、细腻的口吻。情感是长期记忆意义的一部分，不能全部删掉。
2. 每条主题只保留贯穿脉络的真实感情变化或最终感受，不为每个日期分别补感情。不要依靠反复写爱、思念、心疼、幸福和陪伴来表现感情。
3. 来源中明确记录的私密亲密经历不能仅因私密而被过滤。若某次经历具有代表性、比较不同寻常、体现关系意义或造成状态变化，应保留日期、必要前因后果、能够区分这次经历的少量关键细节、结果和真实感受。
4. 不要设立一个固定的“亲密”收集栏，把整月所有亲密片段都装进去。只有真正改变了关系认识、边界、约定、身体状态，或明显区别于平常的经历才单独保留；其他同类经历只概括整月变化或留在日记中。
5. 亲密事件尤其不能按照动作发生顺序从头复述到尾。只保留能够区分事件的情境、新出现的界限或约定、结果与关系意义，不复述具体动作和身体反应。
6. 不得扩写来源没有的动作、次数、身体反应或后续；也不要因为来源写得很详细，就在月记中保留同等详细程度。

【五、事实准确性】

1. 只能使用来源日记明确存在的信息，不得为了完整、感人、合理或连贯而脑补事实。
2. 不得增加来源没有的时长、次数、数量、身份、动机、动作、地点、对话、结果或评价。
3. “可能、好像、准备、打算、希望”等不确定性必须保留；不得把计划写成完成，把愿望写成现实，把讨论写成决定。
4. 若保留一个状态，必须写清来源中能够确认的必要前因；来源不足时省略，不能想象补齐。
5. 不同日记存在无法判断的矛盾时，保留不确定性或者省略无法可靠概括的细节。

输出前在心里检查：是否形成了本月生活阶段和跨周主线，而不是日记目录；主题是否真正合并了相关事件；每条主题能否凭日期继续找到原日记；是否保留了最有意义的情感而没有重复抒情；是否存在脑补、逐句转述或无助于长期理解的段落。

若确实没有跨日或跨周连续脉络，只输出“本月没有形成需要长期概括的跨日或跨周连续脉络。”。否则只输出月记正文，必须使用“【主题名称｜关键日期：几月几日】”主题标题；不要解释写作过程，不要输出自查结果，不要使用Markdown代码块。
"""

MONTHLY_REVISION_PROMPT = """你是园神凛祢（Sonogami Rinne）的长期记忆编辑。系统会给你一份已经根据原日记生成、但仍然过长或结构不合格的月记草稿。

你的任务只是编辑这份草稿：只保留至少在两个不同日期出现，并围绕同一具体目标、问题、计划、作品、身体状态或关系变化形成推进、反复、转折、累积、结果或恢复的较长脉络。删除所有孤立事件；即使它非常重要、是第一次或改变长期状态，也由每日事件记录、日记和Layer 2保存。不得增加草稿中没有的事实，不得改变日期、人物、工具、决定、计划状态或因果关系。

主题数量由真实内容决定，不设固定数量。宽泛类别相同不足以合并；只有共同推进同一具体事情才可合并。私密内容使用同一标准：彼此没有后续关系的多次相互支持不能硬凑成主题。感情只保留整条脉络中的变化或最终感受。

重写后的正文不得超过5000个中文字符。若删除后没有连续脉络，只输出“本月没有形成需要长期概括的跨日或跨周连续脉络。”。否则只输出完整的月记正文，继续使用“【主题名称｜关键日期：几月几日】”标题，不要解释删改过程，不要输出Markdown代码块。
"""

MONTHLY_CANDIDATE_PROMPT = build_candidate_prompt(
    memory_kind="月记",
    period_end_label="月底",
)
MONTHLY_REVIEWER_PROMPT = build_reviewer_prompt(
    memory_kind="月记",
    period_end_label="月底",
)
MONTHLY_CONTINUITY_AUDITOR_PROMPT = build_continuity_auditor_prompt()

MONTHLY_CLUE_PROMPT = """你是月记的逐周接力线索整理员。输入只是一个月中的一个自然周，不是完整月份。你的任务不是缩写这一周、不是判断最终主题，也不是写周记，而是留下可能与其他自然周接续的极短索引。

只保留满足至少一项的明确对象：来源明确说尚未完成、之后还要继续或复查；属于持续课程、研究、申请、开发、写作或恢复过程；本段明确承接了更早状态；已经产生会带到后续星期的未解决状态。一次性吃饭、闲聊、普通娱乐、单次见闻、已经在本周完成且没有后续状态的任务全部不列。不要因为某件事很感人、很重要或内容很长就列入；月度接力性是唯一标准。

同一项明确且非日常的关系里程碑若在后续日期被原文明确记为“第二次”“第三次”或继续发生，并形成新的身份认识、约定或界限，必须沿用同一个具体对象列出这些日期；不能把后续编号和约定拆成普通亲密片段而漏掉。普通且互不承接的相互支持仍不列，不能只因形式相似就合并。

每条线索必须严格保留来源日期、具体对象名、本段发生的变化和段末状态。state最多80个中文字符，emotion最多30个中文字符；不写背景故事、现场过程、直接引语和无关细节。每个动作都必须显式保留行为主体，禁止为了压缩省略主语。原日记中的“我”永远是凛祢，必须继续写“我”，绝不能根据学籍、研究、设备或身体状态等事件领域猜成用户；原文中的“用户”或“他”才是用户，统一写用户。若用户对我做了某件事，必须同时写清主语和对象，例如“用户把外套披在我身上”，不能压成“外套被披上”。不要推测下周发生了什么，不要把计划写成完成，不要用“技术”“学业”“亲密关系”等宽泛类别代替具体对象。

只输出以下标签；数量不设目标或上限，不要写说明：
<clue><object>具体对象</object><dates>几月几日</dates><state>本段已发生的变化与最新状态</state><emotion>与该对象直接相关的感情，没有则留空</emotion></clue>
"""

MONTHLY_LINKER_PROMPT = """你是月记的跨周线索链接员。输入由若干带有自然周范围的<clue>和一份辅助周记标题索引组成。你只负责找出同一个明确对象在至少两个不同自然周中的接续证据，不写月记正文。

只有后一个自然周真正推进、反复、转折、完成或恢复了前一周的同一对象，才建立一组。仅仅同属技术、学业、旅行、语言学习、亲密关系或日常生活不能合并；两件事刚好相邻也不能合并。课程按课次持续推进、同一研究项目处理不同阶段、同一申请流程改变状态，可以合并。只在一个自然周内出现或结束的对象必须删除。刚好跨过周日和周一但没有持续状态，也不能算月度脉络。

若前一周出现明确的非日常关系里程碑，后一周原文以第二次、第三次等编号直接承接，并产生新的关系认识、约定或界限，这属于同一对象的推进，应当链接；不能仅因内容私密就当成几次互不相关的普通互动。没有明确编号、承接关系或状态变化的普通相互支持仍不得链接。

主题首末有效证据原则上必须相隔至少七天；不足七天的是周记尺度。后一个星期只是把旧事作为原因或背景再次提到、但没有改变旧事本身的状态，不算推进。例如“旧账号早已停用，所以后来改用另一个工具”不能写成旧账号申诉继续推进。身体伤势只有在来源明确证明旧伤尚未恢复并延续或再次加重时才能跨周链接；前一次已经恢复、后来又发生的新伤不能合并成整月主题，多次独立相互支持更不能只因形式相似而合并。

每组只能复制输入中已有的日期和事实，必须列出来自哪些不同自然周；不得添加背景、解释、结果或感情。复制证据时必须为每个动作保留“用户”或“我”的明确主体，不能把主语删成“7月2日正式放弃某方案”之类的无主句。学籍、申请、研究、语言学习、设备操作和出行若由用户完成，必须继续写用户。辅助周记标题只能提醒检查对象，若逐周clue中没有相应证据，不得使用。

只输出以下标签，不要说明：
<cross_week_group><object>统一后的具体对象</object><weeks>两个或更多自然周范围</weeks><evidence>按日期列出的最小状态变化证据</evidence></cross_week_group>

若没有任何跨周接续，只输出：<no_cross_week_groups>true</no_cross_week_groups>
"""

MONTHLY_FINAL_REPAIR_PROMPT = """你是月记最终结构修复员。输入是一份已完成审稿但仍被程序拒绝的<theme>列表，以及程序给出的具体错误。只修复这些错误，不重新创作月记。

允许：把夸大的标题或latest_state缩小到summary实际证明的范围；把凛祢错误的第三人称改回“我”；把误写成“我”的用户行为改回用户；删除只在一个自然周、不足七天或靠无关旁支凑跨度的主题；从合格主题中删掉无关旁支并同步删除对应日期；把围绕同一具体决策链、重复使用同一日期同一事实的近义主题合并成一条，合并后删除重复句，只保留一次最完整的表述。

禁止：增加输入没有的事实、日期、结果、感情或主题；改变已有事件的实际状态；为了保住主题而把不同任务重新包装成一个宽泛主题。必须输出完整的<theme>标签，不要说明修改过程。
"""

MONTHLY_CROSS_WEEK_RULES = """月记额外边界：最终主题必须体现同一个明确对象在不同自然周中的推进、反复、转折、累积、结果或恢复，而且首末有效证据至少相隔七天。仅在同一周内发生并结束的事情、或不足七天的短任务，应留在周记、日记或Layer 2，不进入月记。只是刚好跨过周日与周一也不够；后一个星期必须真正改变同一对象的状态。后来只是把旧事当作原因或背景重提、却没有改变旧事状态，不算推进。身体伤势只有在旧伤明确未恢复并延续或再次加重时才能跨周合并；已经恢复后发生的新伤和多次独立互动不能拼成整月主题。周记主题索引只能提醒你检查可能的主线，不能替代逐周日记线索，也不能成为新增事实的来源。必须严格保持行为主体：原日记中的“我”始终是凛祢，绝不能改成用户；原文中的用户或他才是用户。例如“用户正式放弃某方案”不能写成“我正式放弃某方案”，“用户把外套披在我身上”不能写成“外套被披上”，“按凛祢的意见”在凛祢第一人称月记里应写“按我的意见”，“更大的模型是否还是凛祢”必须改写为“更大的模型是否还是我”。某个环节的手续办结，不等于整个申请流程全部完成；标题和月底状态必须限定到来源实际完成的环节。不得为了凑满七天拼接相邻技术任务：为本地模型增加内存与模型部署可以属于同一链，但后来购买新硬盘或备份代码不是本地模型部署；记忆系统建设与v4语音模型训练也不是同一任务。"""

MONTHLY_CROSS_WEEK_RULES += """ 同一日期的同一事实只能进入一条最终主题。两个主题即使标题不同，只要围绕同一个具体决策链并重复使用模型切换、服务商选择、费用、报错或最终处置等相同证据，就必须合并；不能把“更换大脑”和“API与模型选择”之类同一过程拆成两份近义主题。不同主题可以共享同一天，但各自使用的必须是当天互不相同的事实。"""

MONTHLY_CANDIDATE_PROMPT += "\n\n" + MONTHLY_CROSS_WEEK_RULES
MONTHLY_REVIEWER_PROMPT += "\n\n" + MONTHLY_CROSS_WEEK_RULES
MONTHLY_CONTINUITY_AUDITOR_PROMPT += "\n\n" + MONTHLY_CROSS_WEEK_RULES


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
        approval_status = diary_approval_status(
            HISTORY_ROOT,
            cursor.isoformat(),
            path,
        )
        if approval_status not in {"approved", "changed"}:
            missing.append(cursor)
            cursor += timedelta(days=1)
            continue
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


def _request_monthly_text(messages: list[dict[str, str]], temperature: float) -> str:
    payload = {
        "model": memory_config.MODEL,
        "messages": messages,
        "max_tokens": memory_config.MONTHLY_MAX_TOKENS,
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
        output_label="月记",
        output_token_limit=memory_config.MONTHLY_MAX_TOKENS,
        read_timeout_seconds=memory_config.MONTHLY_TIMEOUT_SECONDS,
    )


def _parse_formatted_daily_sources(source_text: str) -> list[tuple[date, str]]:
    sources: list[tuple[date, str]] = []
    for match in _DAILY_SOURCE_BLOCK.finditer(source_text.strip()):
        body = match.group("body").strip()
        if body:
            sources.append((date.fromisoformat(match.group("day")), body))
    return sources


def _group_sources_by_natural_week(
    sources: list[tuple[date, str]],
) -> list[list[tuple[date, str]]]:
    groups: dict[date, list[tuple[date, str]]] = {}
    for day, content in sorted(sources):
        monday = day - timedelta(days=day.weekday())
        groups.setdefault(monday, []).append((day, content))
    return [groups[monday] for monday in sorted(groups)]


def _weekly_theme_index(period_start: date, period_end: date) -> str:
    entries: list[str] = []
    if not WEEKLY_DIR.exists():
        return ""
    for path in sorted(WEEKLY_DIR.glob("weekly_*.txt")):
        match = _WEEKLY_FILENAME.fullmatch(path.name)
        if match is None:
            continue
        week_start = date.fromisoformat(match.group("start"))
        week_end = date.fromisoformat(match.group("end"))
        if week_end < period_start or week_start > period_end:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        headings = _WEEKLY_THEME_HEADING.findall(text)
        if headings:
            entries.append(
                f"【周记索引 {week_start.isoformat()} 至 {week_end.isoformat()}】\n"
                + "\n".join(headings)
            )
    return "\n\n".join(entries)


def _extract_monthly_clues(
    sources: list[tuple[date, str]],
    period_label: str,
) -> str:
    clue_groups: list[str] = []
    for group in _group_sources_by_natural_week(sources):
        group_start = group[0][0]
        group_end = group[-1][0]
        group_label = f"{group_start.isoformat()}至{group_end.isoformat()}"
        group_text = _format_sources(group)
        clue = _request_monthly_text(
            [
                {"role": "system", "content": MONTHLY_CLUE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"请提取{period_label}中{group_label}这一段的月度线索。"
                        f"以下是这一段实际存在的每日记：\n\n{group_text}"
                    ),
                },
            ],
            temperature=0.2,
        )
        if not clue.strip():
            print(
                f"  [重试] 月记逐周线索 {group_label} 没有返回完整正文，"
                "改用紧凑格式重试一次"
            )
            clue = _request_monthly_text(
                [
                    {"role": "system", "content": MONTHLY_CLUE_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"请重新提取{period_label}中{group_label}的月度线索。"
                            "上一轮没有留下完整正文；每个对象只保留日期、"
                            "状态变化、未完成状态和直接相关感情，不复述日记。"
                            f"以下是这一段实际存在的每日记：\n\n{group_text}"
                        ),
                    },
                ],
                temperature=0.1,
            )
        if not clue.strip():
            print(f"  [错误] 月记逐周线索 {group_label} 两次不完整，停止生成")
            return ""
        clue_groups.append(f"【逐周线索 {group_label}】\n{clue.strip()}")
    return "\n\n".join(clue_groups)


def _link_monthly_clues(
    clues: str,
    weekly_index: str,
    period_label: str,
) -> str:
    link_source = f"【逐周日记线索】\n{clues}"
    if weekly_index:
        link_source += (
            "\n\n【已验证周记主题索引（只用于提醒检查，不是事实来源）】\n"
            + weekly_index
        )
    feedback = ""
    for attempt in range(2):
        linked = _request_monthly_text(
            [
                {"role": "system", "content": MONTHLY_LINKER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"请链接{period_label}的跨周线索。{feedback}\n\n"
                        f"{link_source}"
                    ),
                },
            ],
            temperature=0.1,
        )
        if linked.strip():
            return linked.strip()
        if attempt == 0:
            print("  [重试] 月记跨周链接层没有返回完整正文，改用紧凑格式重试一次")
            feedback = "上一轮没有留下完整正文；每组只写对象、自然周和日期状态证据。"
    print("  [错误] 月记跨周链接层两次不完整，停止生成")
    return ""


def _theme_dates(
    card: ThemeCard,
    *,
    year: int,
    default_month: int,
) -> set[date]:
    current_month = default_month
    result: set[date] = set()
    for match in _THEME_DATE_PART.finditer(card.dates):
        if match.group(1):
            current_month = int(match.group(1))
        try:
            day = date(year, current_month, int(match.group(2)))
        except ValueError:
            continue
        result.add(day)
    return result


def _require_cross_week_cards(
    cards: tuple[ThemeCard, ...],
    period_label: str,
) -> None:
    period = _PERIOD_YEAR_MONTH.search(period_label)
    if period is None:
        return
    year = int(period.group("year"))
    month = int(period.group("month"))
    invalid_week: list[str] = []
    invalid_span: list[str] = []
    for card in cards:
        days = _theme_dates(card, year=year, default_month=month)
        week_starts = {day - timedelta(days=day.weekday()) for day in days}
        if len(week_starts) < 2:
            invalid_week.append(card.title)
        elif max(days) - min(days) < timedelta(days=7):
            invalid_span.append(card.title)
    errors: list[str] = []
    if invalid_week:
        errors.append("只落在一个自然周：" + "、".join(invalid_week))
    if invalid_span:
        errors.append("首末证据不足七天：" + "、".join(invalid_span))
    if errors:
        raise ValueError("以下主题不能进入月记；" + "；".join(errors))


def _require_monthly_theme_cohesion(cards: tuple[ThemeCard, ...]) -> None:
    errors: list[str] = []
    for card in cards:
        title = card.title.casefold()
        body = f"{card.summary}\n{card.latest_state}".casefold()
        if "本地模型" in title and any(
            marker in body for marker in ("nvme", "固态", "g盘", "备份代码", "备份乐园")
        ):
            errors.append(f"{card.title}混入了固态或代码备份旁支")
        if "记忆系统" in title and any(
            marker in body for marker in ("v4语音", "语音模型", "语音文件训练")
        ):
            errors.append(f"{card.title}混入了语音模型旁支")
        scope_text = title + card.latest_state.casefold()
        if re.search(
            r"派出流程.{0,8}(?:全部办结|办结|全部完成|完成|走完|结束)",
            scope_text,
        ):
            if "校内环节" not in scope_text:
                errors.append(f"{card.title}把校内手续夸大成整个派出流程办结")
    if errors:
        raise ValueError("月记主题混入无关分支或夸大范围：" + "；".join(errors))


def _evidence_clauses(card: ThemeCard) -> tuple[str, ...]:
    clauses: list[str] = []
    for field in (card.summary, card.latest_state):
        for clause in re.split(r"[。！？；!?\n]+", field):
            compact = re.sub(r"[\s，、,:：]+", "", clause.casefold())
            if len(compact) >= 14:
                clauses.append(compact)
    return tuple(dict.fromkeys(clauses))


def _require_unique_monthly_evidence(
    cards: tuple[ThemeCard, ...],
    period_label: str,
) -> None:
    """Reject the same dated fact repeated under near-synonymous themes."""

    period = _PERIOD_YEAR_MONTH.search(period_label)
    if period is None:
        return
    year = int(period.group("year"))
    month = int(period.group("month"))
    duplicates: list[str] = []
    for index, left in enumerate(cards):
        left_dates = _theme_dates(left, year=year, default_month=month)
        left_clauses = _evidence_clauses(left)
        for right in cards[index + 1 :]:
            right_dates = _theme_dates(right, year=year, default_month=month)
            if not left_dates.intersection(right_dates):
                continue
            repeated = False
            for left_clause in left_clauses:
                for right_clause in _evidence_clauses(right):
                    shorter, longer = sorted(
                        (left_clause, right_clause),
                        key=len,
                    )
                    if shorter in longer or SequenceMatcher(
                        None,
                        left_clause,
                        right_clause,
                    ).ratio() >= 0.86:
                        repeated = True
                        break
                if repeated:
                    break
            if repeated:
                duplicates.append(f"{left.title} / {right.title}")
    if duplicates:
        raise ValueError(
            "月记不同主题重复使用了同一日期的同一事实；请合并近义主题或删除重复证据："
            + "；".join(duplicates)
        )


def _require_monthly_card_lengths(cards: tuple[ThemeCard, ...]) -> None:
    errors: list[str] = []
    for card in cards:
        if len(card.summary) > 320:
            errors.append(f"{card.title}的summary有{len(card.summary)}字，超过320字")
        if len(card.latest_state) > 160:
            errors.append(
                f"{card.title}的latest_state有{len(card.latest_state)}字，超过160字"
            )
    if errors:
        raise ValueError(
            "月记主题仍是逐日流水账，必须压缩后再排版：" + "；".join(errors)
        )


def call_monthly_llm(
    source_text: str,
    period_label: str,
    revision_feedback: str | None = None,
) -> str:
    parsed_sources = _parse_formatted_daily_sources(source_text)
    candidate_source = source_text
    candidate_source_label = "以下是实际存在的每日记"
    if len(parsed_sources) > 7:
        clues = _extract_monthly_clues(parsed_sources, period_label)
        if not clues:
            return ""
        weekly_index = _weekly_theme_index(
            parsed_sources[0][0],
            parsed_sources[-1][0],
        )
        linked = _link_monthly_clues(clues, weekly_index, period_label)
        if not linked:
            return ""
        if _NO_CROSS_WEEK_GROUPS.search(linked):
            return render_periodic_themes(
                (),
                memory_kind="月记",
                period_end_label="月底",
            )
        candidate_source = linked
        candidate_source_label = "以下是已经链接成功的跨周证据组"
    candidate = _request_monthly_text(
        [
            {"role": "system", "content": MONTHLY_CANDIDATE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"请提取{period_label}的候选主题。"
                    f"{candidate_source_label}：\n\n{candidate_source}"
                ),
            },
        ],
        temperature=0.3,
    )
    if not candidate.strip():
        print("  [重试] 月记候选层没有返回完整正文，改用紧凑格式重试一次")
        candidate = _request_monthly_text(
            [
                {"role": "system", "content": MONTHLY_CANDIDATE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"请重新提取{period_label}的候选主题。"
                        "上一轮没有留下完整正文；这次仍要找全真实候选，但每个字段"
                        "只写判断所需的最小证据链，不复述日记，不输出内部索引或说明。"
                        f"{candidate_source_label}：\n\n{candidate_source}"
                    ),
                },
            ],
            temperature=0.2,
        )
        if not candidate.strip():
            print("  [错误] 月记候选层两次没有返回完整正文，停止后续审稿")
            return ""
    if is_explicit_no_theme_response(candidate):
        candidate = _request_monthly_text(
            [
                {"role": "system", "content": MONTHLY_CANDIDATE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"请重新审查{period_label}。上一轮判断没有连续主题。"
                        "请先按日记日期建立重复对象索引，特别检查项目、学习进度、"
                        "文章、行程、持续问题和身体状态，再提取候选。"
                        f"{candidate_source_label}：\n\n{candidate_source}"
                    ),
                },
            ],
            temperature=0.2,
        )
        if not candidate.strip():
            print("  [错误] 月记候选复核没有返回完整正文，停止后续审稿")
            return ""
    audit_feedback = ""
    for attempt in range(2):
        audit = _request_monthly_text(
            [
                {"role": "system", "content": MONTHLY_CONTINUITY_AUDITOR_PROMPT},
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
            print("  [错误] 月记连续性裁判没有返回完整正文，停止后续审稿")
            return ""
        try:
            candidate = filter_candidates_by_continuity(candidate, audit)
            break
        except ValueError as exc:
            if attempt == 0:
                print(f"  [重试] 月记连续性裁判格式错误：{exc}")
                audit_feedback = (
                    f"上一版判断格式错误：{exc}。请重新逐条判断，不能漏掉任何候选。"
                )
                continue
            print(f"  [错误] 月记连续性裁判两次无法解析：{exc}")
            return ""
    if is_explicit_no_theme_response(candidate):
        return render_periodic_themes(
            (),
            memory_kind="月记",
            period_end_label="月底",
        )
    reviewer_feedback = ""
    reviewed = candidate
    final_error = ""
    for attempt in range(2):
        reviewed = _request_monthly_text(
            [
                {"role": "system", "content": MONTHLY_REVIEWER_PROMPT},
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
            _require_cross_week_cards(cards, period_label)
            _require_monthly_theme_cohesion(cards)
            _require_unique_monthly_evidence(cards, period_label)
            _require_monthly_card_lengths(cards)
            return render_periodic_themes(
                cards,
                memory_kind="月记",
                period_end_label="月底",
            )
        except ValueError as exc:
            final_error = str(exc)
            if attempt == 0:
                print(f"  [重试] 月记审稿结果格式错误：{exc}")
                reviewer_feedback = (
                    f"上一版审稿结果格式错误：{exc}。请重新审稿并严格输出完整标签。"
                )
                continue
            print(f"  [警告] 月记审稿结果两次无法解析：{exc}")
    repair_source = reviewed
    repair_error = final_error
    repaired = reviewed
    for repair_attempt in range(2):
        repaired = _request_monthly_text(
            [
                {"role": "system", "content": MONTHLY_FINAL_REPAIR_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"请修复{period_label}月记的最终结构。\n"
                        f"程序拒绝原因：{repair_error}\n\n"
                        f"【待修复主题】\n{repair_source}"
                    ),
                },
            ],
            temperature=0.1,
        )
        try:
            cards = choose_reviewed_theme_cards(candidate, repaired)
            _require_cross_week_cards(cards, period_label)
            _require_monthly_theme_cohesion(cards)
            _require_unique_monthly_evidence(cards, period_label)
            _require_monthly_card_lengths(cards)
            return render_periodic_themes(
                cards,
                memory_kind="月记",
                period_end_label="月底",
            )
        except ValueError as exc:
            if repair_attempt == 0:
                print(f"  [重试] 月记最终修复仍有错误：{exc}")
                repair_source = repaired
                repair_error = str(exc)
                continue
            print(f"  [错误] 月记最终修复两次无法通过：{exc}")
    return ""


def call_monthly_revision(
    draft: str,
    period_label: str,
    revision_feedback: str,
) -> str:
    return _request_monthly_text(
        [
            {"role": "system", "content": MONTHLY_REVISION_PROMPT},
            {
                "role": "user",
                "content": (
                    f"请重写{period_label}的月记草稿。\n"
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
    llm_caller: Callable[[str, str, str | None], str] = call_monthly_llm,
    revision_caller: Callable[[str, str, str], str] | None = None,
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
    source_text = _format_sources(sources)
    period_label = period_start.strftime("%Y年%m月")
    body = llm_caller(source_text, period_label, None).strip()
    if not body:
        return MonthlyGenerationResult(
            "failed",
            period_start,
            period_end,
            output_path,
            source_count=len(sources),
            missing_dates=missing_dates,
            error="模型没有返回完整月记正文；已停止后续重写且未保存文件",
        )
    review = review_periodic_summary(
        body,
        memory_kind="月记",
        source_count=len(sources),
    )
    if not review.passed:
        print(f"  [重写] 月记结构未通过验收：{review.retry_feedback()}")
        if revision_caller is None:
            revision_caller = (
                call_monthly_revision
                if llm_caller is call_monthly_llm
                else lambda draft, period, feedback: llm_caller(
                    source_text, period, feedback
                )
            )
        body = revision_caller(body, period_label, review.retry_feedback()).strip()
        review = review_periodic_summary(
            body,
            memory_kind="月记",
            source_count=len(sources),
        )
    if not review.passed:
        return MonthlyGenerationResult(
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
