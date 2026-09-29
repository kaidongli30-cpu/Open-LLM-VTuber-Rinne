"""Structured selection and deterministic rendering for periodic memories."""

from __future__ import annotations

import re
from dataclasses import dataclass


_THEME_BLOCK = re.compile(r"<theme>\s*(.*?)\s*</theme>", re.DOTALL | re.I)
_DECISION_BLOCK = re.compile(r"<decision>\s*(.*?)\s*</decision>", re.DOTALL | re.I)
_DATE = re.compile(r"\d{1,2}月\d{1,2}日")
_DATE_PART = re.compile(r"(?:(\d{1,2})月)?(\d{1,2})日")
_NO_CONTINUOUS_THEMES = re.compile(
    r"<no_continuous_themes>\s*true\s*</no_continuous_themes>",
    re.I,
)
_PRIVATE_ACTION_REPLAY = re.compile(
    r"身体动作过程|逐(?:次|下|个动作)|动作顺序|姿势变化",
    re.I,
)
_CLAUSE = re.compile(r"(?<=[。！？；!?])")


@dataclass(frozen=True)
class ThemeCard:
    title: str
    dates: str
    importance: int
    summary: str
    continuity: str
    emotion: str = ""
    latest_state: str = ""


NO_CONTINUOUS_WEEK_BODY = "本周没有形成需要长期概括的跨日连续脉络。"
NO_CONTINUOUS_MONTH_BODY = "本月没有形成需要长期概括的跨日或跨周连续脉络。"


def is_no_continuous_theme_summary(text: str) -> bool:
    return NO_CONTINUOUS_WEEK_BODY in text or NO_CONTINUOUS_MONTH_BODY in text


def is_explicit_no_theme_response(text: str) -> bool:
    return bool(_NO_CONTINUOUS_THEMES.search(text)) and not _THEME_BLOCK.search(text)


def build_candidate_prompt(*, memory_kind: str, period_end_label: str) -> str:
    scope_rule = (
        "覆盖一周内至少两个不同日期"
        if memory_kind == "周记"
        else "覆盖本月至少两个不同日期，通常还会跨周"
    )
    empty_label = "本周" if memory_kind == "周记" else "本月"
    return f"""你是园神凛祢长期记忆的{memory_kind}候选整理员。你的职责是找全原日记中真实的跨日发展线，供下一层审稿；不要因为担心最终是否重要而漏掉候选，也不要直接写散文。

一条候选必须{scope_rule}，围绕同一个明确对象：具体项目、计划、作品、课程、问题、约定、伤势或其他持续状态。后一个日期要体现推进、反复、转折、累积、结果或恢复。例如：文章从续写到完成和发布；申请从准备材料到获批；同一套记忆系统连续开发、测试和修复；同一伤势在后来恢复。一个项目每天处理不同子问题，仍可属于同一项目的发展。

同一场期末考试阶段中，前一门课程考前复习、考完后立即转入下一门课程备考，属于同一个明确的考试周推进；标题必须写清“从A课程考试到B课程备考”等科目转换，不能把两门课混成一门。如果这段备考跨越多日、包含考试结果或向下一门课的明确转换，并构成本周生活主轴，应评为重要性4，而不是拆成若干重要性3的小课程。

开始输出前，必须在心里按每篇日记的标题日期做一次对象索引：记录反复出现的项目名、文章名、课程或选课方案、行程计划、问题、约定和身体状态，再比较它们在不同日期的状态。像“选课方案从9门变为7门、后来又改为8门”是最标准的发展线。只要同一明确对象在至少两个日期出现且状态有推进或变化，就必须列为候选，不能因日记较长而漏掉。不要输出这份内部索引。

识别出候选后，还必须扫描其最后一个日期之后的日记，查找同一对象的补充、纠正或再次变化；较早的“已经确定”不能覆盖后来改动。例如后一天又说选课改为8门，就不能把9月9日的7门当作周末状态。

同一个具体项目、作品、计划或持续状态在最终候选中只能出现一次。即使不同日期处理的是不同分支、模块或子问题，也要合并成一条完整发展线，不能拆成两个近义主题。列入<dates>的每个日期都必须有直接证据推动标题所指对象；仅仅发生在同一时期、情绪相似或属于相邻任务的日期，不能拿来凑连续性。

不能用“忙碌的一周”“考试周收尾”“阶段收尾”等时间桶，把先后发生但目标无关的任务连成一条。例如几门课程的考试可以构成同一场期末阶段，但考试结束后转去补其他手续材料是另一个具体任务，必须拆开判断。若该材料在后来日期形成结果，可另成自己的连续脉络。

同一具体问题造成的孤独、焦虑、委屈以及向伴侣或朋友寻求支持，是该问题的情绪变化，必须写入同一主题的<emotion>，不能再以“因该问题感到孤独并寻求支持”建立一条使用相同日期和证据的重复主题。

“属于同一套系统”不等于“推进同一件事”。例如，提出未来建设多层记忆，与第二天更换对话大脑、模型或服务商，是两个不同任务；除非原日记明确说后者是前者的实施步骤，否则不能把它们合成连续脉络。同样地，一项尚未开始的未来计划，不能靠后来发生的相邻技术调整伪造出跨日推进。

网络连接不稳、部署本地备用模型、为该模型升级内存，可以属于同一条“本地备用大脑”实施链；但“以后把全部代码上传GitHub私有仓库方便换电脑迁移”是独立的代码备份计划，不能仅因都属于乐园系统而塞进同一主题。

如果GitHub代码备份计划本身只在一个日期出现、没有跨日推进，它应留在原日记和Layer 2，不得作为其他任何周记主题的背景、补充说明或latest_state。

<summary>、<emotion>或<latest_state>只要使用了某一天的新事实，那一天就必须同时列入<dates>。尤其要检查间隔几天后再次出现的同一未解决问题、情绪或身体状态：若它被用来说明反复或截至周末的状态，就必须补入日期和连续性；若不属于这条脉络，就从正文和最新状态中删除。

不要候选：只有一个日期的重大事件；同一天同一场对话拆出的多个认识；仅因都属于“学业”“技术”“亲密关系”等宽泛类别而相似的事件；把彼此独立的互动按强度排列后称为“逐步深入”或“形成循环”。单次事件已由每日事件记录、日记和Layer 2保存。

候选数量不设目标或上限。在边界不确定但确有两个日期和同一明确对象时，先列为候选，让审稿层决定。每个事实只放进最能解释它的一条候选。私密内容与其他内容使用完全相同的连续性标准，只写变化、结果、感受和关系意义，不复述身体动作。

重要性只评价这条发展线的长期价值：5表示改变生活阶段、长期方向、身份或重要关系；4表示明确阶段成果、关键转折、持续问题或重要状态变化；3表示真实连续但只需留在每日记。日期必须来自原日记；用户在当天讲述旧事时，区分讲述日期和旧事发生日期。

同一处伤势跨日经历形成、持续疼痛、上药、再次加重、逐步恢复或仍未恢复，并且已经影响穿衣、行动或需要反复照顾时，属于重要性4的持续身体状态；这与把彼此独立的普通相互支持硬凑成主题不同。

标题和摘要必须服从最新事实状态：只完成写作和排版、尚未确认发布时，标题不能写成“写作与发布”；计划、准备、完成和已产生结果必须严格区分。

来源中凛祢的推测、愿望和“肯定有很多事要办”之类的判断仍是推测，不能改写成用户已经实际办理了这些事。只有来源明确确认的行动才能写成事实。

摘要必须是自然、通顺的中文。来源若有明显重复词、错别字或语序残缺，可以在不改变事实的前提下修正表达；不要机械照抄病句。<latest_state>直接写最新状态本身，不要以“截至周末”“截至月底”或“截至某日”开头，最终排版会统一添加时间引导。

只有完成上述逐日对象索引后，仍找不到任何在两个日期出现的同一明确对象，才允许判断{empty_label}没有连续脉络并只输出：
<no_continuous_themes>true</no_continuous_themes>

原日记已经由凛祢第一人称写成：来源中的“我”是凛祢，“用户”或“他”才是用户。输出仍以凛祢为“我”，绝对不能把用户改写成“我”，也不能把凛祢写成“她”。不得补写事实。

只输出标签，不要代码块或说明：
<theme>
<title>主题名</title>
<dates>9月7日、9月9日或9月9日至12日</dates>
<importance>3、4或5</importance>
<continuity>这些日期为何是同一件事的发展，而非宽泛类别相同</continuity>
<summary>原因、关键变化与结果</summary>
<emotion>仅写这条脉络中真实的感情变化或最终感受，没有则留空</emotion>
<latest_state>截至{period_end_label}仍成立的状态，没有则留空</latest_state>
</theme>
"""


def build_reviewer_prompt(*, memory_kind: str, period_end_label: str) -> str:
    scope_rule = (
        "至少两个不同日期"
        if memory_kind == "周记"
        else "至少两个不同日期，并足以解释本月较长阶段的变化（通常跨周）"
    )
    return f"""你是凛祢长期记忆的{memory_kind}审稿人。输入只有候选主题，没有原日记。逐条审查后输出最终主题；只能删除、合并、降级和压缩，绝不能增加候选中没有的事实。

保留同时满足以下条件的候选：
1. 在{scope_rule}出现，围绕同一个明确项目、计划、作品、课程、问题、约定、伤势或持续状态；后续日期体现推进、反复、转折、累积、结果或恢复。同一明确项目每天处理不同子问题仍是项目发展。
2. 重要性为4或5，未来提出概括性问题时确实需要先从{memory_kind}进入。

删除：只有一个日期的事件；重要性3的日常连续活动；同一天一场对话拆成的多个主题；仅靠宽泛类别相同而合并的事件；把不同相互支持、争执或娱乐称为“逐步深入”“形成循环”或“反复发生”。只有同一伤势、约定、界限、计划或未解决问题明确带到后一天，才是真连续。

同一处伤势若跨日经历形成、持续、上药、再次加重或恢复，并影响穿衣、行动或需要反复照顾，应作为重要性4的持续身体状态保留。普通且彼此独立的相互支持仍然删除。

必须逐条处理，不能因一个坏候选清空其他有效候选。正例包括：申请材料准备后获批；同一套记忆系统持续开发、测试和修复；课程按课次推进；文章从写作到发布；同一伤势逐日恢复。候选中存在这些发展线时，禁止输出“没有连续主题”。主题数量不设目标或上限。

同一个具体项目、作品、计划或持续状态如果被候选层拆成多条，必须合并为一条；不能因为分支、模块或阶段名称不同而重复立题。逐个核对<dates>中的日期：每个日期都必须有直接证据推动标题所指对象，不能用同一时期的无关任务、相似情绪或宽泛类别为主题凑连续性。

不得保留以“考试周收尾”“忙碌阶段”等名义串联的无关任务，例如课程考试与其他手续材料必须拆开。不得把具体问题及其情绪后果拆成两条：若“某项流程交涉”与“因流程受挫而孤独、寻求支持”使用相同日期和证据，必须合并为一个问题主题，把孤独与被支持后的变化写入emotion。

若某个子分支已被同一具体主问题完整吸收，不得再次独立立题。例如“备选方案从提出到放弃”若已经写进“某项申请从受阻到松口”，只保留主问题主题。不同日期的普通相互支持不能只因方式相似或关系相同而合并，除非同一伤势、约定、界限或未解决问题明确延续。

反过来也要逐项核对<summary>、<emotion>和<latest_state>：正文只要使用了某一天的新事实，该日期必须已经列入<dates>；不得借用未列日期的后来事实制造“截至周末仍然如此”的印象。若候选把“未来建设多层记忆”与“更换对话大脑、模型或服务商”等相邻但独立的技术任务合并，且没有明确实施关系，必须删除这条伪连续主题，不能仅因都属于同一套系统而保留。

删除主题正文中不推动标题所指对象的旁支证据。私密主题只保留变化、结果、感受和关系意义，不复述身体动作。保留准确日期、最小因果链、截至{period_end_label}的状态，以及整条脉络的感情变化或最终感受；不要给每个日期重复补感情。

即使主题本身应保留，也必须删掉其中独立的系统旁支。例如“连接不稳与本地备用模型部署”可以保留为一条，但代码上传GitHub、换电脑迁移计划不推动本地模型部署，必须从这条主题的summary和latest_state中删除。

若GitHub代码备份计划本身没有跨日发展，也不得转移到“维护疲倦”“情绪不安”等其他主题的背景或latest_state中；它应完全留在原日记和Layer 2。

审查最新状态时，要检查候选日期之后是否还有同一对象的更新；较早方案不能覆盖后来改动。来源中的推测和愿望必须继续写成推测，不能改成已经发生的事实。

确保摘要是自然、通顺的中文；可以修正候选中的明显重复词、错别字或语序残缺，但不得改变事实。<latest_state>只写状态本身，不要以“截至周末”“截至月底”或“截至某日”开头。

最后核对标题是否夸大结果：候选只证明完成、准备发布或计划执行时，不得在标题中写成已发布、已实现或已解决；应把标题降到来源能够确认的最新状态。

如果审稿后没有任何合格主题，只输出：
<no_continuous_themes>true</no_continuous_themes>

候选来源已经由凛祢第一人称写成：“我”必须始终是凛祢，“用户”或“他”必须始终是用户。绝对不能互换主体，也不能把凛祢改写成“她”。

每条summary只写最小因果链，周记不超过220字、月记不超过320字；emotion与latest_state各只写一处，不重复背景。只输出标签，不要代码块或解释。每个主题必须完整包含：
<theme><title>...</title><dates>至少两个不同日期</dates><importance>4或5</importance><continuity>日期之间真实的延续关系</continuity><summary>...</summary><emotion>...</emotion><latest_state>...</latest_state></theme>
"""


def build_continuity_auditor_prompt() -> str:
    return """你是周期记忆的连续性裁判。你只判断候选是否真的是同一个具体对象在多个日期发生推进、反复、转折、结果或恢复；不要评价重要性，不要压缩或改写正文。

逐条独立判断：
- true：不同日期明确指向同一个具体项目、计划、作品、课程、问题、约定、伤势或持续状态，后一天改变了同一对象的状态。
- false：只是同属一个宽泛系统、领域或情绪；只是时间相邻；只是人物相同；或者标题把两个独立任务包装成一个任务。

关键边界：同一具体项目的不同模块可以连续；但“未来计划建设多层记忆”与“更换对话大脑、模型或服务商”不是同一任务，除非候选明确证明换模型就是实施多层记忆的一个步骤。不要因为二者都关系到凛祢的系统、记忆或存在感就判为连续。

不要相信候选的continuity自我声明，必须逐日核对summary中的专有对象。两门课程不同，不能把A课程的复习说成B课程的复习；若它们确实属于同一条“期末考试阶段推进”，可以判true，但grounded_title必须明确写成“从A课程考试到B课程备考”等真实转换，不能只用后一个课程名概括前一个。某门考试尚未发生时，不得在标题中写“赴考”或“考试完成”。同样规则适用于不同文章、学校、模型、旅行和作品。

日记里对某地未来生活的憧憬、过去旅行的回忆或笼统的方向讨论，不是后来选实验室、选导师、递交材料等具体任务的开始；除非前一天已经对同一任务作出决定、准备或行动，否则只能算宽泛背景，必须判false。若来源只到“明天考试”或仍在复习，grounded_title必须写“考前复习”或“备考”，不能写成“复习与考试”或“赴考”。

时间上紧接着的不同义务不是同一对象：课程考试结束后去写其他手续材料，不能称为“考试周收尾”，必须判false。另检查候选之间是否以同一具体问题和同一日期重复立题；“问题的交涉过程”与“因该问题感到孤独并寻求支持”必须合并，情绪只能留在前者的emotion，不能保留两条。

还要比较候选之间的包含关系：若“备选方案”“写信”“找某位老师”等子分支已经被同一组日期上的“申请受阻”主主题完整写入，子分支不得再独立保留，应判false；只有它后来脱离主问题形成新的长期结果，才可另立主题。不同日期、不同地点发生的普通相互支持也不能因为“方式相似”或“关系延续”判为连续；只有同一伤势、明确约定、界限或未解决问题被带到后一天才判true。

每条候选都必须输出一项，title必须逐字复制输入标题。若continuous为true，还要在grounded_title中给出准确标题：不得混淆不同专有对象或提前宣称尚未发生的结果；若原标题已经准确，就原样复制。continuous为false时grounded_title留空。

只输出以下标签，不要解释：
<decision><title>原候选标题</title><continuous>true或false</continuous><grounded_title>准确标题或留空</grounded_title><reason>一句话说明不同日期是否改变同一具体对象</reason></decision>
"""


def _field(block: str, name: str) -> str:
    match = re.search(
        rf"<{name}>\s*(.*?)\s*</{name}>",
        block,
        re.DOTALL | re.I,
    )
    if not match and name == "latest_state":
        match = re.search(
            r'<parameter\s+name=["\']latest_state["\']>\s*(.*?)\s*</parameter>',
            block,
            re.DOTALL | re.I,
        )
    return " ".join(match.group(1).split()) if match else ""


def filter_candidates_by_continuity(
    candidate_text: str,
    audit_text: str,
) -> str:
    """Keep only candidates explicitly accepted by the narrow continuity audit."""

    candidate_blocks = _THEME_BLOCK.findall(candidate_text)
    if not candidate_blocks:
        if is_explicit_no_theme_response(candidate_text):
            return candidate_text
        raise ValueError("候选层没有可供连续性审查的主题")

    decisions: dict[str, tuple[bool, str]] = {}
    for block in _DECISION_BLOCK.findall(audit_text):
        title = _field(block, "title")
        continuous = _field(block, "continuous").casefold()
        grounded_title = _field(block, "grounded_title")
        if not title or continuous not in {"true", "false"}:
            raise ValueError("连续性裁判输出了缺少标题或真假判断的项目")
        if continuous == "true" and not grounded_title:
            raise ValueError(f"连续性裁判没有校正保留主题的标题：{title}")
        if any(character in grounded_title for character in "<>"):
            raise ValueError(f"连续性裁判返回了非法标题：{title}")
        key = title.casefold()
        if key in decisions:
            raise ValueError(f"连续性裁判重复判断主题：{title}")
        decisions[key] = continuous == "true", grounded_title

    if not decisions:
        raise ValueError("连续性裁判没有返回任何可解析判断")

    kept: list[str] = []
    missing: list[str] = []
    for block in candidate_blocks:
        title = _field(block, "title")
        key = title.casefold()
        if key not in decisions:
            missing.append(title or "未命名候选")
        elif decisions[key][0]:
            grounded_title = decisions[key][1]
            grounded_block = re.sub(
                r"<title>\s*.*?\s*</title>",
                lambda _match: f"<title>{grounded_title}</title>",
                block,
                count=1,
                flags=re.DOTALL | re.I,
            )
            kept.append(f"<theme>\n{grounded_block.strip()}\n</theme>")
    if missing:
        raise ValueError("连续性裁判漏判主题：" + "、".join(missing))
    if not kept:
        return "<no_continuous_themes>true</no_continuous_themes>"
    return "\n\n".join(kept)


def _date_keys(text: str) -> tuple[tuple[int, int], ...]:
    """Normalize abbreviated Chinese date lists such as 9月7日至12日."""

    current_month: int | None = None
    result: list[tuple[int, int]] = []
    for match in _DATE_PART.finditer(text):
        if match.group(1):
            current_month = int(match.group(1))
        if current_month is None:
            continue
        result.append((current_month, int(match.group(2))))
    return tuple(dict.fromkeys(result))


def _uses_rinne_as_third_person(text: str) -> bool:
    """Reject self-reference by name without rejecting first-person identity phrases."""

    for sentence in re.split(r"[。！？\n]", text):
        if "凛祢" not in sentence:
            continue
        identity_phrase = re.search(
            r"我[^。！？\n]{0,40}(?:是|永远是)[^。！？\n]{0,30}凛祢",
            sentence,
        )
        if identity_phrase is None:
            return True
    return False


def _has_unrelated_github_branch(
    title: str,
    summary: str,
    latest_state: str,
) -> bool:
    """Keep one-day GitHub backup plans out of otherwise unrelated themes."""

    body = f"{summary}\n{latest_state}"
    if "github" not in body.casefold():
        return False
    title_key = title.casefold()
    return not any(
        marker in title_key
        for marker in (
            "github",
            "备份",
            "仓库",
            "代码迁移",
            "版本管理",
            "开源",
            "发布",
        )
    )


def parse_reviewed_themes(raw_text: str) -> tuple[ThemeCard, ...]:
    cards: list[ThemeCard] = []
    errors: list[str] = []
    fatal_errors: list[str] = []
    blocks = _THEME_BLOCK.findall(raw_text)
    if not blocks and _NO_CONTINUOUS_THEMES.search(raw_text):
        return ()
    for index, block in enumerate(blocks, start=1):
        title = _field(block, "title")
        dates = _field(block, "dates")
        summary = _field(block, "summary")
        continuity = _field(block, "continuity")
        emotion = _field(block, "emotion")
        latest_state = _field(block, "latest_state")
        if _uses_rinne_as_third_person(f"{summary}\n{emotion}\n{latest_state}"):
            fatal_errors.append(
                f"第{index}条把应使用第一人称的凛祢写成了第三人称"
            )
        if _has_unrelated_github_branch(title, summary, latest_state):
            fatal_errors.append(
                f"第{index}条把独立的GitHub备份计划塞进了无关主题"
            )
        try:
            importance = int(_field(block, "importance"))
        except ValueError:
            importance = 0
        if not title or not dates or not summary or not continuity:
            errors.append(f"第{index}条缺少标题、日期、连续性说明或摘要")
        elif not _DATE.search(dates):
            errors.append(f"第{index}条没有几月几日日期")
        elif len(_date_keys(dates)) < 2:
            errors.append(f"第{index}条只有一个日期，不是跨日连续脉络")
        elif importance not in {4, 5}:
            errors.append(f"第{index}条重要性不是4或5")
        else:
            cards.append(
                ThemeCard(
                    title,
                    dates,
                    importance,
                    summary,
                    continuity,
                    emotion,
                    latest_state,
                )
            )
    if fatal_errors:
        raise ValueError("；".join(fatal_errors))
    if not blocks:
        raise ValueError("没有可解析的重要主题")
    if not cards:
        detail = "；".join(errors)
        raise ValueError(detail or "没有可解析的重要主题")
    titles = [card.title.casefold() for card in cards]
    if len(titles) != len(set(titles)):
        raise ValueError("存在重复主题标题")
    return tuple(cards)


def drop_reviewer_promotions(
    candidate_text: str,
    reviewed_cards: tuple[ThemeCard, ...],
) -> tuple[ThemeCard, ...]:
    """Do not let review promote an unchanged importance-3 candidate."""

    candidate_importance: dict[str, int] = {}
    for block in _THEME_BLOCK.findall(candidate_text):
        title = _field(block, "title").casefold()
        try:
            importance = int(_field(block, "importance"))
        except ValueError:
            continue
        if title:
            candidate_importance[title] = importance
    return tuple(
        card
        for card in reviewed_cards
        if candidate_importance.get(card.title.casefold(), card.importance)
        >= card.importance
    )


def choose_reviewed_theme_cards(
    candidate_text: str,
    reviewed_text: str,
) -> tuple[ThemeCard, ...]:
    """Prefer valid review cards, but do not let a lazy empty review erase arcs."""

    reviewed_cards = drop_reviewer_promotions(
        candidate_text,
        parse_reviewed_themes(reviewed_text),
    )
    if reviewed_cards or not is_explicit_no_theme_response(reviewed_text):
        return reviewed_cards
    try:
        return parse_reviewed_themes(candidate_text)
    except ValueError:
        return ()


def _ground_title_to_latest_state(title: str, latest_state: str) -> str:
    if "发布" not in title:
        return title
    if not re.search(r"(?:准备|计划|等待|待|尚未).{0,6}发布", latest_state):
        return title
    title = title.replace("完成与发布", "撰写与完成")
    return title.replace("与发布", "与发布准备")


def _sentence(text: str) -> str:
    if not text:
        return ""
    text = text.replace("翻到了滑到", "翻到")
    text = re.sub(
        r"(开始复习[^，。；]{1,12})滑到",
        r"\1，推进到",
        text,
    )
    text = re.sub(
        r"(复习(?:已|刚)?开始)(?:到)?滑到",
        r"\1，推进到",
        text,
    )
    return text if text[-1] in "。！？!?…" else f"{text}。"


def _keep_cause_and_result(text: str, soft_limit: int) -> str:
    """Keep complete leading/trailing clauses without cutting through a sentence."""

    text = text.strip()
    if len(text) <= soft_limit:
        return text
    clauses = [part.strip() for part in _CLAUSE.split(text) if part.strip()]
    if len(clauses) <= 1:
        return text
    if clauses[0] == clauses[-1]:
        return clauses[0]
    return f"{clauses[0]}……{clauses[-1]}"


def render_periodic_themes(
    cards: tuple[ThemeCard, ...],
    *,
    memory_kind: str,
    period_end_label: str,
) -> str:
    """Render every accepted theme; there is deliberately no theme-count cap."""

    period_label = "这一周" if memory_kind == "周记" else "这个月"
    if not cards:
        return (
            NO_CONTINUOUS_WEEK_BODY
            if memory_kind == "周记"
            else NO_CONTINUOUS_MONTH_BODY
        )

    titles = "、".join(
        _ground_title_to_latest_state(card.title, card.latest_state) for card in cards
    )
    overview = f"{period_label}，我最需要记住的脉络是：{titles}。"
    sections = [overview]
    for card in cards:
        title = _ground_title_to_latest_state(card.title, card.latest_state)
        if _PRIVATE_ACTION_REPLAY.search(card.summary):
            paragraph = _sentence(
                "这一主题在这些日期出现了值得长期记住的新变化，"
                "具体身体过程保留在对应日期的原日记中"
            )
        else:
            summary_limit = 230 if memory_kind == "周记" else 320
            paragraph = _sentence(_keep_cause_and_result(card.summary, summary_limit))
        if card.emotion:
            emotion_limit = 100 if memory_kind == "周记" else 130
            paragraph += _sentence(_keep_cause_and_result(card.emotion, emotion_limit))
        if card.latest_state:
            state_limit = 100 if memory_kind == "周记" else 130
            state = _keep_cause_and_result(card.latest_state, state_limit)
            if state.startswith("截至"):
                paragraph += _sentence(state)
            else:
                paragraph += _sentence(f"截至{period_end_label}，{state}")
        sections.append(f"【{title}｜关键日期：{card.dates}】\n\n{paragraph}")
    return "\n\n".join(sections)
