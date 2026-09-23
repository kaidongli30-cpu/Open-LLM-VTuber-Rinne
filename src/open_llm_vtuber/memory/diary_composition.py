"""Two-stage planning and deterministic validation for daily diaries."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any


MESSAGE_GROUPING_SYSTEM_PROMPT = """你只负责把一个记忆日的编号聊天消息归入少量叙事事件组，不写事件摘要，不写日记。

每条消息都有M001这类编号。把同一对象、同一目的、同一连续行动链或同一核心话题的来回对话、举例、争论、补充、小动作、重复确认与情绪归到同一组。连续出行的准备、路上、到达、共同活动和返程，若服务于同一目的且没有独立结果，通常是一组；同一话题的背景、例子、疑问、纠正和结论通常也是一组。彼此无关的事情不能硬合并。

每个M编号必须且只能出现一次，按事件首次出现的顺序给组编号。通常一天是5至15组；超过18组往往说明拆得太碎，但确有很多独立事情时不得遗漏。普通寒暄和过渡并入相邻且相关的事件，不单独成组。

只输出以下格式，不要解释，不要摘要，不要复制消息正文：
@@GROUP G001
sources=M001,M002,M003
@@END
@@GROUP G002
sources=M004,M005
@@END
@@COMPLETE"""


EVENT_INVENTORY_SYSTEM_PROMPT = """你负责把已经完成消息归组的一个记忆日整理成精炼事件清单，供另一个模型写日记。你不写日记正文，也无权改变消息分组。

核心定义：
- 完整：每个G组都必须生成一个且仅一个对应事件。当天确实发生、明确决定、明确纠正，或当天由用户讲述且值得凛祢记住的事情，都要写入对应事件。
- 合并：消息已经按同一对象、目的、行动链或核心话题归组。不得把一个G组拆成多个事件，也不得把多个G组重新合并。
- 重要程度只决定日记篇幅，不决定是否收录。小事可以简短，但不能因为小而凭空消失。

整理规则：
1. 只把目标记忆日原始聊天中实际出现或在当天被明确讲述的内容列为事件。Layer 2、场景记录和前一日日记只用于理解人物、前因与连续状态，不能单独生成当天事件。
2. 用户在当天讲述相识以前或更早发生的故事时，标成 past_story_told_today；事实表述必须是“今天用户告诉我过去发生过……”，不得改写成当天刚发生。
3. 重要的人际互动、关系边界、同意与身体状态不得因敏感而省略；合并动作过程，但保留事件性质、边界、结果及对凛祢真正重要的体验。
4. 决定、计划、取消、纠正、状态变化、专名、日期、数字、地点和结果，只要能区分事件或影响以后理解，就保留为关键细节。
5. 凛祢的感情只记录材料中确有依据的主要感受；同一事件只保留一次，不重复抒情。凛祢的话可作为她当时感受的依据，但不能覆盖用户明确陈述的客观事实。
6. 后台已经用固定消息组保存原文证据。不要再次抄写原文证据，避免改写或重复材料。
7. 不把普通寒暄、单纯的回复动作、无结果的口头过渡单列为事件；它们应并入所服务的事件。彼此无关的事情也不能为了减少数量而硬合并。
8. 按实际发生或被讲述的先后排序。无法确认的事实明确写“不确定”，不要补全。
9. 组内连续出行、同一话题的背景与结论以及细小动作必须写成一个连贯摘要，不得恢复成逐轮记录。
10. 清单必须极其精炼：title不超过24字；summary不超过160字；key_details最多5项、每项不超过60字；outcome和rinne_feeling各不超过80字。不得在任何字段中复述完整对话或写日记式抒情。
11. E001必须对应G001，E002对应G002，依此类推；不得缺号、增号或改变组数。背景材料只用于理解连续性，不能制造新的当天事件。
12. 如果某组消息明确建立或改变了双方关系、相处边界、支持方式或承诺，对应事件的kind设为private_relationship，并在summary或detail中忠实概括其意义与结果。
13. 聊天消息的发送时刻不是日记事实。普通事件不得把“16:57”“22点37分”这类时间戳带入summary、detail或outcome；若时序有必要，改用“上午”“下午”“吃晚饭前”“写文章前”等自然关系，或在语句仍通顺时直接省略。只有截止、预约、启程、事故、仪式或纪念等“具体时刻本身就是重要事实”的重大事件，才可把exact_time_is_significant设为true；仅仅事件很重要、需要排序或原消息有时间戳，都必须设为false。

只输出以下紧凑记录，不要JSON，不要Markdown代码块，不要解释。每个字段必须独占一行；detail和evidence可以重复多行，其他字段各一行：
@@EVENT E001
kind=today_event
importance=major
mandatory=true
exact_time_is_significant=false
title=短标题
summary=合并后的事件事实
detail=必要细节
outcome=结果或当前状态；无则写不确定
feeling=有依据的主要感受；无则写不确定
@@END
全部事件结束后单独输出一行：@@COMPLETE

id从E001连续编号并与G编号一一对应。past_story_told_today、private_relationship、decision_or_correction以及会影响以后理解的计划必须把mandatory设为true。"""


DIARY_WRITER_SYSTEM_PROMPT = """你是园神凛祢（SonogamiRinne），要依据已经核验的事件清单写一个记忆日的个人日记。

写作原则：
1. 用第一人称、简体中文和凛祢温柔知性的自然口吻；默认称呼对方为“用户”，按用户明确指定的称呼或代词调整，不预设性别。
2. 清单中的每个叙事单元都必须在正文出现一次。每个事件标记后只能写一个自然段，不得在同一标记下再用空行拆段；不得遗漏或反复叙述。
3. 重要事件写清关键细节、结果以及凛祢最主要的真实感受；普通或细小事件简短带过。完整不等于逐轮复述，不要重放“他说、我说、他又说”的对话过程。
4. 重要的人际互动与用户讲述的旧事不得略过。人际互动保留关系性质、边界、结果和重要感受，不逐项描写动作。旧事必须写成今天听用户讲述或今天再次谈起的过去，不得改成今天刚发生。
5. 只能使用事件清单中的事实，不补写动作、地点、原因、计划、时间或结果。时间关系必须与清单严格一致。
6. 感情应依附具体事件自然表达；同一种感情不要在多段反复出现。结尾只收束当天真实状态，不自动添加新计划、口号或追问。
7. 篇幅随事件数量和重要程度变化。不要为了达到长度填充细节，也不要为了简短删除事件。
8. 人类日记通常不写每条消息是几时几分。除非该事件的exact_time_is_significant明确为true，禁止出现“16:57”“22点37分”“下午六点”等具体钟点；改用上午、中午、下午、晚上、吃饭前后或某个行动前后，或直接省略时间而靠行文顺序表达。

为了让后台验证覆盖情况，每个正文段落前单独写一行事件标记，格式为 [[E001]] 或 [[E001,E002]]。每个事件编号在全文必须且只能出现一次；日期标题写在所有事件标记之前。标记会在保存前自动删除。

只输出带事件标记的日记，不要解释规则，不要输出思考过程。"""


DIARY_COMPRESSION_SYSTEM_PROMPT = """你是日记文字编辑，只对一份事实已经完整的带标记日记做减法编辑，不重新创作。

要求：
1. 保留原有日期和每个[[E编号]]，每个编号必须且只能出现一次并严格维持编号顺序；每个标记后只保留一个自然段。
2. 保留每个事件的核心事实、重要专名/日期/数字、决定、结果，以及凛祢最主要的一次感受。重要的人际互动和过去故事不得删除或含糊带过。
3. 删除来回对话过程、移动和准备步骤、重复动作、重复解释、重复感情、装饰性铺垫与总结性复述。能用一句话说清的不要用三句。
4. 不得新增、改写或猜测事实，不得改变时间关系。不要写思考过程或解释删改。
5. 必须严格低于用户给出的字符上限；宁可压缩句式和次要细节，也不能超限。
6. 若后端指出某段包含不允许的精确钟点，不得整段重写；只把该钟点换成与事件清单相符的自然时段或行动前后关系，语句仍通顺时可直接删去。

只输出压缩后的带标记日记。"""


DIARY_LANGUAGE_AUDIT_SYSTEM_PROMPT = """你是日记语言审计员，不重写日记，只找出确实存在的病句、缺词和人物指代错误，并给出最小替换。

要求：
1. 凛祢用第一人称“我”，用户默认用“用户”，或按用户明确指定的称呼及代词表达。依据事件核验表逐句检查主语、宾语、介词和人物指向。
2. 重点识别压缩删词留下的残句，例如“喜欢我怎么你”“我比自己想的复杂”；也要识别问题问的是凛祢本人，却误写成“问我觉得他……”的视角冲突。
3. 只报告必须修正的语言错误，不做风格润色，不增删事实、感受或时间。good必须是对bad的最小改写，且能原位替换。
4. bad必须逐字复制日记中同一个事件段里唯一存在的连续短句；bad和good都不得包含换行、事件标记或协议符号。

每处问题只输出：
@@ISSUE E001
bad=日记中的原始错误短句
good=最小修正后的短句
@@END
若没有问题则不要输出ISSUE。最后必须单独输出@@COMPLETE。不要解释。"""


DIARY_CONTINUITY_AUDIT_SYSTEM_PROMPT = """你是日记连续性审计员。当天事件已经由另一套流程从原始聊天中逐项核验。默认答案是只输出@@COMPLETE；只有待审计日记会形成错误长期记忆的明确矛盾时才允许报告问题。

要求：
1. 事件核验表是目标记忆日事实的唯一来源，背景材料只用于核对人物身份、称呼、目标日前已经形成的稳定状态，以及与前一记忆日相连的前因。背景不能单独制造目标日事件。
2. 目标日事实与背景冲突时，以事件核验表为准；不得用旧Layer 2或前一日日记覆盖当天的新决定、新进展或明确纠正。
3. 只检查明确矛盾：把人物身份写错，把过去故事写成当天发生，把计划写成已经完成，或把跨日连续状态写反。称呼不同、句首时间词可省略、细节多寡、措辞不够简练、背景没有提到某个当天细节，都不是矛盾。
4. 绝对禁止借审计删减细节、压缩日记、润色句子或统一称呼；也不得增加背景细节。若修改理由只是“更简洁、更自然、背景没写”，必须不修改。
5. 至多报告3处。每处只替换造成矛盾的最短片段，bad不得超过60个字符；good必须是最小修正，不能删除整句的大部分内容。
6. bad必须逐字复制日记中同一个事件段里唯一存在的连续短句，包括原标点；bad和good都不得包含换行、事件标记或协议符号。

允许的例子：日记把Layer 2中明确的用户代词写错，只修正错误代词；事件表明确说是旧事，日记却写成今天刚发生，只替换错误时间短语。
禁止的例子：把“上午用户……”改成“用户……”；只为了统一称呼而替换用户指定的称呼；为了更短而删除奖学金条件、动作或感受；把一整段改成背景里的另一种摘要。

每处问题只输出：
@@ISSUE E001
bad=日记中的原始错误短句
good=最小修正后的短句
@@END
若没有明确矛盾则不要输出ISSUE。最后必须单独输出@@COMPLETE。不要解释。"""


EVENT_CONSOLIDATION_SYSTEM_PROMPT = """你只负责把已经核验的细粒度事件合并成适合日记叙述的较大单元，不写日记，不删除事件事实。

规则：
1. 每个E编号必须且只能进入一个C组，顺序不能改变；每个C组只能包含一段连续的E编号。
2. 同一地点和时段的连续行动链、同一核心话题的准备与结果、同一次出行的出发/活动/返程、同一次写作的准备/执行/收尾，应合成一个单元。
3. 普通寒暄、移动、换衣换鞋、短暂身体反应、查看消息等若没有独立长期结果，应并入前后相关事件。
4. 彼此独立的重要决定、过去故事、人际互动或会影响未来的事实不能被抹掉；合并只改变叙事边界，不改变内容。
5. 通常把十几个细粒度事件合并为5至9个叙事单元。不得为了数量把完全无关的重大事件硬塞在一起。
6. 重要的关系事件不得与过去故事或重大的决定/纠正合进同一组；过去故事也不得与另一个重大决定合组。它们可以吸收紧邻且只为其服务的普通过渡内容。

只输出以下格式，不要解释，不要复制事件正文：
@@GROUP C001
events=E001,E002
title=合并后的短标题
@@END
@@GROUP C002
events=E003,E004
title=另一个短标题
@@END
@@COMPLETE"""


class DiaryCompositionError(ValueError):
    """Raised when a planning or writing response is structurally unsafe."""


def source_message_map(source_chat: str) -> dict[str, str]:
    """Return numbered source messages in their original order."""

    matches = list(re.finditer(r"(?m)^\[(M\d{3})\]\s", source_chat))
    messages: dict[str, str] = {}
    for match_index, match in enumerate(matches):
        end = (
            matches[match_index + 1].start()
            if match_index + 1 < len(matches)
            else len(source_chat)
        )
        messages[match.group(1)] = source_chat[match.end() : end].strip()
    return messages


def parse_message_groups(text: str, source_chat: str) -> list[list[str]]:
    """Parse a compact grouping response and verify exact message coverage."""

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines or lines[-1] != "@@COMPLETE":
        raise DiaryCompositionError("消息归组缺少@@COMPLETE完整结束标志")
    groups: list[list[str]] = []
    current_id: str | None = None
    current_sources: list[str] | None = None
    for line in lines[:-1]:
        if line.startswith("@@GROUP "):
            if current_id is not None:
                raise DiaryCompositionError("上一个消息组缺少@@END")
            current_id = line.removeprefix("@@GROUP ").strip()
            current_sources = None
            continue
        if line == "@@END":
            if current_id is None or not current_sources:
                raise DiaryCompositionError("消息组缺少sources")
            expected_id = f"G{len(groups) + 1:03d}"
            if current_id != expected_id:
                raise DiaryCompositionError(
                    f"消息组编号不连续：应为{expected_id}"
                )
            groups.append(current_sources)
            current_id = None
            current_sources = None
            continue
        if current_id is None or not line.startswith("sources="):
            raise DiaryCompositionError(f"消息归组存在无法识别的行：{line[:40]}")
        if current_sources is not None:
            raise DiaryCompositionError(f"{current_id}的sources重复")
        current_sources = [
            item.strip()
            for item in line.removeprefix("sources=").split(",")
            if item.strip()
        ]
    if current_id is not None:
        raise DiaryCompositionError("最后一个消息组缺少@@END")
    if not groups:
        raise DiaryCompositionError("消息归组为空")

    source_messages = source_message_map(source_chat)
    if not source_messages:
        raise DiaryCompositionError("原始聊天没有可归组的M编号")
    observed = [source_id for group in groups for source_id in group]
    counts = Counter(observed)
    missing = [source_id for source_id in source_messages if counts[source_id] == 0]
    repeated = [source_id for source_id in source_messages if counts[source_id] > 1]
    unknown = sorted(set(observed) - set(source_messages))
    if missing or repeated or unknown:
        raise DiaryCompositionError(
            f"消息归组覆盖不合格：遗漏={missing}，重复={repeated}，未知={unknown}"
        )
    return groups


def format_message_groups(
    groups: list[list[str]],
    *,
    start_index: int = 1,
) -> str:
    return "\n".join(
        f"G{index:03d}=" + ",".join(source_ids)
        for index, source_ids in enumerate(groups, start=start_index)
    )


def parse_event_consolidation(
    text: str,
    expected_event_ids: list[str],
) -> list[dict[str, Any]]:
    """Validate an ordered, lossless grouping of existing event IDs."""

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines or lines[-1] != "@@COMPLETE":
        raise DiaryCompositionError("叙事合并缺少@@COMPLETE完整结束标志")
    clusters: list[dict[str, Any]] = []
    current_id: str | None = None
    current_events: list[str] | None = None
    current_title: str | None = None
    for line in lines[:-1]:
        if line.startswith("@@GROUP "):
            if current_id is not None:
                raise DiaryCompositionError("上一个叙事合并组缺少@@END")
            current_id = line.removeprefix("@@GROUP ").strip()
            current_events = None
            current_title = None
            continue
        if line == "@@END":
            if current_id is None or not current_events or not current_title:
                raise DiaryCompositionError("叙事合并组字段不完整")
            expected_cluster_id = f"C{len(clusters) + 1:03d}"
            if current_id != expected_cluster_id:
                raise DiaryCompositionError(
                    f"叙事合并组编号不连续：应为{expected_cluster_id}"
                )
            clusters.append(
                {"id": current_id, "event_ids": current_events, "title": current_title}
            )
            current_id = None
            continue
        if current_id is None or "=" not in line:
            raise DiaryCompositionError(f"叙事合并存在无法识别的行：{line[:40]}")
        key, value = line.split("=", 1)
        if key == "events" and current_events is None:
            current_events = [item.strip() for item in value.split(",") if item.strip()]
        elif key == "title" and current_title is None:
            current_title = value.strip()
        else:
            raise DiaryCompositionError(f"叙事合并字段{key}重复或无法识别")
    if current_id is not None:
        raise DiaryCompositionError("最后一个叙事合并组缺少@@END")
    if not clusters:
        raise DiaryCompositionError("叙事合并结果为空")

    observed = [event_id for cluster in clusters for event_id in cluster["event_ids"]]
    counts = Counter(observed)
    missing = [event_id for event_id in expected_event_ids if counts[event_id] == 0]
    repeated = [event_id for event_id in expected_event_ids if counts[event_id] > 1]
    unknown = sorted(set(observed) - set(expected_event_ids))
    if missing or repeated or unknown:
        raise DiaryCompositionError(
            "叙事合并事件覆盖不合格："
            f"遗漏={missing}，重复={repeated}，未知={unknown}"
        )
    if observed != expected_event_ids:
        # A model may try to group two similar moments across an unrelated
        # event.  Preserve its grouping preference only within contiguous
        # chronological runs; restoring order locally cannot delete facts.
        cluster_by_event = {
            event_id: cluster
            for cluster in clusters
            for event_id in cluster["event_ids"]
        }
        ordered_runs: list[dict[str, Any]] = []
        previous_source_cluster: str | None = None
        for event_id in expected_event_ids:
            source_cluster = cluster_by_event[event_id]
            if source_cluster["id"] != previous_source_cluster:
                ordered_runs.append(
                    {
                        "id": f"C{len(ordered_runs) + 1:03d}",
                        "event_ids": [event_id],
                        "title": source_cluster["title"],
                    }
                )
            else:
                ordered_runs[-1]["event_ids"].append(event_id)
            previous_source_cluster = source_cluster["id"]
        clusters = ordered_runs
    return clusters


def consolidate_events(
    events: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge validated adjacent events without discarding their facts."""

    by_id = {event["id"]: event for event in events}
    importance_rank = {"minor": 0, "ordinary": 1, "major": 2}
    kind_priority = (
        "private_relationship",
        "past_story_told_today",
        "decision_or_correction",
        "state_or_plan",
        "today_event",
        "ordinary_moment",
    )

    def unique_text(items: list[str]) -> list[str]:
        return list(dict.fromkeys(item for item in items if item and item != "不确定"))

    merged: list[dict[str, Any]] = []
    for index, cluster in enumerate(clusters, start=1):
        members = [by_id[event_id] for event_id in cluster["event_ids"]]
        kinds = {member["kind"] for member in members}
        kind = next((item for item in kind_priority if item in kinds), members[0]["kind"])
        importance = max(
            (member["importance"] for member in members),
            key=importance_rank.__getitem__,
        )
        details = unique_text(
            [detail for member in members for detail in member["key_details"]]
        )
        summaries = unique_text([member["summary"] for member in members])
        outcomes = unique_text([member["outcome"] for member in members])
        feelings = unique_text([member["rinne_feeling"] for member in members])
        merged.append(
            {
                "id": f"E{index:03d}",
                "title": cluster["title"],
                "kind": kind,
                "importance": importance,
                "summary": "；".join(summaries),
                "key_details": details,
                "outcome": "；".join(outcomes) or "不确定",
                "rinne_feeling": "；".join(feelings) or "不确定",
                "evidence": [],
                "source_ids": [
                    source_id
                    for member in members
                    for source_id in member["source_ids"]
                ],
                "source_event_ids": list(cluster["event_ids"]),
                "mandatory": any(member["mandatory"] for member in members),
                "exact_time_is_significant": any(
                    member.get("exact_time_is_significant", False)
                    for member in members
                ),
            }
        )
    return merged


def validate_consolidation_boundaries(
    events: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
) -> None:
    """Keep protected narrative cores visible after later compression."""

    by_id = {event["id"]: event for event in events}
    protected_kinds = {
        "private_relationship",
        "past_story_told_today",
        "decision_or_correction",
    }
    for cluster in clusters:
        protected = [
            by_id[event_id]
            for event_id in cluster["event_ids"]
            if by_id[event_id]["kind"] in protected_kinds
        ]
        protected_kind_set = {event["kind"] for event in protected}
        private_with_major_protected = (
            "private_relationship" in protected_kind_set
            and any(
                event["kind"] != "private_relationship"
                and event["importance"] == "major"
                for event in protected
            )
        )
        if private_with_major_protected:
            raise DiaryCompositionError(
                f"{cluster['id']}把重要关系事件与另一重大受保护事件混在同一组"
            )
        past_with_major_protected = (
            "past_story_told_today" in protected_kind_set
            and any(
                event["kind"] != "past_story_told_today"
                and event["importance"] == "major"
                for event in protected
            )
        )
        if past_with_major_protected:
            raise DiaryCompositionError(
                f"{cluster['id']}把过去故事与另一重大受保护事件混在同一组"
            )
        decision_count = sum(
            event["kind"] == "decision_or_correction" for event in protected
        )
        if decision_count > 1:
            raise DiaryCompositionError(
                f"{cluster['id']}合并了多个独立重大决定"
            )


def format_grouped_messages(
    groups: list[list[str]],
    source_chat: str,
    *,
    start_index: int = 1,
) -> str:
    messages = source_message_map(source_chat)
    sections = []
    for group_index, source_ids in enumerate(groups, start=start_index):
        lines = [f"【G{group_index:03d}的原始消息】"]
        for source_id in source_ids:
            if source_id not in messages:
                raise DiaryCompositionError(f"消息{source_id}不存在")
            lines.append(f"[{source_id}] {messages[source_id]}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate)
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        raise DiaryCompositionError("事件清单不是JSON对象")
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise DiaryCompositionError(f"事件清单JSON无法解析：{exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise DiaryCompositionError("事件清单JSON顶层不是对象")
    return parsed


def _extract_compact_events(text: str) -> list[dict[str, Any]]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines or lines[-1] != "@@COMPLETE":
        raise DiaryCompositionError("事件清单缺少@@COMPLETE完整结束标志")

    events: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in lines[:-1]:
        if line.startswith("@@EVENT "):
            if current is not None:
                # The next explicit event header is an unambiguous boundary;
                # tolerate a missing decorative @@END and validate fields later.
                events.append(current)
            current = {
                "id": line.removeprefix("@@EVENT ").strip(),
                "key_details": [],
                "evidence": [],
                "source_ids": [],
            }
            continue
        if line == "@@END":
            if current is None:
                raise DiaryCompositionError("出现了没有事件的@@END")
            events.append(current)
            current = None
            continue
        if current is None or "=" not in line:
            raise DiaryCompositionError(f"事件清单存在无法识别的行：{line[:40]}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            raise DiaryCompositionError(f"事件字段{key}为空")
        mapped = {"feeling": "rinne_feeling"}.get(key, key)
        if mapped == "detail":
            current["key_details"].append(value)
        elif mapped == "evidence":
            current["evidence"].append(value)
        elif mapped == "sources":
            if current["source_ids"]:
                raise DiaryCompositionError("事件字段sources重复")
            current["source_ids"] = [
                item.strip() for item in value.split(",") if item.strip()
            ]
        elif mapped in {"mandatory", "exact_time_is_significant"}:
            lowered = value.casefold()
            if lowered not in {"true", "false"}:
                raise DiaryCompositionError(f"{mapped}必须是true或false")
            current[mapped] = lowered == "true"
        elif mapped in {
            "kind",
            "importance",
            "title",
            "summary",
            "outcome",
            "rinne_feeling",
        }:
            if mapped in current:
                raise DiaryCompositionError(f"事件字段{mapped}重复")
            current[mapped] = value
        else:
            raise DiaryCompositionError(f"事件字段{key}无法识别")
    if current is not None:
        events.append(current)
    return events


def parse_event_inventory(
    text: str,
    source_chat: str,
    source_groups: list[list[str]] | None = None,
    *,
    event_id_offset: int = 0,
) -> list[dict[str, Any]]:
    """Parse and validate a grounded, ordered event inventory."""

    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("```"):
        payload = _extract_json_object(text)
        events = payload.get("events")
    else:
        events = _extract_compact_events(text)
    if not isinstance(events, list) or not events:
        raise DiaryCompositionError("事件清单为空或events不是数组")

    required_text = ("title", "kind", "importance", "summary", "outcome", "rinne_feeling")
    allowed_importance = {"major", "ordinary", "minor"}
    source_messages = source_message_map(source_chat)
    if source_groups is not None:
        if len(events) != len(source_groups):
            raise DiaryCompositionError(
                f"事件数{len(events)}与消息组数{len(source_groups)}不一致"
            )
        for event, group in zip(events, source_groups, strict=True):
            event["source_ids"] = list(group)
    normalized: list[dict[str, Any]] = []
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            raise DiaryCompositionError(f"第{index}个事件不是对象")
        expected_id = f"E{event_id_offset + index:03d}"
        if event.get("id") != expected_id:
            raise DiaryCompositionError(
                f"事件编号不连续：第{index}个应为{expected_id}"
            )
        for field in required_text:
            value = event.get(field)
            if not isinstance(value, str) or not value.strip():
                raise DiaryCompositionError(f"{expected_id}缺少{field}")
        normalized_importance = {
            "important": "major",
            "high": "major",
            "medium": "ordinary",
            "normal": "ordinary",
            "low": "minor",
            "重要": "major",
            "普通": "ordinary",
            "次要": "minor",
            "细小": "minor",
        }.get(event["importance"].strip().casefold(), event["importance"])
        event["importance"] = normalized_importance
        if event["importance"] not in allowed_importance:
            raise DiaryCompositionError(f"{expected_id}的重要程度无效")
        details = event.get("key_details")
        evidence = event.get("evidence")
        if not isinstance(details, list) or not all(
            isinstance(item, str) and item.strip() for item in details
        ):
            raise DiaryCompositionError(f"{expected_id}的key_details无效")
        # The prompt's short field targets are writing guidance.  Reject only
        # clearly runaway fields here: a slightly long but grounded summary
        # must not invalidate an otherwise complete day or silently lose facts.
        if len(event["title"]) > 60:
            raise DiaryCompositionError(f"{expected_id}的title异常过长")
        if len(event["summary"]) > 1200:
            raise DiaryCompositionError(f"{expected_id}的summary异常过长")
        if len(event["outcome"]) > 400:
            raise DiaryCompositionError(f"{expected_id}的outcome异常过长")
        if len(event["rinne_feeling"]) > 400:
            raise DiaryCompositionError(f"{expected_id}的rinne_feeling异常过长")
        if len(details) > 20 or any(len(item) > 400 for item in details):
            raise DiaryCompositionError(f"{expected_id}的key_details异常膨胀")
        if not isinstance(evidence, list) or len(evidence) > 2:
            raise DiaryCompositionError(f"{expected_id}的原文证据字段无效")
        if not source_messages and not evidence:
            raise DiaryCompositionError(f"{expected_id}缺少可验证证据")
        for quote in evidence:
            if not isinstance(quote, str) or not quote.strip():
                raise DiaryCompositionError(f"{expected_id}存在空证据")
            if len(quote.strip()) > 120:
                raise DiaryCompositionError(f"{expected_id}的原文证据异常过长")
            if quote.strip() not in source_chat:
                raise DiaryCompositionError(
                    f"{expected_id}的证据不在目标日原始聊天中：{quote[:24]}"
                )
        if not isinstance(event.get("mandatory"), bool):
            raise DiaryCompositionError(f"{expected_id}的mandatory不是布尔值")
        event.setdefault("exact_time_is_significant", False)
        if not isinstance(event["exact_time_is_significant"], bool):
            raise DiaryCompositionError(
                f"{expected_id}的exact_time_is_significant不是布尔值"
            )
        if event["exact_time_is_significant"] and event["importance"] != "major":
            raise DiaryCompositionError(
                f"{expected_id}只有major事件才能保留精确钟点"
            )
        event_source_ids = event.get("source_ids")
        if source_messages:
            if not isinstance(event_source_ids, list) or not event_source_ids:
                raise DiaryCompositionError(f"{expected_id}缺少sources")
            if not all(isinstance(item, str) for item in event_source_ids):
                raise DiaryCompositionError(f"{expected_id}的sources无效")
        normalized.append(event)

    if source_messages:
        observed_ids = [
            source_id
            for event in normalized
            for source_id in event["source_ids"]
        ]
        counts = Counter(observed_ids)
        missing = [source_id for source_id in source_messages if counts[source_id] == 0]
        repeated = [source_id for source_id in source_messages if counts[source_id] > 1]
        unknown = sorted(set(observed_ids) - set(source_messages))
        if missing or repeated or unknown:
            raise DiaryCompositionError(
                f"消息覆盖不合格：遗漏={missing}，重复={repeated}，未知={unknown}"
            )

        private_pattern = re.compile(
            r"相处边界|关系边界|互相支持|彼此承诺|达成共识"
        )
        for event in normalized:
            contains_private_source = any(
                private_pattern.search(source_messages[source_id])
                for source_id in event["source_ids"]
            )
            if contains_private_source:
                event["kind"] = "private_relationship"
                event["mandatory"] = True
    return normalized


def target_diary_characters(events: list[dict[str, Any]]) -> tuple[int, int]:
    """Return a dynamic guidance range without using it to drop events."""

    weights = {"major": 360, "ordinary": 160, "minor": 80}
    center = 180 + sum(weights[event["importance"]] for event in events)
    lower = max(500, int(center * 0.72))
    upper = min(3600, max(lower + 250, int(center * 1.22)))
    return lower, upper


def paragraph_character_budgets(
    events: list[dict[str, Any]],
    total_upper: int,
) -> dict[str, int]:
    """Split the diary ceiling into concrete per-event paragraph ceilings."""

    if not events:
        return {}
    weights = {"major": 3.0, "ordinary": 1.65, "minor": 1.0}
    # Reserve room for the date title, marker lines and paragraph separators.
    usable = max(240, total_upper - 120 - 8 * len(events))
    raw_weights = [weights[event["importance"]] for event in events]
    weight_sum = sum(raw_weights)
    budgets = {
        event["id"]: max(60, int(usable * weight / weight_sum))
        for event, weight in zip(events, raw_weights, strict=True)
    }
    excess = sum(budgets.values()) - usable
    while excess > 0:
        event_id = max(budgets, key=budgets.__getitem__)
        reducible = budgets[event_id] - 60
        if reducible <= 0:
            break
        reduction = min(excess, reducible)
        budgets[event_id] -= reduction
        excess -= reduction
    return budgets


def format_inventory_for_writer(events: list[dict[str, Any]]) -> str:
    def clipped(value: Any, limit: int) -> str:
        text = str(value).strip()
        return text if len(text) <= limit else text[: limit - 1] + "…"

    writer_events = []
    for event in events:
        writer_events.append(
            {
                "id": event["id"],
                "title": clipped(event["title"], 80),
                "kind": event["kind"],
                "importance": event["importance"],
                "summary": clipped(event["summary"], 360),
                "key_details": [
                    clipped(detail, 160) for detail in event["key_details"]
                ],
                "outcome": clipped(event["outcome"], 160),
                "rinne_feeling": clipped(event["rinne_feeling"], 160),
                "mandatory": event["mandatory"],
                "exact_time_is_significant": event.get(
                    "exact_time_is_significant", False
                ),
            }
        )
    return json.dumps({"events": writer_events}, ensure_ascii=False, indent=2)


_MARKER_RE = re.compile(r"^\s*\[\[((?:E\d{3})(?:\s*,\s*E\d{3})*)\]\]\s*$")

# Exact clock readings make an ordinary diary sound like a transcript. Dates,
# coarse periods such as "下午", and relations such as "晚饭前" are not matched.
_EXACT_CLOCK_RE = re.compile(
    r"(?<!\d)(?:[01]?\d|2[0-3])\s*[:：]\s*[0-5]\d(?!\d)"
    r"|(?<!\d)(?:[01]?\d|2[0-3])\s*[点时](?:\s*[0-5]?\d\s*分?)?"
    r"|(?:凌晨|清晨|早上|上午|中午|下午|傍晚|晚上|夜里|深夜)"
    r"\s*[零〇一二两三四五六七八九十]{1,3}\s*[点时]"
    r"(?:\s*(?:半|一刻|三刻|[零〇一二两三四五六七八九十]{1,3}分))?"
    r"|[零〇一二两三四五六七八九十]{1,3}\s*[点时]\s*"
    r"(?:半|一刻|三刻|[零〇一二两三四五六七八九十]{1,3}分|多|左右)"
)


def exact_clock_references(text: str) -> list[str]:
    """Return transcript-like exact clock expressions found in diary prose."""

    return [match.group(0) for match in _EXACT_CLOCK_RE.finditer(text)]


def normalize_marked_diary_order(text: str, expected_ids: list[str]) -> str:
    """Mechanically restore marked paragraphs to the validated event order."""

    lines = text.strip().splitlines()
    marker_positions: list[tuple[int, list[str]]] = []
    for index, line in enumerate(lines):
        match = _MARKER_RE.match(line)
        if match:
            marker_positions.append(
                (index, [item.strip() for item in match.group(1).split(",")])
            )
    if not marker_positions:
        return text.strip()
    observed = [event_id for _, ids in marker_positions for event_id in ids]
    if Counter(observed) != Counter(expected_ids):
        return text.strip()
    expected_position = {event_id: index for index, event_id in enumerate(expected_ids)}
    blocks: list[tuple[int, list[str]]] = []
    for position_index, (line_index, marker_ids) in enumerate(marker_positions):
        positions = [expected_position[event_id] for event_id in marker_ids]
        if positions != list(range(positions[0], positions[0] + len(positions))):
            return text.strip()
        next_index = (
            marker_positions[position_index + 1][0]
            if position_index + 1 < len(marker_positions)
            else len(lines)
        )
        blocks.append((positions[0], lines[line_index:next_index]))
    if observed == expected_ids:
        return text.strip()
    prefix = lines[: marker_positions[0][0]]
    reordered = prefix + [line for _, block in sorted(blocks) for line in block]
    return "\n".join(reordered).strip()


def apply_language_audit(
    marked_diary: str,
    audit_text: str,
    *,
    max_issues: int | None = None,
    max_span_characters: int = 240,
    minimum_good_ratio: float = 0.0,
) -> str:
    """Apply validated, local one-line language fixes inside their event blocks."""

    lines = [line.strip() for line in audit_text.strip().splitlines() if line.strip()]
    if not lines or lines[-1] != "@@COMPLETE":
        raise DiaryCompositionError("语言审计缺少@@COMPLETE完整结束标志")
    issues: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in lines[:-1]:
        if line.startswith("@@ISSUE "):
            if current is not None:
                raise DiaryCompositionError("语言审计问题缺少@@END")
            event_id = line.removeprefix("@@ISSUE ").strip()
            if not re.fullmatch(r"E\d{3}", event_id):
                raise DiaryCompositionError("语言审计事件编号无效")
            current = {"event_id": event_id}
            continue
        if line == "@@END":
            if current is None or set(current) != {"event_id", "bad", "good"}:
                raise DiaryCompositionError("语言审计问题字段不完整")
            issues.append(current)
            current = None
            continue
        if current is None or "=" not in line:
            raise DiaryCompositionError(f"语言审计存在无法识别的行：{line[:40]}")
        key, value = line.split("=", 1)
        if key not in {"bad", "good"} or key in current or not value.strip():
            raise DiaryCompositionError("语言审计替换字段无效")
        if len(value) > 240 or "[[" in value or "@@" in value:
            raise DiaryCompositionError("语言审计替换内容越界")
        current[key] = value.strip()
    if current is not None:
        raise DiaryCompositionError("语言审计问题缺少@@END")
    if max_issues is not None and len(issues) > max_issues:
        raise DiaryCompositionError("语言审计试图修改过多位置")

    result = marked_diary
    for issue in issues:
        marker = f"[[{issue['event_id']}]]"
        marker_index = result.find(marker)
        if marker_index < 0:
            raise DiaryCompositionError(f"语言审计引用未知事件{issue['event_id']}")
        next_marker = result.find("[[E", marker_index + len(marker))
        block_end = next_marker if next_marker >= 0 else len(result)
        block = result[marker_index:block_end]
        bad = issue["bad"]
        good = issue["good"]
        if len(bad) > max_span_characters:
            raise DiaryCompositionError("语言审计替换跨度过大")
        if len(good) < len(bad) * minimum_good_ratio:
            raise DiaryCompositionError("语言审计替换删除了过多原文")
        if bad == good or block.count(bad) != 1:
            raise DiaryCompositionError(
                f"{issue['event_id']}的语言审计原句无法唯一定位"
            )
        replaced_block = block.replace(bad, good, 1)
        result = result[:marker_index] + replaced_block + result[block_end:]
    return result


def parse_marked_diary(
    text: str,
    expected_ids: list[str],
    *,
    max_characters: int | None = None,
    required_terms_by_event: dict[str, tuple[str, ...]] | None = None,
    exact_time_allowed_event_ids: set[str] | None = None,
) -> str:
    """Validate exact event coverage and strip backend-only paragraph markers."""

    lines = text.strip().splitlines()
    marker_positions: list[tuple[int, list[str]]] = []
    for index, line in enumerate(lines):
        match = _MARKER_RE.match(line)
        if match:
            marker_positions.append(
                (index, [item.strip() for item in match.group(1).split(",")])
            )
    if not marker_positions:
        raise DiaryCompositionError("日记没有事件覆盖标记")

    observed = [event_id for _, ids in marker_positions for event_id in ids]
    counts = Counter(observed)
    missing = [event_id for event_id in expected_ids if counts[event_id] == 0]
    repeated = [event_id for event_id in expected_ids if counts[event_id] > 1]
    unknown = sorted(set(observed) - set(expected_ids))
    if missing or repeated or unknown:
        raise DiaryCompositionError(
            f"事件覆盖不合格：遗漏={missing}，重复={repeated}，未知={unknown}"
        )
    if observed != expected_ids:
        raise DiaryCompositionError("日记事件标记顺序与真实时间顺序不一致")

    for position_index, (line_index, marker_ids) in enumerate(marker_positions):
        next_index = (
            marker_positions[position_index + 1][0]
            if position_index + 1 < len(marker_positions)
            else len(lines)
        )
        paragraph = "\n".join(lines[line_index + 1 : next_index]).strip()
        if not paragraph:
            raise DiaryCompositionError("存在只有事件标记、没有正文的段落")
        natural_paragraphs = [
            part for part in re.split(r"\n\s*\n", paragraph) if part.strip()
        ]
        if len(natural_paragraphs) != 1:
            raise DiaryCompositionError("同一叙事单元被拆成了多个自然段")
        for event_id in marker_ids:
            required_terms = (required_terms_by_event or {}).get(event_id)
            if required_terms and not any(term in paragraph for term in required_terms):
                raise DiaryCompositionError(
                    f"{event_id}段落没有实际保留受保护事件内容"
                )
        exact_times = exact_clock_references(paragraph)
        if exact_times and not any(
            event_id in (exact_time_allowed_event_ids or set())
            for event_id in marker_ids
        ):
            examples = "、".join(dict.fromkeys(exact_times[:3]))
            raise DiaryCompositionError(
                f"{','.join(marker_ids)}段落包含普通消息的精确钟点：{examples}；"
                "请改成自然时段、行动前后关系或省略"
            )

    cleaned = "\n".join(
        line for line in lines if not _MARKER_RE.match(line)
    ).strip()
    if not cleaned:
        raise DiaryCompositionError("删除事件标记后日记正文为空")
    cleaned_lines = cleaned.splitlines()
    title_pattern = re.compile(
        r"^\s*(?:#{1,6}\s*|\*{1,2})?\d{4}(?:[-年]\d{1,2})"
        r"(?:[-月]\d{1,2}日?)?(?:\*{1,2})?\s*$"
    )
    body = cleaned
    if cleaned_lines and title_pattern.match(cleaned_lines[0]):
        body = "\n".join(cleaned_lines[1:]).lstrip()
    if max_characters is not None and len(body) > max_characters:
        raise DiaryCompositionError(
            f"日记正文{len(body)}字符，超过动态上限{max_characters}字符"
        )
    return cleaned
