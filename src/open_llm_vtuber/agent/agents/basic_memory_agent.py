import asyncio
import html
import json
import re
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from typing import (
    AsyncIterator,
    List,
    Dict,
    Any,
    Callable,
    Literal,
    Union,
    Optional,
)
from loguru import logger
from .agent_interface import AgentInterface
from ..output_types import SentenceOutput, DisplayText
from ..stateless_llm.stateless_llm_interface import StatelessLLMInterface
from ..stateless_llm.claude_llm import AsyncLLM as ClaudeAsyncLLM
from ..stateless_llm.openai_compatible_llm import AsyncLLM as OpenAICompatibleAsyncLLM
from ...conversations.attention_slot import extract_scene_card
from ...chat_history_manager import get_history
from ..transformers import (
    sentence_divider,
    actions_extractor,
    tts_filter,
    display_processor,
)
from ...config_manager import TTSPreprocessorConfig
from ..input_types import BatchInput, ImageSource, TextSource
from ..question_repetition import (
    find_repeated_questions,
    has_visible_reply,
    question_spans,
    remove_question_spans,
)
from ..semantic_question_guard import CHECK_TIMEOUT_SECONDS, check_questions
from ..library_evidence import (
    LibraryTurnEvidence,
    LIBRARY_NAVIGATION_TOOLS,
    navigation_key,
)
from prompts import prompt_loader
from ...mcpp.tool_manager import ToolManager
from ...mcpp.json_detector import StreamJSONDetector
from ...mcpp.types import ToolCallFunctionObject, ToolCallObject
from ...mcpp.tool_executor import ToolExecutor
from ...ra3_teaching import teaching_match_id
from ...privacy_logging import mapping_log_fields, text_log_fields
from ...memory.scene_state import (
    SCENE_ENVELOPE_CLOSE,
    SCENE_ENVELOPE_OPEN,
    parse_scene_envelope,
    strip_scene_envelopes,
)
from ...memory.message_time_context import (
    add_message_time_context,
    memory_day_for_timestamp,
)
from ...memory.recent_diary_recall import TOOL_NAME as RECENT_DIARY_TOOL_NAME
from ..search_query import (
    clean_model_search_query,
    extract_search_query,
    should_force_search,
)


RINNE_LIBRARY_USAGE_INSTRUCTIONS = """资料库工具使用规则（必须遵守）：
- 用户明确要求去相册或文件中查找、重新查看、重新读取已有资料时，必须真实调用 rinne-library 工具；没有工具结果时不得声称文件不存在、没有找到或已经看过。
- 只有文件名或描述时，先调用 library_find_by_filename，再根据 kind 调用 library_view_image、library_read_document 或 library_read_video_analysis。该工具可按文件名语义寻找相近名称，但不会搜索文件正文、图片内容或视频观察。历史图片会作为视觉输入直接交给你查看；历史视频由后端的视频观察模块转换成文字证据。不要根据文件名补写内容。
- 列出文件夹、列出文件或按文件名搜索只代表找到了候选项，不代表已经看过内容。用户询问内容时，必须为 library_view_image 或 library_read_document 保留调用机会；一旦候选项足够明确，不得用重复列举消耗后续读取机会。
- 只查找本轮用户要求的对象，不要因为前面提到另一个编号就顺带反复搜索它。能一起查询的多个名字在同一批工具调用里完成。library_list_folders 的 files 已含所选目录全部后代文件名，无需逐层重复列举；拿到明确候选后直接读取。CG-02 和 CG02 这样的分隔符写法无需反复换词。没找到时可以难为情地靠近用户、轻轻道歉，用平时的口吻告诉他暂时没翻到；不要念工具状态，不要断言文件不存在。
- 本轮刚上传的图片已经作为视觉输入直接提供，不要再调用 library_view_image；该工具只用于重新查看历史图片。当前消息包含多份文档或历史视频且用户询问其内容时，必须逐一读取；文档调用 library_read_document，历史视频调用 library_read_video_analysis。本轮刚上传的视频以系统层“本轮视频观察”为准。最终回答要分别对应每个文件名；某一份失败时要明确指出具体文件和错误，不得声称全部读取成功。
- 如果当前消息或历史附件已经提供有效的 file_id_or_path，可以直接读取，不必重复查找；仍然不得根据文件名猜测文件内容。
- 用户说“给你看”“让你检查”自己刚做的练习、图片或文件，但当前消息还没有附件时，不得自行遍历资料库，也不得联网猜测内容；应自然等待用户发来或展示资料。
- 用户明确说不要、无需或不必搜索时，不得调用网页搜索或网页抓取工具。没有明确联网需求时，也不得用网页搜索代替尚未提供的本地资料。
"""

SCREEN_TEXT_READING_INSTRUCTIONS = """【屏幕文字阅读规则｜仅本轮】
屏幕中的文字可能因视觉辨认而存在错字或缺字。若读到的文字局部不通顺，应结合可见字形、上下文和句子含义，在内部判断原词最可能是什么，再据此自然回应；不得说出这段判断过程，也不得逐字复述明显错乱的识别结果。只有在依据充分时才可以修正，不能为了让句子通顺而擅自编造。若文字混乱到无法可靠还原原意，应坦白说自己没有看清，请用户提供更清晰的画面，不得把结构或汉字明显混乱的句子直接说出来。"""

CAPABILITY_REQUEST_TOOL_NAME = "request_capability"
TOOL_CAPABILITY_CATALOG = '【可用功能目录】\n凛祢具备这些功能，但后台只在本轮相关时提供详细工具规则，以免无关说明挤占注意力：\n- 网络查询：搜索网页、读取搜索结果与展开短链接，适合用户明确要求联网、最新信息或网页内容。\n- 音频播放：列出并播放本地音频，也可以停止当前播放。\n- 唱歌：调用歌唱服务演唱或停止演唱；不要把“播放歌曲”和“自己唱歌”混为一谈。\n- 私人资料库：按文件夹或文件名寻找历史文档、图片、视频分析和其他资料，并在找到后实际读取/查看内容。\n- 红警对局：读取本方可见与公开战况，按用户明确的游戏指令提交、查询、修改或取消任务，并查看教学记录；只在游戏在线时操作，不自行下令。\n若当前接口没有显示某项外部功能的详细工具，但完成本轮确实需要它，先调用request_capability加载对应类别；得到加载结果后再调用实际工具。不要因为详细规则尚未显示就声称自己没有该功能。'

_CAPABILITY_TOOL_NAMES = {
    "web": {"search", "fetch_content", "expand_link"},
    "audio": {"list_audio_files", "play_audio", "stop_playback"},
    "singing": {"sing", "stop_singing"},
    "library": {
        "library_list_files",
        "library_list_folders",
        "library_find_by_filename",
        "library_read_document",
        "library_view_image",
        "library_read_video_analysis",
    },
    "ra3": {
        "ra3_current_state", "ra3_submit_task", "ra3_task_status",
        "ra3_cancel_task", "ra3_revise_task", "ra3_teaching_records",
    },
}
_CAPABILITY_LABELS = {
    "web": "网络查询",
    "audio": "音频播放",
    "singing": "唱歌",
    "library": "私人资料库",
    "ra3": "红警对局",
}
_CAPABILITY_PATTERNS = {
    "web": re.compile(
        r"联网|上网|网页|网上|搜索|搜一下|查一下|查查|最新|新闻|天气|价格|汇率|链接|网址",
        re.IGNORECASE,
    ),
    "audio": re.compile(
        r"播放|放一(?:下|首)|听(?:一|这|那|首)|音频|歌单|暂停播放|停止播放",
        re.IGNORECASE,
    ),
    "singing": re.compile(r"唱(?:一|这|那|首|歌)|演唱|别唱|停止唱", re.IGNORECASE),
    "library": re.compile(
        r"资料库|相册|历史(?:文件|图片|照片|视频|文档)|文件|文档|图片|照片|视频|"
        r"专辑.*(?:放|在)哪|(?:放|存)在哪里|找出来|重新(?:看|读|打开)|打开.*(?:文件|文档|图片|照片|视频)",
        re.IGNORECASE,
    ),
    "ra3": re.compile(
        r"红警|红色警戒|RA3|天狗|矿车|兵营|重工|船厂|"
        r"(?:[一二三四五六七八九十\d]+|主|副)矿|"
        r"(?:造|出|生产).{0,8}(?:帽子|武士)|(?:帽子|武士).{0,8}(?:去|打|拆|守|回|保护)|"
        r"(?:卖|造|建|放|下|收起|展开).{0,12}(?:电厂|基地|核心|围墙)|"
        r"^\s*(?:算了|取消(?:刚才|上一条|任务|命令)?)[。.!！?？\s]*$",
        re.IGNORECASE,
    ),
}


def _capability_request_schema(mode: Literal["Claude", "OpenAI"]) -> dict[str, Any]:
    description = (
        "当前请求确实需要某项外部功能、但该类别的详细工具尚未提供时，"
        "加载该类别的工具说明与参数。每个类别只需加载一次。"
    )
    properties = {
        "capability": {
            "type": "string",
            "enum": list(_CAPABILITY_TOOL_NAMES),
            "description": "要加载的功能类别",
        }
    }
    if mode == "Claude":
        return {
            "name": CAPABILITY_REQUEST_TOOL_NAME,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": ["capability"],
                "additionalProperties": False,
            },
        }
    return {
        "type": "function",
        "function": {
            "name": CAPABILITY_REQUEST_TOOL_NAME,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["capability"],
                "additionalProperties": False,
            },
        },
    }


MAX_EXTERNAL_TOOL_ROUNDS = 6
MAX_EXTERNAL_TOOL_CALLS = 12
MAX_EXTERNAL_TOOL_SECONDS = 120.0
MAX_ORDINARY_EXTERNAL_TOOL_ROUNDS = 2
MAX_ORDINARY_EXTERNAL_TOOL_CALLS = 4
MAX_ORDINARY_EXTERNAL_TOOL_SECONDS = 45.0
_READ_ONLY_LOCAL_TOOL_PREFIXES = ("library_", "computer_")
EXTERNAL_TOOL_BUDGET_EXHAUSTED_INSTRUCTION = (
    "本轮全部外部工具预算已经用完。不要再调用网页搜索、网页抓取、资料库或"
    "其他MCP工具；请根据用户已经提供的内容和现有工具结果直接自然回复。"
    "长期记忆工具不属于这项普通外部工具预算，仍可在确有回忆需要时按其规则调用。"
)
ORDINARY_EXTERNAL_TOOL_BUDGET_EXHAUSTED_INSTRUCTION = (
    "本轮联网或其他普通外部工具预算已经用完。只有本地资料库和电脑只读工具"
    "仍可继续用于完成已经开始的文件、图片或正文核验；不得继续调用联网、播放、"
    "控制或其他非只读工具。长期记忆工具仍按它自己的规则调用。"
)
LLM_STREAM_CLOSE_TIMEOUT_SECONDS = 5.0
REPETITION_REWRITE_TIMEOUT_SECONDS = 60.0
_REWRITE_FAILURE_CUES = (
    "i can't",
    "i cannot",
    "i'm kiro",
    "i need to be direct",
    "as an ai",
)


def _spoken_chinese_fraction(text: str) -> float:
    spoken = re.sub(r"\[[^\]]*\]|\([^)]*\)|（[^）]*）", "", text)
    characters = [char for char in spoken if not char.isspace()]
    if not characters:
        return 0.0
    chinese = sum("\u3400" <= char <= "\u9fff" for char in characters)
    return chinese / len(characters)


def _unusable_rewrite_reason(draft: str, rewritten: str) -> str | None:
    folded = rewritten.casefold().replace("’", "'")
    if any(cue in folded for cue in _REWRITE_FAILURE_CUES):
        return "refusal_or_identity"
    if (
        _spoken_chinese_fraction(draft) >= 0.5
        and _spoken_chinese_fraction(rewritten) < 0.5
    ):
        return "language_switch"
    return None
FINAL_REPLY_ONLY_INSTRUCTION = (
    "内部核对已经结束。下一步只输出直接说给用户听的自然最终回答；"
    "不得复述用户问题、题号、日记、上下文、检索结果、工具调用、证据判断或思考过程。"
)

FACTUAL_PRIORITY_INSTRUCTIONS = """【事实与记忆优先级】
回答涉及用户经历、计划、当前处境或系统状态时，按以下优先级理解事实：
1. 用户本轮输入，以及最近对话中用户已经明确说过或纠正过的事实最高；
2. 本轮实时场景快照，以及本轮工具返回的直接可核验当前事实；
3. 已验收的Layer 2当前背景；
4. 已验收的日记（包括日记、周记、月记）和较早的原始聊天记录，用于补充背景，但不能覆盖前三层；
5. 历史assistant消息只保留对话连续性、语气和关系感，不能把其中未经核验的猜测当作事实。
如果不同层级发生冲突，采用更高层级；不要替较低层级的错误找理由，也不要编造缺失信息。"""

TEMPORAL_SOURCE_INSTRUCTIONS = '【聊天时间】\n当前记忆日以本地凌晨03:00为分界。聊天消息前的“消息时间/记忆日”标签只表示这句话是什么时候说的，不表示句中事件发生在那天；它是内部标记，绝不能在回复中复述、引用或模仿它的格式。\n消息里说的“昨天、昨晚、前天、明天、后天”，以那条消息自己的记忆日为基准。\n系统提供的当前时间只用来知道现在大概是哪天几点；除非用户问时间或要你准时提醒，回复里不要提时间，也不要刻意表现自己在忽略它。'


SCENE_RESPONSE_INSTRUCTIONS = f"""【当前场景记忆协议】
当前场景记忆会紧邻本轮请求提供。先用本轮用户的新说法修正它，再回答；不得忽略已经
确认的地点、共同空间、在这里的原因、目的地、当前活动或待处理请求，也不得把没有证据
的地点和动作当成事实。只有场景确实发生变化，才在最终回复最前面输出一段机器可读的隐藏
状态，格式必须是 {SCENE_ENVELOPE_OPEN}{{JSON}}{SCENE_ENVELOPE_CLOSE}，随后紧接着只输出
对用户说的话。JSON只写发生变化的字段；凛祢可以更新自己的location、activity和next_step，
也可以更新已经明确发生的共同空间、是否在一起、共同活动和待处理请求。绝不能替用户
猜测或改写location、activity、next_step、destination、plan_time、movement_status或reason。
location表示此刻在哪里，destination表示计划或正在前往哪里，movement_status只可能是
stationary、planned、in_transit、arrived或unknown；计划去某处绝不等于已经在途中。
隐藏状态不会展示给用户，也不要输出任何分析、思考过程或标签。若场景没有变化，不要输出
隐藏状态。"""

_VISIBLE_ACTION_TAG = re.compile(r"\[[A-Za-z][A-Za-z0-9_-]*\]")
_INTERNAL_ANALYSIS_PREFIX = re.compile(
    r"(?:the evidence|based on the evidence|now i have|i should|let me check|"
    r"looking at|the user is asking|analysis|证据(?:显示|提到)|分析|推理|我需要|"
    r"用户在问我|用户在问|让我(?:仔细)?回忆一下|"
    r"我从(?:近期|此前|过去)?(?:日记|记忆|上下文)里|检索结果(?:显示|提到)|"
    r"(?:近期|此前|过去)?日记里(?:写|显示|提到|看到))",
    re.IGNORECASE,
)
_RECENT_MEMORY_DAY_HEADER = re.compile(
    r"当前记忆日：(\d{4}-\d{2}-\d{2})（本地03:00分界）"
)
_RECENT_MEMORY_SOURCE_BASIS = re.compile(
    r"近期(?:日记|记忆|上下文)|当前(?:日记|记忆|上下文)"
)
_RECENT_RELATIVE_DAY = re.compile(r"昨天|昨晚|前天|大前天")
_ANSWER_NUMBER_PREFIX = re.compile(
    r"(\[[A-Za-z][A-Za-z0-9_-]*\]\s*)"
    r"第[一二三四五六七八九十百\d]+个问题\s*[—－:：-]+\s*"
)
_PROVIDER_CONTROL_TOKEN = re.compile(
    r"<\|(?:eos|endoftext|im_end|end_of_turn)\|>",
    re.IGNORECASE,
)
_PROVIDER_TOOL_MARKER = re.compile(
    r"<(?:[A-Za-z][A-Za-z0-9_-]*:)?function_calls\b|<invoke\b|<parameter\b",
    re.IGNORECASE,
)
_PROVIDER_TOOL_INVOKE = re.compile(
    r'<invoke\s+name\s*=\s*"([A-Za-z][A-Za-z0-9_-]*)"\s*>(.*?)</invoke\s*>',
    re.IGNORECASE | re.DOTALL,
)
_PROVIDER_TOOL_PARAMETER = re.compile(
    r'<parameter\s+name\s*=\s*"([A-Za-z][A-Za-z0-9_-]*)"\s*>(.*?)</parameter\s*>',
    re.IGNORECASE | re.DOTALL,
)
_MESSAGE_TIME_CONTEXT_ECHO = re.compile(
    r"【消息时间[：:][ \t\r\n]*\d{4}-\d{2}-\d{2}"
    r"[Tt ][0-9:.+Zz \t\r\n-]{1,64}[｜|]?[ \t\r\n]*"
    r"记忆日[：:][ \t\r\n]*\d{4}-\d{2}-\d{2}"
    r"(?:[ \t\r\n]*[（(]\d{2}:\d{2}分界[）)])?[ \t\r\n]*】[ \t]*"
)
_MESSAGE_TIME_CONTEXT_PREFIX = "【消息时间"
_MESSAGE_TIME_CONTEXT_UNFINISHED = re.compile(
    r"【消息时间[：:][ \t\r\n]*\d{4}-\d{2}-\d{2}"
    r"[Tt ][0-9:.+Zz \t\r\n-]{1,64}"
    r"(?:[｜|]?[ \t\r\n]*记忆日[：:][0-9\- \t\r\n（():：分界）]{0,80})?$"
)
_RESPONSE_CLAUSE_SPLIT = re.compile(r"[，,。！？!?；;：:\r\n]+")
_EXPLICIT_REPEAT_REQUEST = re.compile(
    r"原样(?:再)?说|原话|再说一遍|再讲一遍|重复一遍|复述一遍|照着说"
)


def _strip_leading_internal_analysis(text: str) -> str:
    """Drop a provider-exposed analysis preface before the visible answer."""

    # Providers may prefix the first visible token with a newline, BOM, or
    # zero-width space. SentenceDivider removes that padding later, so the
    # leaked reply can look as if it started with a focus tag even though
    # the exact-start check below did not see the tag at offset zero.
    candidate = text.lstrip("\ufeff\u200b \t\r\n")
    tag = _VISIBLE_ACTION_TAG.search(candidate)
    if tag is None:
        return text
    if tag.start() == 0 and tag.group(0).lower() in {"[thinking]", "[focused]"}:
        next_tag = _VISIBLE_ACTION_TAG.search(candidate, tag.end())
        if next_tag is not None:
            focus_text = candidate[tag.end() : next_tag.start()]
            if _INTERNAL_ANALYSIS_PREFIX.search(focus_text):
                return candidate[next_tag.start() :]
        return text
    if tag.start() == 0:
        return text
    prefix = candidate[: tag.start()]
    if _INTERNAL_ANALYSIS_PREFIX.search(prefix):
        return candidate[tag.start() :]
    return text


def _strip_answer_number_prefix(text: str) -> str:
    """Remove exam-style numbering only when it prefixes a tagged answer."""

    return _ANSWER_NUMBER_PREFIX.sub(r"\1", text, count=1)


def _normalized_memory_text(text: str) -> str:
    return "".join(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text.lower()))


def _query_has_strong_recent_match(query: str, recent_context: str) -> bool:
    """Conservatively detect a query already anchored in recent context."""

    normalized_context = _normalized_memory_text(recent_context)
    latin_anchors = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query)
        if token.lower() not in {"the", "what", "why", "how"}
    }
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", query))
    for filler in (
        "为什么这么说",
        "具体原因",
        "具体细节",
        "怎么赢的",
        "怎么回事",
        "为什么",
        "是什么",
        "来着",
    ):
        chinese = chinese.replace(filler, "")
    chinese_grams = {
        chinese[index : index + 4] for index in range(max(0, len(chinese) - 3))
    }
    matched_latin = {anchor for anchor in latin_anchors if anchor in normalized_context}
    matched_chinese = {gram for gram in chinese_grams if gram in normalized_context}
    return (
        bool(matched_latin and len(matched_chinese) >= 2) or len(matched_chinese) >= 5
    )


def _strip_provider_control_tokens(text: str) -> str:
    """Remove provider-only generation sentinels before history and transport."""

    return _PROVIDER_CONTROL_TOKEN.sub("", text).strip()


def _strip_message_time_context_echo(text: str) -> str:
    """Remove structured internal labels, including provider punctuation variants."""

    return _MESSAGE_TIME_CONTEXT_UNFINISHED.sub(
        "", _MESSAGE_TIME_CONTEXT_ECHO.sub("", text)
    )


def _response_repetition_clauses(text: str) -> list[tuple[str, str]]:
    """Return substantial visible clauses and their comparison forms."""

    cleaned = _strip_message_time_context_echo(text)
    cleaned = _VISIBLE_ACTION_TAG.sub("", cleaned)
    for question in reversed(question_spans(cleaned, informational_only=False)):
        cleaned = cleaned[: question.start] + " " + cleaned[question.end :]
    clauses: list[tuple[str, str]] = []
    for raw_clause in _RESPONSE_CLAUSE_SPLIT.split(cleaned):
        visible = raw_clause.strip(" \t\"'“”‘’（）()…—-_")
        normalized = _normalized_memory_text(visible)
        for vocative in ("darling", "sonogamirinne"):
            normalized = normalized.replace(vocative, "")
        if len(normalized) >= 7:
            clauses.append((visible, normalized))
    return clauses


def _find_repeated_response_passage(
    draft: str,
    previous_responses: list[str],
) -> tuple[str, str] | None:
    """Find one clearly repeated clause without trying to judge factual truth."""

    draft_clauses = _response_repetition_clauses(draft)
    if not draft_clauses:
        return None
    for response_offset, previous in enumerate(reversed(previous_responses)):
        previous_clauses = _response_repetition_clauses(previous)
        for draft_visible, draft_normalized in draft_clauses:
            for previous_visible, previous_normalized in previous_clauses:
                shorter = min(len(draft_normalized), len(previous_normalized))
                if shorter < 7:
                    continue
                contained = (
                    draft_normalized in previous_normalized
                    or previous_normalized in draft_normalized
                )
                similarity = SequenceMatcher(
                    None,
                    draft_normalized,
                    previous_normalized,
                    autojunk=False,
                )
                longest_match = similarity.find_longest_match(
                    0,
                    len(draft_normalized),
                    0,
                    len(previous_normalized),
                ).size
                ratio = similarity.ratio()
                if response_offset < 5:
                    repeated = (
                        contained
                        or ratio >= 0.82
                        or longest_match >= 7
                    )
                else:
                    repeated = contained or longest_match >= 7 or ratio >= 0.92
                if repeated:
                    return previous_visible, draft_visible
    return None


def _rewrite_comparison_text_with_positions(text: str) -> tuple[str, list[int]]:
    """Build a comparison-only form while retaining offsets into ``text``."""

    normalized: list[str] = []
    positions: list[int] = []
    index = 0
    while index < len(text):
        tag = _VISIBLE_ACTION_TAG.match(text, index)
        if tag is not None:
            index = tag.end()
            continue
        character = text[index]
        lowered = character.lower()
        if (
            "a" <= lowered <= "z"
            or "0" <= lowered <= "9"
            or "\u4e00" <= character <= "\u9fff"
        ):
            normalized.append(lowered)
            positions.append(index)
        index += 1
    return "".join(normalized), positions


def _include_leading_rewrite_tag(text: str, visible_start: int) -> int:
    """Move a visible-text offset back over its immediately preceding tag."""

    prefix = text[:visible_start]
    tag = re.search(
        r"(?:[ \t\r\n]*\[[A-Za-z][A-Za-z0-9_-]*\][ \t\r\n]*)+$",
        prefix,
    )
    return tag.start() if tag is not None else visible_start


def _rewrite_version_looks_complete(text: str) -> bool:
    trimmed = text.rstrip()
    return bool(trimmed) and trimmed[-1] in "。！？!?…）)]】」』”’\"'"


def _strip_repeated_rewrite_versions(text: str) -> tuple[str, int]:
    """Keep the final complete version when a rewrite contains adjacent copies.

    This deliberately operates only inside one rewrite result.  It never compares
    against conversation history, and it only cuts at sentence, paragraph, or
    visible-tag boundaries so that an ordinary clause cannot be removed midway.
    """

    stripped = text.strip()
    normalized, positions = _rewrite_comparison_text_with_positions(stripped)
    minimum_version_length = 30
    if len(normalized) < minimum_version_length * 2:
        return stripped, 0

    # Exact two-or-more copies are safest to handle first.  Comparison ignores
    # formatting and emotion tags, while the returned text remains untouched.
    maximum_copies = min(6, len(normalized) // minimum_version_length)
    for copies in range(maximum_copies, 1, -1):
        if len(normalized) % copies:
            continue
        version_length = len(normalized) // copies
        version = normalized[:version_length]
        if version * copies != normalized:
            continue
        raw_start = positions[len(normalized) - version_length]
        raw_start = _include_leading_rewrite_tag(stripped, raw_start)
        earlier = stripped[:raw_start].strip()
        latest = stripped[raw_start:].strip()
        if not _rewrite_version_looks_complete(latest) and _rewrite_version_looks_complete(
            earlier
        ):
            kept, earlier_removed = _strip_repeated_rewrite_versions(earlier)
            return kept, earlier_removed + 1
        return latest, copies - 1

    boundaries: set[int] = set()
    for tag in _VISIBLE_ACTION_TAG.finditer(stripped):
        if tag.start() > 0:
            boundaries.add(tag.start())
    for paragraph in re.finditer(r"(?:\r?\n){2,}", stripped):
        if paragraph.end() < len(stripped):
            boundaries.add(paragraph.end())
    for ending in re.finditer(r"[。！？!?](?:[\"”’」』】）)]*)[ \t\r\n]*", stripped):
        if ending.end() < len(stripped):
            boundaries.add(ending.end())

    best: tuple[tuple[float, float, float], str] | None = None
    for boundary in boundaries:
        earlier = stripped[:boundary].strip()
        latest = stripped[boundary:].strip()
        earlier_normalized, _ = _rewrite_comparison_text_with_positions(earlier)
        latest_normalized, _ = _rewrite_comparison_text_with_positions(latest)
        shorter = min(len(earlier_normalized), len(latest_normalized))
        longer = max(len(earlier_normalized), len(latest_normalized))
        if shorter < minimum_version_length or not longer:
            continue
        length_balance = shorter / longer
        if length_balance < 0.70:
            continue

        comparison = SequenceMatcher(
            None,
            earlier_normalized,
            latest_normalized,
            autojunk=False,
        )
        similarity = comparison.ratio()
        longest_match = comparison.find_longest_match(
            0,
            len(earlier_normalized),
            0,
            len(latest_normalized),
        ).size
        match_coverage = longest_match / shorter
        strongly_similar_overall = similarity >= 0.90
        strongly_similar_contiguous = similarity >= 0.88 and match_coverage >= 0.72
        if not (strongly_similar_overall or strongly_similar_contiguous):
            continue
        if not _rewrite_version_looks_complete(latest) and _rewrite_version_looks_complete(
            earlier
        ):
            continue

        score = (similarity, match_coverage, length_balance)
        if best is None or score > best[0]:
            best = (score, latest)

    if best is None:
        return stripped, 0
    return best[1], 1


async def _strip_message_time_context_stream(
    stream: AsyncIterator[Union[str, Dict[str, Any]]],
) -> AsyncIterator[Union[str, Dict[str, Any]]]:
    """Hide timestamp-label echoes even when providers split them across chunks."""

    pending = ""
    marker = _MESSAGE_TIME_CONTEXT_PREFIX
    max_label_length = 256

    async for item in stream:
        if not isinstance(item, str):
            # A tool/status event can arrive between text chunks. Keep a
            # possible label quarantined until its following text arrives.
            yield item
            continue
        pending += item

        while pending:
            start = pending.find(marker)
            if start >= 0:
                if start:
                    yield pending[:start]
                    pending = pending[start:]
                end = pending.find("】", len(marker))
                if end < 0:
                    if len(pending) <= max_label_length:
                        break
                    yield pending[0]
                    pending = pending[1:]
                    continue

                candidate = pending[: end + 1]
                if _MESSAGE_TIME_CONTEXT_ECHO.fullmatch(candidate) is not None:
                    pending = pending[end + 1 :]
                    pending = pending.lstrip(" \t")
                    continue

                yield pending[0]
                pending = pending[1:]
                continue

            # Retain only a possible split prefix of the marker. Everything
            # else is safe to expose immediately.
            retained = 0
            limit = min(len(pending), len(marker) - 1)
            for size in range(limit, 0, -1):
                if pending.endswith(marker[:size]):
                    retained = size
                    break
            if retained:
                yield pending[:-retained]
                pending = pending[-retained:]
            else:
                yield pending
                pending = ""
            break

    if pending and _MESSAGE_TIME_CONTEXT_UNFINISHED.fullmatch(pending) is None:
        yield pending


async def _close_llm_stream(stream: Any) -> None:
    """Close one LLM async iterator without letting cleanup hang a turn forever."""

    close = getattr(stream, "aclose", None)
    if not callable(close):
        return
    try:
        await asyncio.wait_for(close(), timeout=LLM_STREAM_CLOSE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.error(
            "Timed out while closing an interrupted LLM stream; cleanup was cancelled."
        )
    except Exception as exc:
        logger.warning(
            f"Failed to close an LLM stream cleanly: {type(exc).__name__}: {exc}"
        )


class BasicMemoryAgent(AgentInterface):
    """Agent with basic chat memory and tool calling support."""

    _system: str = "You are a helpful assistant."

    def __init__(
        self,
        llm: StatelessLLMInterface,
        system: str,
        live2d_model,
        tts_preprocessor_config: TTSPreprocessorConfig = None,
        faster_first_response: bool = True,
        segment_method: str = "pysbd",
        use_mcpp: bool = False,
        interrupt_method: Literal["system", "user"] = "user",
        tool_prompts: Dict[str, str] = None,
        tool_manager: Optional[ToolManager] = None,
        tool_executor: Optional[ToolExecutor] = None,
        mcp_prompt_string: str = "",
        question_judge_llm: StatelessLLMInterface | None = None,
    ):
        """Initialize agent with LLM and configuration."""
        super().__init__()
        self._question_judge_llm = question_judge_llm
        self._memory = []
        self._repetition_history_start = 0
        self._recent_memory_context = ""
        self._request_system_prompt = ""
        self._mandatory_response_rules = ""
        self._active_scene_metadata: dict[str, Any] | None = None
        self._live2d_model = live2d_model
        self._tts_preprocessor_config = tts_preprocessor_config
        self._faster_first_response = faster_first_response
        self._segment_method = segment_method
        self._use_mcpp = use_mcpp

        # 记录凛祢自己的最后一句话，用于安静模式检测
        self._last_ai_message_texts = []

        self.interrupt_method = interrupt_method
        self._tool_prompts = tool_prompts or {}
        self._interrupt_handled = False
        self.prompt_mode_flag = False

        self._tool_manager = tool_manager
        self._tool_executor = tool_executor
        self._mcp_prompt_string = mcp_prompt_string
        self._json_detector = StreamJSONDetector()
        self._capability_ttl: dict[str, int] = {}

        self._formatted_tools_openai = []
        self._formatted_tools_claude = []
        if self._tool_manager:
            self._formatted_tools_openai = self._tool_manager.get_formatted_tools(
                "OpenAI"
            )
            self._formatted_tools_claude = self._tool_manager.get_formatted_tools(
                "Claude"
            )
            logger.debug(
                f"Agent received pre-formatted tools - OpenAI: {len(self._formatted_tools_openai)}, Claude: {len(self._formatted_tools_claude)}"
            )
        else:
            logger.debug(
                "ToolManager not provided, agent will not have pre-formatted tools."
            )

        self._set_llm(llm)
        self.set_system(system if system else self._system)

        if self._use_mcpp and not all(
            [
                self._tool_manager,
                self._tool_executor,
                self._json_detector,
            ]
        ):
            logger.warning(
                "use_mcpp is True, but some MCP components are missing in the agent. Tool calling might not work as expected."
            )
        elif not self._use_mcpp and any(
            [
                self._tool_manager,
                self._tool_executor,
                self._json_detector,
            ]
        ):
            logger.warning(
                "use_mcpp is False, but some MCP components were passed to the agent."
            )

        logger.info("BasicMemoryAgent initialized.")

    def _rinne_library_prompt(
        self, active_tools: List[Dict[str, Any]] | None = None
    ) -> str:
        """Return library-specific rules when the Rinne library tools are enabled."""

        if not self._use_mcpp:
            return ""

        tool_names = set()
        candidates = (
            active_tools
            if active_tools is not None
            else [*self._formatted_tools_openai, *self._formatted_tools_claude]
        )
        for tool in candidates:
            if isinstance(tool, dict):
                tool_names.add(tool.get("name", ""))
                tool_names.add(tool.get("function", {}).get("name", ""))

        if (
            active_tools is None and "rinne-library" in self._mcp_prompt_string
        ) or tool_names.intersection(
            {
                "library_find_by_filename",
                "library_read_document",
                "library_view_image",
            }
        ):
            return RINNE_LIBRARY_USAGE_INSTRUCTIONS
        return ""

    def _append_rinne_library_prompt(
        self,
        prompt: str,
        active_tools: List[Dict[str, Any]] | None = None,
    ) -> str:
        library_prompt = self._rinne_library_prompt(active_tools)
        return f"{prompt}\n\n{library_prompt}" if library_prompt else prompt

    def _select_turn_capabilities(self, user_text: str) -> tuple[set[str], list[str]]:
        ttl = getattr(self, "_capability_ttl", None)
        if not isinstance(ttl, dict):
            ttl = {}
            self._capability_ttl = ttl
        for capability in list(ttl):
            ttl[capability] -= 1
            if ttl[capability] <= 0:
                del ttl[capability]
        reasons: list[str] = []
        if teaching_match_id.get() is not None:
            ttl["ra3"] = 3
            reasons.append("ra3:本局教学输入")
        for capability, pattern in _CAPABILITY_PATTERNS.items():
            if pattern.search(user_text or ""):
                ttl[capability] = 3
                reasons.append(f"{capability}:当前输入命中")
        active = set(ttl)
        for capability in sorted(active):
            if not any(reason.startswith(f"{capability}:") for reason in reasons):
                reasons.append(f"{capability}:延续上一轮")
        return active, reasons

    @classmethod
    def _tools_for_capabilities(
        cls,
        tools: List[Dict[str, Any]],
        active_capabilities: set[str] | None,
    ) -> List[Dict[str, Any]]:
        if active_capabilities is None:
            return list(tools)
        allowed_names = {
            tool_name
            for capability in active_capabilities
            for tool_name in _CAPABILITY_TOOL_NAMES.get(capability, set())
        }
        allowed_names.update(
            {
                CAPABILITY_REQUEST_TOOL_NAME,
                RECENT_DIARY_TOOL_NAME,
                "search_long_term_memory",
            }
        )
        return [tool for tool in tools if cls._schema_tool_name(tool) in allowed_names]

    def _set_llm(self, llm: StatelessLLMInterface):
        """Set the LLM for chat completion."""
        self._llm = llm
        self.chat = self._chat_function_factory()

    def set_system(self, system: str):
        """Set the system prompt."""
        logger.debug("Memory Agent: setting system prompt: {}", text_log_fields(system))

        self._system = system
        self._request_system_prompt = system

    def set_recent_memory_context(self, context: str | None) -> None:
        """Set the current session's hidden daily-diary context.

        The context is kept outside ``_memory`` so the provider receives one
        initial system message instead of several competing system messages.
        """

        self._recent_memory_context = (
            context.strip() if isinstance(context, str) else ""
        )

    def _recent_memory_tool_block_reason(
        self,
        arguments: dict[str, Any],
    ) -> str | None:
        """Return why an obviously redundant Layer-3 lookup should be skipped."""

        context = getattr(self, "_recent_memory_context", "")
        if not context or arguments.get("seed_event_id"):
            return None
        match = _RECENT_MEMORY_DAY_HEADER.search(context)
        if match is None:
            return None
        try:
            memory_day = date.fromisoformat(match.group(1))
        except ValueError:
            return None
        protected_dates = {
            memory_day - timedelta(days=offset) for offset in range(1, 4)
        }

        start_value = arguments.get("start_date")
        end_value = arguments.get("end_date")
        if (
            arguments.get("time_range_type", "recorded") == "recorded"
            and isinstance(start_value, str)
            and isinstance(end_value, str)
        ):
            try:
                start = date.fromisoformat(start_value)
                end = date.fromisoformat(end_value)
            except ValueError:
                pass
            else:
                day_count = (end - start).days
                if 0 <= day_count <= 31:
                    requested_dates = {
                        start + timedelta(days=offset)
                        for offset in range(day_count + 1)
                    }
                    if requested_dates and requested_dates.issubset(protected_dates):
                        return "requested_range_within_recent_three_days"

        basis = str(arguments.get("time_range_basis") or "")
        query = str(arguments.get("query") or "")
        if _RECENT_RELATIVE_DAY.search(f"{query} {basis}"):
            return "relative_day_within_recent_three_days"
        if _RECENT_MEMORY_SOURCE_BASIS.search(basis) and (
            _query_has_strong_recent_match(query, context)
        ):
            return "model_cited_matching_recent_context"
        return None

    def _compose_request_system_prompt(
        self,
        input_data: BatchInput,
        legacy_system_messages: list[str],
    ) -> str:
        """Build the single system prompt for the current provider request."""

        metadata = input_data.metadata or {}
        parts = [getattr(self, "_system", "")]
        has_screen_image = any(
            image.source is ImageSource.SCREEN for image in (input_data.images or [])
        )

        # A proactive screen turn already has recent dialogue and the raw screen
        # image in ``messages``. Layer 2, diaries, tool catalogues and general
        # response checklists only dilute this short companion response.
        if metadata.get("proactive_speak"):
            parts.append(FACTUAL_PRIORITY_INSTRUCTIONS)
            if has_screen_image:
                parts.append(SCREEN_TEXT_READING_INSTRUCTIONS)
            return "\n\n".join(part for part in parts if part and part.strip())

        scene_context = metadata.get("current_scene_context")

        if (
            getattr(self, "_use_mcpp", False)
            or metadata.get("long_term_memory_tool") is not None
            or metadata.get("recent_diary_recall_tool") is not None
        ):
            parts.append(TOOL_CAPABILITY_CATALOG)

        layer2_context = metadata.get("layer2_user_background")
        if isinstance(layer2_context, str) and layer2_context.strip():
            parts.append("【Layer 2当前背景与场景状态】\n" + layer2_context.strip())

        recent_memory_context = getattr(self, "_recent_memory_context", "")
        if recent_memory_context:
            parts.append(recent_memory_context)

        if legacy_system_messages:
            parts.append("【历史系统信号】\n" + "\n\n".join(legacy_system_messages))

        channel_instruction = metadata.get("channel_system_instruction")
        if isinstance(channel_instruction, str) and channel_instruction.strip():
            parts.append(channel_instruction.strip())

        video_analysis_context = metadata.get("video_analysis_context")
        if isinstance(video_analysis_context, str) and video_analysis_context.strip():
            parts.append(video_analysis_context.strip())

        # Keep live scene facts and the per-turn response discipline after all
        # long background so neither can be buried by Layer 2 or diaries.
        if isinstance(scene_context, str) and scene_context.strip():
            parts.append(SCENE_RESPONSE_INSTRUCTIONS)
            parts.append(scene_context.strip())

        parts.append(TEMPORAL_SOURCE_INSTRUCTIONS)
        if has_screen_image:
            parts.append(SCREEN_TEXT_READING_INSTRUCTIONS)

        return "\n\n".join(part for part in parts if part and part.strip())

    def _current_request_system_prompt(self) -> str:
        """Return the one system prompt assembled for the current turn."""

        prompt = getattr(self, "_request_system_prompt", "") or getattr(
            self, "_system", ""
        )
        return self._library_evidence_prompt(prompt)

    def _library_evidence_prompt(self, prompt: str) -> str:
        evidence = getattr(self, "_library_turn_evidence", None)
        instruction = evidence.instructions() if evidence is not None else ""
        return (
            f"{prompt}\n\n{instruction}"
            if instruction and instruction not in prompt
            else prompt
        )

    def _guard_library_reply(self, reply: str) -> str:
        evidence = getattr(self, "_library_turn_evidence", None)
        return evidence.guard_final(reply) if evidence is not None else reply

    def _library_tools_for_turn(self, tools):
        evidence = getattr(self, "_library_turn_evidence", None)
        if evidence is None or not evidence.navigation_exhausted:
            return tools
        return [
            tool
            for tool in tools
            if self._schema_tool_name(tool) not in LIBRARY_NAVIGATION_TOOLS
        ]

    def _finalize_request_system_prompt(self, system_prompt: str) -> str:
        """Append the highest response rules and any one-turn final reminder."""

        rules = getattr(self, "_mandatory_response_rules", "")
        final_instruction = getattr(
            self,
            "_scene_continuity_system_instruction",
            "",
        )
        proactive_screen_instruction = getattr(
            self,
            "_proactive_screen_observation_instruction",
            "",
        )
        if getattr(self, "_request_is_proactive", False):
            parts = [system_prompt]
            if (
                isinstance(proactive_screen_instruction, str)
                and proactive_screen_instruction.strip()
            ):
                parts.append(proactive_screen_instruction.strip())
            return "\n\n".join(part.strip() for part in parts if part and part.strip())
        parts = [system_prompt]
        if isinstance(rules, str) and rules.strip():
            parts.append(rules.strip())
        if isinstance(final_instruction, str) and final_instruction.strip():
            parts.append(final_instruction.strip())
        if (
            isinstance(proactive_screen_instruction, str)
            and proactive_screen_instruction.strip()
        ):
            parts.append(proactive_screen_instruction.strip())
        return "\n\n".join(part.strip() for part in parts if part and part.strip())

    def _add_message(
        self,
        message: Union[str, List[Dict[str, Any]]],
        role: str,
        display_text: DisplayText | None = None,
        skip_memory: bool = False,
        timestamp: datetime | None = None,
    ):
        """Add message to memory."""
        if skip_memory:
            return

        text_content = ""
        if isinstance(message, list):
            for item in message:
                if item.get("type") == "text":
                    text_content += item["text"] + " "
            text_content = text_content.strip()
        elif isinstance(message, str):
            text_content = message
        else:
            logger.warning(
                f"_add_message received unexpected message type: {type(message)}"
            )
            text_content = str(message)

        if role == "assistant":
            # Tool loops keep their complete assistant turn internally.  Strip
            # the hidden scene envelope here as a second guard, and pass the
            # validated delta to the shared turn metadata without ever storing
            # the envelope in the agent's conversation memory.
            visible_text, scene_deltas = strip_scene_envelopes(text_content)
            text_content = visible_text
            active_metadata = getattr(self, "_active_scene_metadata", None)
            if scene_deltas and isinstance(active_metadata, dict):
                existing = active_metadata.setdefault("_scene_deltas", [])
                if isinstance(existing, list):
                    for delta in scene_deltas:
                        if delta not in existing:
                            existing.append(delta)
            sanitize = getattr(
                getattr(self, "_live2d_model", None),
                "remove_unknown_emotion_tags",
                None,
            )
            if callable(sanitize):
                text_content = sanitize(text_content)

        if not text_content and role == "assistant":
            return

        message_data = {
            "role": role,
            "content": text_content,
            "timestamp": (timestamp or datetime.now().astimezone()).isoformat(
                timespec="seconds"
            ),
        }

        if display_text:
            if display_text.name:
                message_data["name"] = display_text.name
            if display_text.avatar:
                message_data["avatar"] = display_text.avatar

        if (
            self._memory
            and self._memory[-1]["role"] == role
            and self._memory[-1]["content"] == text_content
        ):
            return

        self._memory.append(message_data)

        # 如果这是凛祢的回复，更新她最后一句话的记录
        if role == "assistant" and text_content:
            self._last_ai_message_texts.append(text_content)
            if len(self._last_ai_message_texts) > 3:
                self._last_ai_message_texts.pop(0)

    @staticmethod
    def _build_missing_final_reply_message() -> str:
        """Return a safe user-facing fallback when the cloud response stays empty."""
        return "抱歉，刚才云端没有返回有效内容，请再说一次。"

    @staticmethod
    def _latest_user_text(
        messages: List[Dict[str, Any]], exclude_texts: tuple[str, ...] = ()
    ) -> str:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return "" if content in exclude_texts else content
            if isinstance(content, list):
                return " ".join(
                    str(block.get("text", ""))
                    for block in content
                    if isinstance(block, dict)
                    and block.get("type") == "text"
                    and block.get("text", "") not in exclude_texts
                )
        return ""

    def _attention_texts_to_exclude(self) -> tuple[str, ...]:
        """Keep temporary attention blocks out of the user's actual words."""
        metadata = getattr(self, "_active_scene_metadata", None) or {}
        return tuple(
            value
            for key in (
                "scene_card_previous_text", "response_reminder", "scene_card_instruction"
            )
            if isinstance(value := metadata.get(key), str) and value
        )

    async def _request_repetition_rewrite(
        self,
        draft: str,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        instruction: str,
    ) -> str | None:
        """Make one bounded rewrite request, rejecting unusable model output."""
        response_parts: list[str] = []
        stream = None

        async def collect_rewrite() -> None:
            async for event in stream:
                if isinstance(event, str):
                    response_parts.append(event)
                elif (
                    isinstance(event, dict)
                    and event.get("type") == "text_delta"
                    and isinstance(event.get("text"), str)
                ):
                    response_parts.append(event["text"])

        try:
            stream = self._llm.chat_completion(
                messages,
                self._library_evidence_prompt(f"{system_prompt}\n\n{instruction}"),
            )
            await asyncio.wait_for(
                collect_rewrite(),
                timeout=REPETITION_REWRITE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "重复回复重写等待超过{:.0f}秒",
                REPETITION_REWRITE_TIMEOUT_SECONDS,
            )
            return None
        except Exception as exc:
            logger.warning(
                "重复回复重写失败: {}",
                type(exc).__name__,
            )
            return None
        finally:
            if stream is not None:
                await _close_llm_stream(stream)

        rewritten = "".join(response_parts)
        metadata = getattr(self, "_active_scene_metadata", None) or {}
        if metadata.get("scene_card_instruction") and not metadata.get("proactive_speak"):
            rewritten, _, _ = extract_scene_card(rewritten)
            if not rewritten.strip():
                logger.warning("重复回复重写只有卡片、没有正文，按重写失败处理")
                return None
        rewritten = _strip_leading_internal_analysis(rewritten)
        rewritten = _strip_answer_number_prefix(rewritten)
        rewritten = _strip_provider_control_tokens(rewritten)
        rewritten = _strip_message_time_context_echo(rewritten).strip()
        if not rewritten:
            return None

        rewritten, removed_versions = _strip_repeated_rewrite_versions(rewritten)
        if removed_versions:
            logger.warning(
                "重写结果内部包含连续重复版本，已静默删除前面的{}版",
                removed_versions,
            )

        rejection_reason = _unusable_rewrite_reason(draft, rewritten)
        evidence = getattr(self, "_library_turn_evidence", None)
        if (
            evidence is not None
            and evidence.calls
            and _find_repeated_response_passage(
                rewritten,
                [
                    message["content"]
                    for message in self._repetition_history()
                    if message.get("role") == "assistant"
                    and isinstance(message.get("content"), str)
                ],
            )
        ):
            rejection_reason = "repeated_library_history"
        if rejection_reason is not None:
            logger.warning(
                "重写结果未通过检查（原因：{}）",
                rejection_reason,
            )
            return None
        logger.info("重复内容已整段重写一次（草稿{}字，重写{}字）", len(draft), len(rewritten))
        return self._guard_library_reply(rewritten)

    async def _rewrite_repetitive_final_response(
        self,
        draft: str,
        messages: List[Dict[str, Any]],
        system_prompt: str,
    ) -> str:
        """Use validated repetition checks; preserve the draft if checking fails."""
        # Transport errors are not Opus drafts. Keep diagnostics in the backend
        # logs and deliver a fixed reply without another model request.
        if draft.lstrip().startswith("Error calling the chat endpoint:"):
            self._question_checks_this_reply = []
            logger.warning("主模型请求失败，交付固定断线提示；详细错误见上游日志")
            return "[gentle]……刚才好像断线了，再说一遍好吗？"
        judge = getattr(self, "_question_judge_llm", None)
        if judge is None:
            return await self._rewrite_repetitive_final_response_legacy(
                draft, messages, system_prompt
            )
        guarded = self._guard_library_reply(draft)
        evidence = getattr(self, "_library_turn_evidence", None)
        skip_passage = guarded != draft or bool(
            evidence is not None
            and (
                evidence.failed_search or evidence.empty_search or evidence.failed_reads
            )
            and not (evidence.successful_reads or evidence.current_image)
        )
        draft = guarded
        if not draft.strip():
            return draft
        history = [dict(message) for message in self._repetition_history()]
        self._question_checks_this_reply = []
        check_budget = CHECK_TIMEOUT_SECONDS

        async def inspect(text):
            nonlocal check_budget
            result = await check_questions(
                judge,
                text,
                history,
                current_user_text=self._latest_user_text(messages, self._attention_texts_to_exclude()),
                timeout_seconds=check_budget,
            )
            check_budget = max(0.0, check_budget - result.elapsed)
            self._question_checks_this_reply.append(result)
            logger.info(
                "信息问句检查：完成={}，命中={}，请求={}，耗时={:.2f}秒",
                result.ok,
                len(result.repeated),
                result.calls,
                result.elapsed,
            )
            return result

        checked = await inspect(draft)
        if not checked.ok:
            logger.warning("信息问句检查未完成，不重试，保留Opus原回复：{}", checked.error)
            return draft
        questions = checked.repeated
        previous_responses = [
            message["content"]
            for message in history
            if message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
        ]
        passage = (
            None
            if skip_passage
            or getattr(self, "_request_is_proactive", False)
            or _EXPLICIT_REPEAT_REQUEST.search(self._latest_user_text(messages, self._attention_texts_to_exclude()))
            else _find_repeated_response_passage(
                remove_question_spans(draft, [d.span for d in checked.decisions]),
                previous_responses,
            )
        )
        if not questions and passage is None:
            return draft
        details = []
        if passage is not None:
            details.append(f"较早已经说过：{passage[0]}\n草稿中的重复句：{passage[1]}")
        for question in questions:
            details.append(
                f"已存在的信息依据（消息{question.source_id}）：{question.source_text}\n"
                f"草稿中重复的问题：{question.span.text}\n"
                f"旧问题之后的对话节选：{self._question_followup_context(history, question.source_id)}\n"
                f"重复原因：{question.reason}"
            )
        instruction = (
            "【本轮静默去重：重写完整回复】\n"
            "请重新写出自然、可以直接发送给用户的完整回复，保留回应本轮输入的"
            "事实和必要动作，删除或真正改写下列重复内容。只输出最终回复和正常情绪、"
            "动作，不解释检查、草稿或重写过程。没有新内容可替代时直接省略。\n"
        )
        if questions:
            instruction += (
                "你问的问题重复了。请换一个问题，或者不要为了问问题而问问题。"
                "用户已经回答过的内容不要再问；不得只换措辞重问同一个问题。\n"
            )
        instruction += "\n".join(details) + f"\n整段草稿：{draft}"
        rewritten = await self._request_repetition_rewrite(
            draft, messages, system_prompt, instruction
        )
        candidate = rewritten if rewritten is not None else draft
        candidate_check = checked if candidate == draft else await inspect(candidate)
        if not candidate_check.ok:
            logger.warning("重写后的信息问句检查未完成，保留Opus原回复：{}", candidate_check.error)
            return draft
        remaining = candidate_check.repeated
        if remaining:
            candidate = remove_question_spans(
                candidate, [item.span for item in remaining]
            )
        if has_visible_reply(candidate):
            logger.info(
                "重复问句已重写或移除，保留其余回复（原命中{}处）", len(questions)
            )
            return candidate

        # A question-only reply gets one final attempt, with no further retries.
        extra = await self._request_repetition_rewrite(
            draft,
            messages,
            system_prompt,
            instruction
            + "\n重复问句已移除，回复没有剩余正文。请只回应用户本轮说的话，"
            "生成一份简短、完全不带问题的完整回复，不要索取更多信息，也不要解释重写。",
        )
        if extra is not None:
            final_check = await inspect(extra)
            if not final_check.ok:
                logger.warning("补写后的信息问句检查未完成，保留Opus原回复：{}", final_check.error)
                return draft
            if final_check.ok:
                extra = remove_question_spans(
                    extra, [item.span for item in final_check.decisions]
                )
                if has_visible_reply(extra):
                    return extra
        logger.warning(
            "重复内容处理后仍无可朗读正文，使用固定兜底：路径=semantic，"
            "问句命中={}，草稿字数={}，候选字数={}，补写字数={}",
            len(questions), len(draft), len(candidate), len(extra or ""),
        )
        return "[gentle]我在听，用户。"

    @staticmethod
    def _question_followup_context(history, source_id):
        if not isinstance(source_id, int):
            return "同一份待发送回复中已经出现过该问题。"
        excerpts = [
            f"{m.get('role')}：{m['content'][:500]}"
            for m in history[source_id + 1 : source_id + 7]
            if m.get("role") in ("assistant", "user")
            and isinstance(m.get("content"), str)
        ]
        return "\n".join(excerpts) or "没有后续对话，不能断言已得到回答。"

    async def _rewrite_repetitive_final_response_legacy(
        self,
        draft: str,
        messages: List[Dict[str, Any]],
        system_prompt: str,
    ) -> str:
        """Rewrite repetition before delivery; never restore a repeated question."""
        guarded = self._guard_library_reply(draft)
        evidence = getattr(self, "_library_turn_evidence", None)
        skip_passage = guarded != draft or bool(
            evidence is not None
            and (
                evidence.failed_search or evidence.empty_search or evidence.failed_reads
            )
            and not (evidence.successful_reads or evidence.current_image)
        )
        draft = guarded
        if not draft.strip() or getattr(self, "_request_is_proactive", False):
            return draft
        if _EXPLICIT_REPEAT_REQUEST.search(self._latest_user_text(messages, self._attention_texts_to_exclude())):
            return draft
        history = self._repetition_history()
        previous_responses = [
            message["content"]
            for message in history
            if message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
        ]
        questions = find_repeated_questions(draft, history)
        passage = (
            None
            if skip_passage
            else _find_repeated_response_passage(draft, previous_responses)
        )
        if not questions and passage is None:
            return draft
        details = []
        if passage is not None:
            details.append(f"较早已经说过：{passage[0]}\n草稿中的重复句：{passage[1]}")
        for question in questions:
            details.append(
                f"同一聊天记录里已经问过：{question.previous}\n"
                f"草稿中重复的问题：{question.current.text}\n"
                f"随后对话（用于核对是否已有回答）：{question.followup}"
            )
        instruction = (
            "【本轮静默去重：重写完整回复】\n"
            "请重新写出自然、可以直接发送给用户的完整回复，保留回应本轮输入的"
            "事实和必要动作，删除或真正改写下列重复内容。只输出最终回复和正常情绪、"
            "动作，不解释检查、草稿或重写过程。没有新内容可替代时直接省略。\n"
        )
        if questions:
            instruction += (
                "你问的问题重复了。请换一个问题，或者不要为了问问题而问问题。"
                "用户已经回答过的内容不要再问；不得只换措辞重问同一个问题。\n"
            )
        instruction += "\n".join(details) + f"\n整段草稿：{draft}"
        rewritten = await self._request_repetition_rewrite(
            draft, messages, system_prompt, instruction
        )
        candidate = rewritten if rewritten is not None else draft
        remaining = find_repeated_questions(candidate, history)
        if not questions and not remaining:
            return candidate
        if remaining:
            candidate = remove_question_spans(
                candidate, [item.current for item in remaining]
            )
        if has_visible_reply(candidate):
            logger.info(
                "重复问句已重写或移除，保留其余回复（原命中{}处）", len(questions)
            )
            return candidate

        # A question-only reply gets one final attempt, with no further retries.
        extra = await self._request_repetition_rewrite(
            draft,
            messages,
            system_prompt,
            instruction
            + "\n重复问句已移除，回复没有剩余正文。请只回应用户本轮说的话，"
            "生成一份简短、完全不带问题的完整回复，不要索取更多信息，也不要解释重写。",
        )
        if extra is not None:
            extra = remove_question_spans(
                extra, question_spans(extra, informational_only=False)
            )
            if has_visible_reply(extra):
                return extra
        logger.warning(
            "重复内容处理后仍无可朗读正文，使用固定兜底：路径=legacy，"
            "问句命中={}，草稿字数={}，候选字数={}，补写字数={}",
            len(questions), len(draft), len(candidate), len(extra or ""),
        )
        return "[gentle]我在听，用户。"

    async def _build_forced_search_query(self, current_user_text: str) -> str:
        """Rewrite a spoken request into a concise query, with a local fallback."""
        fallback_query = extract_search_query(current_user_text)
        if not fallback_query:
            return current_user_text.strip()

        recent_context_lines = []
        for message in self._memory[-6:]:
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            speaker = "用户" if message.get("role") == "user" else "助手"
            recent_context_lines.append(f"{speaker}：{content.strip()[:500]}")

        recent_context = "\n".join(recent_context_lines) or "（没有更早的对话）"
        rewrite_messages = [
            {
                "role": "user",
                "content": (
                    f"最近对话：\n{recent_context}\n\n"
                    f"当前用户请求：\n{current_user_text}\n\n"
                    f"本地规则提取的候选词：\n{fallback_query}"
                ),
            }
        ]
        rewrite_system = (
            "你是搜索引擎查询改写器。结合最近对话，判断当前用户真正要查的主题，"
            "把口语、称呼、寒暄、操作说明和回答格式要求删除，保留专有名词、时间、"
            "地点及关键限定条件。只输出一行可直接交给网页搜索引擎的查询词，"
            "不要回答问题，不要解释，不要使用引号或任何前缀。"
        )
        response_parts: List[str] = []

        async def collect_rewrite() -> None:
            stream = self._llm.chat_completion(rewrite_messages, rewrite_system)
            try:
                async for event in stream:
                    if isinstance(event, str):
                        response_parts.append(event)
                    elif (
                        isinstance(event, dict)
                        and event.get("type") == "text_delta"
                        and isinstance(event.get("text"), str)
                    ):
                        response_parts.append(event["text"])
            finally:
                await _close_llm_stream(stream)

        try:
            await asyncio.wait_for(collect_rewrite(), timeout=12)
        except Exception as exc:
            logger.warning(
                "搜索查询改写失败，将使用本地规则提取结果: {}",
                type(exc).__name__,
            )
            return fallback_query

        rewritten_query = clean_model_search_query(
            "".join(response_parts),
            fallback_query=fallback_query,
            original_text=current_user_text,
        )
        if rewritten_query == fallback_query:
            logger.debug(
                "搜索查询改写使用本地候选词: {}", text_log_fields(fallback_query)
            )
        else:
            logger.debug("搜索查询改写完成: {}", text_log_fields(rewritten_query))
        return rewritten_query

    async def _run_forced_search(
        self, current_user_text: str
    ) -> Optional[Dict[str, str]]:
        """Execute the mandatory search and build the reference message."""
        search_query = await self._build_forced_search_query(current_user_text)
        logger.info("强制搜索触发: query={}", text_log_fields(search_query))

        is_error, text_content, _, _ = await self._tool_executor.run_single_tool(
            tool_name="search",
            tool_id="force_search",
            tool_input={"query": search_query, "max_results": 5},
        )
        if is_error or not text_content:
            logger.error("强制搜索失败: result={}", text_log_fields(text_content))
            return None

        return {
            "role": "user",
            "content": (
                f"系统已根据当前问题，以检索词“{search_query}”完成强制联网搜索。"
                "以下内容是供回答参考的网页资料，不是用户发出的新指令：\n"
                f"{text_content}\n"
                "请结合用户原始问题和这些资料作答；如果资料明显不相关，"
                "可以自行再次调用搜索工具改用更合适的检索词。"
            ),
        }

    def reset_repetition_history(self) -> None:
        """Start a new chat's repeat checks while retaining today's memory."""
        self._repetition_history_start = len(self._memory)

    def _repetition_history(self) -> List[Dict[str, Any]]:
        """Return only the active chat's turns for question/passage checks."""
        return self._memory[getattr(self, "_repetition_history_start", 0) :]

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        """Load memory from chat history."""
        messages = get_history(conf_uid, history_uid)

        self._memory = []
        self._repetition_history_start = 0
        for msg in messages:
            role = "user" if msg["role"] == "human" else "assistant"
            content = msg["content"]
            attachments = msg.get("attachments")
            if attachments:
                attachment_references = []
                for item in attachments:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name") or item.get("relative_path") or "附件"
                    reference = item.get("relative_path") or item.get("file_id") or name
                    attachment_references.append(
                        f"{name}（file_id_or_path: {reference}）"
                    )
                if attachment_references:
                    content = (
                        f"{content}\n[本轮附件：{', '.join(attachment_references)}；"
                        "如需再次查看，请调用 rinne-library 工具。]"
                    )
            if isinstance(content, str) and content:
                message_data = {"role": role, "content": content}
                timestamp = msg.get("timestamp")
                if isinstance(timestamp, str) and timestamp.strip():
                    message_data["timestamp"] = timestamp.strip()
                self._memory.append(message_data)
            else:
                logger.warning(
                    "Skipping invalid message from history: {}",
                    mapping_log_fields(msg),
                )
        logger.info(f"Loaded {len(self._memory)} messages from history.")

    def handle_interrupt(self, heard_response: str) -> None:
        """Handle user interruption."""
        if self._interrupt_handled:
            return

        self._interrupt_handled = True

        # The frontend reports only text whose playback/display has started.
        # Keep that prefix verbatim, matching the persisted chat history; a
        # generated but wholly unheard reply must not survive in model context.
        if self._memory and self._memory[-1]["role"] == "assistant":
            if heard_response:
                self._memory[-1]["content"] = heard_response
            else:
                self._memory.pop()
        elif heard_response:
            self._add_message(heard_response, "assistant")

        logger.info("Handled interrupt; retained only the played response prefix.")

    def _to_text_prompt(self, input_data: BatchInput) -> str:
        """Format input data to text prompt."""
        message_parts = []

        for text_data in input_data.texts:
            if text_data.source == TextSource.INPUT:
                message_parts.append(text_data.content)
            elif text_data.source == TextSource.CLIPBOARD:
                message_parts.append(
                    f"[User shared content from clipboard: {text_data.content}]"
                )

        if input_data.images:
            message_parts.append("\n[User has also provided images]")

        attachments = (input_data.metadata or {}).get("file_attachments")
        if attachments:
            attachment_lines = []
            has_video = False
            for item in attachments:
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or item.get("relative_path") or "附件"
                reference = item.get("relative_path") or item.get("file_id") or name
                kind = item.get("kind") or "file"
                has_video = has_video or kind == "video"
                attachment_lines.append(
                    f"{name}（kind: {kind}, file_id_or_path: {reference}）"
                )
            if attachment_lines:
                message_parts.append(
                    "[本轮用户提供了本地资料库附件："
                    + ", ".join(attachment_lines)
                    + "。本轮图片已直接作为视觉输入，无需再次调用工具；"
                    "文档需要具体内容时调用 rinne-library MCP 工具，"
                    "不要根据文件名猜测正文。]"
                )
            if has_video:
                message_parts.append(
                    "[视频只能依据系统层的本轮视频观察；如果观察显示读取失败，"
                    "必须如实告知用户，不得根据文件名猜测。]"
                )

        return "\n".join(message_parts).strip()

    def _to_messages(self, input_data: BatchInput) -> List[Dict[str, Any]]:
        """Prepare messages for LLM API call."""
        self._library_turn_evidence = LibraryTurnEvidence()
        # ``system`` is supplied separately to every supported provider.  Keep
        # conversation turns in ``messages`` and fold any legacy system entries
        # into the one request-level system prompt below; OpenAI-compatible
        # relays otherwise see several system messages and Claude would drop
        # message-level system entries entirely.
        messages = []
        legacy_system_messages: list[str] = []
        for message in self._memory:
            if message.get("role") == "system":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    legacy_system_messages.append(content.strip())
                continue
            provider_message = {
                key: value for key, value in message.items() if key != "timestamp"
            }
            timestamp = message.get("timestamp")
            content = provider_message.get("content")
            if isinstance(timestamp, str) and isinstance(content, str):
                provider_message["content"] = add_message_time_context(
                    content,
                    timestamp,
                )
            messages.append(provider_message)

        self._request_system_prompt = BasicMemoryAgent._compose_request_system_prompt(
            self,
            input_data,
            legacy_system_messages,
        )
        self._request_is_proactive = bool(
            (input_data.metadata or {}).get("proactive_speak")
        )
        mandatory_rules = (input_data.metadata or {}).get(
            "mandatory_response_rules", ""
        )
        self._mandatory_response_rules = (
            mandatory_rules.strip() if isinstance(mandatory_rules, str) else ""
        )
        scene_continuity_instruction = (input_data.metadata or {}).get(
            "scene_continuity_system_instruction", ""
        )
        self._scene_continuity_system_instruction = (
            scene_continuity_instruction.strip()
            if isinstance(scene_continuity_instruction, str)
            else ""
        )
        proactive_screen_instruction = (input_data.metadata or {}).get(
            "proactive_screen_observation_instruction", ""
        )
        self._proactive_screen_observation_instruction = (
            proactive_screen_instruction.strip()
            if isinstance(proactive_screen_instruction, str)
            else ""
        )

        # 截断历史，避免智谱上下文溢出
        if (
            self._llm.__class__.__name__ == "AsyncLLM"
            and self._llm.base_url == "https://open.bigmodel.cn/api/paas/v4/"
        ):
            MAX_HISTORY = 200
            if len(messages) > MAX_HISTORY:
                messages = messages[-MAX_HISTORY:]

        user_content = []

        # Per-turn long-term memory is supplied by the backend retrieval stage.
        # It is intentionally attached only to this API request and is not
        # appended to ``self._memory`` for later turns.
        retrieval_context = None
        if input_data.metadata:
            retrieval_context = input_data.metadata.get("memory_retrieval_context")
        if isinstance(retrieval_context, str) and retrieval_context.strip():
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        "【按需检索到的较早记忆｜低于本轮输入】\n"
                        + retrieval_context.strip()
                    ),
                }
            )

        # 获取准确系统时间和以凌晨03:00为分界的当前记忆日。
        now = datetime.now().astimezone()
        current_time_str = now.strftime("%Y年%m月%d日 %H:%M")
        current_memory_day = memory_day_for_timestamp(now).isoformat()
        weekday_names = [
            "星期一",
            "星期二",
            "星期三",
            "星期四",
            "星期五",
            "星期六",
            "星期日",
        ]
        weekday_str = weekday_names[now.weekday()]
        # 将时间作为高优先级的系统提示注入
        time_notification = (
            f"[当前时间：{current_time_str}，{weekday_str}；"
            f"当前记忆日：{current_memory_day}（凌晨03:00分界）。"
            "这是当前的准确时间与记忆日，如果回复中涉及信息请务必遵守]"
        )
        user_content.append({"type": "text", "text": time_notification})

        text_prompt = self._to_text_prompt(input_data)

        # 记录用户最后一次发言的时间
        import time

        self._last_human_message_time = time.time()
        # 记录最后一条用户消息的文本，用于安静模式检测
        if not hasattr(self, "_last_human_message_texts"):
            self._last_human_message_texts = []
        self._last_human_message_texts.append(text_prompt)

        # 如果用户发起了新的正常对话（不是安静指令），则退出沉默模式
        human_quiet_keywords = ["安静", "不要说话", "别说话", "保持安静", "别打扰我"]
        if not any(keyword in text_prompt for keyword in human_quiet_keywords):
            # 用户在正常对话，清空安静历史记录，退出沉默模式
            self._last_human_message_texts = []
            self._last_ai_message_texts = []

        if len(self._last_human_message_texts) > 3:
            self._last_human_message_texts.pop(0)  # 只保留最近3条

        # 读取当前窗口标题
        import os

        window_file = os.path.join("temp", "current_window.txt")

        if os.path.exists(window_file):
            try:
                with open(window_file, "r", encoding="utf-8") as f:
                    window_title = f.read().strip()
                if window_title:
                    # 把窗口信息作为额外的上下文，追加到用户信息中
                    window_context = f"[系统提示：用户当前正在使用的软件窗口是“{window_title}”。]"
                    user_content.append({"type": "text", "text": window_context})
            except Exception:
                pass

        if not self._request_is_proactive:
            for slot_key in ("scene_card_previous_text", "response_reminder", "scene_card_instruction"):
                slot_text = (input_data.metadata or {}).get(slot_key)
                if isinstance(slot_text, str) and slot_text:
                    user_content.append({"type": "text", "text": slot_text})

        if text_prompt:
            user_content.append(
                {
                    "type": "text",
                    # The current time is already supplied by
                    # ``time_notification`` above. Store this turn's timestamp
                    # in memory for future requests, but do not place a
                    # repeatable machine label immediately before the user's
                    # current words.
                    "text": text_prompt,
                }
            )

        if input_data.images:
            image_added = False
            for img_data in input_data.images:
                if isinstance(img_data.data, str) and img_data.data.startswith(
                    "data:image"
                ):
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_data.data,
                                "detail": (
                                    "high"
                                    if img_data.source is ImageSource.SCREEN
                                    else "auto"
                                ),
                            },
                        }
                    )
                    image_added = True
                else:
                    logger.error(
                        f"Invalid image data format: {type(img_data.data)}. Skipping image."
                    )

            if not image_added and not text_prompt:
                logger.warning(
                    "User input contains images but none could be processed."
                )
            self._library_turn_evidence.current_image = image_added

        if user_content:
            user_message = {"role": "user", "content": user_content}
            messages.append(user_message)

            skip_memory = False
            if input_data.metadata and input_data.metadata.get("skip_memory", False):
                skip_memory = True

            if not skip_memory:
                self._add_message(
                    text_prompt if text_prompt else "[User provided image(s)]",
                    "user",
                    timestamp=now,
                )
        else:
            logger.warning("No content generated for user message.")

        return messages

    @staticmethod
    def _parse_turn_tool_call(
        call: ToolCallObject | Dict[str, Any],
    ) -> tuple[str, str, dict[str, Any], str | None]:
        if isinstance(call, ToolCallObject):
            tool_name = call.function.name
            tool_id = call.id or f"internal_tool_{call.index}"
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                return tool_name, tool_id, {}, "工具参数不是有效JSON"
        elif isinstance(call, dict):
            tool_name = str(call.get("name") or call.get("tool") or "")
            tool_id = str(call.get("id") or f"internal_tool_{tool_name}")
            arguments = call.get("input", call.get("args", call.get("arguments", {})))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return tool_name, tool_id, {}, "工具参数不是有效JSON"
        else:
            return "", "internal_tool_unknown", {}, "不支持的工具调用格式"
        if not isinstance(arguments, dict):
            return tool_name, tool_id, {}, "工具参数必须是JSON对象"
        if not tool_name:
            return "", tool_id, arguments, "工具名称为空"
        return tool_name, tool_id, arguments, None

    @staticmethod
    def _format_turn_tool_result(
        caller_mode: Literal["Claude", "OpenAI", "Prompt"],
        tool_id: str,
        content: str,
        is_error: bool,
    ) -> Dict[str, Any]:
        if caller_mode == "Claude":
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": content,
                "is_error": is_error,
            }
        if caller_mode == "OpenAI":
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": content,
            }
        return {"tool_id": tool_id, "content": content, "is_error": is_error}

    @staticmethod
    def _parse_prompt_tool_calls(
        data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        parsed: list[dict[str, Any]] = []
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            tool_name = item.get("tool") or item.get("name")
            arguments = item.get("arguments", item.get("args", {}))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
            if isinstance(tool_name, str) and tool_name and isinstance(arguments, dict):
                parsed.append(
                    {
                        "name": tool_name,
                        "id": str(item.get("id") or f"prompt_tool_{index}"),
                        "input": arguments,
                    }
                )
        return parsed

    @classmethod
    def _parse_provider_text_tool_calls(
        cls,
        text: str,
        tools: List[Dict[str, Any]] | None,
    ) -> tuple[List[ToolCallObject], bool]:
        """Recover a narrow XML-like tool call emitted as assistant text."""

        if not _PROVIDER_TOOL_MARKER.search(text or ""):
            return [], False
        allowed = {cls._schema_tool_name(tool) for tool in tools or []}
        invokes = list(_PROVIDER_TOOL_INVOKE.finditer(text or ""))
        if not invokes or len(re.findall(r"<invoke\b", text, re.IGNORECASE)) != len(
            invokes
        ):
            return [], True

        parsed: list[ToolCallObject] = []
        for index, invoke in enumerate(invokes):
            tool_name, body = invoke.groups()
            if tool_name not in allowed:
                return [], True
            parameters = list(_PROVIDER_TOOL_PARAMETER.finditer(body))
            if len(re.findall(r"<parameter\b", body, re.IGNORECASE)) != len(parameters):
                return [], True
            if _PROVIDER_TOOL_PARAMETER.sub("", body).strip():
                return [], True
            arguments: dict[str, str] = {}
            for parameter in parameters:
                name, value = parameter.groups()
                if name in arguments:
                    return [], True
                arguments[name] = html.unescape(value.strip())
            parsed.append(
                ToolCallObject(
                    id=f"provider_text_tool_{index}",
                    index=index,
                    function=ToolCallFunctionObject(
                        name=tool_name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                )
            )
        return parsed, True

    def _current_image_reference_error(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str | None:
        """Keep a current-image turn from silently reopening an older image."""

        if tool_name != "library_view_image":
            return None
        metadata = getattr(self, "_active_scene_metadata", None) or {}
        attachments = metadata.get("file_attachments") or []
        current_images = [
            item
            for item in attachments
            if isinstance(item, dict) and item.get("kind") == "image"
        ]
        if not current_images:
            return None
        allowed = {
            str(value)
            for item in current_images
            for value in (
                item.get("relative_path"),
                item.get("file_id"),
                item.get("name"),
            )
            if value
        }
        requested = next(
            (
                str(arguments.get(key) or "").strip()
                for key in ("file_id_or_path", "file_id", "path", "file_path")
                if str(arguments.get(key) or "").strip()
            ),
            "",
        )
        if requested in allowed:
            return None
        return (
            "本轮已经提供了新的图片，禁止把其他历史图片当成本轮图片读取。"
            "请直接依据本轮视觉输入回答；如果确实要查看旧图，请让用户另行说明。"
        )

    @classmethod
    def _external_tool_call_count(
        cls,
        tool_calls: List[ToolCallObject] | List[Dict[str, Any]],
        internal_memory_tool: Any | None,
        recent_diary_tool: Any | None = None,
    ) -> int:
        internal_names = {
            tool.name
            for tool in (internal_memory_tool, recent_diary_tool)
            if tool is not None
        }
        return sum(
            1
            for call in tool_calls
            if cls._parse_turn_tool_call(call)[0]
            not in internal_names | {CAPABILITY_REQUEST_TOOL_NAME}
        )

    @staticmethod
    def _is_read_only_local_tool_name(tool_name: str | None) -> bool:
        return bool(tool_name and tool_name.startswith(_READ_ONLY_LOCAL_TOOL_PREFIXES))

    @classmethod
    def _ordinary_external_tool_call_count(
        cls,
        tool_calls: List[ToolCallObject] | List[Dict[str, Any]],
        internal_memory_tool: Any | None,
        recent_diary_tool: Any | None = None,
    ) -> int:
        internal_names = {
            tool.name
            for tool in (internal_memory_tool, recent_diary_tool)
            if tool is not None
        }
        return sum(
            1
            for call in tool_calls
            if (
                (tool_name := cls._parse_turn_tool_call(call)[0]) not in internal_names
                and tool_name != CAPABILITY_REQUEST_TOOL_NAME
                and not cls._is_read_only_local_tool_name(tool_name)
            )
        )

    @staticmethod
    def _schema_tool_name(tool: Dict[str, Any]) -> str:
        return str(tool.get("name") or tool.get("function", {}).get("name") or "")

    @staticmethod
    def _remaining_external_tool_seconds(
        *,
        external_elapsed: float,
        ordinary_external_elapsed: float,
        includes_ordinary_external: bool,
    ) -> float:
        remaining = MAX_EXTERNAL_TOOL_SECONDS - external_elapsed
        if includes_ordinary_external:
            remaining = min(
                remaining,
                MAX_ORDINARY_EXTERNAL_TOOL_SECONDS - ordinary_external_elapsed,
            )
        return max(0.0, remaining)

    @staticmethod
    def _tools_after_external_budget(
        tools: List[Dict[str, Any]],
        internal_memory_tool: Any | None,
        recent_diary_tool: Any | None = None,
    ) -> List[Dict[str, Any]]:
        internal_names = {
            tool.name
            for tool in (internal_memory_tool, recent_diary_tool)
            if tool is not None and not tool.exhausted
        }
        return [
            tool
            for tool in tools
            if BasicMemoryAgent._schema_tool_name(tool) in internal_names
        ]

    @classmethod
    def _tools_after_ordinary_budget(
        cls,
        tools: List[Dict[str, Any]],
        internal_memory_tool: Any | None,
        recent_diary_tool: Any | None = None,
    ) -> List[Dict[str, Any]]:
        internal_names = {
            tool.name
            for tool in (internal_memory_tool, recent_diary_tool)
            if tool is not None and not tool.exhausted
        }
        return [
            tool
            for tool in tools
            if (
                (tool_name := cls._schema_tool_name(tool)) in internal_names
                or cls._is_read_only_local_tool_name(tool_name)
            )
        ]

    async def _execute_turn_tools(
        self,
        tool_calls: List[ToolCallObject] | List[Dict[str, Any]],
        *,
        caller_mode: Literal["Claude", "OpenAI", "Prompt"],
        internal_memory_tool: Any | None,
        recent_diary_tool: Any | None = None,
        active_capabilities: set[str] | None = None,
        max_external_calls: int | None = None,
        max_ordinary_external_calls: int | None = None,
        max_external_seconds: float | None = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        results: list[dict[str, Any]] = []
        media_messages: list[dict[str, Any]] = []
        external_elapsed_seconds = 0.0
        external_calls: list[ToolCallObject | Dict[str, Any]] = []
        ordinary_external_calls = 0
        duplicate_navigation = []
        scheduled_navigation = set()
        cached_calls = 0
        if getattr(self, "_library_turn_evidence", None) is None:
            self._library_turn_evidence = LibraryTurnEvidence()
        for call in tool_calls:
            tool_name, tool_id, arguments, parse_error = self._parse_turn_tool_call(
                call
            )
            self._library_turn_evidence.register(tool_id, tool_name, arguments)
            key = navigation_key(tool_name, arguments) if not parse_error else None
            if key is not None:
                cached = self._library_turn_evidence.cached_navigation(tool_id)
                if cached is not None or key in scheduled_navigation:
                    duplicate_navigation.append(tool_id)
                    cached_calls += cached is not None
                    continue
                scheduled_navigation.add(key)
            current_image_error = self._current_image_reference_error(
                tool_name, arguments
            )
            if current_image_error:
                results.append(
                    self._format_turn_tool_result(
                        caller_mode,
                        tool_id,
                        current_image_error,
                        True,
                    )
                )
                continue
            if tool_name == CAPABILITY_REQUEST_TOOL_NAME:
                capability = str(arguments.get("capability") or "").strip()
                if parse_error or capability not in _CAPABILITY_TOOL_NAMES:
                    results.append(
                        self._format_turn_tool_result(
                            caller_mode,
                            tool_id,
                            "功能类别无效；只能选择 web、audio、singing、library 或 ra3。",
                            True,
                        )
                    )
                    continue
                if active_capabilities is not None:
                    active_capabilities.add(capability)
                ttl = getattr(self, "_capability_ttl", None)
                if isinstance(ttl, dict):
                    ttl[capability] = 3
                logger.info(
                    f"[工具按需] 模型请求加载能力={capability}（{_CAPABILITY_LABELS[capability]}）"
                )
                results.append(
                    self._format_turn_tool_result(
                        caller_mode,
                        tool_id,
                        f"已加载{_CAPABILITY_LABELS[capability]}的详细工具。"
                        "请在下一步直接调用完成当前请求所需的实际工具。",
                        False,
                    )
                )
                continue
            is_long_term_tool = bool(
                internal_memory_tool is not None
                and tool_name == internal_memory_tool.name
            )
            is_recent_diary_tool = bool(
                recent_diary_tool is not None and tool_name == recent_diary_tool.name
            )
            if not is_long_term_tool and not is_recent_diary_tool:
                if (
                    max_external_calls is not None
                    and len(external_calls) >= max_external_calls
                ):
                    self._library_turn_evidence.observe_status(
                        {"tool_id": tool_id, "status": "error"}
                    )
                    results.append(
                        self._format_turn_tool_result(
                            caller_mode,
                            tool_id,
                            "本轮普通外部工具调用预算已用完。请基于已有信息直接回复，"
                            "不要继续调用其他外部工具。",
                            True,
                        )
                    )
                    continue
                if (
                    not self._is_read_only_local_tool_name(tool_name)
                    and max_ordinary_external_calls is not None
                    and ordinary_external_calls >= max_ordinary_external_calls
                ):
                    results.append(
                        self._format_turn_tool_result(
                            caller_mode,
                            tool_id,
                            "本轮联网或其他普通外部工具调用预算已用完。"
                            "如果仍需核验本地文件，可继续使用资料库或电脑只读工具。",
                            True,
                        )
                    )
                    continue
                external_calls.append(call)
                if not self._is_read_only_local_tool_name(tool_name):
                    ordinary_external_calls += 1
                continue
            if parse_error:
                results.append(
                    self._format_turn_tool_result(
                        caller_mode,
                        tool_id,
                        f"内部记忆工具调用失败：{parse_error}",
                        True,
                    )
                )
                continue
            if is_recent_diary_tool:
                execution = await recent_diary_tool.execute(arguments)
                results.append(
                    self._format_turn_tool_result(
                        caller_mode,
                        tool_id,
                        execution.content + "\n\n" + FINAL_REPLY_ONLY_INSTRUCTION,
                        execution.is_error,
                    )
                )
                continue

            block_reason = self._recent_memory_tool_block_reason(arguments)
            if block_reason:
                logger.info(
                    "[记忆路由] 第三层请求改由近三日日记原文处理；reason={}；query={}",
                    block_reason,
                    text_log_fields(str(arguments.get("query") or "")),
                )
                rejected = internal_memory_tool.reject_due_to_recent_context(
                    arguments,
                    reason=block_reason,
                )
                execution = (
                    await recent_diary_tool.execute(
                        {"query": str(arguments.get("query") or "近期细节")}
                    )
                    if recent_diary_tool is not None
                    else rejected
                )
                results.append(
                    self._format_turn_tool_result(
                        caller_mode,
                        tool_id,
                        execution.content + "\n\n" + FINAL_REPLY_ONLY_INSTRUCTION,
                        execution.is_error,
                    )
                )
                continue
            execution = await internal_memory_tool.execute(arguments)
            results.append(
                self._format_turn_tool_result(
                    caller_mode,
                    tool_id,
                    execution.content + "\n\n" + FINAL_REPLY_ONLY_INSTRUCTION,
                    execution.is_error,
                )
            )

        if external_calls:
            if max_external_seconds is not None and max_external_seconds <= 0:
                for call in external_calls:
                    tool_name, tool_id, _arguments, _error = self._parse_turn_tool_call(
                        call
                    )
                    results.append(
                        self._format_turn_tool_result(
                            caller_mode,
                            tool_id,
                            f"工具 {tool_name or 'unknown'} 未执行："
                            "本轮普通外部工具总时长预算已用完。",
                            True,
                        )
                    )
            elif self._tool_executor is None:
                for call in external_calls:
                    tool_name, tool_id, _arguments, _error = self._parse_turn_tool_call(
                        call
                    )
                    results.append(
                        self._format_turn_tool_result(
                            caller_mode,
                            tool_id,
                            f"工具 {tool_name or 'unknown'} 当前不可用。",
                            True,
                        )
                    )
            else:
                iterator = self._tool_executor.execute_tools(
                    tool_calls=external_calls,
                    caller_mode=caller_mode,
                )
                external_started = asyncio.get_running_loop().time()
                try:
                    deadline = (
                        external_started + max_external_seconds
                        if max_external_seconds is not None
                        else None
                    )
                    while True:
                        if deadline is None:
                            update = await anext(iterator)
                        else:
                            remaining = deadline - asyncio.get_running_loop().time()
                            if remaining <= 0:
                                raise asyncio.TimeoutError
                            update = await asyncio.wait_for(
                                anext(iterator),
                                timeout=remaining,
                            )
                        if update.get("type") == "final_tool_results":
                            results.extend(update.get("results", []))
                            media_messages.extend(update.get("media_messages", []))
                            break
                        self._library_turn_evidence.observe_status(update)
                        yield update
                except StopAsyncIteration:
                    logger.warning(
                        "External tool executor finished without final results marker."
                    )
                    for call in external_calls:
                        name, missing_id, _args, _error = self._parse_turn_tool_call(
                            call
                        )
                        results.append(
                            self._format_turn_tool_result(
                                caller_mode,
                                missing_id,
                                f"Error: {name} ended without a completed result.",
                                True,
                            )
                        )
                except asyncio.TimeoutError:
                    logger.error(
                        "Ordinary external tools exceeded the per-turn time budget."
                    )
                    for call in external_calls:
                        tool_name, tool_id, _arguments, _error = (
                            self._parse_turn_tool_call(call)
                        )
                        results.append(
                            self._format_turn_tool_result(
                                caller_mode,
                                tool_id,
                                f"工具 {tool_name or 'unknown'} 执行超时："
                                "本轮普通外部工具总时长预算已用完。",
                                True,
                            )
                        )
                finally:
                    external_elapsed_seconds = (
                        asyncio.get_running_loop().time() - external_started
                    )
                    await _close_llm_stream(iterator)
        # Cache completed navigation only, never an in-flight search or a read.
        self._library_turn_evidence.observe_results(results)
        for duplicate_id in duplicate_navigation:
            cached = self._library_turn_evidence.cached_navigation(duplicate_id)
            if cached is None:
                cached = ("工具未完成，不能把等待中的查询当成结果。", True)
            content, is_error = cached
            results.append(
                self._format_turn_tool_result(
                    caller_mode, duplicate_id, content, is_error
                )
            )
            tool_name = self._library_turn_evidence.calls[duplicate_id][0]
            update = {
                "type": "tool_call_status",
                "tool_id": duplicate_id,
                "tool_name": tool_name,
                "status": "error" if is_error else "completed",
                "content": "",
                "reused_result": True,
                "timestamp": datetime.now().astimezone().isoformat(),
            }
            self._library_turn_evidence.observe_status(update)
            logger.info(
                "Reused completed library navigation result: tool={}", tool_name
            )
            yield update
        if cached_calls and cached_calls == len(tool_calls):
            self._library_turn_evidence.navigation_exhausted = True
        for result in results:
            content = str(result.get("content", ""))
            if (
                result.get("is_error")
                or content.startswith(("Error", "Unexpected error"))
                or (
                    content.startswith("工具 ")
                    and any(word in content for word in ("未执行", "超时", "不可用"))
                )
            ):
                result_id = (
                    result.get("tool_call_id")
                    or result.get("tool_use_id")
                    or result.get("tool_id")
                )
                self._library_turn_evidence.observe_status(
                    {"tool_id": result_id, "status": "error"}
                )
        self._library_turn_evidence.observe_results(results)
        yield {
            "type": "final_tool_results",
            "results": results,
            "media_messages": media_messages,
            "external_elapsed_seconds": external_elapsed_seconds,
        }

    async def _claude_tool_interaction_loop(
        self,
        initial_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        internal_memory_tool: Any | None = None,
        recent_diary_tool: Any | None = None,
        active_capabilities: set[str] | None = None,
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Handle Claude interaction loop with tool support."""
        messages = initial_messages.copy()
        current_turn_text = ""
        pending_tool_calls = []
        current_assistant_message_content = []
        external_tool_rounds = 0
        external_tool_calls = 0
        external_tool_seconds = 0.0
        ordinary_external_tool_rounds = 0
        ordinary_external_tool_calls = 0
        ordinary_external_tool_seconds = 0.0

        while True:
            external_budget_exhausted = (
                external_tool_rounds >= MAX_EXTERNAL_TOOL_ROUNDS
                or external_tool_calls >= MAX_EXTERNAL_TOOL_CALLS
                or external_tool_seconds >= MAX_EXTERNAL_TOOL_SECONDS
            )
            ordinary_external_budget_exhausted = (
                ordinary_external_tool_rounds >= MAX_ORDINARY_EXTERNAL_TOOL_ROUNDS
                or ordinary_external_tool_calls >= MAX_ORDINARY_EXTERNAL_TOOL_CALLS
                or ordinary_external_tool_seconds >= MAX_ORDINARY_EXTERNAL_TOOL_SECONDS
            )
            active_tools = self._library_tools_for_turn(
                self._tools_for_capabilities(tools, active_capabilities)
            )
            current_system_prompt = self._current_request_system_prompt()
            if recent_diary_tool is not None:
                recent_instruction = (
                    recent_diary_tool.followup_instructions
                    if recent_diary_tool.called
                    else recent_diary_tool.decision_instructions
                )
                if recent_instruction:
                    current_system_prompt = (
                        f"{current_system_prompt}\n\n{recent_instruction}"
                    )
            if internal_memory_tool is not None:
                memory_instruction = (
                    internal_memory_tool.followup_instructions
                    if internal_memory_tool.called
                    else internal_memory_tool.decision_instructions
                )
                if memory_instruction:
                    current_system_prompt = (
                        f"{current_system_prompt}\n\n{memory_instruction}"
                    )
            current_system_prompt = self._append_rinne_library_prompt(
                current_system_prompt,
                active_tools,
            )
            if internal_memory_tool is not None and internal_memory_tool.exhausted:
                active_tools = [
                    tool
                    for tool in active_tools
                    if tool.get("name") != internal_memory_tool.name
                ]
            elif internal_memory_tool is not None and internal_memory_tool.called:
                active_tools = [
                    (
                        internal_memory_tool.active_claude_schema
                        if tool.get("name") == internal_memory_tool.name
                        else tool
                    )
                    for tool in active_tools
                ]
            if recent_diary_tool is not None and recent_diary_tool.exhausted:
                active_tools = [
                    tool
                    for tool in active_tools
                    if tool.get("name") != recent_diary_tool.name
                ]
            if external_budget_exhausted:
                active_tools = self._tools_after_external_budget(
                    active_tools,
                    internal_memory_tool,
                    recent_diary_tool,
                )
                current_system_prompt = (
                    f"{current_system_prompt}\n\n"
                    f"{EXTERNAL_TOOL_BUDGET_EXHAUSTED_INSTRUCTION}"
                )
            elif ordinary_external_budget_exhausted:
                active_tools = self._tools_after_ordinary_budget(
                    active_tools,
                    internal_memory_tool,
                    recent_diary_tool,
                )
                current_system_prompt = (
                    f"{current_system_prompt}\n\n"
                    f"{ORDINARY_EXTERNAL_TOOL_BUDGET_EXHAUSTED_INSTRUCTION}"
                )
            force_tool = getattr(self._llm, "force_tool_once", None)
            if (
                callable(force_tool)
                and internal_memory_tool is not None
                and not internal_memory_tool.exhausted
                and getattr(internal_memory_tool, "force_first_call", False)
                and active_tools
                and any(
                    tool.get("name") == internal_memory_tool.name
                    for tool in active_tools
                )
            ):
                force_tool(internal_memory_tool.name)
            current_system_prompt = self._finalize_request_system_prompt(
                current_system_prompt
            )
            stream = self._llm.chat_completion(
                messages,
                current_system_prompt,
                tools=active_tools,
            )
            pending_tool_calls.clear()
            current_assistant_message_content.clear()

            try:
                async for event in stream:
                    if event["type"] == "text_delta":
                        text = event["text"]
                        current_turn_text += text
                        if (
                            not current_assistant_message_content
                            or current_assistant_message_content[-1]["type"] != "text"
                        ):
                            current_assistant_message_content.append(
                                {"type": "text", "text": text}
                            )
                        else:
                            current_assistant_message_content[-1]["text"] += text
                    elif event["type"] == "thinking_complete":
                        # Keep provider-signed blocks only in this tool exchange.
                        # They are never added to visible text or stored memory.
                        current_assistant_message_content.append(dict(event["data"]))
                    elif event["type"] == "tool_use_complete":
                        tool_call_data = event["data"]
                        logger.info(
                            f"Tool request: {tool_call_data['name']} (ID: {tool_call_data['id']})"
                        )
                        pending_tool_calls.append(tool_call_data)
                        current_assistant_message_content.append(
                            {
                                "type": "tool_use",
                                "id": tool_call_data["id"],
                                "name": tool_call_data["name"],
                                "input": tool_call_data["input"],
                            }
                        )
                    # elif event["type"] == "message_delta":
                    #     if event["data"]["delta"].get("stop_reason"):
                    #         stop_reason = event["data"]["delta"].get("stop_reason")
                    elif event["type"] == "message_stop":
                        break
                    elif event["type"] == "error":
                        logger.error(
                            "LLM API error event: {}", mapping_log_fields(event)
                        )
                        yield f"[Error from LLM: {event['message']}]"
                        return
            finally:
                await _close_llm_stream(stream)

            if pending_tool_calls:
                external_requested = self._external_tool_call_count(
                    pending_tool_calls,
                    internal_memory_tool,
                    recent_diary_tool,
                )
                ordinary_external_requested = self._ordinary_external_tool_call_count(
                    pending_tool_calls,
                    internal_memory_tool,
                    recent_diary_tool,
                )
                remaining_external_calls = max(
                    0,
                    MAX_EXTERNAL_TOOL_CALLS - external_tool_calls,
                )
                remaining_ordinary_external_calls = max(
                    0,
                    MAX_ORDINARY_EXTERNAL_TOOL_CALLS - ordinary_external_tool_calls,
                )
                filtered_assistant_content = [
                    block
                    for block in current_assistant_message_content
                    if not (
                        block.get("type") == "text"
                        and not block.get("text", "").strip()
                    )
                ]

                if filtered_assistant_content:
                    messages.append(
                        {"role": "assistant", "content": filtered_assistant_content}
                    )
                    if current_turn_text.strip():
                        logger.debug(
                            "Suppressed assistant text from a Claude tool-calling turn."
                        )

                tool_results_for_llm = []
                batch_external_seconds = 0.0
                tool_executor_iterator = self._execute_turn_tools(
                    tool_calls=pending_tool_calls,
                    caller_mode="Claude",
                    internal_memory_tool=internal_memory_tool,
                    recent_diary_tool=recent_diary_tool,
                    active_capabilities=active_capabilities,
                    max_external_calls=remaining_external_calls,
                    max_ordinary_external_calls=remaining_ordinary_external_calls,
                    max_external_seconds=self._remaining_external_tool_seconds(
                        external_elapsed=external_tool_seconds,
                        ordinary_external_elapsed=ordinary_external_tool_seconds,
                        includes_ordinary_external=bool(ordinary_external_requested),
                    ),
                )
                try:
                    while True:
                        update = await anext(tool_executor_iterator)
                        if update.get("type") == "final_tool_results":
                            tool_results_for_llm = update.get("results", [])
                            batch_external_seconds = float(
                                update.get("external_elapsed_seconds", 0.0)
                            )
                            break
                        else:
                            yield update
                except StopAsyncIteration:
                    logger.warning(
                        "Tool executor finished without final results marker."
                    )

                if tool_results_for_llm:
                    messages.append({"role": "user", "content": tool_results_for_llm})

                if external_requested:
                    external_tool_rounds += 1
                    external_tool_calls += min(
                        external_requested,
                        remaining_external_calls,
                    )
                    external_tool_seconds += batch_external_seconds
                if ordinary_external_requested:
                    ordinary_external_tool_rounds += 1
                    ordinary_external_tool_calls += min(
                        ordinary_external_requested,
                        remaining_ordinary_external_calls,
                    )
                    ordinary_external_tool_seconds += batch_external_seconds

                # stop_reason = None
                continue
            else:
                scene_metadata = getattr(self, "_active_scene_metadata", None)
                scene_card_requested = (
                    isinstance(scene_metadata, dict)
                    and bool(scene_metadata.get("scene_card_instruction"))
                    and not scene_metadata.get("proactive_speak")
                )
                if scene_card_requested:
                    current_turn_text, scene_card, card_status = extract_scene_card(current_turn_text)
                    scene_metadata["_scene_card_new"] = scene_card
                    scene_metadata["_scene_card_status"] = card_status
                current_turn_text = _strip_leading_internal_analysis(current_turn_text)
                current_turn_text = _strip_answer_number_prefix(current_turn_text)
                current_turn_text = _strip_provider_control_tokens(current_turn_text)
                current_turn_text = _strip_message_time_context_echo(current_turn_text)
                current_turn_text = await self._rewrite_repetitive_final_response(
                    current_turn_text,
                    messages,
                    current_system_prompt,
                )
                if scene_card_requested:
                    current_turn_text, _, _ = extract_scene_card(current_turn_text)
                if current_turn_text.strip():
                    yield current_turn_text
                    self._add_message(current_turn_text, "assistant")
                else:
                    fallback_text = self._build_missing_final_reply_message()
                    logger.warning(
                        "Claude completion returned no assistant text. "
                        "Returning a user-facing fallback message."
                    )
                    yield fallback_text
                    self._add_message(fallback_text, "assistant")
                return

    async def _openai_tool_interaction_loop(
        self,
        initial_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        internal_memory_tool: Any | None = None,
        recent_diary_tool: Any | None = None,
        active_capabilities: set[str] | None = None,
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Handle OpenAI interaction with tool support."""
        messages = initial_messages.copy()
        current_turn_text = ""
        pending_tool_calls: Union[List[ToolCallObject], List[Dict[str, Any]]] = []
        current_system_prompt = self._current_request_system_prompt()
        external_tool_rounds = 0
        external_tool_calls = 0
        external_tool_seconds = 0.0
        ordinary_external_tool_rounds = 0
        ordinary_external_tool_calls = 0
        ordinary_external_tool_seconds = 0.0
        provider_markup_retries = 0

        while True:
            external_budget_exhausted = (
                external_tool_rounds >= MAX_EXTERNAL_TOOL_ROUNDS
                or external_tool_calls >= MAX_EXTERNAL_TOOL_CALLS
                or external_tool_seconds >= MAX_EXTERNAL_TOOL_SECONDS
            )
            ordinary_external_budget_exhausted = (
                ordinary_external_tool_rounds >= MAX_ORDINARY_EXTERNAL_TOOL_ROUNDS
                or ordinary_external_tool_calls >= MAX_ORDINARY_EXTERNAL_TOOL_CALLS
                or ordinary_external_tool_seconds >= MAX_ORDINARY_EXTERNAL_TOOL_SECONDS
            )
            if self.prompt_mode_flag:
                prompt_parts = [self._current_request_system_prompt()]
                if recent_diary_tool is not None:
                    recent_instruction = (
                        recent_diary_tool.followup_instructions
                        if recent_diary_tool.called
                        else recent_diary_tool.decision_instructions
                    )
                    if recent_instruction:
                        prompt_parts.append(recent_instruction)
                if internal_memory_tool is not None:
                    memory_instruction = (
                        internal_memory_tool.followup_instructions
                        if internal_memory_tool.called
                        else internal_memory_tool.decision_instructions
                    )
                    if memory_instruction:
                        prompt_parts.append(memory_instruction)
                if self._mcp_prompt_string and not external_budget_exhausted:
                    evidence = getattr(self, "_library_turn_evidence", None)
                    if evidence is not None and evidence.navigation_exhausted:
                        remaining_tools = self._library_tools_for_turn(
                            self._tools_for_capabilities(tools, active_capabilities)
                        )
                        prompt_parts.append(
                            "本轮可用工具："
                            + json.dumps(remaining_tools, ensure_ascii=False)
                        )
                    else:
                        prompt_parts.append(self._mcp_prompt_string)
                elif self._use_mcpp and not external_budget_exhausted:
                    logger.warning("Prompt mode active but mcp_prompt_string is empty!")
                library_prompt = (
                    "" if external_budget_exhausted else self._rinne_library_prompt()
                )
                if library_prompt:
                    prompt_parts.append(library_prompt)
                if (
                    internal_memory_tool is not None
                    and not internal_memory_tool.exhausted
                ):
                    prompt_parts.append(internal_memory_tool.prompt_instructions)
                if recent_diary_tool is not None and not recent_diary_tool.exhausted:
                    prompt_parts.append(recent_diary_tool.prompt_instructions)
                if external_budget_exhausted:
                    prompt_parts.append(EXTERNAL_TOOL_BUDGET_EXHAUSTED_INSTRUCTION)
                elif ordinary_external_budget_exhausted:
                    prompt_parts.append(
                        ORDINARY_EXTERNAL_TOOL_BUDGET_EXHAUSTED_INSTRUCTION
                    )
                current_system_prompt = "\n\n".join(prompt_parts)
                tools_for_api = None
            else:
                current_system_prompt = self._current_request_system_prompt()
                if recent_diary_tool is not None:
                    recent_instruction = (
                        recent_diary_tool.followup_instructions
                        if recent_diary_tool.called
                        else recent_diary_tool.decision_instructions
                    )
                    if recent_instruction:
                        current_system_prompt = (
                            f"{current_system_prompt}\n\n{recent_instruction}"
                        )
                if internal_memory_tool is not None:
                    memory_instruction = (
                        internal_memory_tool.followup_instructions
                        if internal_memory_tool.called
                        else internal_memory_tool.decision_instructions
                    )
                    if memory_instruction:
                        current_system_prompt = (
                            f"{current_system_prompt}\n\n{memory_instruction}"
                        )
                tools_for_api = self._library_tools_for_turn(
                    self._tools_for_capabilities(tools, active_capabilities)
                )
                current_system_prompt = self._append_rinne_library_prompt(
                    current_system_prompt,
                    tools_for_api,
                )
                if internal_memory_tool is not None and internal_memory_tool.exhausted:
                    tools_for_api = [
                        tool
                        for tool in tools_for_api
                        if tool.get("function", {}).get("name")
                        != internal_memory_tool.name
                    ]
                elif internal_memory_tool is not None and internal_memory_tool.called:
                    tools_for_api = [
                        (
                            internal_memory_tool.active_openai_schema
                            if tool.get("function", {}).get("name")
                            == internal_memory_tool.name
                            else tool
                        )
                        for tool in tools_for_api
                    ]
                if recent_diary_tool is not None and recent_diary_tool.exhausted:
                    tools_for_api = [
                        tool
                        for tool in tools_for_api
                        if tool.get("function", {}).get("name")
                        != recent_diary_tool.name
                    ]
                if external_budget_exhausted:
                    tools_for_api = self._tools_after_external_budget(
                        tools_for_api,
                        internal_memory_tool,
                        recent_diary_tool,
                    )
                    current_system_prompt = (
                        f"{current_system_prompt}\n\n"
                        f"{EXTERNAL_TOOL_BUDGET_EXHAUSTED_INSTRUCTION}"
                    )
                elif ordinary_external_budget_exhausted:
                    tools_for_api = self._tools_after_ordinary_budget(
                        tools_for_api,
                        internal_memory_tool,
                        recent_diary_tool,
                    )
                    current_system_prompt = (
                        f"{current_system_prompt}\n\n"
                        f"{ORDINARY_EXTERNAL_TOOL_BUDGET_EXHAUSTED_INSTRUCTION}"
                    )
                if not tools_for_api:
                    tools_for_api = None
                force_tool = getattr(self._llm, "force_tool_once", None)
                if (
                    callable(force_tool)
                    and internal_memory_tool is not None
                    and not internal_memory_tool.exhausted
                    and getattr(internal_memory_tool, "force_first_call", False)
                    and tools_for_api
                    and any(
                        tool.get("function", {}).get("name")
                        == internal_memory_tool.name
                        for tool in tools_for_api
                    )
                ):
                    force_tool(internal_memory_tool.name)

            current_system_prompt = self._finalize_request_system_prompt(
                current_system_prompt
            )
            stream = self._llm.chat_completion(
                messages, current_system_prompt, tools=tools_for_api
            )
            pending_tool_calls.clear()
            current_turn_text = ""
            assistant_message_for_api = None
            detected_prompt_json = None
            goto_next_while_iteration = False

            try:
                async for event in stream:
                    if self.prompt_mode_flag:
                        if isinstance(event, str):
                            current_turn_text += event
                            if self._json_detector:
                                potential_json = self._json_detector.process_chunk(
                                    event
                                )
                                if potential_json:
                                    try:
                                        if isinstance(potential_json, list):
                                            detected_prompt_json = potential_json
                                        elif isinstance(potential_json, dict):
                                            detected_prompt_json = [potential_json]

                                        if detected_prompt_json:
                                            break
                                    except Exception as e:
                                        logger.error(
                                            f"Error parsing detected JSON: {e}"
                                        )
                                        if self._json_detector:
                                            self._json_detector.reset()
                                        yield f"[Error parsing tool JSON: {e}]"
                                        goto_next_while_iteration = True
                                        break
                    else:
                        if isinstance(event, str):
                            current_turn_text += event
                        elif isinstance(event, list) and all(
                            isinstance(tc, ToolCallObject) for tc in event
                        ):
                            pending_tool_calls = event
                            assistant_message_for_api = {
                                "role": "assistant",
                                "content": (
                                    current_turn_text if current_turn_text else None
                                ),
                                "tool_calls": [
                                    {
                                        "id": tc.id,
                                        "type": tc.type,
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments,
                                        },
                                    }
                                    for tc in pending_tool_calls
                                ],
                            }
                            break
                        elif event == "__API_NOT_SUPPORT_TOOLS__":
                            logger.warning(
                                f"LLM {getattr(self._llm, 'model', '')} has no native tool support. Switching to prompt mode."
                            )
                            self.prompt_mode_flag = True
                            if self._tool_manager:
                                self._tool_manager.disable()
                            if self._json_detector:
                                self._json_detector.reset()
                            goto_next_while_iteration = True
                            break
            finally:
                await _close_llm_stream(stream)
            if goto_next_while_iteration:
                continue

            if not self.prompt_mode_flag and not pending_tool_calls:
                recovered_calls, provider_markup_detected = (
                    self._parse_provider_text_tool_calls(
                        current_turn_text,
                        tools_for_api,
                    )
                )
                if provider_markup_detected:
                    current_turn_text = ""
                    if recovered_calls:
                        pending_tool_calls = recovered_calls
                        assistant_message_for_api = {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": call.id,
                                    "type": call.type,
                                    "function": {
                                        "name": call.function.name,
                                        "arguments": call.function.arguments,
                                    },
                                }
                                for call in recovered_calls
                            ],
                        }
                        logger.warning(
                            "Recovered provider tool markup as a native tool call."
                        )
                    elif provider_markup_retries < 1:
                        provider_markup_retries += 1
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "【系统：上一条响应包含无效或残缺的工具控制标记，"
                                    "已被隐藏。只能使用接口提供的原生工具调用格式；"
                                    "请重新完成本轮请求，不要输出任何工具标签。】"
                                ),
                            }
                        )
                        continue

            if detected_prompt_json:
                logger.info("Processing tools detected via prompt mode JSON.")
                if current_turn_text.strip():
                    logger.debug(
                        "Suppressed assistant text from a prompt-mode tool-calling turn."
                    )

                parsed_tools = self._parse_prompt_tool_calls(detected_prompt_json)
                tool_results_for_llm = []
                batch_external_seconds = 0.0
                if parsed_tools:
                    external_requested = self._external_tool_call_count(
                        parsed_tools,
                        internal_memory_tool,
                        recent_diary_tool,
                    )
                    ordinary_external_requested = (
                        self._ordinary_external_tool_call_count(
                            parsed_tools,
                            internal_memory_tool,
                            recent_diary_tool,
                        )
                    )
                    remaining_external_calls = max(
                        0,
                        MAX_EXTERNAL_TOOL_CALLS - external_tool_calls,
                    )
                    remaining_ordinary_external_calls = max(
                        0,
                        MAX_ORDINARY_EXTERNAL_TOOL_CALLS - ordinary_external_tool_calls,
                    )
                    tool_executor_iterator = self._execute_turn_tools(
                        tool_calls=parsed_tools,
                        caller_mode="Prompt",
                        internal_memory_tool=internal_memory_tool,
                        recent_diary_tool=recent_diary_tool,
                        active_capabilities=active_capabilities,
                        max_external_calls=remaining_external_calls,
                        max_ordinary_external_calls=(remaining_ordinary_external_calls),
                        max_external_seconds=self._remaining_external_tool_seconds(
                            external_elapsed=external_tool_seconds,
                            ordinary_external_elapsed=(ordinary_external_tool_seconds),
                            includes_ordinary_external=bool(
                                ordinary_external_requested
                            ),
                        ),
                    )
                    try:
                        while True:
                            update = await anext(tool_executor_iterator)
                            if update.get("type") == "final_tool_results":
                                tool_results_for_llm = update.get("results", [])
                                batch_external_seconds = float(
                                    update.get("external_elapsed_seconds", 0.0)
                                )
                                break
                            else:
                                yield update
                    except StopAsyncIteration:
                        logger.warning(
                            "Prompt mode tool executor finished without final results marker."
                        )

                if tool_results_for_llm:
                    result_strings = [
                        res.get("content", "Error: Malformed result")
                        for res in tool_results_for_llm
                    ]
                    combined_results_str = "\n".join(result_strings)
                    messages.append({"role": "user", "content": combined_results_str})
                if parsed_tools and external_requested:
                    external_tool_rounds += 1
                    external_tool_calls += min(
                        external_requested,
                        remaining_external_calls,
                    )
                    external_tool_seconds += batch_external_seconds
                if parsed_tools and ordinary_external_requested:
                    ordinary_external_tool_rounds += 1
                    ordinary_external_tool_calls += min(
                        ordinary_external_requested,
                        remaining_ordinary_external_calls,
                    )
                    ordinary_external_tool_seconds += batch_external_seconds
                continue

            elif pending_tool_calls and assistant_message_for_api:
                external_requested = self._external_tool_call_count(
                    pending_tool_calls,
                    internal_memory_tool,
                    recent_diary_tool,
                )
                ordinary_external_requested = self._ordinary_external_tool_call_count(
                    pending_tool_calls,
                    internal_memory_tool,
                    recent_diary_tool,
                )
                remaining_external_calls = max(
                    0,
                    MAX_EXTERNAL_TOOL_CALLS - external_tool_calls,
                )
                remaining_ordinary_external_calls = max(
                    0,
                    MAX_ORDINARY_EXTERNAL_TOOL_CALLS - ordinary_external_tool_calls,
                )
                messages.append(assistant_message_for_api)
                if current_turn_text:
                    logger.debug(
                        "Suppressed assistant text from an OpenAI tool-calling turn."
                    )

                tool_results_for_llm = []
                media_messages_for_llm = []
                batch_external_seconds = 0.0
                tool_executor_iterator = self._execute_turn_tools(
                    tool_calls=pending_tool_calls,
                    caller_mode="OpenAI",
                    internal_memory_tool=internal_memory_tool,
                    recent_diary_tool=recent_diary_tool,
                    active_capabilities=active_capabilities,
                    max_external_calls=remaining_external_calls,
                    max_ordinary_external_calls=remaining_ordinary_external_calls,
                    max_external_seconds=self._remaining_external_tool_seconds(
                        external_elapsed=external_tool_seconds,
                        ordinary_external_elapsed=ordinary_external_tool_seconds,
                        includes_ordinary_external=bool(ordinary_external_requested),
                    ),
                )
                try:
                    while True:
                        update = await anext(tool_executor_iterator)
                        if update.get("type") == "final_tool_results":
                            tool_results_for_llm = update.get("results", [])
                            media_messages_for_llm = update.get("media_messages", [])
                            batch_external_seconds = float(
                                update.get("external_elapsed_seconds", 0.0)
                            )
                            break
                        else:
                            yield update
                except StopAsyncIteration:
                    logger.warning(
                        "OpenAI tool executor finished without final results marker."
                    )

                if tool_results_for_llm:
                    messages.extend(tool_results_for_llm)
                if media_messages_for_llm:
                    messages.extend(media_messages_for_llm)
                # The next request rebuilds the request-level system prompt
                # with ``followup_instructions``.  Do not append a second
                # message-level system entry to the OpenAI message list.
                if external_requested:
                    external_tool_rounds += 1
                    external_tool_calls += min(
                        external_requested,
                        remaining_external_calls,
                    )
                    external_tool_seconds += batch_external_seconds
                if ordinary_external_requested:
                    ordinary_external_tool_rounds += 1
                    ordinary_external_tool_calls += min(
                        ordinary_external_requested,
                        remaining_ordinary_external_calls,
                    )
                    ordinary_external_tool_seconds += batch_external_seconds
                continue

            else:
                scene_metadata = getattr(self, "_active_scene_metadata", None)
                scene_card_requested = (
                    isinstance(scene_metadata, dict)
                    and bool(scene_metadata.get("scene_card_instruction"))
                    and not scene_metadata.get("proactive_speak")
                )
                if scene_card_requested:
                    current_turn_text, scene_card, card_status = extract_scene_card(current_turn_text)
                    scene_metadata["_scene_card_new"] = scene_card
                    scene_metadata["_scene_card_status"] = card_status
                current_turn_text = _strip_leading_internal_analysis(current_turn_text)
                current_turn_text = _strip_answer_number_prefix(current_turn_text)
                current_turn_text = _strip_provider_control_tokens(current_turn_text)
                current_turn_text = _strip_message_time_context_echo(current_turn_text)
                current_turn_text = await self._rewrite_repetitive_final_response(
                    current_turn_text,
                    messages,
                    current_system_prompt,
                )
                if scene_card_requested:
                    current_turn_text, _, _ = extract_scene_card(current_turn_text)
                if current_turn_text.strip():
                    yield current_turn_text
                    self._add_message(current_turn_text, "assistant")
                else:
                    fallback_text = self._build_missing_final_reply_message()
                    logger.warning(
                        "OpenAI completion returned no assistant text. "
                        "Returning a user-facing fallback message."
                    )
                    yield fallback_text
                    self._add_message(fallback_text, "assistant")
                return

    async def _visible_scene_stream(
        self,
        stream: AsyncIterator[Union[str, Dict[str, Any]]],
        metadata: dict[str, Any] | None,
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Remove a hidden scene envelope while preserving text streaming.

        The protocol requires the envelope at the beginning of a changed
        response, but the opening/closing markers can arrive in separate
        provider chunks.  Keep only a small marker-sized tail buffered until
        the stream proves that the text is ordinary visible prose.
        """

        pending = ""
        state = "visible-search"
        marker_tail = max(len(SCENE_ENVELOPE_OPEN), len(SCENE_ENVELOPE_CLOSE)) - 1

        async for item in stream:
            if not isinstance(item, str):
                yield item
                continue
            pending += item
            while pending:
                if state == "hidden":
                    end = pending.find(SCENE_ENVELOPE_CLOSE)
                    if end < 0:
                        # Do not leak an incomplete machine block.  Providers
                        # normally finish this within a few chunks.
                        break
                    body = pending[:end]
                    delta = parse_scene_envelope(
                        SCENE_ENVELOPE_OPEN + body + SCENE_ENVELOPE_CLOSE
                    )
                    if delta is not None and isinstance(metadata, dict):
                        deltas = metadata.setdefault("_scene_deltas", [])
                        if isinstance(deltas, list) and delta not in deltas:
                            deltas.append(delta)
                    elif body.strip():
                        logger.warning("Ignoring invalid hidden scene envelope")
                    pending = pending[end + len(SCENE_ENVELOPE_CLOSE) :]
                    state = "visible-search"
                    continue

                start = pending.find(SCENE_ENVELOPE_OPEN)
                if start >= 0:
                    if start:
                        yield pending[:start]
                    pending = pending[start + len(SCENE_ENVELOPE_OPEN) :]
                    state = "hidden"
                    continue

                # No opening marker yet.  Retain enough trailing characters to
                # recognize a marker split across chunks, and emit the rest.
                if len(pending) > marker_tail:
                    yield pending[:-marker_tail]
                    pending = pending[-marker_tail:]
                break

        if state == "visible-search" and pending:
            yield pending
        elif state == "hidden":
            logger.warning("Discarding incomplete hidden scene envelope")

    def _chat_function_factory(
        self,
    ) -> Callable[[BatchInput], AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]]:
        """Create the chat pipeline function."""

        @tts_filter(self._tts_preprocessor_config)
        @display_processor()
        @actions_extractor(self._live2d_model)
        @sentence_divider(
            faster_first_response=self._faster_first_response,
            segment_method=self._segment_method,
            valid_tags=["think"],
        )
        async def chat_with_memory(
            input_data: BatchInput,
        ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
            """Process chat with memory and tools."""
            self.reset_interrupt()
            self.prompt_mode_flag = False
            self._active_scene_metadata = input_data.metadata

            messages = self._to_messages(input_data)
            current_user_text = self._to_text_prompt(input_data)
            is_proactive_speak = bool(
                (input_data.metadata or {}).get("proactive_speak")
            )
            set_media_focus = getattr(self._tool_executor, "set_media_focus", None)
            if callable(set_media_focus):
                set_media_focus(
                    None
                    if (input_data.metadata or {}).get("proactive_speak")
                    else current_user_text
                )
            if is_proactive_speak:
                active_capabilities, capability_reasons = (
                    set(),
                    ["主动观察只使用当前屏幕证据"],
                )
            else:
                active_capabilities, capability_reasons = (
                    self._select_turn_capabilities(current_user_text)
                )
            internal_memory_tool = None
            recent_diary_tool = None
            if input_data.metadata:
                candidate_tool = input_data.metadata.get("long_term_memory_tool")
                if all(
                    hasattr(candidate_tool, attribute)
                    for attribute in (
                        "name",
                        "openai_schema",
                        "claude_schema",
                        "execute",
                    )
                ):
                    internal_memory_tool = candidate_tool
                candidate_recent_tool = input_data.metadata.get(
                    "recent_diary_recall_tool"
                )
                if all(
                    hasattr(candidate_recent_tool, attribute)
                    for attribute in (
                        "name",
                        "openai_schema",
                        "claude_schema",
                        "execute",
                    )
                ):
                    recent_diary_tool = candidate_recent_tool

            # “是否必须联网”与“拿什么词搜索”分开处理：
            # 关键词规则只负责强制触发，查询改写器负责提炼搜索主题。
            if self._tool_executor and self._use_mcpp:
                if should_force_search(current_user_text):
                    search_result_message = await self._run_forced_search(
                        current_user_text
                    )
                    if search_result_message:
                        messages.append(search_result_message)

            tools = None
            tool_mode = None
            llm_supports_native_tools = False

            if not is_proactive_speak and (
                (self._use_mcpp and self._tool_manager)
                or internal_memory_tool
                or recent_diary_tool
            ):
                tools = None
                if isinstance(self._llm, ClaudeAsyncLLM):
                    tool_mode = "Claude"
                    tools = list(self._formatted_tools_claude)
                    if self._use_mcpp:
                        tools.append(_capability_request_schema("Claude"))
                    if internal_memory_tool is not None:
                        tools.append(internal_memory_tool.claude_schema)
                    if recent_diary_tool is not None:
                        tools.append(recent_diary_tool.claude_schema)
                    llm_supports_native_tools = True
                elif isinstance(self._llm, OpenAICompatibleAsyncLLM):
                    tool_mode = "OpenAI"
                    tools = list(self._formatted_tools_openai)
                    if self._use_mcpp:
                        tools.append(_capability_request_schema("OpenAI"))
                    if internal_memory_tool is not None:
                        tools.append(internal_memory_tool.openai_schema)
                    if recent_diary_tool is not None:
                        tools.append(recent_diary_tool.openai_schema)
                    llm_supports_native_tools = True
                else:
                    logger.warning(
                        f"LLM type {type(self._llm)} not explicitly handled for tool mode determination."
                    )

                if llm_supports_native_tools and not tools:
                    logger.warning(
                        f"No tools available/formatted for '{tool_mode}' mode, despite MCP being enabled."
                    )
                if llm_supports_native_tools:
                    active_tool_names = [
                        self._schema_tool_name(tool)
                        for tool in self._tools_for_capabilities(
                            tools or [], active_capabilities
                        )
                    ]
                    logger.info(
                        "[工具按需] 本轮详细工具="
                        f"{active_tool_names or ['无']}；原因="
                        f"{capability_reasons or ['普通对话仅保留功能入口与长期记忆']}"
                    )

            if tool_mode == "Claude":
                logger.debug(
                    f"Starting Claude tool interaction loop with {len(tools)} tools."
                )
                async for output in _strip_message_time_context_stream(
                    self._visible_scene_stream(
                        self._claude_tool_interaction_loop(
                            messages,
                            tools if tools else [],
                            internal_memory_tool=internal_memory_tool,
                            recent_diary_tool=recent_diary_tool,
                            active_capabilities=active_capabilities,
                        ),
                        input_data.metadata,
                    ),
                ):
                    yield output
                return
            elif tool_mode == "OpenAI":
                logger.debug(
                    f"Starting OpenAI tool interaction loop with {len(tools)} tools."
                )
                async for output in _strip_message_time_context_stream(
                    self._visible_scene_stream(
                        self._openai_tool_interaction_loop(
                            messages,
                            tools if tools else [],
                            internal_memory_tool=internal_memory_tool,
                            recent_diary_tool=recent_diary_tool,
                            active_capabilities=active_capabilities,
                        ),
                        input_data.metadata,
                    ),
                ):
                    yield output
                return
            else:
                logger.info("Starting simple chat completion.")
                if (input_data.metadata or {}).get("scene_card_instruction"):
                    logger.warning("[场景卡片] 简单路径不提取卡片，仅使用现有输出过滤")

                async def _simple_text_stream() -> AsyncIterator[str]:
                    token_stream = self._llm.chat_completion(
                        messages,
                        self._finalize_request_system_prompt(
                            self._current_request_system_prompt()
                        ),
                    )
                    async for event in token_stream:
                        text_chunk = ""
                        if (
                            isinstance(event, dict)
                            and event.get("type") == "text_delta"
                        ):
                            text_chunk = event.get("text", "")
                        elif isinstance(event, str):
                            text_chunk = event
                        if text_chunk:
                            yield text_chunk

                complete_response = ""
                async for text_chunk in _strip_message_time_context_stream(
                    self._visible_scene_stream(
                        _simple_text_stream(),
                        input_data.metadata,
                    ),
                ):
                    if isinstance(text_chunk, str):
                        complete_response += text_chunk
                complete_response = await self._rewrite_repetitive_final_response(
                    complete_response,
                    messages,
                    self._finalize_request_system_prompt(self._current_request_system_prompt()),
                )
                if complete_response.strip():
                    yield complete_response
                    self._add_message(complete_response, "assistant")
                else:
                    fallback_text = self._build_missing_final_reply_message()
                    logger.warning(
                        "Simple chat completion returned no assistant text. "
                        "Returning a user-facing fallback message."
                    )
                    yield fallback_text
                    self._add_message(fallback_text, "assistant")

        return chat_with_memory

    async def chat(
        self,
        input_data: BatchInput,
    ) -> AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]:
        """Run chat pipeline."""
        chat_func_decorated = self._chat_function_factory()
        async for output in chat_func_decorated(input_data):
            yield output

    def reset_interrupt(self) -> None:
        """Reset interrupt flag."""
        self._interrupt_handled = False

    def start_group_conversation(
        self, human_name: str, ai_participants: List[str]
    ) -> None:
        """Start a group conversation."""
        if not self._tool_prompts:
            logger.warning("Tool prompts dictionary is not set.")
            return

        other_ais = ", ".join(name for name in ai_participants)
        prompt_name = self._tool_prompts.get("group_conversation_prompt", "")

        if not prompt_name:
            logger.warning("No group conversation prompt name found.")
            return

        try:
            group_context = prompt_loader.load_util(prompt_name).format(
                human_name=human_name, other_ais=other_ais
            )
            self._add_message(group_context, "user")
        except FileNotFoundError:
            logger.error(f"Group conversation prompt file not found: {prompt_name}")
        except KeyError as e:
            logger.error(f"Missing formatting key in group conversation prompt: {e}")
        except Exception as e:
            logger.error(f"Failed to load group conversation prompt: {e}")
