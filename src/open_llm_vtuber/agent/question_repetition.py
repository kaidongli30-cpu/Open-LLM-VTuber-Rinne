"""Locate repeated information questions without an extra classifier request."""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


_HIDDEN = re.compile(
    r"\[[^\]\r\n]*\]|（[^）]*）|\([^)]*\)|“[^”]*”|‘[^’]*’|\"[^\"\r\n]*\""
)
_CLAUSE = re.compile(r"[^，,。！？!?；;\r\n]+[，,。！？!?；;\r\n]*")
_TONE_QUESTIONS = {
    "好吗",
    "好么",
    "好不好",
    "是吗",
    "是么",
    "是不是",
    "是吧",
    "对吗",
    "对吧",
    "对不对",
    "行吗",
    "行不行",
    "可以吗",
    "好吧",
    "嗯",
    "嗯哼",
    "真的吗",
    "真的么",
    "呢",
    "ok",
    "okay",
    "right",
    "isntit",
    "really",
}
_INTERROGATIVE = re.compile(
    r"什么|怎么|怎样|哪[里儿个些天种位边]?|谁|几[点时天个岁号]|多少|为何|为什么|"
    r"是否|能否|有没有|要不要|想不想|会不会|去不去|吃不吃|[吗么]$|"
    r"^(?:what|when|where|who|why|how|which)\b",
    re.IGNORECASE,
)
_DECLARATION = re.compile(r"^(?:我(?:知道|明白|记得)|无论|不管|不用|不必)")
_CHOICE_QUESTION = re.compile(r"([\u4e00-\u9fff]{1,2})不\1[呀啊呢吧]*$")
_QUESTION_OPENING = re.compile(
    r"^(?:那|所以|那么|对了|你|您|用户|今天|明天|现在|我们|什么|为什么|怎么|"
    r"怎样|哪|谁|几|多少|是否|能否|有没有|要不要|想不想|请问)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QuestionSpan:
    text: str
    normalized: str
    start: int
    end: int


@dataclass(frozen=True)
class RepeatedQuestion:
    previous: str
    current: QuestionSpan
    followup: str


def _comparison(text: str) -> str:
    text = re.sub(r"用户|darling|sonogamirinne", "", text, flags=re.IGNORECASE)
    text = "".join(re.findall(r"[a-z0-9\u4e00-\u9fff]", text.lower()))
    text = re.sub(r"^(?:那么|对了|话说|嗯|呐)+", "", text)
    text = text.replace("早饭", "早餐").replace("午饭", "午餐").replace("晚饭", "晚餐")
    text = text.replace("了没有", "了吗").replace("了没", "了吗")
    return text.rstrip("呀啊嘛")


def question_spans(text: str, *, informational_only: bool = True) -> list[QuestionSpan]:
    """Keep offsets in the original reply, excluding actions and quoted questions."""
    visible = _HIDDEN.sub(lambda match: " " * len(match.group()), text)
    spans = []
    for match in _CLAUSE.finditer(visible):
        raw = match.group()
        body = raw.rstrip("，,。！？!?；;\r\n \t")
        normalized = _comparison(body)
        if not normalized:
            continue
        explicit = "?" in raw or "？" in raw
        choice_question = _CHOICE_QUESTION.search(normalized)
        if not explicit and not (
            (_INTERROGATIVE.search(body.strip()) or choice_question)
            and not _DECLARATION.search(body.strip())
            and (
                _QUESTION_OPENING.search(body.strip())
                or re.search(r"[吗么]$", normalized)
                or choice_question
            )
        ):
            continue
        if informational_only and normalized in _TONE_QUESTIONS:
            continue
        start = match.start() + len(body) - len(body.lstrip())
        # Stop before masked trailing actions/quotes, retaining them on removal.
        end = match.start() + len(body)
        while end < match.end() and text[end] in "，,。！？!?；;":
            end += 1
        spans.append(QuestionSpan(text[start:end], normalized, start, end))
    return spans


def _question_intent(text: str) -> str | None:
    """Normalize a few common information requests, not all related topics."""
    if re.fullmatch(
        r"(?:那|那么|你|您|今天|现在|最近|还|都|一切|过得|过的)*"
        r"(?:好吗|好么|怎么样|如何|顺利吗|顺不顺利|还好吧)",
        text,
    ):
        return "general_wellbeing"
    if re.fullmatch(
        r"(?:那|你|您|今天|现在|最近|这两天|还|也|是不是|会不会|有没有|有点|一点|"
        r"很|太|这么|那么|没|不|觉得|感觉)*"
        r"(?:累不累|累|疲惫|疲劳)(?:了|吗|么|吧|呢|不|呀|啊)*",
        text,
    ):
        return "fatigue"
    return None


def _same_question(first: str, second: str) -> bool:
    if first == second:
        return True
    # A shared sentence frame must not turn a different day, meal or number
    # into the same information request.
    for details in (("今天", "昨天", "明天", "前天", "后天"), ("早餐", "午餐", "晚餐")):
        left = {word for word in details if word in first}
        right = {word for word in details if word in second}
        if left and right and left != right:
            return False
    left_numbers = re.findall(r"\d+", first)
    right_numbers = re.findall(r"\d+", second)
    if left_numbers and right_numbers and left_numbers != right_numbers:
        return False
    first_intent = _question_intent(first)
    if first_intent is not None and first_intent == _question_intent(second):
        return True
    return (
        min(len(first), len(second)) >= 3
        and SequenceMatcher(None, first, second, autojunk=False).ratio() >= 0.82
    )


def find_repeated_questions(
    draft: str, history: list[dict[str, Any]]
) -> list[RepeatedQuestion]:
    """Compare every information question with the whole active chat history."""
    earlier: list[tuple[QuestionSpan, str]] = []
    for index, message in enumerate(history):
        content = message.get("content")
        if message.get("role") != "assistant" or not isinstance(content, str):
            continue
        following = []
        for later in history[index + 1 : index + 5]:
            if later.get("role") == "user" and isinstance(later.get("content"), str):
                following.append("用户：" + later["content"][:500])
                break
        followup = "\n".join(following) or "未记录到随后回答，不能断言已经回答。"
        earlier.extend((question, followup) for question in question_spans(content))
    repeats = []
    for current in question_spans(draft):
        for previous, followup in reversed(earlier):
            if _same_question(previous.normalized, current.normalized):
                repeats.append(RepeatedQuestion(previous.text, current, followup))
                break
        earlier.append((current, "同一份待发送回复中已经出现过该问题。"))
    return repeats


def remove_question_spans(text: str, spans: list[QuestionSpan]) -> str:
    for span in sorted(spans, key=lambda item: item.start, reverse=True):
        text = text[: span.start] + text[span.end :]
    return re.sub(r"[，,：:；;]\s*$", "。", text).strip()


def has_visible_reply(text: str) -> bool:
    """Check for spoken content, not the normalized meaning of a question."""
    spoken = re.sub(r"\[[^\]\r\n]*\]|（[^）]*）|\([^)]*\)", "", text)
    # Names and short interjections remain audible replies. The comparison
    # normalizer deliberately deletes them and must not decide emptiness.
    return any(character.isalnum() for character in spoken)
