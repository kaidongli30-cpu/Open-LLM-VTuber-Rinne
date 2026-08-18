"""Utilities for turning an explicit search request into a web-search query."""

from __future__ import annotations

import re


FORCED_SEARCH_KEYWORDS = ("搜索",)
MAX_SEARCH_QUERY_LENGTH = 160

_SEARCH_KEYWORD_RE = re.compile(r"搜索")
_LEADING_MODIFIER_RE = re.compile(
    r"^\s*(?:(?:一下|下|一搜|看看)\s*)?(?:(?:关于|有关)\s*)?[:：]?\s*"
)
_STRONG_SENTENCE_END_RE = re.compile(r"[。！？!?；;\r\n]")
_FOLLOW_UP_INSTRUCTION_RE = re.compile(
    r"[,，]\s*(?=(?:然后|再|并(?:且)?|顺便|之后|接着|最后|记得|"
    r"告诉|回答|总结|解释|分析|对比|比较|给我))"
)
_TRAILING_POLITENESS_RE = re.compile(
    r"(?:可以吗|行吗|好吗|好不好|谢谢|拜托了?|麻烦了?)\s*$"
)
_COMMAND_PREFIX_RE = re.compile(
    r"(?:请|麻烦|帮我|给我|替我|可以|能不能|能|去|联网|上网|网上)\s*$"
)
_MODEL_PREFIX_RE = re.compile(
    r"^(?:搜索查询|搜索词|检索词|查询词|query)\s*[:：]\s*",
    re.IGNORECASE,
)
_LIST_PREFIX_RE = re.compile(r"^(?:[-*•]|\d+[.)、])\s*")
_ERROR_RESPONSE_PREFIXES = (
    "error calling the chat endpoint",
    "[error",
)


def should_force_search(text: str) -> bool:
    """Return whether the current compatibility rule requires a web search."""
    return bool(text) and any(keyword in text for keyword in FORCED_SEARCH_KEYWORDS)


def _command_score(text: str, match: re.Match[str]) -> int:
    """Score how likely a ``搜索`` occurrence is being used as a command."""
    score = 0
    suffix = text[match.end() :]
    prefix = text[max(0, match.start() - 12) : match.start()]

    if re.match(r"\s*(?:一下|下|一搜|看看|关于|有关|[:：])", suffix):
        score += 4
    elif suffix[:1].isspace():
        score += 3

    if _COMMAND_PREFIX_RE.search(prefix):
        score += 2

    return score


def _trim_query_clause(text: str) -> str:
    """Remove response instructions that are useful to the assistant, not the engine."""
    query = _STRONG_SENTENCE_END_RE.split(text, maxsplit=1)[0]
    query = _FOLLOW_UP_INSTRUCTION_RE.split(query, maxsplit=1)[0]
    query = _TRAILING_POLITENESS_RE.sub("", query)
    query = query.strip(" \t,，。.!！?？;；:：\"'“”‘’")

    if len(query) > MAX_SEARCH_QUERY_LENGTH:
        query = query[:MAX_SEARCH_QUERY_LENGTH].rstrip()

    return query


def extract_search_query(user_text: str) -> str:
    """Deterministically extract a useful fallback query from a spoken request.

    This parser deliberately remains a fallback. It handles explicit commands such
    as ``帮我搜索一下 X`` without pretending that regular expressions can resolve
    conversational references such as ``它`` or ``刚才那部电影``.
    """
    normalized = re.sub(r"\s+", " ", user_text or "").strip()
    if not normalized:
        return ""

    matches = list(_SEARCH_KEYWORD_RE.finditer(normalized))
    if not matches:
        return _trim_query_clause(normalized)

    scored_matches = [
        (_command_score(normalized, match), match.start(), match) for match in matches
    ]
    score, _, best_match = max(scored_matches, key=lambda item: (item[0], item[1]))

    # With no command cue, keep the keyword. For example, stripping "搜索" from
    # "搜索算法是什么" would turn a good query into the unrelated "算法是什么".
    if score <= 0:
        return _trim_query_clause(normalized)

    candidate = _LEADING_MODIFIER_RE.sub("", normalized[best_match.end() :])
    candidate = _trim_query_clause(candidate)
    return candidate or _trim_query_clause(normalized)


def clean_model_search_query(
    model_text: str,
    *,
    fallback_query: str,
    original_text: str,
) -> str:
    """Validate and normalize the query-rewriter model's output."""
    cleaned = (model_text or "").strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("```text", "").replace("```", "").strip()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return fallback_query

    query = _LIST_PREFIX_RE.sub("", lines[0])
    query = _MODEL_PREFIX_RE.sub("", query)
    query = query.strip(" \t\"'“”‘’`")

    if not query or query.lower().startswith(_ERROR_RESPONSE_PREFIXES):
        return fallback_query
    if len(query) > MAX_SEARCH_QUERY_LENGTH:
        return fallback_query

    normalized_query = re.sub(r"\s+", "", query)
    normalized_original = re.sub(r"\s+", "", original_text or "")
    if (
        fallback_query
        and normalized_query == normalized_original
        and len(fallback_query) + 12 < len(original_text or "")
    ):
        return fallback_query

    return query
