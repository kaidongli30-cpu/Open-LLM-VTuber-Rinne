"""One-turn grounding rules for proactive desktop screen observations."""

from __future__ import annotations

from typing import Any, Iterable


PROACTIVE_SCREEN_OBSERVATION_INSTRUCTION = """【主动观察｜仅本轮】
直接查看本轮屏幕图像，结合最近对话理解用户此刻正在做什么，再像平常陪伴用户那样自然说一两句；不要说明观察或分析过程，也不要复述整段屏幕内容。只提与当前情境直接相关且确实看清的内容，不得主动逐字朗读，看不清就不猜。如果最近对话的话题与屏幕完全无关，忽略屏幕并自然接续原话题，不要借屏幕另起话题。"""


def get_proactive_screen_observation_instruction(
    images: Iterable[dict[str, Any]] | None,
    *,
    proactive_speak: bool,
) -> str:
    """Return the final system reminder only for proactive screen captures."""

    if not proactive_speak:
        return ""
    for image in images or ():
        if isinstance(image, dict) and image.get("source") == "screen":
            return PROACTIVE_SCREEN_OBSERVATION_INSTRUCTION
    return ""


__all__ = [
    "PROACTIVE_SCREEN_OBSERVATION_INSTRUCTION",
    "get_proactive_screen_observation_instruction",
]
