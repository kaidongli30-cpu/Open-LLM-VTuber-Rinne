"""Request-only scene continuity guidance across conversation channels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..memory.today_history_loader import TodayHistoryLoader


SCENE_CONTINUITY_SYSTEM_INSTRUCTION = """【场景连续性提醒】
请先仔细阅读最近几轮对话，确认用户和凛祢最后已知的位置，尤其确认两人是否同行、正在做什么以及移动方向，并在此基础上自然延续当前场景。不要自行默认凛祢在家、用户从外面回来，也不要自行改变人物位置、同行关系或正在进行的行动。只有上文明确支持时，才能描述迎接、会合、同行、到家或分开；证据不足时不要补写位置和动作，直接回应用户本轮所说的话。"""

_QQ_SOURCE_CHANNELS = frozenset({"qq", "qq_private"})


@dataclass(frozen=True, slots=True)
class SceneContinuityDecision:
    """Whether this turn needs the final scene-continuity reminder."""

    instruction: str = ""
    reason: str = "not_needed"

    @property
    def enabled(self) -> bool:
        return bool(self.instruction)


def decide_scene_continuity_instruction(
    history_root: str | Path,
    *,
    source_channel: str,
    reference_time: datetime,
) -> SceneContinuityDecision:
    """Enable the reminder for every QQ turn or the first desktop turn after QQ."""

    normalized_source = source_channel.strip().casefold() or "desktop"
    if normalized_source in _QQ_SOURCE_CHANNELS:
        return SceneContinuityDecision(
            instruction=SCENE_CONTINUITY_SYSTEM_INSTRUCTION,
            reason="qq_turn",
        )
    if normalized_source != "desktop":
        return SceneContinuityDecision(reason="unsupported_source_channel")

    history = TodayHistoryLoader(history_root).load(reference_time=reference_time)
    if not history.turns:
        return SceneContinuityDecision(reason="no_prior_turn")

    latest_source = str(
        history.turns[-1].metadata.get("source_channel") or "desktop"
    ).casefold()
    if latest_source in _QQ_SOURCE_CHANNELS:
        return SceneContinuityDecision(
            instruction=SCENE_CONTINUITY_SYSTEM_INSTRUCTION,
            reason="first_desktop_turn_after_qq",
        )
    return SceneContinuityDecision(reason="latest_turn_not_qq")


__all__ = [
    "SCENE_CONTINUITY_SYSTEM_INSTRUCTION",
    "SceneContinuityDecision",
    "decide_scene_continuity_instruction",
]
