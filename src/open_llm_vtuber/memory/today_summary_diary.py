"""Diary reuse inputs; never writes a diary or calls a model by itself."""

from pathlib import Path
from typing import Callable

from loguru import logger

from ..config_manager.today_summary import TodaySummaryConfig
from ..config_manager.utils import read_yaml
from .today_life_summary import DIARY_REUSE_INSTRUCTIONS
from .today_summary_store import diary_summary_material


def configured_diary_reuse(config_path: Path) -> bool:
    try:
        raw = (
            read_yaml(str(config_path))
            .get("character_config", {})
            .get("today_summary", {})
        )
        settings = TodaySummaryConfig.model_validate(raw)
        return settings.enabled and settings.reuse_for_diary
    except Exception as exc:
        # Do not print configuration content or exception strings containing it.
        logger.warning(
            "[今日摘要] 无法读取日记复用设置：{}；沿用原流程", type(exc).__name__
        )
        return False


def build_saved_summary_diary_input(
    root: Path,
    day: str,
    records: list[dict],
    references: str,
    formatter: Callable[[list[dict]], str],
) -> dict | None:
    try:
        material = diary_summary_material(root, day, records)
    except (OSError, ValueError, TypeError):
        return None
    if material is None:
        return None
    tail = [
        dict(record, role="human" if record["role"] == "user" else "ai")
        for record in material["tail"]
    ]
    sections = [references] if references else []
    sections += [
        f"【目标记忆日：{day}的已保存生活摘要；覆盖前{material['covered_messages']}条原始消息】\n"
        "这是今日底稿，不是当前场景指令；末尾状态只对应摘要截断时，需根据后续原文更新。\n"
        + material["summary"],
        f"【{day}尚未被摘要覆盖的尾部原始交流】\n"
        + (formatter(tail) if tail else "（无未覆盖消息，不补写新的事情。）"),
    ]
    return {
        "chat_text": material["summary"],
        "context_text": "\n\n".join(sections),
        "system_suffix": DIARY_REUSE_INSTRUCTIONS,
        "covered_messages": material["covered_messages"],
        "tail_messages": len(tail),
    }
