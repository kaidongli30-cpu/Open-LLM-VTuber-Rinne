"""Command-line preview for the independent today-message partition prototype."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from .today_history_loader import TodayHistoryLoader
from .today_partition import (
    TodayMessagePartitioner,
    build_today_llm_delivery_preview,
)


def _parse_as_of(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "时间格式无效，请使用例如 2026-05-21T20:00:00+08:00"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        local_timezone = datetime.now().astimezone().tzinfo
        if local_timezone is None:
            raise argparse.ArgumentTypeError("无法确定本机时区")
        parsed = parsed.replace(tzinfo=local_timezone)
    return parsed


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须是正整数") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "从临时目录中的历史 JSON 模拟指定时刻之前会递送给 LLM 的最近消息。"
        )
    )
    parser.add_argument(
        "--history-root",
        type=Path,
        required=True,
        help="存放实验 JSON 副本的临时目录；只扫描该目录直属 JSON。",
    )
    parser.add_argument(
        "--as-of",
        type=_parse_as_of,
        required=True,
        help="模拟递送时间，例如 2026-05-21T20:00:00+08:00。",
    )
    parser.add_argument(
        "--recent-messages",
        type=_positive_integer,
        default=20,
        help="保留的最近原文消息数，默认20。",
    )
    parser.add_argument(
        "--current-input",
        required=True,
        help="模拟本轮刚刚收到的用户文本；它会作为最后一条user消息。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="固定预览JSON路径；重复运行时原地更新，必须位于实验目录的子目录中。",
    )
    return parser


def _validate_paths(history_root: Path, output: Path) -> tuple[Path, Path]:
    resolved_root = history_root.resolve()
    resolved_output = output.resolve()
    if not resolved_root.is_dir():
        raise ValueError(f"实验目录不存在：{resolved_root}")
    if resolved_output.parent == resolved_root:
        raise ValueError("输出文件必须放在实验目录的子目录中，不能混入源 JSON。")
    if resolved_root not in resolved_output.parents:
        raise ValueError("输出文件必须位于实验目录内部，便于整体清理。")
    return resolved_root, resolved_output


def _atomically_update_json(output: Path, payload: dict) -> None:
    """Replace one fixed preview file without accumulating per-turn files."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        history_root, output = _validate_paths(
            arguments.history_root,
            arguments.output,
        )
        loader = TodayHistoryLoader(
            history_root,
            local_timezone=arguments.as_of.tzinfo,
        )
        history = loader.load(reference_time=arguments.as_of)
        partition = TodayMessagePartitioner(arguments.recent_messages).partition(
            history, as_of=arguments.as_of
        )
        preview = build_today_llm_delivery_preview(
            partition,
            current_user_input=arguments.current_input,
        )
        _atomically_update_json(output, preview.to_dict())
    except (OSError, ValueError) as error:
        parser.exit(2, f"无法生成预览：{error}\n")

    diagnostics = partition.diagnostics
    print(f"使用的记忆日：{partition.memory_day.isoformat()}")
    print(f"模拟递送时间（不包含该时刻）：{partition.as_of_exclusive.isoformat()}")
    print(f"扫描 JSON 数量：{history.diagnostics.scanned_files}")
    print(f"成功解析 JSON 数量：{history.diagnostics.parsed_files}")
    print(f"该时刻以前的消息数量：{diagnostics.messages_strictly_before_as_of}")
    print(f"最近原文消息数量：{diagnostics.recent_messages}")
    print(
        f"加上本次用户输入后的messages总数：{len(preview.simulated_messages_for_llm)}"
    )
    print(f"较早今日消息数量：{diagnostics.older_today_messages}")
    print(
        "最近原文中的完整 user/assistant 对数："
        f"{diagnostics.complete_user_assistant_pairs_in_recent}"
    )
    print(f"正式聊天链路已连接：{preview.live_pipeline_connected}")
    print(f"固定预览文件（已原地更新）：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
