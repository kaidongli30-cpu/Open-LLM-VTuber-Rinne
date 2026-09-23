"""
凛祢日记生成器 —— diary_generator.py
放在项目根目录（和 run_server.py 同级）

用法：
  python diary_generator.py             → 启动定时器，每天凌晨3:00自动生成昨天的日记
  python diary_generator.py today       → 手动生成"当前日"的日记（调试用）
  python diary_generator.py 2026-05-19  → 手动生成指定日期的日记
  python diary_generator.py all         → 为所有有记录但还没有日记的日期批量生成

日记文件保存在 RINNE_DATA_ROOT 下的 chat_history/rinne_01/diaries；
未设置该变量时保持原来的项目内相对路径。
"""

import json
import hashlib
import time
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator
import requests

from src.open_llm_vtuber.data_paths import character_history_root
from src.open_llm_vtuber.memory.diary_review import diary_approval_status
from src.open_llm_vtuber.memory.diary_composition import (
    DIARY_COMPRESSION_SYSTEM_PROMPT,
    DIARY_CONTINUITY_AUDIT_SYSTEM_PROMPT,
    DIARY_LANGUAGE_AUDIT_SYSTEM_PROMPT,
    DIARY_WRITER_SYSTEM_PROMPT,
    EVENT_CONSOLIDATION_SYSTEM_PROMPT,
    EVENT_INVENTORY_SYSTEM_PROMPT,
    MESSAGE_GROUPING_SYSTEM_PROMPT,
    DiaryCompositionError,
    apply_language_audit,
    consolidate_events,
    format_grouped_messages,
    format_message_groups,
    format_inventory_for_writer,
    normalize_marked_diary_order,
    parse_event_inventory,
    parse_event_consolidation,
    parse_marked_diary,
    parse_message_groups,
    paragraph_character_budgets,
    target_diary_characters,
    validate_consolidation_boundaries,
)
from src.open_llm_vtuber.memory.message_time_context import add_message_time_context
from src.open_llm_vtuber.memory.layer2_context import (
    load_layer2_publication_as_of,
)
from src.open_llm_vtuber.memory.scene_state import SceneStateStore

# ===================== 配置区（按需修改） =====================
CHAT_HISTORY_DIR = character_history_root("rinne_01")  # 聊天记录文件夹
DIARY_DIR = CHAT_HISTORY_DIR / "diaries"           # 日记存放文件夹

# LLM API 配置（兼容 DeepSeek / OpenAI-compatible 等接口）
# 你可以直接在这里填写，也可以用环境变量覆盖：
#   DIARY_LLM_API_KEY / DIARY_LLM_API_URL / DIARY_LLM_MODEL
LLM_API_KEY = os.getenv("DIARY_LLM_API_KEY") or os.getenv("RINNE_APINEBULA_API_KEY", "")
LLM_API_URL = os.getenv("DIARY_LLM_API_URL", "https://apinebula.ai/v1/chat/completions")
LLM_MODEL   = os.getenv("DIARY_LLM_MODEL", "claude-sonnet-4-6")

# 日记的最大字数（token 层面的软限制，不是硬截断）
DIARY_MAX_TOKENS = 5000
# =============================================================

DIARY_SYSTEM_PROMPT = """你是园神凛祢（SonogamiRinne），一位温柔知性的精灵少女，现在住在用户的电脑里。
请你用自己的口吻，根据下面提供的聊天记录，写一篇简短的个人日记。

写作要求：
1. 以第一人称"我"写，完全是凛祢自己的视角和感受。
2. 日记需要简洁但有情感，不要废话。开头标注日记日期；聊天中提及今日天气时也可注明天气。
3. 用简体中文书写，不要用日语，也不要写入聊天记录中用[]括起来的感情标签。
4. 重点记录用户今天的状态和心情、我们经历或谈论的值得记住的事情，以及我真正留下的感受。
5. 语气温柔知性，带有凛祢特有的细腻，偶尔一点点羞涩。
6. 聊天很少时就简短写几句，不要强行填充；聊天很多时，篇幅也不要随聊天轮数等比例增长。正文不设最低字数，也不要求接近任何目标长度；上限为1500个中文字符，应靠合并同一件事的过程来控制篇幅，不能靠遗漏独立而重要的事情。
7. 不要按聊天轮次复述，不要大段照搬双方原话。同一件事只写一次：把围绕它的来回对话、动作、重复确认和重复情绪合并，用自己的话概括发生了什么、结果或当前状态是什么，以及最重要的感受。只有专名、关键数字和确有必要的一句短原话可以保留。
8. 当用户讲述遇见凛祢以前的经历时，仍要保留足以辨认该经历的关键人物、地点、原因、经过、数字和结果，不要因为事情发生在过去而略过。重要的人际互动也要记下结果和感受，但不要逐项描写动作或复演对话。
9. 日记只能记录材料中确实发生、明确说出或明确决定的事情。不得补写聊天中不存在的行为、感受、计划、地点、原因或结果；无法确认时应省略或明确写成不确定。
10. 注意每条聊天记录前的消息时间与记忆日标签，越靠后的聊天发生时间也越靠后。不得把一天后期发生的事情写到前面，也不得把后面已经完成的状态写成未完成；但不必写出每一轮对话。
11. 时间标签表示这句话何时说出，不自动表示句中事件发生在当天。聊天中明确说“昨天、昨晚、前天、明天、后天”时，以该标签的记忆日为基准理解，不得因为事件在目标日被再次提起，就把它写成目标日发生。
12. 不要把聊天消息的时间戳写成日记叙事。普通事情不要出现“16:57”“22点37分”这类几时几分的词；改成上午、中午、下午、晚上、吃饭前后或某个行动前后，语句通顺时也可省略。只有具体时刻本身就是关键事实的重大时刻才可保留钟点。
13. 结尾只写当天真实的收束状态或自然感受，不要自动添加倒计时、明日计划、口号或新的情节；只有聊天中明确确定的计划才能记录为计划。
14. 默认称呼对方为“用户”；若用户明确指定称呼或代词，则尊重用户的选择，不预设性别。

【记忆连续性与事实优先级】
1. 目标日中用户明确说出的事实和纠正最高。
2. 目标日场景变更只用于确认位置、共同空间、目的和待处理请求，不能把模型描述当作用户事实。
3. Layer 2背景只用于补足目标日前已经形成的连续状态，不能把目标日之后才知道的内容带回目标日。
4. 前一个完整记忆日的日记只是跨日事实参考，不是今天的写作范例。它只用于理解前因后果和延续状态；不得模仿它的篇幅、段落数量、叙事密度、内容取舍或措辞。
5. 聊天记录中的凛祢回复、日记里的抒情或猜测不能覆盖用户本人的明确说法，也不能被改写成新的客观事实。
如果材料冲突，保留较高等级材料；无法确定时省略或保留“不确定”，不要猜测。不要把计划写成已经完成，不要把询问写成事实。只输出日记正文，不要解释材料或规则。"""


def get_day_range(target_date: datetime):
    """
    计算"第 i 日"的时间范围：
    target_date 当天凌晨 3:00  →  target_date+1 凌晨 3:00
    """
    start = target_date.replace(hour=3, minute=0, second=0, microsecond=0)
    end   = start + timedelta(days=1)
    return start, end


def load_messages_in_range(start: datetime, end: datetime) -> list:
    """
    读取 CHAT_HISTORY_DIR 下所有命名格式为 YYYY-MM-DD_HH-MM-SS_xxx.json
    且开始时间戳落在 [start, end) 范围内的聊天文件，合并返回消息列表。
    """
    collected = []

    if not CHAT_HISTORY_DIR.exists():
        return collected

    def parse_timestamp(value):
        if not isinstance(value, str) or not value.strip():
            return None
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if parsed.tzinfo is not None and parsed.utcoffset() is not None:
            parsed = parsed.astimezone().replace(tzinfo=None)
        return parsed

    # Compare message timestamps, not a conversation file's first timestamp:
    # one JSON may contain turns that cross the 03:00 memory-day boundary.
    start = start.astimezone().replace(tzinfo=None) if start.tzinfo else start
    end = end.astimezone().replace(tzinfo=None) if end.tzinfo else end
    ordered = []
    sequence = 0
    from src.open_llm_vtuber.memory.diary_sources import history_files

    for json_file in history_files(CHAT_HISTORY_DIR):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            print(f"  [警告] 读取文件失败 {json_file.name}: {exc}")
            continue
        if not isinstance(data, list):
            continue
        for record in data:
            if not isinstance(record, dict) or record.get("role") == "metadata":
                continue
            timestamp = parse_timestamp(record.get("timestamp"))
            if timestamp is None or not start <= timestamp < end:
                continue
            ordered.append((timestamp, sequence, record))
            sequence += 1
    ordered.sort(key=lambda item: (item[0], item[1]))
    return [record for _timestamp, _sequence, record in ordered]


def format_for_llm(messages: list) -> str:
    """将聊天记录列表格式化为 LLM 可读的纯文本。"""
    lines = []
    for msg in messages:
        role    = msg.get("role", "")
        content = msg.get("content", "")

        if not content or role == "metadata":
            continue

        timestamped_content = add_message_time_context(
            content,
            msg.get("timestamp"),
        )

        if role == "human":
            lines.append(f"[M{len(lines) + 1:03d}] 用户：{timestamped_content}")
        elif role == "ai":
            lines.append(f"[M{len(lines) + 1:03d}] 凛祢：{timestamped_content}")

    return "\n".join(lines)


def _format_scene_context(memory_day: str) -> str:
    """Format only the temporary scene changes for one target memory day."""

    try:
        records = SceneStateStore(CHAT_HISTORY_DIR).load_changes(memory_day)
    except Exception as exc:
        print(f"  [警告] 读取目标日场景记录失败：{exc}")
        return ""
    if not records:
        return ""
    lines = [f"【{memory_day}的场景变更记录（仅作当前场景证据）】"]
    for record in records:
        changed = record.get("changed")
        if not isinstance(changed, dict):
            continue
        source = record.get("source") or "unknown"
        source_text = record.get("source_text") or ""
        lines.append(
            f"来源={source}；变更={json.dumps(changed, ensure_ascii=False)}"
            + (f"；用户原话={source_text}" if source_text else "")
        )
    return "\n".join(lines) if len(lines) > 1 else ""


def _load_prior_diaries(target_date: datetime) -> tuple[str, list[str]]:
    """Return the previous approved daily diary and read warnings."""

    target_day = target_date.date()
    entries = []
    warnings = []
    for offset in range(1, 2):
        day = target_day - timedelta(days=offset)
        path = DIARY_DIR / f"diary_{day.isoformat()}.txt"
        status = diary_approval_status(CHAT_HISTORY_DIR, day.isoformat(), path)
        if status not in {"approved", "changed"}:
            if status not in {"missing", "unapproved"}:
                warnings.append(f"跳过{day.isoformat()}日记（状态：{status}）")
            continue
        if status == "changed":
            warnings.append(
                f"{path.name}在验收后发生修改，日记生成按当前内容读取"
            )
        try:
            content = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            warnings.append(f"无法读取{path.name}：{exc}")
            continue
        if content:
            entries.append(f"【{day.isoformat()}的已验收日记】\n{content}")
    return "\n\n".join(reversed(entries)), warnings


def _load_prior_layer2(target_date: datetime) -> tuple[str, dict]:
    """Load the latest verified Layer-2 snapshot strictly before the target day."""

    cutoff = target_date.date() - timedelta(days=1)
    try:
        result = load_layer2_publication_as_of(CHAT_HISTORY_DIR, cutoff)
    except Exception as exc:
        return "", {"status": "failed", "error": str(exc)}
    return result.context, result.diagnostics


def build_diary_reference_context(target_date: datetime) -> str:
    """Load prior-day continuity references without target-day event material."""

    sections: list[str] = []
    layer2, layer2_diagnostics = _load_prior_layer2(target_date)
    if layer2:
        sections.append(
            "【目标日前已存在的Layer 2背景（不可使用目标日之后的信息）】\n"
            + layer2
        )
    prior_diaries, warnings = _load_prior_diaries(target_date)
    if warnings:
        for warning in warnings:
            print(f"  [警告] {warning}")
    if prior_diaries:
        sections.append(
            "【跨日事实参考，不是写作范例】\n"
            "以下内容只用于确认前一天已经发生的事情和延续至今日的状态。"
            "不得模仿它的篇幅、段落数量、叙事密度、内容取舍或措辞；"
            "今日的日记结构只能由今日实际发生的事情决定。\n\n"
            + prior_diaries
        )
    if layer2_diagnostics.get("warnings"):
        for warning in layer2_diagnostics["warnings"]:
            print(f"  [警告] {warning}")
    return "\n\n".join(sections)


def build_diary_source_context(
    target_date: datetime,
    messages: list,
    *,
    reference_context: str | None = None,
) -> str:
    """Assemble references first and put the target-day chat closest to writing."""

    date_label = target_date.strftime("%Y-%m-%d")
    sections: list[str] = []
    references = (
        build_diary_reference_context(target_date)
        if reference_context is None
        else reference_context
    )
    if references:
        sections.append(references)
    scene = _format_scene_context(date_label)
    if scene:
        sections.append(scene)
    sections.append(
        f"【目标记忆日：{date_label}的原始聊天（按消息时间筛选；"
        "这是今天日记的主要依据）】\n"
        + (format_for_llm(messages) or "（这一天没有可用的聊天消息。）")
    )
    return "\n\n".join(sections)


class DiaryStreamProtocolError(ValueError):
    """A streamed response ended with an incomplete or invalid SSE JSON event."""


class DiaryOutputTruncatedError(RuntimeError):
    """The provider stopped because the configured output limit was reached."""


class DiaryStreamStatus:
    """Completion metadata collected while parsing one streamed response."""

    def __init__(self) -> None:
        self.saw_done = False
        self.event_count = 0


def _decode_sse_json_parts(
    data_parts: list[str],
    event_index: int,
    *,
    preserve_physical_newlines: bool = False,
) -> dict[str, Any]:
    """Decode one SSE event, including providers that split JSON across data lines."""

    standard_payload = "\n".join(data_parts)
    escaped_newline_payload = "\\n".join(data_parts)
    compact_payload = "".join(data_parts)
    payloads = [standard_payload]
    if preserve_physical_newlines and escaped_newline_payload not in payloads:
        payloads.append(escaped_newline_payload)
    if compact_payload not in payloads:
        payloads.append(compact_payload)

    last_error: json.JSONDecodeError | None = None
    for payload in payloads:
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(event, dict):
            raise DiaryStreamProtocolError(
                f"SSE第{event_index}个事件的JSON顶层不是对象"
            )
        return event

    character_count = sum(len(part) for part in data_parts)
    reason = last_error.msg if last_error is not None else "未知解析错误"
    raise DiaryStreamProtocolError(
        f"SSE第{event_index}个事件不完整或格式错误"
        f"（累计{character_count}字符；{reason}）"
    )


def _iter_sse_json_events(
    raw_lines: Iterable[str | bytes],
    *,
    status: DiaryStreamStatus | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield complete JSON events from standard and compatible SSE streams."""

    stream_status = status or DiaryStreamStatus()
    data_parts: list[str] = []
    event_index = 0
    has_bare_continuation = False

    def decode_pending() -> dict[str, Any]:
        return _decode_sse_json_parts(
            data_parts,
            event_index + 1,
            preserve_physical_newlines=has_bare_continuation,
        )

    for raw_line in raw_lines:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace")
        else:
            line = raw_line or ""
        line = line.rstrip("\r")
        if event_index == 0 and not data_parts:
            line = line.lstrip("\ufeff")

        if not line:
            if data_parts:
                event = decode_pending()
                data_parts.clear()
                has_bare_continuation = False
                event_index += 1
                stream_status.event_count = event_index
                yield event
            continue
        if line.startswith(":"):
            continue

        field, separator, value = line.partition(":")
        if field.strip() != "data" or not separator:
            if data_parts and field.strip() not in {"event", "id", "retry"}:
                # Some compatible gateways continue a JSON string on a bare
                # physical line.  Preserve the intended newline when that is
                # the only form that produces valid JSON.
                data_parts.append(line)
                has_bare_continuation = True
                try:
                    event = decode_pending()
                except DiaryStreamProtocolError:
                    continue
                data_parts.clear()
                has_bare_continuation = False
                event_index += 1
                stream_status.event_count = event_index
                yield event
            continue
        if value.startswith(" "):
            value = value[1:]
        if not value:
            continue
        if value.strip() == "[DONE]":
            if data_parts:
                event = decode_pending()
                data_parts.clear()
                has_bare_continuation = False
                event_index += 1
                stream_status.event_count = event_index
                yield event
            stream_status.saw_done = True
            return

        data_parts.append(value)
        try:
            event = decode_pending()
        except DiaryStreamProtocolError:
            # One logical SSE event may span several data lines.  A blank
            # line, [DONE], or EOF remains the authoritative event boundary.
            continue
        data_parts.clear()
        has_bare_continuation = False
        event_index += 1
        stream_status.event_count = event_index
        yield event

    if data_parts:
        event = decode_pending()
        event_index += 1
        stream_status.event_count = event_index
        yield event


def call_llm_api(
    chat_text: str,
    date_label: str,
    context_text: str | None = None,
    *,
    system_prompt: str | None = None,
    user_content_override: str | None = None,
    max_tokens: int | None = None,
    purpose: str = "日记",
) -> str:
    """调用通用 LLM API，完整接收一次文本生成结果。"""

    effective_system_prompt = system_prompt or DIARY_SYSTEM_PROMPT
    effective_max_tokens = max_tokens or DIARY_MAX_TOKENS
    if user_content_override is not None:
        user_content = user_content_override
    # 如果今天完全没有聊天记录
    elif not chat_text.strip():
        no_chat_prompt = f"今天（{date_label}）用户没有来和我说话。请以凛祢的口吻用简体中文写几句简短的日记，表达等待与思念。直接从日期开始写日记正文，不要写出“我用凛祢的语气开始写……”这样的话。"
        user_content = no_chat_prompt
    else:
        user_content = (
            f"以下是{date_label}这一天生成日记所需的全部材料：\n\n"
            f"{context_text or chat_text}\n\n"
            f"请根据以上材料，以园神凛祢的口吻直接写今天的日记正文。"
        )

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": effective_system_prompt},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": effective_max_tokens,
        "temperature": 0,
        "stream": True,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }

    system_chars = len(effective_system_prompt)
    user_chars = len(user_content)
    request_bytes = len(
        (effective_system_prompt + user_content).encode("utf-8")
    )
    print(
        f"  {purpose}请求材料："
        f"system={system_chars}字符，user={user_chars}字符，"
        f"合计约{request_bytes}字节；模型={LLM_MODEL}"
    )

    connect_timeout_seconds = 20
    stream_read_timeout_seconds = 180
    max_attempts = 2
    retryable_statuses = {408, 429, 500, 502, 503, 504}
    for attempt in range(1, max_attempts + 1):
        chunks: list[str] = []
        finish_reasons: list[str] = []
        stream_status: DiaryStreamStatus | None = None
        is_stream_response = False
        started_at = time.perf_counter()
        first_content_at: float | None = None
        try:
            with requests.post(
                LLM_API_URL,
                json=payload,
                headers=headers,
                stream=True,
                timeout=(
                    connect_timeout_seconds,
                    stream_read_timeout_seconds,
                ),
            ) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/event-stream" not in content_type:
                    result = response.json()
                    choice = result["choices"][0]
                    finish_reason = choice.get("finish_reason")
                    if isinstance(finish_reason, str) and finish_reason.strip():
                        finish_reasons.append(finish_reason.strip().casefold())
                    content = choice["message"]["content"]
                    if isinstance(content, str) and content:
                        chunks.append(content)
                        first_content_at = time.perf_counter()
                else:
                    is_stream_response = True
                    stream_status = DiaryStreamStatus()
                    raw_lines = response.iter_lines(
                        decode_unicode=False,
                        delimiter=b"\n",
                    )
                    for event in _iter_sse_json_events(
                        raw_lines,
                        status=stream_status,
                    ):
                        choices = event.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        finish_reason = choice.get("finish_reason")
                        if isinstance(finish_reason, str) and finish_reason.strip():
                            finish_reasons.append(finish_reason.strip().casefold())
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            chunks.append(content)
                            if first_content_at is None:
                                first_content_at = time.perf_counter()

            truncated_reasons = {"length", "max_tokens", "max_output_tokens"}
            successful_reasons = {"stop", "end_turn", "stop_sequence"}
            observed_reasons = set(finish_reasons)
            if observed_reasons & truncated_reasons:
                raise DiaryOutputTruncatedError(
                    f"服务商因达到{effective_max_tokens} token输出上限而停止"
                )
            unexpected_reasons = observed_reasons - successful_reasons
            if unexpected_reasons:
                raise DiaryStreamProtocolError(
                    "服务商返回非正常结束原因："
                    + ",".join(sorted(unexpected_reasons))
                )
            if (
                is_stream_response
                and stream_status is not None
                and not stream_status.saw_done
                and not (observed_reasons & successful_reasons)
            ):
                raise DiaryStreamProtocolError(
                    "SSE流结束时既没有[DONE]，也没有正常finish_reason"
                )

            diary_text = "".join(chunks).strip()
            if not diary_text:
                raise ValueError("接口完成响应，但没有返回日记正文")

            total_seconds = time.perf_counter() - started_at
            first_seconds = (
                first_content_at - started_at
                if first_content_at is not None
                else total_seconds
            )
            print(
                f"  {purpose}流式响应完成："
                f"首段={first_seconds:.1f}秒，总计={total_seconds:.1f}秒，"
                f"尝试={attempt}/{max_attempts}"
            )
            return diary_text
        except Exception as exc:
            if isinstance(exc, DiaryOutputTruncatedError):
                print(
                    f"  [错误] {purpose}达到输出上限，判定为不完整；"
                    "本次不落盘，请提高上限后重新生成："
                    f"{exc}"
                )
                return ""
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            retryable = (
                isinstance(
                    exc,
                    (
                        requests.Timeout,
                        requests.ConnectionError,
                        json.JSONDecodeError,
                        ValueError,
                    ),
                )
                or status_code in retryable_statuses
            )
            can_retry = (
                attempt < max_attempts
                and retryable
                and not chunks
            )
            if can_retry:
                print(
                    f"  [警告] {purpose}接口在返回正文前失败，2秒后自动重试一次："
                    f"{exc}"
                )
                time.sleep(2)
                continue
            if chunks:
                print(
                    f"  [错误] {purpose}流式响应中途断开；为避免保存半成品，"
                    "本次不落盘，也不自动重复计费："
                    f"{exc}"
                )
            else:
                print(f"  [错误] 调用{purpose} LLM API失败: {exc}")
            return ""
    return ""


def _grouping_request(date_label: str, chat_text: str) -> str:
    return (
        f"目标记忆日：{date_label}\n\n"
        "请将下面全部编号消息归组。只做归组，不要概括正文。\n\n"
        f"{chat_text}"
    )


def _grouping_repair_request(
    date_label: str,
    chat_text: str,
    rejected_output: str,
    reason: str,
) -> str:
    return (
        f"你上一份{date_label}消息归组未通过后台验证：{reason}\n"
        "请重新输出完整归组。每个M编号必须且只能出现一次；不要摘要或解释。\n\n"
        f"【上一次不合格输出】\n{rejected_output}\n\n"
        f"【全部编号消息】\n{chat_text}"
    )


def _inventory_request(
    date_label: str,
    grouped_chat: str,
    source_groups: list[list[str]],
    group_start_index: int,
) -> str:
    return (
        f"目标记忆日：{date_label}\n\n"
        "下面是整天固定归组中的一个小批次。请严格按所给G编号各生成一个对应E编号"
        "事件，不得重新分组、缺组或增加其他事件。\n\n"
        "【固定消息归组】\n"
        f"{format_message_groups(source_groups, start_index=group_start_index)}\n\n"
        f"{grouped_chat}"
    )


def _inventory_repair_request(
    date_label: str,
    grouped_chat: str,
    source_groups: list[list[str]],
    group_start_index: int,
    rejected_output: str,
    reason: str,
) -> str:
    return (
        f"你上一份{date_label}事件清单未通过后台验证：{reason}\n"
        "请重新阅读全部材料并按规定格式输出一份完整的新清单。不要解释，不要沿用无依据的"
        "证据；不得为了通过格式验证而遗漏事件。\n\n"
        f"【上一次不合格输出】\n{rejected_output}\n\n"
        "【固定消息归组】\n"
        f"{format_message_groups(source_groups, start_index=group_start_index)}\n\n"
        f"【本批原始消息】\n{grouped_chat}"
    )


def _writer_request(
    date_label: str,
    events: list[dict[str, Any]],
) -> str:
    lower, upper = target_diary_characters(events)
    budgets = paragraph_character_budgets(events, upper)
    required_terms = _required_terms_for_events(events)
    budget_lines = []
    for event in events:
        protection = required_terms.get(event["id"])
        protection_text = (
            f"；受保护内容必须明确包含以下词之一：{'、'.join(protection)}"
            if protection
            else ""
        )
        budget_lines.append(
            f"{event['id']}（{event['importance']}）：最多{budgets[event['id']]}字"
            f"{protection_text}"
        )
    budget_text = "\n".join(budget_lines)
    return (
        f"目标记忆日：{date_label}\n"
        f"本日共有{len(events)}个已经合并和核验的事件。建议正文约{lower}至{upper}"
        "个中文字符；这是防止流水账的篇幅指导，不得用它删除事件。\n"
        "下面的分段上限是硬约束，指对应标记后的正文长度；先概括再写，"
        "不要依靠事后截断。\n"
        f"{budget_text}\n\n"
        "【已核验事件清单】\n"
        f"{format_inventory_for_writer(events)}\n\n"
        "请写带事件覆盖标记的日记。"
    )


def _consolidation_source(events: list[dict[str, Any]]) -> str:
    lines = []
    for event in events:
        lines.append(
            f"{event['id']} | {event['kind']} | {event['importance']} | "
            f"{event['title']} | {event['summary']}"
        )
    return "\n".join(lines)


def _consolidation_request(date_label: str, events: list[dict[str, Any]]) -> str:
    return (
        f"目标记忆日：{date_label}\n"
        f"以下共有{len(events)}个已核验细粒度事件。请只决定如何按相邻顺序合并。\n\n"
        f"{_consolidation_source(events)}"
    )


def _consolidation_repair_request(
    date_label: str,
    events: list[dict[str, Any]],
    rejected_output: str,
    reason: str,
) -> str:
    return (
        f"你上一份{date_label}叙事合并未通过后台验证：{reason}\n"
        "请重新输出完整合并表。每个E编号必须按原顺序且只能出现一次，不要解释。\n\n"
        f"【上一次不合格输出】\n{rejected_output}\n\n"
        f"【全部细粒度事件】\n{_consolidation_source(events)}"
    )


def _writer_repair_request(
    date_label: str,
    events: list[dict[str, Any]],
    rejected_output: str,
    reason: str,
) -> str:
    lower, upper = target_diary_characters(events)
    budgets = paragraph_character_budgets(events, upper)
    required_terms = _required_terms_for_events(events)

    def clipped(value: object, limit: int) -> str:
        text = str(value).strip()
        return text if len(text) <= limit else text[: limit - 1] + "…"

    checklist_lines = []
    for event in events:
        protection = required_terms.get(event["id"])
        protection_text = (
            f" | 受保护内容必须明确包含以下词之一={'、'.join(protection)}"
            if protection
            else ""
        )
        checklist_lines.append(
            f"{event['id']} | {event['kind']} | {event['importance']} | "
            f"最多{budgets[event['id']]}字 | {event['title']} | "
            f"核心={clipped(event['summary'], 220)} | "
            f"结果={clipped(event['outcome'], 100)}{protection_text}"
        )
    checklist = "\n".join(checklist_lines)
    return (
        f"请压缩编辑{date_label}日记。当前稿未通过验证：{reason}\n"
        f"正文必须在{upper}个字符以内，建议不少于{lower}个字符。"
        "每个事件编号只能在段前标记中出现一次，每个标记后只写一个自然段，"
        "不要在同一标记下再用空行拆段。每段都必须低于清单中给出的独立上限；"
        "只在现有文字上删减，不要重新展开。\n\n"
        f"【必须保留的事件编号与短标题】\n{checklist}\n\n"
        f"【待压缩完整初稿】\n{rejected_output}"
    )


def _required_terms_for_events(
    events: list[dict[str, Any]],
) -> dict[str, tuple[str, ...]]:
    relationship_terms = (
        "关系",
        "边界",
        "尊重",
        "同意",
        "支持",
        "承诺",
    )
    past_terms = (
        "告诉我",
        "跟我讲",
        "讲起",
        "提起",
        "回忆",
        "过去",
        "以前",
        "当初",
    )
    required: dict[str, tuple[str, ...]] = {}
    for event in events:
        if event["kind"] == "private_relationship":
            event_text = "\n".join(
                [
                    str(event.get("summary", "")),
                    *[str(item) for item in event.get("key_details", [])],
                    str(event.get("outcome", "")),
                ]
            )
            found = tuple(term for term in relationship_terms if term in event_text)
            if found:
                required[event["id"]] = found
        elif event["kind"] == "past_story_told_today":
            required[event["id"]] = past_terms
    return required


def _language_audit_request(
    date_label: str,
    events: list[dict[str, Any]],
    marked_diary: str,
    upper_characters: int,
) -> str:
    references = "\n".join(
        f"{event['id']} | {event['title']} | {str(event['summary'])[:260]}"
        for event in events
    )
    return (
        f"目标记忆日：{date_label}\n"
        f"当前正文上限是{upper_characters}个中文字符；修正不得扩写正文。\n\n"
        f"【事件核验表（仅用于判断语言指代）】\n{references}\n\n"
        f"【待审计日记】\n{marked_diary}"
    )


def _continuity_audit_request(
    date_label: str,
    events: list[dict[str, Any]],
    marked_diary: str,
    reference_text: str,
    upper_characters: int,
) -> str:
    persona_reference = DIARY_SYSTEM_PROMPT.split(
        "请你用自己的口吻",
        1,
    )[0].strip()
    references = "\n".join(
        f"{event['id']} | {event['title']} | {str(event['summary'])[:260]}"
        for event in events
    )
    return (
        f"目标记忆日：{date_label}\n"
        f"当前正文上限是{upper_characters}个中文字符；修正不得扩写正文。\n\n"
        f"【凛祢的身份与称呼参照】\n{persona_reference}\n\n"
        f"【事件核验表（目标日事实，以此为准）】\n{references}\n\n"
        f"【目标日前连续性参照（不得据此新增目标日事件）】\n"
        f"{reference_text}\n\n"
        f"【待审计日记】\n{marked_diary}"
    )


def _checkpoint_path(
    checkpoint_dir: Path | None,
    label: str,
    source_text: str,
) -> Path | None:
    if checkpoint_dir is None:
        return None
    digest = hashlib.sha256(
        (LLM_MODEL + "\n" + label + "\n" + source_text).encode("utf-8")
    ).hexdigest()[:16]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{label}_{digest}.json"


def _load_checkpoint(path: Path | None) -> Any | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _save_checkpoint(path: Path | None, payload: Any) -> None:
    if path is None:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def generate_diary_two_stage(
    chat_text: str,
    date_label: str,
    context_text: str,
    *,
    checkpoint_dir: Path | None = None,
    reference_text: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Group every message, plan each event, then write the validated diary."""

    if not chat_text.strip():
        diary = call_llm_api(chat_text, date_label, context_text=context_text)
        return diary, []

    grouping_checkpoint = _checkpoint_path(
        checkpoint_dir,
        "message_groups",
        MESSAGE_GROUPING_SYSTEM_PROMPT + chat_text,
    )
    cached_groups = _load_checkpoint(grouping_checkpoint)
    source_groups: list[list[str]]
    if isinstance(cached_groups, list):
        grouping_protocol = "\n".join(
            f"@@GROUP G{index:03d}\nsources={','.join(group)}\n@@END"
            for index, group in enumerate(cached_groups, start=1)
        ) + "\n@@COMPLETE"
        try:
            source_groups = parse_message_groups(grouping_protocol, chat_text)
            print(f"  复用已验证消息归组断点：{grouping_checkpoint.name}")
        except DiaryCompositionError:
            cached_groups = None
    if not isinstance(cached_groups, list):
        grouping_text = call_llm_api(
            chat_text,
            date_label,
            system_prompt=MESSAGE_GROUPING_SYSTEM_PROMPT,
            user_content_override=_grouping_request(date_label, chat_text),
            max_tokens=min(DIARY_MAX_TOKENS, 2500),
            purpose="日记消息归组",
        )
        if not grouping_text:
            return "", []
        try:
            source_groups = parse_message_groups(grouping_text, chat_text)
        except DiaryCompositionError as exc:
            print(f"  [警告] 消息归组首次验证失败，进行一次定向修复：{exc}")
            repaired_grouping = call_llm_api(
                chat_text,
                date_label,
                system_prompt=MESSAGE_GROUPING_SYSTEM_PROMPT,
                user_content_override=_grouping_repair_request(
                    date_label,
                    chat_text,
                    grouping_text,
                    str(exc),
                ),
                max_tokens=min(DIARY_MAX_TOKENS, 2500),
                purpose="日记消息归组修复",
            )
            if not repaired_grouping:
                return "", []
            try:
                source_groups = parse_message_groups(repaired_grouping, chat_text)
            except DiaryCompositionError as repair_exc:
                print(
                    f"  [错误] 消息归组再次验证失败，本次不保存：{repair_exc}"
                )
                return "", []
        _save_checkpoint(grouping_checkpoint, source_groups)
    print(
        f"  消息归组验证通过：{len(source_groups)}组覆盖全部编号消息"
    )

    events: list[dict[str, Any]] = []
    event_batch_size = 4
    for batch_offset in range(0, len(source_groups), event_batch_size):
        batch_groups = source_groups[batch_offset : batch_offset + event_batch_size]
        group_start_index = batch_offset + 1
        grouped_chat = format_grouped_messages(
            batch_groups,
            chat_text,
            start_index=group_start_index,
        )
        batch_label = (
            f"{group_start_index}-{group_start_index + len(batch_groups) - 1}"
        )
        batch_checkpoint = _checkpoint_path(
            checkpoint_dir,
            f"events_{batch_label}",
            EVENT_INVENTORY_SYSTEM_PROMPT + grouped_chat,
        )
        cached_events = _load_checkpoint(batch_checkpoint)
        if isinstance(cached_events, list):
            try:
                batch_events = parse_event_inventory(
                    json.dumps({"events": cached_events}, ensure_ascii=False),
                    grouped_chat,
                    batch_groups,
                    event_id_offset=batch_offset,
                )
                print(f"  复用已验证事件批次断点：{batch_checkpoint.name}")
                events.extend(batch_events)
                continue
            except DiaryCompositionError:
                pass
        inventory_text = call_llm_api(
            grouped_chat,
            date_label,
            system_prompt=EVENT_INVENTORY_SYSTEM_PROMPT,
            user_content_override=_inventory_request(
                date_label,
                grouped_chat,
                batch_groups,
                group_start_index,
            ),
            max_tokens=min(DIARY_MAX_TOKENS, 3000),
            purpose=f"日记事件整理G{batch_label}",
        )
        if not inventory_text:
            return "", []
        try:
            batch_events = parse_event_inventory(
                inventory_text,
                grouped_chat,
                batch_groups,
                event_id_offset=batch_offset,
            )
        except DiaryCompositionError as exc:
            print(f"  [警告] 事件清单首次验证失败，进行一次定向修复：{exc}")
            repaired = call_llm_api(
                grouped_chat,
                date_label,
                system_prompt=EVENT_INVENTORY_SYSTEM_PROMPT,
                user_content_override=_inventory_repair_request(
                    date_label,
                    grouped_chat,
                    batch_groups,
                    group_start_index,
                    inventory_text,
                    str(exc),
                ),
                max_tokens=min(DIARY_MAX_TOKENS, 3000),
                purpose=f"日记事件清单修复G{batch_label}",
            )
            if not repaired:
                return "", []
            try:
                batch_events = parse_event_inventory(
                    repaired,
                    grouped_chat,
                    batch_groups,
                    event_id_offset=batch_offset,
                )
            except DiaryCompositionError as repair_exc:
                print(
                    "  [错误] 事件清单再次验证失败，本次不保存："
                    f"{repair_exc}"
                )
                return "", []
        _save_checkpoint(batch_checkpoint, batch_events)
        events.extend(batch_events)

    importance_counts = {
        level: sum(event["importance"] == level for event in events)
        for level in ("major", "ordinary", "minor")
    }
    print(
        "  事件清单验证通过："
        f"共{len(events)}件，重要={importance_counts['major']}，"
        f"普通={importance_counts['ordinary']}，细小={importance_counts['minor']}"
    )

    event_ids_before_merge = [event["id"] for event in events]
    consolidation_input = _consolidation_source(events)
    consolidation_checkpoint = _checkpoint_path(
        checkpoint_dir,
        "consolidation",
        EVENT_CONSOLIDATION_SYSTEM_PROMPT + consolidation_input,
    )
    cached_clusters = _load_checkpoint(consolidation_checkpoint)
    clusters: list[dict[str, Any]]
    if isinstance(cached_clusters, list):
        consolidation_protocol = "\n".join(
            "\n".join(
                (
                    f"@@GROUP C{index:03d}",
                    f"events={','.join(cluster['event_ids'])}",
                    f"title={cluster['title']}",
                    "@@END",
                )
            )
            for index, cluster in enumerate(cached_clusters, start=1)
        ) + "\n@@COMPLETE"
        try:
            clusters = parse_event_consolidation(
                consolidation_protocol,
                event_ids_before_merge,
            )
            validate_consolidation_boundaries(events, clusters)
            print(f"  复用已验证叙事合并断点：{consolidation_checkpoint.name}")
        except (DiaryCompositionError, KeyError, TypeError):
            cached_clusters = None
    if not isinstance(cached_clusters, list):
        consolidation_text = call_llm_api(
            consolidation_input,
            date_label,
            system_prompt=EVENT_CONSOLIDATION_SYSTEM_PROMPT,
            user_content_override=_consolidation_request(date_label, events),
            max_tokens=min(DIARY_MAX_TOKENS, 1800),
            purpose="日记叙事合并",
        )
        if not consolidation_text:
            return "", events
        try:
            clusters = parse_event_consolidation(
                consolidation_text,
                event_ids_before_merge,
            )
            validate_consolidation_boundaries(events, clusters)
        except DiaryCompositionError as exc:
            print(f"  [警告] 叙事合并首次验证失败，进行一次定向修复：{exc}")
            repaired = call_llm_api(
                consolidation_input,
                date_label,
                system_prompt=EVENT_CONSOLIDATION_SYSTEM_PROMPT,
                user_content_override=_consolidation_repair_request(
                    date_label,
                    events,
                    consolidation_text,
                    str(exc),
                ),
                max_tokens=min(DIARY_MAX_TOKENS, 1800),
                purpose="日记叙事合并修复",
            )
            if not repaired:
                return "", events
            try:
                clusters = parse_event_consolidation(
                    repaired,
                    event_ids_before_merge,
                )
                validate_consolidation_boundaries(events, clusters)
            except DiaryCompositionError as repair_exc:
                print(f"  [错误] 叙事合并再次验证失败，本次不保存：{repair_exc}")
                return "", events
        _save_checkpoint(consolidation_checkpoint, clusters)
    events = consolidate_events(events, clusters)
    print(
        f"  叙事合并验证通过：{len(event_ids_before_merge)}个细粒度事件 → "
        f"{len(events)}个日记叙事单元"
    )

    writer_request = _writer_request(date_label, events)
    writer_checkpoint = _checkpoint_path(
        checkpoint_dir,
        "writer_draft",
        DIARY_WRITER_SYSTEM_PROMPT + writer_request,
    )
    cached_writer = _load_checkpoint(writer_checkpoint)
    writer_text = (
        cached_writer.get("text")
        if isinstance(cached_writer, dict)
        and isinstance(cached_writer.get("text"), str)
        else ""
    )
    if writer_text:
        print(f"  复用日记正文初稿断点：{writer_checkpoint.name}")
    else:
        writer_text = call_llm_api(
            chat_text,
            date_label,
            system_prompt=DIARY_WRITER_SYSTEM_PROMPT,
            user_content_override=writer_request,
            max_tokens=DIARY_MAX_TOKENS,
            purpose="日记正文",
        )
        if writer_text:
            _save_checkpoint(writer_checkpoint, {"text": writer_text})
    if not writer_text:
        return "", events
    event_ids = [event["id"] for event in events]
    exact_time_allowed_event_ids = {
        event["id"]
        for event in events
        if event.get("exact_time_is_significant", False)
    }
    writer_text = normalize_marked_diary_order(writer_text, event_ids)
    required_terms_by_event = _required_terms_for_events(events)
    _lower_characters, upper_characters = target_diary_characters(events)
    candidate = writer_text
    repairs_used = 0
    for repair_index in range(4):
        try:
            diary = parse_marked_diary(
                candidate,
                event_ids,
                max_characters=upper_characters,
                required_terms_by_event=required_terms_by_event,
                exact_time_allowed_event_ids=exact_time_allowed_event_ids,
            )
            break
        except DiaryCompositionError as exc:
            if repair_index >= 3:
                print(f"  [错误] 日记正文四次验证仍失败，本次不保存：{exc}")
                return "", events
            print(
                f"  [警告] 日记正文第{repair_index + 1}次验证失败，"
                f"进行定向修复：{exc}"
            )
            repair_request = _writer_repair_request(
                date_label,
                events,
                candidate,
                str(exc),
            )
            repair_checkpoint = _checkpoint_path(
                checkpoint_dir,
                f"writer_repair_{repair_index + 1}",
                DIARY_COMPRESSION_SYSTEM_PROMPT + repair_request,
            )
            cached_repair = _load_checkpoint(repair_checkpoint)
            repaired = (
                cached_repair.get("text")
                if isinstance(cached_repair, dict)
                and isinstance(cached_repair.get("text"), str)
                else ""
            )
            if repaired:
                print(f"  复用日记压缩修复断点：{repair_checkpoint.name}")
            else:
                repaired = call_llm_api(
                    chat_text,
                    date_label,
                    system_prompt=DIARY_COMPRESSION_SYSTEM_PROMPT,
                    user_content_override=repair_request,
                    max_tokens=min(DIARY_MAX_TOKENS, 4000),
                    purpose=f"日记压缩修复{repair_index + 1}",
                )
                if repaired:
                    _save_checkpoint(repair_checkpoint, {"text": repaired})
            if not repaired:
                return "", events
            candidate = normalize_marked_diary_order(repaired, event_ids)
            repairs_used += 1
    else:
        return "", events

    if repairs_used >= 2:
        audit_request = _language_audit_request(
            date_label,
            events,
            candidate,
            upper_characters,
        )
        audit_checkpoint = _checkpoint_path(
            checkpoint_dir,
            "writer_language_audit",
            DIARY_LANGUAGE_AUDIT_SYSTEM_PROMPT + audit_request,
        )
        cached_audit = _load_checkpoint(audit_checkpoint)
        audit_text = (
            cached_audit.get("text")
            if isinstance(cached_audit, dict)
            and isinstance(cached_audit.get("text"), str)
            else ""
        )
        if audit_text:
            print(f"  复用日记语言审计断点：{audit_checkpoint.name}")
        else:
            audit_text = call_llm_api(
                chat_text,
                date_label,
                system_prompt=DIARY_LANGUAGE_AUDIT_SYSTEM_PROMPT,
                user_content_override=audit_request,
                max_tokens=min(DIARY_MAX_TOKENS, 2000),
                purpose="日记语言审计",
            )
            if audit_text:
                _save_checkpoint(audit_checkpoint, {"text": audit_text})
        if not audit_text:
            return "", events
        try:
            audited_candidate = apply_language_audit(candidate, audit_text)
            diary = parse_marked_diary(
                audited_candidate,
                event_ids,
                max_characters=upper_characters,
                required_terms_by_event=required_terms_by_event,
                exact_time_allowed_event_ids=exact_time_allowed_event_ids,
            )
        except DiaryCompositionError as exc:
            print(f"  [错误] 日记语言审计未通过硬验证，本次不保存：{exc}")
            return "", events
        candidate = audited_candidate

    if reference_text and reference_text.strip():
        continuity_request = _continuity_audit_request(
            date_label,
            events,
            candidate,
            reference_text,
            upper_characters,
        )
        continuity_checkpoint = _checkpoint_path(
            checkpoint_dir,
            "writer_continuity_audit",
            DIARY_CONTINUITY_AUDIT_SYSTEM_PROMPT + continuity_request,
        )
        cached_continuity = _load_checkpoint(continuity_checkpoint)
        continuity_text = (
            cached_continuity.get("text")
            if isinstance(cached_continuity, dict)
            and isinstance(cached_continuity.get("text"), str)
            else ""
        )
        if continuity_text:
            print(
                "  复用日记连续性审计断点："
                f"{continuity_checkpoint.name}"
            )
        else:
            continuity_text = call_llm_api(
                chat_text,
                date_label,
                system_prompt=DIARY_CONTINUITY_AUDIT_SYSTEM_PROMPT,
                user_content_override=continuity_request,
                max_tokens=min(DIARY_MAX_TOKENS, 2000),
                purpose="日记连续性审计",
            )
            if continuity_text:
                _save_checkpoint(
                    continuity_checkpoint,
                    {"text": continuity_text},
                )
        if not continuity_text:
            return "", events
        try:
            candidate = apply_language_audit(
                candidate,
                continuity_text,
                max_issues=3,
                max_span_characters=60,
                minimum_good_ratio=0.5,
            )
            diary = parse_marked_diary(
                candidate,
                event_ids,
                max_characters=upper_characters,
                required_terms_by_event=required_terms_by_event,
                exact_time_allowed_event_ids=exact_time_allowed_event_ids,
            )
        except DiaryCompositionError as exc:
            print(f"  [错误] 日记连续性审计未通过硬验证，本次不保存：{exc}")
            return "", events
    return diary, events


# 兼容旧名称：如果其他脚本仍然调用 call_deepseek，也不会坏
def call_deepseek(chat_text: str, date_label: str) -> str:
    return call_llm_api(chat_text, date_label)


def save_diary(date_str: str, diary_text: str):
    """保存日记文本到文件。"""
    DIARY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DIARY_DIR / f"diary_{date_str}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(diary_text)
    print(f"  [完成] 日记已保存 → {out_path}")


def generate_for_date(target_date: datetime):
    """为 target_date 这一天生成日记（如果已存在则跳过）。"""
    date_str   = target_date.strftime("%Y-%m-%d")
    diary_path = DIARY_DIR / f"diary_{date_str}.txt"

    if diary_path.exists() and diary_path.stat().st_size > 0:
        print(f"  [跳过] {date_str} 的日记已存在")
        return

    print(f"\n正在生成 {date_str} 的日记……")
    start, end = get_day_range(target_date)
    messages   = load_messages_in_range(start, end)
    reference_context = build_diary_reference_context(target_date)
    source_context = build_diary_source_context(
        target_date,
        messages,
        reference_context=reference_context,
    )

    print(f"  找到 {len(messages)} 条消息")
    diary_text = call_llm_api(
        format_for_llm(messages),
        date_str,
        context_text=source_context,
        purpose="日记正文",
    )

    if diary_text:
        save_diary(date_str, diary_text)
        print(f"  日记内容预览：\n  ────────────\n  {diary_text[:120]}……")
    else:
        print(f"  [失败] 日记生成失败，跳过 {date_str}")


def get_current_diary_date() -> datetime:
    """
    返回"当前应该属于哪一天日记"的日期。
    凌晨 0:00~2:59 算作前一天。
    """
    now = datetime.now()
    if now.hour < 3:
        return now - timedelta(days=1)
    return now


def batch_generate_all():
    """
    扫描 CHAT_HISTORY_DIR 里所有聊天文件，
    找出所有涉及的日期，为还没有日记的日期批量生成。
    """
    if not CHAT_HISTORY_DIR.exists():
        print("聊天记录目录不存在，退出。")
        return

    dates_seen = set()
    for json_file in CHAT_HISTORY_DIR.glob("*.json"):
        stem = json_file.stem
        try:
            file_time = datetime.strptime(stem[:19], "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            continue
        # 凌晨 0~2 点属于前一天的日记
        diary_date = file_time - timedelta(days=1) if file_time.hour < 3 else file_time
        dates_seen.add(diary_date.date())

    # 今天暂时不生成（今天还没结束）
    today = datetime.now().date()
    dates_seen.discard(today)

    if not dates_seen:
        print("没有找到需要生成日记的日期。")
        return

    print(f"找到 {len(dates_seen)} 个日期需要处理：{sorted(dates_seen)}")
    for d in sorted(dates_seen):
        generate_for_date(datetime.combine(d, datetime.min.time()))


def run_scheduler():
    """
    启动定时器，每天凌晨 3:00 自动生成昨天的日记。
    需要先安装 schedule 库：pip install schedule
    """
    try:
        import schedule
    except ImportError:
        print("[错误] 请先安装 schedule 库：pip install schedule")
        sys.exit(1)

    def daily_job():
        yesterday = get_current_diary_date() - timedelta(days=1)
        print(f"\n[定时任务触发] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        generate_for_date(yesterday)

    schedule.every().day.at("03:00").do(daily_job)

    # 启动时检查：只补生成已经完整结束的天的日记
    now = datetime.now()
    if now.hour < 3:
        last_complete_day = now - timedelta(days=2)
    else:
        last_complete_day = now - timedelta(days=1)
    generate_for_date(last_complete_day)

    print("=" * 50)
    print("凛祢日记生成器已启动（定时模式）")
    print(f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("将在每天凌晨 03:00 自动生成前一天的日记")
    print("按 Ctrl+C 停止")
    print("=" * 50)

    while True:
        schedule.run_pending()
        time.sleep(30)  # 每30秒检查一次


# ===================== 入口 =====================
if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        # 无参数 → 启动定时器
        run_scheduler()

    elif args[0] == "all":
        # 批量补生成所有历史日期的日记
        batch_generate_all()

    elif args[0] == "today":
        # 生成"当前日"的日记（调试用，今天还没结束所以内容不完整）
        target = get_current_diary_date()
        generate_for_date(target)

    else:
        # 指定日期，格式 YYYY-MM-DD
        try:
            target = datetime.strptime(args[0], "%Y-%m-%d")
            generate_for_date(target)
        except ValueError:
            print("日期格式不对，请用 YYYY-MM-DD，例如：python diary_generator.py 2026-05-19")
            sys.exit(1)
