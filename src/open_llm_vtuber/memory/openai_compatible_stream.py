"""Safely collect text from OpenAI-compatible streaming chat completions."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

import requests


class StreamProtocolError(ValueError):
    """A streamed response ended with incomplete or invalid SSE data."""


class OutputTruncatedError(RuntimeError):
    """The provider stopped because the configured output limit was reached."""


class StreamStatus:
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
            raise StreamProtocolError(f"SSE第{event_index}个事件的JSON顶层不是对象")
        return event

    character_count = sum(len(part) for part in data_parts)
    reason = last_error.msg if last_error is not None else "未知解析错误"
    raise StreamProtocolError(
        f"SSE第{event_index}个事件不完整或格式错误"
        f"（累计{character_count}字符；{reason}）"
    )


def iter_sse_json_events(
    raw_lines: Iterable[str | bytes],
    *,
    status: StreamStatus | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield complete JSON events from standard and compatible SSE streams."""

    stream_status = status or StreamStatus()
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
        line = (
            raw_line.decode("utf-8", errors="replace")
            if isinstance(raw_line, bytes)
            else (raw_line or "")
        )
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
                data_parts.append(line)
                has_bare_continuation = True
                try:
                    event = decode_pending()
                except StreamProtocolError:
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
        except StreamProtocolError:
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


def request_chat_completion_text(
    *,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    output_label: str,
    output_token_limit: int,
    read_timeout_seconds: float,
    connect_timeout_seconds: float = 20,
    max_attempts: int = 2,
    retry_delay_seconds: float = 2,
    report: Callable[[str], None] = print,
) -> str:
    """Return only a complete response; never return a partial stream."""

    request_payload = dict(payload)
    request_payload["stream"] = True
    retryable_statuses = {408, 429, 500, 502, 503, 504}

    for attempt in range(1, max_attempts + 1):
        chunks: list[str] = []
        finish_reasons: list[str] = []
        stream_status: StreamStatus | None = None
        is_stream_response = False
        started_at = time.perf_counter()
        first_content_at: float | None = None
        try:
            with requests.post(
                url,
                json=request_payload,
                headers=dict(headers),
                stream=True,
                timeout=(connect_timeout_seconds, read_timeout_seconds),
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
                    stream_status = StreamStatus()
                    for event in iter_sse_json_events(
                        response.iter_lines(decode_unicode=False, delimiter=b"\n"),
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
                raise OutputTruncatedError(
                    f"服务商因达到{output_token_limit} token输出上限而停止"
                )
            unexpected_reasons = observed_reasons - successful_reasons
            if unexpected_reasons:
                raise StreamProtocolError(
                    "服务商返回非正常结束原因："
                    + ",".join(sorted(unexpected_reasons))
                )
            if (
                is_stream_response
                and stream_status is not None
                and not stream_status.saw_done
                and not (observed_reasons & successful_reasons)
            ):
                raise StreamProtocolError(
                    "SSE流结束时既没有[DONE]，也没有正常finish_reason"
                )

            text = "".join(chunks).strip()
            if not text:
                raise ValueError(f"接口完成响应，但没有返回{output_label}正文")

            total_seconds = time.perf_counter() - started_at
            first_seconds = (
                first_content_at - started_at
                if first_content_at is not None
                else total_seconds
            )
            report(
                f"  {output_label}流式响应完成："
                f"首段={first_seconds:.1f}秒，总计={total_seconds:.1f}秒，"
                f"尝试={attempt}/{max_attempts}"
            )
            return text
        except Exception as exc:
            if isinstance(exc, OutputTruncatedError):
                report(
                    f"  [错误] {output_label}达到输出上限，判定为不完整；"
                    f"本次不落盘，请提高上限后重新生成：{exc}"
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
            can_retry = attempt < max_attempts and retryable and not chunks
            if can_retry:
                report(
                    f"  [警告] {output_label}接口在返回正文前失败，"
                    f"{retry_delay_seconds:g}秒后自动重试一次：{exc}"
                )
                time.sleep(retry_delay_seconds)
                continue
            if chunks:
                report(
                    f"  [错误] {output_label}流式响应中途断开；"
                    "为避免保存残缺内容，本次不落盘，也不自动重复计费："
                    f"{exc}"
                )
            else:
                report(f"  [错误] {output_label}API调用失败：{exc}")
            return ""
    return ""
