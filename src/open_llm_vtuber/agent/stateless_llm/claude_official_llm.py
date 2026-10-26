"""Isolated official-route client. Production configuration does not select it."""

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from time import monotonic

from anthropic import AsyncAnthropic

from .claude_llm import AsyncLLM as ClaudeAsyncLLM
from .request_limiter import build_backend_key, limit_request_concurrency
from ...official_route.gate import OfficialContentBlocked, RequestGate

OFFICIAL_URL = "https://api.anthropic.com"


def split_system(
    text: str, stable_prefix: str | None = None
) -> tuple[list[dict], bool]:
    if (
        stable_prefix
        and text.startswith(stable_prefix)
        and len(stable_prefix) < len(text)
    ):
        position = len(stable_prefix)
    else:
        positions = [
            text.find(marker)
            for marker in ("【历史系统信号】", "【当前场景记忆协议】", "【聊天时间】")
            if text.find(marker) > 0
        ]
        position = min(positions) if positions else len(text)
    blocks = [
        dict(type="text", text=text[:position], cache_control={"type": "ephemeral"})
    ]
    if position < len(text):
        blocks.append(dict(type="text", text=text[position:]))
    return blocks, len(blocks) == 2


def _error_category(exc):
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return "authentication"
    if status == 429:
        return "rate_limit"
    # Inspect provider error metadata in memory; never persist its message.
    body = getattr(exc, "body", {})
    code = json.dumps(body).casefold() if isinstance(body, (dict, list)) else ""
    if status == 402 or any(
        word in code
        for word in (
            "credit_balance",
            "insufficient_quota",
            "billing_error",
            "credit balance",
            "insufficient credit",
        )
    ):
        return "quota"
    if status is not None and status >= 500:
        return "upstream_5xx"
    if "timeout" in type(exc).__name__.casefold():
        return "timeout"
    if "connection" in type(exc).__name__.casefold():
        return "connection"
    return (
        "invalid_response"
        if isinstance(exc, (ValueError, KeyError))
        else "provider_error"
    )


class AsyncLLM(ClaudeAsyncLLM):
    def __init__(
        self,
        *,
        gate: RequestGate,
        ledger_path: Path | str,
        model="claude-opus-4-6",
        thinking=False,
        stable_system_prefix=None,
        http_client=None,
        max_concurrent_requests=1,
        min_request_interval_seconds=0.0,
    ):
        key = os.environ.get("RINNE_OFFICIAL_ANTHROPIC_KEY", "")
        if not key:
            raise ValueError("official_key_missing")
        self.model = model
        self.base_url = OFFICIAL_URL
        self.system = None
        self.gate = gate
        self.thinking = bool(thinking)
        self.stable_system_prefix = stable_system_prefix
        self.ledger_path = Path(ledger_path)
        self.max_concurrent_requests = max(1, int(max_concurrent_requests))
        self.min_request_interval_seconds = max(
            0.0, float(min_request_interval_seconds)
        )
        self._forced_tool_name_once = None
        self.client = AsyncAnthropic(
            api_key=key,
            base_url=OFFICIAL_URL,
            auth_token="",
            max_retries=0,
            http_client=http_client,
        )
        # An empty string suppresses env lookup during construction, but SDK
        # 0.76 still emits a Bearer header for it. None after construction omits it.
        self.client.auth_token = None
        self._limiter_key = build_backend_key(
            provider_name="claude_official_llm", base_url=OFFICIAL_URL, api_key=key
        )

    def _messages(self, messages):
        if any(message.get("role") == "system" for message in messages):
            raise ValueError("mid_conversation_system_unsupported")
        converted = [
            deepcopy(self._convert_message_format(message)) for message in messages
        ]
        if converted and converted[-1].get("role") == "assistant":
            raise ValueError("assistant_prefill_unsupported")
        if len(converted) > 1:
            previous = converted[-2]
            if isinstance(previous.get("content"), str):
                previous["content"] = [dict(type="text", text=previous["content"])]
            for block in reversed(previous.get("content", [])):
                if block.get("type") in ("text", "tool_use"):
                    block["cache_control"] = {"type": "ephemeral"}
                    break
        return converted

    async def chat_completion(self, messages, system=None, tools=None):
        started = monotonic()
        row = dict(
            time=datetime.now(timezone.utc).isoformat(),
            group="official_thinking" if self.thinking else "official",
            model=self.model,
            thinking=self.thinking,
            gate="not_checked",
            input_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            output_tokens=0,
            estimated_usd=0.0,
            ttft_seconds=None,
            total_seconds=0.0,
            stop_reason=None,
            error_category="",
            forced_tool_changed_to_auto=False,
        )
        try:
            system_blocks, did_split = split_system(
                system if system is not None else (self.system or ""),
                self.stable_system_prefix,
            )
            row["system_split"] = did_split
            payload = dict(
                model=self.model,
                max_tokens=16000 if self.thinking else 4096,
                system=system_blocks,
                messages=self._messages(messages),
            )
            if tools:
                payload["tools"] = deepcopy(tools)
            forced = self._forced_tool_name_once
            self._forced_tool_name_once = None
            if forced and tools:
                payload["tool_choice"] = (
                    {"type": "auto"}
                    if self.thinking
                    else {"type": "tool", "name": forced}
                )
                row["forced_tool_changed_to_auto"] = self.thinking
            if self.thinking:
                payload["extra_body"] = {"thinking": {"type": "adaptive"}}
            gate_result = await self.gate.require(payload)
            row["gate"] = "allowed"
            row["checked_lines"] = gate_result.checked_lines
            async with limit_request_concurrency(
                self._limiter_key,
                self.max_concurrent_requests,
                self.min_request_interval_seconds,
            ):
                async with self.client.messages.stream(**payload) as stream:
                    async for event in stream:
                        if event.type == "message_start":
                            usage = event.message.usage.model_dump(exclude_none=True)
                            for name in (
                                "input_tokens",
                                "cache_creation_input_tokens",
                                "cache_read_input_tokens",
                                "output_tokens",
                            ):
                                row[name] = usage.get(name, 0) or 0
                            yield dict(
                                type="message_start",
                                data=event.message.model_dump(exclude_none=True),
                            )
                        elif (
                            event.type == "content_block_start"
                            and event.content_block.type == "tool_use"
                        ):
                            block = event.content_block
                            yield dict(
                                type="tool_use_start",
                                data=dict(
                                    id=block.id,
                                    name=block.name,
                                    input=None,
                                    index=event.index,
                                ),
                            )
                        elif (
                            event.type == "content_block_delta"
                            and event.delta.type == "text_delta"
                        ):
                            if row["ttft_seconds"] is None:
                                row["ttft_seconds"] = round(monotonic() - started, 6)
                            yield dict(type="text_delta", text=event.delta.text)
                        elif event.type == "content_block_stop":
                            block = stream.current_message_snapshot.content[
                                event.index
                            ].model_dump(exclude_none=True)
                            if block["type"] in ("thinking", "redacted_thinking"):
                                self.gate.trust_official_thinking(block)
                                yield dict(type="thinking_complete", data=block)
                            elif block["type"] == "tool_use":
                                yield dict(type="tool_use_complete", data=block)
                        elif event.type == "message_delta":
                            row["stop_reason"] = event.delta.stop_reason
                            usage = event.usage.model_dump(exclude_none=True)
                            for name in (
                                "input_tokens",
                                "cache_creation_input_tokens",
                                "cache_read_input_tokens",
                                "output_tokens",
                            ):
                                if usage.get(name) is not None:
                                    row[name] = usage[name]
                            yield dict(
                                type="message_delta",
                                data=dict(
                                    delta=event.delta.model_dump(exclude_none=True),
                                    usage=usage,
                                ),
                            )
                        elif event.type == "message_stop":
                            if row["stop_reason"] == "refusal":
                                row["error_category"] = "refusal"
                            yield dict(type="message_stop")
        except OfficialContentBlocked as exc:
            row["gate"] = "blocked"
            row["blocked"] = exc.result.blocked
            row["error_category"] = "content_blocked"
            raise
        except asyncio.CancelledError:
            row["error_category"] = "cancelled"
            raise
        except Exception as exc:
            row["error_category"] = _error_category(exc)
            yield dict(type="error", message=row["error_category"])
            raise
        finally:
            row["total_seconds"] = round(monotonic() - started, 6)
            row["estimated_usd"] = (
                row["input_tokens"] * 5
                + row["cache_creation_input_tokens"] * 6.25
                + row["cache_read_input_tokens"] * 0.5
                + row["output_tokens"] * 25
            ) / 1e6
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
