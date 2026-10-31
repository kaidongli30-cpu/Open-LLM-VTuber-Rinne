"""Isolated official-route client. Production configuration does not select it."""

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import monotonic

from anthropic import AsyncAnthropic

from .claude_llm import AsyncLLM as ClaudeAsyncLLM
from .request_limiter import build_backend_key, limit_request_concurrency
from ...official_route.gate import OfficialContentBlocked, RequestGate
from ...official_route.model_profiles import get_profile, request_options
from ...official_route.record_storage import append_row

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
        effort=None,
    ):
        key = os.environ.get("RINNE_OFFICIAL_ANTHROPIC_KEY", "")
        if not key:
            raise ValueError("official_key_missing")
        self.model = model
        self.base_url = OFFICIAL_URL
        self.system = None
        self.gate = gate
        self.thinking = bool(thinking)
        self.effort = effort
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
        # Anchor the history breakpoint before the current user question.
        # Tool-result continuations must not move it and rewrite the already
        # checked history prefix (including cache metadata) mid-turn.
        question_index = next(
            (
                index
                for index in range(len(converted) - 1, -1, -1)
                if converted[index].get("role") == "user"
                and not (
                    isinstance(converted[index].get("content"), list)
                    and any(
                        block.get("type") == "tool_result"
                        for block in converted[index]["content"]
                    )
                )
            ),
            0,
        )
        if question_index > 0:
            previous = converted[question_index - 1]
            if isinstance(previous.get("content"), str):
                previous["content"] = [dict(type="text", text=previous["content"])]
            for block in reversed(previous.get("content", [])):
                if block.get("type") in ("text", "tool_use"):
                    block["cache_control"] = {"type": "ephemeral"}
                    break
        return converted

    def _write_ledger(self, row):
        """Small override point for runtime-only, sanitized telemetry."""
        append_row(self.ledger_path, row, lock_name=".state.lock")

    def build_request(self, messages, system=None, tools=None) -> dict:
        """Build exactly the SDK payload used by chat_completion, without I/O.

        Does not consume force_tool_once or mutate caller-owned messages/tools;
        route preflight can call this repeatedly before the final gate check.
        Thinking is merged into the HTTP body by SDK 0.76.0 via extra_body.
        """
        options = request_options(
            self.model,
            self.thinking,
            self._forced_tool_name_once,
            effort=getattr(self, "effort", None),
        )
        if not tools:
            options.pop("tool_choice", None)
        system_blocks, _ = split_system(
            system if system is not None else (self.system or ""),
            self.stable_system_prefix,
        )
        payload = dict(
            system=system_blocks,
            messages=self._messages(messages),
            **options,
        )
        if tools:
            payload["tools"] = deepcopy(tools)
        return payload

    def build_wire_request(self, messages, system=None, tools=None):
        """Exact JSON body after SDK extra_body merge and stream insertion."""
        return self.wire_payload(self.build_request(messages, system, tools))

    @staticmethod
    def wire_payload(payload):
        result = deepcopy(payload)
        result.update(result.pop("extra_body", {}))
        result["stream"] = True
        return result

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
            request_sha256=None,
            gate_lines=0,
            latency_category="not_started",
        )
        try:
            payload = self.build_request(messages, system=system, tools=tools)
            profile = get_profile(payload["model"])
            row["model"] = payload["model"]
            row["thinking"] = profile.effective_thinking(self.thinking)
            row["group"] = "official_thinking" if row["thinking"] else "official"
            row["system_split"] = len(payload["system"]) == 2
            forced = self._forced_tool_name_once
            self._forced_tool_name_once = None
            row["forced_tool_changed_to_auto"] = bool(
                forced and tools and payload.get("tool_choice") == {"type": "auto"}
            )
            row["request_sha256"] = hashlib.sha256(
                json.dumps(
                    self.wire_payload(payload),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            gate_result = await self.gate.require(self.wire_payload(payload))
            row["gate"] = "allowed"
            row["checked_lines"] = gate_result.checked_lines
            row["gate_lines"] = gate_result.checked_lines
            row["latency_category"] = "waiting_first_text"
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
                                row["latency_category"] = "first_text"
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
            row["gate_lines"] = exc.result.checked_lines
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
            # No payload (e.g. unknown model) means no HTTP and zero usage.
            if row["request_sha256"] is not None:
                row["estimated_usd"] = get_profile(row["model"]).usage_cost(row)
            self._write_ledger(row)
