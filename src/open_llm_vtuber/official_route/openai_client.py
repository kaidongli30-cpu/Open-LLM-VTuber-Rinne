"""Responses transport for the existing guarded Claude request pipeline.

Only the transport and sanitized accounting differ. In particular, the inherited
chat_completion, preparation, tool projection, budgets and final require run
unchanged. No trial helpers or real chat-history paths are used here.
"""

from __future__ import annotations

from copy import deepcopy
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
from openai import AsyncOpenAI

from ..agent.stateless_llm.request_limiter import build_backend_key
from . import model_profiles
from .content_check import normalized_lines
from .record_storage import append_row
from .runtime_client import (
    GuardedOfficialClient,
    OfficialRouteFailure,
    _BudgetedGate,
    _REQUEST_CONTEXT,
)

OPENAI_URL = "https://api.openai.com/v1"
EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
_HTTP_BODY = ContextVar("official_openai_http_body", default=None)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def request_keyword_hit(value, checker, *, field_name=False):
    """Scan every key/value, preserving exact reviewed memory-line identities.

    Serializing the whole request first destroys those identities (JSON wraps
    and escapes them). The preceding gate and subset audit still apply to the
    complete payload; keys never receive the memory-line exception.
    """
    if isinstance(value, str):
        if field_name:
            raw = getattr(checker, "first", checker)
            if getattr(checker, "single_safety_pass", False):
                # Field names remain protocol data, not conversational prose.
                folded = value.casefold()
                return any(word in folded for word in raw.keywords)
            return raw.keyword_hit(value)
        return any(checker.keyword_hit(line) for line in normalized_lines(value))
    if isinstance(value, dict):
        return any(
            request_keyword_hit(key, checker, field_name=True)
            or request_keyword_hit(child, checker)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(request_keyword_hit(child, checker) for child in value)
    return False


def _sha(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def read_openai_api_key():
    """Dedicated process variable, then HKCU; never set global SDK variables."""
    value = os.environ.get("RINNE_OPENAI_API_KEY", "")
    if not value and os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value = winreg.QueryValueEx(key, "RINNE_OPENAI_API_KEY")[0]
        except (ImportError, OSError):
            value = ""
    return value if isinstance(value, str) else ""


def make_openai_sdk(api_key, *, http_client=None):
    http = http_client or httpx.AsyncClient(
        trust_env=False, timeout=300, follow_redirects=False
    )
    # Empty values suppress SDK 2.15.0's environment lookups. None afterward
    # suppresses organization/project headers entirely.
    sdk = AsyncOpenAI(
        api_key=api_key,
        base_url=OPENAI_URL,
        organization="",
        project="",
        webhook_secret="",
        max_retries=0,
        timeout=300,
        http_client=http,
    )
    sdk.organization = sdk.project = sdk.webhook_secret = None

    async def inspect(request):
        expected = _HTTP_BODY.get()
        if (
            expected is None
            or request.method != "POST"
            or str(request.url) != OPENAI_URL + "/responses"
            or json.loads(request.content) != expected
            or "openai-organization" in request.headers
            or "openai-project" in request.headers
        ):
            _fail("gate_blocked")

    http.event_hooks["request"].append(inspect)
    return sdk


def get_openai_profile(model):
    return model_profiles.get_openai_profile(model)


def _fail(category):
    context = _REQUEST_CONTEXT.get()
    if context is not None:
        context["failure"] = category
    raise OfficialRouteFailure(category)


def _metrics():
    context = _REQUEST_CONTEXT.get()
    if context is None:
        raise OfficialRouteFailure("gate_blocked")
    return context.setdefault("openai", {})


class _Box:
    """Only the small Anthropic event surface consumed by the native client."""

    def __init__(self, **values):
        self.values = values

    def __getattr__(self, key):
        value = self.values[key]
        if isinstance(value, dict):
            return _Box(**value)
        if isinstance(value, list):
            return [_Box(**v) if isinstance(v, dict) else v for v in value]
        return value

    def model_dump(self, **unused):
        return deepcopy(self.values)


def _plain(value):
    return (
        deepcopy(value)
        if isinstance(value, dict)
        else value.model_dump(mode="json", exclude_none=True)
    )


def _strip_cache(blocks):
    # Strip only protocol metadata, never keys inside tool arguments.
    return [
        {k: deepcopy(v) for k, v in b.items() if k != "cache_control"} for b in blocks
    ]


class ReplayStore:
    def __init__(self):
        self.outputs = {}

    def register(self, blocks, output):
        self.outputs[_sha(_strip_cache(blocks))] = deepcopy(output)

    def replay(self, blocks):
        return deepcopy(self.outputs.get(_sha(_strip_cache(blocks))))


def convert_request(claude, *, model, effort, replays):
    """Convert exactly one just-required wire body; unsupported data fails closed."""
    if effort is not None and effort not in EFFORTS:
        raise ValueError("unsupported_effort")
    result = dict(
        model=model,
        store=False,
        stream=True,
        include=["reasoning.encrypted_content"],
        max_output_tokens=32000,
        service_tier="default",
        instructions="\n".join(b["text"] for b in claude["system"]),
        input=[],
    )
    if effort is not None:
        result["reasoning"] = {"effort": effort}
    for message in claude["messages"]:
        role, blocks = message["role"], message["content"]
        if role not in ("user", "assistant"):
            raise ValueError("unsupported_role")
        if isinstance(blocks, str):
            blocks = [dict(type="text", text=blocks)]
        replay = replays.replay(blocks) if role == "assistant" else None
        if replay is not None:
            result["input"].extend(replay)
            continue
        for block in blocks:
            kind = block["type"]
            if kind == "text" and isinstance(block["text"], str):
                result["input"].append(dict(role=role, content=block["text"]))
            elif kind == "tool_use" and role == "assistant":
                result["input"].append(
                    dict(
                        type="function_call",
                        call_id=block["id"],
                        name=block["name"],
                        arguments=_canonical(block["input"]),
                    )
                )
            elif kind == "tool_result" and role == "user":
                output = block.get("content", "")
                if isinstance(output, list):
                    if any(
                        not isinstance(b, dict)
                        or b.get("type") != "text"
                        or set(b) - {"type", "text", "cache_control"}
                        or not isinstance(b.get("text"), str)
                        for b in output
                    ):
                        raise ValueError("unsupported_tool_result")
                    output = "\n".join(b["text"] for b in output)
                if not isinstance(output, str):
                    raise ValueError("unsupported_tool_result")
                result["input"].append(
                    dict(
                        type="function_call_output",
                        call_id=block["tool_use_id"],
                        output=output,
                    )
                )
            else:
                raise ValueError("unsupported_or_untrusted_block")
    if claude.get("tools"):
        result["tools"] = [
            dict(
                type="function",
                name=t["name"],
                description=t.get("description", ""),
                parameters=deepcopy(t["input_schema"]),
                strict=False,
            )
            for t in claude["tools"]
        ]
        choice = claude.get("tool_choice", {"type": "auto"})
        if choice["type"] == "auto":
            result["tool_choice"] = "auto"
        elif choice["type"] == "tool":
            result["tool_choice"] = dict(type="function", name=choice["name"])
        else:
            raise ValueError("unsupported_tool_choice")
    return result


def _lines(value, path=(), *, responses=False, model=None, authorized=()):
    """Exact lines, with exemptions restricted to exact protocol paths/values.

    JSON-encoded function arguments are decoded before comparison. Arbitrary
    schema/argument keys also count as text; whitespace is never normalized.
    """
    if isinstance(value, str):
        exempt = False
        if responses:
            exempt = (
                (path == ("model",) and value == model)
                or (path == ("reasoning", "effort") and value in EFFORTS)
                or (path == ("service_tier",) and value == "default")
                or (
                    len(path) == 2
                    and path[0] == "include"
                    and value == "reasoning.encrypted_content"
                )
                or (path == ("tool_choice",) and value == "auto")
                or (path == ("tool_choice", "type") and value == "function")
                or (
                    len(path) == 3
                    and path[0] == "tools"
                    and path[2] == "type"
                    and value == "function"
                )
                or (
                    len(path) == 3
                    and path[0] == "input"
                    and path[2] == "role"
                    and value in ("user", "assistant")
                )
                or (
                    len(path) == 3
                    and path[0] == "input"
                    and path[2] == "type"
                    and value
                    in ("message", "reasoning", "function_call", "function_call_output")
                )
                or (
                    len(path) == 5
                    and path[0] == "input"
                    and path[2] == "content"
                    and path[4] == "type"
                    and value == "output_text"
                )
                or path in authorized
            )
            if len(path) == 3 and path[0] == "input" and path[2] == "arguments":
                yield from _lines(json.loads(value))
                return
        if not exempt:
            yield from (line for line in value.splitlines() if line)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _lines(
                item,
                (*path, i),
                responses=responses,
                model=model,
                authorized=authorized,
            )
    elif isinstance(value, dict):
        for key, item in value.items():
            # Protocol field names are structure. Schema keys have no exemption.
            if not responses or "parameters" in path:
                yield from str(key).splitlines()
            yield from _lines(
                item,
                (*path, key),
                responses=responses,
                model=model,
                authorized=authorized,
            )


def subset_check(claude, responses, replays, *, model="gpt-6.1-sol"):
    authorized = set()
    known = [item for output in replays.outputs.values() for item in output]
    for i, item in enumerate(responses.get("input", [])):
        if item in known:
            for key in ("id", "status", "phase"):
                if key in item:
                    authorized.add(("input", i, key))
    source = list(_lines(claude))
    sent = list(_lines(responses, responses=True, model=model, authorized=authorized))
    allowed = set(source)
    missing = sum(line not in allowed for line in sent)
    return dict(
        passed=not missing,
        missing_lines=missing,
        claude_text_lines=len(source),
        openai_text_lines=len(sent),
    )


class _PinnedGate(_BudgetedGate):
    async def require(self, payload):
        result = await super().require(payload)
        _REQUEST_CONTEXT.get()["openai_approved_wire"] = deepcopy(payload)
        return result


class GuardedOpenAIClient(GuardedOfficialClient):
    def __init__(
        self,
        *,
        gate,
        ledger_path,
        checker=None,
        model="claude-opus-4-6",
        thinking=True,
        openai_model="gpt-6.1-sol",
        provider_model=None,
        effort="high",
        first_text_timeout_seconds=40,
        stable_system_prefix=None,
        http_client=None,
        sdk=None,
        max_concurrent_requests=1,
        min_request_interval_seconds=0.0,
        check_budget_seconds=20.0,
        foreground_deadline=None,
    ):
        if provider_model is not None:
            if openai_model != "gpt-6.1-sol" and openai_model != provider_model:
                raise ValueError("conflicting_provider_model")
            openai_model = provider_model
        profile = get_openai_profile(openai_model)
        if effort is not None and effort not in EFFORTS:
            raise ValueError("unsupported_effort")
        if hasattr(profile, "validate_effort"):
            profile.validate_effort(effort)
        # The native constructor requires an Anthropic key and creates an unused
        # Anthropic SDK. Initialize its state locally instead of altering env or
        # constructing a second provider. All its methods remain inherited.
        self.checker = checker if checker is not None else gate.checker
        self.check_budget_seconds = max(0.0, float(check_budget_seconds))
        self._remaining_check_seconds = self.check_budget_seconds
        self._foreground_deadline = foreground_deadline
        self.first_text_timeout_seconds = max(0.0, float(first_text_timeout_seconds))
        self.last_request = None
        self.usage_rows = []
        self._schema_checked = set()
        self._bound_view = None
        self._raw_prefix, self._safe_prefix, self.check_audit = [], [], []
        self.model, self.thinking = model, bool(thinking)
        model_profiles.get_profile(model)
        self.system, self.stable_system_prefix = None, stable_system_prefix
        self.provider_model, self.openai_effort = openai_model, effort
        # Reserve the native serializer's effort attribute for Claude. Its null
        # default preserves the approved request if the parent gains that option.
        self.effort = None
        # Always use the independent ledger, even if a caller supplies usage.jsonl.
        self.ledger_path = Path(ledger_path).with_name("openai_usage.jsonl")
        self.max_concurrent_requests = max(1, int(max_concurrent_requests))
        self.min_request_interval_seconds = max(
            0.0, float(min_request_interval_seconds)
        )
        self._forced_tool_name_once = None
        self.gate = _PinnedGate(self, gate)
        gate.cache_only, gate.model = True, model
        self.replays = ReplayStore()
        key = "injected-test-sdk" if sdk is not None else read_openai_api_key()
        if not key:
            raise ValueError("official_key_missing")
        self.sdk = (
            sdk if sdk is not None else make_openai_sdk(key, http_client=http_client)
        )
        self.base_url = OPENAI_URL
        self._limiter_key = build_backend_key(
            provider_name="official_openai", base_url=OPENAI_URL, api_key=key
        )
        self.client = SimpleNamespace(
            messages=SimpleNamespace(stream=lambda **p: _ResponsesTransport(self, p)),
            close=self.close,
        )

    async def close(self):
        await self.sdk.close()

    def _write_ledger(self, row):
        context = _REQUEST_CONTEXT.get()
        if context is not None:
            context["ledger_written"] = True
            if not context.get("flushing_ledger"):
                context["pending_row"] = deepcopy(row)
                return
        meta = (context or {}).get("openai", {})
        # Allowlist only numeric/category/hash metadata, never provider prose,
        # request/reply bodies, encrypted reasoning or credentials.
        safe = dict(
            time=row.get("time", datetime.now(timezone.utc).isoformat()),
            provider="openai",
            model=self.provider_model,
            effort=self.openai_effort,
            gate=row.get("gate", "not_dispatched"),
            gate_lines=row.get("gate_lines", 0),
            line_subset=meta.get("line_subset", False),
            missing_lines=meta.get("missing_lines", 0),
            keyword_hits=meta.get("keyword_hits"),
            request_sha256=meta.get("request_sha256"),
            claude_request_sha256=meta.get("claude_request_sha256"),
            input_tokens=meta.get("input_tokens", 0),
            cached_tokens=meta.get("cached_tokens", 0),
            cache_write_tokens=meta.get("cache_write_tokens", 0),
            output_tokens=meta.get("output_tokens", 0),
            reasoning_tokens=meta.get("reasoning_tokens", 0),
            estimated_usd=meta.get("estimated_usd", 0.0),
            usage_unknown=meta.get("usage_unknown", bool(meta.get("dispatched"))),
            dispatched=meta.get("dispatched", False),
            ttft_seconds=row.get("ttft_seconds"),
            total_seconds=row.get("total_seconds", 0.0),
            check_seconds=(context or {}).get("check_seconds", 0.0),
            stop_reason=meta.get("stop_reason", row.get("stop_reason")),
            error_category=(context or {}).get("failure")
            or row.get("error_category", ""),
            emitted_text=bool((context or {}).get("emitted_text")),
            service_tier="default",
        )
        safe["latency_category"] = (
            "error_after_text"
            if safe["error_category"] and safe["emitted_text"]
            else "timeout_before_text"
            if safe["error_category"] == "timeout"
            else "error_before_text"
            if safe["error_category"]
            else "first_text"
            if safe["ttft_seconds"] is not None
            else "completed_without_text"
        )
        self.last_request = deepcopy(safe)
        self.usage_rows.append(deepcopy(safe))
        append_row(self.ledger_path, safe, lock_name=".state.lock")


class _ResponsesTransport:
    def __init__(self, owner, payload):
        self.owner, self.payload = owner, deepcopy(payload)
        self.stream = self.iterator = None
        self.current_message_snapshot = _Box(content=[])

    async def __aenter__(self):
        o, meta = self.owner, _metrics()
        wire = o.wire_payload(self.payload)
        # Native AsyncLLM required this same wire immediately before the limiter.
        # Do not reset its first-text clock or spend another checking allocation.
        if wire != _REQUEST_CONTEXT.get().get("openai_approved_wire"):
            _fail("gate_blocked")
        meta["claude_request_sha256"] = _sha(wire)
        try:
            body = convert_request(
                wire, model=o.provider_model, effort=o.openai_effort, replays=o.replays
            )
            audit = subset_check(wire, body, o.replays, model=o.provider_model)
        except (ValueError, KeyError, TypeError):
            _fail("gate_blocked")
        meta.update(line_subset=audit["passed"], missing_lines=audit["missing_lines"])
        if not audit["passed"]:
            _fail("gate_blocked")
        meta["keyword_hits"] = int(request_keyword_hit(body, o.checker))
        if meta["keyword_hits"]:
            _fail("gate_blocked")
        meta["request_sha256"] = _sha(body)
        token = _HTTP_BODY.set(deepcopy(body))
        try:
            meta["dispatched"] = True
            self.stream = await o.sdk.responses.create(**body)
        except Exception as exc:
            self._error(exc)
            raise
        finally:
            _HTTP_BODY.reset(token)
        return self

    def _error(self, exc):
        if isinstance(exc, OfficialRouteFailure):
            _fail(exc.category)
        status = getattr(exc, "status_code", None)
        body = getattr(exc, "body", {})
        error = body.get("error", body) if isinstance(body, dict) else {}
        code = error.get("code") if isinstance(error, dict) else None
        category = (
            "authentication"
            if status in (401, 403)
            else "quota"
            if status == 402
            or code in ("insufficient_quota", "billing_error", "credit_balance")
            else "rate_limit"
            if status == 429 or code in ("rate_limit_exceeded", "rate_limit")
            else "upstream_5xx"
            if (status is not None and status >= 500) or code == "server_error"
            else "refusal"
            if code
            in ("content_policy_violation", "content_filter", "safety_violation")
            else "timeout"
            if isinstance(exc, TimeoutError)
            or "timeout" in type(exc).__name__.casefold()
            else "connection"
        )
        _fail(category)

    def __aiter__(self):
        self.iterator = self.events()
        return self.iterator

    def _usage(self, usage):
        meta = _metrics()
        meta["usage_unknown"] = True
        if not isinstance(usage, dict):
            _fail("connection")
        ins, outs = (
            usage.get("input_tokens_details") or {},
            usage.get("output_tokens_details") or {},
        )
        counts = dict(
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cached_tokens=ins.get("cached_tokens"),
            cache_write_tokens=ins.get("cache_write_tokens", 0),
            reasoning_tokens=outs.get("reasoning_tokens"),
        )
        if any(type(v) is not int or v < 0 for v in counts.values()):
            _fail("connection")
        if (
            counts["cached_tokens"] + counts["cache_write_tokens"]
            > counts["input_tokens"]
            or counts["reasoning_tokens"] > counts["output_tokens"]
        ):
            _fail("connection")
        meta.update(counts)
        meta["estimated_usd"] = get_openai_profile(
            self.owner.provider_model
        ).usage_cost(usage)
        meta["usage_unknown"] = False
        return dict(
            input_tokens=counts["input_tokens"], output_tokens=counts["output_tokens"]
        )

    async def events(self):
        o, meta, streamed = self.owner, _metrics(), ""
        try:
            yield _Box(type="message_start", message=dict(usage={}))
            terminal = None
            async for event in self.stream:
                raw = _plain(event)
                kind = raw["type"]
                if kind == "response.output_text.delta":
                    text = raw["delta"]
                    if not isinstance(text, str):
                        _fail("connection")
                    streamed += text
                    yield _Box(
                        type="content_block_delta",
                        index=0,
                        delta=dict(type="text_delta", text=text),
                    )
                elif kind in ("response.refusal.delta", "response.refusal.done"):
                    _fail("refusal")
                elif kind in (
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                ):
                    terminal = raw["response"]
                    break
                elif kind == "error":
                    self._error(SimpleNamespace(status_code=None, body=raw))
            if terminal is None:
                _fail("connection")
            meta["usage_unknown"] = True
            compat = (
                self._usage(terminal["usage"])
                if terminal.get("usage") is not None
                else None
            )
            if terminal.get("service_tier") not in (None, "default"):
                _fail("connection")
            if terminal.get("status") != "completed":
                reason = (terminal.get("incomplete_details") or {}).get("reason")
                meta["stop_reason"] = (
                    "max_output_tokens"
                    if reason == "max_output_tokens"
                    else "incomplete"
                )
                if terminal.get("error"):
                    self._error(
                        SimpleNamespace(status_code=None, body=terminal["error"])
                    )
                _fail("refusal" if reason == "content_filter" else "connection")
            if compat is None:
                _fail("connection")
            output, blocks = terminal.get("output", []), []
            for item in output:
                kind = item["type"]
                if kind == "reasoning":
                    if not item.get("encrypted_content") or item.get("summary"):
                        _fail("connection")
                    blocks.append(
                        dict(type="redacted_thinking", data=item["encrypted_content"])
                    )
                elif kind == "function_call":
                    args = json.loads(item["arguments"])
                    if not isinstance(args, dict):
                        _fail("connection")
                    blocks.append(
                        dict(
                            type="tool_use",
                            id=item["call_id"],
                            name=item["name"],
                            input=args,
                        )
                    )
                elif kind == "message":
                    for part in item["content"]:
                        if part["type"] == "refusal":
                            _fail("refusal")
                        if part["type"] != "output_text" or not isinstance(
                            part["text"], str
                        ):
                            _fail("connection")
                        if blocks and blocks[-1]["type"] == "text":
                            blocks[-1]["text"] += part["text"]
                        else:
                            blocks.append(dict(type="text", text=part["text"]))
                else:
                    _fail("connection")
            text = "".join(b["text"] for b in blocks if b["type"] == "text")
            if streamed and streamed != text:
                _fail("connection")
            if not blocks:
                _fail("connection")
            # The native tool loop groups all already-streamed text before the
            # completion-only reasoning/tool events. Key replay by what that
            # loop actually receives, while preserving original Responses items.
            replay_blocks = (
                [
                    dict(type="text", text=streamed),
                    *[b for b in blocks if b["type"] != "text"],
                ]
                if streamed
                else blocks
            )
            o.replays.register(replay_blocks, output)
            self.current_message_snapshot = _Box(content=blocks)
            for i, block in enumerate(blocks):
                if block["type"] == "text":
                    if not streamed:
                        yield _Box(
                            type="content_block_delta",
                            index=i,
                            delta=dict(type="text_delta", text=block["text"]),
                        )
                else:
                    if block["type"] == "tool_use":
                        yield _Box(
                            type="content_block_start", index=i, content_block=block
                        )
                    yield _Box(type="content_block_stop", index=i)
            meta["stop_reason"] = (
                "tool_use"
                if any(b["type"] == "tool_use" for b in blocks)
                else "end_turn"
            )
            yield _Box(
                type="message_delta",
                delta=dict(stop_reason=meta["stop_reason"]),
                usage=compat,
            )
            yield _Box(type="message_stop")
        except Exception as exc:
            self._error(exc)

    async def __aexit__(self, typ, value, tb):
        if self.iterator is not None:
            await self.iterator.aclose()
        if self.stream is not None:
            await self.stream.close()
