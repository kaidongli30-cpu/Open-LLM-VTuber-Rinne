"""Checked official client; routing and whole-turn fallback belong to runtime."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
import json
import re
from time import monotonic

from ..agent.stateless_llm.claude_official_llm import (
    AsyncLLM as OfficialAsyncLLM,
    _error_category,
)
from .clean_view import (
    FIXED_ANCHORS,
    ProtectedContentBlocked,
    clean_background,
    fixed_wrapper_parts,
)
from .content_check import fingerprint
from .checks_deadline import is_pending, check_deadline, check_scope
from .gate import OfficialContentBlocked
from .program_text import schema_texts


class OfficialRouteFailure(RuntimeError):
    """Only a category crosses the runtime boundary, never provider prose."""

    def __init__(self, category: str, *, emitted_text: bool = False):
        self.category = category
        self.emitted_text = emitted_text
        super().__init__(category)


_REQUEST_CONTEXT = ContextVar("official_runtime_request", default=None)
_RAW_ERROR_PREFIXES = (
    "error calling the chat endpoint",
    "claude api error",
    "error code:",
)


class _BudgetedGate:
    def __init__(self, owner, delegate):
        self.owner = owner
        self.delegate = delegate

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    async def require(self, payload):
        context = _REQUEST_CONTEXT.get()
        try:
            result = await self.owner._checking(lambda: self.delegate.require(payload))
        except OfficialContentBlocked as error:
            category = "presend_unrecorded" if error.result.pending else "gate_blocked"
            if context is not None:
                context["failure"] = category
            raise OfficialRouteFailure(category) from None
        if context is not None:
            context["gate_finished"] = monotonic()
            context["gate_ready"].set()
        self.owner._approved_prefix = deepcopy(self.owner._safe_prefix)
        return result


class GuardedOfficialClient(OfficialAsyncLLM):
    """A ClaudeAsyncLLM subclass, usable by the existing native tool loop.

    One instance is intended for one routed conversation turn, including its
    memory-tool continuations. The 20s checking budget is shared across all
    cleaning/schema/gate work, independent of official first-text waiting.
    """

    def __init__(
        self,
        *,
        gate,
        ledger_path,
        checker=None,
        model="claude-opus-4-6",
        thinking=True,
        stable_system_prefix=None,
        http_client=None,
        max_concurrent_requests=1,
        min_request_interval_seconds=0.0,
        check_budget_seconds=20.0,
        foreground_deadline=None,
        first_text_timeout_seconds=None,
        effort=None,
    ):
        self.checker = checker if checker is not None else gate.checker
        self.check_budget_seconds = max(0.0, float(check_budget_seconds))
        self._remaining_check_seconds = self.check_budget_seconds
        self._foreground_deadline = foreground_deadline
        from .model_profiles import model_is_thinking

        effective_thinking = model_is_thinking(model, thinking)
        self.first_text_timeout_seconds = (
            (40.0 if effective_thinking else 20.0)
            if first_text_timeout_seconds is None
            else max(0.0, float(first_text_timeout_seconds))
        )
        self.last_request = None
        self.usage_rows = []
        self._schema_checked = set()
        self._bound_view = None
        self._raw_prefix = []
        self._safe_prefix = []
        self._approved_prefix = []
        self.check_audit = []
        super().__init__(
            gate=gate,
            ledger_path=ledger_path,
            model=model,
            thinking=thinking,
            stable_system_prefix=stable_system_prefix,
            http_client=http_client,
            max_concurrent_requests=max_concurrent_requests,
            min_request_interval_seconds=min_request_interval_seconds,
            effort=effort,
        )
        self.gate = _BudgetedGate(self, gate)
        gate.cache_only = True
        gate.model = model

    def bind_initial_view(self, messages, system, tools):
        """Pin the view checked by routing; tool continuations only append."""
        self._bound_view = (deepcopy(messages), system, deepcopy(tools))
        self._raw_prefix = deepcopy(messages)
        self._safe_prefix = deepcopy(messages)
        self._initial_request_pending = True

    def checked_tool_continuation(self):
        """Reuse only the final-gate-approved tool tail across providers."""
        if self._bound_view is None:
            return []
        initial, _, _ = self._bound_view
        tail = deepcopy(self._approved_prefix[len(initial) :])
        # Thinking signatures and encrypted reasoning belong to their provider.
        # Text and tool pairs still pass the other provider's complete gate.
        for message in tail:
            content = message.get("content")
            if isinstance(content, list):
                message["content"] = [
                    block
                    for block in content
                    if block.get("type") not in {"thinking", "redacted_thinking"}
                ]
        return [message for message in tail if message.get("content")]

    async def _prepare_bound(self, messages, system, tools):
        _, fixed_system, fixed_tools = self._bound_view
        allowed = {t["name"]: t for t in fixed_tools or []}
        current = tools or []
        valid_tools = len({t.get("name") for t in current}) == len(current) and all(
            allowed.get(t.get("name")) == t for t in current
        )
        if (
            system != fixed_system
            or not valid_tools
            or messages[: len(self._raw_prefix)] != self._raw_prefix
        ):
            raise OfficialRouteFailure("gate_blocked")
        additions = deepcopy(messages[len(self._raw_prefix) :])
        if self._uses_tool_projection():
            await self._prepare_tool_additions(additions)
        else:
            structures = []
            for message in additions:
                if message.get("role") not in ("user", "assistant"):
                    raise OfficialRouteFailure("gate_blocked")
                message["content"] = await self._clean_content(
                    message.get("content", ""), structures=structures
                )
            # Legacy/synthetic checkers retain their existing contracts.
            await self._check_structure(structures)
        prepared = [*deepcopy(self._safe_prefix), *additions]
        self._raw_prefix = deepcopy(messages)
        self._safe_prefix = deepcopy(prepared)
        return prepared, fixed_system, deepcopy(tools)

    def set_check_deadline(self, deadline):
        """Share the foreground deadline already started by runtime routing."""
        self._foreground_deadline = deadline

    def set_check_budget_seconds(self, seconds):
        """Set the unused part of runtime's foreground checking budget."""
        self._remaining_check_seconds = max(0.0, float(seconds))
        self._foreground_deadline = None

    @property
    def remaining_check_seconds(self):
        return self._remaining_check_seconds

    async def _checking(self, operation):
        started = monotonic()
        if self._foreground_deadline is not None:
            self._remaining_check_seconds = min(
                self._remaining_check_seconds,
                max(0.0, self._foreground_deadline - started),
            )
            # Official response waiting must not consume the checking budget.
            self._foreground_deadline = None
        context = _REQUEST_CONTEXT.get()
        try:
            if self._remaining_check_seconds <= 0:
                raise TimeoutError
            allocation = self._remaining_check_seconds
            result = await asyncio.wait_for(operation(), allocation)
            # wait_for cannot interrupt synchronous work that returns without
            # yielding. Its elapsed time still belongs to the same 20s budget;
            # never dispatch after a late checker/gate result slips through.
            if monotonic() - started > allocation:
                raise TimeoutError
            return result
        except (TimeoutError, asyncio.TimeoutError):
            if context is not None:
                context["failure"] = "check_failed"
            raise OfficialRouteFailure("check_failed") from None
        finally:
            elapsed = monotonic() - started
            self._remaining_check_seconds = max(
                0.0, self._remaining_check_seconds - elapsed
            )
            if context is not None:
                context["check_seconds"] += elapsed

    async def _clean_text(self, text, *, anchors=(), layer2=False):
        cleaned = await clean_background(
            text, self.checker, anchors=anchors, layer2=layer2
        )
        return cleaned.text

    async def _clean_system(self, text):
        # Preserve exact Layer2 prefix/suffix, even inside a composed system.
        pieces = []
        cursor = 0
        for match in re.finditer(
            r"【第二层记忆：截至 (\d{4}-\d{2}-\d{2}) 的用户当前个人背景】", text
        ):
            if match.start() < cursor:
                continue
            parts = fixed_wrapper_parts(match[1])
            if parts is None or not text.startswith(parts[0], match.start()):
                raise OfficialRouteFailure("content_blocked")
            closing = text.find(parts[1], match.start() + len(parts[0]))
            if closing < 0:
                raise OfficialRouteFailure("content_blocked")
            before = text[cursor : match.start()]
            anchors = tuple(anchor for anchor in FIXED_ANCHORS if anchor in before)
            pieces.append(await self._clean_text(before, anchors=anchors))
            end = closing + len(parts[1])
            pieces.append(
                await self._clean_text(text[match.start() : end], layer2=True)
            )
            cursor = end
        remainder = text[cursor:]
        anchors = tuple(anchor for anchor in FIXED_ANCHORS if anchor in remainder)
        pieces.append(await self._clean_text(remainder, anchors=anchors))
        return "".join(pieces)

    async def _check_structure(self, value):
        texts = list(
            dict.fromkeys(text for text in schema_texts(value) if text.strip())
        )
        if not texts:
            return
        records = await self.checker.check_many(texts, kind="line")
        if len(records) != len(texts) or any(is_pending(r) for r in records):
            raise OfficialRouteFailure("check_failed")
        if any(r.label != "A" for r in records):
            raise OfficialRouteFailure("gate_blocked")
        for text, record in zip(texts, records):
            self.checker.record_lines(text, record)

    def _uses_tool_projection(self):
        return getattr(self.checker, "runtime_two_pass", False) and hasattr(
            self.checker, "memory_checks"
        )

    def _collect_tool_content(
        self, owner, field, texts, slots, structures, *, tool_result=False
    ):
        """Collect mutable result slots without changing speech or protocol data."""
        content = owner.get(field, "")
        if isinstance(content, str):
            if tool_result:
                texts.append(content)
                slots.append((owner, field))
            return
        if not isinstance(content, list):
            raise OfficialRouteFailure("content_blocked")
        for block in content:
            if not isinstance(block, dict):
                raise OfficialRouteFailure("content_blocked")
            kind = block.get("type")
            if kind in ("thinking", "redacted_thinking"):
                # The final gate still requires exact provider-emitted blocks.
                continue
            if kind == "tool_result":
                self._collect_tool_content(
                    block, "content", texts, slots, structures, tool_result=True
                )
                structures.append(self.gate._block_metadata(block, kind))
            elif kind == "text":
                if not isinstance(block.get("text", ""), str):
                    raise OfficialRouteFailure("content_blocked")
                if tool_result:
                    texts.append(block.get("text", ""))
                    slots.append((block, "text"))
                structures.append(self.gate._block_metadata(block, kind))
            elif kind == "tool_use":
                structures.extend(
                    value if key.startswith("argument:") else {key: value}
                    for key, value in self.gate._block_metadata(block, kind).items()
                )
            else:
                raise OfficialRouteFailure("content_blocked")

    async def _prepare_tool_additions(self, additions):
        texts, slots, structures = [], [], []
        speech = []
        if getattr(self.checker, "single_safety_pass", False):
            # New speech in a tool continuation has not been seen by the
            # pre-turn pass. Check it once, independently of memory rewriting;
            # cached lines are reused and the final gate never buys a recheck.
            for message in additions:
                content = message.get("content", "")
                if isinstance(content, str):
                    speech.append(content)
                elif isinstance(content, list):
                    speech.extend(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
        for message in additions:
            if message.get("role") not in ("user", "assistant"):
                raise OfficialRouteFailure("gate_blocked")
            self._collect_tool_content(message, "content", texts, slots, structures)
            structures.append(
                {k: v for k, v in message.items() if k not in ("role", "content")}
            )
        if getattr(self.checker, "single_safety_pass", False):
            await self._check_structure(list(dict.fromkeys(speech)))
        structure_texts = list(
            dict.fromkeys(text for text in schema_texts(structures) if text.strip())
        )
        if not texts and not structure_texts:
            return
        # One Opus batch covers all new retrieved text AND immutable arguments/
        # metadata. Only returned text slots may be substituted. The memory
        # checker records exact safe lines for the final cache-only RequestGate.
        with check_scope("tool_results"):
            projected, status = await self.checker.memory_checks.prepare_tool_results(
                texts, structure_texts=structure_texts
            )
        if status != "ready":
            raise OfficialRouteFailure(
                "gate_blocked" if status == "blocked" else "check_failed"
            )
        if (
            not isinstance(projected, list)
            or len(projected) != len(slots)
            or any(not isinstance(text, str) for text in projected)
        ):
            raise OfficialRouteFailure("check_failed")
        for (owner, field), text in zip(slots, projected):
            owner[field] = text

    async def _clean_content(self, content, *, tool_result=False, structures=None):
        """Clean content blocks and check their metadata in one batch.

        With ``structures`` given, metadata is collected there instead and
        the caller checks everything it collected with one call.
        """
        if isinstance(content, str):
            # Only retrieved memory content may be substituted. User/assistant
            # speech stays intact and the final gate can reject it wholesale.
            if tool_result:
                with check_scope("tool_results"):
                    return await self._clean_text(content)
            return content
        if not isinstance(content, list):
            raise OfficialRouteFailure("content_blocked")
        pending = [] if structures is None else structures
        result = []
        for original in content:
            if not isinstance(original, dict):
                raise OfficialRouteFailure("content_blocked")
            block = deepcopy(original)
            kind = block.get("type")
            if kind in ("thinking", "redacted_thinking"):
                # Do not trust or edit caller-provided reasoning. RequestGate
                # accepts only an exact block it saw emitted by this provider.
                result.append(block)
                continue
            if kind == "tool_result":
                block["content"] = await self._clean_content(
                    block.get("content", ""), tool_result=True, structures=pending
                )
                pending.append(self.gate._block_metadata(block, kind))
            elif kind == "text":
                text = block.get("text", "")
                if not isinstance(text, str):
                    raise OfficialRouteFailure("content_blocked")
                if tool_result:
                    with check_scope("tool_results"):
                        block["text"] = await self._clean_text(text)
                # This branch has validated the exact protocol tag. Only
                # extra/variable metadata need classification; the final
                # RequestGate independently validates the complete block.
                pending.append(self.gate._block_metadata(block, kind))
            elif kind == "tool_use":
                pending.extend(
                    value if key.startswith("argument:") else {key: value}
                    for key, value in self.gate._block_metadata(block, kind).items()
                )
            else:
                # Unknown structures/media are not assumed to be safe JSON.
                raise OfficialRouteFailure("content_blocked")
            result.append(block)
        if structures is None:
            await self._check_structure(pending)
        return result

    async def _prepare(self, messages, system, tools):
        clean_system = await self._clean_system(
            system if system is not None else (self.system or "")
        )
        clean_messages = deepcopy(messages)
        if self._uses_tool_projection():
            await self._prepare_tool_additions(clean_messages)
        else:
            for message in clean_messages:
                if message.get("role") not in ("user", "assistant"):
                    raise OfficialRouteFailure("content_blocked")
                message["content"] = await self._clean_content(
                    message.get("content", "")
                )
                await self._check_structure(
                    {
                        key: value
                        for key, value in message.items()
                        if key not in ("role", "content")
                    }
                )
        clean_tools = deepcopy(tools)
        if clean_tools:
            digest = fingerprint(
                json.dumps(clean_tools, ensure_ascii=False, sort_keys=True)
            )
            if digest not in self._schema_checked:
                await self._check_structure(clean_tools)
                self._schema_checked.add(digest)
        return clean_messages, clean_system, clean_tools

    def _write_ledger(self, row):
        context = _REQUEST_CONTEXT.get()
        if context is not None:
            context["ledger_written"] = True
            if not context.get("flushing_ledger"):
                # Native generator teardown may precede wrapper validation
                # (e.g. an incomplete error prefix). Keep its usage, then write
                # exactly one row once this request's final category is known.
                context["pending_row"] = deepcopy(row)
                return
            if context.get("failure"):
                row["error_category"] = context["failure"]
            row["check_seconds"] = round(context["check_seconds"], 6)
            row["emitted_text"] = bool(context.get("emitted_text"))
        error = row.get("error_category")
        has_text = row.get("emitted_text", row.get("ttft_seconds") is not None)
        row["latency_category"] = (
            "timeout_before_text"
            if error == "timeout" and not has_text
            else "error_after_text"
            if error and has_text
            else "error_before_text"
            if error
            else "first_text"
            if row.get("ttft_seconds") is not None
            else "completed_without_text"
        )
        self.last_request = deepcopy(row)
        self.usage_rows.append(deepcopy(row))
        super()._write_ledger(row)

    async def chat_completion(self, messages, system=None, tools=None):
        started = monotonic()
        context = {
            "failure": "",
            "check_seconds": 0.0,
            "ledger_written": False,
            "gate_ready": asyncio.Event(),
            "gate_finished": None,
        }
        token = _REQUEST_CONTEXT.set(context)
        emitted_text = False
        stream = None
        text_prefix = ""
        seen_stop = False
        task = None
        gate_wait = None
        try:
            if self._bound_view is not None and self._initial_request_pending:
                # Routing checked this exact initial payload. No cleaning or
                # model calls are allowed between that gate and final send.
                prepared = await self._prepare_bound(messages, system, tools)
                self._initial_request_pending = False
            else:
                self._remaining_check_seconds = (
                    20.0
                    if self._bound_view is not None
                    else self._remaining_check_seconds
                )
                self._foreground_deadline = (
                    None if self._bound_view is not None else self._foreground_deadline
                )
                with (
                    check_deadline(self._remaining_check_seconds),
                    check_scope(
                        phase="tool_loop" if self._bound_view else "foreground",
                        audit=self.check_audit,
                    ),
                ):
                    prepare = (
                        self._prepare_bound
                        if self._bound_view is not None
                        else self._prepare
                    )
                    prepared = await self._checking(
                        lambda: prepare(messages, system, tools)
                    )
                    result = await self._checking(
                        lambda: self.gate.check(
                            self.build_wire_request(*prepared), cached_only=True
                        )
                        if getattr(self.checker, "single_safety_pass", False)
                        else self.gate.check(self.build_wire_request(*prepared))
                    )
                    if not result.allowed:
                        raise OfficialRouteFailure(
                            "check_failed" if result.pending else "gate_blocked"
                        )
            stream = super().chat_completion(*prepared).__aiter__()
            # Gate checking is still before dispatch. Start the official wait
            # clock after its budgeted work, not at foreground routing start.
            while True:
                if not emitted_text:
                    task = asyncio.create_task(anext(stream))
                    if context["gate_finished"] is None:
                        gate_wait = asyncio.create_task(context["gate_ready"].wait())
                        await asyncio.wait(
                            {task, gate_wait}, return_when=asyncio.FIRST_COMPLETED
                        )
                        if not gate_wait.done():
                            gate_wait.cancel()
                            with suppress(asyncio.CancelledError):
                                await gate_wait
                    remaining = (
                        self.first_text_timeout_seconds
                        - (monotonic() - context["gate_finished"])
                        if context["gate_finished"] is not None
                        else self.first_text_timeout_seconds
                    )
                    done, _ = await asyncio.wait({task}, timeout=max(0.0, remaining))
                    if not done:
                        context["failure"] = "timeout"
                        task.cancel()
                        with suppress(asyncio.CancelledError, Exception):
                            await task
                        raise OfficialRouteFailure("timeout")
                    try:
                        event = task.result()
                    except StopAsyncIteration:
                        break
                else:
                    try:
                        event = await anext(stream)
                    except StopAsyncIteration:
                        break
                if not isinstance(event, dict):
                    raise OfficialRouteFailure("connection", emitted_text=emitted_text)
                kind = event.get("type")
                if kind in ("error", "provider_error"):
                    # The native client's generic exception handler emits a
                    # provider_error event even when our budgeted gate raised
                    # OfficialRouteFailure(timeout). Preserve the authoritative
                    # local classification instead of relabeling it connection.
                    category = context.get("failure") or event.get(
                        "category", event.get("message", "")
                    )
                    category = (
                        category
                        if category
                        in {
                            "authentication",
                            "quota",
                            "connection",
                            "timeout",
                            "upstream_5xx",
                            "content_blocked",
                            "refusal",
                            "rate_limit",
                            "check_failed",
                            "gate_blocked",
                            "presend_unrecorded",
                        }
                        else "connection"
                    )
                    raise OfficialRouteFailure(category, emitted_text=emitted_text)
                if (
                    kind == "message_delta"
                    and event.get("data", {}).get("delta", {}).get("stop_reason")
                    == "refusal"
                ):
                    context["failure"] = "refusal"
                    raise OfficialRouteFailure("refusal", emitted_text=emitted_text)
                if kind == "text_delta":
                    value = event.get("text", "")
                    if not isinstance(value, str):
                        raise OfficialRouteFailure(
                            "connection", emitted_text=emitted_text
                        )
                    if not value:
                        continue
                    text_prefix += value
                    folded = text_prefix.lstrip().casefold()
                    if any(folded.startswith(prefix) for prefix in _RAW_ERROR_PREFIXES):
                        raise OfficialRouteFailure(
                            "connection", emitted_text=emitted_text
                        )
                    if any(prefix.startswith(folded) for prefix in _RAW_ERROR_PREFIXES):
                        continue
                    event = dict(type="text_delta", text=text_prefix)
                    text_prefix = ""
                    emitted_text = True
                    context["emitted_text"] = True
                # Keep native message/tool/thinking events for the tool loop.
                if kind == "message_stop":
                    seen_stop = True
                # BasicMemoryAgent closes streams through wait_for(aclose()),
                # which runs in a different task/context. Do not keep a token
                # bound across a yield: reset it before returning to the caller
                # and bind a fresh token in whichever task resumes/closes us.
                _REQUEST_CONTEXT.reset(token)
                token = None
                try:
                    yield event
                finally:
                    token = _REQUEST_CONTEXT.set(context)
            if text_prefix:
                raise OfficialRouteFailure("connection", emitted_text=emitted_text)
            if context.get("pending_row", {}).get("error_category") == "refusal":
                raise OfficialRouteFailure("refusal", emitted_text=emitted_text)
            if not seen_stop:
                raise OfficialRouteFailure("connection", emitted_text=emitted_text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            category = (
                exc.category
                if isinstance(exc, OfficialRouteFailure)
                else "check_failed"
                if isinstance(exc, OfficialContentBlocked) and exc.result.pending
                else "gate_blocked"
                if isinstance(exc, (OfficialContentBlocked, ProtectedContentBlocked))
                else _error_category(exc)
            )
            context["failure"] = category
            if stream is not None:
                await stream.aclose()
            if not context["ledger_written"]:
                self._write_ledger(
                    dict(
                        time=datetime.now(timezone.utc).isoformat(),
                        group="official_thinking" if self.thinking else "official",
                        model=self.model,
                        thinking=self.thinking,
                        gate="not_dispatched",
                        request_sha256=None,
                        gate_lines=0,
                        input_tokens=0,
                        cache_creation_input_tokens=0,
                        cache_read_input_tokens=0,
                        output_tokens=0,
                        estimated_usd=0.0,
                        ttft_seconds=None,
                        total_seconds=round(monotonic() - started, 6),
                        stop_reason=None,
                        error_category=category,
                    )
                )
            raise OfficialRouteFailure(category, emitted_text=emitted_text) from None
        finally:
            for pending_task in (task, gate_wait):
                if pending_task is not None and not pending_task.done():
                    pending_task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await pending_task
            if stream is not None:
                await stream.aclose()
            if "pending_row" in context:
                context["flushing_ledger"] = True
                self._write_ledger(context["pending_row"])
            _REQUEST_CONTEXT.reset(token)
