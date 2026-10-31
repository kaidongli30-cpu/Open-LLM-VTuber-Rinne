"""Per-turn official routing with shared state and fail-closed background checks."""

from __future__ import annotations

import asyncio
from contextvars import Context
from copy import deepcopy, copy
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from loguru import logger
from pathlib import Path
from time import monotonic
from typing import Any

from .checks_deadline import check_deadline, is_pending, check_scope
from .clean_view import (
    BACKGROUND_PENDING,
    BACKGROUND_PLACEHOLDER,
    background_parts,
    clean_background,
    clean_history,
    FIXED_ANCHORS,
)
from .content_check import CheckRecord, fingerprint, CheckBudgetExceeded
from .gate import RequestGate
from .state import OfficialRouteState, estimate_turn_reserve
from ..memory.message_time_context import add_message_time_context


def _now(metadata=None):
    value = (metadata or {}).get("official_route_now") or (metadata or {}).get(
        "request_received_at"
    )
    if isinstance(value, datetime):
        return value if value.tzinfo else value.astimezone()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).astimezone()
        except ValueError:
            pass
    return datetime.now().astimezone()


def _turn_text(turn):
    return json.dumps(
        dict(user=turn["user"], assistant=turn.get("assistant", "")), ensure_ascii=False
    )


def _pending(text, kind, version):
    return CheckRecord(
        "",
        fingerprint(text),
        kind,
        version,
        "",
        "U",
        0,
        "check_failure",
        "",
        "无法确认",
    )


def _failure_category(record):
    reason = getattr(record, "reason", "")
    return (
        "timeout"
        if reason == "超时"
        else "format"
        if reason == "格式错误"
        else "connection"
    )


class _CachedOnly:
    """Previously attempted missing data becomes a neutral omission, never a new call."""

    def __init__(self, checker, memory_kind=None):
        self.checker = checker
        self.memory_kind = memory_kind

    def __getattr__(self, key):
        return getattr(self.checker, key)

    async def check_many(self, texts, kind="segment", *, contexts=None):
        contexts = contexts or [()] * len(texts)
        return [
            self.checker.cached(t, self.memory_kind or kind, context=tuple(c))
            or _pending(t, self.memory_kind or kind, self.checker.version)
            for t, c in zip(texts, contexts)
        ]


@dataclass
class RouteChoice:
    provider: str = ""
    fallback_from: str | None = None
    fallback_reason: str | None = None
    provider_state: Any = None
    prior_cost_usd: float = 0.0
    prior_check_seconds: float = 0.0
    route: str = "relay"
    reason: str = "disabled"
    messages: list = field(default_factory=list)
    system: str = ""
    client: Any = None
    token: str | None = None
    check_seconds: float = 0.0
    started: float = field(default_factory=monotonic)
    now: datetime | None = None
    input_text: str = ""
    history: list = field(default_factory=list)
    request_sha256: str = ""
    gate_lines: int = 0
    backgrounds: list = field(default_factory=list)
    check_audit: list = field(default_factory=list)


class OfficialRouteRuntime:
    def __init__(self, config, data_dir, relay_llm, checker, official_factory=None):
        self.config = (
            config.model_dump() if hasattr(config, "model_dump") else dict(config)
        )
        self.data_dir = Path(data_dir)
        self.relay_llm = relay_llm
        self.checker = checker
        self.gate = RequestGate(
            checker, model=self.config.get("model", "claude-opus-4-6")
        )
        self.state = OfficialRouteState(self.config, self.data_dir)
        self.official_factory = official_factory
        self._tasks = set()
        self._jobs = {}
        self._attempted = set()
        self._ready_events = {}
        self._retry_delay = {}
        self._shutdown = False
        self._background_sem = asyncio.Semaphore(2)
        self._background_versions = {}

    @property
    def enabled(self):
        return bool(self.config.get("enabled", False))

    def parse_command(self, text):
        return self.state.parse_manual_command(text) if self.enabled else None

    async def handle_command(self, text, metadata=None):
        mode = self.parse_command(text)
        if mode is None:
            return None
        if mode == "invalid":
            return "⚙ 没有这个指令。可用：/anthropic、/openai、/relay（/apinebula），/official 回到上次的官方通道"
        await self.state.set_manual(mode, now=_now(metadata))
        await self.state.render_status(now=_now(metadata))
        selected = (await self.state.snapshot(now=_now(metadata)))["route"]
        return {
            "anthropic": "⚙ 已切到 Anthropic",
            "openai": "⚙ 已切到 OpenAI",
            "apinebula": "⚙ 已切到中转站（APINebula）",
        }[selected]

    def _history(self, agent, current_text=None):
        memory = deepcopy(getattr(agent, "_memory", []))
        if (
            memory
            and memory[-1].get("role") == "user"
            and memory[-1].get("content") == current_text
        ):
            memory.pop()
        turns = []
        for m in memory:
            if not isinstance(m.get("content"), str):
                continue
            if m.get("role") == "user":
                turns.append(
                    dict(user=m["content"], assistant="", timestamp=m.get("timestamp"))
                )
            elif m.get("role") == "assistant" and turns:
                turns[-1]["assistant"] = m["content"]
                turns[-1]["assistant_timestamp"] = m.get("timestamp")
        return turns

    def _backgrounds(self, agent, metadata):
        persona = getattr(agent, "_system", "")
        if "【核心原则】" not in persona:
            raise ValueError("protected_anchor_missing")
        memory_rules = "\n\n".join(
            getattr(metadata.get(name), "decision_instructions", "")
            for name in ("recent_diary_recall_tool", "long_term_memory_tool")
            if metadata.get(name) is not None
        )
        backgrounds = [
            (
                "persona",
                persona,
                dict(anchors=tuple(a for a in FIXED_ANCHORS if a in persona)),
            ),
            ("layer2", metadata.get("layer2_user_background", ""), dict(layer2=True)),
            ("recent", getattr(agent, "_recent_memory_context", ""), dict(recent=True)),
            ("rules", metadata.get("mandatory_response_rules", ""), {}),
            ("memory_rules", memory_rules, {}),
            # SDK-generated field outside the protocol exemption list. Check
            # it once in background, then only reuse its A record at send.
            ("wire_protocol", "stream", {}),
        ]
        approved = getattr(self.checker, "approved_memory", None)
        if approved is None or not approved.enabled:
            return backgrounds
        return [
            (
                name,
                approved.project(
                    text,
                    name,
                    background_parts(
                        text, **{k: v for k, v in options.items() if k != "recent"}
                    ),
                )
                if name in {"layer2", "recent", "rules"} and text
                else text,
                options,
            )
            for name, text, options in backgrounds
        ]

    def _key(self, texts, kind, contexts):
        return fingerprint(json.dumps([kind, texts, contexts], ensure_ascii=False))

    def _wire_protocol_entries(self, tools):
        """Use the native serializer for protocol fields without opening a client."""
        from ..agent.stateless_llm.claude_official_llm import AsyncLLM

        serializer = AsyncLLM.__new__(AsyncLLM)
        serializer.model = self.config.get("model", "claude-opus-4-6")
        serializer.thinking = self.config.get("thinking", True)
        serializer.effort = self.config.get("effort")
        serializer.system = None
        serializer.stable_system_prefix = None
        serializer._forced_tool_name_once = None
        wire = serializer.build_wire_request([], "", tools)
        extra = self.gate._metadata(
            {k: v for k, v in wire.items() if k not in {"system", "messages", "tools"}}
        )
        return [(text, "line", ()) for text in self.checker.schema_texts(extra)]

    def _spawn(self, coro):
        # Background retries must not inherit a finished foreground deadline.
        task = Context().run(asyncio.create_task, coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _check_batch(self, texts, kind, contexts, now=None):
        try:
            async with self._background_sem:
                with check_scope(
                    phase="background", day=_now({"official_route_now": now}).date()
                ):
                    records = await self.checker.check_many(
                        texts, kind=kind, contexts=contexts, timeout_seconds=60
                    )
            good = any(not is_pending(r) for r in records)
            if good:
                await self.state.note_check_success(now=now)
            return records
        except asyncio.CancelledError:
            raise
        except CheckBudgetExceeded as error:
            self._check_budget_error = error
            raise
        except Exception:
            return [_pending(t, kind, self.checker.version) for t in texts]

    async def _background_job(
        self, texts, kind, contexts, key, now=None, timestamps=None
    ):
        records = await self._check_batch(texts, kind, contexts, now)
        self._attempted.add(key)
        for n, r in enumerate(records):
            if not is_pending(r) and r.label != "A" and timestamps and timestamps[n]:
                await self.state.note_blocked_turn(timestamps[n], now=now)
        failed = [
            i
            for i, r in enumerate(records)
            if is_pending(r) and self._retry_allowed(texts[i], kind, contexts[i], now)
        ]
        if failed and not self._shutdown:
            level = self._retry_delay.get(key, 0)
            self._retry_delay[key] = min(level + 1, 3)
            seconds = [60, 120, 300, 600][min(level, 3)]
            self._spawn(
                self._retry_after(
                    seconds,
                    [texts[i] for i in failed],
                    kind,
                    [contexts[i] for i in failed],
                    key,
                    now,
                    [timestamps[i] for i in failed] if timestamps else None,
                )
            )
        else:
            self._retry_delay.pop(key, None)
        await self.state.set_pending_count(
            len(self.checker.pending)
            + sum(
                not t.done() and t is not asyncio.current_task()
                for t in self._jobs.values()
            ),
            now=now,
        )

    async def _retry_after(self, seconds, texts, kind, contexts, key, now, timestamps):
        await asyncio.sleep(seconds)
        if not self._shutdown:
            self._jobs[key] = self._spawn(
                self._background_job(texts, kind, contexts, key, None, timestamps)
            )

    def _schedule(
        self, texts, kind="segment", contexts=None, now=None, timestamps=None
    ):
        if not texts:
            return None
        contexts = contexts or [()] * len(texts)
        needed = [
            i
            for i, t in enumerate(texts)
            if self.checker.cached(t, kind, context=contexts[i]) is None
            and self._retry_allowed(t, kind, contexts[i], now)
        ]
        if not needed:
            return None
        texts = [texts[i] for i in needed]
        contexts = [contexts[i] for i in needed]
        timestamps = [timestamps[i] for i in needed] if timestamps else None
        key = self._key(texts, kind, contexts)
        current = self._jobs.get(key)
        if current is None or current.done() and key not in self._retry_delay:
            self._jobs[key] = self._spawn(
                self._background_job(texts, kind, contexts, key, now, timestamps)
            )
        return key

    def _retry_allowed(self, text, kind, context, now=None):
        with check_scope(day=_now({"official_route_now": now}).date()):
            return not hasattr(
                self.checker, "retry_allowed"
            ) or self.checker.retry_allowed(text, kind, context)

    async def prime(self, agent, input_data, tools=None):
        if not self.enabled:
            return
        metadata = input_data.metadata or {}
        now = _now(metadata)
        if (await self.state.snapshot(now)).get("effective_route") == "apinebula":
            return
        backgrounds = self._backgrounds(agent, metadata)
        for name, text, options in backgrounds:
            if getattr(self.checker, "single_safety_pass", False) and name not in {
                "layer2",
                "recent",
            }:
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            parts = background_parts(text, **options)
            ordinary = [
                p["text"] for p in parts if not p["protected"] and p["text"].strip()
            ]
            signature = fingerprint(name + "\n" + text)
            if signature not in self._background_versions:
                kind = "memory_" + name if name in {"layer2", "recent"} else "segment"
                self._background_versions[signature] = self._schedule(
                    ordinary, kind=kind, now=now
                )
        history = self._history(agent, agent._to_text_prompt(input_data))
        texts = [_turn_text(t) for t in history]
        contexts = [tuple(texts[max(0, i - 3) : i]) for i in range(len(texts))]
        await self._cached_cooldown(history, texts, contexts, now)
        if tools and not getattr(self.checker, "single_safety_pass", False):
            self._schedule(
                list(dict.fromkeys(self.checker.schema_texts(tools))),
                kind="line",
                now=now,
            )
        await self.state.set_pending_count(
            sum(not t.done() for t in self._jobs.values()), now=now
        )

    async def wait_prechecks(self):
        active = [t for t in self._jobs.values() if not t.done()]
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        if getattr(self, "_check_budget_error", None) is not None:
            raise self._check_budget_error

    async def _cached_cooldown(self, history, texts, contexts, now):
        if getattr(self.checker, "single_safety_pass", False):
            return
        timestamps = []
        for turn, text, context in zip(history, texts, contexts):
            record = self.checker.cached(text, "turn", context=context)
            stamp = turn.get("timestamp")
            if stamp and not is_pending(record) and record.label != "A":
                timestamps.append(stamp)
        if timestamps:
            latest = max(
                timestamps,
                key=lambda x: datetime.fromisoformat(x) if isinstance(x, str) else x,
            )
            await self.state.note_blocked_turn(latest, now=now)

    async def _background_ready(self, agent, metadata, now):
        for name, text, options in self._backgrounds(agent, metadata):
            if getattr(self.checker, "single_safety_pass", False) and name not in {
                "layer2",
                "recent",
            }:
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            values = [
                p["text"]
                for p in background_parts(text, **options)
                if not p["protected"] and p["text"].strip()
            ]
            kind = "memory_" + name if name in {"layer2", "recent"} else "segment"
            memory = getattr(self.checker, "memory_checks", None)
            if (
                name in {"layer2", "recent"}
                and memory is not None
                and memory.recovery_blocked
                and any(
                    memory.project_unit(t, kind)
                    in {BACKGROUND_PENDING, BACKGROUND_PLACEHOLDER}
                    for t in values
                )
            ):
                return False
            missing = [t for t in values if self.checker.cached(t, kind) is None]
            signature = fingerprint(name + "\n" + text)
            key = self._background_versions.get(signature)
            if (
                name in {"layer2", "recent"}
                and key
                and key not in self._attempted
                and any(is_pending(self.checker.cached(t, kind)) for t in values)
            ):
                return False
            if not missing:
                continue
            if signature not in self._background_versions:
                key = self._schedule(values, kind=kind, now=now)
                self._background_versions[signature] = key
            if key and key not in self._attempted:
                return False
        return True

    async def _clean_request(self, agent, input_data, messages, user_text, history):
        metadata = input_data.metadata or {}
        cached = _CachedOnly(self.checker)
        cleaned = {}
        manifests = []
        for name, text, options in self._backgrounds(agent, metadata):
            if text:
                memory_kind = "memory_" + name if name in {"layer2", "recent"} else None
                if memory_kind and hasattr(self.checker, "memory_checks"):
                    view = self.checker.memory_checks.clean_view(
                        text, memory_kind, options
                    )
                else:
                    view = await clean_background(
                        text, _CachedOnly(self.checker, memory_kind), **options
                    )
                cleaned[name] = view.text
                manifests += view.manifest
            else:
                cleaned[name] = ""
        # Preserve the existing composer, substituting only the cleaned components.
        shadow = dict(metadata)
        shadow["layer2_user_background"] = cleaned["layer2"]
        shadow["mandatory_response_rules"] = cleaned["rules"]
        for key in [
            "channel_system_instruction",
            "current_scene_context",
            "scene_continuity_system_instruction",
        ]:
            value = shadow.get(key)
            if isinstance(value, str) and value:
                shadow[key] = (await clean_background(value, cached)).text
        # A small proxy avoids mutation of the live agent's persona or memory.
        from types import SimpleNamespace

        proxy = SimpleNamespace(
            _system=cleaned["persona"],
            _recent_memory_context=cleaned["recent"],
            _use_mcpp=False,
        )
        from ..agent.agents.basic_memory_agent import BasicMemoryAgent

        shadow.pop("long_term_memory_tool", None)
        shadow.pop("recent_diary_recall_tool", None)
        clone = copy(input_data)
        clone.metadata = shadow
        legacy = [
            m["content"]
            for m in getattr(agent, "_memory", [])
            if m.get("role") == "system" and isinstance(m.get("content"), str)
        ]
        legacy = [(await clean_background(x, cached)).text for x in legacy]
        system = BasicMemoryAgent._compose_request_system_prompt(proxy, clone, legacy)
        extras = [
            cleaned["rules"],
            shadow.get("scene_continuity_system_instruction", ""),
        ]
        system = "\n\n".join(x for x in [system, *extras] if x and x.strip())
        if cleaned["memory_rules"]:
            system += "\n\n" + cleaned["memory_rules"]
        if metadata.get("source_channel") == "qq" and not metadata.get("proactive_speak"):
            qq_instruction = shadow.get("channel_system_instruction", "")
            if isinstance(qq_instruction, str) and qq_instruction.strip():
                # Append only the checked view, never the unreviewed original.
                system += "\n\n" + qq_instruction.strip()
        view = await clean_history(history, cached, neighbor_radius=1)
        wire = []
        for turn in view.turns:
            for role, key in [("user", "user"), ("assistant", "assistant")]:
                text = turn.get(key, "")
                if not text:
                    continue
                stamp = turn.get(
                    "timestamp" if role == "user" else "assistant_timestamp"
                )
                content = (
                    add_message_time_context(text, stamp)
                    if isinstance(stamp, str)
                    else text
                )
                wire.append(dict(role=role, content=content))
        current = deepcopy(messages[-1])
        blocks = current.get("content", [])
        if isinstance(blocks, str):
            blocks = [dict(type="text", text=blocks)]
        for block in blocks:
            if block.get("type") == "text" and block.get("text") != user_text:
                block["text"] = (
                    await clean_background(block.get("text", ""), cached)
                ).text
        current["content"] = blocks
        wire.append(current)
        return system, wire, manifests

    async def _prepare_provider_turn(
        self, agent, input_data, messages, user_text, active_capabilities, tools
    ):
        metadata = input_data.metadata or {}
        now = _now(metadata)
        choice = RouteChoice(
            messages=messages,
            client=self.relay_llm,
            now=now,
            input_text=user_text,
            history=self._history(agent, user_text),
        )
        choice.provider_state = self.state
        choice.provider = getattr(self, "_provider", "anthropic")
        if not self.enabled:
            return choice
        from .model_profiles import configuration_issue

        issue = configuration_issue(self.config)
        if issue:
            choice.reason = issue
            return choice
        key_ok = (
            bool(os.environ.get("RINNE_OFFICIAL_ANTHROPIC_KEY"))
            or self.official_factory is not None
        )
        if not key_ok:
            choice.reason = "missing_key"
            return choice
        history_texts = [_turn_text(t) for t in choice.history]
        history_contexts = [
            tuple(history_texts[max(0, i - 3) : i]) for i in range(len(history_texts))
        ]
        await self._cached_cooldown(
            choice.history, history_texts, history_contexts, now
        )
        prelim = await self.state.preflight(0, now)
        if not prelim["allow"]:
            choice.reason = prelim["reason"]
            return choice
        if (
            input_data.images
            or metadata.get("official_route_has_media")
            or metadata.get("proactive_speak")
            or metadata.get("video_analysis_context")
        ):
            choice.reason = "media"
            return choice
        from ..agent.agents.basic_memory_agent import should_force_search

        if (
            active_capabilities
            or metadata.get("official_route_forced_search")
            or should_force_search(user_text)
        ):
            choice.reason = "capability"
            return choice
        from ..memory.cloud_memory_tool import (
            LongTermRecallPolicy,
            classify_long_term_recall_policy,
        )

        memory_tool = metadata.get("long_term_memory_tool")
        if classify_long_term_recall_policy(
            user_text
        ) is LongTermRecallPolicy.FORCE or getattr(
            memory_tool, "force_first_call", False
        ):
            # The relay can force retrieval before answering. Official adaptive
            # thinking only permits auto, so keep mandatory recall on the relay.
            choice.reason = "forced_recall"
            return choice
        original_estimate = estimate_turn_reserve(
            len(
                json.dumps(
                    dict(
                        system=agent._finalize_request_system_prompt(
                            agent._current_request_system_prompt()
                        ),
                        messages=messages,
                        tools=tools,
                    ),
                    ensure_ascii=False,
                )
            ),
            model=getattr(
                self, "_budget_model", self.config.get("model", "claude-opus-4-6")
            ),
            provider=getattr(self, "_provider", "anthropic"),
        )
        budget = await self.state.preflight(original_estimate, now)
        if not budget["allow"]:
            choice.reason = budget["reason"]
            return choice
        try:
            await self.prime(agent, input_data, tools)
            if not await self._background_ready(agent, metadata, now):
                choice.reason = "background_pending"
                return choice
        except Exception as exc:
            choice.reason = "check_failed"
            await self.state.note_check_failure(now)
            logger.warning(
                "Official route check_failed: phase=prime category={}",
                type(exc).__name__,
            )
            return choice
        started = monotonic()
        deadline = started + 20
        try:
            with (
                check_deadline(20, deadline=deadline),
                check_scope(
                    phase="turn_input", audit=choice.check_audit, day=now.date()
                ),
            ):
                history_texts = [_turn_text(t) for t in choice.history]
                contexts = [
                    tuple(history_texts[max(0, i - 3) : i])
                    for i in range(len(history_texts))
                ]
                missing_turns = [
                    i
                    for i, text in enumerate(history_texts)
                    if self.checker.cached(text, "turn", context=contexts[i]) is None
                ]
                official_parts = [
                    i
                    for i in missing_turns
                    if self.checker.route_mark(history_texts[i]) == "official"
                    and not is_pending(
                        self.checker.official_input_record(
                            history_texts[i], contexts[i]
                        )
                    )
                ]
                relay_turns = [i for i in missing_turns if i not in official_parts][
                    -16:
                ]
                blocks = messages[-1].get("content", [])
                if isinstance(blocks, str):
                    blocks = [dict(type="text", text=blocks)]
                small = [
                    b["text"]
                    for b in blocks
                    if isinstance(b, dict)
                    and b.get("type") == "text"
                    and b.get("text") != user_text
                ]
                small += [
                    v
                    for k in [
                        "channel_system_instruction",
                        "current_scene_context",
                        "scene_continuity_system_instruction",
                    ]
                    if isinstance((v := metadata.get(k)), str) and v
                ]
                current_entry = (user_text, "input", tuple(history_texts[-3:]))
                entries = [
                    (
                        choice.history[i]["assistant"],
                        "segment",
                        self.checker.official_reply_context(
                            choice.history[i]["user"], contexts[i]
                        ),
                    )
                    for i in official_parts
                ]
                entries += [(text, "segment", ()) for text in small]
                single_safety = getattr(self.checker, "single_safety_pass", False)
                if single_safety:
                    # Stable backgrounds, scenes, schemas and this turn share
                    # one Sonnet request. Memory keeps its independent Opus path.
                    for name, text, options in self._backgrounds(agent, metadata):
                        if name in {"layer2", "recent"} or not text:
                            continue
                        entries.extend(
                            (part["text"], "segment", ())
                            for part in background_parts(text, **options)
                            if not part["protected"] and part["text"].strip()
                        )
                    entries.extend(
                        (m["content"], "segment", ())
                        for m in getattr(agent, "_memory", [])
                        if m.get("role") == "system"
                        and isinstance(m.get("content"), str)
                    )
                    if tools:
                        entries.extend(
                            (text, "line", ())
                            for text in self.checker.schema_texts(tools)
                        )
                    entries.extend(self._wire_protocol_entries(tools))
                entries.append(current_entry)
                # Keep the newest unchecked items in a single <=16-item request.
                selected = [
                    entry
                    for entry in entries
                    if self.checker.cached(entry[0], entry[1], context=entry[2]) is None
                ]
                if single_safety:
                    selected = list(dict.fromkeys(selected))
                    previous_entries = [
                        (history_texts[i], "turn", contexts[i]) for i in relay_turns
                    ]
                    combined = await asyncio.wait_for(
                        self.checker.check_segments([*selected, *previous_entries]),
                        max(0, deadline - monotonic()),
                    )
                    results = [combined[: len(selected)], combined[len(selected) :]]
                else:
                    selected = selected[-16:]
                    work = [self.checker.check_segments(selected)]
                    if relay_turns:
                        work.append(
                            self.checker.check_many(
                                [history_texts[i] for i in relay_turns],
                                kind="turn",
                                contexts=[contexts[i] for i in relay_turns],
                            )
                        )
                    results = await asyncio.wait_for(
                        asyncio.gather(*work, return_exceptions=True),
                        max(0, deadline - monotonic()),
                    )
                for result in results:
                    if isinstance(result, BaseException):
                        raise result
                current = dict(zip(selected, results[0])).get(current_entry)
                if current is None:
                    current = self.checker.cached(
                        user_text, "input", context=current_entry[2]
                    )
                if is_pending(current):
                    choice.reason = "check_failed"
                    await self.state.note_check_failure(
                        now, category=_failure_category(current)
                    )
                    return choice
                await self.state.note_check_success(now)
                if any(is_pending(record) for group in results for record in group):
                    # Preserve the existing current-input failure streak rule.
                    choice.reason = "check_failed"
                    return choice
                if current.label != "A":
                    choice.reason = "current_not_a"
                    await self.state.note_blocked_turn(now, now=now)
                    return choice
                for i in official_parts:
                    self.checker.record_official_turn(history_texts[i], contexts[i])
                await self._cached_cooldown(
                    choice.history, history_texts, contexts, now
                )
                system, wire, manifest = await self._clean_request(
                    agent, input_data, messages, user_text, choice.history
                )
                continuation = getattr(self, "_checked_continuation", [])
                if continuation:
                    wire = [*wire, *continuation]
                    original_estimate = estimate_turn_reserve(
                        len(
                            json.dumps(
                                dict(system=system, messages=wire, tools=tools),
                                ensure_ascii=False,
                            )
                        ),
                        model=getattr(
                            self,
                            "_budget_model",
                            self.config.get("model", "claude-opus-4-6"),
                        ),
                        provider=getattr(self, "_provider", "anthropic"),
                    )
                if self.official_factory is None:
                    from .runtime_client import GuardedOfficialClient

                    factory = GuardedOfficialClient
                else:
                    factory = self.official_factory
                turn_gate = copy(self.gate)
                turn_gate._official_thinking = set()
                turn_gate._tool_specs = {}
                if single_safety:
                    turn_gate.cache_only = True
                client = factory(
                    gate=turn_gate,
                    ledger_path=self.data_dir / "usage.jsonl",
                    checker=self.checker,
                    model=self.config.get("model", "claude-opus-4-6"),
                    thinking=self.config.get("thinking", True),
                    check_budget_seconds=max(0, 20 - (monotonic() - started)),
                    first_text_timeout_seconds=self.config.get(
                        "first_text_timeout_seconds", 40
                    ),
                    effort=self.config.get("effort"),
                )
                client.check_audit = choice.check_audit
                for name in ("long_term_memory_tool", "recent_diary_recall_tool"):
                    helper = metadata.get(name)
                    if (
                        helper
                        and getattr(helper, "force_first_call", False)
                        and hasattr(client, "force_tool_once")
                    ):
                        client.force_tool_once(helper.name)
                # Use the very same builder as the official client. Construction
                # does not dispatch and no credentials/payload enter metadata.
                payload = client.build_wire_request(wire, system, tools)
                with check_scope(phase="route_gate"):
                    gate = await asyncio.wait_for(
                        turn_gate.check(payload, cached_only=True)
                        if single_safety
                        else turn_gate.check(payload),
                        max(0, deadline - monotonic()),
                    )
                if not gate.allowed:
                    choice.reason = "check_failed" if gate.pending else "gate_blocked"
                    if gate.pending:
                        await self.state.note_check_failure(now, category="timeout")
                    close = getattr(getattr(client, "client", None), "close", None)
                    if close:
                        await close()
                    return choice
                if hasattr(client, "bind_initial_view"):
                    client.bind_initial_view(wire, system, tools)
                reservation = await self.state.reserve(original_estimate, now)
                if not reservation["allow"]:
                    close = getattr(getattr(client, "client", None), "close", None)
                    if close:
                        await close()
                    choice.reason = reservation["reason"]
                    return choice
                choice.token = reservation["token"]
                choice.route = "official"
                choice.reason = "official"
                choice.system = system
                choice.messages = wire
                choice.backgrounds = manifest
                choice.gate_lines = gate.checked_lines
                choice.request_sha256 = fingerprint(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                choice.client = client
                return choice
        except asyncio.TimeoutError:
            choice.reason = "check_failed"
            await self.state.note_check_failure(now, category="timeout")
            return choice
        except asyncio.CancelledError:
            raise
        except Exception:
            choice.reason = "check_failed"
            await self.state.note_check_failure(now)
            return choice
        finally:
            choice.check_seconds = monotonic() - started

    async def prepare_turn(
        self, agent, input_data, messages, user_text, active_capabilities, tools
    ):
        if not self.enabled:
            return RouteChoice(
                messages=messages, client=self.relay_llm, now=_now(input_data.metadata)
            )
        snapshot = await self.state.snapshot(now=_now(input_data.metadata))
        selected = snapshot.get("route", "anthropic")
        if selected == "apinebula":
            return RouteChoice(
                reason="manual_relay",
                messages=messages,
                client=self.relay_llm,
                now=_now(input_data.metadata),
                input_text=user_text,
            )
        return await self._choose_provider(
            selected, agent, input_data, messages, user_text, active_capabilities, tools
        )

    async def _choose_provider(
        self,
        selected,
        agent,
        input_data,
        messages,
        user_text,
        active_capabilities,
        tools,
        *,
        allow_other=True,
        continuation=None,
    ):
        own_reasons = {
            "official_model_unpriced",
            "unsupported_effort",
            "missing_key",
            "daily_budget",
            "credit_floor",
            "credit_expired",
            "credit_not_started",
            "error_pause",
            "disabled_today",
            "authentication",
            "quota",
        }
        first_reason = None
        pool = (
            [selected, "openai" if selected == "anthropic" else "anthropic"]
            if allow_other
            else [selected]
        )
        for provider in pool:
            view = self.state.for_provider(provider)
            available = await view.preflight(0, now=_now(input_data.metadata))
            reason = None if available["allow"] else available["reason"]
            if reason is None and self.official_factory is None:
                if provider == "openai":
                    from .openai_client import read_openai_api_key

                    if not read_openai_api_key():
                        reason = "missing_key"
                elif not os.environ.get("RINNE_OFFICIAL_ANTHROPIC_KEY"):
                    reason = "missing_key"
            if reason:
                first_reason = first_reason or reason
                if reason not in own_reasons:
                    return RouteChoice(
                        reason=reason,
                        messages=messages,
                        client=self.relay_llm,
                        now=_now(input_data.metadata),
                        input_text=user_text,
                    )
                continue
            shadow = copy(self)
            shadow._checked_continuation = continuation or []
            shadow.state = view
            shadow.config = dict(self.config)
            shadow._provider = provider
            shadow._budget_model = view.config.model
            if provider == "openai":
                shadow.config.update(
                    model="claude-opus-4-6",
                    effort=None,
                    first_text_timeout_seconds=view.config.first_text_timeout_seconds,
                )
                shadow.gate = copy(self.gate)
                shadow.gate.model = "claude-opus-4-6"
                if self.official_factory is None:
                    from .openai_client import GuardedOpenAIClient

                    def factory(**kwargs):
                        kwargs["effort"] = view.config.effort
                        return GuardedOpenAIClient(
                            **kwargs, openai_model=view.config.model
                        )

                    shadow.official_factory = factory
            else:
                shadow.config.update(
                    effort=view.config.effort,
                    first_text_timeout_seconds=view.config.first_text_timeout_seconds,
                )
            choice = await shadow._prepare_provider_turn(
                agent, input_data, messages, user_text, active_capabilities, tools
            )
            if choice.route == "official":
                if provider != selected:
                    choice.fallback_from, choice.fallback_reason = (
                        selected,
                        first_reason,
                    )
                return choice
            first_reason = first_reason or choice.reason
            if choice.reason not in own_reasons:
                return choice
        return RouteChoice(
            reason=first_reason or "missing_key",
            messages=messages,
            client=self.relay_llm,
            now=_now(input_data.metadata),
            input_text=user_text,
        )

    async def prepare_alternate(
        self,
        choice,
        category,
        agent,
        input_data,
        messages,
        user_text,
        active_capabilities,
        tools,
    ):
        if choice.fallback_from or category not in {
            "connection",
            "provider_error",
            "upstream_5xx",
            "server_error",
            "rate_limit",
            "quota",
            "authentication",
        }:
            return None
        other = "openai" if choice.provider == "anthropic" else "anthropic"
        checked_tail = getattr(choice.client, "checked_tool_continuation", lambda: [])()
        alternate = await self._choose_provider(
            other,
            agent,
            input_data,
            messages,
            user_text,
            active_capabilities,
            tools,
            allow_other=False,
            continuation=checked_tail,
        )
        if alternate.route != "official":
            return None
        settled = {"route": "official", "route_reason": "official_error"}
        await self.finish_turn(choice, settled, error=category, attempt_only=True)
        alternate.prior_cost_usd = settled.get("official_usd", 0)
        alternate.prior_check_seconds = settled.get("official_check_seconds", 0)
        alternate.fallback_from, alternate.fallback_reason = choice.provider, category
        return alternate

    async def finish_turn(
        self, choice, metadata, reply="", error=None, *, attempt_only=False
    ):
        state = choice.provider_state or self.state
        usage = (
            getattr(choice.client, "usage_rows", [])
            if choice.route == "official"
            else []
        )
        amount = sum(x.get("estimated_usd", 0) for x in usage)
        gate_lines = (
            sum(x.get("gate_lines", x.get("checked_lines", 0)) for x in usage)
            or choice.gate_lines
        )
        sha = (
            usage[-1].get("request_sha256", choice.request_sha256)
            if usage
            else choice.request_sha256
        )
        ttft = next(
            (x.get("ttft_seconds") for x in usage if x.get("ttft_seconds") is not None),
            None,
        )
        predispatch = {
            "check_failed",
            "gate_blocked",
            "content_blocked",
            "presend_unrecorded",
        }
        if error == "check_failed":
            await state.note_check_failure(choice.now, category="timeout")
        if error and error not in predispatch:
            await state.note_error(error, choice.now)
            if error == "refusal":
                await state.note_blocked_turn(choice.now, now=choice.now)
        await state.finish(
            choice.token,
            route=metadata.get("route", choice.route),
            reason=metadata.get("route_reason", choice.reason),
            official_usd=amount,
            ttft_seconds=ttft,
            now=choice.now,
            gate_lines=gate_lines,
            request_sha=sha,
            check_seconds=choice.check_seconds,
            fallback_from=choice.fallback_from,
            fallback_reason=choice.fallback_reason,
            attempt_only=attempt_only,
        )
        metadata.update(
            official_usd=amount + choice.prior_cost_usd,
            gate_lines=gate_lines,
            official_request_sha256=sha,
            official_check_seconds=choice.check_seconds + choice.prior_check_seconds,
            provider=choice.provider
            if metadata.get("route", choice.route) == "official"
            else "",
            fallback_from=choice.fallback_from,
            fallback_reason=choice.fallback_reason,
        )
        if reply and not metadata.get("official_route_command"):
            text = json.dumps(
                dict(user=choice.input_text, assistant=reply), ensure_ascii=False
            )
            if choice.reason == "manual_relay":
                self.checker.mark_route(text, "manual_relay", choice.now)
            elif metadata.get("route", choice.route) == "official" and not error:
                self.checker.mark_route(text, "official", choice.now)
        if choice.route == "official":
            close = getattr(getattr(choice.client, "client", None), "close", None)
            if close:
                await close()

    async def close(self):
        self._shutdown = True
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*list(self._tasks), return_exceptions=True)
