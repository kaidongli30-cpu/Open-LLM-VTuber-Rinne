"""Locked shared routing state, budget reservations and metadata-only status.

All public operations are async and serialized by normalized directory across
connections, threads and processes. ``expected_usd`` is the caller's one-request
reserve (``estimate_turn_reserve``); tool loops settle from the ledger
afterwards. ``finish`` settles once and never invokes
``note_error`` itself. The official client's usage.jsonl is authoritative; the
sum of completed-turn costs is a conservative floor for fake/absent ledgers.
No method accepts or persists dialogue, arbitrary exception text or API keys.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import math
import os
import re
import threading
import unicodedata
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from ..config_manager.agent import OfficialRouteConfig
from ..data_paths import character_history_root
from ..memory.message_time_context import MEMORY_DAY_BOUNDARY_HOUR
from .model_profiles import (
    MODEL_PROFILES,
    OPENAI_MODEL_PROFILES,
    configuration_issue,
    get_openai_profile,
    get_profile,
)
from .record_storage import process_lock

_REGISTRY_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_ASYNC_LOCKS: dict[tuple[str, asyncio.AbstractEventLoop], asyncio.Lock] = {}

REASONS = frozenset(
    {
        "official",
        "disabled",
        "missing_key",
        "disabled_today",
        "manual_relay",
        "cooldown",
        "media",
        "capability",
        "daily_budget",
        "pending_checks",
        "current_not_a",
        "gate_blocked",
        "official_error",
        "official_refusal",
        "error_pause",
        "credit_floor",
        "credit_expired",
        "credit_not_started",
        "authentication",
        "quota",
        "unknown",
        "cancelled",
        "timeout",
        "provider_error",
        "connection",
        "content_blocked",
        "refusal",
        "background_pending",
        "history_pending",
        "budget",
        "manual",
        "auto",
        "non_memory_tool",
        "search",
        "image",
        "video",
        "proactive",
        "checks_paused",
        "check_failed",
        "check_pending",
        "presend_unrecorded",
        "official_model_unpriced",
        "unsupported_effort",
        "invalid_check_model",
        "invalid_response",
        "upstream_5xx",
        "rate_limit",
        "server_error",
    }
)
ERRORS = frozenset(
    {
        "authentication",
        "quota",
        "timeout",
        "connection",
        "provider_error",
        "cancelled",
        "refusal",
        "rate_limit",
        "server_error",
        "upstream_5xx",
        "invalid_response",
    }
)


class StateStorageError(RuntimeError):
    """Metadata could not be read safely. The caller must use the relay."""


def _timestamp(value: datetime | str | None = None) -> datetime:
    if value is None:
        return datetime.now().astimezone()
    result = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(value.replace("Z", "+00:00"))
    )
    if result.tzinfo is None:
        result = result.astimezone()
    return result.astimezone()


def _day(now: datetime) -> str:
    return (now - timedelta(hours=MEMORY_DAY_BOUNDARY_HOUR)).date().isoformat()


def _money(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("amount_must_be_finite_and_nonnegative")
    return result


def _category(value: str, allowed: frozenset[str] = REASONS) -> str:
    return value if isinstance(value, str) and value in allowed else "unknown"


def _error_category(value: str) -> str:
    aliases = {
        "401": "authentication",
        "403": "authentication",
        "402": "quota",
        "auth": "authentication",
        "credit_balance": "quota",
    }
    return _category(aliases.get(value, value), ERRORS)


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _process_lock(path: Path):
    with process_lock(
        path, timeout_error=StateStorageError("state_lock_timeout"), poll_interval=0.01
    ):
        yield


def _status_html(snapshot: dict) -> str:
    def cash(value):
        return f"${value:.4f}（约 ¥{value * 7:.2f}）"

    def safe(value):
        return html.escape(str(value if value is not None else "—"))

    explicit_only = snapshot["chat_check_mode"] == "explicit_only"
    check_models = (
        f"显式检查：{safe(snapshot['first_pass_model'])}；聊天二审：禁用"
        if explicit_only
        else f"第一遍：{safe(snapshot['first_pass_model'])}；二审：{safe(snapshot['second_pass_model'])}"
    )

    provider_sections = "".join(
        f"<h2>{safe(name)}</h2>"
        f"<p>模型：{safe(info['official_model'])}；思考强度：{safe(info['effort'])}；"
        f"首字等待：{safe(info['first_text_timeout_seconds'])}秒</p>"
        f"<p>今天花费：{cash(info['today_usd'])}；今天上限：{cash(info['daily_limit_usd'])}；"
        f"待结算预留：{cash(info['reserved_usd'])}</p>"
        f"<p>本账期花费：{cash(info['period_usd'])}；剩余：{cash(info['remaining_usd'])}；"
        f"到期日：{safe(info['credit_expires'])}</p>"
        f"<p>可用状态：{safe(info['availability_reason'])}；{safe(info['configuration_status'])}；"
        f"错误暂停到：{safe(info['pause_until'])}；当天停用：{safe(info['disabled_day'])}；"
        f"最近错误：{safe(info['last_error'])}</p>"
        f"<p>今天官方：{info['turn_count']}轮；换入：{info['fallback_count']}次；换家原因："
        + (
            "、".join(
                f"{safe(k)}：{v}"
                for k, v in sorted(info["fallback_reason_counts"].items())
            )
            or "无"
        )
        + "</p>"
        for name, info in snapshot["providers"].items()
    )

    rows = "".join(
        "<tr>"
        + f"<td>{safe(row['time'])}</td>"
        + f"<td>{safe(row['route'])} {safe(row.get('provider', ''))}</td>"
        + f"<td>{safe(row['reason'])}"
        + (
            f"（从 {safe(row['fallback_from'])} 换来：{safe(row.get('fallback_reason'))}）"
            if row.get("fallback_from")
            else ""
        )
        + "</td>"
        + f"<td>{safe(row['official_usd'])}</td><td>{safe(row['ttft_seconds'])}</td>"
        + "</tr>"
        for row in reversed(snapshot["recent_turns"])
    )
    reasons = (
        "、".join(
            f"{safe(k)}：{v}" for k, v in sorted(snapshot["reason_counts"].items())
        )
        or "无"
    )
    return (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>官方通道状态</title>'
        "<style>body{font:16px sans-serif;margin:2em;line-height:1.7}table{border-collapse:collapse}"
        "td,th{border:1px solid #ccc;padding:.4em .7em}</style><h1>官方通道状态</h1>"
        f"<p>当前选择：{safe(snapshot['route'])}；实际选择：{safe(snapshot['effective_route'])}；"
        f"最近官方：{safe(snapshot['last_official'])}</p>"
        f"<p>官方模型：{safe(snapshot['official_model'])}；思考：{safe(snapshot['thinking_status'])}；"
        f"{check_models}</p>"
        f"<p>聊天检查模式：{safe(snapshot['chat_check_mode'])}；"
        f"内容冷却：{'禁用' if explicit_only else '启用'}；"
        f"检查失败暂停：{'禁用（失败仅当轮回中转）' if explicit_only else '启用'}</p>"
        f"<p>通道配置：{safe(snapshot['configuration_status'])}</p>"
        f"<p>记忆日：{safe(snapshot['memory_day'])}（凌晨3点换日）</p>"
        f"<p>今天花费：{cash(snapshot['today_usd'])}；今天上限：{cash(snapshot['daily_limit_usd'])}</p>"
        f"<p>本账期花费：{cash(snapshot['period_usd'])}；估计剩余：{cash(snapshot['remaining_usd'])}</p>"
        f"<p>待结算预留：{cash(snapshot['reserved_usd'])}；到期日：{safe(snapshot['credit_expires'])}</p>"
        f"<p>模式：{'手动中转站' if snapshot['manual_mode'] == 'relay' else '自动'}；"
        f"冷却到：{safe(snapshot['cooldown_until'])}；错误暂停到：{safe(snapshot['pause_until'])}</p>"
        f"<p>检查暂停到：{safe(snapshot['checks_paused_until'])}；待检查：{snapshot['pending_count']}；"
        f"连续检查失败：{snapshot['check_failure_streak']}；今天暂停检查：{snapshot['check_pause_count']}次</p>"
        "<p>今天检查失败："
        + "、".join(
            f"{safe(k)}：{v}"
            for k, v in sorted(snapshot["check_failure_counts"].items())
        )
        + "</p>"
        f"<p>今天官方：{snapshot['route_counts'].get('official', 0)}轮；"
        f"中转站：{snapshot['route_counts'].get('relay', 0)}轮；"
        f"最近错误：{safe(snapshot['last_error']) or '无'}</p><p>原因计数：{reasons}</p>"
        + provider_sections
        + "<h2>最近20轮</h2><table><thead><tr><th>时间</th><th>通道／厂家</th>"
        "<th>原因类别／换家</th>"
        "<th>美元</th><th>开口秒数</th></tr></thead><tbody>"
        + rows
        + "</tbody></table></html>"
    )


class OfficialRouteState:
    def __init__(
        self,
        config: OfficialRouteConfig | dict,
        data_dir: str | Path | None = None,
        *,
        provider: str = "anthropic",
    ):
        if provider not in {"anthropic", "openai"}:
            raise ValueError("provider_must_be_anthropic_or_openai")
        self.provider = provider
        self.root_config = (
            config
            if isinstance(config, OfficialRouteConfig)
            else OfficialRouteConfig(**config)
        )
        self.config = self.root_config
        if provider == "openai":
            self.config = self.root_config.model_copy(
                update={**self.root_config.openai.model_dump(), "thinking": True}
            )
        self.data_dir = (
            Path(
                data_dir
                or self.config.data_dir
                or character_history_root() / "official_route"
            )
            .expanduser()
            .resolve()
        )
        self.state_path = self.data_dir / "state.json"
        self.usage_path = self.data_dir / (
            "openai_usage.jsonl" if provider == "openai" else "usage.jsonl"
        )
        self.status_path = self.data_dir / "状态.html"
        self._key = os.path.normcase(str(self.data_dir))
        with _REGISTRY_GUARD:
            self._thread_lock = _THREAD_LOCKS.setdefault(self._key, threading.RLock())

    @property
    def _chat_check_mode(self):
        return getattr(self.root_config, "chat_check_mode", "two_pass")

    @property
    def _explicit_only(self):
        return self._chat_check_mode == "explicit_only"

    def for_provider(self, provider: str) -> OfficialRouteState:
        """Return an independent budget/client view using the same locked state.

        The default object remains the legacy Anthropic view. The returned
        config is flattened to that provider's model/effort/credit settings.
        Selection, cooldown, check state, history and locking remain shared.
        """
        return OfficialRouteState(self.root_config, self.data_dir, provider=provider)

    @staticmethod
    def _new_provider_state():
        return {
            "disabled_day": None,
            "pause_until": None,
            "consecutive_errors": 0,
            "last_error": "",
            "reservations": {},
            "settled_days": {},
            "completed_tokens": [],
        }

    def _provider_state(self, state):
        # Anthropic fields stay at the root for old code's rollback compatibility.
        return state if self.provider == "anthropic" else state["providers"]["openai"]

    def _read_state(self):
        state = self._read_legacy_state()
        try:
            providers = state.setdefault("providers", {})
            if not isinstance(providers, dict):
                raise ValueError()
            budget = providers.setdefault("openai", self._new_provider_state())
            if not isinstance(budget, dict):
                raise ValueError()
            for name, kind in (
                ("reservations", dict),
                ("settled_days", dict),
                ("completed_tokens", list),
            ):
                if not isinstance(budget.get(name), kind):
                    raise ValueError()
            for name in (
                "disabled_day",
                "pause_until",
                "consecutive_errors",
                "last_error",
            ):
                if name not in budget:
                    raise ValueError()
            configured = self.root_config.route
            routes = {"anthropic", "openai", "apinebula"}
            if "route" not in state:
                state["route"] = (
                    "apinebula" if state["manual_mode"] == "relay" else configured
                )
                state["route_conf_seen"] = configured
                state["last_official"] = (
                    state["route"] if state["route"] != "apinebula" else "anthropic"
                )
            if (
                state["route"] not in routes
                or state.get("route_conf_seen") not in routes
            ):
                raise ValueError()
            if state.get("last_official") not in {"anthropic", "openai"}:
                raise ValueError()
            if state["route_conf_seen"] != configured:
                state["route"] = configured
                state["route_conf_seen"] = configured
                if configured != "apinebula":
                    state["last_official"] = configured
            state["manual_mode"] = "relay" if state["route"] == "apinebula" else "auto"
            return state
        except (ValueError, TypeError, KeyError) as exc:
            raise StateStorageError("state_unreadable") from exc

    @staticmethod
    def parse_manual_command(text: str) -> str | None:
        command = re.fullmatch(r"[/／、]([A-Za-z]+)", text.strip())
        if command:
            name = command[1].casefold()
            if name in {"anthropic", "openai", "apinebula"}:
                return name
            if name in {"relay", "official"}:
                return "relay" if name == "relay" else "auto"
            if name not in {"end", "cancel", "status"}:
                return "invalid"
        normalized = "".join(
            c
            for c in text
            if not c.isspace() and not unicodedata.category(c).startswith("P")
        )
        return {"切到中转站": "relay", "切回自动": "auto"}.get(normalized)

    async def _transaction(self, callback: Callable, now=None):
        timestamp = _timestamp(now)
        loop = asyncio.get_running_loop()
        with _REGISTRY_GUARD:
            lock = _ASYNC_LOCKS.setdefault((self._key, loop), asyncio.Lock())
        async with lock:
            # Cancellation cannot abandon a still-running file transaction and
            # allow another task to interleave before its reservation is saved.
            worker = asyncio.create_task(
                asyncio.to_thread(self._transaction_sync, callback, timestamp)
            )
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                await worker
                raise

    def _transaction_sync(self, callback, now):
        with self._thread_lock:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with _process_lock(self.data_dir / ".state.lock"):
                state = self._read_state()
                snapshot = self._snapshot(state, now)
                result = callback(state, snapshot, now)
                _atomic_write(
                    self.state_path, json.dumps(state, ensure_ascii=False, indent=2)
                )
                return result

    def _read_legacy_state(self):
        if not self.state_path.exists():
            return {
                "version": 1,
                "manual_mode": "auto",
                "cooldown_until": None,
                "disabled_day": None,
                "pause_until": None,
                "consecutive_errors": 0,
                "last_error": "",
                "reservations": {},
                "recent_turns": [],
                "day_stats": {},
                "settled_days": {},
                "completed_tokens": [],
                "checks_paused_until": None,
                "check_failure_streak": 0,
                "pending_owners": {},
            }
        try:
            result = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(result, dict) or result.get("version") != 1:
                raise ValueError()
            for name, kind in (
                ("reservations", dict),
                ("recent_turns", list),
                ("day_stats", dict),
                ("settled_days", dict),
            ):
                if not isinstance(result.get(name), kind):
                    raise ValueError()
            return result
        except (OSError, ValueError, TypeError) as exc:
            raise StateStorageError("state_unreadable") from exc

    def _ledger(self, now):
        rows = []
        if not self.usage_path.exists():
            return rows
        try:
            # Writers append complete JSON + newline. An incomplete final append
            # is not counted yet; its still-held reservation covers the cost.
            data = self.usage_path.read_bytes()
            for line in data.splitlines(keepends=True):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    if not line.endswith((b"\n", b"\r")):
                        continue
                    raise
                if "time" not in row and "timestamp" not in row:
                    raise ValueError("usage_timestamp_missing")
                amount = _money(row.get("estimated_usd", 0))
                stamp = _timestamp(row.get("time", row.get("timestamp")))
                stamp = stamp.astimezone(now.tzinfo)
                rows.append((stamp, amount))
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise StateStorageError("usage_unreadable") from exc
        return rows

    def _provider_snapshot(self, state, now):
        budget = self._provider_state(state)
        credit_start = _timestamp(self.config.credit_as_of).astimezone(now.tzinfo)
        day = _day(now)
        ledger_days = Counter()
        for stamp, amount in self._ledger(now):
            if credit_start <= stamp <= now:
                ledger_days[_day(stamp)] += amount
        cycle = hashlib.sha256(
            f"{self.config.credit_as_of}|{self.config.credit_usd}|{self.config.credit_expires}".encode()
        ).hexdigest()
        if budget.get("credit_cycle") != cycle:
            budget["credit_cycle"] = cycle
            budget["settled_days"] = {}
            budget["ledger_baseline_days"] = dict(ledger_days)
            budget["budget_day"] = None
            budget["daily_limits"] = {}
        # The ledger is authoritative. A completed-turn aggregate acts only as
        # a floor when fake clients have not written a ledger; never add both.
        accounted = dict(ledger_days)
        for settled_day, amount in budget["settled_days"].items():
            floor = amount + budget.get("ledger_baseline_days", {}).get(settled_day, 0)
            accounted[settled_day] = max(accounted.get(settled_day, 0), floor)
        period = sum(accounted.values())
        today = accounted.get(day, 0.0)
        remaining = max(0.0, self.config.credit_usd - period)
        days = max(
            0,
            (
                date.fromisoformat(self.config.credit_expires) - date.fromisoformat(day)
            ).days
            + 1,
        )
        daily_limits = budget.setdefault("daily_limits", {})
        if not isinstance(daily_limits, dict):
            raise StateStorageError("daily_limits_unreadable")
        # Migrate the old single-day cache before selecting a different day.
        # A delayed old turn must not discard today's already-fixed limit.
        previous_day = budget.get("budget_day")
        if previous_day and "daily_limit_usd" in budget:
            daily_limits.setdefault(previous_day, _money(budget["daily_limit_usd"]))
        if day not in daily_limits:
            daily_limits[day] = (
                max(0.0, self.config.credit_usd - (period - today)) / days
                if days
                else 0.0
            )
        budget["budget_day"] = day
        budget["daily_limit_usd"] = _money(daily_limits[day])
        reservations = budget["reservations"]
        reserved = sum(_money(r["usd"]) for r in reservations.values())
        reserved_today = sum(
            _money(r["usd"]) for r in reservations.values() if r["memory_day"] == day
        )
        stats = state["day_stats"].get(day, {})
        issue = configuration_issue(self.config, self.provider)
        # Check-model validity is shared, even when the provider's model is bad.
        shared_issue = configuration_issue(
            self.root_config.model_copy(
                update={"model": "claude-opus-4-6", "effort": None}
            )
        )
        issue = shared_issue or issue
        profiles = (
            OPENAI_MODEL_PROFILES if self.provider == "openai" else MODEL_PROFILES
        )
        profile = profiles.get(self.config.model)
        active_thinking = bool(
            profile and profile.effective_thinking(self.config.thinking)
        )
        thinking_status = (
            "不可用（模型不在价目表里）"
            if profile is None
            else "开（此模型关不掉思考）"
            if profile.always_thinking
            else "开"
            if active_thinking
            else "关"
        )
        provider_stats = stats.get("providers", {}).get(self.provider, {})
        if (
            not provider_stats
            and self.provider == "anthropic"
            and "providers" not in stats
        ):
            provider_stats = stats
        return {
            "provider": self.provider,
            "effort": self.config.effort,
            "first_text_timeout_seconds": self.config.first_text_timeout_seconds,
            "official_model": self.config.model,
            "thinking_requested": self.config.thinking,
            "thinking_effective": active_thinking,
            "thinking_status": thinking_status,
            "first_pass_model": self.config.first_pass_model,
            "second_pass_model": self.config.second_pass_model,
            "configuration_issue": issue,
            "configuration_status": {
                "official_model_unpriced": "此家不可用：官方模型不在价目表里",
                "unsupported_effort": "此家不可用：模型不支持该思考强度",
                "invalid_check_model": "全走中转站：检查模型只支持 claude-sonnet-4-6 或 claude-opus-4-6",
            }.get(issue, "配置可用"),
            "memory_day": day,
            "manual_mode": state["manual_mode"],
            "chat_check_mode": self._chat_check_mode,
            "content_cooldown_enabled": not self._explicit_only,
            "check_failure_pause_enabled": not self._explicit_only,
            "cooldown_until": None if self._explicit_only else state["cooldown_until"],
            "disabled_day": budget["disabled_day"],
            "pause_until": budget["pause_until"],
            "consecutive_errors": budget["consecutive_errors"],
            "last_error": budget["last_error"],
            "today_usd": today,
            "checks_paused_until": (
                None if self._explicit_only else state.get("checks_paused_until")
            ),
            "check_failure_streak": (
                0 if self._explicit_only else state.get("check_failure_streak", 0)
            ),
            "pending_count": sum(state.get("pending_owners", {}).values()),
            "check_failure_counts": dict(stats.get("check_failures", {})),
            "check_pause_count": stats.get("check_pauses", 0),
            "daily_limit_usd": budget["daily_limit_usd"],
            "period_usd": period,
            "remaining_usd": remaining,
            "reserved_usd": reserved,
            "reserved_today_usd": reserved_today,
            "available_usd": max(0.0, remaining - reserved),
            "credit_expires": self.config.credit_expires,
            "days_remaining": days,
            "route_counts": dict(provider_stats.get("routes", {})),
            "turn_count": provider_stats.get("routes", {}).get("official", 0),
            "fallback_count": provider_stats.get("fallback_count", 0),
            "fallback_reason_counts": dict(provider_stats.get("fallback_reasons", {})),
            "reason_counts": dict(provider_stats.get("reasons", {})),
            "recent_turns": list(state["recent_turns"]),
            "credit_started": now >= credit_start,
        }

    def _snapshot(self, state, now):
        providers = {}
        for provider in ("anthropic", "openai"):
            view = self if self.provider == provider else self.for_provider(provider)
            providers[provider] = view._provider_snapshot(state, now)
            providers[provider]["availability_reason"] = view._decision(
                providers[provider], 0.0, now
            )["reason"]
        result = dict(providers[self.provider])
        result.update(
            route=state["route"],
            route_conf_seen=state["route_conf_seen"],
            last_official=state["last_official"],
            effective_route=state["route"] if self.root_config.enabled else "apinebula",
            providers=providers,
        )
        stats = state["day_stats"].get(result["memory_day"], {})
        # Keep the old top-level aggregate fields for existing runtime callers.
        result["route_counts"] = dict(stats.get("routes", {}))
        result["reason_counts"] = dict(stats.get("reasons", {}))
        return result

    def _decision(self, snapshot, expected, now):
        reason = "official"
        if not self.config.enabled:
            reason = "disabled"
        elif snapshot["manual_mode"] == "relay":
            reason = "manual_relay"
        elif (
            not self._explicit_only
            and snapshot["cooldown_until"]
            and _timestamp(snapshot["cooldown_until"]) > now
        ):
            reason = "cooldown"
        elif (
            not self._explicit_only
            and snapshot["checks_paused_until"]
            and _timestamp(snapshot["checks_paused_until"]) > now
        ):
            reason = "checks_paused"
        elif snapshot["configuration_issue"]:
            reason = snapshot["configuration_issue"]
        elif snapshot["disabled_day"] == snapshot["memory_day"]:
            reason = "disabled_today"
        elif snapshot["pause_until"] and _timestamp(snapshot["pause_until"]) > now:
            reason = "error_pause"
        elif not snapshot["credit_started"]:
            reason = "credit_not_started"
        elif not snapshot["days_remaining"]:
            reason = "credit_expired"
        elif snapshot["available_usd"] < 1.0 - 1e-9:
            reason = "credit_floor"
        elif (
            snapshot["today_usd"] + snapshot["reserved_today_usd"] + expected
            > snapshot["daily_limit_usd"] + 1e-9
            or expected > snapshot["available_usd"] + 1e-9
        ):
            reason = "daily_budget"
        return {"allow": reason == "official", "reason": reason, "token": None}

    async def snapshot(self, now=None):
        return await self._transaction(lambda state, snapshot, stamp: snapshot, now)

    async def preflight(self, expected_usd=0.0, now=None):
        expected = _money(expected_usd)
        return await self._transaction(
            lambda state, snapshot, stamp: self._decision(snapshot, expected, stamp),
            now,
        )

    async def reserve(self, expected_usd, now=None):
        expected = _money(expected_usd)
        if expected == 0:
            raise ValueError("reservation_must_be_positive")

        def reserve(state, snapshot, stamp):
            decision = self._decision(snapshot, expected, stamp)
            if decision["allow"]:
                token = uuid.uuid4().hex
                self._provider_state(state)["reservations"][token] = {
                    "usd": expected,
                    "time": stamp.isoformat(),
                    "memory_day": snapshot["memory_day"],
                }
                decision["token"] = token
            return decision

        return await self._transaction(reserve, now)

    async def set_manual(self, mode, now=None):
        if mode not in {"relay", "auto", "anthropic", "openai", "apinebula"}:
            raise ValueError("invalid_manual_route")

        def update(state, snapshot, stamp):
            route = (
                "apinebula"
                if mode == "relay"
                else state["last_official"]
                if mode == "auto"
                else mode
            )
            state["route"] = route
            state["route_conf_seen"] = self.root_config.route
            if route != "apinebula":
                state["last_official"] = route
            state["manual_mode"] = "relay" if route == "apinebula" else "auto"
            _atomic_write(self.status_path, _status_html(self._snapshot(state, stamp)))
            return mode

        return await self._transaction(update, now)

    async def note_blocked_turn(self, timestamp, now=None):
        if self._explicit_only:
            return None
        original = _timestamp(timestamp)
        until = original + timedelta(minutes=self.config.cooldown_minutes)

        def update(state, snapshot, stamp):
            previous = state["cooldown_until"]
            if not previous or _timestamp(previous) < until:
                state["cooldown_until"] = until.isoformat()
            return state["cooldown_until"]

        return await self._transaction(update, now)

    async def note_error(self, category, now=None):
        category = _error_category(category)

        def update(state, snapshot, stamp):
            budget = self._provider_state(state)
            budget["last_error"] = category
            budget["consecutive_errors"] += 1
            if category in {"authentication", "quota"}:
                budget["disabled_day"] = snapshot["memory_day"]
            if budget["consecutive_errors"] >= 3:
                budget["pause_until"] = (stamp + timedelta(minutes=30)).isoformat()
            return category

        return await self._transaction(update, now)

    async def note_check_failure(self, now=None, category="connection"):
        category = (
            category if category in {"connection", "timeout", "format"} else "format"
        )

        def update(state, snapshot, stamp):
            stats = state["day_stats"].setdefault(
                snapshot["memory_day"], {"routes": {}, "reasons": {}}
            )
            failures = stats.setdefault("check_failures", {})
            failures[category] = failures.get(category, 0) + 1
            if self._explicit_only:
                return self._snapshot(state, stamp)
            state["check_failure_streak"] = state.get("check_failure_streak", 0) + 1
            previous = state.get("checks_paused_until")
            if state["check_failure_streak"] >= 2 and (
                not previous or _timestamp(previous) <= stamp
            ):
                state["checks_paused_until"] = (
                    stamp + timedelta(minutes=5)
                ).isoformat()
                stats["check_pauses"] = stats.get("check_pauses", 0) + 1
            return self._snapshot(state, stamp)

        return await self._transaction(update, now)

    async def note_check_success(self, now=None):
        def update(state, snapshot, stamp):
            state["check_failure_streak"] = 0
            state["checks_paused_until"] = None
            return self._snapshot(state, stamp)

        return await self._transaction(update, now)

    async def set_pending_count(self, count, now=None, owner="default"):
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("pending_count_must_be_nonnegative_integer")
        owner_key = hashlib.sha256(str(owner).encode()).hexdigest()

        def update(state, snapshot, stamp):
            owners = state.setdefault("pending_owners", {})
            if count:
                owners[owner_key] = count
            else:
                owners.pop(owner_key, None)
            return sum(owners.values())

        return await self._transaction(update, now)

    async def finish(
        self,
        token=None,
        *,
        route="relay",
        reason="unknown",
        official_usd=0.0,
        ttft_seconds=None,
        now=None,
        gate_lines=0,
        request_sha="",
        check_seconds=0.0,
        fallback_from=None,
        fallback_reason=None,
        attempt_only=False,
    ):
        """Settle this provider; failed attempts may skip final-turn statistics.

        attempt_only releases the reservation and records actual cost exactly
        once per token, preserving error streaks and all turn/fallback counters.
        """
        if route not in {"official", "relay"}:
            raise ValueError("route_must_be_official_or_relay")
        amount = _money(official_usd)
        reason = _category(reason)
        fallback_from = (
            fallback_from if fallback_from in {"anthropic", "openai"} else None
        )
        fallback_reason = _category(fallback_reason) if fallback_from else None
        ttft = None if ttft_seconds is None else _money(ttft_seconds)
        checks = _money(check_seconds)
        sha = (
            request_sha
            if isinstance(request_sha, str)
            and len(request_sha) == 64
            and all(c in "0123456789abcdef" for c in request_sha)
            else ""
        )

        def settle(state, snapshot, stamp):
            budget = self._provider_state(state)
            if token and token in budget["completed_tokens"]:
                return self._snapshot(state, stamp)
            reservation = budget["reservations"].pop(token, None) if token else None
            if token and reservation is None:
                raise ValueError("unknown_reservation_token")
            day = snapshot["memory_day"]
            if amount:
                # Actual usage can exceed the reserve: once begun, an official
                # turn is allowed to finish. The next turn observes the excess.
                started = _timestamp(reservation["time"]) if reservation else stamp
                if started >= _timestamp(self.config.credit_as_of):
                    charged_day = reservation["memory_day"] if reservation else day
                    budget["settled_days"][charged_day] = (
                        budget["settled_days"].get(charged_day, 0) + amount
                    )
            if attempt_only:
                if token:
                    budget["completed_tokens"].append(token)
                    budget["completed_tokens"] = budget["completed_tokens"][-2000:]
                result = self._snapshot(state, stamp)
                _atomic_write(self.status_path, _status_html(result))
                return result
            if route == "official":
                budget["consecutive_errors"] = 0
            stats = state["day_stats"].setdefault(day, {"routes": {}, "reasons": {}})
            provider_stats = stats.setdefault(
                "providers",
                {
                    "anthropic": {
                        "routes": dict(stats.get("routes", {})),
                        "reasons": dict(stats.get("reasons", {})),
                    }
                },
            ).setdefault(self.provider, {"routes": {}, "reasons": {}})
            provider_stats["routes"][route] = provider_stats["routes"].get(route, 0) + 1
            provider_stats["reasons"][reason] = (
                provider_stats["reasons"].get(reason, 0) + 1
            )
            if fallback_from:
                provider_stats["fallback_count"] = (
                    provider_stats.get("fallback_count", 0) + 1
                )
                reasons = provider_stats.setdefault("fallback_reasons", {})
                reasons[fallback_reason] = reasons.get(fallback_reason, 0) + 1
            stats["routes"][route] = stats["routes"].get(route, 0) + 1
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            state["recent_turns"].append(
                {
                    "time": stamp.isoformat(),
                    "route": route,
                    "provider": self.provider if route == "official" else "",
                    "fallback_from": fallback_from,
                    "fallback_reason": fallback_reason,
                    "reason": reason,
                    "official_usd": amount,
                    "ttft_seconds": ttft,
                    "gate_lines": max(0, int(gate_lines)),
                    "request_sha": sha,
                    "check_seconds": checks,
                }
            )
            state["recent_turns"] = state["recent_turns"][-20:]
            if token:
                budget["completed_tokens"].append(token)
                budget["completed_tokens"] = budget["completed_tokens"][-2000:]
            result = self._snapshot(state, stamp)
            _atomic_write(self.status_path, _status_html(result))
            return result

        return await self._transaction(settle, now)

    async def render_status(self, now=None):
        def render(state, snapshot, stamp):
            _atomic_write(self.status_path, _status_html(snapshot))
            return self.status_path

        return await self._transaction(render, now)


# Production sample: about 0.82 prompt tokens per character of the JSON request.
TOKENS_PER_REQUEST_CHAR = 0.9
# Observed official outputs, thinking included, stayed under 500 tokens.
TURN_OUTPUT_RESERVE_TOKENS = 4000


def estimate_turn_reserve(
    payload_chars: int,
    output_tokens: int = TURN_OUTPUT_RESERVE_TOKENS,
    *,
    model: str = "claude-opus-4-6",
    provider: str = "anthropic",
) -> float:
    """One cold request priced from characters, not UTF-8 bytes.

    Bytes counted each Chinese character three times and the reserve was taken
    twice, so a ~$0.03 turn held ~$1.9 and days stopped at about a third of
    their share. Tool loops and long outputs settle from the ledger afterwards.
    """
    if payload_chars < 0 or output_tokens < 0:
        raise ValueError("reserve_inputs_must_be_nonnegative")
    if provider not in {"anthropic", "openai"}:
        raise ValueError("provider_must_be_anthropic_or_openai")
    profile = get_openai_profile(model) if provider == "openai" else get_profile(model)
    input_tokens = payload_chars * TOKENS_PER_REQUEST_CHAR
    input_multiplier, output_multiplier = (
        profile.price_multipliers(input_tokens) if provider == "openai" else (1.0, 1.0)
    )
    return (
        input_tokens
        * max(profile.input_usd, profile.cache_write_usd)
        * input_multiplier
        + output_tokens * profile.output_usd * output_multiplier
    ) / 1_000_000


def estimate_two_request_reserve(
    payload_bytes: int, max_tokens: int = 16000, *, model: str = "claude-opus-4-6"
) -> float:
    """Byte-based reserve, taken twice; the router uses ``estimate_turn_reserve``."""
    if payload_bytes < 0 or max_tokens < 0:
        raise ValueError("reserve_inputs_must_be_nonnegative")
    profile = get_profile(model)
    return (
        2
        * (
            payload_bytes * max(profile.input_usd, profile.cache_write_usd)
            + max_tokens * profile.output_usd
        )
        / 1_000_000
    )
