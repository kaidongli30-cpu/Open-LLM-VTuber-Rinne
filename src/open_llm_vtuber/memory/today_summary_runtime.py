"""Background daily summaries.  Never trims or injects the reply history."""

from __future__ import annotations

import hashlib
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from .message_time_context import memory_day_for_timestamp
from .openai_compatible_stream import request_chat_completion_text
from .today_life_summary import (
    SUMMARY_PROMPT_VERSION,
    SummaryValidationError,
    build_summary_request,
    checkpoint,
    source_digest,
    validate_plain_summary,
)
from .today_summary_store import TodaySummaryStore, load_day_messages, summary_lock


def request_summary_text(messages, provider: dict[str, Any], settings, *, review=False):
    base = str(provider["base_url"]).rstrip("/")
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    model = settings.model or provider["model"]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": settings.max_output_tokens,
        "thinking": {
            "type": "disabled" if settings.reasoning_effort == "none" else "enabled"
        },
        "reasoning_effort": settings.reasoning_effort,
    }
    text = request_chat_completion_text(
        url=url,
        headers={
            "Authorization": "Bearer " + provider["llm_api_key"],
            "Content-Type": "application/json",
        },
        payload=payload,
        output_label="今日摘要复核" if review else "今日摘要初稿",
        output_token_limit=settings.max_output_tokens,
        read_timeout_seconds=settings.timeout_seconds,
        max_attempts=1,
        report=lambda _message: None,
        proxy_url=provider.get("proxy_url"),
    )
    if not text:
        raise SummaryValidationError("provider_no_complete_content")
    return text


def summary_provider(character_config) -> dict[str, Any]:
    """Resolve only the independent configured provider, never the brain."""
    settings = character_config.today_summary
    configured = getattr(
        character_config.agent_config.llm_configs, settings.llm_provider, None
    )
    if configured is None:
        raise SummaryValidationError("summary_provider_missing")
    provider = (
        configured.model_dump()
        if hasattr(configured, "model_dump")
        else dict(configured)
    )
    if (
        not provider.get("base_url")
        or not provider.get("llm_api_key")
        or not (settings.model or provider.get("model"))
    ):
        raise SummaryValidationError("summary_provider_incomplete")
    return provider


def generate_checkpoint(
    root: Path,
    character_config,
    *,
    memory_day: str | None = None,
    call=request_summary_text,
) -> str:
    settings = character_config.today_summary
    if not settings.enabled:
        return "disabled"
    day = memory_day or memory_day_for_timestamp(datetime.now()).isoformat()
    with summary_lock(root, day):
        source = load_day_messages(root, day)
        turns, count = checkpoint(source, settings.interval_turns)
        if not count:
            return "below_checkpoint"
        covered = source[:count]
        store = TodaySummaryStore(root)
        existing = store.load(day, source)
        if existing and existing["covered_messages"] >= count:
            return "current"
        provider = summary_provider(character_config)
        model = settings.model or provider["model"]
        signature = hashlib.sha256(
            (
                source_digest(covered)
                + SUMMARY_PROMPT_VERSION
                + model
                + str(provider["base_url"])
                + str(settings.max_output_tokens)
                + settings.reasoning_effort
            ).encode("utf-8")
        ).hexdigest()
        previous = store.previous_attempt(day)
        if previous.get("signature") == signature and previous.get("status") in {
            "started",
            "failed",
        }:
            return "previous_attempt_not_repeated"
        attempt = {
            "signature": signature,
            "memory_day": day,
            "turns": turns,
            "model": model,
            "status": "started",
        }
        # Persist intent before any billable work, also checking write access.
        store.record_attempt(day, attempt)
        started = time.perf_counter()
        logger.info(
            "[今日摘要] 开始：记忆日={}，覆盖={}轮/{}条，模型={}",
            day,
            turns,
            count,
            model,
        )
        try:
            initial = call(
                build_summary_request(covered, day), provider, settings, review=False
            )
            summary = validate_plain_summary(initial)
            elapsed = round(time.perf_counter() - started, 3)
            store.save(day, covered, summary, turns=turns, model=model, seconds=elapsed)
            attempt.update(status="saved", seconds=elapsed)
            store.record_attempt(day, attempt)
            logger.info(
                "[今日摘要] 已保存：{}，{}字符，耗时{:.1f}秒；对话历史未压缩",
                day,
                len(summary["text"]),
                elapsed,
            )
            return "saved"
        except Exception as exc:
            code = (
                str(exc)
                if isinstance(exc, SummaryValidationError)
                else type(exc).__name__
            )
            attempt.update(
                status="failed",
                error=code,
                seconds=round(time.perf_counter() - started, 3),
            )
            store.record_attempt(day, attempt)
            logger.warning(
                "[今日摘要] 本次未发布：{}，原因={}；保留上一版，同一检查点不重复调用",
                day,
                code,
            )
            return "failed"


_guard = threading.Lock()
_running: set[str] = set()
_latest: dict[str, tuple[Path, Any]] = {}


def request_today_summary(root: Path | str, character_config) -> None:
    """Return immediately; desktop and private QQ use the same entry point."""
    settings = getattr(character_config, "today_summary", None)
    if settings is None or not settings.enabled:
        return
    path = Path(root).resolve()
    key = str(path).casefold()
    with _guard:
        _latest[key] = (path, character_config)
        if key in _running:
            return
        _running.add(key)

    def worker():
        try:
            while True:
                with _guard:
                    job = _latest.pop(key, None)
                if job is None:
                    break
                try:
                    status = generate_checkpoint(*job)
                    # Reload after a successful generation: new turns may have
                    # arrived while the API was busy, including another process.
                    if status == "saved":
                        with _guard:
                            _latest.setdefault(key, job)
                except OSError:
                    logger.debug(
                        "[今日摘要] 检查点由另一个入口处理，或数据目录暂不可写"
                    )
                except Exception as exc:
                    logger.warning(
                        "[今日摘要] 后台任务不可用：{}；不影响对话", type(exc).__name__
                    )
        finally:
            with _guard:
                _running.discard(key)
                pending = _latest.pop(key, None)
            if pending is not None:
                request_today_summary(*pending)

    try:
        threading.Thread(target=worker, name="rinne-today-summary", daemon=True).start()
    except Exception:
        with _guard:
            _running.discard(key)
        raise
