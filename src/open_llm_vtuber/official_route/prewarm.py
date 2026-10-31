"""Schedule background route priming using side-effect-free memory readers."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from ..memory.layer2_context import load_current_layer2_context
from ..memory.mandatory_response_rules import load_mandatory_response_rules
from ..memory.today_history_loader import TodayHistoryLoader


def schedule_prewarm(runtime, agent):
    """Return a runtime-owned task, or defer safely until called in a loop.

    Creating an agent before the server event loop starts must not create an
    unawaited coroutine, a new event loop, or a worker thread. The caller can
    call this again from its next async lifecycle hook; ``_prewarm_pending``
    records that the earlier call was deferred. Classifier retries remain the
    runtime's responsibility, including its 1/2/5/10-minute backoff.
    """
    if runtime is None or not getattr(runtime, "enabled", True):
        return None
    if getattr(runtime, "_shutdown", False):
        return None
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        runtime._prewarm_pending = True
        return None

    coroutine = prewarm(runtime, agent)
    try:
        task = runtime._spawn(coroutine)
    except Exception:
        coroutine.close()
        runtime._prewarm_pending = True
        return None
    runtime._prewarm_pending = False

    def completed(finished):
        # Retrieve unexpected task errors without logging exception text, which
        # could include private loader data. A later lifecycle call may retry.
        if finished.cancelled() or finished.exception() is not None:
            runtime._prewarm_pending = True

    task.add_done_callback(completed)
    return task


def _read_contexts(history_root, reference_time, include_history):
    layer2 = load_current_layer2_context(history_root).context
    rules = load_mandatory_response_rules(history_root).context
    memory = []
    if include_history:
        loaded = TodayHistoryLoader(history_root).load(reference_time=reference_time)
        memory = [
            {
                "role": turn.role,
                "content": turn.content,
                "timestamp": turn.timestamp.isoformat(),
            }
            for turn in loaded.turns
        ]
    return layer2, rules, memory


async def prewarm(runtime, agent):
    """Prime current backgrounds/history without waiting for model checks.

    The proxy snapshots agent fields; neither the real agent nor its memory is
    modified. In particular, this module never loads recent diary context,
    because that loader can write the diary registration JSON.
    """
    if runtime is None or not getattr(runtime, "enabled", True):
        return
    if getattr(runtime, "_shutdown", False):
        return
    reference_time = datetime.now().astimezone()
    history_root = Path(runtime.data_dir).parent
    memory = deepcopy(getattr(agent, "_memory", []) or [])
    proxy = SimpleNamespace(
        _system=getattr(agent, "_system", ""),
        _recent_memory_context=getattr(agent, "_recent_memory_context", ""),
        _memory=memory,
        _to_text_prompt=lambda _input: "",
    )
    if hasattr(agent, "_recent"):
        proxy._recent = getattr(agent, "_recent")
    layer2, rules, today = await asyncio.to_thread(
        _read_contexts, history_root, reference_time, not memory
    )
    if not memory:
        proxy._memory = today
    input_data = SimpleNamespace(
        metadata={
            "layer2_user_background": layer2,
            "mandatory_response_rules": rules,
        },
        images=None,
    )
    await runtime.prime(proxy, input_data)


__all__ = ["prewarm", "schedule_prewarm"]
