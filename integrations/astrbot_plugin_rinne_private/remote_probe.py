"""Silent QQ-server round-trip probe for the private OneBot transport."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Protocol

from .transport_health import TransportHealthStore


class OneBotActionClient(Protocol):
    async def call_action(self, action: str, **params: object) -> object: ...


class OneBotRemoteProbe:
    """Verify the upstream QQ session without sending a visible message."""

    def __init__(
        self,
        *,
        get_bot: Callable[[], OneBotActionClient | None],
        sender_id: str,
        health_store: TransportHealthStore,
        timeout_seconds: float = 15.0,
        healthy_interval_seconds: float = 60.0,
        failure_interval_seconds: float = 15.0,
    ) -> None:
        if not sender_id.isdecimal():
            raise ValueError("sender_id must be a decimal QQ identifier")
        if min(
            timeout_seconds,
            healthy_interval_seconds,
            failure_interval_seconds,
        ) <= 0:
            raise ValueError("probe intervals must be positive")
        self._get_bot = get_bot
        self._sender_id = sender_id
        self._health_store = health_store
        self._timeout_seconds = timeout_seconds
        self._healthy_interval_seconds = healthy_interval_seconds
        self._failure_interval_seconds = failure_interval_seconds

    async def probe_once(self) -> bool:
        bot = self._get_bot()
        if bot is None:
            self._health_store.record_remote_probe_failure("adapter_unavailable")
            return False
        try:
            response = await asyncio.wait_for(
                bot.call_action(
                    "get_stranger_info",
                    user_id=self._sender_id,
                    no_cache=True,
                ),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError:
            self._health_store.record_remote_probe_failure("timeout")
            return False
        except (ConnectionError, OSError):
            self._health_store.record_remote_probe_failure("connection")
            return False
        except asyncio.CancelledError:
            raise
        except Exception:
            self._health_store.record_remote_probe_failure("action_failed")
            return False

        if not self._is_expected_response(response):
            self._health_store.record_remote_probe_failure("invalid_response")
            return False
        self._health_store.record_remote_probe_success()
        return True

    async def run(self) -> None:
        while True:
            healthy = await self.probe_once()
            await asyncio.sleep(
                self._healthy_interval_seconds
                if healthy
                else self._failure_interval_seconds
            )

    def _is_expected_response(self, response: object) -> bool:
        if not isinstance(response, dict):
            return False
        returned_id: Any = response.get("user_id")
        return str(returned_id).strip() == self._sender_id
