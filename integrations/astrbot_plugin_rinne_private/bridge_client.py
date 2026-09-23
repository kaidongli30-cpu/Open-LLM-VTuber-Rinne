"""Loopback-only NDJSON client for Rinne's private backend bridge."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping
from urllib.parse import urlsplit

import aiohttp


BRIDGE_TOKEN_ENV = "RINNE_PRIVATE_BRIDGE_TOKEN"


class BridgeClientConfigurationError(ValueError):
    pass


class BridgeTurnError(RuntimeError):
    pass


def _require_loopback_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}:
        raise BridgeClientConfigurationError("bridge URL must use loopback HTTP")
    if parsed.username or parsed.password or not parsed.port:
        raise BridgeClientConfigurationError("bridge URL must contain only host and port")
    if parsed.path != "/agent/private/v1/turn" or parsed.query or parsed.fragment:
        raise BridgeClientConfigurationError("bridge URL path is invalid")
    return url.strip()


@dataclass(frozen=True, slots=True)
class BridgeClientConfig:
    url: str
    token_environment_name: str = BRIDGE_TOKEN_ENV

    @classmethod
    def from_values(cls, url: str) -> "BridgeClientConfig":
        return cls(url=_require_loopback_url(url))

    def resolve_token(self, environ: Mapping[str, str] | None = None) -> str:
        values = os.environ if environ is None else environ
        token = values.get(self.token_environment_name, "").strip()
        if len(token) < 32:
            raise BridgeClientConfigurationError(
                f"{self.token_environment_name} is missing or too short"
            )
        return token


class RinneBridgeClient:
    def __init__(self, config: BridgeClientConfig) -> None:
        self.config = config
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=300, connect=5, sock_read=180)
            self._session = aiohttp.ClientSession(timeout=timeout, trust_env=False)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    def _sibling_url(self, suffix: str) -> str:
        return self.config.url.rsplit("/", 1)[0] + suffix

    async def transcribe_audio(
        self,
        sender_id: str,
        message_id: str,
        audio: dict[str, Any],
    ) -> str:
        session = await self._get_session()
        token = self.config.resolve_token()
        async with session.post(
            self._sibling_url("/transcribe"),
            json={
                "sender_id": sender_id,
                "message_id": message_id,
                "audio": audio,
            },
            headers={"Authorization": f"Bearer {token}"},
        ) as response:
            if response.status != 200:
                raise BridgeTurnError(
                    f"bridge transcription returned HTTP {response.status}"
                )
            try:
                payload = await response.json()
            except (aiohttp.ContentTypeError, ValueError) as exc:
                raise BridgeTurnError(
                    "bridge transcription returned invalid JSON"
                ) from exc
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str):
            raise BridgeTurnError("bridge transcription text is invalid")
        return text.strip()

    async def end_session(self, sender_id: str, message_id: str) -> bool:
        """Seal one canonical QQ history JSON without stopping the bot."""

        session = await self._get_session()
        token = self.config.resolve_token()
        async with session.post(
            self._sibling_url("/session/end"),
            json={"sender_id": sender_id, "message_id": message_id},
            headers={"Authorization": f"Bearer {token}"},
        ) as response:
            if response.status != 200:
                raise BridgeTurnError(
                    f"bridge session end returned HTTP {response.status}"
                )
            try:
                payload = await response.json()
            except (aiohttp.ContentTypeError, ValueError) as exc:
                raise BridgeTurnError("bridge session end returned invalid JSON") from exc
        closed = payload.get("closed") if isinstance(payload, dict) else None
        if not isinstance(closed, bool):
            raise BridgeTurnError("bridge session end status is invalid")
        return closed

    async def health(self, sender_id: str) -> dict[str, Any]:
        """Check the authenticated local bridge without calling the LLM."""

        session = await self._get_session()
        token = self.config.resolve_token()
        try:
            async with session.post(
                self._sibling_url("/health"),
                json={"sender_id": sender_id},
                headers={"Authorization": f"Bearer {token}"},
            ) as response:
                if response.status != 200:
                    raise BridgeTurnError(
                        f"bridge health returned HTTP {response.status}"
                    )
                try:
                    payload = await response.json()
                except (aiohttp.ContentTypeError, ValueError) as exc:
                    raise BridgeTurnError(
                        "bridge health returned invalid JSON"
                    ) from exc
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise BridgeTurnError("bridge health request failed") from exc
        if not isinstance(payload, dict) or not isinstance(
            payload.get("ready_to_reply"), bool
        ):
            raise BridgeTurnError("bridge health status is invalid")
        return payload

    async def stream_turn(
        self,
        sender_id: str,
        message_id: str,
        text: str,
        media: tuple[dict[str, Any], ...] = (),
    ) -> AsyncIterator[str]:
        session = await self._get_session()
        token = self.config.resolve_token()
        payload = {
            "sender_id": sender_id,
            "message_id": message_id,
            "text": text,
            "media": list(media),
        }
        done = False
        turn_timeout = aiohttp.ClientTimeout(total=None, connect=5, sock_read=45)
        try:
            async with session.post(
                self.config.url,
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
                timeout=turn_timeout,
            ) as response:
                if response.status != 200:
                    raise BridgeTurnError(f"bridge returned HTTP {response.status}")
                async for raw_line in response.content:
                    if len(raw_line) > 65_536:
                        raise BridgeTurnError("bridge event is too large")
                    try:
                        event = json.loads(raw_line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise BridgeTurnError("bridge returned invalid NDJSON") from exc
                    if not isinstance(event, dict):
                        raise BridgeTurnError("bridge event must be an object")
                    event_type = event.get("type")
                    if event_type == "sentence":
                        if done:
                            raise BridgeTurnError("bridge sent a sentence after done")
                        sentence = event.get("text")
                        if not isinstance(sentence, str) or not sentence.strip():
                            raise BridgeTurnError("bridge sentence is invalid")
                        yield sentence.strip()
                    elif event_type == "heartbeat":
                        if done:
                            raise BridgeTurnError("bridge sent a heartbeat after done")
                    elif event_type == "done":
                        if done:
                            raise BridgeTurnError("bridge sent duplicate done events")
                        done = True
                    elif event_type == "error":
                        raise BridgeTurnError("Rinne backend could not finish the turn")
                    else:
                        raise BridgeTurnError("bridge event type is invalid")
        except asyncio.CancelledError:
            raise
        except BridgeTurnError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise BridgeTurnError("bridge turn request failed") from exc
        if not done:
            raise BridgeTurnError("bridge stream ended before done")
