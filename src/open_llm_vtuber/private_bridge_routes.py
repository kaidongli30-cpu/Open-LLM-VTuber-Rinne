"""FastAPI routes for the loopback-only AstrBot-to-Rinne bridge."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import AsyncIterator

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from starlette.responses import StreamingResponse

from .agent_runtime.access_policy import ChannelAccessDenied
from .agent_runtime.private_bridge import (
    PrivateBridgeAuthenticationError,
    PrivateBridgeConfigurationError,
    PrivateBridgeNotStarted,
    PrivateChannelBridge,
    is_loopback_host,
)
from .agent_runtime.private_media import PrivateMediaValidationError
from .agent_runtime.text_turn import normalize_qq_response_text

PRIVATE_BRIDGE_HEARTBEAT_SECONDS = 15.0


class PrivateMediaPayload(BaseModel):
    kind: str = Field(pattern="^(image|document|video)$")
    name: str = Field(min_length=1, max_length=255)
    relative_path: str = Field(min_length=1, max_length=255)
    size: int = Field(gt=0, le=100 * 1024 * 1024)
    sha256: str = Field(pattern="^[0-9a-fA-F]{64}$")

    @field_validator("size")
    @classmethod
    def enforce_kind_limit(cls, value: int, info) -> int:
        if info.data.get("kind") != "video" and value > 50 * 1024 * 1024:
            raise ValueError("image and document media must not exceed 50 MB")
        return value


class PrivateTextTurnPayload(BaseModel):
    message_id: str = Field(min_length=1, max_length=256)
    sender_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=50_000)
    media: list[PrivateMediaPayload] = Field(default_factory=list, max_length=16)


class PrivateAudioPayload(BaseModel):
    kind: str = Field(pattern="^audio$")
    name: str = Field(min_length=1, max_length=255)
    relative_path: str = Field(min_length=1, max_length=255)
    size: int = Field(gt=0, le=20 * 1024 * 1024)
    sha256: str = Field(pattern="^[0-9a-fA-F]{64}$")


class PrivateTranscriptionPayload(BaseModel):
    message_id: str = Field(min_length=1, max_length=256)
    sender_id: str = Field(min_length=1, max_length=128)
    audio: PrivateAudioPayload


class PrivateSessionEndPayload(BaseModel):
    message_id: str = Field(min_length=1, max_length=256)
    sender_id: str = Field(min_length=1, max_length=128)


class PrivateHealthPayload(BaseModel):
    sender_id: str = Field(min_length=1, max_length=128)


def _encode_ndjson(event: dict) -> bytes:
    return (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


_PRIVATE_TRANSIENT_TEXT = {"Thinking...", "Connection established"}
_PRIVATE_INTERNAL_ERROR_PREFIXES = (
    "Error calling the chat endpoint:",
    "See the logs for details.",
    "Troubleshooting with documentation:",
)
def _private_visible_sentence(text: object) -> str | None:
    """Keep internal status out of QQ while preserving desktop-style expression."""

    if not isinstance(text, str):
        return None
    sentence = text.strip()
    if not sentence or sentence in _PRIVATE_TRANSIENT_TEXT:
        return None
    if sentence.startswith(_PRIVATE_INTERNAL_ERROR_PREFIXES):
        return None
    return normalize_qq_response_text(sentence).strip() or None


def _is_private_internal_error(text: object) -> bool:
    return isinstance(text, str) and text.strip().startswith(
        _PRIVATE_INTERNAL_ERROR_PREFIXES
    )


def _require_private_request(
    bridge: PrivateChannelBridge,
    request: Request,
    authorization: str | None,
    sender_id: str,
) -> None:
    peer_host = request.client.host if request.client else None
    if not is_loopback_host(peer_host):
        raise HTTPException(status_code=403, detail="loopback access required")
    try:
        bridge.config.require_bearer(authorization)
    except PrivateBridgeAuthenticationError as exc:
        raise HTTPException(status_code=401, detail="unauthorized") from exc
    if not sender_id:
        raise HTTPException(status_code=422, detail="blank sender field")
    try:
        bridge.require_sender(sender_id)
    except ChannelAccessDenied as exc:
        raise HTTPException(status_code=403, detail="sender not allowed") from exc
    if not bridge.started:
        raise HTTPException(status_code=503, detail="bridge not ready")


def init_private_bridge_routes(bridge: PrivateChannelBridge) -> APIRouter:
    router = APIRouter(prefix="/agent/private/v1")

    @router.post("/turn")
    async def run_private_turn(
        payload: PrivateTextTurnPayload,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> StreamingResponse:
        sender_id = payload.sender_id.strip()
        message_id = payload.message_id.strip()
        text = payload.text.strip()
        _require_private_request(bridge, request, authorization, sender_id)
        if not message_id or not text:
            raise HTTPException(status_code=422, detail="blank turn field")

        async def stream_events() -> AsyncIterator[bytes]:
            queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=64)
            failure_code: str | None = None

            async def report_failure(code: str) -> None:
                nonlocal failure_code
                if failure_code is None:
                    failure_code = code
                    await queue.put({"type": "error", "code": code})

            async def observe(event: dict) -> None:
                event_type = event.get("type")
                if event_type == "full-text":
                    if _is_private_internal_error(event.get("text")):
                        await report_failure("backend_generation_failed")
                        return
                    sentence = _private_visible_sentence(event.get("text"))
                    if sentence and failure_code is None:
                        await queue.put({"type": "sentence", "text": sentence})
                elif event_type == "error":
                    await report_failure("backend_error")

            async def execute() -> None:
                try:
                    turn_kwargs = {
                        "sender_id": sender_id,
                        "message_id": message_id,
                        "text": text,
                        "event_observer": observe,
                    }
                    if payload.media:
                        turn_kwargs["media"] = tuple(
                            item.model_dump() for item in payload.media
                        )
                    response = await bridge.run_text_turn(**turn_kwargs)
                    if failure_code is None:
                        await queue.put(
                            {"type": "done", "response_chars": len(response or "")}
                        )
                except PrivateBridgeNotStarted:
                    await report_failure("bridge_not_ready")
                except Exception:
                    await report_failure("turn_failed")
                finally:
                    await queue.put(None)

            task = asyncio.create_task(execute())
            try:
                # Flush the streaming response immediately, then keep the private
                # transport alive while the shared desktop/QQ turn pipeline is
                # preparing and analyzing long video attachments.
                yield _encode_ndjson({"type": "heartbeat"})
                while True:
                    try:
                        event = await asyncio.wait_for(
                            queue.get(), timeout=PRIVATE_BRIDGE_HEARTBEAT_SECONDS
                        )
                    except asyncio.TimeoutError:
                        yield _encode_ndjson({"type": "heartbeat"})
                        continue
                    if event is None:
                        break
                    yield _encode_ndjson(event)
            finally:
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        return StreamingResponse(
            stream_events(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store", "X-Rinne-Bridge-Version": "4"},
        )

    @router.post("/transcribe")
    async def transcribe_private_audio(
        payload: PrivateTranscriptionPayload,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, str]:
        sender_id = payload.sender_id.strip()
        _require_private_request(bridge, request, authorization, sender_id)
        try:
            text = await bridge.transcribe_audio(
                sender_id=sender_id,
                audio=payload.audio.model_dump(),
            )
        except PrivateMediaValidationError as exc:
            raise HTTPException(status_code=422, detail="invalid audio") from exc
        except (PrivateBridgeNotStarted, PrivateBridgeConfigurationError) as exc:
            raise HTTPException(status_code=503, detail="ASR not ready") from exc
        return {"text": text}

    @router.post("/session/end")
    async def end_private_session(
        payload: PrivateSessionEndPayload,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, bool]:
        sender_id = payload.sender_id.strip()
        _require_private_request(bridge, request, authorization, sender_id)
        try:
            closed = await bridge.end_history_session(sender_id=sender_id)
        except PrivateBridgeNotStarted as exc:
            raise HTTPException(status_code=503, detail="bridge not ready") from exc
        return {"closed": closed}

    @router.post("/health")
    async def private_health(
        payload: PrivateHealthPayload,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict:
        sender_id = payload.sender_id.strip()
        _require_private_request(bridge, request, authorization, sender_id)
        try:
            return bridge.health_snapshot()
        except PrivateBridgeNotStarted as exc:
            raise HTTPException(status_code=503, detail="bridge not ready") from exc

    return router


__all__ = [
    "PrivateAudioPayload",
    "PrivateHealthPayload",
    "PrivateMediaPayload",
    "PrivateSessionEndPayload",
    "PrivateTextTurnPayload",
    "PrivateTranscriptionPayload",
    "init_private_bridge_routes",
]
