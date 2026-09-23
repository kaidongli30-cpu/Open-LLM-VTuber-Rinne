"""Authenticated and serialized processing for remote text channels."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, TypeVar

from .access_policy import ChannelIdentityPolicy
from .message import InboundChannelMessage


_ResultT = TypeVar("_ResultT")
TurnProcessor = Callable[[InboundChannelMessage], Awaitable[str]]
TextDeliverer = Callable[[InboundChannelMessage, str], Awaitable[None]]


class ChannelTurnStatus(str, Enum):
    DELIVERED = "delivered"
    DUPLICATE = "duplicate"
    EMPTY = "empty"


@dataclass(frozen=True, slots=True)
class ChannelTurnResult:
    status: ChannelTurnStatus
    response_text: str


@dataclass(slots=True)
class _ProcessedResponse:
    response_text: str
    delivered: bool = False


class ConversationTurnCoordinator:
    """Serialize turns that share Rinne's stateful conversation brain."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def run(
        self,
        operation: Callable[[], Awaitable[_ResultT]],
    ) -> _ResultT:
        async with self._lock:
            return await operation()


class ChannelMessageRuntime:
    """Apply identity, duplicate-delivery and global turn-order guarantees."""

    def __init__(
        self,
        *,
        identity_policy: ChannelIdentityPolicy,
        turn_processor: TurnProcessor,
        text_deliverer: TextDeliverer,
        coordinator: ConversationTurnCoordinator | None = None,
        duplicate_cache_size: int = 512,
    ) -> None:
        if duplicate_cache_size < 1:
            raise ValueError("duplicate_cache_size must be positive")
        self._identity_policy = identity_policy
        self._turn_processor = turn_processor
        self._text_deliverer = text_deliverer
        self._coordinator = coordinator or ConversationTurnCoordinator()
        self._duplicate_cache_size = duplicate_cache_size
        self._processed: OrderedDict[
            tuple[str, str, str], _ProcessedResponse
        ] = OrderedDict()

    async def handle_message(
        self,
        message: InboundChannelMessage,
    ) -> ChannelTurnResult:
        self._identity_policy.require_allowed(message.channel, message.sender_id)

        async def process_serialized() -> ChannelTurnResult:
            return await self._handle_serialized(message)

        return await self._coordinator.run(process_serialized)

    async def _handle_serialized(
        self,
        message: InboundChannelMessage,
    ) -> ChannelTurnResult:
        cache_key = (
            message.channel.value,
            message.sender_id,
            message.message_id,
        )
        processed = self._processed.get(cache_key)
        if processed is not None and processed.delivered:
            self._processed.move_to_end(cache_key)
            return ChannelTurnResult(
                status=ChannelTurnStatus.DUPLICATE,
                response_text=processed.response_text,
            )

        if processed is None:
            response_text = await self._turn_processor(message)
            processed = _ProcessedResponse(response_text=response_text)
            self._processed[cache_key] = processed

        if processed.response_text.strip():
            await self._text_deliverer(message, processed.response_text)
            status = ChannelTurnStatus.DELIVERED
        else:
            status = ChannelTurnStatus.EMPTY

        processed.delivered = True
        self._processed.move_to_end(cache_key)
        self._trim_duplicate_cache()
        return ChannelTurnResult(
            status=status,
            response_text=processed.response_text,
        )

    def _trim_duplicate_cache(self) -> None:
        while len(self._processed) > self._duplicate_cache_size:
            self._processed.popitem(last=False)
