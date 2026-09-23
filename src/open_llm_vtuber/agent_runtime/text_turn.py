"""Bridge authenticated channel text into Rinne's existing conversation core."""

from __future__ import annotations

import json
import re
from typing import Awaitable, Callable

from ..conversations.single_conversation import process_single_conversation
from ..conversations.types import ConversationOutputMode
from ..service_context import ServiceContext
from .message import InboundChannelMessage


ConversationEventObserver = Callable[[dict], Awaitable[None]]


QQ_DESKTOP_STYLE_SYSTEM_INSTRUCTION = """【私人QQ回复格式】
你正在通过私人QQ与用户进行真实的手机文字聊天。保持凛祢的人格、记忆、关系和语气不变。
继续严格采用与电脑桌宠相同的回复格式：保留[happy]等已登记的表情标签，也可以自然使用全角括号包围的动作、神态或场景描写。
“发送消息”和“正在输入”只是QQ通信过程，不是凛祢需要描写的动作；不要在回复开头或每条回复中添加“（发消息）”“（发送消息）”或同义占位描写。
不要解释、复述或提及这些格式规则。"""


_LEADING_QQ_SEND_ACTION = re.compile(
    r"^(?P<prefix>\s*(?:\[[^\]\r\n]{1,32}\]\s*)*)"
    r"[（(]\s*(?:发|发送)消息\s*[）)]\s*"
)


def normalize_qq_response_text(text: str) -> str:
    """Remove only the obsolete leading QQ send-action placeholder."""

    return _LEADING_QQ_SEND_ACTION.sub(r"\g<prefix>", text, count=1)


class RinneTextTurnProcessor:
    """Run a remote text turn with the same persona, tools and memory pipeline.

    Platform identifiers are deliberately not copied into model metadata.  The
    surrounding channel runtime retains them for access control and delivery.
    """

    def __init__(
        self,
        *,
        context: ServiceContext,
        client_uid: str,
        event_observer: ConversationEventObserver | None = None,
    ) -> None:
        if not client_uid.strip():
            raise ValueError("client_uid must not be blank")
        self._context = context
        self._client_uid = client_uid
        self._event_observer = event_observer

    async def __call__(self, message: InboundChannelMessage) -> str:
        async def observe_event(payload: str) -> None:
            if self._event_observer is None:
                return
            try:
                event = json.loads(payload)
            except (TypeError, json.JSONDecodeError):
                return
            if isinstance(event, dict):
                await self._event_observer(event)

        images = message.metadata.get("images")
        attachments = message.metadata.get("attachments")
        return await process_single_conversation(
            context=self._context,
            websocket_send=observe_event,
            client_uid=self._client_uid,
            user_input=message.text,
            images=list(images) if isinstance(images, (list, tuple)) else None,
            attachments=(
                list(attachments)
                if isinstance(attachments, (list, tuple))
                else None
            ),
            metadata={
                "source_channel": message.channel.value,
                "channel_system_instruction": QQ_DESKTOP_STYLE_SYSTEM_INSTRUCTION,
            },
            output_mode=ConversationOutputMode.TEXT_ONLY,
            response_text_filter=normalize_qq_response_text,
        )
