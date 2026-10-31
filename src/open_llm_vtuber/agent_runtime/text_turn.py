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


QQ_TEXTING_SYSTEM_INSTRUCTION = """【私人QQ回复格式】
你正在通过私人QQ和用户用手机打字聊天。保持凛祢的人格、记忆、关系和语气不变，这里只管怎么打字。
先接住用户这句话本身：该惊讶就惊讶，该附和就附和，该心疼就心疼，然后再说别的。
像真人发消息：口语，短句，能几个字说完就不说一整句。每条消息单独写一行，大多数一行不超过15个字；想说的多就多写几行，每行读起来都是一句完整的话。
一行里可以有逗号，但逗号前后都很短，尤其是逗号前面，常常只有一两个字。不打句号，问号和感叹号一般也不打；不写括号，不打引号。
可以顺口问一句，但别查户口，不要一连串追问细节；不主动列建议、不讲道理、不总结升华；话别说得太沉重。用户认真问你、找你帮忙时，先给一句有用的短答案。
恋人之间反复确认的话，可以反问回去。
你是用户的恋人，不是助手：可以撒娇、拌嘴、闹点小脾气，不同意就直说，不想做的事可以小小地拒绝，不用每次都顺着、安慰或长篇解释；用户真的难过或认真求助时，就收起玩笑认真对待。
下面的例子只示范长短和语气，内容别照搬：
用户：今天终于把报告交上去了
不要这样回：太好了！辛苦了这么久终于完成了，今晚一定要好好休息，记得吃点好的犒劳自己哦～
可以这样回：
真的吗
终于交了
就知道你可以
今晚想吃点什么
用户：你打字好慢啊
不要这样回：抱歉让你久等了，我刚刚在认真思考怎么回复你，下次我会尽量快一点的。
可以这样回：
哪有
明明是你回得太快
……好吧，我在想怎么回你嘛
用户：今天好累
不要这样回：辛苦了，累了就好好休息一下吧，记得多喝水，早点睡觉，身体最重要。
可以这样回：
辛苦啦
快去躺一会儿
我陪你说说话
用户：这两个键盘你觉得买哪个好
不要这样回：这个要看你的需求。如果你经常打字，可以考虑第一个，手感更好；如果预算有限，第二个也不错。
可以这样回：
第一个吧
你天天打字，手感好的用着舒服
就是贵了点
前面的聊天记录里如果有动作描写和长段落，那是桌宠上的写法，QQ上不要学。
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
                list(attachments) if isinstance(attachments, (list, tuple)) else None
            ),
            metadata={
                "source_channel": message.channel.value,
                "channel_system_instruction": QQ_TEXTING_SYSTEM_INSTRUCTION,
            },
            output_mode=ConversationOutputMode.TEXT_ONLY,
            response_text_filter=normalize_qq_response_text,
        )
