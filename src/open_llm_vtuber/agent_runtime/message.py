"""Channel-neutral message types.

These types deliberately contain no transport credentials.  The QQ adapter
translates a platform event into this small envelope before the message reaches
Rinne's conversation runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping


class AgentChannel(str, Enum):
    """Supported user-facing conversation channels."""

    DESKTOP = "desktop"
    QQ = "qq"


@dataclass(frozen=True, slots=True)
class InboundChannelMessage:
    """One authenticated inbound text message from a conversation channel."""

    channel: AgentChannel
    message_id: str
    sender_id: str
    conversation_id: str
    text: str
    received_at: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("message_id", "sender_id", "conversation_id"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be blank")
        if not self.text.strip():
            raise ValueError("text must not be blank")
        if self.received_at.tzinfo is None:
            raise ValueError("received_at must include a timezone")
