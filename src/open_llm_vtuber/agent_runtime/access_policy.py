"""Default-deny identity policy for remote conversation channels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .message import AgentChannel


class ChannelAccessDenied(PermissionError):
    """Raised when a remote sender is not on the exact channel allowlist."""


@dataclass(frozen=True, slots=True)
class ChannelIdentityPolicy:
    """Allow only explicitly paired sender identifiers on each channel.

    Platform identifiers are intentionally case-sensitive.  They are opaque
    values, so changing case or accepting partial matches could authorize a
    different account.
    """

    allowed_senders: Mapping[AgentChannel, frozenset[str]] = field(
        default_factory=dict
    )

    @classmethod
    def from_mapping(
        cls,
        allowed_senders: Mapping[AgentChannel | str, Iterable[str]],
    ) -> "ChannelIdentityPolicy":
        normalized: dict[AgentChannel, frozenset[str]] = {}
        for channel_value, sender_values in allowed_senders.items():
            channel = AgentChannel(channel_value)
            senders = frozenset(
                sender.strip() for sender in sender_values if sender.strip()
            )
            if senders:
                normalized[channel] = senders
        return cls(allowed_senders=normalized)

    def is_allowed(self, channel: AgentChannel, sender_id: str) -> bool:
        sender = sender_id.strip()
        return bool(sender) and sender in self.allowed_senders.get(
            channel, frozenset()
        )

    def require_allowed(self, channel: AgentChannel, sender_id: str) -> None:
        if not self.is_allowed(channel, sender_id):
            raise ChannelAccessDenied(
                f"sender is not paired for channel {channel.value}"
            )
