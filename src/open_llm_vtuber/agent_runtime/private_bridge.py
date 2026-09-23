"""Authenticated, loopback-only bridge into Rinne's shared conversation core."""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Protocol

from .access_policy import ChannelIdentityPolicy
from .channel_runtime import ConversationTurnCoordinator
from .message import AgentChannel, InboundChannelMessage
from .private_history import PrivateHistorySessionManager
from .private_media import import_private_audio, import_private_media
from .read_only_config import (
    READ_ONLY_MCP_SERVER_NAME,
    ReadOnlyComputerConfig,
    ReadOnlyComputerConfigurationError,
)
from .text_turn import RinneTextTurnProcessor


PRIVATE_BRIDGE_ENABLED_ENV = "RINNE_PRIVATE_BRIDGE_ENABLED"
PRIVATE_BRIDGE_TOKEN_ENV = "RINNE_PRIVATE_BRIDGE_TOKEN"
PRIVATE_BRIDGE_ALLOWED_SENDERS_ENV = "RINNE_PRIVATE_QQ_ALLOWED_USER_IDS"
PRIVATE_BRIDGE_INBOX_ROOT_ENV = "RINNE_PRIVATE_QQ_INBOX_ROOT"
PRIVATE_BRIDGE_LIBRARY_ROOT_ENV = "RINNE_LIBRARY_ROOT"
PRIVATE_BRIDGE_CLIENT_UID = "rinne-private-qq"
PRIVATE_BRIDGE_MIN_TOKEN_CHARS = 32


class PrivateBridgeConfigurationError(ValueError):
    """Raised when an enabled private bridge is not safely configured."""


class PrivateBridgeNotStarted(RuntimeError):
    """Raised when a turn arrives before the isolated context is ready."""


class PrivateBridgeAuthenticationError(PermissionError):
    """Raised when a bearer credential is missing or incorrect."""


class ContextFactory(Protocol):
    async def __call__(
        self,
        *,
        client_uid: str,
        send_text: Callable[[str], Awaitable[None]],
    ) -> Any: ...


PrivateBridgeEventObserver = Callable[[dict[str, Any]], Awaitable[None]]
ProcessorFactory = Callable[..., RinneTextTurnProcessor]
ContextRefresher = Callable[[Any, str], Awaitable[None]]


def _reconcile_media_labels(
    text: str,
    staged: tuple[Mapping[str, Any], ...],
    saved: list[dict[str, Any]],
) -> str:
    """Make visible QQ labels match the canonical file actually saved."""

    labels = {"image": "图片", "document": "文件", "video": "视频"}
    reconciled = text
    for descriptor, record in zip(staged, saved):
        kind = str(descriptor.get("kind") or "")
        label = labels.get(kind)
        old_name = str(descriptor.get("name") or "").strip()
        new_name = str(record.get("name") or "").strip()
        if label and old_name and new_name and old_name != new_name:
            reconciled = reconciled.replace(
                f"[{label}：{old_name}]",
                f"[{label}：{new_name}]",
                1,
            )
    return reconciled


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise PrivateBridgeConfigurationError("enabled flag must be true or false")


def is_loopback_host(host: str | None) -> bool:
    """Return whether a peer address is an IP loopback address."""

    if not host:
        return False
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class PrivateBridgeConfig:
    enabled: bool = False
    token: str = ""
    allowed_sender_ids: tuple[str, ...] = ()
    inbox_root: Path | None = None
    library_root: Path | None = None

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "PrivateBridgeConfig":
        values = os.environ if environ is None else environ
        enabled = _parse_bool(values.get(PRIVATE_BRIDGE_ENABLED_ENV))
        if not enabled:
            return cls()

        token = values.get(PRIVATE_BRIDGE_TOKEN_ENV, "").strip()
        if len(token) < PRIVATE_BRIDGE_MIN_TOKEN_CHARS:
            raise PrivateBridgeConfigurationError(
                f"{PRIVATE_BRIDGE_TOKEN_ENV} must contain at least "
                f"{PRIVATE_BRIDGE_MIN_TOKEN_CHARS} characters"
            )
        senders = tuple(
            dict.fromkeys(
                item.strip()
                for item in values.get(
                    PRIVATE_BRIDGE_ALLOWED_SENDERS_ENV, ""
                ).split(",")
                if item.strip()
            )
        )
        if not senders:
            raise PrivateBridgeConfigurationError(
                f"{PRIVATE_BRIDGE_ALLOWED_SENDERS_ENV} must contain a sender ID"
            )

        data_root_value = str(values.get("RINNE_DATA_ROOT", "")).strip()
        default_data_root = _require_scoped_absolute_path(
            Path(data_root_value) if data_root_value else Path("G:/Rinne-Agent-Data"),
            environment_name="RINNE_DATA_ROOT",
        )
        inbox_root = _require_private_data_path(
            values.get(PRIVATE_BRIDGE_INBOX_ROOT_ENV),
            default=default_data_root / "qq_private" / "inbox",
            environment_name=PRIVATE_BRIDGE_INBOX_ROOT_ENV,
        )
        library_root = _require_project_content_path(
            values.get(PRIVATE_BRIDGE_LIBRARY_ROOT_ENV),
            default=default_data_root / "rinne_library" / "rinne_01",
            environment_name=PRIVATE_BRIDGE_LIBRARY_ROOT_ENV,
            project_root=default_data_root,
        )
        return cls(
            enabled=True,
            token=token,
            allowed_sender_ids=senders,
            inbox_root=inbox_root,
            library_root=library_root,
        )

    def require_bearer(self, authorization: str | None) -> None:
        prefix = "Bearer "
        candidate = ""
        if authorization and authorization.startswith(prefix):
            candidate = authorization[len(prefix) :]
        if not candidate or not hmac.compare_digest(candidate, self.token):
            raise PrivateBridgeAuthenticationError("invalid bearer credential")


def _require_private_data_path(
    value: str | None,
    *,
    default: Path,
    environment_name: str,
) -> Path:
    candidate = Path(str(value or "").strip() or default).expanduser()
    if not candidate.is_absolute():
        raise PrivateBridgeConfigurationError(
            f"{environment_name} must be an absolute path"
        )
    resolved = candidate.resolve(strict=False)
    if resolved == Path(resolved.anchor).resolve(strict=False):
        raise PrivateBridgeConfigurationError(
            f"{environment_name} must not be a drive root"
        )
    if os.name == "nt" and resolved.drive.casefold() != "g:":
        raise PrivateBridgeConfigurationError(
            f"{environment_name} must stay on the G drive"
        )
    return resolved


def _require_scoped_absolute_path(
    candidate: Path,
    *,
    environment_name: str,
) -> Path:
    candidate = candidate.expanduser()
    if not candidate.is_absolute():
        raise PrivateBridgeConfigurationError(
            f"{environment_name} must be an absolute path"
        )
    resolved = candidate.resolve(strict=False)
    if resolved == Path(resolved.anchor).resolve(strict=False):
        raise PrivateBridgeConfigurationError(
            f"{environment_name} must not be a drive root"
        )
    return resolved


def _require_project_content_path(
    value: str | None,
    *,
    default: Path,
    environment_name: str,
    project_root: Path,
) -> Path:
    """Keep canonical shared content inside the explicitly selected project."""

    resolved = _require_scoped_absolute_path(
        Path(str(value or "").strip() or default),
        environment_name=environment_name,
    )
    root = project_root.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise PrivateBridgeConfigurationError(
            f"{environment_name} must stay inside RINNE_DATA_ROOT"
        )
    return resolved


def _nested_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


class PrivateChannelBridge:
    """Own one persistent, tool-restricted context for private QQ turns."""

    def __init__(
        self,
        *,
        config: PrivateBridgeConfig,
        context_factory: ContextFactory,
        coordinator: ConversationTurnCoordinator | None = None,
        processor_factory: ProcessorFactory = RinneTextTurnProcessor,
        context_refresher: ContextRefresher | None = None,
    ) -> None:
        if not config.enabled:
            raise PrivateBridgeConfigurationError("private bridge is disabled")
        self.config = config
        self._context_factory = context_factory
        self._coordinator = coordinator or ConversationTurnCoordinator()
        self._processor_factory = processor_factory
        self._context_refresher = context_refresher
        self._identity_policy = ChannelIdentityPolicy.from_mapping(
            {AgentChannel.QQ: config.allowed_sender_ids}
        )
        self._context: Any | None = None
        self._history: PrivateHistorySessionManager | None = None

    @property
    def started(self) -> bool:
        return self._context is not None

    async def start(self) -> None:
        if self._context is not None:
            return

        async def discard_transport_event(_payload: str) -> None:
            return None

        self._context = await self._context_factory(
            client_uid=PRIVATE_BRIDGE_CLIENT_UID,
            send_text=discard_transport_event,
        )
        self._history = PrivateHistorySessionManager(self._context)
        self._history.resume_latest_open()

    async def stop(self) -> None:
        context = self._context
        self._context = None
        self._history = None
        if context is not None:
            await context.close()

    def require_sender(self, sender_id: str) -> None:
        self._identity_policy.require_allowed(AgentChannel.QQ, sender_id)

    def health_snapshot(self) -> dict[str, Any]:
        """Return a credential-free snapshot of the private reply core."""

        context = self._context
        if context is None:
            raise PrivateBridgeNotStarted("private bridge context is not ready")

        agent_engine = getattr(context, "agent_engine", None)
        agent_ready = bool(agent_engine and callable(getattr(agent_engine, "chat", None)))
        asr_engine = getattr(context, "asr_engine", None)
        asr_ready = bool(
            asr_engine and callable(getattr(asr_engine, "async_transcribe_np", None))
        )
        memory_ready = getattr(context, "live_memory_retrieval_service", None) is not None

        tool_manager = getattr(context, "tool_manager", None)
        raw_tools = getattr(tool_manager, "tools", {}) if tool_manager else {}
        loaded_servers = sorted(
            {
                str(getattr(tool, "related_server", "")).strip()
                for tool in raw_tools.values()
                if str(getattr(tool, "related_server", "")).strip()
            }
        )
        library_ready = "rinne-library" in loaded_servers

        character_config = getattr(context, "character_config", None)
        agent_config = _nested_value(character_config, "agent_config")
        agent_settings = _nested_value(agent_config, "agent_settings")
        basic_settings = _nested_value(agent_settings, "basic_memory_agent")
        configured_servers = list(
            dict.fromkeys(_nested_value(basic_settings, "mcp_enabled_servers", ()) or ())
        )

        try:
            read_only_config = ReadOnlyComputerConfig.from_environment()
            if read_only_config.enabled:
                read_only_config.create_service()
                read_only_status = (
                    "ready"
                    if READ_ONLY_MCP_SERVER_NAME in loaded_servers
                    else "unavailable"
                )
                if READ_ONLY_MCP_SERVER_NAME not in configured_servers:
                    configured_servers.append(READ_ONLY_MCP_SERVER_NAME)
            else:
                read_only_status = "not_enabled"
        except ReadOnlyComputerConfigurationError:
            read_only_status = "unavailable"

        essential_servers = {"rinne-library", READ_ONLY_MCP_SERVER_NAME}
        unavailable_optional = sorted(
            server
            for server in configured_servers
            if server not in loaded_servers and server not in essential_servers
        )
        ready_to_reply = agent_ready and memory_ready and library_ready
        return {
            "version": 1,
            "ready_to_reply": ready_to_reply,
            "agent_ready": agent_ready,
            "asr_ready": asr_ready,
            "memory_ready": memory_ready,
            "library_ready": library_ready,
            "read_only_computer": read_only_status,
            "loaded_mcp_servers": loaded_servers,
            "optional_mcp_unavailable": unavailable_optional,
            "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }

    async def transcribe_audio(
        self,
        *,
        sender_id: str,
        audio: Mapping[str, Any],
    ) -> str:
        """Transcribe one staged QQ voice without starting an LLM turn."""

        context = self._context
        if context is None:
            raise PrivateBridgeNotStarted("private bridge context is not ready")
        self.require_sender(sender_id)
        if self.config.inbox_root is None:
            raise PrivateBridgeConfigurationError(
                "private media inbox is not configured"
            )
        asr_engine = context.asr_engine
        if asr_engine is None:
            raise PrivateBridgeConfigurationError("private bridge ASR is not loaded")
        samples = await asyncio.to_thread(
            import_private_audio,
            audio,
            inbox_root=self.config.inbox_root,
        )

        async def process_serialized() -> str:
            return str(await asr_engine.async_transcribe_np(samples)).strip()

        return await self._coordinator.run(process_serialized)

    async def end_history_session(
        self,
        *,
        sender_id: str,
    ) -> bool:
        """Seal the active QQ JSON without stopping any runtime process."""

        if self._context is None or self._history is None:
            raise PrivateBridgeNotStarted("private bridge context is not ready")
        self.require_sender(sender_id)

        async def close_serialized() -> bool:
            return self._history.close(reason="qq_end_command")

        return await self._coordinator.run(close_serialized)

    async def run_text_turn(
        self,
        *,
        sender_id: str,
        message_id: str,
        text: str,
        media: tuple[Mapping[str, Any], ...] = (),
        event_observer: PrivateBridgeEventObserver,
    ) -> str:
        context = self._context
        if context is None:
            raise PrivateBridgeNotStarted("private bridge context is not ready")
        self.require_sender(sender_id)
        images: list[dict[str, Any]] = []
        attachments: list[dict[str, Any]] = []
        if media:
            if self.config.inbox_root is None or self.config.library_root is None:
                raise PrivateBridgeConfigurationError(
                    "private media roots are not configured"
                )
            images, attachments = await asyncio.to_thread(
                import_private_media,
                media,
                inbox_root=self.config.inbox_root,
                library_root=self.config.library_root,
            )
            text = _reconcile_media_labels(text, media, attachments)
        message = InboundChannelMessage(
            channel=AgentChannel.QQ,
            message_id=message_id,
            sender_id=sender_id,
            conversation_id=f"private:{sender_id}",
            text=text,
            received_at=datetime.now(timezone.utc),
            metadata={"images": images, "attachments": attachments},
        )

        async def process_serialized() -> str:
            if self._history is None:
                raise PrivateBridgeNotStarted("private history session is not ready")
            self._history.ensure_open()
            if self._context_refresher is not None:
                await self._context_refresher(context, PRIVATE_BRIDGE_CLIENT_UID)
            processor = self._processor_factory(
                context=context,
                client_uid=PRIVATE_BRIDGE_CLIENT_UID,
                event_observer=event_observer,
            )
            return await processor(message)

        return await self._coordinator.run(process_serialized)
