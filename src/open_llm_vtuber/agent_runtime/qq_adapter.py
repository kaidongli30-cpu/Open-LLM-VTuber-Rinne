"""Official QQ Bot C2C text adapter.

The transport SDK is imported only when an explicitly enabled adapter starts.
Tests inject small fakes, so the safety boundary is testable without credentials
or a network connection.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol

from loguru import logger
from ..privacy_logging import text_log_fields

from .access_policy import ChannelAccessDenied, ChannelIdentityPolicy
from .channel_runtime import (
    ChannelMessageRuntime,
    ChannelTurnResult,
    ConversationTurnCoordinator,
    TurnProcessor,
)
from .message import AgentChannel, InboundChannelMessage
from .qq_config import QQChannelConfig
from .remote_approval import (
    ReadOnlyRemoteApprovalBroker,
    RemoteApprovalDecision,
    RemoteApprovalRequest,
    RemoteApprovalResolution,
    parse_remote_approval_button_data,
)


class QQSDKUnavailable(RuntimeError):
    """Raised when the optional official QQ SDK is not installed."""


class QQApiClientProtocol(Protocol):
    async def ensure_token(self) -> str: ...

    async def get_gateway_url(self) -> str: ...

    def ensure_token_sync(self) -> str: ...

    def get_gateway_url_sync(self) -> str: ...

    def clear_token(self) -> None: ...

    async def send_text(
        self,
        chat_type: str,
        chat_id: str,
        content: str,
        *,
        reply_to: str | None = None,
    ) -> Any: ...

    def build_text_body(
        self,
        content: str,
        reply_to: str | None = None,
        markdown: bool = True,
        max_length: int = 4000,
    ) -> Any: ...

    async def post_c2c_message(
        self,
        user_id: str,
        message: Any,
        *,
        keyboard: Any | None = None,
    ) -> Any: ...

    async def acknowledge_interaction(
        self,
        interaction_id: str,
        code: int = 0,
        data: Mapping[str, Any] | None = None,
    ) -> None: ...


class QQEventParserProtocol(Protocol):
    def parse(self, event_type: str, raw: Mapping[str, Any]) -> Any | None: ...


class QQWebSocketProtocol(Protocol):
    def start(
        self,
        gateway_url: str,
        main_loop: asyncio.AbstractEventLoop,
    ) -> None: ...

    async def async_stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class QQSDKBindings:
    create_api_client: Callable[..., QQApiClientProtocol]
    create_event_parser: Callable[[], QQEventParserProtocol]
    create_callbacks: Callable[..., Any]
    create_websocket: Callable[..., QQWebSocketProtocol]
    parse_interaction_event: Callable[[Mapping[str, Any]], Any] | None = None
    create_read_only_approval_keyboard: (
        Callable[[RemoteApprovalRequest], Any] | None
    ) = None


def load_official_qq_sdk() -> QQSDKBindings:
    """Load Tencent's official SDK only for an enabled QQ runtime."""

    try:
        from qqbot_agent_sdk import (  # type: ignore[import-not-found]
            EventParser,
            QQApiClient,
            QQWebSocket,
            WSCallbacks,
            parse_interaction_event,
        )
        from qqbot_agent_sdk.dto import (  # type: ignore[import-not-found]
            InlineKeyboard,
            KeyboardButton,
            KeyboardButtonAction,
            KeyboardButtonPermission,
            KeyboardButtonRenderData,
            KeyboardContent,
            KeyboardRow,
        )
    except ImportError as exc:
        raise QQSDKUnavailable(
            "QQ support requires the optional qqbot-agent-sdk package"
        ) from exc

    def create_read_only_approval_keyboard(request: RemoteApprovalRequest):
        def button(
            *,
            button_id: str,
            label: str,
            visited_label: str,
            decision,
            style: int,
        ):
            return KeyboardButton(
                id=button_id,
                render_data=KeyboardButtonRenderData(
                    label=label,
                    visited_label=visited_label,
                    style=style,
                ),
                action=KeyboardButtonAction(
                    type=1,
                    data=request.button_data(decision),
                    permission=KeyboardButtonPermission(type=2),
                    click_limit=1,
                ),
                group_id="rinne-read-only-approval",
            )

        return InlineKeyboard(
            content=KeyboardContent(
                rows=[
                    KeyboardRow(
                        buttons=[
                            button(
                                button_id="rinne-allow-once",
                                label="✅ 允许一次",
                                visited_label="已允许",
                                decision=RemoteApprovalDecision.ALLOW_ONCE,
                                style=1,
                            ),
                            button(
                                button_id="rinne-deny",
                                label="❌ 拒绝",
                                visited_label="已拒绝",
                                decision=RemoteApprovalDecision.DENY,
                                style=0,
                            ),
                        ]
                    )
                ]
            )
        )

    return QQSDKBindings(
        create_api_client=QQApiClient,
        create_event_parser=EventParser,
        create_callbacks=WSCallbacks,
        create_websocket=QQWebSocket,
        parse_interaction_event=parse_interaction_event,
        create_read_only_approval_keyboard=create_read_only_approval_keyboard,
    )


class QQChannelAdapter:
    """Receive and reply to allowlisted official QQ Bot C2C text messages."""

    def __init__(
        self,
        *,
        config: QQChannelConfig,
        turn_processor: TurnProcessor,
        coordinator: ConversationTurnCoordinator | None = None,
        environ: Mapping[str, str] | None = None,
        sdk_loader: Callable[[], QQSDKBindings] = load_official_qq_sdk,
        approval_broker: ReadOnlyRemoteApprovalBroker | None = None,
    ) -> None:
        self._config = config
        self._turn_processor = turn_processor
        self._coordinator = coordinator
        self._environ = environ
        self._sdk_loader = sdk_loader
        self._approval_broker = approval_broker
        self._api: QQApiClientProtocol | None = None
        self._parser: QQEventParserProtocol | None = None
        self._websocket: QQWebSocketProtocol | None = None
        self._runtime: ChannelMessageRuntime | None = None
        self._parse_interaction_event: Callable[[Mapping[str, Any]], Any] | None = None
        self._create_approval_keyboard: (
            Callable[[RemoteApprovalRequest], Any] | None
        ) = None

    @property
    def is_running(self) -> bool:
        return self._websocket is not None

    async def start(self) -> bool:
        """Start the official WebSocket transport, or no-op when disabled."""

        if not self._config.enabled:
            return False
        if self.is_running:
            return True

        credentials = self._config.resolve_credentials(self._environ)
        sdk = self._sdk_loader()
        api = sdk.create_api_client(
            app_id=credentials.app_id,
            client_secret=credentials.client_secret,
        )
        parser = sdk.create_event_parser()
        identity_policy = ChannelIdentityPolicy.from_mapping(
            {AgentChannel.QQ: self._config.allowed_user_openids}
        )

        self._api = api
        self._parser = parser
        self._parse_interaction_event = sdk.parse_interaction_event
        self._create_approval_keyboard = sdk.create_read_only_approval_keyboard
        self._runtime = ChannelMessageRuntime(
            identity_policy=identity_policy,
            turn_processor=self._turn_processor,
            text_deliverer=self._deliver_text,
            coordinator=self._coordinator,
        )
        try:
            callbacks = sdk.create_callbacks(
                on_message_event=self.handle_raw_event,
                on_connected=self._on_connected,
                on_disconnected=self._on_disconnected,
                on_fatal_error=self._on_fatal_error,
                get_token=api.ensure_token_sync,
                get_session=lambda: (None, None),
                set_session=lambda _session_id, _seq: None,
                set_heartbeat_interval=lambda _interval: None,
                clear_token=api.clear_token,
                fail_pending=self._on_pending_failed,
                get_gateway_url=api.get_gateway_url_sync,
                on_interaction_event=(
                    self.handle_raw_interaction
                    if self._approval_broker is not None
                    else None
                ),
            )
            websocket = sdk.create_websocket(
                callbacks=callbacks,
                log_tag="Rinne-QQ",
            )
            await api.ensure_token()
            gateway_url = await api.get_gateway_url()
            websocket.start(gateway_url, asyncio.get_running_loop())
        except Exception:
            self._clear_runtime()
            raise

        self._websocket = websocket
        logger.info("[QQ] 官方 C2C 文字通道已启动")
        return True

    async def stop(self) -> None:
        websocket = self._websocket
        try:
            if websocket is not None:
                await websocket.async_stop()
        finally:
            self._clear_runtime()

    async def send_proactive_text(
        self,
        *,
        recipient_id: str,
        text: str,
    ) -> Any:
        """Submit one proactive C2C text only to an allowlisted QQ OpenID."""

        recipient = recipient_id.strip()
        if recipient not in self._config.allowed_user_openids:
            raise ChannelAccessDenied("proactive QQ recipient is not paired")
        content = text.strip()
        if not content:
            raise ValueError("proactive text must not be blank")
        api = self._api
        if api is None or not self.is_running:
            raise RuntimeError("QQ adapter is not running")
        return await api.send_text("c2c", recipient, content)

    async def send_read_only_approval(
        self,
        request: RemoteApprovalRequest,
    ) -> Any:
        """Send a two-button, one-time approval without exposing an executor."""

        broker = self._approval_broker
        api = self._api
        create_keyboard = self._create_approval_keyboard
        if broker is None:
            raise RuntimeError("QQ read-only approval broker is not configured")
        if api is None or not self.is_running:
            raise RuntimeError("QQ adapter is not running")
        if request.channel is not AgentChannel.QQ:
            raise ValueError("QQ approval request must use the QQ channel")
        if request.requester_id not in self._config.allowed_user_openids:
            raise ChannelAccessDenied("QQ approval recipient is not paired")
        if not broker.is_pending(request):
            raise ValueError("QQ approval request is not pending")
        if create_keyboard is None:
            raise QQSDKUnavailable(
                "installed QQ SDK does not support safe approval keyboards"
            )

        text = (
            "🔐 **凛祢只读操作确认**\n\n"
            f"操作：{request.action.value}\n"
            f"目标：{request.summary}\n\n"
            "本次许可只能使用一次，参数变化或超时后自动失效。"
        )
        message = api.build_text_body(text, markdown=True)
        keyboard = create_keyboard(request)
        try:
            return await api.post_c2c_message(
                request.requester_id,
                message,
                keyboard=keyboard,
            )
        except Exception:
            broker.invalidate_pending(request)
            raise

    async def handle_raw_interaction(
        self,
        event_type: str,
        raw: Mapping[str, Any],
    ) -> RemoteApprovalResolution | None:
        """Resolve only Rinne's C2C one-time approval button payloads."""

        if event_type != "INTERACTION_CREATE":
            return None
        broker = self._approval_broker
        api = self._api
        parse_interaction = self._parse_interaction_event
        if broker is None or api is None or parse_interaction is None:
            return None
        event = parse_interaction(raw)
        event_data = getattr(event, "data", None)
        resolved = getattr(event_data, "resolved", None)
        button_data = str(getattr(resolved, "button_data", "")).strip()
        if parse_remote_approval_button_data(button_data) is None:
            return None
        interaction_id = str(getattr(event, "id", "")).strip()
        operator_id = str(getattr(event, "operator_openid", "")).strip()
        if (
            not interaction_id
            or not operator_id
            or not bool(getattr(event, "is_c2c", False))
        ):
            return None
        try:
            resolution = broker.resolve_button(
                button_data=button_data,
                channel=AgentChannel.QQ,
                requester_id=operator_id,
            )
        except ChannelAccessDenied:
            resolution = RemoteApprovalResolution.NOT_FOUND
        await api.acknowledge_interaction(interaction_id, code=0)
        logger.info(f"[QQ审批] 按钮处理完成：resolution={resolution.value}")
        return resolution

    async def handle_raw_event(
        self,
        event_type: str,
        raw: Mapping[str, Any],
    ) -> ChannelTurnResult | None:
        """Parse one SDK event and ignore everything outside safe C2C text."""

        parser = self._parser
        runtime = self._runtime
        if parser is None or runtime is None:
            return None

        event = parser.parse(event_type, raw)
        if event is None or getattr(event, "chat_scope", "") != "c2c":
            return None
        content = str(getattr(event, "content", "")).strip()
        if not content:
            return None

        message_id = str(getattr(event, "message_id", "")).strip()
        sender_id = str(getattr(event, "user_id", "")).strip()
        chat_id = str(getattr(event, "chat_id", "")).strip()
        if not message_id or not sender_id or not chat_id:
            logger.warning("[QQ] 已忽略字段不完整的 C2C 消息")
            return None

        message = InboundChannelMessage(
            channel=AgentChannel.QQ,
            message_id=message_id,
            sender_id=sender_id,
            conversation_id=chat_id,
            text=content,
            received_at=_parse_qq_timestamp(getattr(event, "timestamp", "")),
            metadata={"source_channel": AgentChannel.QQ.value},
        )
        try:
            return await runtime.handle_message(message)
        except ChannelAccessDenied:
            logger.warning("[QQ] 已拒绝一个未配对账号的 C2C 消息")
            return None

    async def _deliver_text(
        self,
        message: InboundChannelMessage,
        response_text: str,
    ) -> None:
        api = self._api
        if api is None:
            raise RuntimeError("QQ API client is not running")
        await api.send_text(
            "c2c",
            message.conversation_id,
            response_text,
            reply_to=message.message_id,
        )

    def _clear_runtime(self) -> None:
        self._websocket = None
        self._runtime = None
        self._parser = None
        self._parse_interaction_event = None
        self._create_approval_keyboard = None
        self._api = None

    @staticmethod
    def _on_connected() -> None:
        logger.info("[QQ] WebSocket 已连接")

    @staticmethod
    def _on_disconnected() -> None:
        logger.warning("[QQ] WebSocket 已断开，等待官方 SDK 重连")

    @staticmethod
    def _on_fatal_error(error_code: str, message: str) -> None:
        logger.error(
            "[QQ] WebSocket 致命错误 {}: {}",
            error_code,
            text_log_fields(message),
        )

    @staticmethod
    def _on_pending_failed(reason: str) -> None:
        logger.warning(
            "[QQ] 连接中断，待发送请求失败: {}", text_log_fields(reason)
        )


def _parse_qq_timestamp(raw_timestamp: Any) -> datetime:
    value = str(raw_timestamp).strip()
    if value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                return parsed
        except ValueError:
            pass
    logger.warning("[QQ] 消息时间无效，改用本机接收时间")
    return datetime.now(timezone.utc)


__all__ = [
    "QQChannelAdapter",
    "QQSDKBindings",
    "QQSDKUnavailable",
    "load_official_qq_sdk",
]
