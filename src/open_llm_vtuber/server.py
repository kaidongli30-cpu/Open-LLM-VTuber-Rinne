"""
Open-LLM-VTuber Server
========================
This module contains the WebSocket server for Open-LLM-VTuber, which handles
the WebSocket connections, serves static files, and manages the web tool.
It uses FastAPI for the server and Starlette for static file serving.
"""

import asyncio
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from loguru import logger
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response
from starlette.staticfiles import StaticFiles as StarletteStaticFiles

from .routes import (
    init_app_info_routes,
    init_client_ws_route,
    init_webtool_routes,
    init_proxy_route,
    init_library_routes,
)
from .service_context import ServiceContext
from .config_manager.utils import Config
from .websocket_handler import WebSocketHandler
from .memory.daily_child_events import ChildEventWorkerLaunch
from .memory.layer2_runtime import Layer2WorkerLaunch
from .data_paths import character_history_root
from .rinne_renderer_profile import RinneRuntimeOutfitController
from .memory.live_retrieval import (
    LiveMemoryRetrievalService,
    live_retrieval_settings_from_config,
)
from .agent_runtime.private_bridge import (
    PrivateBridgeConfig,
    PrivateBridgeConfigurationError,
    PrivateChannelBridge,
)
from .private_bridge_routes import init_private_bridge_routes
from .agent_runtime import (
    AgentChannel,
    ChannelIdentityPolicy,
    ConversationTurnCoordinator,
    InMemoryReminderScheduler,
    ProactiveCategory,
    ProactiveDeliveryReceipt,
    ProactiveMessageDispatcher,
    ProactiveMessageGate,
    ProactiveRuntimeConfig,
    QQChannelAdapter,
    QQChannelConfig,
    ReadOnlyRemoteApprovalBroker,
    RinneTextTurnProcessor,
    ScheduledReminder,
    task_completion_request,
)


# Create a custom StaticFiles class that adds CORS headers
class CORSStaticFiles(StarletteStaticFiles):
    """
    Static files handler that adds CORS headers to all responses.
    Needed because Starlette StaticFiles might bypass standard middleware.
    """

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        if path.endswith(".js"):
            response.headers["Content-Type"] = "application/javascript"
        # 对 index.html 禁用缓存
        if path in ("", "index.html"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        return response


class AvatarStaticFiles(CORSStaticFiles):
    """
    Avatar files handler with security restrictions and CORS headers
    """

    async def get_response(self, path: str, scope):
        allowed_extensions = (".jpg", ".jpeg", ".png", ".gif", ".svg")
        if not any(path.lower().endswith(ext) for ext in allowed_extensions):
            return Response("Forbidden file type", status_code=403)
        response = await super().get_response(path, scope)
        return response


class WebSocketServer:
    """
    API server for Open-LLM-VTuber. This contains the websocket endpoint for the client, hosts the web tool, and serves static files.

    Creates and configures a FastAPI app, registers all routes
    (WebSocket, web tools, proxy) and mounts static assets with CORS.

    Args:
        config (Config): Application configuration containing system settings.
        default_context_cache (ServiceContext, optional):
            Pre‑initialized service context for sessions' service context to reference to.
            **If omitted, `initialize()` method needs to be called to load service context.**

    Notes:
        - If default_context_cache is omitted, call `await initialize()` to load service context cache.
        - Use `clean_cache()` to clear and recreate the local cache directory.
    """

    def __init__(
        self,
        config: Config,
        default_context_cache: ServiceContext = None,
        rinne_outfit_controller: RinneRuntimeOutfitController | None = None,
    ):
        self.app = FastAPI(title="Open-LLM-VTuber Server")  # Added title for clarity
        self.config = config
        self.default_context_cache = (
            default_context_cache or ServiceContext()
        )  # Use provided context or initialize a new empty one waiting to be loaded
        self.ws_handler = WebSocketHandler(
            self.default_context_cache,
            rinne_outfit_controller=rinne_outfit_controller,
        )
        self._child_event_worker_launches: list[ChildEventWorkerLaunch] = []
        self._layer2_worker_launches: list[Layer2WorkerLaunch] = []
        self._background_tasks: set[asyncio.Task] = set()
        self._channel_turn_coordinator = ConversationTurnCoordinator()
        self._private_bridge: PrivateChannelBridge | None = None
        self._qq_adapter: QQChannelAdapter | None = None
        self._qq_context: ServiceContext | None = None
        self._qq_read_only_approval_broker: ReadOnlyRemoteApprovalBroker | None = None
        self._qq_proactive_config: ProactiveRuntimeConfig | None = None
        self._qq_proactive_gate: ProactiveMessageGate | None = None
        self._qq_proactive_dispatcher: ProactiveMessageDispatcher | None = None
        self._qq_reminder_scheduler: InMemoryReminderScheduler | None = None
        self._qq_reminder_task: asyncio.Task | None = None
        self._pending_qq_task_notifications: list[tuple[str, dict]] = []
        try:
            private_bridge_config = PrivateBridgeConfig.from_environment()
        except PrivateBridgeConfigurationError as exc:
            logger.error(
                "[私人QQ桥接] 配置无效，桥接保持关闭：{}", type(exc).__name__
            )
        else:
            if private_bridge_config.enabled:
                self._private_bridge = PrivateChannelBridge(
                    config=private_bridge_config,
                    context_factory=self.ws_handler.create_text_channel_context,
                    coordinator=self._channel_turn_coordinator,
                    context_refresher=self.ws_handler.refresh_text_channel_memory,
                )
                self.app.include_router(
                    init_private_bridge_routes(self._private_bridge)
                )
        self.app.add_event_handler("startup", self._start_private_bridge)
        self.app.add_event_handler("startup", self._start_qq_channel)
        self.app.add_event_handler(
            "startup", self._start_child_event_worker_monitors
        )
        self.app.add_event_handler("shutdown", self._stop_qq_channel)
        self.app.add_event_handler("shutdown", self._stop_private_bridge)
        # It will be populated during the initialize method call

        # Add global CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Include routes, passing the context instance
        # The context will be populated during the initialize step
        self.app.include_router(
            init_client_ws_route(
                default_context_cache=self.default_context_cache,
                ws_handler=self.ws_handler,
            ),
        )
        self.app.include_router(
            init_webtool_routes(default_context_cache=self.default_context_cache),
        )
        self.app.include_router(init_library_routes())
        self.app.include_router(init_app_info_routes())

        # Initialize and include proxy routes if proxy is enabled
        system_config = config.system_config
        if hasattr(system_config, "enable_proxy") and system_config.enable_proxy:
            # Construct the server URL for the proxy
            host = system_config.host
            port = system_config.port
            server_url = f"ws://{host}:{port}/client-ws"
            self.app.include_router(
                init_proxy_route(server_url=server_url),
            )

        # Mount cache directory first (to ensure audio file access)
        if not os.path.exists("cache"):
            os.makedirs("cache")
        self.app.mount(
            "/cache",
            CORSStaticFiles(directory="cache"),
            name="cache",
        )

        # Mount static files with CORS-enabled handlers
        self.app.mount(
            "/live2d-models",
            CORSStaticFiles(directory="live2d-models"),
            name="live2d-models",
        )
        self.app.mount(
            "/bg",
            CORSStaticFiles(directory="backgrounds"),
            name="backgrounds",
        )
        self.app.mount(
            "/avatars",
            AvatarStaticFiles(directory="avatars"),
            name="avatars",
        )

        # Mount web tool directory separately from frontend
        self.app.mount(
            "/web-tool",
            CORSStaticFiles(directory="web_tool", html=True),
            name="web_tool",
        )

        # Mount main frontend last (as catch-all)
        self.app.mount(
            "/",
            CORSStaticFiles(directory="frontend", html=True),
            name="frontend",
        )

    def watch_child_event_worker(self, launch: ChildEventWorkerLaunch) -> None:
        """Arrange for one launched daily worker to notify the desktop UI."""

        self._child_event_worker_launches.append(launch)

    def watch_layer2_worker(self, launch: Layer2WorkerLaunch) -> None:
        """Arrange for one Layer-2 worker to notify the desktop UI."""

        self._layer2_worker_launches.append(launch)

    def queue_system_notification(self, notification: dict) -> None:
        self.ws_handler.queue_system_notification(notification)
        if notification.get("type") == "memory-notification":
            task_id = self._notification_task_id("startup", notification)
            self._pending_qq_task_notifications.append(
                (task_id, dict(notification))
            )
            del self._pending_qq_task_notifications[:-32]

    async def _start_child_event_worker_monitors(self) -> None:
        for launch in self._child_event_worker_launches:
            task = asyncio.create_task(self._monitor_child_event_worker(launch))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        for launch in self._layer2_worker_launches:
            task = asyncio.create_task(self._monitor_layer2_worker(launch))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _start_qq_channel(self) -> None:
        """Start the optional official QQ channel without risking desktop boot."""

        if self._private_bridge is not None:
            logger.info("[QQ] 私人 AstrBot 桥接已配置，官方 QQ 通道保持关闭")
            return

        qq_context: ServiceContext | None = None
        qq_adapter: QQChannelAdapter | None = None
        approval_broker: ReadOnlyRemoteApprovalBroker | None = None
        try:
            qq_config = QQChannelConfig.from_environment()
            if not qq_config.enabled:
                logger.debug("[QQ] 通道保持关闭")
                return

            # Validate credentials and the exact sender allowlist before cloning
            # a conversation context. Credentials remain outside project files.
            qq_config.resolve_credentials()
            approval_broker = ReadOnlyRemoteApprovalBroker(
                identity_policy=ChannelIdentityPolicy.from_mapping(
                    {AgentChannel.QQ: qq_config.allowed_user_openids}
                )
            )

            async def discard_transport_event(_payload: str) -> None:
                return None

            qq_context = await self.ws_handler.create_text_channel_context(
                client_uid="rinne-qq",
                send_text=discard_transport_event,
            )
            turn_processor = RinneTextTurnProcessor(
                context=qq_context,
                client_uid="rinne-qq",
            )
            qq_adapter = QQChannelAdapter(
                config=qq_config,
                turn_processor=turn_processor,
                coordinator=self._channel_turn_coordinator,
                approval_broker=approval_broker,
            )
            await qq_adapter.start()
            try:
                await self._start_qq_proactive_runtime(
                    qq_config=qq_config,
                    qq_adapter=qq_adapter,
                )
            except Exception as proactive_exc:
                await self._stop_qq_proactive_runtime()
                logger.error(
                    "[QQ主动消息] 安全配置无效，主动消息保持关闭；"
                    "QQ被动私聊继续运行："
                    f"{type(proactive_exc).__name__}"
                )
        except Exception as exc:
            await self._stop_qq_proactive_runtime()
            if qq_adapter is not None:
                try:
                    await qq_adapter.stop()
                except Exception as cleanup_exc:
                    logger.warning(
                        "[QQ] 适配器启动失败后的清理也失败："
                        f"{type(cleanup_exc).__name__}"
                    )
            if qq_context is not None:
                try:
                    await qq_context.close()
                except Exception as cleanup_exc:
                    logger.warning(
                        "[QQ] 会话启动失败后的清理也失败："
                        f"{type(cleanup_exc).__name__}"
                    )
            logger.error(
                "[QQ] 通道启动失败，桌面端继续运行："
                f"{type(exc).__name__}"
            )
            return

        self._qq_context = qq_context
        self._qq_adapter = qq_adapter
        self._qq_read_only_approval_broker = approval_broker

    async def _start_private_bridge(self) -> None:
        bridge = self._private_bridge
        if bridge is None:
            logger.debug("[私人QQ桥接] 通道保持关闭")
            return
        try:
            await bridge.start()
        except Exception as exc:
            logger.error(
                "[私人QQ桥接] 隔离会话启动失败，桌面端继续运行：{}",
                type(exc).__name__,
            )
            return
        logger.info("[私人QQ桥接] 本机认证入口已就绪，电脑工具保持关闭")

    async def _stop_private_bridge(self) -> None:
        bridge = self._private_bridge
        if bridge is None:
            return
        try:
            await bridge.stop()
        except Exception as exc:
            logger.warning(
                "[私人QQ桥接] 关闭隔离会话失败：{}", type(exc).__name__
            )

    async def _stop_qq_channel(self) -> None:
        """Stop QQ transport and its isolated conversation context."""

        qq_adapter = self._qq_adapter
        qq_context = self._qq_context
        self._qq_adapter = None
        self._qq_context = None
        self._qq_read_only_approval_broker = None
        try:
            await self._stop_qq_proactive_runtime()
            if qq_adapter is not None:
                await qq_adapter.stop()
        finally:
            if qq_context is not None:
                await qq_context.close()

    async def _start_qq_proactive_runtime(
        self,
        *,
        qq_config: QQChannelConfig,
        qq_adapter: QQChannelAdapter,
    ) -> None:
        proactive_config = ProactiveRuntimeConfig.from_environment()
        if not proactive_config.enabled:
            self._pending_qq_task_notifications.clear()
            logger.debug("[QQ主动消息] 总开关保持关闭")
            return

        recipient_id = proactive_config.require_qq_recipient(
            qq_config.allowed_user_openids
        )
        gate = ProactiveMessageGate(proactive_config.to_policy())

        async def deliver(request):
            if request.channel is not AgentChannel.QQ:
                return ProactiveDeliveryReceipt(accepted=False)
            if request.recipient_id != recipient_id:
                return ProactiveDeliveryReceipt(accepted=False)
            await qq_adapter.send_proactive_text(
                recipient_id=recipient_id,
                text=request.text,
            )
            return ProactiveDeliveryReceipt(
                accepted=True,
                delivery_confirmed=None,
            )

        dispatcher = ProactiveMessageDispatcher(
            gate=gate,
            deliverer=deliver,
        )
        scheduler = InMemoryReminderScheduler()
        self._qq_proactive_config = proactive_config
        self._qq_proactive_gate = gate
        self._qq_proactive_dispatcher = dispatcher
        self._qq_reminder_scheduler = scheduler
        reminder_task = asyncio.create_task(self._qq_reminder_loop())
        self._qq_reminder_task = reminder_task
        self._background_tasks.add(reminder_task)
        reminder_task.add_done_callback(self._background_tasks.discard)
        await self._flush_pending_qq_task_notifications(recipient_id)
        logger.info(
            "[QQ主动消息] 已启用确定性提醒和任务完成通知；"
            "陪伴问候="
            f"{'开启' if proactive_config.companion_enabled else '关闭'}"
        )

    async def _stop_qq_proactive_runtime(self) -> None:
        reminder_task = getattr(self, "_qq_reminder_task", None)
        self._qq_reminder_task = None
        self._qq_proactive_config = None
        self._qq_proactive_gate = None
        self._qq_proactive_dispatcher = None
        self._qq_reminder_scheduler = None
        if reminder_task is not None:
            reminder_task.cancel()
            try:
                await reminder_task
            except asyncio.CancelledError:
                pass

    async def _qq_reminder_loop(self) -> None:
        while True:
            config = self._qq_proactive_config
            scheduler = self._qq_reminder_scheduler
            dispatcher = self._qq_proactive_dispatcher
            if config is None or scheduler is None or dispatcher is None:
                return
            await scheduler.dispatch_due(
                dispatcher,
                now=datetime.now(timezone.utc),
            )
            await asyncio.sleep(config.reminder_poll_seconds)

    def schedule_qq_reminder(
        self,
        *,
        reminder_id: str,
        due_at: datetime,
        text: str,
    ) -> None:
        """Schedule an explicit in-memory reminder from trusted local code."""

        config = self._qq_proactive_config
        scheduler = self._qq_reminder_scheduler
        if config is None or scheduler is None:
            raise RuntimeError("QQ proactive reminder runtime is not enabled")
        scheduler.schedule(
            ScheduledReminder(
                reminder_id=reminder_id,
                due_at=due_at,
                channel=AgentChannel.QQ,
                recipient_id=config.qq_recipient_openid,
                text=text,
            )
        )

    def pause_qq_proactive_messages(self) -> None:
        gate = self._qq_proactive_gate
        if gate is not None:
            gate.pause()

    def resume_qq_proactive_messages(self) -> None:
        gate = self._qq_proactive_gate
        if gate is not None:
            gate.resume()

    def set_qq_companion_messages_enabled(self, enabled: bool) -> None:
        gate = self._qq_proactive_gate
        if gate is not None:
            gate.set_category_enabled(ProactiveCategory.COMPANION, enabled)

    async def _flush_pending_qq_task_notifications(
        self,
        recipient_id: str,
    ) -> None:
        pending = list(self._pending_qq_task_notifications)
        self._pending_qq_task_notifications.clear()
        for task_id, notification in pending:
            await self._dispatch_qq_task_notification(
                task_id=task_id,
                notification=notification,
                recipient_id=recipient_id,
            )

    async def _dispatch_qq_task_notification(
        self,
        *,
        task_id: str,
        notification: dict,
        recipient_id: str | None = None,
    ) -> None:
        dispatcher = getattr(self, "_qq_proactive_dispatcher", None)
        config = getattr(self, "_qq_proactive_config", None)
        if dispatcher is None or config is None:
            return
        recipient = recipient_id or config.qq_recipient_openid
        message = str(notification.get("message", "")).strip()
        description = str(notification.get("description", "")).strip()
        text = "：".join(part for part in (message, description) if part)
        if not text:
            return
        result = await dispatcher.dispatch(
            task_completion_request(
                task_id=task_id,
                channel=AgentChannel.QQ,
                recipient_id=recipient,
                text=text,
            ),
            now=datetime.now(timezone.utc),
        )
        logger.info(
            "[QQ主动消息] 任务通知处理完成："
            f"status={result.status.value}, reason="
            f"{result.reason.value if result.reason else 'none'}"
        )

    @staticmethod
    def _notification_task_id(prefix: str, notification: dict) -> str:
        canonical = json.dumps(
            notification,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}:{digest}"

    async def _publish_task_notification(
        self,
        *,
        task_id: str,
        notification: dict,
    ) -> None:
        await self.ws_handler.publish_system_notification(notification)
        await self._dispatch_qq_task_notification(
            task_id=task_id,
            notification=notification,
        )

    async def _monitor_layer2_worker(self, launch: Layer2WorkerLaunch) -> None:
        exit_code = await asyncio.to_thread(launch.process.wait)
        result = self._read_child_event_worker_result(launch.result_path)
        status = result.get("status") if result else None
        if exit_code == 0 and status == "published":
            notification = {
                "type": "memory-notification",
                "level": "success",
                "message": "个人背景已更新",
            }
            await self._publish_task_notification(
                task_id=f"layer2:{launch.memory_day}:published",
                notification=notification,
            )
            return
        if exit_code == 0 and status in {
            "already_current",
            "already_running",
            "disabled",
        }:
            return
        notification = {
            "type": "memory-notification",
            "level": "error",
            "message": "个人背景更新失败",
            "description": "继续使用上一份有效背景，请查看后端日志",
        }
        await self._publish_task_notification(
            task_id=(
                f"layer2:{launch.memory_day}:"
                f"{status or 'worker_failed'}"
            ),
            notification=notification,
        )

    async def _monitor_child_event_worker(
        self, launch: ChildEventWorkerLaunch
    ) -> None:
        exit_code = await asyncio.to_thread(launch.process.wait)
        result = self._read_child_event_worker_result(launch.result_path)
        status = result.get("status") if result else None
        if exit_code == 0 and status == "published":
            notification = {
                "type": "memory-notification",
                "level": "success",
                "message": "昨日事件已整理完成",
            }
            await self._publish_task_notification(
                task_id=f"child-events:{launch.memory_day}:published",
                notification=notification,
            )
            return
        if exit_code == 0 and status in {
            "already_published",
            "already_running",
            "disabled",
        }:
            return
        description = self._brief_child_event_error(result)
        notification = {
            "type": "memory-notification",
            "level": "error",
            "message": "昨日事件整理失败",
            "description": description,
        }
        await self._publish_task_notification(
            task_id=(
                f"child-events:{launch.memory_day}:"
                f"{status or 'worker_failed'}"
            ),
            notification=notification,
        )

    @staticmethod
    def _read_child_event_worker_result(path: Path) -> dict | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _brief_child_event_error(result: dict | None) -> str:
        if not result:
            return "后台任务未返回有效结果，请查看后端日志"
        error_type = str(result.get("error_type", ""))
        error = str(result.get("error", ""))
        if error_type == "ValidationError":
            return "24B输出未通过格式校验"
        if error_type == "PublicationConflict":
            return "事件目录状态冲突，未覆盖现有文件"
        if "diary_missing_or_empty" in error:
            return "待整理日记不存在或内容为空"
        if "provider_connection_error" in error or "urlopen" in error.lower():
            return "无法连接事件生成服务"
        if "provider_http_error" in error:
            return "事件生成接口返回错误"
        return "请查看后端日志中的详细原因"

    async def initialize(self):
        """Asynchronously load the service context from config.
        Calling this function is needed if default_context_cache was not provided to the constructor."""
        await self.default_context_cache.load_from_config(self.config)
        retrieval_config = (
            self.default_context_cache.character_config.agent_config.agent_settings
            .basic_memory_agent.long_term_memory_retrieval
        )
        llm = getattr(self.default_context_cache.agent_engine, "_llm", None)
        llm_base_url = str(getattr(llm, "base_url", ""))
        is_local_ollama = (
            "localhost:11434" in llm_base_url
            or "127.0.0.1:11434" in llm_base_url
        )
        if retrieval_config.enabled and not is_local_ollama:
            try:
                history_root = character_history_root(
                    self.default_context_cache.character_config.conf_uid
                )
                logger.info("[长期记忆] 正在预热本地检索模型……")
                service = await asyncio.to_thread(
                    LiveMemoryRetrievalService,
                    history_root,
                    settings=live_retrieval_settings_from_config(retrieval_config),
                )
                warmup = await asyncio.to_thread(service.warm_runtime)
                self.default_context_cache.live_memory_retrieval_service = service
                logger.info(
                    "[长期记忆] 本地检索模型预热完成："
                    f"{warmup['total_seconds']}秒；等待云端按需调用"
                )
            except Exception as exc:
                logger.warning(
                    "[长期记忆] 预热失败，将在云端实际请求记忆时重试："
                    f"{type(exc).__name__}: {exc}"
                )

    @staticmethod
    def clean_cache():
        """Clean the cache directory by removing and recreating it."""
        cache_dir = "cache"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)
