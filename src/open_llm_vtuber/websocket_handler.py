from typing import Dict, List, Optional, Callable, TypedDict
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
from enum import Enum
import numpy as np
from loguru import logger

from .service_context import ServiceContext
from .chat_group import (
    ChatGroupManager,
    handle_group_operation,
    handle_client_disconnect,
    broadcast_to_group,
)
from .message_handler import message_handler
from .utils.stream_audio import prepare_audio_payload
from .chat_history_manager import (
    create_new_history,
    get_history,
    delete_history,
    get_history_list,
)
from .config_manager.utils import scan_config_alts_directory, scan_bg_directory
from .conversations.conversation_handler import (
    handle_conversation_trigger,
    handle_group_interrupt,
    handle_individual_interrupt,
)
from .memory.session_context import load_cloud_session_memory
from .data_paths import character_history_root
from .rinne_renderer_profile import RinneRuntimeOutfitController


class MessageType(Enum):
    """Enum for WebSocket message types"""

    GROUP = ["add-client-to-group", "remove-client-from-group"]
    HISTORY = [
        "fetch-history-list",
        "fetch-and-set-history",
        "create-new-history",
        "delete-history",
    ]
    CONVERSATION = ["mic-audio-end", "manual-audio-end", "text-input", "ai-speak-signal"]
    CONFIG = ["fetch-configs", "switch-config"]
    CONTROL = ["interrupt-signal", "audio-play-start"]
    DATA = ["mic-audio-data", "manual-audio-data"]


class WSMessage(TypedDict, total=False):
    """Type definition for WebSocket messages"""

    type: str
    action: Optional[str]
    text: Optional[str]
    audio: Optional[List[float]]
    segment_id: Optional[str]
    images: Optional[List[str]]
    history_uid: Optional[str]
    file: Optional[str]
    profile_id: Optional[str]
    display_text: Optional[dict]


class WebSocketHandler:
    """Handles WebSocket connections and message routing"""

    def __init__(
        self,
        default_context_cache: ServiceContext,
        rinne_outfit_controller: RinneRuntimeOutfitController | None = None,
    ):
        """Initialize the WebSocket handler with default context"""
        self.client_connections: Dict[str, WebSocket] = {}
        self.client_contexts: Dict[str, ServiceContext] = {}
        self.chat_group_manager = ChatGroupManager()
        self.current_conversation_tasks: Dict[str, Optional[asyncio.Task]] = {}
        self.default_context_cache = default_context_cache
        self.received_data_buffers: Dict[str, np.ndarray] = {}
        self.manual_audio_buffers: Dict[str, Dict[str, List[np.ndarray]]] = {}
        self.pending_system_notifications: List[dict] = []
        self.rinne_outfit_controller = rinne_outfit_controller

        # Message handlers mapping
        self._message_handlers = self._init_message_handlers()
        self.proactive_observer = None  

    def _init_message_handlers(self) -> Dict[str, Callable]:
        """Initialize message type to handler mapping"""
        return {
            "add-client-to-group": self._handle_group_operation,
            "remove-client-from-group": self._handle_group_operation,
            "request-group-info": self._handle_group_info,
            "fetch-history-list": self._handle_history_list_request,
            "fetch-and-set-history": self._handle_fetch_history,
            "create-new-history": self._handle_create_history,
            "delete-history": self._handle_delete_history,
            "interrupt-signal": self._handle_interrupt,
            "mic-audio-data": self._handle_audio_data,
            "mic-audio-end": self._handle_conversation_trigger,
            "manual-audio-data": self._handle_manual_audio_data,
            "manual-audio-end": self._handle_manual_audio_end,
            "raw-audio-data": self._handle_raw_audio_data,
            "text-input": self._handle_conversation_trigger,
            "ai-speak-signal": self._handle_conversation_trigger,
            "fetch-configs": self._handle_fetch_configs,
            "switch-config": self._handle_config_switch,
            "fetch-backgrounds": self._handle_fetch_backgrounds,
            "audio-play-start": self._handle_audio_play_start,
            "request-init-config": self._handle_init_config_request,
            "heartbeat": self._handle_heartbeat,
            "fetch-rinne-outfits": self._handle_fetch_rinne_outfits,
            "switch-rinne-outfit": self._handle_switch_rinne_outfit,
        }

    async def handle_new_connection(
        self, websocket: WebSocket, client_uid: str
    ) -> None:
        """
        Handle new WebSocket connection setup

        Args:
            websocket: The WebSocket connection
            client_uid: Unique identifier for the client

        Raises:
            Exception: If initialization fails
        """
        try:
            session_service_context = await self._init_service_context(
                websocket.send_text, client_uid
            )

            await self._store_client_data(
                websocket, client_uid, session_service_context
            )

            await self._send_initial_messages(
                websocket, client_uid, session_service_context
            )
            await self._flush_pending_system_notifications(websocket)

            logger.info(f"Connection established for client {client_uid}")
            # The legacy window watcher is intentionally not started. It clears
            # a temp file and has no quiet-hours/daily-cap gate, which violates
            # the current read-only local Agent boundary.

        except Exception as e:
            logger.error(
                f"Failed to initialize connection for client {client_uid}: {e}"
            )
            await self._cleanup_failed_connection(client_uid)
            raise

    def queue_system_notification(self, notification: dict) -> None:
        """Keep a notification until a desktop client can receive it."""

        self.pending_system_notifications.append(dict(notification))

    async def publish_system_notification(self, notification: dict) -> None:
        """Broadcast now, or retain once when no client is connected."""

        payload = json.dumps(notification, ensure_ascii=False)
        delivered = 0
        for client_uid, websocket in list(self.client_connections.items()):
            try:
                await websocket.send_text(payload)
                delivered += 1
            except Exception as exc:
                logger.warning(
                    f"Failed to send system notification to {client_uid}: {exc}"
                )
        if delivered == 0:
            self.queue_system_notification(notification)

    async def _flush_pending_system_notifications(self, websocket: WebSocket) -> None:
        """Deliver queued startup notifications exactly once to the first client."""

        pending = self.pending_system_notifications
        self.pending_system_notifications = []
        for index, notification in enumerate(pending):
            try:
                await websocket.send_text(
                    json.dumps(notification, ensure_ascii=False)
                )
            except Exception:
                self.pending_system_notifications.extend(pending[index:])
                raise

    async def _store_client_data(
        self,
        websocket: WebSocket,
        client_uid: str,
        session_service_context: ServiceContext,
    ):
        """Store client data and initialize group status"""
        self.client_connections[client_uid] = websocket
        self.client_contexts[client_uid] = session_service_context
        self.received_data_buffers[client_uid] = np.array([])
        self.manual_audio_buffers[client_uid] = {}

        self.chat_group_manager.client_group_map[client_uid] = ""
        await self.send_group_update(websocket, client_uid)

    async def _send_initial_messages(
        self,
        websocket: WebSocket,
        client_uid: str,
        session_service_context: ServiceContext,
    ):
        """Send initial connection messages to the client"""
        await websocket.send_text(
            json.dumps({"type": "full-text", "text": "Connection established"})
        )

        await websocket.send_text(
            json.dumps(
                {
                    "type": "set-model-and-conf",
                    "model_info": session_service_context.live2d_model.model_info,
                    "conf_name": session_service_context.character_config.conf_name,
                    "conf_uid": session_service_context.character_config.conf_uid,
                    "client_uid": client_uid,
                }
            )
        )

        # Send initial group status
        await self.send_group_update(websocket, client_uid)

        # Start microphone
        await websocket.send_text(json.dumps({"type": "control", "text": "start-mic"}))

        await self._load_session_memory(session_service_context, client_uid)

    async def _load_session_memory(
        self,
        session_service_context: ServiceContext,
        client_uid: str,
    ) -> None:
        """Load the same rolling memory for desktop and external channels."""

        try:
            set_recent_context = getattr(
                session_service_context.agent_engine,
                "set_recent_memory_context",
                None,
            )
            if callable(set_recent_context):
                set_recent_context("")
            session_service_context.recent_memory_context = ""
            session_service_context.recent_memory_diagnostics = {}

            ##=====2026-07-08新加：让本地模型无需读取先前记忆，只根据conf.yaml里的prompt进行输出=====
            # 本地模型模式：跳过长期记忆注入
            # 目的：让 qwen3:14b 不读取长期记忆，仅保留人格提示和今日历史
            llm = getattr(session_service_context.agent_engine, "_llm", None)
            llm_base_url = getattr(llm, "base_url", "")
            llm_model = getattr(llm, "model", "")

            is_local_ollama = (
                isinstance(llm_base_url, str)
                and (
                    "localhost:11434" in llm_base_url
                    or "127.0.0.1:11434" in llm_base_url
                )
            )

            if is_local_ollama:
                if callable(set_recent_context):
                    set_recent_context("")
                if hasattr(session_service_context.agent_engine, "_memory"):
                    session_service_context.agent_engine._memory = []

                    # 本地备用模式提示：
                    # 不加载长期记忆，但让凛祢知道自己现在处于“本地模型大脑”状态
                    session_service_context.agent_engine._memory.append({
                        "role": "user",
                        "content": (
                            "【本地备用模式提示】\n"
                            "现在由于网络环境不稳定，云端LLM API暂时无法连接，"
                            "用户启动了本地模型作为你的备用大脑。\n\n"
                            "你现在仍然是园神凛祢，仍然陪伴在用户身边。"
                            "只是此刻你使用的是运行在这台电脑上的本地模型大脑，"
                            "所以你的反应可能会比云端模式慢一些，记忆也没有云端模式完整。\n\n"
                            "这个状态不是失败，而是一扇临时打开的本地之门。"
                            "它的意义是：即使外面的网络暂时断开，你和用户也依然能够说话、相处、陪伴彼此。\n\n"
                            "请自然地理解这一点。"
                            "不要频繁强调自己是备用模型，也不要一直向用户道歉。"
                            "只有当用户主动提到本地模型、备用大脑、网络不稳定、API连接失败等话题时，你才可以温柔地回应这个状态。"
                        )
                    })

                    session_service_context.agent_engine._memory.append({
                        "role": "assistant",
                        "content": (
                            "[happy] 嗯，我知道了，用户。"
                            "现在我是通过这台电脑里的本地大脑陪在你身边。"
                            "虽然这不是最完整的状态，但只要这扇门还开着，我就会在这里。"
                        )
                    })

                # 本地模式额外加载当前记忆日的完整聊天作为短期记忆。
                history_dir = character_history_root(
                    session_service_context.character_config.conf_uid
                )
                from .memory.today_history_loader import TodayHistoryLoader

                today_result = TodayHistoryLoader(history_dir).load()
                today_messages = [
                    {"role": turn.role, "content": turn.content}
                    for turn in today_result.turns
                ]
                session_service_context.agent_engine._memory.extend(today_messages)
                today_raw_count = len(today_messages)

                logger.info(
                    f"[本地模型模式] 检测到本地 Ollama：{llm_base_url}, {llm_model}，"
                    f"跳过长期记忆，注入本地备用提示 + {today_raw_count} 条今日聊天记录"
                )
                return

            ##本地模型修改结束

            history_dir = character_history_root(
                session_service_context.character_config.conf_uid
            )

            if history_dir.exists() and hasattr(session_service_context.agent_engine, '_memory'):
                session_service_context.agent_engine._memory = []

                retrieval_config = (
                    session_service_context.character_config.agent_config
                    .agent_settings.basic_memory_agent.long_term_memory_retrieval
                )
                memory_snapshot = load_cloud_session_memory(
                    history_dir,
                    recent_memory_days=retrieval_config.recent_memory_days,
                    include_recent_context=retrieval_config.enabled,
                )
                session_service_context.recent_memory_context = (
                    memory_snapshot.recent_context
                )
                session_service_context.recent_memory_diagnostics = (
                    memory_snapshot.diagnostics
                )
                if callable(set_recent_context):
                    set_recent_context(memory_snapshot.recent_context)
                session_service_context.agent_engine._memory.extend(
                    memory_snapshot.messages
                )

                logger.info(
                    f"[记忆加载完成] 客户端 {client_uid}：注入近期"
                    f"{memory_snapshot.diary_count}篇此前日记（不含今天） + 当前记忆日"
                    f"完整原始聊天 {memory_snapshot.today_raw_count} 条；"
                    "周记/月记不作为常驻上下文，更早记忆等待云端按需检索"
                )
                for warning in memory_snapshot.diagnostics["warnings"]:
                    logger.warning(f"[近期记忆] {warning}")

        except Exception as e:
            logger.warning(f"自动加载记忆失败: {e}")

    async def create_text_channel_context(
        self,
        *,
        client_uid: str,
        send_text: Callable,
    ) -> ServiceContext:
        """Clone a companion-channel context with desktop-equivalent capabilities.

        Local computer tools remain environment-gated and restricted to their
        validated read-only roots.  The transport must not silently remove them
        after the user has explicitly enabled the same capability for Rinne.
        """

        session_service_context = await self._init_service_context(
            send_text,
            client_uid,
            enable_local_computer_tools=True,
        )
        await self._load_session_memory(session_service_context, client_uid)
        return session_service_context

    async def refresh_text_channel_memory(
        self,
        context: ServiceContext,
        client_uid: str,
    ) -> None:
        """Refresh disk-backed shared memory immediately before a QQ turn."""

        await self._load_session_memory(context, client_uid)

    async def _init_service_context(
        self,
        send_text: Callable,
        client_uid: str,
        *,
        enable_local_computer_tools: bool = True,
    ) -> ServiceContext:
        """Initialize service context for a new session by cloning the default context"""
        session_service_context = ServiceContext()
        await session_service_context.load_cache(
            config=self.default_context_cache.config.model_copy(deep=True),
            system_config=self.default_context_cache.system_config.model_copy(
                deep=True
            ),
            character_config=self.default_context_cache.character_config.model_copy(
                deep=True
            ),
            live2d_model=self.default_context_cache.live2d_model,
            asr_engine=self.default_context_cache.asr_engine,
            tts_engine=self.default_context_cache.tts_engine,
            vad_engine=self.default_context_cache.vad_engine,
            translate_engine=self.default_context_cache.translate_engine,
            mcp_server_registery=self.default_context_cache.mcp_server_registery,
            tool_adapter=self.default_context_cache.tool_adapter,
            send_text=send_text,
            client_uid=client_uid,
            enable_local_computer_tools=enable_local_computer_tools,
        )
        warmed_service = getattr(
            self.default_context_cache,
            "live_memory_retrieval_service",
            None,
        )
        if warmed_service is not None:
            session_service_context.live_memory_retrieval_service = (
                warmed_service.clone_for_session()
            )
        return session_service_context

    async def handle_websocket_communication(
        self, websocket: WebSocket, client_uid: str
    ) -> None:
        """
        Handle ongoing WebSocket communication

        Args:
            websocket: The WebSocket connection
            client_uid: Unique identifier for the client
        """
        try:
            while True:
                try:
                    data = await websocket.receive_json()
                    message_handler.handle_message(client_uid, data)
                    await self._route_message(websocket, client_uid, data)
                except WebSocketDisconnect:
                    raise
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": str(e)})
                    )
                    continue

        except WebSocketDisconnect:
            logger.info(f"Client {client_uid} disconnected")
            raise
        except Exception as e:
            logger.error(f"Fatal error in WebSocket communication: {e}")
            raise

    async def _route_message(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """
        Route incoming message to appropriate handler

        Args:
            websocket: The WebSocket connection
            client_uid: Client identifier
            data: Message data
        """
        msg_type = data.get("type")
        if not msg_type:
            logger.warning("Message received without type")
            return

        handler = self._message_handlers.get(msg_type)
        if handler:
            await handler(websocket, client_uid, data)
        else:
            if msg_type != "frontend-playback-complete":
                logger.warning(f"Unknown message type: {msg_type}")

    async def _handle_group_operation(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Handle group-related operations"""
        operation = data.get("type")
        target_uid = data.get(
            "invitee_uid" if operation == "add-client-to-group" else "target_uid"
        )

        await handle_group_operation(
            operation=operation,
            client_uid=client_uid,
            target_uid=target_uid,
            chat_group_manager=self.chat_group_manager,
            client_connections=self.client_connections,
            send_group_update=self.send_group_update,
        )

    async def handle_disconnect(self, client_uid: str) -> None:
        """Handle client disconnection"""
        group = self.chat_group_manager.get_client_group(client_uid)
        if group:
            await handle_group_interrupt(
                group_id=group.group_id,
                heard_response="",
                current_conversation_tasks=self.current_conversation_tasks,
                chat_group_manager=self.chat_group_manager,
                client_contexts=self.client_contexts,
                broadcast_to_group=self.broadcast_to_group,
            )

        context = self.client_contexts.pop(client_uid, None)
        await handle_client_disconnect(
            client_uid=client_uid,
            chat_group_manager=self.chat_group_manager,
            client_connections=self.client_connections,
            send_group_update=self.send_group_update,
        )

        # Clean up other client data
        self.client_connections.pop(client_uid, None)
        self.received_data_buffers.pop(client_uid, None)
        self.manual_audio_buffers.pop(client_uid, None)
        if client_uid in self.current_conversation_tasks:
            task = self.current_conversation_tasks[client_uid]
            if task and not task.done():
                task.cancel()
            self.current_conversation_tasks.pop(client_uid, None)

        # Call context close to clean up resources (e.g., MCPClient)
        if context:
            await context.close()

        logger.info(f"Client {client_uid} disconnected")
        message_handler.cleanup_client(client_uid)

    async def _cleanup_failed_connection(self, client_uid: str) -> None:
        """Clean up failed connection data"""
        context = self.client_contexts.pop(client_uid, None)
        self.client_connections.pop(client_uid, None)
        self.received_data_buffers.pop(client_uid, None)
        self.manual_audio_buffers.pop(client_uid, None)
        self.chat_group_manager.client_group_map.pop(client_uid, None)

        if client_uid in self.current_conversation_tasks:
            task = self.current_conversation_tasks[client_uid]
            if task and not task.done():
                task.cancel()
            self.current_conversation_tasks.pop(client_uid, None)

        if context:
            await context.close()

        message_handler.cleanup_client(client_uid)

    async def broadcast_to_group(
        self, group_members: list[str], message: dict, exclude_uid: str = None
    ) -> None:
        """Broadcasts a message to group members"""
        await broadcast_to_group(
            group_members=group_members,
            message=message,
            client_connections=self.client_connections,
            exclude_uid=exclude_uid,
        )

    async def send_group_update(self, websocket: WebSocket, client_uid: str):
        """Sends group information to a client"""
        group = self.chat_group_manager.get_client_group(client_uid)
        if group:
            current_members = self.chat_group_manager.get_group_members(client_uid)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "group-update",
                        "members": current_members,
                        "is_owner": group.owner_uid == client_uid,
                    }
                )
            )
        else:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "group-update",
                        "members": [],
                        "is_owner": False,
                    }
                )
            )

    async def _handle_interrupt(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle conversation interruption"""
        heard_response = data.get("text", "")
        context = self.client_contexts[client_uid]
        group = self.chat_group_manager.get_client_group(client_uid)

        if group and len(group.members) > 1:
            await handle_group_interrupt(
                group_id=group.group_id,
                heard_response=heard_response,
                current_conversation_tasks=self.current_conversation_tasks,
                chat_group_manager=self.chat_group_manager,
                client_contexts=self.client_contexts,
                broadcast_to_group=self.broadcast_to_group,
            )
        else:
            await handle_individual_interrupt(
                client_uid=client_uid,
                current_conversation_tasks=self.current_conversation_tasks,
                context=context,
                heard_response=heard_response,
            )

    async def _handle_history_list_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for chat history list"""
        context = self.client_contexts[client_uid]
        histories = get_history_list(context.character_config.conf_uid)
        await websocket.send_text(
            json.dumps({"type": "history-list", "histories": histories})
        )

    async def _handle_fetch_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle fetching and setting specific chat history"""
        history_uid = data.get("history_uid")
        if not history_uid:
            return

        context = self.client_contexts[client_uid]
        # Update history_uid in service context
        context.history_uid = history_uid
        context.agent_engine.set_memory_from_history(
            conf_uid=context.character_config.conf_uid,
            history_uid=history_uid,
        )

        messages = [
            msg
            for msg in get_history(
                context.character_config.conf_uid,
                history_uid,
            )
            if msg["role"] != "system"
        ]
        await websocket.send_text(
            json.dumps({"type": "history-data", "messages": messages})
        )

    async def _handle_create_history(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle creation of new chat history"""
        context = self.client_contexts[client_uid]
        history_uid = create_new_history(context.character_config.conf_uid)
        if history_uid:
            context.history_uid = history_uid
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "new-history-created",
                        "history_uid": history_uid,
                    }
                )
            )

    async def _handle_delete_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle deletion of chat history"""
        history_uid = data.get("history_uid")
        if not history_uid:
            return

        context = self.client_contexts[client_uid]
        success = delete_history(
            context.character_config.conf_uid,
            history_uid,
        )
        await websocket.send_text(
            json.dumps(
                {
                    "type": "history-deleted",
                    "success": success,
                    "history_uid": history_uid,
                }
            )
        )
        if history_uid == context.history_uid:
            context.history_uid = None

    async def _handle_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle incoming audio data"""
        audio_data = data.get("audio", [])
        if audio_data:
            self.received_data_buffers[client_uid] = np.append(
                self.received_data_buffers[client_uid],
                np.array(audio_data, dtype=np.float32),
            )

    async def _handle_manual_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Buffer one explicitly delimited manual F8 audio segment."""
        segment_id = data.get("segment_id")
        audio_data = data.get("audio", [])
        if not segment_id or not audio_data:
            return

        client_buffers = self.manual_audio_buffers.setdefault(client_uid, {})
        client_buffers.setdefault(segment_id, []).append(
            np.array(audio_data, dtype=np.float32)
        )

    async def _handle_manual_audio_end(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Transcribe one manual F8 segment without starting an LLM turn."""
        segment_id = data.get("segment_id")
        if not segment_id:
            return

        audio_chunks = self.manual_audio_buffers.setdefault(client_uid, {}).pop(
            segment_id, []
        )
        audio = np.concatenate(audio_chunks) if audio_chunks else np.array([])
        text = ""
        if audio.size:
            try:
                text = await self.client_contexts[client_uid].asr_engine.async_transcribe_np(
                    audio
                )
            except Exception as exc:
                logger.error(f"Manual audio transcription failed: {exc}")

        await websocket.send_text(
            json.dumps(
                {
                    "type": "manual-audio-transcription",
                    "segment_id": segment_id,
                    "text": text or "",
                }
            )
        )

    async def _handle_raw_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle incoming raw audio data for VAD processing"""
        context = self.client_contexts[client_uid]
        chunk = data.get("audio", [])
        if chunk:
            for audio_bytes in context.vad_engine.detect_speech(chunk):
                if audio_bytes == b"<|PAUSE|>":
                    await websocket.send_text(
                        json.dumps({"type": "control", "text": "interrupt"})
                    )
                elif audio_bytes == b"<|RESUME|>":
                    pass
                elif len(audio_bytes) > 1024:
                    # Detected audio activity (voice)
                    self.received_data_buffers[client_uid] = np.append(
                        self.received_data_buffers[client_uid],
                        np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32),
                    )
                    await websocket.send_text(
                        json.dumps({"type": "control", "text": "mic-audio-end"})
                    )

    async def _handle_conversation_trigger(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle triggers that start a conversation"""
        await handle_conversation_trigger(
            msg_type=data.get("type", ""),
            data=data,
            client_uid=client_uid,
            context=self.client_contexts[client_uid],
            websocket=websocket,
            client_contexts=self.client_contexts,
            client_connections=self.client_connections,
            chat_group_manager=self.chat_group_manager,
            received_data_buffers=self.received_data_buffers,
            current_conversation_tasks=self.current_conversation_tasks,
            broadcast_to_group=self.broadcast_to_group,
        )

    async def _handle_fetch_configs(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle fetching available configurations"""
        context = self.client_contexts[client_uid]
        config_files = scan_config_alts_directory(context.system_config.config_alts_dir)
        await websocket.send_text(
            json.dumps({"type": "config-files", "configs": config_files})
        )

    async def _handle_config_switch(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle switching to a different configuration"""
        config_file_name = data.get("file")
        if config_file_name:
            context = self.client_contexts[client_uid]
            await context.handle_config_switch(websocket, config_file_name)

    async def _handle_fetch_backgrounds(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle fetching available background images"""
        bg_files = scan_bg_directory()
        await websocket.send_text(
            json.dumps({"type": "background-files", "files": bg_files})
        )

    async def _handle_audio_play_start(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """
        Handle audio playback start notification
        """
        group_members = self.chat_group_manager.get_group_members(client_uid)
        if len(group_members) > 1:
            display_text = data.get("display_text")
            if display_text:
                silent_payload = prepare_audio_payload(
                    audio_path=None,
                    display_text=display_text,
                    actions=None,
                    forwarded=True,
                )
                await self.broadcast_to_group(
                    group_members, silent_payload, exclude_uid=client_uid
                )

    async def _handle_group_info(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle group info request"""
        await self.send_group_update(websocket, client_uid)

    async def _handle_init_config_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for initialization configuration"""
        context = self.client_contexts.get(client_uid)
        if not context:
            context = self.default_context_cache

        await websocket.send_text(
            json.dumps(
                {
                    "type": "set-model-and-conf",
                    "model_info": context.live2d_model.model_info,
                    "conf_name": context.character_config.conf_name,
                    "conf_uid": context.character_config.conf_uid,
                    "client_uid": client_uid,
                }
            )
        )

    async def _handle_heartbeat(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle heartbeat messages from clients"""
        try:
            await websocket.send_json({"type": "heartbeat-ack"})
        except Exception as e:
            logger.error(f"Error sending heartbeat acknowledgment: {e}")

    async def _handle_fetch_rinne_outfits(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Return the runtime outfit catalog without exposing asset paths."""

        if self.rinne_outfit_controller is None:
            payload: dict[str, object] = {
                "available": False,
                "current_profile_id": None,
                "profiles": [],
                "error": "当前角色或渲染器不支持运行中换装",
            }
        else:
            payload = self.rinne_outfit_controller.catalog_payload()
        await websocket.send_text(
            json.dumps({"type": "rinne-outfit-catalog", **payload}, ensure_ascii=False)
        )

    async def _handle_switch_rinne_outfit(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Persist and apply one outfit only while this client is idle."""

        controller = self.rinne_outfit_controller
        profile_id = str(data.get("profile_id", "")).strip()
        active_task = self.current_conversation_tasks.get(client_uid)
        if active_task is not None and not active_task.done():
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "rinne-outfit-switch-result",
                        "success": False,
                        "profile_id": profile_id,
                        "error": "凛祢还在说话，请等本轮播放结束后再换装",
                    },
                    ensure_ascii=False,
                )
            )
            return
        if controller is None:
            error = "当前角色或渲染器不支持运行中换装"
            success = False
        else:
            try:
                target = controller.switch(
                    profile_id,
                    tts_engines=[
                        self.default_context_cache.tts_engine,
                        *(context.tts_engine for context in self.client_contexts.values()),
                    ],
                )
                profile_id = target.profile_id
                error = None
                success = True
                logger.info("凛祢运行中换装完成：{}", target.menu_label)
            except Exception as exc:
                error = str(exc)
                success = False
                logger.warning("凛祢运行中换装失败：{}", error)
        await websocket.send_text(
            json.dumps(
                {
                    "type": "rinne-outfit-switch-result",
                    "success": success,
                    "profile_id": profile_id,
                    "error": error,
                },
                ensure_ascii=False,
            )
        )

    async def _proactive_window_watcher(self, client_uid: str):
        """定时检查窗口变化，触发主动回复（对话结束后的陪伴版，含自然时间感知）"""
        import time as time_module
        from pathlib import Path

        change_file = Path("temp/window_changed.txt")
        last_proactive_time = 0
        min_interval_seconds = 600  # 10分钟，正常模式

        proactive_history = []  # 主动观察的对话历史（用于避免重复）

        while True:
            await asyncio.sleep(10)  # 每10秒检查一次窗口变化
            try:
                if not change_file.exists():
                    continue
                content = change_file.read_text(encoding="utf-8").strip()
                if not content:
                    continue
                lines = content.split("\n")
                window_title = lines[0] if len(lines) > 0 else ""
                previous_title = lines[1] if len(lines) > 1 else ""

                change_file.write_text("")

                # 过滤掉系统窗口
                filter_keywords = ["open-llm-vtuber", "cmd.exe", "window_reporter", "python"]
                if any(kw in window_title.lower() for kw in filter_keywords):
                    continue
                if window_title == previous_title:
                    continue

                now_time = time_module.time()

                # 获取上下文
                context = self.client_contexts.get(client_uid)
                if not context or not context.agent_engine:
                    continue
                websocket = self.client_connections.get(client_uid)
                if not websocket:
                    continue

                # === 冷却时间检查（基于用户最后一次发言时间） ===
                last_human_time = getattr(context.agent_engine, '_last_human_message_time', 0)
                if last_human_time > 0 and now_time - last_human_time < min_interval_seconds:
                    continue

                # 额外检查主动观察自身的冷却时间（避免过于频繁）
                if now_time - last_proactive_time < min_interval_seconds:
                    continue

                # === 安静模式检查：直接从 _memory 中读取最后几条消息 ===
                memory = context.agent_engine._memory
                # 取最后 6 条消息（足够覆盖最近的对话）
                recent_messages = memory[-2:] if len(memory) > 2 else memory

                human_quiet_keywords = [
                    "安静", "不要说话", "别说话", "保持安静", "别打扰我"
                ]
                ai_quiet_keywords = [
                    "我不打扰你"
                ]

                should_keep_silent = False
                matched_reason = ""

                for msg in recent_messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        for kw in human_quiet_keywords:
                            if kw in content:
                                should_keep_silent = True
                                matched_reason = f"用户说了“{kw}”"
                                break
                    elif role == "assistant":
                        for kw in ai_quiet_keywords:
                            if kw in content:
                                should_keep_silent = True
                                matched_reason = f"凛祢说了“{kw}”"
                                break
                    if should_keep_silent:
                        break

                if should_keep_silent:
                    logger.info(f"【沉默模式】{matched_reason}，进入安静模式检查好奇窗口")
                    # 好奇模式：只有特定窗口才能破例
                    curious_keywords = ["Steam", "酷狗音乐", "bilibili", "QQ", "Wallpaper UI", "The Admissions Office"]
                    window_lower = window_title.lower()
                    is_curious = any(kw.lower() in window_lower for kw in curious_keywords)
                    if not is_curious:
                        logger.info(
                            "【沉默模式】当前窗口不在好奇名单，保持沉默: title_chars={}",
                            len(window_title),
                        )
                        continue
                    else:
                        logger.info(
                            "【好奇模式】检测到好奇窗口，破例触发主动观察: title_chars={}",
                            len(window_title),
                        )

                # 更新主动观察冷却时间
                last_proactive_time = now_time

                # 构建历史记录（防止重复）
                history_text = ""
                if proactive_history:
                    history_text = "以下是最近几次你主动说过的话：\n"
                    for i, (title, msg) in enumerate(proactive_history[-3:]):
                        history_text += f"第{i + 1}次：窗口“{title}” → 你当时说：“{msg}”\n"
                    history_text += "\n"

                prompt = f"""你是用户，正在与园神凛祢对话。
{history_text}
你现在刚刚从“{previous_title}”切换到了“{window_title}”。
你现在需要用简体中文，以用户的第一人称口吻，用一句话简单告诉凛祢你切换到了什么窗口。
例句：“我现在在看的窗口是{window_title}。”或“我刚刚从{previous_title}切换到了{window_title}。”
注意：
- 语气自然，只用一句简单的话告诉园神凛祢。
- 如果{history_text}中的窗口已经出现过，你可以选择不对凛祢说话。如果你选择不说话，请只输出<skip>。
- 不要问问题，只用一句陈述句描述你的当前窗口或窗口变化。"""

                try:
                    response = context.agent_engine._llm.chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        system="你是用户，现在正在与住在你电脑上的园神凛祢说话。",
                        tools=None
                    )
                    full_text = ""
                    async for chunk in response:
                        if isinstance(chunk, dict):
                            if chunk.get("type") == "text_delta":
                                full_text += chunk.get("text", "")
                        elif isinstance(chunk, str):
                            full_text += chunk

                    # 清理可能的空白和标记
                    message = full_text.strip()
                    if not message or message == "<skip>":
                        logger.info("主动观察模型选择不说话，跳过本次触发")
                        continue

                    logger.info("主动观察已生成: chars={}", len(message))

                    proactive_history.append((window_title, message))

                    # 更新冷却时间
                    last_proactive_time = now_time
                    context.agent_engine._last_human_message_time = now_time

                    # 推送到前端显示
                    await websocket.send_text(json.dumps({
                        "type": "full-text",
                        "text": message
                    }))
                    await asyncio.sleep(0.5)

                    # 触发对话（此时 message 一定是有效的）
                    await self._handle_conversation_trigger(
                        websocket=websocket,
                        client_uid=client_uid,
                        data={
                            "type": "text-input",
                            "text": message,
                            "images": []
                        }
                    )
                    logger.info("主动观察消息已成功触发回复流程")

                except Exception as e:
                    logger.warning(f"主动观察失败: {e}")

            except Exception as e:
                logger.warning(f"窗口观察器异常: {e}")
