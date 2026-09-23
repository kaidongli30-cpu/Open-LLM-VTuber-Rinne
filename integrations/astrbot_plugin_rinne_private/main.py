"""AstrBot Star that turns a personal QQ account into private Rinne transport."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import os
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import File, Image, Plain, Record, Video
from astrbot.api.star import Context, Star, register

from .bridge_client import (
    BridgeClientConfig,
    BridgeClientConfigurationError,
    BridgeTurnError,
    RinneBridgeClient,
)
from .core import AtomicPendingStore, PendingStoreError, StrictOverController
from .media_inbox import MediaInbox, MediaInboxError, VIDEO_SUFFIXES
from .remote_probe import OneBotRemoteProbe
from .transport_health import TransportHealthStore


@register(
    "astrbot_plugin_rinne_private",
    "local",
    "只允许指定 QQ 私聊并在 over 后交给既有凛祢后端",
    "0.7.1",
)
class RinnePrivatePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.allowed_sender_id = str(config.get("allowed_sender_id", "")).strip()
        self._controller: StrictOverController | None = None
        self._bridge_client: RinneBridgeClient | None = None
        self._media_inbox: MediaInbox | None = None
        self._transport_health: TransportHealthStore | None = None
        self._remote_probe: OneBotRemoteProbe | None = None
        self._remote_probe_task: asyncio.Task[None] | None = None
        self._configuration_error = ""
        try:
            raw_data_root = str(config.get("data_root", "")).strip()
            if not raw_data_root:
                raise ValueError("private QQ data_root must be configured")
            data_root = Path(raw_data_root).expanduser()
            if not data_root.is_absolute() or data_root.resolve() == Path(
                data_root.anchor
            ).resolve():
                raise ValueError("private QQ data_root must be an absolute non-root path")
            store = AtomicPendingStore(data_root)
            self._media_inbox = MediaInbox(data_root)
            self._transport_health = TransportHealthStore(data_root)
            self._remote_probe = OneBotRemoteProbe(
                get_bot=self._get_onebot_client,
                sender_id=self.allowed_sender_id,
                health_store=self._transport_health,
            )
            self._controller = StrictOverController(
                store=store,
                max_pending_messages=int(config.get("max_pending_messages", 64)),
                max_pending_chars=int(config.get("max_pending_chars", 45_000)),
            )
            bridge_config = BridgeClientConfig.from_values(
                os.environ.get(
                    "RINNE_PRIVATE_BRIDGE_URL",
                    str(
                        config.get(
                            "bridge_url",
                            "http://127.0.0.1:12393/agent/private/v1/turn",
                        )
                    ),
                )
            )
            self._bridge_client = RinneBridgeClient(bridge_config)
        except (OSError, ValueError, PendingStoreError) as exc:
            self._configuration_error = type(exc).__name__
            logger.error(f"私人凛祢插件配置无效：{type(exc).__name__}")

    def _get_onebot_client(self) -> Any | None:
        platform = self.context.get_platform("aiocqhttp")
        return getattr(platform, "bot", None) if platform is not None else None

    @filter.on_astrbot_loaded()
    async def start_remote_probe(self) -> None:
        if self._remote_probe is None:
            return
        if self._remote_probe_task is None or self._remote_probe_task.done():
            self._remote_probe_task = asyncio.create_task(
                self._remote_probe.run(),
                name="rinne-private-qq-remote-probe",
            )
            logger.info("私人 QQ 远端静默探针已启动")

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=1000)
    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    async def block_all_group_messages(self, event: AstrMessageEvent):
        """This personal Rinne account never participates in group chats."""

        event.stop_event()

    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE, priority=1000)
    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    async def handle_private_message(self, event: AstrMessageEvent):
        """Buffer one private message, or submit the batch after `over`."""

        event.stop_event()
        sender_id = str(event.get_sender_id()).strip()
        if not self.allowed_sender_id or sender_id != self.allowed_sender_id:
            logger.warning("已静默拦截非白名单 QQ 私聊")
            return

        if self._transport_health is not None:
            self._transport_health.record_inbound()

        async def send_text(text: str) -> None:
            await event.send(event.plain_result(text))
            if self._transport_health is not None:
                self._transport_health.record_reply_submitted()

        if self._controller is None or self._bridge_client is None:
            await send_text("私人凛祢插件还没有配置完成。")
            return

        message_id = str(event.message_obj.message_id).strip()
        parts: list[dict[str, object]] = []
        try:
            for component in event.message_obj.message:
                if isinstance(component, Plain):
                    text = str(component.text or "").strip()
                    if text:
                        parts.append({"type": "text", "text": text})
                elif isinstance(component, Image):
                    if self._media_inbox is None:
                        raise MediaInboxError("QQ image URL is unavailable")
                    image_url = str(component.url or "").strip()
                    if image_url:
                        media = await self._media_inbox.stage_url(
                            kind="image",
                            name=Path(str(component.file or "image")).name,
                            url=image_url,
                        )
                    else:
                        try:
                            image_path = await component.convert_to_file_path()
                        except Exception as exc:
                            raise MediaInboxError(
                                "QQ image could not be resolved"
                            ) from exc
                        media = await self._media_inbox.stage_local_media(
                            kind="image",
                            name=Path(str(component.file or image_path)).name,
                            path=image_path,
                        )
                    parts.append({"type": "image", **media})
                elif isinstance(component, Record):
                    if self._media_inbox is None:
                        raise MediaInboxError("QQ audio inbox is unavailable")
                    audio_path = await component.convert_to_file_path()
                    raw_name = str(component.file or "voice").replace("\\", "/")
                    audio_name = f"{Path(raw_name).stem or 'voice'}.wav"
                    audio = await self._media_inbox.stage_local_audio(
                        name=audio_name,
                        path=audio_path,
                    )
                    transcript = await self._bridge_client.transcribe_audio(
                        sender_id,
                        message_id,
                        audio,
                    )
                    if not transcript:
                        await send_text("这段语音没有听清，可以再说一次。")
                        return
                    parts.append({"type": "audio_text", "text": transcript})
                elif isinstance(component, Video):
                    if self._media_inbox is None:
                        raise MediaInboxError("QQ video inbox is unavailable")
                    try:
                        video_path = await component.convert_to_file_path()
                    except Exception as exc:
                        raise MediaInboxError(
                            "QQ video could not be resolved"
                        ) from exc
                    media = await self._media_inbox.stage_local_video(
                        name=Path(str(component.file or video_path)).name,
                        path=video_path,
                    )
                    parts.append({"type": "video", **media})
                elif isinstance(component, File):
                    if self._media_inbox is None:
                        raise MediaInboxError("QQ file inbox is unavailable")
                    file_name = str(component.name or "file")
                    if Path(file_name).suffix.casefold() in VIDEO_SUFFIXES:
                        try:
                            video_path = await component.get_file()
                        except Exception as exc:
                            raise MediaInboxError(
                                "QQ video file could not be resolved"
                            ) from exc
                        media = await self._media_inbox.stage_local_video(
                            name=file_name,
                            path=video_path,
                        )
                        parts.append({"type": "video", **media})
                        continue
                    if not str(component.url or "").strip():
                        raise MediaInboxError("QQ file URL is unavailable")
                    media = await self._media_inbox.stage_url(
                        kind="document",
                        name=file_name,
                        url=str(component.url),
                    )
                    parts.append({"type": "document", **media})
        except MediaInboxError:
            await send_text(
                "这条语音、图片、文件或视频没能安全接收；语音需在 5 分钟内，"
                "图片和文档需在 50MB 内，视频需在 100MB 内。"
            )
            return
        except (BridgeClientConfigurationError, BridgeTurnError):
            await send_text("凛祢的语音识别暂时没有接通，请稍后重发这段语音。")
            return

        async def deliver_sentence(sentence: str) -> None:
            await send_text(sentence)

        async def provide_status(status_sender_id: str) -> str:
            try:
                health = await self._bridge_client.health(status_sender_id)
            except (BridgeClientConfigurationError, BridgeTurnError):
                return (
                    "QQ 通道可以收发消息，但 12394 凛祢后端当前没有接通。\n"
                    "状态：暂时不可回复。"
                )

            ready = bool(health.get("ready_to_reply"))
            lines = [
                "凛祢：可回复" if ready else "凛祢：尚未完全就绪",
                (
                    "本次 /status 已到达 12394；你看到这条回复，"
                    "即表示 NapCat、OneBot、AstrBot 与 QQ 往返正常。"
                ),
                "模型会话核心：已加载（本次未调用模型 API）"
                if health.get("agent_ready")
                else "模型会话核心：未就绪",
                "ASR 语音识别：就绪"
                if health.get("asr_ready")
                else "ASR 语音识别：未就绪",
                "三层记忆：就绪"
                if health.get("memory_ready")
                else "三层记忆：未就绪",
                "图片、文件与视频读取：就绪"
                if health.get("library_ready")
                else "图片、文件与视频读取：未就绪",
            ]
            read_only_status = health.get("read_only_computer")
            if read_only_status == "ready":
                lines.append("电脑文件：只读能力就绪")
            elif read_only_status == "not_enabled":
                lines.append("电脑文件：只读能力未启用（安全关闭）")
            else:
                lines.append("电脑文件：只读能力配置异常")
            unavailable = health.get("optional_mcp_unavailable")
            if isinstance(unavailable, list) and unavailable:
                lines.append(
                    "可选 MCP 降级："
                    + "、".join(str(item) for item in unavailable)
                    + "（不影响基础私聊）"
                )
            checked_at = health.get("checked_at")
            if isinstance(checked_at, str) and checked_at:
                lines.append(f"后端检查时间：{checked_at}")
            return "\n".join(lines)

        try:
            outcome = await self._controller.handle_message(
                sender_id=sender_id,
                message_id=message_id,
                parts=parts,
                submitter=self._bridge_client.stream_turn,
                deliver_sentence=deliver_sentence,
                end_session=self._bridge_client.end_session,
                status_provider=provide_status,
            )
        except BridgeClientConfigurationError:
            await send_text("本机桥接令牌还没有配置，刚才的内容仍然保留着。")
            return
        except BridgeTurnError:
            await send_text(
                "这次没有成功生成回复，刚才的内容还保留着；稍后再发一次 over 就好。"
            )
            return
        except Exception as exc:
            logger.error(f"私人凛祢消息处理失败：{type(exc).__name__}")
            await send_text("这次没有成功提交，内容仍然保留着。")
            return

        if outcome.reply_text:
            await send_text(outcome.reply_text)

    async def terminate(self):
        if self._remote_probe_task is not None:
            self._remote_probe_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._remote_probe_task
            self._remote_probe_task = None
        if self._bridge_client is not None:
            await self._bridge_client.close()
