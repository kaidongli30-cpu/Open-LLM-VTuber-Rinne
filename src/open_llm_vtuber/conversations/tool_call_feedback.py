import re
import asyncio
from typing import Any, Optional, Set

from loguru import logger

from ..agent.output_types import DisplayText
from ..live2d_model import Live2dModel
from ..tts.tts_interface import TTSInterface
from .tts_manager import TTSTaskManager
from .types import WebSocketSend


class ToolCallFeedbackManager:
    """Emit waiting speech/text for search tools (once per conversation turn)."""

    def __init__(
        self,
        *,
        tts_manager: TTSTaskManager,
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
        websocket_send: WebSocketSend,
        character_name: str,
        character_avatar: str,
        translate_engine: Optional[Any] = None,
        waiting_text: str = "\u68c0\u7d22\u4e2d...\u68c0\u7d22\u4e2d...",
    ) -> None:
        self._tts_manager = tts_manager
        self._live2d_model = live2d_model
        self._tts_engine = tts_engine
        self._websocket_send = websocket_send
        self._character_name = character_name
        self._character_avatar = character_avatar
        self._translate_engine = translate_engine
        self._waiting_text = waiting_text

        self._running_search_tool_ids: Set[str] = set()
        self._announced_in_this_turn = False

    async def handle_tool_status(self, status_update: dict) -> None:
        """Announce waiting status only on the first search tool call in this turn."""
        status = status_update.get("status")
        tool_id = status_update.get("tool_id")
        tool_name = status_update.get("tool_name", "")
        is_search_tool = self._is_search_related_tool(tool_name)

        if status == "running" and is_search_tool:
            if tool_id:
                self._running_search_tool_ids.add(tool_id)
            if not self._announced_in_this_turn:
                self._announced_in_this_turn = True
                await self._announce_waiting_once()
            return

        if status in {"completed", "error"} and tool_id:
            self._running_search_tool_ids.discard(tool_id)

    async def _announce_waiting_once(self) -> None:
        tts_text = self._waiting_text

        display_text = DisplayText(
            text=self._waiting_text,
            name=self._character_name,
            avatar=self._character_avatar,
        )
        await self._tts_manager.speak(
            tts_text=tts_text,
            display_text=display_text,
            actions=None,
            live2d_model=self._live2d_model,
            tts_engine=self._tts_engine,
            websocket_send=self._websocket_send,
        )

    async def stop(self) -> None:
        self._running_search_tool_ids.clear()
        self._announced_in_this_turn = False

    @staticmethod
    def _is_search_related_tool(tool_name: str) -> bool:
        """
        Return True only for search/retrieval style MCP tools.
        This avoids announcing waiting text for non-search tools (e.g. audio player).
        """
        if not tool_name:
            return False

        normalized = tool_name.strip().lower()
        keywords = (
            "search",
            "web_search",
            "ddg",
            "fetch_content",
            "fetch_url",
            "crawl",
            "scrape",
            "query",
            "lookup",
            "retrieve",
            "browser_search",
        )
        return any(keyword in normalized for keyword in keywords)
