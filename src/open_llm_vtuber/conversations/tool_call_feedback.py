import asyncio
from typing import Any, Optional, Set

from ..agent.output_types import DisplayText
from ..live2d_model import Live2dModel
from ..tts.tts_interface import TTSInterface
from .conversation_utils import _translate_text_if_needed
from .tts_manager import TTSTaskManager
from .types import WebSocketSend


class ToolCallFeedbackManager:
    """Emit waiting speech/text once per kind of lookup in a conversation turn."""

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
        waiting_text: str = "我先搜索一下",
        enabled: bool = True,
    ) -> None:
        self._tts_manager = tts_manager
        self._live2d_model = live2d_model
        self._tts_engine = tts_engine
        self._websocket_send = websocket_send
        self._character_name = character_name
        self._character_avatar = character_avatar
        self._translate_engine = translate_engine
        self._waiting_text = waiting_text
        self._enabled = enabled

        self._running_search_tool_ids: Set[str] = set()
        self._announced_in_this_turn: Set[str] = set()

    async def handle_tool_status(self, status_update: dict) -> None:
        """Announce waiting status only on the first search tool call in this turn."""
        if not self._enabled:
            return
        status = status_update.get("status")
        tool_id = status_update.get("tool_id")
        tool_name = status_update.get("tool_name", "")
        is_search_tool = self._is_search_related_tool(tool_name)

        if status == "running" and is_search_tool:
            if tool_id:
                self._running_search_tool_ids.add(tool_id)
            await self._announce_waiting_once("search", self._waiting_text)
            return

        if status in {"completed", "error"} and tool_id:
            self._running_search_tool_ids.discard(tool_id)

    async def announce_memory_recall(self) -> None:
        """Called when the model's hidden long-term-memory lookup starts."""
        await self._announce_waiting_once("memory", "我仔细想想")

    async def _announce_waiting_once(self, kind: str, text: str) -> None:
        if not self._enabled or kind in self._announced_in_this_turn:
            return
        self._announced_in_this_turn.add(kind)
        tts_text = await _translate_text_if_needed(text, self._translate_engine)

        display_text = DisplayText(
            text=text,
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
        # The tool runs as soon as this callback returns.  Send the prompt first,
        # including when the tool blocks the event loop or returns immediately.
        if self._tts_manager.task_list:
            await asyncio.gather(*self._tts_manager.task_list)
        await self._tts_manager.wait_until_payloads_sent()

    async def stop(self) -> None:
        self._running_search_tool_ids.clear()
        self._announced_in_this_turn.clear()

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
