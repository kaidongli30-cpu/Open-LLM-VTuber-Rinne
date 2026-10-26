"""Text-only local teaching, reusing the existing conversation and MCP session."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import ipaddress
import json
import re
from typing import Any
from urllib.parse import urlsplit


teaching_match_id: ContextVar[str | None] = ContextVar(
    "ra3_teaching_match_id", default=None
)
RA3_MUTATIONS = frozenset({"ra3_submit_task", "ra3_cancel_task", "ra3_revise_task"})
_REQUEST_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_MAX_REQUESTS = 1024  # Never evict a key and accidentally execute it twice.


def _valid_match(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and len(value) <= 512
        and "\x00" not in value
    )


def _request_id(data: dict) -> str | None:
    value = data.get("request_id")
    return value if isinstance(value, str) and _REQUEST_ID.fullmatch(value) else None


async def _payload(client: Any, tool: str, arguments: dict) -> dict:
    result = await client.call_tool("rinne-ra3", tool, arguments)
    items = result.get("content_items")
    if (
        not isinstance(items, list)
        or len(items) != 1
        or not isinstance(items[0], dict)
        or items[0].get("type") != "text"
    ):
        raise ValueError("Invalid game response")
    value = json.loads(items[0]["text"])
    if not isinstance(value, dict):
        raise ValueError("Invalid game response")
    return value


async def _current_state(client: Any) -> tuple[str | None, bool]:
    value = await _payload(client, "ra3_current_state", {})
    current = value.get("current")
    if (
        value.get("status") not in {"current", "offline"}
        or not isinstance(value.get("offline"), bool)
        or (current is not None and not isinstance(current, dict))
    ):
        raise ValueError("Invalid game state")
    if (
        value.get("status") == "current"
        and value.get("offline") is False
        and isinstance(current, dict)
        and current.get("online") is True
        and _valid_match(current.get("match_id"))
    ):
        return current["match_id"], True
    return None, True


async def current_match(client: Any) -> str | None:
    """Only the MCP server decides freshness; never trust a browser match ID."""
    match_id, _ = await _current_state(client)
    return match_id


async def teaching_tool_rejection(
    client: Any, tool: str, arguments: Any, server: str
) -> str | None:
    """Check immediately before a teaching turn dispatches a game mutation."""
    captured = teaching_match_id.get()
    if captured is None or tool not in RA3_MUTATIONS:
        return None
    try:
        if server != "rinne-ra3" or not isinstance(arguments, dict):
            return "教学工具参数无效，操作未提交。"
        if await current_match(client) != captured:
            return "教学对应的对局已离线或变化，操作未提交。"
        if tool == "ra3_submit_task":
            if arguments.get("match_id") != captured:
                return "不能把本轮教学提交到其他对局。"
        else:
            task_id = arguments.get("task_id")
            if not isinstance(task_id, str) or not task_id or len(task_id) > 128:
                return "任务编号无效，操作未提交。"
            value = await _payload(
                client, "ra3_task_status", {"task_id": task_id, "match_id": captured}
            )
            task = value.get("task")
            if (
                value.get("status") != "task"
                or value.get("offline") is not False
                or not isinstance(task, dict)
                or task.get("id") != task_id
                or task.get("match_id") != captured
            ):
                return "这条任务不属于本轮教学对局，操作未提交。"
            # The task lookup also awaited I/O: recheck before dispatch.
            if await current_match(client) != captured:
                return "教学对应的对局已离线或变化，操作未提交。"
    except Exception:
        return "无法核对教学对应的对局，操作未提交。"
    return None


def _loopback(websocket: Any) -> bool:
    try:
        # A tunnel can make a remote browser appear to originate at loopback.
        # Teaching is only the direct local desktop connection, not that proxy.
        headers = websocket.headers
        proxy_headers = {
            "forwarded",
            "x-forwarded-for",
            "x-real-ip",
            "cf-connecting-ip",
            "true-client-ip",
            "x-forwarded-host",
        }
        if any(name in headers for name in proxy_headers):
            return False
        hostname = urlsplit("//" + headers.get("host", "")).hostname
        if hostname not in {"127.0.0.1", "localhost", "::1"}:
            return False
        origin = headers.get("origin")
        if origin not in {None, "null", "file://"}:
            if urlsplit(origin).hostname not in {"127.0.0.1", "localhost", "::1"}:
                return False
        address = ipaddress.ip_address(websocket.client.host)
        return address.is_loopback or (
            isinstance(address, ipaddress.IPv6Address)
            and address.ipv4_mapped is not None
            and address.ipv4_mapped.is_loopback
        )
    except (ValueError, AttributeError, TypeError):
        return False


@dataclass
class _Session:
    # Hash only the submitted match/text pair; no transcript or attachments.
    requests: dict[str, tuple[str, asyncio.Future]] = field(default_factory=dict)
    admitting: bool = False


class TeachingWebSocket:
    """Tag only this turn's outgoing packets; delegate the original connection."""

    def __init__(self, websocket: Any, request_id: str) -> None:
        self._websocket = websocket
        self._request_id = request_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._websocket, name)

    def _tag(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {**data, "game_command_request_id": self._request_id}
        return data

    async def send_text(self, data: str) -> None:
        try:
            value = json.loads(data)
        except (ValueError, TypeError):
            await self._websocket.send_text(data)
            return
        if isinstance(value, dict):
            data = json.dumps(self._tag(value), ensure_ascii=False)
        await self._websocket.send_text(data)

    async def send_json(self, data: Any, **kwargs: Any) -> None:
        await self._websocket.send_json(self._tag(data), **kwargs)


class Ra3TeachingGateway:
    """Per-connection admission and idempotency; no game command queue."""

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}

    def discard(self, client_uid: str) -> None:
        self._sessions.pop(client_uid, None)

    @staticmethod
    def _allowed(handler: Any, websocket: Any, uid: str) -> bool:
        return (
            _loopback(websocket)
            and handler.client_connections.get(uid) is websocket
            and uid in handler.client_contexts
            and handler.chat_group_manager.get_client_group(uid) is None
        )

    async def state(self, handler: Any, websocket: Any, uid: str, data: dict) -> None:
        request_id = _request_id(data)
        match_id = None
        paired = False
        status = "rejected"
        conclusion = "教学状态请求无效。"
        if (
            self._allowed(handler, websocket, uid)
            and request_id
            and set(data) == {"type", "request_id"}
        ):
            try:
                context = handler.client_contexts[uid]
                match_id, paired = await _current_state(context.mcp_client)
                if (
                    not self._allowed(handler, websocket, uid)
                    or handler.client_contexts.get(uid) is not context
                ):
                    match_id = None
                    paired = False
                status = "online" if match_id else "offline" if paired else "rejected"
                conclusion = (
                    "当前对局可接收教学。"
                    if match_id
                    else "当前对局离线，不能发送教学。"
                )
            except Exception:
                status = "unavailable"
                conclusion = "暂时无法读取游戏状态。"
        await websocket.send_text(
            json.dumps(
                {
                    "type": "ra3-teaching-state",
                    "request_id": request_id,
                    "match_id": match_id,
                    "online": match_id is not None,
                    "paired": paired,
                    "status": status,
                    "conclusion": conclusion,
                },
                ensure_ascii=False,
            )
        )

    async def submit(self, handler: Any, websocket: Any, uid: str, data: dict) -> None:
        request_id = _request_id(data)

        def result(accepted: bool, conclusion: str) -> dict:
            return dict(
                type="ra3-teaching-result",
                request_id=request_id,
                accepted=accepted,
                conclusion=conclusion,
            )

        async def reply(payload: dict) -> None:
            await websocket.send_text(json.dumps(payload, ensure_ascii=False))

        if not self._allowed(handler, websocket, uid):
            await reply(result(False, "教学仅支持本机的独立聊天连接。"))
            return
        text = data.get("text")
        if (
            request_id is None
            or set(data) != {"type", "request_id", "match_id", "text"}
            or not _valid_match(data.get("match_id"))
            or not isinstance(text, str)
            or not text.strip()
            or len(text) > 2000
        ):
            await reply(
                result(False, "教学只接收1至2000字纯文字，请勿附带图像或附件。")
            )
            return

        session = self._sessions.setdefault(uid, _Session())
        fingerprint = hashlib.sha256(
            json.dumps([data["match_id"], text], ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        previous = session.requests.get(request_id)
        if previous:
            if previous[0] != fingerprint:
                await reply(result(False, "请求编号已用于不同内容，未启动新对话。"))
            else:
                await reply(await asyncio.shield(previous[1]))
            return
        if len(session.requests) >= _MAX_REQUESTS:
            await reply(result(False, "本连接的教学请求记录已满，请重新连接后再发送。"))
            return

        future = asyncio.get_running_loop().create_future()
        session.requests[request_id] = (fingerprint, future)
        response = result(False, "凛祢正在回应，请等本轮结束后再发送。")
        if session.admitting or self._busy(handler, uid):
            future.set_result(response)
            await reply(response)
            return
        session.admitting = True
        context = handler.client_contexts[uid]
        try:
            match_id = await current_match(context.mcp_client)
            if match_id is None or match_id != data["match_id"]:
                response = result(
                    False, "对局已离线或变化，教学未发送，也不会自动重发。"
                )
            elif (
                not self._allowed(handler, websocket, uid)
                or handler.client_contexts.get(uid) is not context
                or self._sessions.get(uid) is not session
            ):
                response = result(False, "聊天连接或分组已变化，教学未发送。")
            elif not self._busy(handler, uid):
                before = handler.current_conversation_tasks.get(uid)
                token = teaching_match_id.set(match_id)
                try:
                    await handler._handle_conversation_trigger(
                        TeachingWebSocket(websocket, request_id),
                        uid,
                        {"type": "text-input", "text": text},
                    )
                finally:
                    teaching_match_id.reset(token)
                after = handler.current_conversation_tasks.get(uid)
                if after is not None and after is not before:
                    response = result(
                        True, "已转入凛祢的对话流程；游戏操作是否完成仍需核对结果。"
                    )
        except asyncio.CancelledError:
            response = result(False, "教学接收已中止，不会自动重发。")
            raise
        except Exception:
            response = result(False, "暂时无法接收教学，未自动重发。")
        finally:
            session.admitting = False
            if not future.done():
                future.set_result(response)
        await reply(response)

    @staticmethod
    def _busy(handler: Any, uid: str) -> bool:
        task = handler.current_conversation_tasks.get(uid)
        return task is not None and not task.done()
