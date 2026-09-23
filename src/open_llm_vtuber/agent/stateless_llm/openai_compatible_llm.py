"""Description: This file contains the implementation of the `AsyncLLM` class.
This class is responsible for handling asynchronous interaction with OpenAI API compatible
endpoints for language generation.
"""

import asyncio
import contextvars
import copy
import json
import time
import uuid
from typing import AsyncIterator, List, Dict, Any, Callable
import httpx
from openai import (
    AsyncOpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
    NotGiven,
    NOT_GIVEN,
)
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from loguru import logger

from .stateless_llm_interface import StatelessLLMInterface
from .request_limiter import build_backend_key, limit_request_concurrency
from ...mcpp.types import ToolCallObject


_NO_STREAM_CHUNK = object()


class _UpstreamResponseTimeout(TimeoutError):
    """The provider returned no headers or stream data before the deadline."""

    def __init__(self, message: str, attempts: list[dict[str, Any]]):
        super().__init__(message)
        self.attempts = attempts


def _forced_tool_call_from_json(
    content: str,
    *,
    forced_tool_name: str,
) -> ToolCallObject | None:
    """Recover a tool call from providers that ignore OpenAI tool_choice."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    if isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    if not isinstance(payload, dict):
        return None
    name = payload.get("name") or payload.get("tool")
    arguments = payload.get("arguments")
    if name is None and {"query", "question_granularity"} <= payload.keys():
        name = forced_tool_name
        arguments = payload
    if name != forced_tool_name:
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError):
            return None
    if not isinstance(arguments, dict):
        return None
    return ToolCallObject.from_dict(
        {
            "index": 0,
            "id": f"protocol-recovery-{uuid.uuid4().hex}",
            "type": "function",
            "function": {
                "name": forced_tool_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }
    )


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("content")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _forced_tool_selection_payload(
    messages: List[Dict[str, Any]],
) -> dict[str, Any]:
    question = ""
    options: list[dict[str, Any]] = []
    for message in reversed(messages):
        if not question and message.get("role") == "user":
            question = _content_text(message.get("content"))
        if not options and message.get("role") == "tool":
            content = _content_text(message.get("content"))
            json_start = content.find("{")
            if json_start >= 0:
                try:
                    payload = json.loads(content[json_start:])
                except (TypeError, ValueError):
                    payload = {}
                retrieval = payload.get("retrieval", {})
                raw_options = retrieval.get("detail_navigation_options", [])
                if isinstance(raw_options, list):
                    options = [
                        {
                            "seed_event_id": item.get("seed_event_id"),
                            "occurrence_date": item.get("occurrence_date"),
                            "navigation_hint": item.get("navigation_hint"),
                        }
                        for item in raw_options
                        if isinstance(item, dict) and item.get("seed_event_id")
                    ]
        if question and options:
            break
    return {"question": question, "navigation_options": options}


class AsyncLLM(StatelessLLMInterface):
    def __init__(
        self,
        model: str,
        base_url: str,
        llm_api_key: str = "z",
        organization_id: str = "z",
        project_id: str = "z",
        temperature: float = 1.0,
        max_concurrent_requests: int = 1,
        min_request_interval_seconds: float = 0.0,
        upstream_warning_seconds: float = 30.0,
        upstream_first_data_timeout_seconds: float = 90.0,
        upstream_max_attempts: int = 2,
        proxy_url: str | None = None,
    ):
        """
        Initializes an instance of the `AsyncLLM` class.

        Parameters:
        - model (str): The model to be used for language generation.
        - base_url (str): The base URL for the OpenAI API.
        - organization_id (str, optional): The organization ID for the OpenAI API. Defaults to "z".
        - project_id (str, optional): The project ID for the OpenAI API. Defaults to "z".
        - llm_api_key (str, optional): The API key for the OpenAI API. Defaults to "z".
        - temperature (float, optional): What sampling temperature to use, between 0 and 2. Defaults to 1.0.
        """
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_concurrent_requests = max(1, int(max_concurrent_requests))
        self.min_request_interval_seconds = max(
            0.0, float(min_request_interval_seconds)
        )
        self.upstream_first_data_timeout_seconds = max(
            5.0, float(upstream_first_data_timeout_seconds)
        )
        self.upstream_warning_seconds = min(
            self.upstream_first_data_timeout_seconds,
            max(1.0, float(upstream_warning_seconds)),
        )
        self.upstream_max_attempts = max(1, min(2, int(upstream_max_attempts)))
        http_client = None
        if proxy_url:
            http_client = httpx.AsyncClient(
                proxy=proxy_url,
                timeout=httpx.Timeout(600.0, connect=10.0),
            )
        self.client = AsyncOpenAI(
            base_url=base_url,
            organization=organization_id,
            project=project_id,
            api_key=llm_api_key,
            max_retries=0,
            **({"http_client": http_client} if http_client is not None else {}),
        )
        self.support_tools = True
        self._trace_sink: Callable[[Dict[str, Any]], None] | None = None
        self._trace_sink_context = contextvars.ContextVar(
            f"openai_compatible_trace_sink_{id(self)}",
            default=None,
        )
        self._trace_request_index = 0
        self._forced_tool_name_once: str | None = None
        self._limiter_key = build_backend_key(
            provider_name="openai_compatible_llm",
            base_url=base_url,
            organization_id=organization_id,
            project_id=project_id,
            api_key=llm_api_key,
        )

        logger.info(
            "Initialized AsyncLLM with the parameters: "
            f"{self.base_url}, {self.model}, max_concurrent_requests={self.max_concurrent_requests}, "
            f"min_request_interval_seconds={self.min_request_interval_seconds}, "
            f"upstream_warning_seconds={self.upstream_warning_seconds}, "
            "upstream_first_data_timeout_seconds="
            f"{self.upstream_first_data_timeout_seconds}, "
            f"upstream_max_attempts={self.upstream_max_attempts}, "
            f"proxy_enabled={bool(proxy_url)}"
        )

    def set_trace_sink(
        self,
        sink: Callable[[Dict[str, Any]], None] | None,
    ) -> None:
        """Attach an optional request trace sink.

        The desktop pet never installs a sink. Isolated evaluations can use it
        to capture the exact messages and tool schemas without changing the API
        request payload.
        """

        self._trace_sink = sink

    def force_tool_once(self, tool_name: str) -> None:
        """Force one OpenAI-compatible request to select a named function."""

        self._forced_tool_name_once = tool_name

    def push_trace_sink(
        self,
        sink: Callable[[Dict[str, Any]], None] | None,
    ):
        """Install a task-local trace sink and return its reset token."""

        trace_context = getattr(self, "_trace_sink_context", None)
        if trace_context is None:
            trace_context = contextvars.ContextVar(
                f"openai_compatible_trace_sink_{id(self)}",
                default=None,
            )
            self._trace_sink_context = trace_context
        return trace_context.set(sink)

    def reset_trace_sink(self, token) -> None:
        """Restore the previous task-local trace sink."""

        self._trace_sink_context.reset(token)

    def _emit_trace(self, payload: Dict[str, Any]) -> None:
        trace_context = getattr(self, "_trace_sink_context", None)
        trace_sink = trace_context.get() if trace_context is not None else None
        if trace_sink is None:
            trace_sink = getattr(self, "_trace_sink", None)
        if trace_sink is None:
            return
        try:
            trace_sink(copy.deepcopy(payload))
        except Exception as exc:  # pragma: no cover - tracing must never break chat
            logger.warning(
                f"OpenAI-compatible trace sink failed: {type(exc).__name__}: {exc}"
            )

    @staticmethod
    def _message_shape(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        shape = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                content_length = len(content)
            elif isinstance(content, list):
                content_length = sum(len(str(item)) for item in content)
            else:
                content_length = 0
            shape.append(
                {
                    "role": message.get("role"),
                    "content_length": content_length,
                    "has_tool_calls": bool(message.get("tool_calls")),
                    "has_tool_call_id": bool(message.get("tool_call_id")),
                }
            )
        return shape

    async def _wait_for_upstream_stage(
        self,
        task: asyncio.Task,
        *,
        attempt_started: float,
        warning_seconds: float,
        timeout_seconds: float,
        trace_request_id: str,
        attempt: int,
        max_attempts: int,
        stage: str,
    ):
        """Wait for headers/data while emitting one actionable slow warning."""

        elapsed = time.perf_counter() - attempt_started
        warning_remaining = max(0.0, warning_seconds - elapsed)
        if warning_remaining:
            done, _pending = await asyncio.wait({task}, timeout=warning_remaining)
            if done:
                return await task
        elif task.done():
            return await task

        elapsed = time.perf_counter() - attempt_started
        if elapsed >= warning_seconds:
            logger.warning(
                "[LLM upstream] still waiting for {} after {:.1f}s; "
                "trace_id={}, attempt={}/{}",
                stage,
                elapsed,
                trace_request_id,
                attempt,
                max_attempts,
            )

        timeout_remaining = max(
            0.0,
            timeout_seconds - (time.perf_counter() - attempt_started),
        )
        if timeout_remaining:
            done, _pending = await asyncio.wait({task}, timeout=timeout_remaining)
            if done:
                return await task

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise asyncio.TimeoutError

    @staticmethod
    async def _close_stream_after_failed_attempt(stream) -> None:
        if stream is None:
            return
        try:
            await asyncio.wait_for(stream.close(), timeout=5.0)
        except Exception as exc:
            logger.warning(
                "[LLM upstream] failed to close abandoned stream: {}",
                exc,
            )

    async def _open_stream_with_first_data_retry(
        self,
        *,
        request_kwargs: dict[str, Any],
        trace_request_id: str,
    ):
        """Open a stream and retry once only before any provider data arrives."""

        warning_seconds = float(getattr(self, "upstream_warning_seconds", 30.0))
        timeout_seconds = float(
            getattr(self, "upstream_first_data_timeout_seconds", 90.0)
        )
        max_attempts = int(getattr(self, "upstream_max_attempts", 2))
        retryable_statuses = {408, 429, 500, 502, 503, 504}
        attempts: list[dict[str, Any]] = []

        for attempt in range(1, max_attempts + 1):
            attempt_started = time.perf_counter()
            stream = None
            create_task = None
            first_chunk_task = None
            logger.info(
                "[LLM upstream] request sent; trace_id={}, attempt={}/{}",
                trace_request_id,
                attempt,
                max_attempts,
            )
            try:
                create_task = asyncio.create_task(
                    self.client.chat.completions.create(**request_kwargs)
                )
                stream = await self._wait_for_upstream_stage(
                    create_task,
                    attempt_started=attempt_started,
                    warning_seconds=warning_seconds,
                    timeout_seconds=timeout_seconds,
                    trace_request_id=trace_request_id,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    stage="response headers",
                )
                stream_iterator = stream.__aiter__()
                first_chunk_task = asyncio.create_task(anext(stream_iterator))
                try:
                    first_chunk = await self._wait_for_upstream_stage(
                        first_chunk_task,
                        attempt_started=attempt_started,
                        warning_seconds=warning_seconds,
                        timeout_seconds=timeout_seconds,
                        trace_request_id=trace_request_id,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        stage="first response data",
                    )
                except StopAsyncIteration:
                    first_chunk = _NO_STREAM_CHUNK

                elapsed = round(time.perf_counter() - attempt_started, 3)
                attempts.append(
                    {
                        "attempt": attempt,
                        "outcome": (
                            "empty_stream"
                            if first_chunk is _NO_STREAM_CHUNK
                            else "first_data"
                        ),
                        "elapsed_seconds": elapsed,
                    }
                )
                logger.info(
                    "[LLM upstream] response started after {:.1f}s; "
                    "trace_id={}, attempt={}/{}",
                    elapsed,
                    trace_request_id,
                    attempt,
                    max_attempts,
                )
                return stream, stream_iterator, first_chunk, attempts
            except asyncio.CancelledError:
                # asyncio.wait() does not cancel the tasks it observes. Keep
                # ownership here until both the request and prefetched read
                # have stopped, before the caller releases its request slot.
                pending_tasks = [
                    task for task in (create_task, first_chunk_task) if task is not None
                ]
                for task in pending_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*pending_tasks, return_exceptions=True)
                if (
                    stream is None
                    and create_task is not None
                    and not create_task.cancelled()
                    and create_task.exception() is None
                ):
                    # Headers may have completed just before cancellation,
                    # without the await above assigning the returned stream.
                    stream = create_task.result()
                await self._close_stream_after_failed_attempt(stream)
                logger.info(
                    "[LLM upstream] interrupted request cleaned up; "
                    "trace_id={}, attempt={}/{}",
                    trace_request_id,
                    attempt,
                    max_attempts,
                )
                raise
            except asyncio.TimeoutError:
                elapsed = round(time.perf_counter() - attempt_started, 3)
                attempts.append(
                    {
                        "attempt": attempt,
                        "outcome": "no_data_timeout",
                        "elapsed_seconds": elapsed,
                    }
                )
                await self._close_stream_after_failed_attempt(stream)
                if attempt < max_attempts:
                    logger.error(
                        "[LLM upstream] no response data within {:.0f}s; "
                        "cancelling and retrying once. A provider may still bill "
                        "an abandoned request; trace_id={}",
                        timeout_seconds,
                        trace_request_id,
                    )
                    continue
                raise _UpstreamResponseTimeout(
                    f"upstream returned no response data after {max_attempts} attempt(s)",
                    attempts,
                )
            except APIError as exc:
                status_code = getattr(exc, "status_code", None)
                elapsed = round(time.perf_counter() - attempt_started, 3)
                attempts.append(
                    {
                        "attempt": attempt,
                        "outcome": "api_error_before_data",
                        "status_code": status_code,
                        "error_type": type(exc).__name__,
                        "elapsed_seconds": elapsed,
                    }
                )
                await self._close_stream_after_failed_attempt(stream)
                retryable_error = (
                    isinstance(exc, APIConnectionError)
                    or status_code in retryable_statuses
                )
                if attempt < max_attempts and retryable_error:
                    logger.warning(
                        "[LLM upstream] retryable status {} before response data; "
                        "retrying once. trace_id={}",
                        status_code,
                        trace_request_id,
                    )
                    continue
                raise

        raise _UpstreamResponseTimeout(
            "upstream returned no response data",
            attempts,
        )

    async def _iterate_stream_with_idle_timeout(
        self,
        stream_iterator,
        first_chunk,
    ):
        """Yield the prefetched chunk, then bound later stream-idle waits."""

        if first_chunk is not _NO_STREAM_CHUNK:
            yield first_chunk
        idle_timeout = float(
            getattr(self, "upstream_stream_idle_timeout_seconds", 90.0)
        )
        while True:
            try:
                chunk = await asyncio.wait_for(
                    anext(stream_iterator), timeout=idle_timeout
                )
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError as exc:
                raise _UpstreamResponseTimeout(
                    "upstream stream stopped sending data after it began",
                    [],
                ) from exc
            yield chunk

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] | NotGiven = NOT_GIVEN,
    ) -> AsyncIterator[str | List[ChoiceDeltaToolCall]]:
        """
        Generates a chat completion using the OpenAI API asynchronously.

        Parameters:
        - messages (List[Dict[str, Any]]): The list of messages to send to the API.
        - system (str, optional): System prompt to use for this completion.
        - tools (List[Dict[str, str]], optional): List of tools to use for this completion.

        Yields:
        - str: The content of each chunk from the API response.
        - List[ChoiceDeltaToolCall]: The tool calls detected in the response.

        Raises:
        - APIConnectionError: When the server cannot be reached
        - RateLimitError: When a 429 status code is received
        - APIError: For other API-related errors
        """
        stream = None
        # Tool call related state variables
        accumulated_tool_calls = {}
        in_tool_call = False
        received_meaningful_output = False
        self._trace_request_index = getattr(self, "_trace_request_index", 0) + 1
        trace_request_index = self._trace_request_index
        trace_request_id = str(uuid.uuid4())
        trace_started = time.perf_counter()
        trace_text_parts: list[str] = []
        trace_tool_calls: list[dict[str, Any]] = []
        trace_finish_reasons: list[str] = []
        trace_usage: dict[str, Any] | None = None
        trace_protocol_recovery: dict[str, Any] | None = None
        trace_upstream_attempts: list[dict[str, Any]] = []
        suppressed_forced_text_parts: list[str] = []
        trace_completed_emitted = False

        def emit_request_completed() -> None:
            nonlocal trace_completed_emitted
            if trace_completed_emitted:
                return
            self._emit_trace(
                {
                    "event": "request_completed",
                    "trace_request_id": trace_request_id,
                    "request_index": trace_request_index,
                    "elapsed_seconds": round(time.perf_counter() - trace_started, 3),
                    "finish_reasons": trace_finish_reasons,
                    "tool_calls": trace_tool_calls,
                    "text": "".join(trace_text_parts),
                    "usage": trace_usage,
                    "protocol_recovery": trace_protocol_recovery,
                    "upstream_attempts": trace_upstream_attempts,
                }
            )
            trace_completed_emitted = True

        try:
            async with limit_request_concurrency(
                self._limiter_key,
                self.max_concurrent_requests,
                self.min_request_interval_seconds,
            ):
                try:
                    # If system prompt is provided, add it to the messages
                    messages_with_system = messages
                    if system:
                        messages_with_system = [
                            {"role": "system", "content": system},
                            *messages,
                        ]
                    available_tools = tools if self.support_tools else NOT_GIVEN
                    forced_tool_name = getattr(
                        self,
                        "_forced_tool_name_once",
                        None,
                    )
                    self._forced_tool_name_once = None
                    tool_choice: Any = NOT_GIVEN
                    if forced_tool_name and available_tools is not NOT_GIVEN:
                        tool_choice = {
                            "type": "function",
                            "function": {"name": forced_tool_name},
                        }

                    logger.debug(
                        "OpenAI-compatible request message shape: "
                        f"{self._message_shape(messages_with_system)}"
                    )
                    self._emit_trace(
                        {
                            "event": "request_started",
                            "trace_request_id": trace_request_id,
                            "request_index": trace_request_index,
                            "model": self.model,
                            "base_url": self.base_url,
                            "temperature": self.temperature,
                            "messages": messages_with_system,
                            "tools": (
                                available_tools
                                if available_tools is not NOT_GIVEN
                                else None
                            ),
                            "tool_choice": (
                                tool_choice
                                if tool_choice is not NOT_GIVEN
                                else "provider_default_auto"
                            ),
                        }
                    )

                    (
                        stream,
                        stream_iterator,
                        first_chunk,
                        trace_upstream_attempts,
                    ) = await self._open_stream_with_first_data_retry(
                        request_kwargs={
                            "messages": messages_with_system,
                            "model": self.model,
                            "stream": True,
                            "temperature": self.temperature,
                            "tools": available_tools,
                            "tool_choice": tool_choice,
                        },
                        trace_request_id=trace_request_id,
                    )
                    logger.debug(
                        f"Tool Support: {self.support_tools}, Available tools: {available_tools}"
                    )

                    async for chunk in self._iterate_stream_with_idle_timeout(
                        stream_iterator,
                        first_chunk,
                    ):
                        finish_reason = (
                            chunk.choices[0].finish_reason if chunk.choices else None
                        )
                        if finish_reason:
                            trace_finish_reasons.append(str(finish_reason))
                        chunk_usage = getattr(chunk, "usage", None)
                        if chunk_usage is not None:
                            trace_usage = (
                                chunk_usage.model_dump()
                                if hasattr(chunk_usage, "model_dump")
                                else dict(chunk_usage)
                            )
                        # Guard against chunks with missing choices field (e.g., from OpenWebUI)
                        if not chunk.choices:
                            continue

                        if self.support_tools:
                            has_tool_calls = (
                                hasattr(chunk.choices[0].delta, "tool_calls")
                                and chunk.choices[0].delta.tool_calls
                            )

                            if has_tool_calls:
                                logger.debug(
                                    f"Tool calls detected in chunk: {chunk.choices[0].delta.tool_calls}"
                                )
                                in_tool_call = True
                                # Process tool calls in the current chunk
                                for tool_call in chunk.choices[0].delta.tool_calls:
                                    index = (
                                        tool_call.index
                                        if hasattr(tool_call, "index")
                                        else 0
                                    )

                                    # Initialize tool call for this index if needed
                                    if index not in accumulated_tool_calls:
                                        accumulated_tool_calls[index] = {
                                            "index": index,
                                            "id": getattr(tool_call, "id", None),
                                            "type": getattr(tool_call, "type", None),
                                            "function": {
                                                "name": "",
                                                "arguments": "",
                                            },
                                        }

                                    # Update tool call information
                                    if hasattr(tool_call, "id") and tool_call.id:
                                        accumulated_tool_calls[index]["id"] = (
                                            tool_call.id
                                        )
                                    if hasattr(tool_call, "type") and tool_call.type:
                                        accumulated_tool_calls[index]["type"] = (
                                            tool_call.type
                                        )

                                    # Update function information
                                    if hasattr(tool_call, "function"):
                                        if (
                                            hasattr(tool_call.function, "name")
                                            and tool_call.function.name
                                        ):
                                            accumulated_tool_calls[index]["function"][
                                                "name"
                                            ] = tool_call.function.name
                                        if (
                                            hasattr(tool_call.function, "arguments")
                                            and tool_call.function.arguments
                                        ):
                                            accumulated_tool_calls[index]["function"][
                                                "arguments"
                                            ] += tool_call.function.arguments

                                continue

                            # If we were in a tool call but now we're not, yield the tool call result
                            elif in_tool_call and not has_tool_calls:
                                in_tool_call = False
                                # Convert accumulated tool calls to the required format and output
                                logger.info(
                                    f"Complete tool calls: {accumulated_tool_calls}"
                                )

                                # Use the from_dict method to create a ToolCallObject instance from a dictionary
                                complete_tool_calls = [
                                    ToolCallObject.from_dict(tool_data)
                                    for tool_data in accumulated_tool_calls.values()
                                ]
                                trace_tool_calls.extend(
                                    copy.deepcopy(list(accumulated_tool_calls.values()))
                                )

                                received_meaningful_output = True
                                yield complete_tool_calls
                                accumulated_tool_calls = {}  # Reset for potential future tool calls

                        # Process regular content chunks
                        if len(chunk.choices) == 0:
                            logger.info("Empty chunk received")
                            continue
                        content = chunk.choices[0].delta.content or ""
                        if content:
                            received_meaningful_output = True
                            if forced_tool_name:
                                suppressed_forced_text_parts.append(content)
                            else:
                                trace_text_parts.append(content)
                                yield content

                    # If stream ends while still in a tool call, make sure to yield the tool call
                    if in_tool_call and accumulated_tool_calls:
                        logger.info(
                            f"Final tool call at stream end: {accumulated_tool_calls}"
                        )

                        # Create a ToolCallObject instance from a dictionary using the from_dict method.
                        complete_tool_calls = [
                            ToolCallObject.from_dict(tool_data)
                            for tool_data in accumulated_tool_calls.values()
                        ]
                        trace_tool_calls.extend(
                            copy.deepcopy(list(accumulated_tool_calls.values()))
                        )

                        received_meaningful_output = True
                        yield complete_tool_calls

                    forced_tool_satisfied = bool(trace_tool_calls)
                    if forced_tool_name and not forced_tool_satisfied:
                        trace_protocol_recovery = {
                            "attempted": True,
                            "forced_tool_name": forced_tool_name,
                            "reason": "provider_did_not_return_forced_tool_call",
                            "suppressed_stream_text": "".join(
                                suppressed_forced_text_parts
                            ),
                            "outcome": "empty_response",
                        }
                        logger.warning(
                            "Provider returned text instead of the forced tool "
                            f"{forced_tool_name}; retrying once non-streaming."
                        )
                        selection_payload = _forced_tool_selection_payload(
                            messages_with_system
                        )
                        retry_messages = [
                            {
                                "role": "system",
                                "content": (
                                    "You are a strict memory-navigation selector, not a "
                                    "conversation assistant. Return "
                                    "only one JSON object with exactly this shape: "
                                    '{"name":"'
                                    f"{forced_tool_name}"
                                    '","arguments":{"query":"...",'
                                    '"question_granularity":"exact_detail",'
                                    '"seed_event_id":"copy one exact available seed"}}. '
                                    "Copy the seed_event_id exactly from the available "
                                    "navigation options. If the user describes an event "
                                    "that already happened, choose a completed event, not "
                                    "an option that only records a plan or intention. Do "
                                    "not use Markdown fences."
                                ),
                            },
                            {
                                "role": "user",
                                "content": json.dumps(
                                    selection_payload,
                                    ensure_ascii=False,
                                ),
                            },
                        ]
                        trace_protocol_recovery["selection_option_count"] = len(
                            selection_payload["navigation_options"]
                        )
                        forced_response = await self.client.chat.completions.create(
                            messages=retry_messages,
                            model=self.model,
                            stream=False,
                            temperature=self.temperature,
                            tools=NOT_GIVEN,
                            tool_choice=NOT_GIVEN,
                        )
                        forced_usage = getattr(forced_response, "usage", None)
                        if forced_usage is not None:
                            trace_usage = (
                                forced_usage.model_dump()
                                if hasattr(forced_usage, "model_dump")
                                else dict(forced_usage)
                            )
                        if forced_response.choices:
                            forced_choice = forced_response.choices[0]
                            forced_message = forced_choice.message
                            forced_finish_reason = getattr(
                                forced_choice,
                                "finish_reason",
                                None,
                            )
                            if forced_finish_reason:
                                trace_finish_reasons.append(str(forced_finish_reason))
                            forced_tool_calls = forced_message.tool_calls or []
                            recovered_json_call = None
                            if not forced_tool_calls and forced_message.content:
                                trace_protocol_recovery["fallback_text"] = (
                                    forced_message.content
                                )
                                recovered_json_call = _forced_tool_call_from_json(
                                    forced_message.content,
                                    forced_tool_name=forced_tool_name,
                                )
                                if recovered_json_call is not None:
                                    forced_tool_calls = [recovered_json_call]
                            if forced_tool_calls:
                                trace_protocol_recovery["outcome"] = (
                                    "recovered_tool_json"
                                    if recovered_json_call is not None
                                    else "recovered_tool_call"
                                )
                                received_meaningful_output = True
                                complete_tool_calls = [
                                    ToolCallObject.from_dict(
                                        {
                                            "index": index,
                                            "id": tool_call.id,
                                            "type": tool_call.type,
                                            "function": {
                                                "name": tool_call.function.name,
                                                "arguments": tool_call.function.arguments,
                                            },
                                        }
                                    )
                                    for index, tool_call in enumerate(forced_tool_calls)
                                ]
                                trace_tool_calls.extend(
                                    {
                                        "index": index,
                                        "id": tool_call.id,
                                        "type": tool_call.type,
                                        "function": {
                                            "name": tool_call.function.name,
                                            "arguments": tool_call.function.arguments,
                                        },
                                    }
                                    for index, tool_call in enumerate(forced_tool_calls)
                                )
                                yield complete_tool_calls
                            elif forced_message.content:
                                trace_protocol_recovery["outcome"] = "invalid_json"
                                received_meaningful_output = True
                                logger.error(
                                    "Forced-tool JSON recovery returned invalid content."
                                )
                                safe_failure_call = ToolCallObject.from_dict(
                                    {
                                        "index": 0,
                                        "id": (
                                            "protocol-recovery-failed-"
                                            f"{uuid.uuid4().hex}"
                                        ),
                                        "type": "function",
                                        "function": {
                                            "name": forced_tool_name,
                                            "arguments": json.dumps(
                                                {
                                                    "query": (
                                                        "provider failed to select "
                                                        "a reliable memory detail"
                                                    ),
                                                    "question_granularity": (
                                                        "exact_detail"
                                                    ),
                                                    "seed_event_id": (
                                                        "__protocol_selection_failed__"
                                                    ),
                                                }
                                            ),
                                        },
                                    }
                                )
                                yield [safe_failure_call]

                    if not received_meaningful_output:
                        logger.warning(
                            "Streaming chat completion ended without text or tool calls. "
                            "Retrying once with a non-streaming request."
                        )
                        if stream:
                            try:
                                await asyncio.wait_for(stream.close(), timeout=5.0)
                            except asyncio.TimeoutError:
                                logger.error(
                                    "Timed out after 5 seconds while closing an empty provider stream."
                                )
                            stream = None
                        fallback_response = await self.client.chat.completions.create(
                            messages=messages_with_system,
                            model=self.model,
                            stream=False,
                            temperature=self.temperature,
                            tools=available_tools,
                            tool_choice=tool_choice,
                        )
                        fallback_usage = getattr(fallback_response, "usage", None)
                        if fallback_usage is not None:
                            trace_usage = (
                                fallback_usage.model_dump()
                                if hasattr(fallback_usage, "model_dump")
                                else dict(fallback_usage)
                            )
                        if fallback_response.choices:
                            fallback_message = fallback_response.choices[0].message
                            fallback_finish_reason = getattr(
                                fallback_response.choices[0],
                                "finish_reason",
                                None,
                            )
                            if fallback_finish_reason:
                                trace_finish_reasons.append(str(fallback_finish_reason))
                            fallback_tool_calls = fallback_message.tool_calls or []
                            if fallback_tool_calls:
                                complete_tool_calls = [
                                    ToolCallObject.from_dict(
                                        {
                                            "index": index,
                                            "id": tool_call.id,
                                            "type": tool_call.type,
                                            "function": {
                                                "name": tool_call.function.name,
                                                "arguments": tool_call.function.arguments,
                                            },
                                        }
                                    )
                                    for index, tool_call in enumerate(
                                        fallback_tool_calls
                                    )
                                ]
                                trace_tool_calls.extend(
                                    {
                                        "index": index,
                                        "id": tool_call.id,
                                        "type": tool_call.type,
                                        "function": {
                                            "name": tool_call.function.name,
                                            "arguments": tool_call.function.arguments,
                                        },
                                    }
                                    for index, tool_call in enumerate(
                                        fallback_tool_calls
                                    )
                                )
                                yield complete_tool_calls
                            elif fallback_message.content:
                                trace_text_parts.append(fallback_message.content)
                                yield fallback_message.content
                            else:
                                logger.error(
                                    "Non-streaming recovery also returned no text "
                                    "or tool calls."
                                )
                        else:
                            logger.error("Non-streaming recovery returned no choices.")
                    emit_request_completed()
                finally:
                    # The agent intentionally stops consuming the first response
                    # as soon as a complete tool call is yielded. Async-generator
                    # finalization must still persist that request's outcome.
                    if received_meaningful_output:
                        emit_request_completed()
                    # make sure the stream is properly closed
                    # so when interrupted, no more tokens will being generated.
                    if stream:
                        logger.debug("Chat completion finished.")
                        try:
                            await asyncio.wait_for(stream.close(), timeout=5.0)
                        except asyncio.TimeoutError:
                            logger.error(
                                "Timed out after 5 seconds while closing the provider stream."
                            )
                        logger.debug("Stream closed.")

        except _UpstreamResponseTimeout as e:
            if e.attempts:
                trace_upstream_attempts = e.attempts
            self._emit_trace(
                {
                    "event": "request_error",
                    "trace_request_id": trace_request_id,
                    "request_index": trace_request_index,
                    "error_class": "upstream_timeout",
                    "error_type": type(e).__name__,
                    "elapsed_seconds": round(time.perf_counter() - trace_started, 3),
                    "upstream_attempts": trace_upstream_attempts,
                    "received_meaningful_output": received_meaningful_output,
                }
            )
            logger.error(
                "[LLM upstream] {}; trace_id={}, received_output={}",
                e,
                trace_request_id,
                received_meaningful_output,
            )
            if not received_meaningful_output:
                attempt_count = len(trace_upstream_attempts) or int(
                    getattr(self, "upstream_max_attempts", 2)
                )
                timeout_seconds = float(
                    getattr(self, "upstream_first_data_timeout_seconds", 90.0)
                )
                attempt_text = "仅请求1次" if attempt_count == 1 else "连续两次请求"
                yield (
                    f"上游模型在每次{timeout_seconds:g}秒的等待期内都没有开始回复，"
                    f"本轮{attempt_text}并已停止。"
                    "请稍后重试，并根据日志中的 trace_id 排查中转站或本地网络。"
                )

        except APIConnectionError as e:
            self._emit_trace(
                {
                    "event": "request_error",
                    "trace_request_id": trace_request_id,
                    "request_index": trace_request_index,
                    "error_class": "connection",
                    "error_type": type(e).__name__,
                    "elapsed_seconds": round(time.perf_counter() - trace_started, 3),
                }
            )
            logger.error(
                f"Error calling the chat endpoint: Connection error. Failed to connect to the LLM API. \nCheck the configurations and the reachability of the LLM backend. \nSee the logs for details. \nTroubleshooting with documentation: https://open-llm-vtuber.github.io/docs/faq#%E9%81%87%E5%88%B0-error-calling-the-chat-endpoint-%E9%94%99%E8%AF%AF%E6%80%8E%E4%B9%88%E5%8A%9E \n{e.__cause__}"
            )
            yield "Error calling the chat endpoint: Connection error. Failed to connect to the LLM API. Check the configurations and the reachability of the LLM backend. See the logs for details. Troubleshooting with documentation: [https://open-llm-vtuber.github.io/docs/faq#%E9%81%87%E5%88%B0-error-calling-the-chat-endpoint-%E9%94%99%E8%AF%AF%E6%80%8E%E4%B9%88%E5%8A%9E]"

        except RateLimitError as e:
            self._emit_trace(
                {
                    "event": "request_error",
                    "trace_request_id": trace_request_id,
                    "request_index": trace_request_index,
                    "error_class": "rate_limit",
                    "error_type": type(e).__name__,
                    "elapsed_seconds": round(time.perf_counter() - trace_started, 3),
                }
            )
            logger.error(
                f"Error calling the chat endpoint: Rate limit exceeded: {e.response}"
            )
            yield "Error calling the chat endpoint: Rate limit exceeded. Please try again later. See the logs for details."

        except APIError as e:
            self._emit_trace(
                {
                    "event": "request_error",
                    "trace_request_id": trace_request_id,
                    "request_index": trace_request_index,
                    "error_class": "api_error",
                    "error_type": type(e).__name__,
                    "elapsed_seconds": round(time.perf_counter() - trace_started, 3),
                }
            )
            if "does not support tools" in str(e):
                self.support_tools = False
                logger.warning(
                    f"{self.model} does not support tools. Disabling tool support."
                )
                yield "__API_NOT_SUPPORT_TOOLS__"
                return
            logger.error(f"LLM API: Error occurred: {e}")
            logger.info(f"Base URL: {self.base_url}")
            logger.info(f"Model: {self.model}")
            logger.info(f"Message shape: {self._message_shape(messages)}")
            logger.info(f"temperature: {self.temperature}")
            yield "Error calling the chat endpoint: Error occurred while generating response. See the logs for details."
