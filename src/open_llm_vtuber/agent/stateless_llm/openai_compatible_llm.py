"""Description: This file contains the implementation of the `AsyncLLM` class.
This class is responsible for handling asynchronous interaction with OpenAI API compatible
endpoints for language generation.
"""

import asyncio
import time
import uuid
from typing import AsyncIterator, List, Dict, Any

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

    def __init__(self, message: str, attempts: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.attempts = attempts or []


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
        upstream_stream_idle_timeout_seconds: float = 90.0,
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
        self.upstream_stream_idle_timeout_seconds = max(
            5.0, float(upstream_stream_idle_timeout_seconds)
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
            "upstream_stream_idle_timeout_seconds="
            f"{self.upstream_stream_idle_timeout_seconds}, "
            f"upstream_max_attempts={self.upstream_max_attempts}, "
            f"proxy_enabled={bool(proxy_url)}"
        )

    async def _wait_for_upstream_stage(
        self,
        task: asyncio.Task,
        *,
        attempt_started: float,
        request_id: str,
        attempt: int,
        stage: str,
    ):
        """Wait for response headers/data and emit one slow-response warning."""

        elapsed = time.perf_counter() - attempt_started
        warning_remaining = max(0.0, self.upstream_warning_seconds - elapsed)
        if warning_remaining:
            done, _pending = await asyncio.wait({task}, timeout=warning_remaining)
            if done:
                return await task
        elif task.done():
            return await task

        elapsed = time.perf_counter() - attempt_started
        logger.warning(
            "[LLM upstream] still waiting for {} after {:.1f}s; "
            "request_id={}, attempt={}/{}",
            stage,
            elapsed,
            request_id,
            attempt,
            self.upstream_max_attempts,
        )

        timeout_remaining = max(
            0.0,
            self.upstream_first_data_timeout_seconds
            - (time.perf_counter() - attempt_started),
        )
        if timeout_remaining:
            done, _pending = await asyncio.wait({task}, timeout=timeout_remaining)
            if done:
                return await task

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise asyncio.TimeoutError

    @staticmethod
    async def _close_stream(stream) -> None:
        if stream is None:
            return
        try:
            await asyncio.wait_for(stream.close(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("[LLM upstream] timed out while closing provider stream")
        except Exception as exc:
            logger.warning("[LLM upstream] failed to close provider stream: {}", exc)

    async def _open_stream_with_first_data_retry(
        self,
        *,
        request_kwargs: dict[str, Any],
        request_id: str,
    ):
        """Retry only while no provider data has been received."""

        retryable_statuses = {408, 429, 500, 502, 503, 504}
        attempts: list[dict[str, Any]] = []

        for attempt in range(1, self.upstream_max_attempts + 1):
            attempt_started = time.perf_counter()
            stream = None
            create_task = None
            first_chunk_task = None
            logger.info(
                "[LLM upstream] request sent; request_id={}, attempt={}/{}",
                request_id,
                attempt,
                self.upstream_max_attempts,
            )
            try:
                create_task = asyncio.create_task(
                    self.client.chat.completions.create(**request_kwargs)
                )
                stream = await self._wait_for_upstream_stage(
                    create_task,
                    attempt_started=attempt_started,
                    request_id=request_id,
                    attempt=attempt,
                    stage="response headers",
                )
                stream_iterator = stream.__aiter__()
                first_chunk_task = asyncio.create_task(anext(stream_iterator))
                try:
                    first_chunk = await self._wait_for_upstream_stage(
                        first_chunk_task,
                        attempt_started=attempt_started,
                        request_id=request_id,
                        attempt=attempt,
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
                    "request_id={}, attempt={}/{}",
                    elapsed,
                    request_id,
                    attempt,
                    self.upstream_max_attempts,
                )
                return stream, stream_iterator, first_chunk
            except asyncio.CancelledError:
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
                    stream = create_task.result()
                await self._close_stream(stream)
                logger.info(
                    "[LLM upstream] interrupted request cleaned up; "
                    "request_id={}, attempt={}/{}",
                    request_id,
                    attempt,
                    self.upstream_max_attempts,
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
                await self._close_stream(stream)
                if attempt < self.upstream_max_attempts:
                    logger.warning(
                        "[LLM upstream] no response data within {:.0f}s; "
                        "cancelling and retrying. request_id={}",
                        self.upstream_first_data_timeout_seconds,
                        request_id,
                    )
                    continue
                raise _UpstreamResponseTimeout(
                    "upstream returned no response data before the deadline",
                    attempts,
                )
            except APIError as exc:
                status_code = getattr(exc, "status_code", None)
                attempts.append(
                    {
                        "attempt": attempt,
                        "outcome": "api_error_before_data",
                        "status_code": status_code,
                        "error_type": type(exc).__name__,
                    }
                )
                await self._close_stream(stream)
                retryable_error = (
                    isinstance(exc, APIConnectionError)
                    or status_code in retryable_statuses
                )
                if attempt < self.upstream_max_attempts and retryable_error:
                    logger.warning(
                        "[LLM upstream] retryable status {} before response data; "
                        "retrying. request_id={}",
                        status_code,
                        request_id,
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
        while True:
            try:
                chunk = await asyncio.wait_for(
                    anext(stream_iterator),
                    timeout=self.upstream_stream_idle_timeout_seconds,
                )
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError as exc:
                raise _UpstreamResponseTimeout(
                    "upstream stream stopped sending data after it began"
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
        request_id = uuid.uuid4().hex[:12]

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
                    logger.debug(
                        "OpenAI-compatible request prepared: message_count={}, "
                        "system_prompt_present={}",
                        len(messages_with_system),
                        bool(system),
                    )

                    available_tools = tools if self.support_tools else NOT_GIVEN

                    (
                        stream,
                        stream_iterator,
                        first_chunk,
                    ) = await self._open_stream_with_first_data_retry(
                        request_kwargs={
                            "messages": messages_with_system,
                            "model": self.model,
                            "stream": True,
                            "temperature": self.temperature,
                            "tools": available_tools,
                        },
                        request_id=request_id,
                    )
                    logger.debug(
                        "Tool support enabled: {}; tools supplied: {}",
                        self.support_tools,
                        available_tools is not NOT_GIVEN,
                    )

                    async for chunk in self._iterate_stream_with_idle_timeout(
                        stream_iterator,
                        first_chunk,
                    ):
                        # Guard against chunks with missing choices field (e.g., from OpenWebUI)
                        if not chunk.choices:
                            continue

                        if self.support_tools:
                            has_tool_calls = (
                                hasattr(chunk.choices[0].delta, "tool_calls")
                                and chunk.choices[0].delta.tool_calls
                            )

                            if has_tool_calls:
                                logger.debug("Tool call data detected in stream chunk")
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
                                    "Completed {} streamed tool call(s)",
                                    len(accumulated_tool_calls),
                                )

                                # Use the from_dict method to create a ToolCallObject instance from a dictionary
                                complete_tool_calls = [
                                    ToolCallObject.from_dict(tool_data)
                                    for tool_data in accumulated_tool_calls.values()
                                ]

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
                        yield content

                    # If stream ends while still in a tool call, make sure to yield the tool call
                    if in_tool_call and accumulated_tool_calls:
                        logger.info(
                            "Completed {} tool call(s) at stream end",
                            len(accumulated_tool_calls),
                        )

                        # Create a ToolCallObject instance from a dictionary using the from_dict method.
                        complete_tool_calls = [
                            ToolCallObject.from_dict(tool_data)
                            for tool_data in accumulated_tool_calls.values()
                        ]

                        received_meaningful_output = True
                        yield complete_tool_calls

                    if not received_meaningful_output:
                        logger.warning(
                            "Streaming chat completion ended without text or tool calls. "
                            "Retrying once with a non-streaming request."
                        )
                        if stream:
                            await self._close_stream(stream)
                            stream = None
                        try:
                            fallback_response = await asyncio.wait_for(
                                self.client.chat.completions.create(
                                    messages=messages_with_system,
                                    model=self.model,
                                    stream=False,
                                    temperature=self.temperature,
                                    tools=available_tools,
                                ),
                                timeout=self.upstream_first_data_timeout_seconds,
                            )
                        except asyncio.TimeoutError as exc:
                            raise _UpstreamResponseTimeout(
                                "non-streaming recovery timed out"
                            ) from exc
                        if fallback_response.choices:
                            fallback_message = fallback_response.choices[0].message
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
                                yield complete_tool_calls
                            elif fallback_message.content:
                                yield fallback_message.content
                            else:
                                logger.error(
                                    "Non-streaming recovery also returned no text "
                                    "or tool calls."
                                )
                        else:
                            logger.error("Non-streaming recovery returned no choices.")
                finally:
                    # make sure the stream is properly closed
                    # so when interrupted, no more tokens will being generated.
                    if stream:
                        logger.debug("Chat completion finished.")
                        await self._close_stream(stream)
                        logger.debug("Stream closed.")

        except _UpstreamResponseTimeout as e:
            logger.error(
                "[LLM upstream] {}; request_id={}, attempts={}, received_output={}",
                e,
                request_id,
                e.attempts,
                received_meaningful_output,
            )
            if not received_meaningful_output:
                yield (
                    "Error calling the chat endpoint: The upstream model did not "
                    "respond before the configured timeout. Please try again later."
                )

        except APIConnectionError as e:
            logger.error(
                f"Error calling the chat endpoint: Connection error. Failed to connect to the LLM API. \nCheck the configurations and the reachability of the LLM backend. \nSee the logs for details. \nTroubleshooting with documentation: https://open-llm-vtuber.github.io/docs/faq#%E9%81%87%E5%88%B0-error-calling-the-chat-endpoint-%E9%94%99%E8%AF%AF%E6%80%8E%E4%B9%88%E5%8A%9E \n{e.__cause__}"
            )
            yield "Error calling the chat endpoint: Connection error. Failed to connect to the LLM API. Check the configurations and the reachability of the LLM backend. See the logs for details. Troubleshooting with documentation: [https://open-llm-vtuber.github.io/docs/faq#%E9%81%87%E5%88%B0-error-calling-the-chat-endpoint-%E9%94%99%E8%AF%AF%E6%80%8E%E4%B9%88%E5%8A%9E]"

        except RateLimitError as e:
            logger.error(
                f"Error calling the chat endpoint: Rate limit exceeded: {e.response}"
            )
            yield "Error calling the chat endpoint: Rate limit exceeded. Please try again later. See the logs for details."

        except APIError as e:
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
            logger.info(f"temperature: {self.temperature}")
            yield "Error calling the chat endpoint: Error occurred while generating response. See the logs for details."
