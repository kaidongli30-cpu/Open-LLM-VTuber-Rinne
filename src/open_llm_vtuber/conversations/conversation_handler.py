import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Optional, Callable

import numpy as np
from fastapi import WebSocket
from loguru import logger

from ..chat_group import ChatGroupManager
from ..chat_history_manager import store_message
from ..file_library import normalize_attachment, save_data_url
from ..service_context import ServiceContext
from .group_conversation import process_group_conversation
from .image_payload_diagnostics import inspect_image_payload
from .proactive_screen_observation import (
    get_proactive_screen_observation_instruction,
)
from .single_conversation import process_single_conversation
from .conversation_utils import EMOJI_LIST
from .types import GroupConversationState
from prompts import prompt_loader


CANCELLED_CONVERSATION_CLEANUP_TIMEOUT_SECONDS = 5.0


async def _wait_for_cancelled_conversation(task: asyncio.Task) -> None:
    """Let a cancelled turn finalize streams and release shared resources."""

    try:
        await asyncio.wait_for(
            asyncio.shield(task),
            timeout=CANCELLED_CONVERSATION_CLEANUP_TIMEOUT_SECONDS,
        )
    except asyncio.CancelledError:
        pass
    except asyncio.TimeoutError:
        logger.error(
            "Timed out waiting for an interrupted conversation to finish cleanup."
        )
    except Exception as exc:
        logger.warning(
            "Interrupted conversation ended with an error during cleanup: "
            f"{type(exc).__name__}: {exc}"
        )


def _conversation_task_is_cancelling(task: asyncio.Task) -> bool:
    if getattr(task, "_rinne_interrupting", False):
        return True
    cancelling = getattr(task, "cancelling", None)
    return bool(cancelling()) if callable(cancelling) else False


def _register_conversation_task(
    task_key: str,
    task: asyncio.Task,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
) -> None:
    current_conversation_tasks[task_key] = task

    def _cleanup_task(completed_task: asyncio.Task) -> None:
        if current_conversation_tasks.get(task_key) is completed_task:
            current_conversation_tasks.pop(task_key, None)

    task.add_done_callback(_cleanup_task)


def _prepare_library_attachments(
    images: list[dict] | None,
    attachments: list[dict] | None,
) -> list[dict]:
    """Persist explicitly marked captures and validate user-upload refs."""

    validated: list[dict] = []
    for attachment in attachments or []:
        record = normalize_attachment(attachment)
        if record is not None:
            validated.append(record)
        else:
            logger.warning("Ignoring an attachment that is not in Rinne's library")

    for image in images or []:
        if not isinstance(image, dict) or not image.get("persist"):
            continue
        source = image.get("source")
        if source not in {"camera", "screen"}:
            continue
        try:
            validated.append(save_data_url(image.get("data", ""), source=source))
        except (TypeError, ValueError, OSError) as exc:
            logger.warning(f"Failed to persist {source} capture: {exc}")
    return validated


async def handle_conversation_trigger(
    msg_type: str,
    data: dict,
    client_uid: str,
    context: ServiceContext,
    websocket: WebSocket,
    client_contexts: Dict[str, ServiceContext],
    client_connections: Dict[str, WebSocket],
    chat_group_manager: ChatGroupManager,
    received_data_buffers: Dict[str, np.ndarray],
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    broadcast_to_group: Callable,
) -> None:
    """Handle triggers that start a conversation"""
    request_received_perf = time.perf_counter()
    request_received_at = datetime.now().astimezone().isoformat()
    metadata = None

    if msg_type == "ai-speak-signal":
        try:
            # Get proactive speak prompt from config
            prompt_name = "proactive_speak_prompt"
            prompt_file = context.system_config.tool_prompts.get(prompt_name)
            if prompt_file:
                user_input = prompt_loader.load_util(prompt_file)
            else:
                logger.warning("Proactive speak prompt not configured, using default")
                user_input = "Please say something."
        except Exception as e:
            logger.error(f"Error loading proactive speak prompt: {e}")
            user_input = "Please say something."

        # Add metadata to indicate this is a proactive speak request
        # that should be skipped in both memory and history
        metadata = {
            "proactive_speak": True,
            "skip_memory": True,  # Skip storing in AI's internal memory
            "skip_history": True,  # Skip storing in local conversation history
        }

        await websocket.send_text(
            json.dumps(
                {
                    "type": "full-text",
                    "text": "AI wants to speak something...",
                }
            )
        )
    elif msg_type == "text-input":
        user_input = data.get("text", "")
    else:  # mic-audio-end
        user_input = received_data_buffers[client_uid]
        received_data_buffers[client_uid] = np.array([])

    metadata = dict(metadata or {})
    metadata["_request_received_perf"] = request_received_perf
    metadata["_request_received_at"] = request_received_at

    images = data.get("images")
    for image in images or []:
        if not isinstance(image, dict) or image.get("source") != "screen":
            continue
        diagnostic = inspect_image_payload(image)
        logger.info(
            "[屏幕图像] 到达后端：proactive={}，declared_mime={}，"
            "actual_mime={}，dimensions={}x{}，decoded_bytes={}，status={}",
            bool(metadata.get("proactive_speak")),
            diagnostic.get("declared_mime"),
            diagnostic.get("actual_mime"),
            diagnostic.get("width"),
            diagnostic.get("height"),
            diagnostic.get("decoded_bytes"),
            diagnostic.get("status"),
        )
    proactive_screen_instruction = get_proactive_screen_observation_instruction(
        images,
        proactive_speak=bool(metadata and metadata.get("proactive_speak")),
    )
    if proactive_screen_instruction:
        metadata["proactive_screen_observation_instruction"] = (
            proactive_screen_instruction
        )
    attachments = _prepare_library_attachments(images, data.get("attachments"))
    if attachments:
        metadata["file_attachments"] = attachments
    session_emoji = np.random.choice(EMOJI_LIST)

    group = chat_group_manager.get_client_group(client_uid)
    if group and len(group.members) > 1:
        # Use group_id as task key for group conversations
        task_key = group.group_id
        if (
            task_key not in current_conversation_tasks
            or current_conversation_tasks[task_key].done()
        ):
            logger.info(f"Starting new group conversation for {task_key}")

            _register_conversation_task(
                task_key,
                asyncio.create_task(
                    process_group_conversation(
                        client_contexts=client_contexts,
                        client_connections=client_connections,
                        broadcast_func=broadcast_to_group,
                        group_members=group.members,
                        initiator_client_uid=client_uid,
                        user_input=user_input,
                        images=images,
                        attachments=attachments,
                        session_emoji=session_emoji,
                        metadata=metadata,
                    )
                ),
                current_conversation_tasks,
            )
        else:
            logger.warning(
                f"Skipping overlapping group conversation trigger for {task_key}."
            )
    else:
        # Use client_uid as task key for individual conversations
        existing_task = current_conversation_tasks.get(client_uid)
        if existing_task and not existing_task.done():
            if _conversation_task_is_cancelling(existing_task):
                logger.info(
                    "Waiting for the interrupted conversation to release resources "
                    f"before handling '{msg_type}' for client {client_uid}."
                )
                await _wait_for_cancelled_conversation(existing_task)
            if not existing_task.done():
                logger.warning(
                    f"Skipping overlapping conversation trigger '{msg_type}' for client {client_uid}."
                )
                return

        _register_conversation_task(
            client_uid,
            asyncio.create_task(
                process_single_conversation(
                    context=context,
                    websocket_send=websocket.send_text,
                    client_uid=client_uid,
                    user_input=user_input,
                    images=images,
                    attachments=attachments,
                    session_emoji=session_emoji,
                    metadata=metadata,
                )
            ),
            current_conversation_tasks,
        )


async def handle_individual_interrupt(
    client_uid: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    context: ServiceContext,
    heard_response: str,
):
    if client_uid in current_conversation_tasks:
        task = current_conversation_tasks[client_uid]
        if task and not task.done():
            setattr(task, "_rinne_interrupting", True)
            task.cancel()
            logger.info("🛑 Conversation task was successfully interrupted")
            await _wait_for_cancelled_conversation(task)

        try:
            context.agent_engine.handle_interrupt(heard_response)
        except Exception as e:
            logger.error(f"Error handling interrupt: {e}")

        if context.history_uid:
            if heard_response:
                store_message(
                    conf_uid=context.character_config.conf_uid,
                    history_uid=context.history_uid,
                    role="ai",
                    content=heard_response,
                    name=context.character_config.character_name,
                    avatar=context.character_config.avatar,
                )
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="system",
                content="[Interrupted by user]",
            )


async def handle_group_interrupt(
    group_id: str,
    heard_response: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    chat_group_manager: ChatGroupManager,
    client_contexts: Dict[str, ServiceContext],
    broadcast_to_group: Callable,
) -> None:
    """Handles interruption for a group conversation"""
    task = current_conversation_tasks.get(group_id)
    if not task or task.done():
        return

    # Get state and speaker info before cancellation
    state = GroupConversationState.get_state(group_id)
    current_speaker_uid = state.current_speaker_uid if state else None

    # Get context from current speaker
    context = None
    group = chat_group_manager.get_group_by_id(group_id)
    if current_speaker_uid:
        context = client_contexts.get(current_speaker_uid)
        logger.info(f"Found current speaker context for {current_speaker_uid}")
    if not context and group and group.members:
        logger.warning(f"No context found for group {group_id}, using first member")
        context = client_contexts.get(next(iter(group.members)))

    # Now cancel the task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        logger.info(f"🛑 Group conversation {group_id} cancelled successfully.")

    current_conversation_tasks.pop(group_id, None)
    GroupConversationState.remove_state(group_id)  # Clean up state after we've used it

    # Store messages with speaker info
    if context and group:
        for member_uid in group.members:
            if member_uid in client_contexts:
                try:
                    member_ctx = client_contexts[member_uid]
                    member_ctx.agent_engine.handle_interrupt(heard_response)
                    store_message(
                        conf_uid=member_ctx.character_config.conf_uid,
                        history_uid=member_ctx.history_uid,
                        role="ai",
                        content=heard_response,
                        name=context.character_config.character_name,
                        avatar=context.character_config.avatar,
                    )
                    store_message(
                        conf_uid=member_ctx.character_config.conf_uid,
                        history_uid=member_ctx.history_uid,
                        role="system",
                        content="[Interrupted by user]",
                    )
                except Exception as e:
                    logger.error(f"Error handling interrupt for {member_uid}: {e}")

    await broadcast_to_group(
        list(group.members),
        {
            "type": "interrupt-signal",
            "text": "conversation-interrupted",
        },
    )
