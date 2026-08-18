import asyncio
import re
from typing import Optional, Union, Any, List, Dict
import numpy as np
import json
from loguru import logger

from ..message_handler import message_handler
from .types import WebSocketSend, BroadcastContext
from .tts_manager import DEFAULT_TTS_REFERENCE, TTSTaskManager
from ..agent.output_types import SentenceOutput, AudioOutput, DisplayText
from ..agent.input_types import BatchInput, TextData, ImageData, TextSource, ImageSource
from ..asr.asr_interface import ASRInterface
from ..live2d_model import Live2dModel
from ..tts.tts_interface import TTSInterface
from ..utils.stream_audio import prepare_audio_payload


SPECIAL_TTS_REFERENCE_EMOTIONS = ("surprise", "shy", "angry")


def _select_tts_reference_emotion(
    actions: Optional[Any],
    live2d_model: Live2dModel,
) -> Optional[str]:
    """Return the first configured special emotion represented by Live2D actions."""
    if not actions or not actions.expressions:
        return None

    for expression in actions.expressions:
        for emotion in SPECIAL_TTS_REFERENCE_EMOTIONS:
            mapped_expression = live2d_model.emo_map.get(emotion)
            if mapped_expression is not None and str(mapped_expression) == str(
                expression
            ):
                return emotion

    # A recognized non-special expression explicitly restores the default reference.
    return DEFAULT_TTS_REFERENCE


# Convert class methods to standalone functions
def create_batch_input(
    input_text: str,
    images: Optional[List[Dict[str, Any]]],
    from_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> BatchInput:
    """Create batch input for agent processing"""
    return BatchInput(
        texts=[
            TextData(source=TextSource.INPUT, content=input_text, from_name=from_name)
        ],
        images=[
            ImageData(
                source=ImageSource(img["source"]),
                data=img["data"],
                mime_type=img["mime_type"],
            )
            for img in (images or [])
        ]
        if images
        else None,
        metadata=metadata,
    )


def _split_text_for_realtime_tts(text: str, max_chunk_chars: int = 48) -> List[str]:
    """
    Split long text into smaller chunks for better translation/TTS responsiveness.
    Prefer punctuation boundaries; fallback to fixed-size chunks.
    """
    if not text:
        return []
    if len(text) <= max_chunk_chars:
        return [text]

    separators = set("。！？!?；;，,\n")
    chunks: List[str] = []
    buf: List[str] = []

    for ch in text:
        buf.append(ch)
        should_split = ch in separators or len(buf) >= max_chunk_chars
        if should_split:
            chunk = "".join(buf).strip()
            if chunk:
                chunks.append(chunk)
            buf = []

    if buf:
        chunk = "".join(buf).strip()
        if chunk:
            chunks.append(chunk)

    return chunks if chunks else [text]


async def _translate_text_if_needed(
    text: str,
    translate_engine: Optional[Any],
) -> str:
    """Translate non-empty text in a worker thread when translation is enabled."""
    if not translate_engine:
        return text

    if not len(re.sub(r"[\s.,!?，。！？、\"“”'\(\)\[\]]+", "", text)):
        return text

    try:
        translated = await asyncio.to_thread(translate_engine.translate, text)
        if not translated or not translated.strip():
            raise ValueError("translation returned empty output")
        return translated
    except Exception as e:
        logger.warning(
            "Translation failed; keeping display text but suppressing TTS for this chunk: "
            f"{e}"
        )
        return ""


async def process_agent_output(
    output: Union[AudioOutput, SentenceOutput],
    character_config: Any,
    live2d_model: Live2dModel,
    tts_engine: TTSInterface,
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
    translate_engine: Optional[Any] = None,
) -> str:
    """Process agent output with character information and optional translation"""
    output.display_text.name = character_config.character_name
    output.display_text.avatar = character_config.avatar

    full_response = ""
    try:
        if isinstance(output, SentenceOutput):
            full_response = await handle_sentence_output(
                output,
                live2d_model,
                tts_engine,
                websocket_send,
                tts_manager,
                translate_engine,
            )
        elif isinstance(output, AudioOutput):
            full_response = await handle_audio_output(output, websocket_send)
        else:
            logger.warning(f"Unknown output type: {type(output)}")
    except Exception as e:
        logger.error(f"Error processing agent output: {e}")
        await websocket_send(
            json.dumps(
                {"type": "error", "message": f"Error processing response: {str(e)}"}
            )
        )

    return full_response


async def handle_sentence_output(
    output: SentenceOutput,
    live2d_model: Live2dModel,
    tts_engine: TTSInterface,
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
    translate_engine: Optional[Any] = None,
) -> str:
    """Handle sentence output type with optional translation support"""
    full_response = ""
    async for display_text, tts_text, actions in output:
        logger.debug(f"Processing output: '''{tts_text}'''...")

        full_response += display_text.text
        reference_emotion = _select_tts_reference_emotion(actions, live2d_model)
        resolved_reference_emotion = tts_manager.resolve_reference_emotion(
            reference_emotion
        )

        # Translate and synthesize in smaller chunks so speech starts earlier.
        display_chunks = _split_text_for_realtime_tts(display_text.text)
        tts_chunks = _split_text_for_realtime_tts(tts_text) if tts_text else [""]
        chunk_count = max(len(display_chunks), len(tts_chunks))

        for idx in range(chunk_count):
            display_chunk = display_chunks[idx] if idx < len(display_chunks) else ""
            tts_chunk = tts_chunks[idx] if idx < len(tts_chunks) else ""

            translated_tts_chunk = await _translate_text_if_needed(
                tts_chunk, translate_engine
            )

            if translate_engine and tts_chunk:
                logger.info(
                    f"Chunk translated: '''{tts_chunk}''' -> '''{translated_tts_chunk}'''"
                )

            if not display_chunk and not translated_tts_chunk:
                continue

            await tts_manager.speak(
                tts_text=translated_tts_chunk,
                display_text=DisplayText(
                    text=display_chunk,
                    name=display_text.name,
                    avatar=display_text.avatar,
                ),
                actions=actions if idx == 0 else None,
                live2d_model=live2d_model,
                tts_engine=tts_engine,
                websocket_send=websocket_send,
                reference_emotion=resolved_reference_emotion,
            )
    return full_response


async def handle_audio_output(
    output: AudioOutput,
    websocket_send: WebSocketSend,
) -> str:
    """Process and send AudioOutput directly to the client"""
    full_response = ""
    async for audio_path, display_text, transcript, actions in output:
        full_response += transcript
        audio_payload = prepare_audio_payload(
            audio_path=audio_path,
            display_text=display_text,
            actions=actions.to_dict() if actions else None,
        )
        await websocket_send(json.dumps(audio_payload))
    return full_response


async def send_conversation_start_signals(websocket_send: WebSocketSend) -> None:
    """Send initial conversation signals"""
    await websocket_send(
        json.dumps(
            {
                "type": "control",
                "text": "conversation-chain-start",
            }
        )
    )
    await websocket_send(json.dumps({"type": "full-text", "text": "Thinking..."}))


async def process_user_input(
    user_input: Union[str, np.ndarray],
    asr_engine: ASRInterface,
    websocket_send: WebSocketSend,
) -> str:
    """Process user input, converting audio to text if needed"""
    if isinstance(user_input, np.ndarray):
        logger.info("Transcribing audio input...")
        input_text = await asr_engine.async_transcribe_np(user_input)
        await websocket_send(
            json.dumps({"type": "user-input-transcription", "text": input_text})
        )
        return input_text
    return user_input


async def finalize_conversation_turn(
    tts_manager: TTSTaskManager,
    websocket_send: WebSocketSend,
    client_uid: str,
    broadcast_ctx: Optional[BroadcastContext] = None,
) -> None:
    """Finalize a conversation turn"""
    if tts_manager.task_list:
        await asyncio.gather(*tts_manager.task_list)
        await tts_manager.wait_until_payloads_sent()
        await websocket_send(json.dumps({"type": "backend-synth-complete"}))

        response = await message_handler.wait_for_response(
            client_uid, "frontend-playback-complete"
        )

        if not response:
            logger.warning(f"No playback completion response from {client_uid}")
            return

    await websocket_send(json.dumps({"type": "force-new-message"}))

    if broadcast_ctx and broadcast_ctx.broadcast_func:
        await broadcast_ctx.broadcast_func(
            broadcast_ctx.group_members,
            {"type": "force-new-message"},
            broadcast_ctx.current_client_uid,
        )

    await send_conversation_end_signal(websocket_send, broadcast_ctx)


async def send_conversation_end_signal(
    websocket_send: WebSocketSend,
    broadcast_ctx: Optional[BroadcastContext],
    session_emoji: str = "😊",
) -> None:
    """Send conversation chain end signal"""
    chain_end_msg = {
        "type": "control",
        "text": "conversation-chain-end",
    }

    await websocket_send(json.dumps(chain_end_msg))

    if broadcast_ctx and broadcast_ctx.broadcast_func and broadcast_ctx.group_members:
        await broadcast_ctx.broadcast_func(
            broadcast_ctx.group_members,
            chain_end_msg,
        )

    logger.info(f"😎👍✅ Conversation Chain {session_emoji} completed!")


def cleanup_conversation(tts_manager: TTSTaskManager, session_emoji: str) -> None:
    """Clean up conversation resources"""
    tts_manager.clear()
    logger.debug(f"🧹 Clearing up conversation {session_emoji}.")


EMOJI_LIST = [
    "🐶",
    "🐱",
    "🐭",
    "🐹",
    "🐰",
    "🦊",
    "🐻",
    "🐼",
    "🐨",
    "🐯",
    "🦁",
    "🐮",
    "🐷",
    "🐸",
    "🐵",
    "🐔",
    "🐧",
    "🐦",
    "🐤",
    "🐣",
    "🐥",
    "🦆",
    "🦅",
    "🦉",
    "🦇",
    "🐺",
    "🐗",
    "🐴",
    "🦄",
    "🐝",
    "🌵",
    "🎄",
    "🌲",
    "🌳",
    "🌴",
    "🌱",
    "🌿",
    "☘️",
    "🍀",
    "🍂",
    "🍁",
    "🍄",
    "🌾",
    "💐",
    "🌹",
    "🌸",
    "🌛",
    "🌍",
    "⭐️",
    "🔥",
    "🌈",
    "🌩",
    "⛄️",
    "🎃",
    "🎄",
    "🎉",
    "🎏",
    "🎗",
    "🀄️",
    "🎭",
    "🎨",
    "🧵",
    "🪡",
    "🧶",
    "🥽",
    "🥼",
    "🦺",
    "👔",
    "👕",
    "👜",
    "👑",
]
