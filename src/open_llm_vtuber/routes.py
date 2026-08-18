import os
import json
from uuid import uuid4
import numpy as np
from datetime import datetime
from fastapi import APIRouter, WebSocket, UploadFile, File, Response
from starlette.responses import JSONResponse, FileResponse
from starlette.websockets import WebSocketDisconnect
from loguru import logger
from .service_context import ServiceContext
from .websocket_handler import WebSocketHandler
from .proxy_handler import ProxyHandler


def _load_live2d_models_from_model_dict() -> list[dict]:
    """Load Live2D model metadata from model_dict.json when available."""
    model_dict_path = "model_dict.json"
    if not os.path.isfile(model_dict_path):
        return []

    try:
        with open(model_dict_path, "r", encoding="utf-8") as f:
            model_entries = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read {model_dict_path}: {e}")
        return []

    if not isinstance(model_entries, list):
        logger.warning(f"Unexpected format in {model_dict_path}: expected a list")
        return []

    supported_extensions = [".png", ".jpg", ".jpeg"]
    valid_characters = []

    for entry in model_entries:
        if not isinstance(entry, dict):
            continue

        name = entry.get("name")
        model_url = entry.get("url")
        if not name or not model_url or not isinstance(model_url, str):
            continue

        local_model_path = model_url.lstrip("/").replace("/", os.sep)
        if not os.path.isfile(local_model_path):
            logger.warning(
                f"Skipping Live2D model '{name}' because file does not exist: {local_model_path}"
            )
            continue

        model_dir = os.path.dirname(local_model_path)
        character_dir = os.path.join("live2d-models", name)

        avatar_file = None
        avatar_candidates = []
        for ext in supported_extensions:
            avatar_candidates.extend(
                [
                    os.path.join(model_dir, f"{name}{ext}"),
                    os.path.join(model_dir, f"icon{ext}"),
                    os.path.join(character_dir, f"{name}{ext}"),
                    os.path.join(character_dir, f"icon{ext}"),
                ]
            )

        for avatar_path in avatar_candidates:
            if os.path.isfile(avatar_path):
                avatar_file = avatar_path.replace("\\", "/")
                break

        valid_characters.append(
            {
                "name": name,
                "avatar": avatar_file,
                "model_path": model_url,
            }
        )

    return valid_characters


def _scan_live2d_models_from_directory() -> list[dict]:
    """Fallback scan for legacy directory layouts when model_dict.json is unavailable."""
    live2d_dir = "live2d-models"
    if not os.path.exists(live2d_dir):
        return []

    supported_extensions = [".png", ".jpg", ".jpeg"]
    valid_characters = []

    for entry in os.scandir(live2d_dir):
        if not entry.is_dir():
            continue

        folder_name = entry.name.replace("\\", "/")
        runtime_dir = os.path.join(live2d_dir, folder_name, "runtime")
        model_candidates = [
            os.path.join(live2d_dir, folder_name, f"{folder_name}.model3.json"),
            os.path.join(runtime_dir, f"{folder_name}.model3.json"),
        ]

        model3_file = next((p for p in model_candidates if os.path.isfile(p)), None)
        if not model3_file and os.path.isdir(runtime_dir):
            runtime_models = [
                os.path.join(runtime_dir, f)
                for f in os.listdir(runtime_dir)
                if f.endswith(".model3.json")
            ]
            if runtime_models:
                model3_file = runtime_models[0]

        if not model3_file:
            continue

        avatar_file = None
        for ext in supported_extensions:
            avatar_candidates = [
                os.path.join(os.path.dirname(model3_file), f"{folder_name}{ext}"),
                os.path.join(os.path.dirname(model3_file), f"icon{ext}"),
                os.path.join(live2d_dir, folder_name, f"{folder_name}{ext}"),
                os.path.join(live2d_dir, folder_name, f"icon{ext}"),
            ]
            avatar_file = next(
                (p.replace("\\", "/") for p in avatar_candidates if os.path.isfile(p)),
                None,
            )
            if avatar_file:
                break

        valid_characters.append(
            {
                "name": folder_name,
                "avatar": avatar_file,
                "model_path": f"/{model3_file.replace(os.sep, '/')}",
            }
        )

    return valid_characters


def init_client_ws_route(
    default_context_cache: ServiceContext,
    ws_handler: WebSocketHandler | None = None,
) -> APIRouter:
    """
    Create and return API routes for handling the `/client-ws` WebSocket connections.

    Args:
        default_context_cache: Default service context cache for new sessions.

    Returns:
        APIRouter: Configured router with WebSocket endpoint.
    """

    router = APIRouter()
    ws_handler = ws_handler or WebSocketHandler(default_context_cache)

    @router.websocket("/client-ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for client connections"""
        await websocket.accept()
        client_uid = str(uuid4())

        try:
            await ws_handler.handle_new_connection(websocket, client_uid)
            await ws_handler.handle_websocket_communication(websocket, client_uid)
        except WebSocketDisconnect:
            await ws_handler.handle_disconnect(client_uid)
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {e}")
            await ws_handler.handle_disconnect(client_uid)
            raise

    return router


def init_proxy_route(server_url: str) -> APIRouter:
    """
    Create and return API routes for handling proxy connections.

    Args:
        server_url: The WebSocket URL of the actual server

    Returns:
        APIRouter: Configured router with proxy WebSocket endpoint
    """
    router = APIRouter()
    proxy_handler = ProxyHandler(server_url)

    @router.websocket("/proxy-ws")
    async def proxy_endpoint(websocket: WebSocket):
        """WebSocket endpoint for proxy connections"""
        try:
            await proxy_handler.handle_client_connection(websocket)
        except Exception as e:
            logger.error(f"Error in proxy connection: {e}")
            raise

    return router


def init_webtool_routes(default_context_cache: ServiceContext) -> APIRouter:
    """
    Create and return API routes for handling web tool interactions.

    Args:
        default_context_cache: Default service context cache for new sessions.

    Returns:
        APIRouter: Configured router with WebSocket endpoint.
    """

    router = APIRouter()

    @router.get("/web-tool")
    async def web_tool_redirect():
        """Redirect /web-tool to /web_tool/index.html"""
        return Response(status_code=302, headers={"Location": "/web-tool/index.html"})

    @router.get("/web_tool")
    async def web_tool_redirect_alt():
        """Redirect /web_tool to /web_tool/index.html"""
        return Response(status_code=302, headers={"Location": "/web-tool/index.html"})

    @router.get("/undefined/{file_path:path}")
    async def mobile_live2d_fallback(file_path: str):
        """
        处理手机 Safari 的 Live2D 路径 bug。
        手机端前端 JS 会将模型路径错误构建为 /undefined/undefined.model3.json，
        此路由将所有 /undefined/* 请求重定向到凛祢模型的真实文件。
        """
        # 文件名中的 undefined.model3.json 需要映射为 rinne.model3.json
        actual_path = file_path.replace("undefined.model3.json", "rinne.model3.json")
        full_path = os.path.join("live2d-models", "rinne", actual_path)

        if os.path.isfile(full_path):
            logger.debug(f"[Mobile Live2D] {file_path} -> {full_path}")
            return FileResponse(full_path)

        logger.warning(f"[Mobile Live2D] 文件未找到: {full_path} (原始请求: {file_path})")
        return Response(status_code=404)

    @router.get("/models/{file_path:path}")
    async def mobile_live2d_models_fallback(file_path: str):
        """
        处理手机端将 /live2d-models/ 错误请求为 /models/ 的情况
        """
        full_path = os.path.join("live2d-models", file_path)
        if os.path.isfile(full_path):
            logger.debug(f"[Mobile Live2D /models] {file_path} -> {full_path}")
            return FileResponse(full_path)
        logger.warning(f"[Mobile Live2D /models] 文件未找到: {full_path}")
        return Response(status_code=404)

    @router.get("/live2d-models/info")
    async def get_live2d_folder_info():
        """Get information about available Live2D models"""
        if not os.path.exists("live2d-models"):
            return JSONResponse(
                {"error": "Live2D models directory not found"}, status_code=404
            )

        valid_characters = _load_live2d_models_from_model_dict()
        if not valid_characters:
            valid_characters = _scan_live2d_models_from_directory()

        return JSONResponse(
            {
                "type": "live2d-models/info",
                "count": len(valid_characters),
                "characters": valid_characters,
            }
        )

    @router.post("/asr")
    async def transcribe_audio(file: UploadFile = File(...)):
        """
        Endpoint for transcribing audio using the ASR engine
        """
        logger.info(f"Received audio file for transcription: {file.filename}")

        try:
            contents = await file.read()

            # Validate minimum file size
            if len(contents) < 44:  # Minimum WAV header size
                raise ValueError("Invalid WAV file: File too small")

            # Decode the WAV header and get actual audio data
            wav_header_size = 44  # Standard WAV header size
            audio_data = contents[wav_header_size:]

            # Validate audio data size
            if len(audio_data) % 2 != 0:
                raise ValueError("Invalid audio data: Buffer size must be even")

            # Convert to 16-bit PCM samples to float32
            try:
                audio_array = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
            except ValueError as e:
                raise ValueError(
                    f"Audio format error: {str(e)}. Please ensure the file is 16-bit PCM WAV format."
                )

            # Validate audio data
            if len(audio_array) == 0:
                raise ValueError("Empty audio data")

            text = await default_context_cache.asr_engine.async_transcribe_np(
                audio_array
            )
            logger.info(f"Transcription result: {text}")
            return {"text": text}

        except ValueError as e:
            logger.error(f"Audio format error: {e}")
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=400,
                media_type="application/json",
            )
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return Response(
                content=json.dumps(
                    {"error": "Internal server error during transcription"}
                ),
                status_code=500,
                media_type="application/json",
            )

    @router.websocket("/tts-ws")
    async def tts_endpoint(websocket: WebSocket):
        """WebSocket endpoint for TTS generation"""
        await websocket.accept()
        logger.info("TTS WebSocket connection established")

        try:
            while True:
                data = await websocket.receive_json()
                text = data.get("text")
                if not text:
                    continue

                logger.info(f"Received text for TTS: {text}")

                # Split text into sentences
                sentences = [s.strip() for s in text.split(".") if s.strip()]

                try:
                    # Generate and send audio for each sentence
                    for sentence in sentences:
                        sentence = sentence + "."  # Add back the period
                        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid4())[:8]}"
                        audio_path = (
                            await default_context_cache.tts_engine.async_generate_audio(
                                text=sentence, file_name_no_ext=file_name
                            )
                        )
                        logger.info(
                            f"Generated audio for sentence: {sentence} at: {audio_path}"
                        )

                        await websocket.send_json(
                            {
                                "status": "partial",
                                "audioPath": audio_path,
                                "text": sentence,
                            }
                        )

                    # Send completion signal
                    await websocket.send_json({"status": "complete"})

                except Exception as e:
                    logger.error(f"Error generating TTS: {e}")
                    await websocket.send_json({"status": "error", "message": str(e)})

        except WebSocketDisconnect:
            logger.info("TTS WebSocket client disconnected")
        except Exception as e:
            logger.error(f"Error in TTS WebSocket connection: {e}")
            await websocket.close()

    return router
