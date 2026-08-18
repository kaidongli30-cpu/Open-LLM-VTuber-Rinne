####
# change from xTTS.py
####

import asyncio
import re
from pathlib import Path
from typing import Any, Optional

import requests
from loguru import logger
from .tts_interface import TTSInterface


class TTSEngine(TTSInterface):
    def __init__(
        self,
        api_url: str = "http://127.0.0.1:9880/tts",
        text_lang: str = "zh",
        ref_audio_path: str = "",
        prompt_lang: str = "zh",
        prompt_text: str = "",
        text_split_method: str = "cut5",
        batch_size: str = "1",
        media_type: str = "wav",
        streaming_mode: str = "ture",
        emotion_references: Optional[dict[str, dict[str, Any]]] = None,
    ):
        self.api_url = api_url
        self.text_lang = text_lang
        self.ref_audio_path = ref_audio_path
        self.prompt_lang = prompt_lang
        self.prompt_text = prompt_text
        self.text_split_method = text_split_method
        self.batch_size = batch_size
        self.media_type = media_type
        self.streaming_mode = streaming_mode
        self.emotion_references = {
            str(emotion).lower(): reference
            for emotion, reference in (emotion_references or {}).items()
            if isinstance(reference, dict)
        }

    def _default_reference(self) -> dict[str, Any]:
        return {
            "ref_audio_path": self.ref_audio_path,
            "aux_ref_audio_paths": [],
            "prompt_lang": self.prompt_lang,
            "prompt_text": self.prompt_text,
        }

    def _resolve_reference(
        self,
        emotion: Optional[str],
    ) -> tuple[dict[str, Any], Optional[str]]:
        if not emotion:
            return self._default_reference(), None

        normalized_emotion = emotion.lower()
        reference = self.emotion_references.get(normalized_emotion)
        if not reference:
            return self._default_reference(), None

        ref_audio_path = str(reference.get("ref_audio_path", "")).strip()
        prompt_lang = str(reference.get("prompt_lang", "")).strip()
        prompt_text = str(reference.get("prompt_text", "")).strip()
        raw_aux_paths = reference.get("aux_ref_audio_paths", [])
        if isinstance(raw_aux_paths, str):
            aux_ref_audio_paths = [raw_aux_paths.strip()] if raw_aux_paths.strip() else []
        elif isinstance(raw_aux_paths, (list, tuple)):
            aux_ref_audio_paths = [
                str(path).strip() for path in raw_aux_paths if str(path).strip()
            ]
        else:
            logger.warning(
                f"Invalid auxiliary reference list for emotion "
                f"'{normalized_emotion}'; using the default reference."
            )
            return self._default_reference(), None

        if not ref_audio_path or not prompt_lang or not prompt_text:
            logger.warning(
                f"Incomplete GPT-SoVITS reference for emotion '{normalized_emotion}'; "
                "using the default reference."
            )
            return self._default_reference(), None
        if not Path(ref_audio_path).is_file():
            logger.warning(
                f"GPT-SoVITS reference audio not found for emotion "
                f"'{normalized_emotion}': {ref_audio_path}; using the default reference."
            )
            return self._default_reference(), None

        missing_aux_paths = [
            path for path in aux_ref_audio_paths if not Path(path).is_file()
        ]
        if missing_aux_paths:
            logger.warning(
                f"GPT-SoVITS auxiliary reference audio not found for emotion "
                f"'{normalized_emotion}': {missing_aux_paths}; "
                "using the default reference."
            )
            return self._default_reference(), None

        return {
            "ref_audio_path": ref_audio_path,
            "aux_ref_audio_paths": aux_ref_audio_paths,
            "prompt_lang": prompt_lang,
            "prompt_text": prompt_text,
        }, normalized_emotion

    def _request_audio(
        self,
        cleaned_text: str,
        reference: dict[str, Any],
    ) -> requests.Response:
        data = {
            "text": cleaned_text,
            "text_lang": self.text_lang,
            **reference,
            "text_split_method": self.text_split_method,
            "batch_size": self.batch_size,
            "media_type": self.media_type,
            "streaming_mode": (
                self.streaming_mode.strip().lower() == "true"
                if isinstance(self.streaming_mode, str)
                else bool(self.streaming_mode)
            ),
        }
        return requests.post(self.api_url, json=data, timeout=300)

    @staticmethod
    def _log_reference_selection(
        selected_emotion: Optional[str],
        reference: dict[str, Any],
    ) -> None:
        aux_paths = reference.get("aux_ref_audio_paths", [])
        aux_display = ", ".join(aux_paths) if aux_paths else "(none)"
        logger.debug(
            "GPT-SoVITS reference selected: "
            f"emotion={selected_emotion or 'default'}, "
            f"main={reference['ref_audio_path']}, "
            f"aux={aux_display}"
        )

    @staticmethod
    def _save_audio(response: requests.Response, file_name: str) -> str:
        with open(file_name, "wb") as audio_file:
            audio_file.write(response.content)
        return file_name

    def generate_audio(self, text, file_name_no_ext=None, emotion=None):
        file_name = self.generate_cache_file_name(file_name_no_ext, self.media_type)
        cleaned_text = re.sub(r"\[.*?\]", "", text)
        reference, selected_emotion = self._resolve_reference(emotion)
        self._log_reference_selection(selected_emotion, reference)

        try:
            response = self._request_audio(cleaned_text, reference)
            if response.status_code == 200:
                return self._save_audio(response, file_name)
        except requests.RequestException as exc:
            if not selected_emotion:
                raise
            logger.warning(
                f"GPT-SoVITS request failed with the '{selected_emotion}' reference: "
                f"{exc}; retrying once with the default reference."
            )
        else:
            if not selected_emotion:
                logger.critical(
                    "Error: Failed to generate audio. "
                    f"Status code: {response.status_code}"
                )
                return None
            logger.warning(
                f"GPT-SoVITS returned HTTP {response.status_code} with the "
                f"'{selected_emotion}' reference; retrying once with the default reference."
            )

        fallback_reference = self._default_reference()
        self._log_reference_selection("default-fallback", fallback_reference)
        fallback_response = self._request_audio(cleaned_text, fallback_reference)
        if fallback_response.status_code == 200:
            return self._save_audio(fallback_response, file_name)

        logger.critical(
            "Error: Failed to generate audio with both special and default "
            f"references. Fallback status code: {fallback_response.status_code}"
        )
        return None

    async def async_generate_audio_with_emotion(
        self,
        text: str,
        emotion: str,
        file_name_no_ext: Optional[str] = None,
    ) -> str:
        return await asyncio.to_thread(
            self.generate_audio,
            text,
            file_name_no_ext,
            emotion,
        )
