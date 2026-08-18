from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import requests
from loguru import logger

from .translate_interface import TranslateInterface


DEFAULT_SYSTEM_PROMPT = (
    "你是中译日专用翻译引擎。把输入完整翻译成自然、口语化且符合女性语气的日语。"
    "只输出本条输入对应的日语译文，不解释，不复述原文，不添加输入中不存在的信息。"
    "忠实保持原句的信息、指代和语气，不增强、不削弱。"
    "禁止残留不属于自然日语的简体中文词。"
)

# These characters are strong Simplified-Chinese signals in Rinne's Chinese input.
# Shared Han characters are deliberately excluded because they are also valid Japanese.
SIMPLIFIED_CHINESE_SIGNALS = frozenset(
    "这们说还没给吗东两发么进问见听边过让从对为与车门书气觉经样总开长爱欢"
    "应该岁层记忆检结课赶复习现够办实转换话语"
)


class OllamaLocalTranslate(TranslateInterface):
    """Translate Chinese TTS chunks locally through Ollama."""

    def __init__(self, config: dict[str, Any]):
        self.api_url = config.get("api_url", "http://127.0.0.1:11434/api/chat")
        self.model = config.get("model", "qwen3.5:4b-q4_K_M")
        self.timeout_seconds = float(config.get("timeout_seconds", 20.0))
        self.keep_alive = config.get("keep_alive", "10m")
        self.num_ctx = int(config.get("num_ctx", 2048))
        self.num_predict = int(config.get("num_predict", 192))
        self.temperature = float(config.get("temperature", 0.0))
        self.max_validation_attempts = int(config.get("max_validation_attempts", 2))

        self.system_prompt = config.get("system_prompt") or self._read_text_file(
            config.get("system_prompt_path")
        )
        if not self.system_prompt:
            self.system_prompt = DEFAULT_SYSTEM_PROMPT

        configured_glossary = config.get("glossary")
        if configured_glossary is not None:
            self.glossary = dict(configured_glossary)
        else:
            self.glossary = self._read_json_file(config.get("glossary_path"))

        if self.timeout_seconds <= 0:
            raise ValueError("Ollama translation timeout_seconds must be positive")
        if self.max_validation_attempts < 1:
            raise ValueError("Ollama max_validation_attempts must be at least 1")

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[3]

    @classmethod
    def _resolve_path(cls, raw_path: Optional[str]) -> Optional[Path]:
        if not raw_path:
            return None
        path = Path(raw_path)
        if not path.is_absolute():
            path = cls._project_root() / path
        return path

    @classmethod
    def _read_text_file(cls, raw_path: Optional[str]) -> str:
        path = cls._resolve_path(raw_path)
        if path is None:
            return ""
        return path.read_text(encoding="utf-8").strip()

    @classmethod
    def _read_json_file(cls, raw_path: Optional[str]) -> dict[str, str]:
        path = cls._resolve_path(raw_path)
        if path is None:
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Ollama glossary must be a JSON object: {path}")
        return {str(source): str(target) for source, target in data.items()}

    def _build_system_prompt(self, text: str, stricter_retry: bool = False) -> str:
        active_terms = [
            f"{source}→{target}"
            for source, target in self.glossary.items()
            if source in text
        ]
        prompt = self.system_prompt
        if active_terms:
            prompt += (
                "本条只使用以下实际出现的词汇表：" + "，".join(active_terms) + "。"
            )
        if stricter_retry:
            prompt += (
                "上一次结果未通过输出检查。请重新翻译，确保非空、不要复述中文原文，"
                "并且不残留任何简体中文。"
            )
        return prompt

    def _request_translation(self, text: str, stricter_retry: bool) -> str:
        response = requests.post(
            self.api_url,
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": self._build_system_prompt(text, stricter_retry),
                    },
                    {"role": "user", "content": text},
                ],
                "stream": False,
                "think": False,
                "keep_alive": self.keep_alive,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.num_predict,
                    "num_ctx": self.num_ctx,
                },
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return str((data.get("message") or {}).get("content") or "").strip()

    @staticmethod
    def _simplified_chinese_residual(text: str) -> str:
        return "".join(sorted(set(text) & SIMPLIFIED_CHINESE_SIGNALS))

    @classmethod
    def _validation_error(cls, source: str, translated: str) -> Optional[str]:
        if not translated:
            return "empty output"
        if "\n" in translated or "\r" in translated:
            return "multi-line output"
        residual = cls._simplified_chinese_residual(translated)
        if residual:
            return f"Simplified-Chinese residual: {residual}"
        if translated == source and cls._simplified_chinese_residual(source):
            return "output is unchanged Simplified-Chinese source"
        return None

    @staticmethod
    def _sanitize_for_tts(text: str) -> str:
        # GPT-SoVITS' current Windows preprocessing fails on the Japanese middle dot.
        return text.replace("・", "")

    def translate(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> str:
        del source_lang, target_lang
        if not text or not text.strip():
            return ""

        last_error = "unknown validation error"
        for attempt in range(self.max_validation_attempts):
            translated = self._request_translation(text, stricter_retry=attempt > 0)
            last_error = self._validation_error(text, translated) or ""
            if not last_error:
                return self._sanitize_for_tts(translated)
            logger.warning(
                "Local Ollama translation rejected "
                f"(attempt={attempt + 1}/{self.max_validation_attempts}): {last_error}"
            )

        raise ValueError(
            "Local Ollama translation failed output validation after "
            f"{self.max_validation_attempts} attempts: {last_error}"
        )

    def translate_sync(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> str:
        return self.translate(text, source_lang, target_lang)
