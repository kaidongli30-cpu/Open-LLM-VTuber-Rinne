from __future__ import annotations

import json
import re
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
    "这们说还给吗东两发么进问见听边过让从对为车门书气觉经样总开长爱欢"
    "应该岁层记忆检结课赶复习现够办实转换话语"
)

# A Han-character run containing a Simplified-Chinese signal is usually the
# smallest useful unit to show the repair pass. For example, passing
# "先生那边的事情" works better than only saying that "边" was rejected.
CJK_RUN_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")

REPAIR_SYSTEM_PROMPT = (
    "你是日语译文修复器。输入会包含中文原文、待修正的日语译文，以及程序检测到的"
    "简体中文残留。把包含残留的整个中文片段改写成自然日语，保留其他已经正确的"
    "日语、原意、指代和语气。只输出修正后的一行日语，不解释，不复述原文。"
)

GLOSSARY_TOKEN_PATTERN = re.compile(r"RINNEGLS\d{4}TOKEN")

# This is the user's name, not the Japanese adjective 静か. Keep its hiragana
# spelling unchanged so the Japanese TTS receives the intended pronunciation.
BUILT_IN_PRESERVED_TERMS = {"しずか": "しずか"}


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
            loaded_glossary = dict(configured_glossary)
        else:
            loaded_glossary = self._read_json_file(config.get("glossary_path"))
        self.glossary = {**loaded_glossary, **BUILT_IN_PRESERVED_TERMS}

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

    @staticmethod
    def _protected_terms_instruction(protected_terms: dict[str, str]) -> str:
        if not protected_terms:
            return ""
        tokens = "，".join(protected_terms)
        return (
            "输入中的以下字符串是程序保护的日语专名或作品名占位符："
            f"{tokens}。翻译句子结构时，必须让每个占位符在译文中恰好出现一次，"
            "并保持字母、数字和顺序完全不变；不得翻译、删除、拆分或重复占位符。"
            "占位符按日语名词处理，最终读音将由程序写回。"
        )

    def _build_system_prompt(
        self,
        protected_terms: dict[str, str],
        stricter_retry: bool = False,
    ) -> str:
        prompt = self.system_prompt
        prompt += self._protected_terms_instruction(protected_terms)
        if stricter_retry:
            prompt += (
                "上一次结果未通过输出检查。请重新翻译，确保非空、不要复述中文原文，"
                "不残留任何简体中文，并严格保留全部专名占位符。"
            )
        return prompt

    def _protect_glossary_terms(self, text: str) -> tuple[str, dict[str, str]]:
        entries = [
            (str(source), str(target))
            for source, target in self.glossary.items()
            if str(source) and str(target) and str(source) in text
        ]
        if not entries:
            return text, {}

        entries.sort(key=lambda item: (-len(item[0]), item[0]))
        targets = dict(entries)
        pattern = re.compile("|".join(re.escape(source) for source, _ in entries))
        protected_terms: dict[str, str] = {}

        def replace(match: re.Match[str]) -> str:
            token_index = len(protected_terms)
            token = f"RINNEGLS{token_index:04d}TOKEN"
            while token in text or token in protected_terms:
                token_index += 1
                token = f"RINNEGLS{token_index:04d}TOKEN"
            protected_terms[token] = targets[match.group(0)]
            return token

        return pattern.sub(replace, text), protected_terms

    @staticmethod
    def _restore_glossary_terms(
        translated: str,
        protected_terms: dict[str, str],
    ) -> str:
        restored = translated
        for token, target in protected_terms.items():
            restored = re.sub(
                rf"[ \t]*{re.escape(token)}[ \t]*",
                lambda _match: target,
                restored,
            )
        return restored

    def _request_messages(self, messages: list[dict[str, str]]) -> str:
        response = requests.post(
            self.api_url,
            json={
                "model": self.model,
                "messages": messages,
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

    def _request_translation(
        self,
        text: str,
        protected_terms: dict[str, str],
        stricter_retry: bool,
    ) -> str:
        return self._request_messages(
            [
                {
                    "role": "system",
                    "content": self._build_system_prompt(
                        protected_terms, stricter_retry
                    ),
                },
                {"role": "user", "content": text},
            ]
        )

    @staticmethod
    def _simplified_chinese_residual(text: str) -> str:
        return "".join(sorted(set(text) & SIMPLIFIED_CHINESE_SIGNALS))

    @classmethod
    def _residual_han_spans(cls, text: str) -> list[str]:
        residual_chars = set(cls._simplified_chinese_residual(text))
        if not residual_chars:
            return []
        return [
            span
            for span in CJK_RUN_PATTERN.findall(text)
            if residual_chars.intersection(span)
        ]

    def _request_repair(
        self,
        source: str,
        translated: str,
        protected_terms: dict[str, str],
    ) -> str:
        residual = self._simplified_chinese_residual(translated)
        residual_spans = self._residual_han_spans(translated)
        prompt = REPAIR_SYSTEM_PROMPT

        prompt += self._protected_terms_instruction(protected_terms)

        repair_details = [
            f"中文原文：{source}",
            f"待修正译文：{translated}",
        ]
        if residual_spans:
            repair_details.append(
                "必须整段改写的中文残留片段：" + "，".join(residual_spans)
            )
        if residual:
            repair_details.append("最终译文不得残留这些简体字符：" + residual)
        if "\n" in translated or "\r" in translated:
            repair_details.append("待修正译文包含多行；最终只能输出一行译文。")

        return self._request_messages(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "\n".join(repair_details)},
            ]
        )

    @classmethod
    def _validation_error(
        cls,
        source: str,
        translated: str,
        protected_terms: Optional[dict[str, str]] = None,
    ) -> Optional[str]:
        if not translated:
            return "empty output"
        if "\n" in translated or "\r" in translated:
            return "multi-line output"
        expected_token_order = list(protected_terms or {})
        expected_tokens = set(expected_token_order)
        observed_tokens = GLOSSARY_TOKEN_PATTERN.findall(translated)
        observed_token_set = set(observed_tokens)
        if observed_token_set != expected_tokens:
            missing = sorted(expected_tokens - observed_token_set)
            unexpected = sorted(observed_token_set - expected_tokens)
            return f"protected glossary token mismatch: missing={missing}, unexpected={unexpected}"
        duplicated = sorted(
            token for token in expected_tokens if observed_tokens.count(token) != 1
        )
        if duplicated:
            return f"protected glossary token duplicated: {duplicated}"
        if observed_tokens != expected_token_order:
            return (
                "protected glossary token order changed: "
                f"expected={expected_token_order}, observed={observed_tokens}"
            )
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

        protected_source, protected_terms = self._protect_glossary_terms(text)
        last_error = "unknown validation error"
        translated = ""
        for attempt in range(self.max_validation_attempts):
            if (
                attempt == 0
                or not translated
                or last_error.startswith("protected glossary token")
            ):
                translated = self._request_translation(
                    protected_source,
                    protected_terms,
                    stricter_retry=attempt > 0,
                )
            else:
                translated = self._request_repair(
                    protected_source,
                    translated,
                    protected_terms,
                )
            last_error = (
                self._validation_error(
                    protected_source,
                    translated,
                    protected_terms,
                )
                or ""
            )
            if not last_error:
                restored = self._restore_glossary_terms(
                    translated,
                    protected_terms,
                )
                return self._sanitize_for_tts(restored)
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
