import aiohttp
import asyncio
from typing import Optional
from loguru import logger

from .translate_interface import TranslateInterface


class DeepSeekTranslate(TranslateInterface):
    """使用 DeepSeek API 进行翻译"""

    def __init__(self, config: dict):
        self.api_url = config.get(
            "api_url", "https://api.deepseek.com/v1/chat/completions"
        )
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "deepseek-v4-pro")
        self.source_lang = config.get("source_lang", "zh")
        self.target_lang = config.get("target_lang", "ja")
        self.prompt = config.get(
            "prompt",
            "你是一个翻译引擎，任务是将输入文本翻译成日语输出。"
            "无论输入是中文、日文还是中日混合，你都必须将全部内容整合后输出纯日语。"
            "如果输入本身已经是日语，直接原样输出即可，不要做任何修改。"
            "严格遵守以下词汇表：园神凛祢→そのがみりんね, 凛祢→りんね, 凛绪→りお, 诱宵美九→いざよいみく, 美九→みく, 约会大作战→デートアライブ, MUA~→ううま,　十香→とうか,　凯东→かいどん, 好啦→はい。"
            "只输出翻译结果，不要输出任何解释或多余的文字。",
        )

    async def _async_translate(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> str:
        """异步发送翻译请求"""
        if not text or not text.strip():
            return ""

        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": text},
        ]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 512,
            "stream": False,
            # Translation is a direct transformation task. DeepSeek V4's default
            # thinking mode can consume the entire token budget before emitting
            # message.content, which previously turned a valid sentence into silence.
            "thinking": {"type": "disabled"},
        }

        for attempt in range(2):
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url, json=payload, headers=headers
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(
                            f"DeepSeek translate error {resp.status}: {error_text}"
                        )

                    data = await resp.json()
                    choice = data["choices"][0]
                    translated_text = (
                        choice.get("message", {}).get("content") or ""
                    ).strip()
                    if translated_text:
                        return translated_text

                    finish_reason = choice.get("finish_reason")
                    completion_tokens = (data.get("usage") or {}).get(
                        "completion_tokens"
                    )
                    if attempt == 0:
                        logger.warning(
                            "DeepSeek translation returned empty content "
                            f"(finish_reason={finish_reason}, "
                            f"completion_tokens={completion_tokens}); retrying once."
                        )
                        continue

                    raise Exception(
                        "DeepSeek translation returned empty content twice "
                        f"(finish_reason={finish_reason}, "
                        f"completion_tokens={completion_tokens})"
                    )

        raise RuntimeError("DeepSeek translation retry loop ended unexpectedly")

    def translate(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> str:
        """同步翻译接口（内部调用异步方法）"""
        return asyncio.run(self._async_translate(text, source_lang, target_lang))

    def translate_sync(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> str:
        return self.translate(text, source_lang, target_lang)