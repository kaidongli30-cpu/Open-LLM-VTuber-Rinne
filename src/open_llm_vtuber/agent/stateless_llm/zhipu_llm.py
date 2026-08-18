"""
智谱 GLM 多模态 LLM 适配器
- 历史消息中的图像内容会被剥离，只保留文字
- 只有最新一条用户消息保留图像，并转换为智谱要求的格式
"""

from typing import List, Dict, Any, AsyncIterator
from openai import NOT_GIVEN
from loguru import logger

from .openai_compatible_llm import AsyncLLM as OpenAICompatibleAsyncLLM


class AsyncLLM(OpenAICompatibleAsyncLLM):

    def __init__(
        self,
        llm_api_key: str,
        model: str = "glm-4.6v",
        temperature: float = 1.0,
        max_concurrent_requests: int = 1,
        **kwargs,
    ):
        super().__init__(
            model=model,
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            llm_api_key=llm_api_key,
            temperature=temperature,
            max_concurrent_requests=max_concurrent_requests,
        )
        logger.info(f"ZhipuLLM 初始化，模型: {model}")

    def _convert_image_url_to_zhipu_format(self, content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将 OpenAI 格式的 image_url 转换为智谱要求的 image 格式。
        同时提取 base64 数据（去掉 data:image/xxx;base64, 前缀）。
        """
        new_content = []
        for item in content_list:
            if item.get("type") == "image_url":
                url = item.get("image_url", {}).get("url", "")
                if not url:
                    continue
                # 如果是 base64 格式，去掉前缀
                if url.startswith("data:image"):
                    base64_data = url.split(",", 1)[-1]
                    new_content.append({
                        "type": "image",
                        "image": base64_data
                    })
                else:
                    # 如果是 http(s) 链接，也尝试用 image 字段（智谱支持 url）
                    new_content.append({
                        "type": "image",
                        "image": url
                    })
            else:
                new_content.append(item)
        return new_content

    def _strip_images_from_history(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        处理历史消息：
        - 非最新带图像的用户消息：剥离图像，只保留文字。
        - 最新一条带图像的用户消息：将图像转换为智谱格式。
        """
        if not messages:
            return messages

        # ========== 新增：快速检查是否有任何图片 ==========
        has_image = False
        for msg in messages:
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        has_image = True
                        break
            if has_image:
                break
        # 如果没有图片，直接原样返回，不进行任何转换
        if not has_image:
            return messages
        # ========== 快速检查结束 ==========

        # 找到最后一条 role==user 且含图像的消息的索引
        last_user_image_idx = -1
        for i, msg in enumerate(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                if any(
                    isinstance(c, dict) and c.get("type") == "image_url"
                    for c in msg["content"]
                ):
                    last_user_image_idx = i

        converted = []
        for i, msg in enumerate(messages):
            # 非列表 content 的消息直接保留
            if not isinstance(msg.get("content"), list):
                converted.append(msg)
                continue

            # 最后一条含图像的用户消息：转换图像格式
            if i == last_user_image_idx:
                new_content = self._convert_image_url_to_zhipu_format(msg["content"])
                converted.append({**msg, "content": new_content})
                logger.debug(f"消息[{i}] 已转换图像格式为智谱要求")
                continue

            # 其余消息：剥离图像，只保留文字
            text_parts = [
                c for c in msg["content"]
                if isinstance(c, dict) and c.get("type") == "text"
            ]

            if not text_parts:
                # 如果这条消息只有图像没有文字，用占位符代替
                converted.append({
                    **msg,
                    "content": "[图像已省略]"
                })
            elif len(text_parts) == len(msg["content"]):
                # 没有图像，原样保留
                converted.append(msg)
            else:
                # 有图像也有文字，只保留文字部分
                converted.append({**msg, "content": text_parts})
                logger.debug(f"消息[{i}] 已剥离图像，保留文字")

        return converted

    async def chat_completion(
            self,
            messages: List[Dict[str, Any]],
            system: str = None,
            tools=NOT_GIVEN,
    ) -> AsyncIterator[str]:
        converted = self._strip_images_from_history(messages)
        logger.debug(f"ZhipuLLM: sending converted messages (first 2): {converted[:2]}")
        async for chunk in super().chat_completion(converted, system, tools):
            logger.debug(
                f"ZhipuLLM: received chunk type: {type(chunk)}, content: {chunk if isinstance(chunk, str) else 'non-str'}")
            yield chunk