import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
import os
from typing import Optional, List, Dict, Any

import httpx
from loguru import logger

from .translate_interface import TranslateInterface


def sign(key, msg):
    """生成 HMAC-SHA256 签名"""
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


class HunyuanTranslate(TranslateInterface):
    """
    腾讯混元大模型翻译 API 实现（使用 TC3-HMAC-SHA256 签名认证）
    严格遵循官方文档：https://cloud.tencent.cn/document/product/1729/113395
    """

    def __init__(
        self,
        secret_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        token: str = "",
        region: str = "ap-guangzhou",
        source_lang: str = "zh",
        target_lang: str = "ja",
        model: str = "hunyuan-translation",
        stream: bool = False,
    ):
        """
        初始化混元翻译客户端

        Args:
            secret_id: 腾讯云 Secret ID，不传则从环境变量 HUNYUAN_SECRET_ID 读取
            secret_key: 腾讯云 Secret Key，不传则从环境变量 HUNYUAN_SECRET_KEY 读取
            token: 临时 token（如果使用临时密钥）
            region: 地域，目前仅支持 ap-guangzhou
            source_lang: 源语言代码
            target_lang: 目标语言代码
            model: 模型名称，可选 hunyuan-translation / hunyuan-translation-lite
            stream: 是否使用流式输出
        """
        self.secret_id = secret_id or os.environ.get("HUNYUAN_SECRET_ID")
        self.secret_key = secret_key or os.environ.get("HUNYUAN_SECRET_KEY")
        if not self.secret_id or not self.secret_key:
            raise ValueError("请设置 HUNYUAN_SECRET_ID 和 HUNYUAN_SECRET_KEY 环境变量或传入参数")

        self.token = token
        self.region = region
        self.service = "hunyuan"
        self.host = "hunyuan.tencentcloudapi.com"
        self.version = "2023-09-01"
        self.action = "ChatTranslations"
        self.algorithm = "TC3-HMAC-SHA256"
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.model = model
        self.stream = stream

        logger.info(f"初始化 HunyuanTranslate，secret_id: {self.secret_id[:4]}..., region: {region}")

    def _create_signature(self, date, service):
        """创建签名密钥"""
        secret_date = sign(("TC3" + self.secret_key).encode("utf-8"), date)
        secret_service = sign(secret_date, service)
        secret_signing = sign(secret_service, "tc3_request")
        return secret_signing

    def _prepare_headers(self, payload: str, timestamp: int, date: str) -> dict:
        """准备请求头（严格遵循官方文档：只签名 content-type 和 host）"""
        ct = "application/json; charset=utf-8"
        canonical_uri = "/"
        canonical_querystring = ""

        # 只包含参与签名的头部：content-type 和 host
        canonical_headers = f"content-type:{ct}\nhost:{self.host}\n"
        signed_headers = "content-type;host"

        hashed_request_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        canonical_request = "\n".join(
            [
                "POST",
                canonical_uri,
                canonical_querystring,
                canonical_headers,
                signed_headers,
                hashed_request_payload,
            ]
        )

        credential_scope = f"{date}/{self.service}/tc3_request"
        hashed_canonical_request = hashlib.sha256(
            canonical_request.encode("utf-8")
        ).hexdigest()
        string_to_sign = f"{self.algorithm}\n{timestamp}\n{credential_scope}\n{hashed_canonical_request}"

        secret_signing = self._create_signature(date, self.service)
        signature = hmac.new(
            secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        authorization = f"{self.algorithm} Credential={self.secret_id}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"

        # 构建完整头部（所有公共参数放在 Header 中）
        headers = {
            "Authorization": authorization,
            "Content-Type": ct,
            "Host": self.host,
            "X-TC-Action": self.action,
            "X-TC-Timestamp": str(timestamp),
            "X-TC-Version": self.version,
        }
        if self.region:
            headers["X-TC-Region"] = self.region
        if self.token:
            headers["X-TC-Token"] = self.token

        return headers

    def translate(
        self,
        text: str,
        field: Optional[str] = None,
        glossary_ids: Optional[List[str]] = None,
        references: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> str:
        """
        翻译文本

        Args:
            text: 待翻译的文本
            field: 可选，文本所属领域（如 "游戏剧情"），提高特定领域翻译质量
            glossary_ids: 可选，术语库 ID 列表
            references: 可选，参考示例列表，格式 [{"Text": "...", "Translation": "..."}]
            **kwargs: 其他可选参数（如 temperature 等）

        Returns:
            翻译后的文本
        """
        timestamp = int(time.time())
        date = datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d")

        # 构建请求参数（混元翻译格式）
        params = {
            "Model": self.model,
            "Stream": self.stream,
            "Text": text,
            "Source": self.source_lang,
            "Target": self.target_lang,
        }
        if field:
            params["Field"] = field
        if glossary_ids:
            params["GlossaryIDs"] = glossary_ids
        if references:
            params["References"] = references
        params.update(kwargs)

        payload = json.dumps(params, ensure_ascii=False)
        headers = self._prepare_headers(payload, timestamp, date)

        try:
            response = httpx.post(
                url=f"https://{self.host}",
                headers=headers,
                data=payload,
                timeout=30,
            )
            res = response.json()
            logger.debug(f"混元翻译响应: {res}")

            if "Response" not in res:
                logger.error(f"响应缺少 Response 字段: {res}")
                raise RuntimeError("翻译失败：响应格式异常")

            resp_data = res["Response"]

            if "Error" in resp_data:
                error_msg = resp_data["Error"].get("Message", "未知错误")
                error_code = resp_data["Error"].get("Code", "")
                logger.error(f"API 错误 [{error_code}]: {error_msg}")

                if error_code == "FailedOperation.FreeResourcePackExhausted":
                    raise RuntimeError("翻译失败：免费资源包已用完，请购买资源包或开通后付费")
                elif error_code == "FailedOperation.ServiceNotActivated":
                    raise RuntimeError("翻译失败：服务未开通，请前往控制台开通")
                else:
                    raise RuntimeError(f"翻译失败: {error_msg}")

            # 提取翻译结果
            if "Choices" in resp_data and len(resp_data["Choices"]) > 0:
                choice = resp_data["Choices"][0]
                if "Message" in choice and "Content" in choice["Message"]:
                    translated_text = choice["Message"]["Content"]

                    if "Usage" in resp_data:
                        usage = resp_data["Usage"]
                        logger.info(
                            f"Token 使用: 输入 {usage.get('PromptTokens', 0)}，"
                            f"输出 {usage.get('CompletionTokens', 0)}，"
                            f"总计 {usage.get('TotalTokens', 0)}"
                        )

                    logger.info(
                        f"翻译成功: {text[:30]}... -> {translated_text[:30]}..."
                    )
                    return translated_text

            logger.error(f"无法从响应中提取翻译结果: {resp_data}")
            raise RuntimeError("翻译失败：无法解析结果")

        except httpx.TimeoutException:
            logger.error("请求超时")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP 错误 {e.response.status_code}: {e.response.text}")
            if e.response.status_code == 401:
                raise RuntimeError("翻译失败：SecretId/SecretKey 无效或未授权")
            elif e.response.status_code == 429:
                raise RuntimeError("翻译失败：请求频率超限")
            else:
                raise
        except Exception as e:
            logger.critical(f"API 调用异常: {e}")
            raise e