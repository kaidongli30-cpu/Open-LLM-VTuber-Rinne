# config_manager/translate.py
from typing import ClassVar, Dict, Literal, Optional

from pydantic import ValidationInfo, Field, model_validator

from .i18n import I18nMixin, Description

# --- Sub-models for specific Translator providers ---


class DeepLXConfig(I18nMixin):
    """Configuration for DeepLX translation service."""

    deeplx_target_lang: str = Field(..., alias="deeplx_target_lang")
    deeplx_api_endpoint: str = Field(..., alias="deeplx_api_endpoint")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "deeplx_target_lang": Description(
            en="Target language code for DeepLX translation",
            zh="DeepLX 翻译的目标语言代码",
        ),
        "deeplx_api_endpoint": Description(
            en="API endpoint URL for DeepLX service", zh="DeepLX 服务的 API 端点 URL"
        ),
    }


class TencentConfig(I18nMixin):
    """Configuration for tencent translation service."""

    secret_id: str = Field(..., description="Tencent Secret ID")
    secret_key: str = Field(..., description="Tencent Secret Key")
    region: str = Field(..., description="Region for Tencent Service")
    source_lang: str = Field(
        ..., description="Source language code for tencent translation"
    )
    target_lang: str = Field(
        ..., description="Target language code for tencent translation"
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "secret_id": Description(en="Tencent Secret ID", zh="腾讯服务的Secret ID"),
        "secret_key": Description(en="Tencent Secret Key", zh="腾讯服务的Secret Key"),
        "region": Description(en="Region for Tencent Service", zh="腾讯服务使用的区域"),
        "source_lang": Description(
            en="Source language code for tencent translation", zh="腾讯翻译的源语言代码"
        ),
        "target_lang": Description(
            en="Target language code for tencent translation",
            zh="腾讯翻译的目标语言代码",
        ),
    }


class HunyuanConfig(I18nMixin):
    """Configuration for Hunyuan translation service."""

    secret_id: str = Field(..., description="Hunyuan Secret ID")
    secret_key: str = Field(..., description="Hunyuan Secret Key")
    region: str = Field(
        default="ap-guangzhou", description="Region for Hunyuan Service"
    )
    source_lang: str = Field(
        default="zh", description="Source language code for Hunyuan translation"
    )
    target_lang: str = Field(
        default="ja", description="Target language code for Hunyuan translation"
    )
    model: str = Field(
        default="hunyuan-translation",
        description="Model name, e.g., hunyuan-translation or hunyuan-translation-lite",
    )
    stream: bool = Field(
        default=False, description="Whether to use streaming output"
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "api_key": Description(en="API Key for Hunyuan", zh="混元的API Key"),
        "region": Description(en="Region for Hunyuan Service", zh="混元服务使用的区域"),
        "source_lang": Description(
            en="Source language code for Hunyuan translation",
            zh="混元翻译的源语言代码",
        ),
        "target_lang": Description(
            en="Target language code for Hunyuan translation",
            zh="混元翻译的目标语言代码",
        ),
        "model": Description(
            en="Model name, e.g., hunyuan-translation or hunyuan-translation-lite",
            zh="模型名称，例如 hunyuan-translation 或 hunyuan-translation-lite",
        ),
        "stream": Description(
            en="Whether to use streaming output",
            zh="是否使用流式输出",
        ),
    }


class OllamaLocalConfig(I18nMixin):
    """Configuration for the local Ollama translation service."""

    api_url: str = Field(default="http://127.0.0.1:11434/api/chat")
    model: str = Field(default="qwen3.5:4b-q4_K_M")
    system_prompt_path: Optional[str] = Field(default=None)
    glossary_path: str = Field(
        default="models/ollama/qwen3.5-4b-q4_K_M/dynamic-glossary.json"
    )
    timeout_seconds: float = Field(default=20.0, gt=0)
    keep_alive: str = Field(default="10m")
    num_ctx: int = Field(default=2048, gt=0)
    num_predict: int = Field(default=192, gt=0)
    temperature: float = Field(default=0.0, ge=0)
    max_validation_attempts: int = Field(default=2, ge=1, le=3)


# --- Main TranslatorConfig model ---


class TranslatorConfig(I18nMixin):
    """Configuration for translation services."""

    translate_audio: bool = Field(..., alias="translate_audio")
    translate_provider: Literal[
        "deeplx", "tencent", "hunyuan", "deepseek", "ollama_local"
    ] = Field(..., alias="translate_provider")
    deeplx: Optional[DeepLXConfig] = Field(None, alias="deeplx")
    tencent: Optional[TencentConfig] = Field(None, alias="tencent")
    hunyuan: Optional[HunyuanConfig] = Field(None, alias="hunyuan")
    deepseek: Optional[dict] = Field(None, alias="deepseek")
    ollama_local: Optional[OllamaLocalConfig] = Field(None, alias="ollama_local")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "translate_audio": Description(
            en="Enable audio translation (requires DeepLX deployment)",
            zh="启用音频翻译（需要部署 DeepLX）",
        ),
        "translate_provider": Description(
            en="Translation service provider to use", zh="要使用的翻译服务提供者"
        ),
        "deeplx": Description(
            en="Configuration for DeepLX translation service", zh="DeepLX 翻译服务配置"
        ),
        "tencent": Description(
            en="Configuration for TenCent translation service", zh="腾讯 翻译服务配置"
        ),
        "hunyuan": Description(
            en="Configuration for Hunyuan translation service", zh="混元 翻译服务配置"
        ),
    }

    @model_validator(mode="after")
    def check_translator_config(cls, values: "TranslatorConfig", info: ValidationInfo):
        translate_audio = values.translate_audio
        translate_provider = values.translate_provider

        if translate_audio:
            if translate_provider == "deeplx" and values.deeplx is None:
                raise ValueError(
                    "DeepLX configuration must be provided when translate_audio is True and translate_provider is 'deeplx'"
                )
            elif translate_provider == "tencent" and values.tencent is None:
                raise ValueError(
                    "Tencent configuration must be provided when translate_audio is True and translate_provider is 'tencent'"
                )
            elif translate_provider == "hunyuan" and values.hunyuan is None:
                raise ValueError(
                    "Hunyuan configuration must be provided when translate_audio is True and translate_provider is 'hunyuan'"
                )
            elif translate_provider == "deepseek" and values.deepseek is None:
                raise ValueError(
                    "DeepSeek configuration must be provided when translate_audio is True and translate_provider is 'deepseek'"
                )
            elif (
                translate_provider == "ollama_local" and values.ollama_local is None
            ):
                raise ValueError(
                    "Ollama local configuration must be provided when translate_audio is True and translate_provider is 'ollama_local'"
                )

        return values


class TTSPreprocessorConfig(I18nMixin):
    """Configuration for TTS preprocessor."""

    remove_special_char: bool = Field(..., alias="remove_special_char")
    ignore_brackets: bool = Field(default=True, alias="ignore_brackets")
    ignore_parentheses: bool = Field(default=True, alias="ignore_parentheses")
    ignore_asterisks: bool = Field(default=True, alias="ignore_asterisks")
    ignore_angle_brackets: bool = Field(default=True, alias="ignore_angle_brackets")
    translator_config: TranslatorConfig = Field(..., alias="translator_config")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "remove_special_char": Description(
            en="Remove special characters from the input text",
            zh="从输入文本中删除特殊字符",
        ),
        "translator_config": Description(
            en="Configuration for translation services", zh="翻译服务的配置"
        ),
    }
