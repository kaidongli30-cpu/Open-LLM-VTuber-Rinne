from .deeplx import DeepLXTranslate
from .tencent import TencentTranslate
from .hunyuan import HunyuanTranslate  # 确保导入混元模块
from .deepseek_translate import DeepSeekTranslate  # 新增deepseek翻译模块
from .ollama_local_translate import OllamaLocalTranslate
from .translate_interface import TranslateInterface


class TranslateFactory:
    @staticmethod
    def get_translator(
        translate_provider: str, translate_provider_config: dict
    ) -> TranslateInterface:
        translate_provider = translate_provider.lower()
        if translate_provider == "deeplx":
            return DeepLXTranslate(
                api_endpoint=translate_provider_config.get("deeplx_api_endpoint"),
                target_lang=translate_provider_config.get("deeplx_target_lang"),
            )
        elif translate_provider == "tencent":
            return TencentTranslate(
                secret_id=translate_provider_config.get("secret_id"),
                secret_key=translate_provider_config.get("secret_key"),
                region=translate_provider_config.get("region"),
                source_lang=translate_provider_config.get("source_lang"),
                target_lang=translate_provider_config.get("target_lang"),
            )
        elif translate_provider == "hunyuan":
            return HunyuanTranslate(
                secret_id=translate_provider_config.get("secret_id"),
                secret_key=translate_provider_config.get("secret_key"),
                model=translate_provider_config.get("model", "hunyuan-translation"),
                source_lang=translate_provider_config.get("source_lang"),
                target_lang=translate_provider_config.get("target_lang"),
                region=translate_provider_config.get("region"),
                stream=translate_provider_config.get("stream", False),
            )
        # 新增deepseek的翻译分支
        elif translate_provider == "deepseek":
            return DeepSeekTranslate(config=translate_provider_config)
        elif translate_provider == "ollama_local":
            return OllamaLocalTranslate(config=translate_provider_config)
        else:
            raise ValueError(f"Unsupported translate provider: {translate_provider}")
