"""Configuration for the daily Layer-2 user-background updater."""

from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from .i18n import Description, I18nMixin


Layer2ProviderConfigName = Literal[
    "deepseek_llm",
    "openai_compatible_llm",
]


class Layer2MemoryGenerationConfig(I18nMixin, BaseModel):
    """Cloud model and runtime policy for the private Layer-2 background.

    Credentials and the base URL are referenced from the existing
    ``agent_config.llm_configs`` pool. This keeps one authoritative API key per
    provider while allowing Layer 2 to select a different model.
    """

    enabled: bool = Field(False, alias="enabled")
    inject_into_conversation: bool = Field(True, alias="inject_into_conversation")
    provider_config_name: Layer2ProviderConfigName = Field(
        "deepseek_llm", alias="provider_config_name"
    )
    model: str = Field("deepseek-v4-pro", alias="model")
    reasoning_effort: Literal["low", "high", "max"] = Field(
        "high", alias="reasoning_effort"
    )
    temperature: float = Field(0.0, ge=0.0, le=2.0, alias="temperature")
    max_output_tokens: int = Field(65536, ge=1, alias="max_output_tokens")
    timeout_seconds: float = Field(180.0, gt=0.0, alias="timeout_seconds")
    retry_attempts: Literal[3] = Field(3, alias="retry_attempts")
    audit_retention_days: int = Field(14, ge=1, alias="audit_retention_days")

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "enabled": Description(
            en="Enable the daily Layer-2 user-background update.",
            zh="是否启用每日第二层用户背景更新。",
        ),
        "inject_into_conversation": Description(
            en="Inject the latest valid background into every conversation turn.",
            zh="是否把最近一份有效个人背景直接注入每轮对话。",
        ),
        "provider_config_name": Description(
            en="Credential and base-URL entry under agent_config.llm_configs.",
            zh="从 agent_config.llm_configs 中读取密钥和地址的配置项名称。",
        ),
        "model": Description(
            en="Model used for ledger update and model-facing projection.",
            zh="用于事实账本更新和模型阅读版投影的模型名称。",
        ),
        "reasoning_effort": Description(
            en="Reasoning effort for compatible providers.",
            zh="兼容接口使用的思考强度。",
        ),
        "temperature": Description(
            en="Sampling temperature; the validated workflow uses zero.",
            zh="采样温度；已验收方案使用零。",
        ),
        "max_output_tokens": Description(
            en="Maximum output tokens for each Layer-2 API call.",
            zh="每次第二层 API 调用允许的最大输出 token。",
        ),
        "timeout_seconds": Description(
            en="Timeout for one provider request.",
            zh="单次接口请求的超时时间。",
        ),
        "retry_attempts": Description(
            en="Fixed safety retry count for the frozen workflow.",
            zh="冻结工作流固定使用的安全重试次数。",
        ),
        "audit_retention_days": Description(
            en="Number of recent daily audit directories to retain.",
            zh="保留最近多少天的每日完整审计目录。",
        ),
    }
