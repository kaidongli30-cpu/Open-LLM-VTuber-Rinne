"""Configuration for the independent daily child-event generator."""

from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from .i18n import Description, I18nMixin


DailyChildEventProvider = Literal[
    "ollama_llm",
    "lmstudio_llm",
    "openai_compatible_llm",
    "openai_llm",
    "gemini_llm",
    "zhipu_llm",
    "deepseek_llm",
    "groq_llm",
    "mistral_llm",
    "claude_llm",
]


class DailyChildEventGenerationConfig(I18nMixin, BaseModel):
    """Independent model connection used to turn one diary into child events.

    The provider is deliberately separate from the conversational LLM.  The
    model name is data, not a backend type, so changing an Ollama model does
    not require changing the execution path.
    """

    enabled: bool = Field(True, alias="enabled")
    llm_provider: DailyChildEventProvider = Field(
        "ollama_llm", alias="llm_provider"
    )
    base_url: str = Field("http://localhost:11434/v1", alias="base_url")
    llm_api_key: str = Field("default_api_key", alias="llm_api_key")
    model: str = Field("mistral-small3.2:24b", alias="model")
    organization_id: str | None = Field(None, alias="organization_id")
    project_id: str | None = Field(None, alias="project_id")
    temperature: float = Field(0.1, ge=0.0, le=2.0, alias="temperature")
    max_output_tokens: int = Field(8192, ge=1, alias="max_output_tokens")
    timeout_seconds: float = Field(300.0, gt=0.0, alias="timeout_seconds")
    max_concurrent_requests: int = Field(1, ge=1, alias="max_concurrent_requests")
    min_request_interval_seconds: float = Field(
        0.0, ge=0.0, alias="min_request_interval_seconds"
    )
    num_ctx: int = Field(32768, ge=1, alias="num_ctx")
    top_p: float = Field(0.9, ge=0.0, le=1.0, alias="top_p")
    top_k: int = Field(20, ge=0, alias="top_k")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0, alias="presence_penalty")
    keep_alive: float = Field(600.0, alias="keep_alive")
    unload_at_exit: bool = Field(True, alias="unload_at_exit")

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "enabled": Description(
            en="Enable the independent daily child-event generator.",
            zh="是否启用独立的每日子事件生成器。",
        ),
        "llm_provider": Description(
            en="Provider type used for diary-to-event generation.",
            zh="将日记整理为事件时使用的接口类型。",
        ),
        "base_url": Description(
            en="API endpoint for the selected provider.",
            zh="所选接口的 API 地址。",
        ),
        "llm_api_key": Description(
            en="API key for the selected provider.",
            zh="所选接口的 API 密钥。",
        ),
        "model": Description(
            en="Model name. For Ollama, changing this switches the local model.",
            zh="模型名称；使用 Ollama 时修改此项即可切换本地模型。",
        ),
        "organization_id": Description(
            en="Optional organization identifier.",
            zh="可选的组织 ID。",
        ),
        "project_id": Description(
            en="Optional project identifier.",
            zh="可选的项目 ID。",
        ),
        "temperature": Description(
            en="Sampling temperature used for structured event extraction.",
            zh="整理结构化事件时使用的采样温度。",
        ),
        "max_output_tokens": Description(
            en="Maximum output tokens for one diary.",
            zh="整理一篇日记时允许输出的最大 token 数。",
        ),
        "timeout_seconds": Description(
            en="Maximum time for one provider request.",
            zh="单次接口请求的最长等待时间。",
        ),
        "max_concurrent_requests": Description(
            en="Maximum concurrent requests for this generator.",
            zh="此生成器允许的最大并发请求数。",
        ),
        "min_request_interval_seconds": Description(
            en="Minimum delay between requests.",
            zh="请求之间的最小间隔。",
        ),
        "num_ctx": Description(
            en="Context length for local Ollama-compatible backends.",
            zh="本地 Ollama 兼容接口使用的上下文长度。",
        ),
        "top_p": Description(en="Top-p sampling value.", zh="Top-p 采样值。"),
        "top_k": Description(en="Top-k sampling value.", zh="Top-k 采样值。"),
        "presence_penalty": Description(
            en="Presence penalty for compatible chat APIs.",
            zh="兼容聊天 API 使用的 presence penalty。",
        ),
        "keep_alive": Description(
            en="Seconds to keep an Ollama model loaded after a request.",
            zh="Ollama 模型请求后继续驻留内存的秒数。",
        ),
        "unload_at_exit": Description(
            en="Unload the selected Ollama model when the worker exits.",
            zh="后台任务退出时是否卸载所选 Ollama 模型。",
        ),
    }
