# config_manager/stateless_llm_models.py
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from .i18n import Description, I18nMixin


class StatelessLLMBaseConfig(I18nMixin):
    """Base configuration shared by stateless LLM backends."""

    interrupt_method: Literal["system", "user"] = Field(
        "user", alias="interrupt_method"
    )
    max_concurrent_requests: int = Field(
        1, alias="max_concurrent_requests", ge=1
    )
    min_request_interval_seconds: float = Field(
        0.0, alias="min_request_interval_seconds", ge=0.0
    )

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "interrupt_method": Description(
            en=(
                "How interruption signals are injected into the conversation. "
                "Use 'system' if the provider supports inserting system messages "
                "mid-conversation, otherwise use 'user'."
            ),
            zh=(
                "How interruption signals are injected into the conversation. "
                "Use 'system' if the provider supports inserting system messages "
                "mid-conversation, otherwise use 'user'."
            ),
        ),
        "max_concurrent_requests": Description(
            en=(
                "Maximum number of concurrent requests allowed for this LLM backend. "
                "Set to 1 to fully serialize requests."
            ),
            zh=(
                "Maximum number of concurrent requests allowed for this LLM backend. "
                "Set to 1 to fully serialize requests."
            ),
        ),
        "min_request_interval_seconds": Description(
            en=(
                "Minimum delay between two requests sent to the same LLM backend. "
                "Useful for providers that enforce QPS-style rate limits."
            ),
            zh=(
                "Minimum delay between two requests sent to the same LLM backend. "
                "Useful for providers that enforce QPS-style rate limits."
            ),
        ),
    }


class StatelessLLMWithTemplate(StatelessLLMBaseConfig):
    """Configuration for template-based stateless LLM backends."""

    base_url: str = Field(..., alias="base_url")
    llm_api_key: str = Field(..., alias="llm_api_key")
    model: str = Field(..., alias="model")
    organization_id: str | None = Field(None, alias="organization_id")
    project_id: str | None = Field(None, alias="project_id")
    template: str | None = Field(None, alias="template")
    temperature: float = Field(1.0, alias="temperature")

    _COMMON_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "base_url": Description(
            en="Base URL for the API endpoint.",
            zh="Base URL for the API endpoint.",
        ),
        "llm_api_key": Description(
            en="API key used to authenticate with the backend.",
            zh="API key used to authenticate with the backend.",
        ),
        "organization_id": Description(
            en="Optional organization identifier for the backend.",
            zh="Optional organization identifier for the backend.",
        ),
        "project_id": Description(
            en="Optional project identifier for the backend.",
            zh="Optional project identifier for the backend.",
        ),
        "model": Description(
            en="Model name to use for requests.",
            zh="Model name to use for requests.",
        ),
        "template": Description(
            en="Prompt template name used for non-ChatML models.",
            zh="Prompt template name used for non-ChatML models.",
        ),
        "temperature": Description(
            en="Sampling temperature, typically between 0 and 2.",
            zh="Sampling temperature, typically between 0 and 2.",
        ),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **StatelessLLMBaseConfig.DESCRIPTIONS,
        **_COMMON_DESCRIPTIONS,
    }


class OpenAICompatibleConfig(StatelessLLMBaseConfig):
    """Configuration for OpenAI-compatible chat backends."""

    base_url: str = Field(..., alias="base_url")
    llm_api_key: str = Field(..., alias="llm_api_key")
    model: str = Field(..., alias="model")
    organization_id: str | None = Field(None, alias="organization_id")
    project_id: str | None = Field(None, alias="project_id")
    temperature: float = Field(1.0, alias="temperature")

    _COMMON_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "base_url": Description(
            en="Base URL for the API endpoint.",
            zh="Base URL for the API endpoint.",
        ),
        "llm_api_key": Description(
            en="API key used to authenticate with the backend.",
            zh="API key used to authenticate with the backend.",
        ),
        "organization_id": Description(
            en="Optional organization identifier for the backend.",
            zh="Optional organization identifier for the backend.",
        ),
        "project_id": Description(
            en="Optional project identifier for the backend.",
            zh="Optional project identifier for the backend.",
        ),
        "model": Description(
            en="Model name to use for requests.",
            zh="Model name to use for requests.",
        ),
        "temperature": Description(
            en="Sampling temperature, typically between 0 and 2.",
            zh="Sampling temperature, typically between 0 and 2.",
        ),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **StatelessLLMBaseConfig.DESCRIPTIONS,
        **_COMMON_DESCRIPTIONS,
    }


class OllamaConfig(OpenAICompatibleConfig):
    """Configuration for Ollama."""

    llm_api_key: str = Field("default_api_key", alias="llm_api_key")
    keep_alive: float = Field(-1, alias="keep_alive")
    unload_at_exit: bool = Field(True, alias="unload_at_exit")
    interrupt_method: Literal["system", "user"] = Field(
        "system", alias="interrupt_method"
    )

    _OLLAMA_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "llm_api_key": Description(
            en="API key for Ollama. Defaults to 'default_api_key'.",
            zh="API key for Ollama. Defaults to 'default_api_key'.",
        ),
        "keep_alive": Description(
            en=(
                "Seconds to keep the model loaded after the last request. "
                "Use -1 to keep it loaded indefinitely."
            ),
            zh=(
                "Seconds to keep the model loaded after the last request. "
                "Use -1 to keep it loaded indefinitely."
            ),
        ),
        "unload_at_exit": Description(
            en="Whether to unload the model when the process exits.",
            zh="Whether to unload the model when the process exits.",
        ),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **OpenAICompatibleConfig.DESCRIPTIONS,
        **_OLLAMA_DESCRIPTIONS,
    }


class LmStudioConfig(OpenAICompatibleConfig):
    """Configuration for LM Studio."""

    llm_api_key: str = Field("default_api_key", alias="llm_api_key")
    base_url: str = Field("http://localhost:1234/v1", alias="base_url")
    interrupt_method: Literal["system", "user"] = Field(
        "system", alias="interrupt_method"
    )


class OpenAIConfig(OpenAICompatibleConfig):
    """Configuration for the official OpenAI API."""

    base_url: str = Field("https://api.openai.com/v1", alias="base_url")
    interrupt_method: Literal["system", "user"] = Field(
        "system", alias="interrupt_method"
    )


class GeminiConfig(OpenAICompatibleConfig):
    """Configuration for Gemini's OpenAI-compatible endpoint."""

    base_url: str = Field(
        "https://generativelanguage.googleapis.com/v1beta/openai/", alias="base_url"
    )
    interrupt_method: Literal["system", "user"] = Field(
        "user", alias="interrupt_method"
    )


class MistralConfig(OpenAICompatibleConfig):
    """Configuration for Mistral."""

    base_url: str = Field("https://api.mistral.ai/v1", alias="base_url")
    interrupt_method: Literal["system", "user"] = Field(
        "user", alias="interrupt_method"
    )


class ZhipuConfig(OpenAICompatibleConfig):
    """Configuration for Zhipu."""

    base_url: str = Field("https://open.bigmodel.cn/api/paas/v4/", alias="base_url")


class DeepseekConfig(OpenAICompatibleConfig):
    """Configuration for DeepSeek."""

    base_url: str = Field("https://api.deepseek.com/v1", alias="base_url")


class GroqConfig(OpenAICompatibleConfig):
    """Configuration for Groq."""

    base_url: str = Field("https://api.groq.com/openai/v1", alias="base_url")
    interrupt_method: Literal["system", "user"] = Field(
        "system", alias="interrupt_method"
    )


class ClaudeConfig(StatelessLLMBaseConfig):
    """Configuration for Claude."""

    base_url: str = Field("https://api.anthropic.com", alias="base_url")
    llm_api_key: str = Field(..., alias="llm_api_key")
    model: str = Field(..., alias="model")
    interrupt_method: Literal["system", "user"] = Field(
        "user", alias="interrupt_method"
    )

    _CLAUDE_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "base_url": Description(
            en="Base URL for the Claude API.",
            zh="Base URL for the Claude API.",
        ),
        "llm_api_key": Description(
            en="API key used to authenticate with Claude.",
            zh="API key used to authenticate with Claude.",
        ),
        "model": Description(
            en="Claude model name to use for requests.",
            zh="Claude model name to use for requests.",
        ),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **StatelessLLMBaseConfig.DESCRIPTIONS,
        **_CLAUDE_DESCRIPTIONS,
    }


class LlamaCppConfig(StatelessLLMBaseConfig):
    """Configuration for local llama.cpp."""

    model_path: str = Field(..., alias="model_path")
    interrupt_method: Literal["system", "user"] = Field(
        "system", alias="interrupt_method"
    )

    _LLAMA_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "model_path": Description(
            en="Filesystem path to the GGUF model file.",
            zh="Filesystem path to the GGUF model file.",
        ),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **StatelessLLMBaseConfig.DESCRIPTIONS,
        **_LLAMA_DESCRIPTIONS,
    }


class StatelessLLMConfigs(I18nMixin, BaseModel):
    """Collection of stateless LLM provider configurations."""

    stateless_llm_with_template: StatelessLLMWithTemplate | None = Field(
        None, alias="stateless_llm_with_template"
    )
    openai_compatible_llm: OpenAICompatibleConfig | None = Field(
        None, alias="openai_compatible_llm"
    )
    ollama_llm: OllamaConfig | None = Field(None, alias="ollama_llm")
    lmstudio_llm: LmStudioConfig | None = Field(None, alias="lmstudio_llm")
    openai_llm: OpenAIConfig | None = Field(None, alias="openai_llm")
    gemini_llm: GeminiConfig | None = Field(None, alias="gemini_llm")
    zhipu_llm: ZhipuConfig | None = Field(None, alias="zhipu_llm")
    deepseek_llm: DeepseekConfig | None = Field(None, alias="deepseek_llm")
    groq_llm: GroqConfig | None = Field(None, alias="groq_llm")
    claude_llm: ClaudeConfig | None = Field(None, alias="claude_llm")
    llama_cpp_llm: LlamaCppConfig | None = Field(None, alias="llama_cpp_llm")
    mistral_llm: MistralConfig | None = Field(None, alias="mistral_llm")

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "stateless_llm_with_template": Description(
            en="Configuration for template-based stateless LLM backends.",
            zh="Configuration for template-based stateless LLM backends.",
        ),
        "openai_compatible_llm": Description(
            en="Configuration for OpenAI-compatible LLM providers.",
            zh="Configuration for OpenAI-compatible LLM providers.",
        ),
        "ollama_llm": Description(
            en="Configuration for Ollama.",
            zh="Configuration for Ollama.",
        ),
        "lmstudio_llm": Description(
            en="Configuration for LM Studio.",
            zh="Configuration for LM Studio.",
        ),
        "openai_llm": Description(
            en="Configuration for the official OpenAI API.",
            zh="Configuration for the official OpenAI API.",
        ),
        "gemini_llm": Description(
            en="Configuration for Gemini.",
            zh="Configuration for Gemini.",
        ),
        "mistral_llm": Description(
            en="Configuration for Mistral.",
            zh="Configuration for Mistral.",
        ),
        "zhipu_llm": Description(
            en="Configuration for Zhipu.",
            zh="Configuration for Zhipu.",
        ),
        "deepseek_llm": Description(
            en="Configuration for DeepSeek.",
            zh="Configuration for DeepSeek.",
        ),
        "groq_llm": Description(
            en="Configuration for Groq.",
            zh="Configuration for Groq.",
        ),
        "claude_llm": Description(
            en="Configuration for Claude.",
            zh="Configuration for Claude.",
        ),
        "llama_cpp_llm": Description(
            en="Configuration for local llama.cpp.",
            zh="Configuration for local llama.cpp.",
        ),
    }
