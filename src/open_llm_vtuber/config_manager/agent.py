"""
This module contains the pydantic model for the configurations of
different types of agents.
"""

from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator
from typing import Dict, ClassVar, Optional, Literal, List
from .i18n import I18nMixin, Description
from .stateless_llm_models import StatelessLLMConfigs

# ======== Configurations for different Agents ========


class LongTermMemoryRetrievalConfig(I18nMixin, BaseModel):
    """Configuration for recent context and cloud-requested long-term recall."""

    enabled: bool = Field(False, alias="enabled")
    recent_memory_days: int = Field(3, ge=1, le=31, alias="recent_memory_days")
    top_k: int = Field(10, alias="top_k")
    embedding_model: str = Field("BAAI/bge-base-zh-v1.5", alias="embedding_model")
    reranker_model: str = Field("BAAI/bge-reranker-base", alias="reranker_model")
    model_cache_dir: Optional[str] = Field(None, alias="model_cache_dir")
    embedding_device: Literal["auto", "cpu", "cuda"] = Field(
        "cpu", alias="embedding_device"
    )
    reranker_device: Literal["auto", "cpu", "cuda"] = Field(
        "cpu", alias="reranker_device"
    )
    reranker_batch_size: int = Field(8, alias="reranker_batch_size")
    max_retrieval_seconds: float = Field(
        60.0, ge=1.0, le=300.0, alias="max_retrieval_seconds"
    )
    legacy_archive_fallback_on_error: bool = Field(
        True, alias="legacy_archive_fallback_on_error"
    )
    trial_logging: bool = Field(True, alias="trial_logging")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "enabled": Description(
            en="Expose recent context and an on-demand memory tool to cloud replies",
            zh="向云端提供近期上下文和按需长期记忆工具",
        ),
        "recent_memory_days": Description(
            en="Number of previous complete memory days of reviewed diary compactions",
            zh="仅常驻加载当前记忆日前若干日的已验收日记压缩版（不含今天、周记和月记）",
        ),
        "max_retrieval_seconds": Description(
            en="Maximum total wait for long-term retrieval in one turn",
            zh="单轮长期记忆检索允许等待的总秒数",
        ),
        "legacy_archive_fallback_on_error": Description(
            en="Temporarily inject the legacy archive only when retrieval is unavailable",
            zh="仅在检索不可用时临时注入旧归档",
        ),
        "trial_logging": Description(
            en="Write private per-turn trial diagnostics under the configured chat history root",
            zh="在已配置的聊天记录根目录下写入私有逐轮试用日志",
        ),
    }


class OpenAIOfficialRouteConfig(I18nMixin, BaseModel):
    """Independent OpenAI settings; invalid model/effort falls back at routing."""

    model: str = "gpt-6.1-sol"
    effort: Optional[str] = "high"
    first_text_timeout_seconds: float = Field(40, ge=10, le=180, allow_inf_nan=False)
    credit_usd: float = Field(0, ge=0, allow_inf_nan=False)
    credit_as_of: str = "2026-11-01T00:00:00+00:00"
    credit_expires: str = "2026-12-31"

    @field_validator("credit_as_of")
    @classmethod
    def valid_credit_start(cls, value: str) -> str:
        if "T" not in value and " " not in value:
            raise ValueError("credit_as_of must include the starting time")
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value

    @field_validator("credit_expires")
    @classmethod
    def valid_credit_expiry(cls, value: str) -> str:
        date.fromisoformat(value)
        return value


class OfficialRouteConfig(I18nMixin, BaseModel):
    """Shared official-route controls; credentials are never stored here."""

    enabled: bool = False
    route: Literal["anthropic", "openai", "apinebula"] = "anthropic"
    model: str = "claude-opus-4-6"
    thinking: bool = True
    effort: Optional[str] = None
    first_text_timeout_seconds: float = Field(40, ge=10, le=180, allow_inf_nan=False)
    openai: OpenAIOfficialRouteConfig = Field(default_factory=OpenAIOfficialRouteConfig)
    chat_check_mode: Literal["two_pass", "explicit_only"] = "two_pass"
    first_pass_model: str = "claude-sonnet-4-6"
    second_pass_model: str = "claude-opus-4-6"
    cooldown_minutes: int = Field(10, ge=0, le=1440)
    credit_usd: float = Field(0, ge=0, allow_inf_nan=False)
    credit_as_of: str = "2026-11-01T00:00:00+00:00"
    credit_expires: str = "2026-12-31"
    data_dir: Optional[str] = None

    @field_validator("credit_as_of")
    @classmethod
    def valid_credit_start(cls, value: str) -> str:
        if "T" not in value and " " not in value:
            raise ValueError("credit_as_of must include the starting time")
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value

    @field_validator("credit_expires")
    @classmethod
    def valid_credit_expiry(cls, value: str) -> str:
        date.fromisoformat(value)
        return value

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "enabled": Description(
            en="Enable automatic official routing", zh="启用官方通道自动切换"
        ),
        "model": Description(en="Official dialogue model", zh="官方通道大脑型号"),
        "route": Description(
            en="Selected provider; the latest configuration/menu change wins",
            zh="选择 Anthropic、OpenAI 或 APINebula；配置与菜单最后改的算数",
        ),
        "effort": Description(
            en="Model-specific effort; null preserves the official default",
            zh="模型支持的思考强度；null 使用官方默认值",
        ),
        "first_text_timeout_seconds": Description(
            en="Maximum wait for first text, 10 to 180 seconds",
            zh="首字最长等待秒数，允许 10—180",
        ),
        "openai": Description(
            en="Independent OpenAI model, effort, timeout and credit settings",
            zh="OpenAI 独立的模型、思考强度、超时和额度设置",
        ),
        "thinking": Description(en="Enable adaptive thinking", zh="启用自适应思考"),
        "chat_check_mode": Description(
            en="Chat checks: two_pass, or explicit_only with one Sonnet 4.6 check and no cooldown",
            zh="聊天检查：two_pass 双审；explicit_only 仅单次 Sonnet 4.6 检查，无冷却",
        ),
        "first_pass_model": Description(
            en="Relay first-pass checker: Sonnet 4.6 or Opus 4.6",
            zh="中转站第一遍检查模型，只支持 Sonnet 4.6 或 Opus 4.6",
        ),
        "second_pass_model": Description(
            en="Relay second-pass checker: Sonnet 4.6 or Opus 4.6",
            zh="中转站二审模型，只支持 Sonnet 4.6 或 Opus 4.6",
        ),
        "cooldown_minutes": Description(
            en="Relay cooldown after a blocked turn", zh="非A轮之后走中转站的分钟数"
        ),
        "credit_usd": Description(
            en="Remaining dollars at the starting time", zh="起点时的剩余美元额度"
        ),
        "credit_as_of": Description(
            en="Timestamp of the remaining-credit reading", zh="剩余额度的读取时间"
        ),
        "credit_expires": Description(
            en="Last inclusive credit date", zh="额度到期日（含当日）"
        ),
        "data_dir": Description(
            en="Shared route data directory; empty uses character history",
            zh="共用通道数据目录；留空用角色聊天记录目录",
        ),
    }


class BasicMemoryAgentConfig(I18nMixin, BaseModel):
    """Configuration for the basic memory agent."""

    llm_provider: Literal[
        "stateless_llm_with_template",
        "openai_compatible_llm",
        "claude_llm",
        "llama_cpp_llm",
        "ollama_llm",
        "lmstudio_llm",
        "openai_llm",
        "gemini_llm",
        "zhipu_llm",
        "deepseek_llm",
        "groq_llm",
        "mistral_llm",
    ] = Field(..., alias="llm_provider")

    question_judge_model: Optional[str] = Field(None, alias="question_judge_model")

    faster_first_response: Optional[bool] = Field(True, alias="faster_first_response")
    segment_method: Literal["regex", "pysbd"] = Field("pysbd", alias="segment_method")
    use_mcpp: Optional[bool] = Field(False, alias="use_mcpp")
    mcp_enabled_servers: Optional[List[str]] = Field([], alias="mcp_enabled_servers")
    scene_memory_enabled: bool = Field(False, alias="scene_memory_enabled")
    response_reminder_enabled: bool = Field(False, alias="response_reminder_enabled")
    scene_card_enabled: bool = Field(False, alias="scene_card_enabled")
    scene_card_position: Literal["end", "start"] = Field(
        "end", alias="scene_card_position"
    )
    bocha_api_key: str = Field("", alias="bocha_api_key", repr=False)
    official_route: OfficialRouteConfig = Field(default_factory=OfficialRouteConfig)
    long_term_memory_retrieval: LongTermMemoryRetrievalConfig = Field(
        default_factory=LongTermMemoryRetrievalConfig,
        alias="long_term_memory_retrieval",
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "llm_provider": Description(
            en="LLM provider to use for this agent",
            zh="Basic Memory Agent 智能体使用的大语言模型选项",
        ),
        "question_judge_model": Description(
            en="Independent model for repeated questions, using the same endpoint and credentials; empty keeps the existing detector",
            zh="独立检测同一聊天记录内重复问句的模型，复用接口和密钥；留空保持原有检测",
        ),
        "faster_first_response": Description(
            en="Whether to respond as soon as encountering a comma in the first sentence to reduce latency (default: True)",
            zh="是否在第一句回应时遇上逗号就直接生成音频以减少首句延迟（默认：True）",
        ),
        "segment_method": Description(
            en="Method for segmenting sentences: 'regex' or 'pysbd' (default: 'pysbd')",
            zh="分割句子的方法：'regex' 或 'pysbd'（默认：'pysbd'）",
        ),
        "use_mcpp": Description(
            en="Whether to use MCP (Model Context Protocol) for the agent (default: True)",
            zh="是否使用为智能体启用 MCP (Model Context Protocol) Plus（默认：False）",
        ),
        "mcp_enabled_servers": Description(
            en="List of MCP servers to enable for the agent",
            zh="为智能体启用 MCP 服务器列表",
        ),
        "scene_memory_enabled": Description(
            en="Keep one shared current-scene snapshot across desktop and private QQ",
            zh="在电脑端和私人QQ之间维护一份共享的当前场景快照",
        ),
        "response_reminder_enabled": Description(
            en="Place the response reminder immediately before the current user text",
            zh="在本轮用户文字之前加入贴身提醒（默认关闭）",
        ),
        "scene_card_enabled": Description(
            en="Write and reuse a private assistant-authored scene card",
            zh="让模型撰写并在下一轮读取私有场景卡片（默认关闭）",
        ),
        "scene_card_position": Description(
            en="Write the private scene card at the end or start of the reply",
            zh="场景卡片放在回复结尾 end 或开头 start（默认 end）",
        ),
        "bocha_api_key": Description(
            en="Private persistent Bocha Web Search API key; BOCHA_API_KEY overrides it",
            zh="私有持久化博查搜索密钥；BOCHA_API_KEY 环境变量可覆盖它",
        ),
        "official_route": Description(
            en="Shared automatic official-route controls",
            zh="桌宠和QQ共用的官方通道设置",
        ),
        "long_term_memory_retrieval": Description(
            en="Live long-term-memory retrieval settings",
            zh="实时长期记忆检索设置",
        ),
    }


class Mem0VectorStoreConfig(I18nMixin, BaseModel):
    """Configuration for Mem0 vector store."""

    provider: str = Field(..., alias="provider")
    config: Dict = Field(..., alias="config")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "provider": Description(
            en="Vector store provider (e.g., qdrant)", zh="向量存储提供者（如 qdrant）"
        ),
        "config": Description(
            en="Provider-specific configuration", zh="提供者特定配置"
        ),
    }


class Mem0LLMConfig(I18nMixin, BaseModel):
    """Configuration for Mem0 LLM."""

    provider: str = Field(..., alias="provider")
    config: Dict = Field(..., alias="config")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "provider": Description(en="LLM provider name", zh="语言模型提供者名称"),
        "config": Description(
            en="Provider-specific configuration", zh="提供者特定配置"
        ),
    }


class Mem0EmbedderConfig(I18nMixin, BaseModel):
    """Configuration for Mem0 embedder."""

    provider: str = Field(..., alias="provider")
    config: Dict = Field(..., alias="config")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "provider": Description(en="Embedder provider name", zh="嵌入模型提供者名称"),
        "config": Description(
            en="Provider-specific configuration", zh="提供者特定配置"
        ),
    }


class Mem0Config(I18nMixin, BaseModel):
    """Configuration for Mem0."""

    vector_store: Mem0VectorStoreConfig = Field(..., alias="vector_store")
    llm: Mem0LLMConfig = Field(..., alias="llm")
    embedder: Mem0EmbedderConfig = Field(..., alias="embedder")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "vector_store": Description(en="Vector store configuration", zh="向量存储配置"),
        "llm": Description(en="LLM configuration", zh="语言模型配置"),
        "embedder": Description(en="Embedder configuration", zh="嵌入模型配置"),
    }


# =================================


class HumeAIConfig(I18nMixin, BaseModel):
    """Configuration for the Hume AI agent."""

    api_key: str = Field(..., alias="api_key")
    host: str = Field("api.hume.ai", alias="host")
    config_id: Optional[str] = Field(None, alias="config_id")
    idle_timeout: int = Field(15, alias="idle_timeout")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "api_key": Description(
            en="API key for Hume AI service", zh="Hume AI 服务的 API 密钥"
        ),
        "host": Description(
            en="Host URL for Hume AI service (default: api.hume.ai)",
            zh="Hume AI 服务的主机地址（默认：api.hume.ai）",
        ),
        "config_id": Description(
            en="Configuration ID for EVI settings", zh="EVI 配置 ID"
        ),
        "idle_timeout": Description(
            en="Idle timeout in seconds before disconnecting (default: 15)",
            zh="空闲超时断开连接的秒数（默认：15）",
        ),
    }


# =================================


class LettaConfig(I18nMixin, BaseModel):
    """Configuration for the Letta agent."""

    host: str = Field("localhost", alias="host")
    port: int = Field(8283, alias="port")
    id: str = Field(..., alias="id")
    faster_first_response: Optional[bool] = Field(True, alias="faster_first_response")
    segment_method: Literal["regex", "pysbd"] = Field("pysbd", alias="segment_method")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "host": Description(
            en="Host address for the Letta server", zh="Letta服务器的主机地址"
        ),
        "port": Description(
            en="Port number for the Letta server (default: 8283)",
            zh="Letta服务器的端口号（默认：8283）",
        ),
        "id": Description(
            en="Agent instance ID running on the Letta server",
            zh="指定Letta服务器上运行的Agent实例id",
        ),
    }


class AgentSettings(I18nMixin, BaseModel):
    """Settings for different types of agents."""

    basic_memory_agent: Optional[BasicMemoryAgentConfig] = Field(
        None, alias="basic_memory_agent"
    )
    mem0_agent: Optional[Mem0Config] = Field(None, alias="mem0_agent")
    hume_ai_agent: Optional[HumeAIConfig] = Field(None, alias="hume_ai_agent")
    letta_agent: Optional[LettaConfig] = Field(None, alias="letta_agent")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "basic_memory_agent": Description(
            en="Configuration for basic memory agent", zh="基础记忆代理配置"
        ),
        "mem0_agent": Description(en="Configuration for Mem0 agent", zh="Mem0代理配置"),
        "hume_ai_agent": Description(
            en="Configuration for Hume AI agent", zh="Hume AI 代理配置"
        ),
        "letta_agent": Description(
            en="Configuration for Letta agent", zh="Letta 代理配置"
        ),
    }


class MediaAnalysisConfig(I18nMixin, BaseModel):
    """Gemini-native observer used for video analysis.

    Static images are sent directly to the configured dialogue model.
    """

    enabled: bool = Field(False, alias="enabled")
    provider: Literal["gemini_native"] = Field("gemini_native", alias="provider")
    base_url: str = Field("https://generativelanguage.googleapis.com", alias="base_url")
    model: str = Field("gemini-2.0-flash", alias="model")
    api_key: str = Field("", alias="api_key", repr=False)
    api_key_file: Optional[str] = Field(None, alias="api_key_file")
    timeout_seconds: float = Field(180.0, ge=10.0, le=900.0, alias="timeout_seconds")
    max_output_tokens: int = Field(4096, ge=512, le=32768, alias="max_output_tokens")
    video_segment_seconds: float = Field(
        45.0, ge=5.0, le=45.0, alias="video_segment_seconds"
    )
    max_concurrent_requests: int = Field(1, ge=1, le=8, alias="max_concurrent_requests")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "enabled": Description(
            en="Enable the video observer",
            zh="启用视频观察模块",
        ),
        "provider": Description(
            en="Native media-observer protocol",
            zh="媒体观察模块使用的原生协议",
        ),
        "base_url": Description(
            en="Gemini-native provider base URL",
            zh="Gemini 原生接口提供商的基础地址",
        ),
        "model": Description(
            en="Model used only for video observation",
            zh="仅用于视频观察的模型",
        ),
        "api_key_file": Description(
            en="Path to a private API-key file",
            zh="私密 API 密钥文件路径",
        ),
        "api_key": Description(
            en="API key for the video observer",
            zh="视频观察模块的 API 密钥",
        ),
    }


class AgentConfig(I18nMixin, BaseModel):
    """This class contains all of the configurations related to agent."""

    conversation_agent_choice: Literal[
        "basic_memory_agent", "mem0_agent", "hume_ai_agent", "letta_agent"
    ] = Field(..., alias="conversation_agent_choice")
    agent_settings: AgentSettings = Field(..., alias="agent_settings")
    media_analysis: MediaAnalysisConfig = Field(
        default_factory=MediaAnalysisConfig, alias="media_analysis"
    )
    llm_configs: StatelessLLMConfigs = Field(..., alias="llm_configs")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "conversation_agent_choice": Description(
            en="Type of conversation agent to use", zh="要使用的对话代理类型"
        ),
        "agent_settings": Description(
            en="Settings for different agent types", zh="不同代理类型的设置"
        ),
        "media_analysis": Description(
            en="Video observer configuration",
            zh="视频观察配置",
        ),
        "llm_configs": Description(
            en="Pool of LLM provider configurations", zh="语言模型提供者配置池"
        ),
        "faster_first_response": Description(
            en="Whether to respond as soon as encountering a comma in the first sentence to reduce latency (default: True)",
            zh="是否在第一句回应时遇上逗号就直接生成音频以减少首句延迟（默认：True）",
        ),
        "segment_method": Description(
            en="Method for segmenting sentences: 'regex' or 'pysbd' (default: 'pysbd')",
            zh="分割句子的方法：'regex' 或 'pysbd'（默认：'pysbd'）",
        ),
    }
