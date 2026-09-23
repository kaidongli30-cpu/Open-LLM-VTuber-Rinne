"""
This module contains the pydantic model for the configurations of
different types of agents.
"""

from pydantic import BaseModel, Field
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
        30.0, ge=1.0, le=300.0, alias="max_retrieval_seconds"
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

    faster_first_response: Optional[bool] = Field(True, alias="faster_first_response")
    segment_method: Literal["regex", "pysbd"] = Field("pysbd", alias="segment_method")
    use_mcpp: Optional[bool] = Field(False, alias="use_mcpp")
    mcp_enabled_servers: Optional[List[str]] = Field([], alias="mcp_enabled_servers")
    scene_memory_enabled: bool = Field(False, alias="scene_memory_enabled")
    bocha_api_key: str = Field("", alias="bocha_api_key", repr=False)
    long_term_memory_retrieval: LongTermMemoryRetrievalConfig = Field(
        default_factory=LongTermMemoryRetrievalConfig,
        alias="long_term_memory_retrieval",
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "llm_provider": Description(
            en="LLM provider to use for this agent",
            zh="Basic Memory Agent 智能体使用的大语言模型选项",
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
        "bocha_api_key": Description(
            en="Private persistent Bocha Web Search API key; BOCHA_API_KEY overrides it",
            zh="私有持久化博查搜索密钥；BOCHA_API_KEY 环境变量可覆盖它",
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
    base_url: str = Field(
        "https://generativelanguage.googleapis.com", alias="base_url"
    )
    model: str = Field("gemini-2.0-flash", alias="model")
    api_key_file: Optional[str] = Field(None, alias="api_key_file")
    timeout_seconds: float = Field(
        180.0, ge=10.0, le=900.0, alias="timeout_seconds"
    )
    max_output_tokens: int = Field(
        4096, ge=512, le=32768, alias="max_output_tokens"
    )
    video_segment_seconds: float = Field(
        45.0, ge=5.0, le=45.0, alias="video_segment_seconds"
    )
    max_concurrent_requests: int = Field(
        1, ge=1, le=8, alias="max_concurrent_requests"
    )

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
