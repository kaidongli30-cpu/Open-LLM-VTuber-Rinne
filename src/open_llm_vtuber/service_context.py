import os
import json
from typing import Callable
from loguru import logger
from fastapi import WebSocket

from prompts import prompt_loader
from .live2d_model import Live2dModel
from .rinne_renderer_profile import active_rinne_emotion_map
from .asr.asr_interface import ASRInterface
from .tts.tts_interface import TTSInterface
from .vad.vad_interface import VADInterface
from .agent.agents.agent_interface import AgentInterface
from .translate.translate_interface import TranslateInterface
from .mcpp.server_registry import ServerRegistry
from .mcpp.tool_manager import ToolManager
from .mcpp.mcp_client import MCPClient
from .mcpp.tool_executor import ToolExecutor
from .video_analysis import settings_from_character_config
from .mcpp.tool_adapter import ToolAdapter
from .agent_runtime.read_only_config import (
    ReadOnlyComputerConfigurationError,
    effective_read_only_mcp_settings,
)
from .data_paths import character_history_root

from .asr.asr_factory import ASRFactory
from .tts.tts_factory import TTSFactory
from .vad.vad_factory import VADFactory
from .agent.agent_factory import AgentFactory
from .translate.translate_factory import TranslateFactory

from .config_manager import (
    Config,
    AgentConfig,
    CharacterConfig,
    SystemConfig,
    ASRConfig,
    TTSConfig,
    VADConfig,
    TranslatorConfig,
    read_yaml,
    validate_config,
)


def _safe_character_config_log_fields(character_config) -> dict[str, str]:
    """Return a minimal diagnostic summary without provider credentials."""

    agent_config = getattr(character_config, "agent_config", None)
    return {
        "conf_name": str(getattr(character_config, "conf_name", "")),
        "conf_uid": str(getattr(character_config, "conf_uid", "")),
        "live2d_model_name": str(
            getattr(character_config, "live2d_model_name", "")
        ),
        "conversation_agent_choice": str(
            getattr(agent_config, "conversation_agent_choice", "")
        ),
    }


def _headless_private_bridge_enabled() -> bool:
    """Return whether this process is serving only the local private bridge."""

    return os.environ.get("RINNE_HEADLESS_PRIVATE_BRIDGE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class ServiceContext:
    """Initializes, stores, and updates the asr, tts, and llm instances and other
    configurations for a connected client."""

    def __init__(self):
        self.config: Config = None
        self.system_config: SystemConfig = None
        self.character_config: CharacterConfig = None

        self.live2d_model: Live2dModel = None
        self.asr_engine: ASRInterface = None
        self.tts_engine: TTSInterface = None
        self.agent_engine: AgentInterface = None
        self.live_memory_retrieval_service = None
        self.recent_memory_context: str = ""
        self.recent_memory_diagnostics: dict = {}
        # translate_engine can be none if translation is disabled
        self.vad_engine: VADInterface | None = None
        self.translate_engine: TranslateInterface | None = None

        self.mcp_server_registery: ServerRegistry | None = None
        self.tool_adapter: ToolAdapter | None = None
        self.tool_manager: ToolManager | None = None
        self.mcp_client: MCPClient | None = None
        self.tool_executor: ToolExecutor | None = None
        self._effective_use_mcpp: bool = False
        self._bocha_api_key: str = ""

        # the system prompt is a combination of the persona prompt and live2d expression prompt
        self.system_prompt: str = None

        # Store the generated MCP prompt string (if MCP enabled)
        self.mcp_prompt: str = ""

        self.history_uid: str = ""  # Add history_uid field

        self.send_text: Callable = None
        self.client_uid: str = None

    def __str__(self):
        character_config = self.character_config
        return (
            f"ServiceContext:\n"
            f"  System Config: {'Loaded' if self.system_config else 'Not Loaded'}\n"
            f"  Live2D Model: {'Loaded' if self.live2d_model else 'Not Loaded'}\n"
            f"  ASR Engine: {type(self.asr_engine).__name__ if self.asr_engine else 'Not Loaded'}\n"
            f"  TTS Engine: {type(self.tts_engine).__name__ if self.tts_engine else 'Not Loaded'}\n"
            f"  LLM Engine: {type(self.agent_engine).__name__ if self.agent_engine else 'Not Loaded'}\n"
            f"  VAD Engine: {type(self.vad_engine).__name__ if self.vad_engine else 'Not Loaded'}\n"
            f"  Character Config: {_safe_character_config_log_fields(character_config) if character_config else 'Not Loaded'}\n"
            f"  System Prompt Chars: {len(self.system_prompt or '')}\n"
            f"  MCP Enabled: {'Yes' if self.mcp_client else 'No'}"
        )

    # ==== Initializers

    def _resolve_mcp_settings(
        self,
        use_mcpp,
        enabled_servers,
        *,
        enable_local_computer_tools: bool,
    ) -> tuple[bool, list[str]]:
        try:
            return effective_read_only_mcp_settings(
                use_mcpp=bool(use_mcpp),
                enabled_servers=enabled_servers,
                enable_local_computer_tools=enable_local_computer_tools,
            )
        except ReadOnlyComputerConfigurationError as exc:
            logger.error(
                "[只读电脑] 配置无效，本次不开放本地文件工具："
                f"{type(exc).__name__}"
            )
            return bool(use_mcpp), list(dict.fromkeys(enabled_servers or ()))

    async def _init_mcp_components(self, use_mcpp, enabled_servers):
        """Initializes MCP components based on configuration, dynamically fetching tool info."""
        logger.debug(
            f"Initializing MCP components: use_mcpp={use_mcpp}, enabled_servers={enabled_servers}"
        )

        # Reset MCP components first
        self.mcp_server_registery = None
        self.tool_manager = None
        self.mcp_client = None
        self.tool_executor = None
        self.json_detector = None
        self.mcp_prompt = ""
        self._effective_use_mcpp = bool(use_mcpp)

        if use_mcpp and enabled_servers:
            # 1. Initialize ServerRegistry
            self.mcp_server_registery = ServerRegistry()
            logger.info("ServerRegistry initialized or referenced.")

            # 2. Use ToolAdapter to get the MCP prompt and tools
            if not self.tool_adapter:
                logger.error(
                    "ToolAdapter not initialized before calling _init_mcp_components."
                )
                self.mcp_prompt = "[Error: ToolAdapter not initialized]"
                return  # Exit if ToolAdapter is mandatory and not initialized

            try:
                (
                    mcp_prompt_string,
                    openai_tools,
                    claude_tools,
                ) = await self.tool_adapter.get_tools(enabled_servers)
                # Store the generated prompt string
                self.mcp_prompt = mcp_prompt_string
                logger.info(
                    f"Dynamically generated MCP prompt string (length: {len(self.mcp_prompt)})."
                )
                logger.info(
                    f"Dynamically formatted tools - OpenAI: {len(openai_tools)}, Claude: {len(claude_tools)}."
                )

                # 3. Initialize ToolManager with the fetched formatted tools

                _, raw_tools_dict = await self.tool_adapter.get_server_and_tool_info(
                    enabled_servers
                )
                self.tool_manager = ToolManager(
                    formatted_tools_openai=openai_tools,
                    formatted_tools_claude=claude_tools,
                    initial_tools_dict=raw_tools_dict,
                )
                logger.info("ToolManager initialized with dynamically fetched tools.")

            except Exception as e:
                logger.error(
                    f"Failed during dynamic MCP tool construction: {e}", exc_info=True
                )
                # Ensure dependent components are not created if construction fails
                self.tool_manager = None
                self.mcp_prompt = "[Error constructing MCP tools/prompt]"

            # 4. Initialize MCPClient
            if self.mcp_server_registery:
                self.mcp_client = MCPClient(
                    self.mcp_server_registery, self.send_text, self.client_uid
                )
                logger.info("MCPClient initialized for this session.")
            else:
                logger.error(
                    "MCP enabled but ServerRegistry not available. MCPClient not created."
                )
                self.mcp_client = None  # Ensure it's None

            # 5. Initialize ToolExecutor
            if self.mcp_client and self.tool_manager:
                self.tool_executor = ToolExecutor(
                    self.mcp_client,
                    self.tool_manager,
                    bocha_api_key=self._bocha_api_key,
                    media_settings=settings_from_character_config(
                        self.character_config
                    ),
                )
                logger.info("ToolExecutor initialized for this session.")
            else:
                logger.warning(
                    "MCPClient or ToolManager not available. ToolExecutor not created."
                )
                self.tool_executor = None  # Ensure it's None

            logger.info("StreamJSONDetector initialized for this session.")

        elif use_mcpp and not enabled_servers:
            logger.warning(
                "use_mcpp is True, but mcp_enabled_servers list is empty. MCP components not initialized."
            )
        else:
            logger.debug(
                "MCP components not initialized (use_mcpp is False or no enabled servers)."
            )

    async def close(self):
        """Clean up resources, especially the MCPClient."""
        logger.info("Closing ServiceContext resources...")
        if self.mcp_client:
            logger.info(f"Closing MCPClient for context instance {id(self)}...")
            await self.mcp_client.aclose()
            self.mcp_client = None
        if self.agent_engine and hasattr(self.agent_engine, "close"):
            await self.agent_engine.close()  # Ensure agent resources are also closed
        self.live_memory_retrieval_service = None
        self.recent_memory_context = ""
        self.recent_memory_diagnostics = {}
        logger.info("ServiceContext closed.")

    async def load_cache(
        self,
        config: Config,
        system_config: SystemConfig,
        character_config: CharacterConfig,
        live2d_model: Live2dModel,
        asr_engine: ASRInterface,
        tts_engine: TTSInterface,
        vad_engine: VADInterface,
        translate_engine: TranslateInterface | None,
        mcp_server_registery: ServerRegistry | None = None,
        tool_adapter: ToolAdapter | None = None,
        send_text: Callable = None,
        client_uid: str = None,
        enable_local_computer_tools: bool = True,
    ) -> None:
        """
        Load the ServiceContext with shared engine references and rebuild
        session-local state that must not be shared across clients.
        """
        if not character_config:
            raise ValueError("character_config cannot be None")
        if not system_config:
            raise ValueError("system_config cannot be None")

        self.config = config
        self.system_config = system_config
        self.character_config = character_config
        self.live2d_model = live2d_model
        self.asr_engine = asr_engine
        self.tts_engine = tts_engine
        self.vad_engine = vad_engine
        self.agent_engine = None
        self.live_memory_retrieval_service = None
        self.recent_memory_context = ""
        self.recent_memory_diagnostics = {}
        self.translate_engine = translate_engine
        # Load potentially shared components by reference
        self.mcp_server_registery = mcp_server_registery
        self.tool_adapter = tool_adapter
        self.send_text = send_text
        self.client_uid = client_uid
        basic_memory_config = (
            self.character_config.agent_config.agent_settings.basic_memory_agent
        )
        self._bocha_api_key = str(
            getattr(basic_memory_config, "bocha_api_key", "") or ""
        ).strip()

        effective_use_mcpp, effective_enabled_servers = self._resolve_mcp_settings(
            basic_memory_config.use_mcpp,
            basic_memory_config.mcp_enabled_servers,
            enable_local_computer_tools=enable_local_computer_tools,
        )

        # Initialize session-specific MCP components
        await self._init_mcp_components(
            effective_use_mcpp,
            effective_enabled_servers,
        )

        await self.init_agent(
            self.character_config.agent_config,
            self.character_config.persona_prompt,
        )

        logger.debug(
            "Loaded service context with cache: {}",
            _safe_character_config_log_fields(character_config),
        )

    async def load_from_config(self, config: Config) -> None:
        """
        Load the ServiceContext with the config.
        Reinitialize the instances if the config is different.

        Parameters:
        - config (Dict): The configuration dictionary.
        """
        if not self.config:
            self.config = config

        if not self.system_config:
            self.system_config = config.system_config

        if not self.character_config:
            self.character_config = config.character_config

        # update all sub-configs

        # init live2d from character config
        self.init_live2d(config.character_config.live2d_model_name)

        # init asr from character config
        self.init_asr(config.character_config.asr_config)

        headless_private_bridge = _headless_private_bridge_enabled()
        if headless_private_bridge:
            logger.info(
                "[私人QQ桥接] 保留语音识别，跳过桌面 TTS、VAD 与翻译引擎。"
            )
        else:
            # init tts from character config
            self.init_tts(config.character_config.tts_config)

            # init vad from character config
            self.init_vad(config.character_config.vad_config)

        basic_memory_config = (
            config.character_config.agent_config.agent_settings.basic_memory_agent
        )
        self._bocha_api_key = str(
            getattr(basic_memory_config, "bocha_api_key", "") or ""
        ).strip()
        effective_use_mcpp, effective_enabled_servers = self._resolve_mcp_settings(
            basic_memory_config.use_mcpp,
            basic_memory_config.mcp_enabled_servers,
            enable_local_computer_tools=True,
        )

        # Initialize shared ToolAdapter if it doesn't exist yet
        if (
            not self.tool_adapter
            and effective_use_mcpp
        ):
            if not self.mcp_server_registery:
                logger.info(
                    "Initializing shared ServerRegistry within load_from_config."
                )
                self.mcp_server_registery = ServerRegistry()
            logger.info("Initializing shared ToolAdapter within load_from_config.")
            self.tool_adapter = ToolAdapter(server_registery=self.mcp_server_registery)

        # Initialize MCP Components before initializing Agent
        await self._init_mcp_components(
            effective_use_mcpp,
            effective_enabled_servers,
        )

        # init agent from character config
        await self.init_agent(
            config.character_config.agent_config,
            config.character_config.persona_prompt,
        )

        if not headless_private_bridge:
            self.init_translate(
                config.character_config.tts_preprocessor_config.translator_config
            )
        
        try:
            # 确定历史记录文件夹路径
            history_dir = character_history_root(config.character_config.conf_uid)
            if history_dir.exists():
                # 获取所有 JSON 文件并按修改时间排序
                files = list(history_dir.glob("*.json"))
                if files:
                    latest_file = max(files, key=lambda path: path.stat().st_mtime)
                    # 从文件名提取 history_uid
                    history_uid = latest_file.stem
                    
                    if self.agent_engine and hasattr(self.agent_engine, 'set_memory_from_history'):
                        self.agent_engine.set_memory_from_history(
                            conf_uid=config.character_config.conf_uid,
                            history_uid=history_uid,
                        )
                        logger.info("自动加载长期记忆成功: history_uid_present=True")
        except Exception as e:
            logger.warning(f"自动加载长期记忆失败: {e}")

        # store typed config references
        self.config = config
        self.system_config = config.system_config or self.system_config
        self.character_config = config.character_config

    def init_live2d(self, live2d_model_name: str) -> None:
        logger.info(f"Initializing Live2D: {live2d_model_name}")
        try:
            self.live2d_model = Live2dModel(
                live2d_model_name,
                emotion_map_override=active_rinne_emotion_map(live2d_model_name),
            )
            self.character_config.live2d_model_name = live2d_model_name
        except Exception as e:
            logger.critical(f"Error initializing Live2D: {e}")
            logger.critical("Try to proceed without Live2D...")

    def init_asr(self, asr_config: ASRConfig) -> None:
        if not self.asr_engine or (self.character_config.asr_config != asr_config):
            logger.info(f"Initializing ASR: {asr_config.asr_model}")
            self.asr_engine = ASRFactory.get_asr_system(
                asr_config.asr_model,
                **getattr(asr_config, asr_config.asr_model).model_dump(),
            )
            # saving config should be done after successful initialization
            self.character_config.asr_config = asr_config
        else:
            logger.info("ASR already initialized with the same config.")

    def init_tts(self, tts_config: TTSConfig) -> None:
        if not self.tts_engine or (self.character_config.tts_config != tts_config):
            logger.info(f"Initializing TTS: {tts_config.tts_model}")
            self.tts_engine = TTSFactory.get_tts_engine(
                tts_config.tts_model,
                **getattr(tts_config, tts_config.tts_model.lower()).model_dump(),
            )
            # saving config should be done after successful initialization
            self.character_config.tts_config = tts_config
        else:
            logger.info("TTS already initialized with the same config.")

    def init_vad(self, vad_config: VADConfig) -> None:
        if vad_config.vad_model is None:
            logger.info("VAD is disabled.")
            self.vad_engine = None
            return

        if not self.vad_engine or (self.character_config.vad_config != vad_config):
            logger.info(f"Initializing VAD: {vad_config.vad_model}")
            self.vad_engine = VADFactory.get_vad_engine(
                vad_config.vad_model,
                **getattr(vad_config, vad_config.vad_model.lower()).model_dump(),
            )
            # saving config should be done after successful initialization
            self.character_config.vad_config = vad_config
        else:
            logger.info("VAD already initialized with the same config.")

    async def init_agent(self, agent_config: AgentConfig, persona_prompt: str) -> None:
        """Initialize or update the LLM engine based on agent configuration."""
        logger.info(f"Initializing Agent: {agent_config.conversation_agent_choice}")

        if (
            self.agent_engine is not None
            and agent_config == self.character_config.agent_config
            and persona_prompt == self.character_config.persona_prompt
        ):
            logger.debug("Agent already initialized with the same config.")
            return

        system_prompt = await self.construct_system_prompt(persona_prompt)

        # Pass avatar to agent factory
        avatar = self.character_config.avatar or ""  # Get avatar from config

        try:
            effective_agent_settings = agent_config.agent_settings.model_dump()
            effective_agent_settings["basic_memory_agent"]["use_mcpp"] = (
                self._effective_use_mcpp
            )
            self.agent_engine = AgentFactory.create_agent(
                conversation_agent_choice=agent_config.conversation_agent_choice,
                agent_settings=effective_agent_settings,
                llm_configs=agent_config.llm_configs.model_dump(),
                system_prompt=system_prompt,
                live2d_model=self.live2d_model,
                tts_preprocessor_config=self.character_config.tts_preprocessor_config,
                character_avatar=avatar,
                system_config=self.system_config.model_dump(),
                tool_manager=self.tool_manager,
                tool_executor=self.tool_executor,
                mcp_prompt_string=self.mcp_prompt,
            )

            logger.debug(f"Agent choice: {agent_config.conversation_agent_choice}")
            logger.debug("System prompt prepared: chars={}", len(system_prompt))

            # Save the current configuration
            self.character_config.agent_config = agent_config
            self.system_prompt = system_prompt

        except Exception as e:
            logger.error("Failed to initialize agent: {}", type(e).__name__)
            raise

    def init_translate(self, translator_config: TranslatorConfig) -> None:
        """Initialize or update the translation engine based on the configuration."""

        if not translator_config.translate_audio:
            logger.debug("Translation is disabled.")
            return

        if (
            not self.translate_engine
            or self.character_config.tts_preprocessor_config.translator_config
            != translator_config
        ):
            logger.info(
                f"Initializing Translator: {translator_config.translate_provider}"
            )
            #获取对应提供商的配置段
            provider_config = getattr(
                translator_config, translator_config.translate_provider
            )
            #兼容Pydantic模型和普通字典
            if hasattr(provider_config, "model_dump"):
                provider_config = provider_config.model_dump()

            self.translate_engine = TranslateFactory.get_translator(
                translator_config.translate_provider,
                provider_config
            )

            self.character_config.tts_preprocessor_config.translator_config = (
                translator_config
            )
        else:
            logger.info("Translation already initialized with the same config.")

    # ==== utils

    async def construct_system_prompt(self, persona_prompt: str) -> str:
        """
        Append tool prompts to persona prompt.

        Parameters:
        - persona_prompt (str): The persona prompt.

        Returns:
        - str: The system prompt with all tool prompts appended.
        """
        logger.debug("Constructing persona prompt: chars={}", len(persona_prompt))

        for prompt_name, prompt_file in self.system_config.tool_prompts.items():
            if (
                prompt_name == "group_conversation_prompt"
                or prompt_name == "proactive_speak_prompt"
            ):
                continue

            prompt_content = prompt_loader.load_util(prompt_file)

            if prompt_name == "live2d_expression_prompt":
                prompt_content = prompt_content.replace(
                    "[<insert_emomap_keys>]", self.live2d_model.emo_str
                )

            if prompt_name == "mcp_prompt":
                continue

            persona_prompt += prompt_content

        logger.debug("System prompt construction complete: chars={}", len(persona_prompt))

        return persona_prompt

    async def handle_config_switch(
        self,
        websocket: WebSocket,
        config_file_name: str,
    ) -> None:
        """
        Handle the configuration switch request.
        Change the configuration to a new config and notify the client.

        Parameters:
        - websocket (WebSocket): The WebSocket connection.
        - config_file_name (str): The name of the configuration file.
        """
        try:
            new_character_config_data = None

            if config_file_name == "conf.yaml":
                # Load base config
                new_character_config_data = read_yaml("conf.yaml").get(
                    "character_config"
                )
            else:
                # Load alternative config and merge with base config
                characters_dir = self.system_config.config_alts_dir
                file_path = os.path.normpath(
                    os.path.join(characters_dir, config_file_name)
                )
                if not file_path.startswith(characters_dir):
                    raise ValueError("Invalid configuration file path")

                alt_config_data = read_yaml(file_path).get("character_config")

                # Start with original config data and perform a deep merge
                new_character_config_data = deep_merge(
                    self.config.character_config.model_dump(), alt_config_data
                )

            if new_character_config_data:
                new_config = {
                    "system_config": self.system_config.model_dump(),
                    "character_config": new_character_config_data,
                }
                new_config = validate_config(new_config)
                await self.load_from_config(new_config)  # Await the async load
                logger.debug(
                    "New character config: {}",
                    _safe_character_config_log_fields(self.character_config),
                )

                # Send responses to client
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "set-model-and-conf",
                            "model_info": self.live2d_model.model_info,
                            "conf_name": self.character_config.conf_name,
                            "conf_uid": self.character_config.conf_uid,
                        }
                    )
                )

                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "config-switched",
                            "message": f"Switched to config: {config_file_name}",
                        }
                    )
                )

                logger.info(f"Configuration switched to {config_file_name}")
            else:
                raise ValueError(
                    f"Failed to load configuration from {config_file_name}"
                )

        except Exception as e:
            logger.error("Error switching configuration: {}", type(e).__name__)
            logger.debug(self)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"Error switching configuration: {str(e)}",
                    }
                )
            )
            raise e


def deep_merge(dict1, dict2):
    """
    Recursively merges dict2 into dict1, prioritizing values from dict2.
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
