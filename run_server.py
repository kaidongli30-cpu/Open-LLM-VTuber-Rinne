import os
import sys
import atexit
import asyncio
import argparse
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import tomli
import uvicorn
from loguru import logger
from upgrade_codes.upgrade_manager import UpgradeManager

from src.open_llm_vtuber.server import WebSocketServer
from src.open_llm_vtuber.config_manager import Config, read_yaml, validate_config
from src.open_llm_vtuber.rinne_renderer_profile import (
    RinneRuntimeOutfitController,
    apply_rinne_renderer_profile_to_config,
    prepare_rinne_renderer_startup,
)
from src.open_llm_vtuber.data_paths import (
    character_history_root,
    resolve_character_history_root,
)
from src.open_llm_vtuber.memory.startup_chat_history_archive import (
    archive_old_chat_history_on_startup,
)

os.environ["HF_HOME"] = str(Path(__file__).parent / "models")
os.environ["MODELSCOPE_CACHE"] = str(Path(__file__).parent / "models")

upgrade_manager = UpgradeManager()


@dataclass
class RinneMemoryStartupOutcome:
    child_event_worker: object | None = None
    layer2_worker: object | None = None
    notifications: list[dict[str, str]] = field(default_factory=list)
    diary_ready_for_archive: bool = False


def _last_complete_memory_day(reference_time: datetime) -> datetime:
    """Return the diary day that fully ended before this backend start."""

    days_back = 2 if reference_time.hour < 3 else 1
    return reference_time - timedelta(days=days_back)


def prepare_rinne_memories_on_startup(
    reference_time: datetime | None = None,
    history_root: str | Path | None = None,
    generation_settings=None,
    layer2_generation_settings=None,
    config_path: str | Path = "conf.yaml",
) -> RinneMemoryStartupOutcome:
    """Generate periodic memories and launch child-event work in background."""

    import diary_generator
    import monthly_generator
    import weekly_generator
    from src.open_llm_vtuber.memory.long_term_archive import (
        load_today_messages,
        select_long_term_memories,
    )
    from src.open_llm_vtuber.memory.daily_child_events import (
        daily_child_event_publication_status,
        launch_daily_child_event_worker,
    )
    from src.open_llm_vtuber.memory.diary_review import wait_for_diary_approval
    from src.open_llm_vtuber.memory.scene_state import SceneStateStore
    from src.open_llm_vtuber.memory.layer2_runtime import (
        layer2_publication_status,
        launch_layer2_worker,
    )

    now = reference_time or datetime.now()
    outcome = RinneMemoryStartupOutcome()
    history_root = resolve_character_history_root(history_root)
    last_complete_day = _last_complete_memory_day(now)
    from src.open_llm_vtuber.memory.diary_sources import pending_chat_diary_days

    # Recover genuine older conversations before processing the latest day.
    # Empty intervening days are intentionally absent from this list.
    for pending_day in pending_chat_diary_days(history_root, last_complete_day.date()):
        pending_path = history_root / "diaries" / f"diary_{pending_day}.txt"
        logger.info(f"[记忆生成] 补查旧聊天日记：{pending_day}")
        try:
            diary_generator.generate_for_date(
                datetime.combine(pending_day, datetime.min.time())
            )
        except Exception as exc:
            logger.error(f"[记忆生成] {pending_day} 补写失败，保留原始聊天：{exc}")
            continue
        if pending_path.is_file() and pending_path.stat().st_size:
            wait_for_diary_approval(history_root, pending_day.isoformat(), pending_path)
        else:
            logger.warning(f"[记忆生成] {pending_day} 日记未生成，保留原始聊天")
    diary_path = (
        history_root
        / "diaries"
        / (f"diary_{last_complete_day.strftime('%Y-%m-%d')}.txt")
    )

    logger.info(f"[记忆生成] 检查上一完整记忆日：{last_complete_day:%Y-%m-%d}")
    try:
        diary_generator.generate_for_date(last_complete_day)
    except Exception as exc:
        logger.error(f"[记忆生成] 日记生成出现异常，暂停周记和月记：{exc}")
    else:
        if not diary_path.exists() or diary_path.stat().st_size == 0:
            logger.error(
                "[记忆生成] 上一完整记忆日的日记未成功生成，"
                "本次暂停周记和月记；后端仍会继续启动。"
            )
        else:
            approval = wait_for_diary_approval(
                history_root,
                last_complete_day.date().isoformat(),
                diary_path,
            )
            logger.info(
                "[日记验收] 已确认可供后续记忆流程使用："
                f"{last_complete_day:%Y-%m-%d}，状态={approval['status']}"
            )
            outcome.diary_ready_for_archive = True
            if approval.get("status") == "changed_after_approval":
                logger.warning(
                    "[日记验收] 这份日记在上次 approve 后发生了手动修改；"
                    "本次按当前内容继续，不自动重建既有第二层或子事件。"
                )
            memory_day = last_complete_day.date().isoformat()
            try:
                # Scene detail is a current-day aid only.  Once the diary has
                # been accepted, discard that day's temporary change log while
                # retaining the compact current snapshot for ongoing scenes.
                SceneStateStore(history_root).discard_change_log(memory_day)
            except Exception as exc:
                logger.warning(f"[当前场景] 临时变更日志清理失败：{exc}")
            publication_status = daily_child_event_publication_status(
                memory_day, history_root
            )
            if publication_status == "current":
                logger.info(
                    f"[每日子事件] {memory_day} 事件已存在，"
                    "跳过整理；未调用事件生成模型。"
                )
            elif publication_status == "stale":
                logger.warning(
                    f"[每日子事件] {memory_day} 事件已存在，但日记已在事件"
                    "生成后修改；为避免自动覆盖，未启动事件生成模型。"
                )
            elif publication_status == "invalid":
                logger.error(
                    f"[每日子事件] {memory_day} 事件目录已存在，但发布清单或"
                    "事件文件不完整；为避免自动覆盖，未启动事件生成模型。"
                )
            elif generation_settings is not None and not generation_settings.enabled:
                logger.info(
                    f"[每日子事件] {memory_day} 尚未生成，但每日事件模型接口已禁用；"
                    "未启动模型。"
                )
            else:
                try:
                    launch_kwargs = {}
                    if generation_settings is not None:
                        launch_kwargs["config_path"] = config_path
                    child_event_worker = launch_daily_child_event_worker(
                        last_complete_day.date(), history_root, **launch_kwargs
                    )
                    outcome.child_event_worker = child_event_worker
                    if generation_settings is None:
                        provider_label = "ollama_llm"
                        model_label = "mistral-small3.2:24b"
                    else:
                        provider_label = generation_settings.llm_provider
                        model_label = generation_settings.model
                    logger.info(
                        f"[每日子事件] {memory_day} 尚未生成，"
                        "已启动后台整理任务："
                        f"接口={provider_label}，模型={model_label}，"
                        f"PID={child_event_worker.process.pid}"
                    )
                except Exception as exc:
                    logger.error(
                        "[每日子事件] 后台任务启动失败；"
                        f"不影响后端继续启动：{exc}"
                    )
                    outcome.notifications.append(
                        {
                            "type": "memory-notification",
                            "level": "error",
                            "message": "昨日事件整理失败",
                            "description": "24B后台任务启动失败，请查看后端日志",
                        }
                    )
            if layer2_generation_settings is None:
                logger.info("[第二层记忆] 未提供运行配置，本次不启动个人背景更新。")
            elif not layer2_generation_settings.enabled:
                logger.info("[第二层记忆] 接口已禁用，本次不启动个人背景更新。")
            else:
                layer2_status = layer2_publication_status(memory_day, history_root)
                if layer2_status == "current":
                    logger.info(
                        f"[第二层记忆] {memory_day} 个人背景已是当前版本，"
                        "跳过更新；未调用云端模型。"
                    )
                elif layer2_status == "stale":
                    logger.warning(
                        f"[第二层记忆] {memory_day} 的日记在个人背景发布后发生变化；"
                        "为避免覆盖，等待人工复核。"
                    )
                    outcome.notifications.append(
                        {
                            "type": "memory-notification",
                            "level": "error",
                            "message": "个人背景等待人工复核",
                            "description": "当日日记在第二层发布后发生变化",
                        }
                    )
                elif layer2_status == "invalid":
                    logger.error(
                        "[第二层记忆] 当前发布不完整或校验失败；"
                        "未启动更新，原始文件未被覆盖。"
                    )
                    outcome.notifications.append(
                        {
                            "type": "memory-notification",
                            "level": "error",
                            "message": "个人背景状态异常",
                            "description": "当前第二层发布未通过完整性校验",
                        }
                    )
                else:
                    try:
                        layer2_worker = launch_layer2_worker(
                            last_complete_day.date(),
                            history_root,
                            config_path=config_path,
                        )
                        outcome.layer2_worker = layer2_worker
                        logger.info(
                            f"[第二层记忆] 已启动 {memory_day} 后台更新："
                            f"接口配置={layer2_generation_settings.provider_config_name}，"
                            f"模型={layer2_generation_settings.model}，"
                            f"PID={layer2_worker.process.pid}"
                        )
                    except Exception as exc:
                        logger.error(
                            "[第二层记忆] 后台任务启动失败；"
                            f"继续使用上一份有效背景：{exc}"
                        )
                        outcome.notifications.append(
                            {
                                "type": "memory-notification",
                                "level": "error",
                                "message": "个人背景更新启动失败",
                                "description": "继续使用上一份有效个人背景",
                            }
                        )
            try:
                weekly_result = weekly_generator.generate_latest_completed_week(now)
            except Exception as exc:
                logger.error(f"[记忆生成] 周记生成出现异常：{exc}")
            else:
                if weekly_result.status == "failed":
                    logger.error(
                        f"[记忆生成] 周记生成失败：{weekly_result.error}"
                    )
            try:
                monthly_result = monthly_generator.generate_latest_completed_month(now)
            except Exception as exc:
                logger.error(f"[记忆生成] 月记生成出现异常：{exc}")
            else:
                if monthly_result.status == "failed":
                    logger.error(f"[记忆生成] 月记生成失败：{monthly_result.error}")

    selection = select_long_term_memories(history_root)
    today_messages = load_today_messages(history_root, now)
    logger.info(
        "[长期记忆可用] "
        f"{len(selection.monthly_entries)} 篇月记 + "
        f"{len(selection.weekly_entries)} 篇周记 + "
        f"{len(selection.diary_entries)} 篇日记 + "
        f"今日 {len(today_messages)} 条消息"
    )
    for warning in selection.diagnostics.warnings:
        logger.warning(f"[长期记忆] {warning}")
    return outcome


def prepare_recent_diary_context_on_startup(
    reference_time: datetime | None = None,
    history_root: str | Path | None = None,
) -> object | None:
    """Publish a compact view of the last approved diary, or safely fall back."""

    import diary_generator
    from src.open_llm_vtuber.memory.diary_review import diary_approval_status
    from src.open_llm_vtuber.memory.recent_diary_compaction import (
        RecentDiaryCompactionError,
        load_matching_compaction,
        publish_compaction,
        request_compaction,
    )

    now = reference_time or datetime.now()
    root = resolve_character_history_root(history_root)
    memory_day = _last_complete_memory_day(now).date().isoformat()
    diary_path = root / "diaries" / f"diary_{memory_day}.txt"
    if not diary_path.is_file() or not diary_path.stat().st_size:
        return None
    approval_status = diary_approval_status(root, memory_day, diary_path)
    if approval_status != "approved":
        logger.warning(
            f"[近期日记压缩] {memory_day} 状态={approval_status}，"
            "不自动生成；本轮回退读取完整日记。"
        )
        return None
    existing = load_matching_compaction(root, diary_path, memory_day)
    if existing is not None:
        logger.info(
            f"[近期日记压缩] {memory_day} 已存在且与日记匹配，"
            f"{existing.source_character_count}→{existing.summary_character_count} 字符。"
        )
        return existing
    try:
        diary_text = diary_path.read_text(encoding="utf-8").strip()
        compaction = request_compaction(
            memory_day=memory_day,
            diary_text=diary_text,
            api_key=diary_generator.LLM_API_KEY,
            api_url=diary_generator.LLM_API_URL,
            model=diary_generator.LLM_MODEL,
            max_attempts=3,
        )
        publish_compaction(root, diary_path, compaction)
    except (OSError, UnicodeError, RecentDiaryCompactionError) as exc:
        logger.warning(
            f"[近期日记压缩] {memory_day} 生成或校验失败：{exc}；"
            "本轮回退读取完整日记。"
        )
        return None
    logger.info(
        f"[近期日记压缩] {memory_day} 已发布："
        f"{compaction.source_character_count}→"
        f"{compaction.summary_character_count} 字符，"
        f"耗时 {compaction.elapsed_seconds:.1f} 秒。"
    )
    return compaction


def get_version() -> str:
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
    return pyproject["project"]["version"]


def init_logger(console_log_level: str = "INFO") -> None:
    logger.remove()
    # Console output
    logger.add(
        sys.stderr,
        level=console_log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | {message}",
        colorize=True,
    )

    # File output
    logger.add(
        "logs/debug_{time:YYYY-MM-DD}.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message} | {extra}",
        backtrace=True,
        diagnose=True,
    )


def check_frontend_submodule(lang=None):
    """
    Check if the frontend submodule is initialized. If not, attempt to initialize it.
    If initialization fails, log an error message.
    """
    if lang is None:
        lang = upgrade_manager.lang

    frontend_path = Path(__file__).parent / "frontend" / "index.html"
    if not frontend_path.exists():
        if lang == "zh":
            logger.warning("未找到前端子模块，正在尝试初始化子模块...")
        else:
            logger.warning(
                "Frontend submodule not found, attempting to initialize submodules..."
            )

        try:
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"], check=True
            )
            if frontend_path.exists():
                if lang == "zh":
                    logger.info("👍 前端子模块（和其他子模块）初始化成功。")
                else:
                    logger.info(
                        "👍 Frontend submodule (and other submodules) initialized successfully."
                    )
            else:
                if lang == "zh":
                    logger.critical(
                        '子模块初始化失败。\n你之后可能会在浏览器中看到 {{"detail":"Not Found"}} 的错误提示。请检查我们的快速入门指南和常见问题页面以获取更多信息。'
                    )
                    logger.error(
                        "初始化子模块后，前端文件仍然缺失。\n"
                        + "你是否手动更改或删除了 `frontend` 文件夹？\n"
                        + "它是一个 Git 子模块 - 你不应该直接修改它。\n"
                        + "如果你这样做了，请使用 `git restore frontend` 丢弃你的更改，然后再试一次。\n"
                    )
                else:
                    logger.critical(
                        'Failed to initialize submodules. \nYou might see {{"detail":"Not Found"}} in your browser. Please check our quick start guide and common issues page from our documentation.'
                    )
                    logger.error(
                        "Frontend files are still missing after submodule initialization.\n"
                        + "Did you manually change or delete the `frontend` folder?  \n"
                        + "It's a Git submodule — you shouldn't modify it directly.  \n"
                        + "If you did, discard your changes with `git restore frontend`, then try again.\n"
                    )
        except Exception as e:
            if lang == "zh":
                logger.critical(
                    f'初始化子模块失败: {e}。\n怀疑你跟 GitHub 之间有网络问题。你之后可能会在浏览器中看到 {{"detail":"Not Found"}} 的错误提示。请检查我们的快速入门指南和常见问题页面以获取更多信息。\n'
                )
            else:
                logger.critical(
                    f'Failed to initialize submodules: {e}. \nYou might see {{"detail":"Not Found"}} in your browser. Please check our quick start guide and common issues page from our documentation.\n'
                )


def parse_args():
    parser = argparse.ArgumentParser(description="Open-LLM-VTuber Server")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--hf_mirror", action="store_true", help="Use Hugging Face mirror"
    )
    return parser.parse_args()


@logger.catch
def run(console_log_level: str):
    init_logger(console_log_level)
    logger.info(f"Open-LLM-VTuber, version v{get_version()}")

    # Get selected language
    lang = upgrade_manager.lang

    # Check if the frontend submodule is initialized
    check_frontend_submodule(lang)

    # Sync user config with default config
    try:
        upgrade_manager.sync_user_config()
    except Exception as e:
        logger.error(f"Error syncing user config: {e}")

    atexit.register(WebSocketServer.clean_cache)

    # Load configurations from yaml file. Rinne's renderer selection is applied
    # in memory; the user's conf.yaml remains untouched.
    config_data = read_yaml("conf.yaml")
    rinne_outfit_controller = None
    if config_data.get("character_config", {}).get("conf_uid") == "rinne_01":
        renderer_startup = prepare_rinne_renderer_startup(
            Path(__file__).parent,
            interactive=False,
        )
        if renderer_startup.profile.renderer == "rinne":
            rinne_outfit_controller = RinneRuntimeOutfitController(
                Path(__file__).parent,
                config_data,
                renderer_startup,
            )
        config_data = apply_rinne_renderer_profile_to_config(
            config_data, renderer_startup.profile
        )
        logger.info(f"Rinne renderer selected: {renderer_startup.profile.menu_label}")
    config: Config = validate_config(config_data)
    server_config = config.system_config
    headless_private_bridge = os.environ.get(
        "RINNE_HEADLESS_PRIVATE_BRIDGE", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    server_host = server_config.host
    server_port = server_config.port
    if headless_private_bridge:
        server_host = "127.0.0.1"
        raw_private_port = os.environ.get(
            "RINNE_HEADLESS_PRIVATE_BRIDGE_PORT", "12394"
        ).strip()
        try:
            server_port = int(raw_private_port)
        except ValueError:
            logger.error("RINNE_HEADLESS_PRIVATE_BRIDGE_PORT 必须是整数")
            sys.exit(1)
        if not 1 <= server_port <= 65535:
            logger.error("RINNE_HEADLESS_PRIVATE_BRIDGE_PORT 超出有效端口范围")
            sys.exit(1)

    memory_startup = RinneMemoryStartupOutcome()
    if config.character_config.conf_uid == "rinne_01":
        history_root = character_history_root(config.character_config.conf_uid)
        logger.info(f"[数据目录] 聊天与记忆目录：{history_root.resolve()}")
        if headless_private_bridge:
            logger.info(
                "[私人QQ桥接] 使用 E 盘既有记忆，跳过重复的日记与记忆生成任务"
            )
        else:
            memory_startup = prepare_rinne_memories_on_startup(
                history_root=history_root,
                generation_settings=config.character_config.daily_child_event_generation,
                layer2_generation_settings=config.character_config.layer2_memory_generation,
                config_path="conf.yaml",
            )
            if memory_startup.diary_ready_for_archive:
                prepare_recent_diary_context_on_startup(history_root=history_root)
                try:
                    archive_old_chat_history_on_startup(history_root)
                except Exception as exc:
                    logger.error(
                        "[chat archive] startup archive failed; continuing server startup: {}",
                        exc,
                    )
            else:
                logger.warning(
                    "[chat archive] diary was not generated and approved; "
                    "keeping source JSON files for the next retry"
                )

    if server_config.enable_proxy:
        logger.info("Proxy mode enabled - /proxy-ws endpoint will be available")

    # Initialize the WebSocket server (synchronous part)
    server = WebSocketServer(
        config=config,
        rinne_outfit_controller=rinne_outfit_controller,
    )
    if memory_startup.child_event_worker is not None:
        server.watch_child_event_worker(memory_startup.child_event_worker)
    if memory_startup.layer2_worker is not None:
        server.watch_layer2_worker(memory_startup.layer2_worker)
    for notification in memory_startup.notifications:
        server.queue_system_notification(notification)

    # Perform asynchronous initialization (loading context, etc.)
    logger.info("Initializing server context...")
    try:
        asyncio.run(server.initialize())
        logger.info("Server context initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize server context: {e}")
        sys.exit(1)  # Exit if initialization fails

    # Run the Uvicorn server
    logger.info(f"Starting server on {server_host}:{server_port}")
    uvicorn.run(
        app=server.app,
        host=server_host,
        port=server_port,
        log_level=console_log_level.lower(),
    )


if __name__ == "__main__":
    args = parse_args()
    console_log_level = "DEBUG" if args.verbose else "INFO"
    if args.verbose:
        logger.info("Running in verbose mode")
    else:
        logger.info(
            "Running in standard mode. For detailed debug logs, use: uv run run_server.py --verbose"
        )
    if args.hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    run(console_log_level=console_log_level)
