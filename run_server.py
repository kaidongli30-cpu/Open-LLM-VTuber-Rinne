import os
import sys
import atexit
import asyncio
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import tomli
import uvicorn
from loguru import logger
from upgrade_codes.upgrade_manager import UpgradeManager

from src.open_llm_vtuber.server import WebSocketServer
from src.open_llm_vtuber.config_manager import Config, read_yaml, validate_config

os.environ["HF_HOME"] = str(Path(__file__).parent / "models")
os.environ["MODELSCOPE_CACHE"] = str(Path(__file__).parent / "models")

upgrade_manager = UpgradeManager()


def _last_complete_memory_day(reference_time: datetime) -> datetime:
    """Return the diary day that fully ended before this backend start."""

    days_back = 2 if reference_time.hour < 3 else 1
    return reference_time - timedelta(days=days_back)


def prepare_rinne_memories_on_startup(
    reference_time: datetime | None = None,
    history_root: str | Path = Path("chat_history/rinne_01"),
) -> None:
    """Run the one-shot diary -> weekly -> monthly startup sequence safely."""

    import diary_generator
    import monthly_generator
    import weekly_generator
    from src.open_llm_vtuber.memory.long_term_archive import (
        load_today_messages,
        select_long_term_memories,
    )

    now = reference_time or datetime.now()
    history_root = Path(history_root)
    last_complete_day = _last_complete_memory_day(now)
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
            try:
                weekly_result = weekly_generator.generate_latest_completed_week(now)
            except Exception as exc:
                logger.error(f"[记忆生成] 周记生成出现异常，暂停月记：{exc}")
            else:
                if weekly_result.status == "failed":
                    logger.error(
                        f"[记忆生成] 周记生成失败，暂停月记：{weekly_result.error}"
                    )
                else:
                    try:
                        monthly_result = (
                            monthly_generator.generate_latest_completed_month(now)
                        )
                    except Exception as exc:
                        logger.error(f"[记忆生成] 月记生成出现异常：{exc}")
                    else:
                        if monthly_result.status == "failed":
                            logger.error(
                                f"[记忆生成] 月记生成失败：{monthly_result.error}"
                            )

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

    # Load configurations from yaml file
    config: Config = validate_config(read_yaml("conf.yaml"))
    server_config = config.system_config

    if config.character_config.conf_uid == "rinne_01":
        prepare_rinne_memories_on_startup()

    if server_config.enable_proxy:
        logger.info("Proxy mode enabled - /proxy-ws endpoint will be available")

    # Initialize the WebSocket server (synchronous part)
    server = WebSocketServer(config=config)

    # Perform asynchronous initialization (loading context, etc.)
    logger.info("Initializing server context...")
    try:
        asyncio.run(server.initialize())
        logger.info("Server context initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize server context: {e}")
        sys.exit(1)  # Exit if initialization fails

    # Run the Uvicorn server
    logger.info(f"Starting server on {server_config.host}:{server_config.port}")
    uvicorn.run(
        app=server.app,
        host=server_config.host,
        port=server_config.port,
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
