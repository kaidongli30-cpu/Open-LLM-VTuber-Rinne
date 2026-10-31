"""Create the optional route without changing the existing relay configuration."""

import os
from pathlib import Path

from loguru import logger

from .runtime import OfficialRouteRuntime
from .runtime_checks import RuntimeContentChecker
from ..data_paths import character_history_root

_RUNTIMES = {}


def create_official_route(
    config, relay_llm, llm_provider, llm_config, conf_uid="rinne_01"
):
    values = (
        config.model_dump() if hasattr(config, "model_dump") else dict(config or {})
    )
    if not values.get("enabled", False):
        return None
    chat_check_mode = values.get("chat_check_mode", "two_pass")
    if chat_check_mode not in ("two_pass", "explicit_only"):
        raise ValueError("invalid_chat_check_mode")
    if not os.environ.get("RINNE_OFFICIAL_ANTHROPIC_KEY") and os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                os.environ["RINNE_OFFICIAL_ANTHROPIC_KEY"] = winreg.QueryValueEx(
                    key, "RINNE_OFFICIAL_ANTHROPIC_KEY"
                )[0]
        except (OSError, ImportError):
            pass
    root = character_history_root(conf_uid)
    from .openai_client import read_openai_api_key

    if not os.environ.get("RINNE_OPENAI_API_KEY"):
        key = read_openai_api_key()
        if key:
            os.environ["RINNE_OPENAI_API_KEY"] = key
    data_dir = Path(values.get("data_dir") or root / "official_route").resolve()
    if os.environ.get("RINNE_DATA_ROOT"):
        # Test data roots always win over a literal production data_dir.
        data_dir = (root / "official_route").resolve()
    cache_key = (
        str(data_dir),
        values.get("model"),
        values.get("thinking"),
        values.get("credit_as_of"),
        values.get("first_pass_model", "claude-sonnet-4-6"),
        values.get("second_pass_model", "claude-opus-4-6"),
        chat_check_mode,
        values.get("route", "anthropic"),
        values.get("effort"),
        values.get("first_text_timeout_seconds", 40),
        str(values.get("openai", {})),
    )
    if cache_key in _RUNTIMES:
        return _RUNTIMES[cache_key]
    keyword_path = data_dir / "关键词.txt"
    if not keyword_path.is_file():
        logger.warning("官方通道缺少关键词表，本次不启用：{}", keyword_path)
        return None
    from ..agent.stateless_llm_factory import LLMFactory

    common = dict(llm_config)
    common.pop("interrupt_method", None)
    common.update(upstream_max_attempts=1, empty_stream_recovery=False)
    from .model_profiles import CHECK_MODELS

    first_name = values.get("first_pass_model", "claude-sonnet-4-6")
    if chat_check_mode == "explicit_only":
        first_name = "claude-sonnet-4-6"
    second_name = values.get("second_pass_model", "claude-opus-4-6")
    first = LLMFactory.create_llm(
        llm_provider=llm_provider,
        **{
            **common,
            "model": first_name if first_name in CHECK_MODELS else "claude-sonnet-4-6",
        },
    )
    second = LLMFactory.create_llm(
        llm_provider=llm_provider,
        **{
            **common,
            "model": second_name if second_name in CHECK_MODELS else "claude-opus-4-6",
        },
    )
    checker = RuntimeContentChecker(
        keyword_path,
        data_dir / "content_checks.jsonl",
        first,
        second,
        data_dir / "rulings.jsonl",
        second_record_paths=[
            data_dir / "content_checks_second_pass.jsonl",
            data_dir / "content_checks_second_pass_masked.jsonl",
        ],
        aliases_path=data_dir / "cache_aliases.jsonl",
        chat_check_mode=chat_check_mode,
    )
    runtime = OfficialRouteRuntime(values, data_dir, relay_llm, checker)
    _RUNTIMES[cache_key] = runtime
    return runtime
