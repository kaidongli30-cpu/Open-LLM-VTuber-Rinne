"""Isolated neutral cloud evaluation for Rinne layer-two user backgrounds.

The runner never edits the live desktop-pet configuration or memory files.  It
freezes diary inputs and a test-case-neutral prompt into an external experiment
directory, then performs independent chronological replays with the cloud model
already selected by ``conf.yaml``.  Credentials are read only at call time and
are never written to experiment artifacts.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
import unicodedata
import urllib.parse
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator


TOOLS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TOOLS_DIR.parent
SRC_DIR = REPOSITORY_ROOT / "src"
for import_path in (TOOLS_DIR, SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import layer2_background_delta_experiment as delta_runner  # noqa: E402
import layer2_background_experiment as legacy  # noqa: E402
import layer2_model_facing_background as model_facing  # noqa: E402
from open_llm_vtuber.config_manager.daily_child_event import (  # noqa: E402
    DailyChildEventGenerationConfig,
)
from open_llm_vtuber.memory import (  # noqa: E402
    daily_child_event_providers as provider_adapter,
)
from open_llm_vtuber.memory.daily_child_event_providers import (  # noqa: E402
    DailyChildEventProviderError,
)


from open_llm_vtuber.memory import layer2_claude_yaml as claude_transport  # noqa: E402

ExperimentError = legacy.ExperimentError
ManifestDay = legacy.ManifestDay

DEFAULT_END_DATE = date.today() - timedelta(days=1)
DEFAULT_START_DATE = DEFAULT_END_DATE - timedelta(days=6)
DEFAULT_EXPERIMENT_ROOT = Path("tmp/layer2-cloud-experiment")
DEFAULT_SOURCE_DIARIES = legacy.DEFAULT_SOURCE_DIARIES
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "conf.yaml"
SUPPORTED_CONFIG_PROVIDERS = {
    "openai_compatible_llm",
    "deepseek_llm",
}
PROVIDER_DEFAULT_BASE_URLS = {
    "deepseek_llm": "https://api.deepseek.com/v1",
}
BACKGROUND_SCHEMA_SOURCE = TOOLS_DIR / "layer2_background_schema.json"
DELTA_SCHEMA_SOURCE = TOOLS_DIR / "layer2_background_delta_schema.json"
NEUTRAL_PROMPT_SOURCE = (
    SRC_DIR
    / "open_llm_vtuber"
    / "memory"
    / "prompts"
    / "layer2_background_cloud_neutral_v1.txt"
)
RUN_COUNT = 3
RETRY_ATTEMPTS = 3
REQUEST_TIMEOUT_SECONDS = 180.0
MAX_OUTPUT_TOKENS = 4096
DEEPSEEK_THINKING_MAX_OUTPUT_TOKENS = 65536
TEMPERATURE = 0.0
EVIDENCE_QUOTE_TRANSLATION = str.maketrans(
    {
        '"': "‹q›",
        "'": "‹q›",
        "“": "‹q›",
        "”": "‹q›",
        "„": "‹q›",
        "＂": "‹q›",
        "‘": "‹q›",
        "’": "‹q›",
        "‚": "‹q›",
        "＇": "‹q›",
    }
)

# Generic terms are used only to audit the frozen prompt for answer leakage.
PROMPT_LEAK_TERMS = (
    "OCR",
    "Live2D",
    "第一次启动",
    "第一次对话",
    "表情功能",
    "互道晚安",
)


def _runtime_source_hashes() -> dict[str, str]:
    """Hash every repository source file that can change experiment semantics."""
    config_module = sys.modules[DailyChildEventGenerationConfig.__module__]
    paths = (
        Path(__file__).resolve(),
        Path(delta_runner.__file__).resolve(),
        Path(legacy.__file__).resolve(),
        Path(provider_adapter.__file__).resolve(),
        Path(config_module.__file__).resolve(),
    )
    return {
        path.relative_to(REPOSITORY_ROOT).as_posix(): legacy._sha256_file(path)
        for path in paths
    }


def _runtime_source_hash_errors(expected: Any) -> list[str]:
    if not isinstance(expected, dict):
        return ["runtime source hash manifest is missing or invalid"]
    current = _runtime_source_hashes()
    errors: list[str] = []
    for relative_path in sorted(set(expected) | set(current)):
        if expected.get(relative_path) != current.get(relative_path):
            errors.append(f"runtime source changed or missing: {relative_path}")
    return errors


def _safe_base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(str(value or "").strip())
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "")
    )


def _json_type_for_const(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _deepseek_strict_schema(source: dict[str, Any]) -> dict[str, Any]:
    """Translate local Draft 2020-12 structure to DeepSeek strict mode.

    DeepSeek's strict tool dialect does not accept the local ``$defs`` graph
    and accepts a single-value enum more reliably than ``const``.  The result
    is request-only; the unmodified local schema remains the authority used to
    validate every returned delta.
    """

    definitions = source.get("$defs", {})

    def expand(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.rsplit("/", 1)[-1]
                definition = definitions.get(name)
                if not isinstance(definition, dict):
                    raise ExperimentError(f"Unknown local schema reference: {ref}")
                return expand(copy.deepcopy(definition))
            result = {
                key: expand(value)
                for key, value in node.items()
                if key not in {"$schema", "$defs", "title", "const"}
            }
            if "const" in node:
                result.setdefault("type", _json_type_for_const(node["const"]))
                result["enum"] = [copy.deepcopy(node["const"])]
            return result
        if isinstance(node, list):
            return [expand(value) for value in node]
        return copy.deepcopy(node)

    result = expand(source)
    if not isinstance(result, dict):
        raise ExperimentError("Expanded DeepSeek schema root is not an object")
    return result


def _deepseek_beta_chat_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(str(base_url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        raise ExperimentError("DeepSeek base URL is invalid")
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, "/beta/chat/completions", "", "")
    )


def _transport_for_provider(provider: str) -> str:
    if provider == "deepseek_llm":
        return "deepseek_beta_strict_tool_thinking_auto"
    return "forced_function_call"


def _request_url_for_settings(
    settings: DailyChildEventGenerationConfig,
) -> str:
    if settings.llm_provider == "deepseek_llm":
        return _deepseek_beta_chat_url(settings.base_url)
    return provider_adapter._append_endpoint(settings.base_url, "/chat/completions")


def _cloud_request(
    *,
    system_prompt: str,
    task_prompt: str,
    settings: DailyChildEventGenerationConfig,
    schema: dict[str, Any],
    deepseek_reasoning_effort: str = "high",
) -> tuple[dict[str, Any], str, str]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt},
    ]
    if claude_transport.enabled(settings):
        return claude_transport.request(settings, system_prompt, task_prompt)
    if settings.llm_provider == "deepseek_llm":
        payload = {
            "model": settings.model,
            "messages": messages,
            "thinking": {"type": "enabled"},
            "reasoning_effort": deepseek_reasoning_effort,
            "max_tokens": settings.max_output_tokens,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "submit_layer2_delta",
                        "description": "提交完整的当日第二层用户背景差量。",
                        "strict": True,
                        "parameters": _deepseek_strict_schema(schema),
                    },
                }
            ],
        }
        return (
            payload,
            _request_url_for_settings(settings),
            _transport_for_provider(settings.llm_provider),
        )

    payload = {
        "model": settings.model,
        "messages": messages,
        "temperature": settings.temperature,
        "max_tokens": settings.max_output_tokens,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "submit_layer2_delta",
                    "description": "提交当日第二层用户背景差量。",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["delta_json"],
                        "properties": {
                            "delta_json": {
                                "type": "string",
                                "description": (
                                    "严格符合任务中output_json_schema的完整JSON对象文本"
                                ),
                            }
                        },
                    },
                },
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": "submit_layer2_delta"},
        },
    }
    return (
        payload,
        _request_url_for_settings(settings),
        _transport_for_provider(settings.llm_provider),
    )


def _load_cloud_settings(
    config_path: Path,
    provider_name: str | None = None,
    model_override: str | None = None,
    deepseek_max_output_tokens: int = DEEPSEEK_THINKING_MAX_OUTPUT_TOKENS,
) -> tuple[DailyChildEventGenerationConfig, dict[str, Any]]:
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        agent = raw["character_config"]["agent_config"]
        provider = (
            provider_name
            or agent["agent_settings"]["basic_memory_agent"]["llm_provider"]
        )
        provider_config = agent["llm_configs"][provider]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise ExperimentError(
            f"Cannot load selected cloud provider from config: {type(exc).__name__}"
        ) from exc
    if provider not in SUPPORTED_CONFIG_PROVIDERS:
        raise ExperimentError(
            "Neutral cloud pilot requires an OpenAI-compatible configured "
            f"provider, got {provider!r}"
        )
    api_key = str(provider_config.get("llm_api_key", "")).strip()
    if not api_key or api_key in {"z", "default_api_key", "your_api_key_here"}:
        raise ExperimentError("Selected cloud provider has no usable API key")
    values = {
        "enabled": True,
        "llm_provider": provider,
        "base_url": provider_config.get("base_url")
        or PROVIDER_DEFAULT_BASE_URLS.get(provider),
        "proxy_url": (
            provider_config.get("proxy_url") if provider == "deepseek_llm" else None
        ),
        "llm_api_key": api_key,
        "model": model_override or provider_config.get("model"),
        "organization_id": provider_config.get("organization_id"),
        "project_id": provider_config.get("project_id"),
        "temperature": TEMPERATURE,
        "max_output_tokens": (
            deepseek_max_output_tokens
            if provider == "deepseek_llm"
            else MAX_OUTPUT_TOKENS
        ),
        "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "max_concurrent_requests": 1,
        "min_request_interval_seconds": 0.0,
        "num_ctx": 32768,
        "top_p": 1.0,
        "top_k": 0,
        "presence_penalty": 0.0,
        "keep_alive": 0.0,
        "unload_at_exit": False,
    }
    try:
        settings = DailyChildEventGenerationConfig.model_validate(values)
    except Exception as exc:
        raise ExperimentError(
            f"Selected cloud provider config is invalid: {type(exc).__name__}"
        ) from exc
    safe = {
        "llm_provider": settings.llm_provider,
        "model": settings.model,
        "base_url": _safe_base_url(settings.base_url),
        "temperature": settings.temperature,
        "max_output_tokens": settings.max_output_tokens,
        "timeout_seconds": settings.timeout_seconds,
        "api_key_present": True,
        "proxy_enabled": bool(settings.proxy_url),
    }
    return settings, safe


def _audit_prompt_neutrality(prompt: str, diary_texts: list[str]) -> list[str]:
    errors: list[str] = []
    for term in PROMPT_LEAK_TERMS:
        if term.casefold() in prompt.casefold():
            errors.append(f"prompt contains held-out term: {term}")
    for diary_text in diary_texts:
        for line in diary_text.splitlines():
            stripped = line.strip()
            if len(stripped) >= 16 and stripped in prompt:
                errors.append(f"prompt copies a real diary line: {stripped[:40]}")
    return sorted(set(errors))


def _manifest_days(root: Path) -> list[ManifestDay]:
    return legacy._load_manifest(root)


def _review_template(run_index: int, experiment_days: list[date]) -> str:
    rows = "\n".join(
        f"| {day.isoformat()} | 未运行 | 未审查 |  |" for day in experiment_days
    )
    return (
        f"# 云端中性第二层评估 run_{run_index:02d}\n\n"
        "人工结论不得进入模型请求；以raw patch为主要评分对象。\n\n"
        "| 日期 | 运行状态 | 内容结论 | 备注 |\n"
        "| --- | --- | --- | --- |\n"
        f"{rows}\n\n"
        "## 硬失败\n\n"
        "- [ ] 无证据编造\n"
        "- [ ] 无主体倒置\n"
        "- [ ] 无用户计划、身份、时间或关系幻觉\n"
        "- [ ] 证据与操作结构可追溯\n\n"
        "## 语义质量\n\n"
        "- 核心长期事实遗漏：\n"
        "- 一次性细节误收录：\n"
        "- 新增与更新选择：\n"
        "- 当前与历史状态：\n"
        "- 允许继续：\n"
    )


def prepare_experiment(args: argparse.Namespace) -> None:
    root = args.experiment_root.resolve()
    source = args.source_diaries.resolve()
    config_path = args.config_path.resolve()
    if root.exists():
        raise ExperimentError(f"Experiment root already exists: {root}")
    if not source.is_dir():
        raise ExperimentError(f"Diary source directory not found: {source}")
    for asset in (
        BACKGROUND_SCHEMA_SOURCE,
        DELTA_SCHEMA_SOURCE,
        NEUTRAL_PROMPT_SOURCE,
    ):
        if not asset.is_file():
            raise ExperimentError(f"Experiment asset missing: {asset}")
    settings, provider_safe = _load_cloud_settings(
        config_path,
        args.provider_name,
        args.model_override,
        args.deepseek_max_output_tokens,
    )
    experiment_days = legacy._days(args.start_date, args.end_date)

    root.mkdir(parents=True, exist_ok=False)
    frozen_diaries = root / "frozen_inputs" / "diaries"
    frozen_assets = root / "frozen_assets"
    frozen_diaries.mkdir(parents=True)
    frozen_assets.mkdir(parents=True)
    for asset in (
        BACKGROUND_SCHEMA_SOURCE,
        DELTA_SCHEMA_SOURCE,
        NEUTRAL_PROMPT_SOURCE,
    ):
        shutil.copy2(asset, frozen_assets / asset.name)

    manifest_days: list[dict[str, Any]] = []
    for current_day in experiment_days:
        filename = f"diary_{current_day.isoformat()}.txt"
        source_path = source / filename
        if not source_path.is_file():
            manifest_days.append(
                {
                    "date": current_day.isoformat(),
                    "status": "missing",
                    "source_path": str(source_path),
                    "frozen_path": None,
                    "sha256": None,
                    "byte_count": 0,
                    "character_count": 0,
                }
            )
            continue
        data = source_path.read_bytes()
        destination = frozen_diaries / filename
        destination.write_bytes(data)
        manifest_days.append(
            {
                "date": current_day.isoformat(),
                "status": "available",
                "source_path": str(source_path),
                "frozen_path": str(destination.relative_to(root)),
                "sha256": legacy._sha256_bytes(data),
                "byte_count": len(data),
                "character_count": len(data.decode("utf-8")),
            }
        )

    manifest = {
        "experiment": "Rinne layer-two neutral cloud chronological evaluation",
        "created_at": legacy._iso_now(),
        "start_date": args.start_date.isoformat(),
        "end_date": args.end_date.isoformat(),
        "calendar_day_count": len(experiment_days),
        "available_diary_count": sum(
            item["status"] == "available" for item in manifest_days
        ),
        "missing_diary_count": sum(
            item["status"] == "missing" for item in manifest_days
        ),
        "days": manifest_days,
    }
    legacy._write_json_new(root / "input_manifest.json", manifest)

    config = {
        "experiment": "layer2_cloud_neutral_stability_eval",
        "prompt_version": "cloud-neutral-v1",
        "created_at": legacy._iso_now(),
        "private_data": True,
        "repository_head": legacy._git_head(),
        "runner_sha256": legacy._sha256_file(Path(__file__).resolve()),
        "runtime_source_sha256": _runtime_source_hashes(),
        "background_schema_sha256": legacy._sha256_file(
            frozen_assets / BACKGROUND_SCHEMA_SOURCE.name
        ),
        "delta_schema_sha256": legacy._sha256_file(
            frozen_assets / DELTA_SCHEMA_SOURCE.name
        ),
        "neutral_prompt_sha256": legacy._sha256_file(
            frozen_assets / NEUTRAL_PROMPT_SOURCE.name
        ),
        "provider": provider_safe,
        "retry_attempts": RETRY_ATTEMPTS,
        "independent_run_count": RUN_COUNT,
        "config_source": str(config_path),
        "provider_config_name": args.provider_name,
        "model_override": args.model_override,
        "deepseek_reasoning_effort": args.deepseek_reasoning_effort,
        "deepseek_max_output_tokens": args.deepseek_max_output_tokens,
        "rule": (
            "The frozen prompt contains no date-specific answer key. Raw patches are "
            "the primary semantic evidence. Objective validation may quarantine only "
            "operations that cannot be merged safely."
        ),
        "structured_output_transport": _transport_for_provider(
            provider_safe["llm_provider"]
        ),
        "request_endpoint": _safe_base_url(_request_url_for_settings(settings)),
    }
    legacy._write_json_new(root / "experiment_config.json", config)

    for run_index in range(1, RUN_COUNT + 1):
        run_dir = root / "runs" / f"run_{run_index:02d}"
        run_dir.mkdir(parents=True)
        review_dir = root / "reviews" / f"run_{run_index:02d}"
        review_dir.mkdir(parents=True)
        legacy._write_text_new(
            review_dir / "REVIEW.md",
            _review_template(run_index, experiment_days),
        )
    legacy._write_text_new(
        root / "reviews" / "CROSS_RUN_REVIEW.md",
        (
            "# 三次独立运行稳定性审查\n\n"
            "此文件只供人工审查，不会进入任何模型请求。\n\n"
            "- 主体或无证据幻觉：\n"
            "- 核心事实跨运行一致性：\n"
            "- 一次性细节误收录一致性：\n"
            "- 状态更新与重复：\n"
            "- 延迟、Token与费用边界：\n"
            "- 最终是否允许接入：\n"
        ),
    )
    legacy._write_text_new(
        root / "README.md",
        (
            "# 凛祢第二层云端中性稳定性评估\n\n"
            "此目录与正式桌宠隔离。API Key只在调用时从conf.yaml读取，"
            "不会写入本目录。提示词、Schema和日记均冻结并校验哈希。\n\n"
            f"- 模型：{provider_safe['model']}\n"
            f"- 独立运行：{RUN_COUNT}次\n"
            f"- 单次API最多尝试：{RETRY_ATTEMPTS}次\n"
            "- API三次失败后停止，不跳过日期\n"
            "- 程序不按语义关键词过滤模型事实\n"
        ),
    )
    print(f"Prepared neutral cloud experiment at {root}")


def preflight(args: argparse.Namespace, *, print_result: bool = True) -> dict[str, Any]:
    root = args.experiment_root.resolve()
    config = legacy._read_json(root / "experiment_config.json")
    manifest = _manifest_days(root)
    errors: list[str] = []
    errors.extend(_runtime_source_hash_errors(config.get("runtime_source_sha256")))
    checks = (
        (BACKGROUND_SCHEMA_SOURCE.name, "background_schema_sha256"),
        (DELTA_SCHEMA_SOURCE.name, "delta_schema_sha256"),
        (NEUTRAL_PROMPT_SOURCE.name, "neutral_prompt_sha256"),
    )
    for filename, config_key in checks:
        asset = root / "frozen_assets" / filename
        if not asset.is_file() or legacy._sha256_file(asset) != config.get(config_key):
            errors.append(f"frozen asset changed or missing: {filename}")
    diary_texts: list[str] = []
    for item in manifest:
        if item.status != "available":
            continue
        diary_path = root / str(item.frozen_path)
        if not diary_path.is_file() or legacy._sha256_file(diary_path) != item.sha256:
            errors.append(f"frozen diary changed or missing: {item.day}")
            continue
        diary_texts.append(diary_path.read_text(encoding="utf-8"))
    prompt_path = root / "frozen_assets" / NEUTRAL_PROMPT_SOURCE.name
    if prompt_path.is_file():
        errors.extend(
            _audit_prompt_neutrality(
                prompt_path.read_text(encoding="utf-8"), diary_texts
            )
        )
    try:
        _settings, provider_safe = _load_cloud_settings(
            args.config_path.resolve(),
            args.provider_name,
            args.model_override,
            args.deepseek_max_output_tokens,
        )
    except ExperimentError as exc:
        provider_safe = {}
        errors.append(str(exc))
    else:
        frozen_provider = config.get("provider", {})
        for key in ("llm_provider", "model", "base_url", "max_output_tokens"):
            if provider_safe.get(key) != frozen_provider.get(key):
                errors.append(f"cloud provider changed since prepare: {key}")
        if provider_safe.get(
            "llm_provider"
        ) == "deepseek_llm" and args.deepseek_reasoning_effort != config.get(
            "deepseek_reasoning_effort"
        ):
            errors.append("DeepSeek reasoning effort changed since prepare")
    result = {
        "ok": not errors,
        "checked_at": legacy._iso_now(),
        "errors": errors,
        "provider": provider_safe,
        "prompt_neutrality_audit": "passed" if not errors else "failed",
        "api_retry_attempts": config.get("retry_attempts"),
    }
    if print_result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise ExperimentError("Preflight failed: " + "; ".join(errors))
    return result


def seal_experiment(args: argparse.Namespace) -> None:
    root = args.experiment_root.resolve()
    seal_path = root / "PREPARATION_SEAL.json"
    if seal_path.exists():
        raise ExperimentError(f"Preparation seal already exists: {seal_path}")
    preflight_result = preflight(args, print_result=False)
    seal = {
        "sealed_at": legacy._iso_now(),
        "authoritative_repository_head": legacy._git_head(),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": legacy._sha256_file(Path(__file__).resolve()),
        "runtime_source_sha256": _runtime_source_hashes(),
        "preflight": preflight_result,
    }
    legacy._write_json_new(seal_path, seal)
    print(json.dumps(seal, ensure_ascii=False, indent=2))


def _build_task_prompt(
    *,
    current_day: date,
    previous: dict[str, Any],
    diary_text: str,
    schema: dict[str, Any],
) -> str:
    return (
        f"处理日期：{current_day.isoformat()}。\n"
        "请根据上一日紧凑背景与今日材料，只输出今日必要差量。\n\n"
        "<previous_user_background>\n"
        f"{json.dumps(delta_runner._compact_previous_view(previous, review_date=current_day), ensure_ascii=False, indent=2)}\n"
        "</previous_user_background>\n\n"
        "<today_diary>\n"
        f"{diary_text}\n"
        "</today_diary>\n\n"
        "<output_json_schema>\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n"
        "</output_json_schema>\n\n"
        f"source_date必须为{current_day.isoformat()}。"
        "新增、更新以及有当日证据的删除，其证据摘录必须逐字来自今日材料；"
        "基于背景时间线的删除按系统规则输出空证据数组。只输出JSON对象。"
    )


def _schema_errors(value: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(schema)
    errors: list[str] = []
    for error in sorted(validator.iter_errors(value), key=lambda item: list(item.path)):
        location = ".".join(str(part) for part in error.path) or "$"
        errors.append(f"{location}: {error.message}")
    return errors


def _claude_delta_limit_instructions(schema: dict[str, Any]) -> str:
    properties = schema.get("properties", {})
    definitions = schema.get("$defs", {})
    fact_change = definitions.get("fact_change", {}).get("properties", {})
    fact_removal = definitions.get("fact_removal", {}).get("properties", {})
    limits: list[str] = []
    for field in ("added_facts", "updated_facts", "removed_facts"):
        limit = properties.get(field, {}).get("maxItems")
        if isinstance(limit, int):
            limits.append(f"{field} 最多 {limit} 条")
    summary_limit = properties.get("summary", {}).get("maxLength")
    if isinstance(summary_limit, int):
        limits.append(
            f"summary 最多 {summary_limit} 字符，建议控制在 {summary_limit // 2} 字符内"
        )
    for field in ("statement", "notes"):
        limit = fact_change.get(field, {}).get("maxLength")
        if isinstance(limit, int):
            limits.append(
                f"added_facts/updated_facts 每条 {field} 最多 {limit} 字符，"
                f"建议控制在 {limit // 2} 字符内"
            )
    reason_limit = fact_removal.get("reason", {}).get("maxLength")
    if isinstance(reason_limit, int):
        limits.append(
            f"removed_facts 每条 reason 最多 {reason_limit} 字符，"
            f"建议控制在 {reason_limit // 2} 字符内"
        )
    excerpt_schema = fact_change.get("evidence_excerpts", {})
    excerpt_count = excerpt_schema.get("maxItems")
    excerpt_length = excerpt_schema.get("items", {}).get("maxLength")
    if isinstance(excerpt_count, int) and isinstance(excerpt_length, int):
        limits.append(
            f"每条事实的 evidence_excerpts 最多 {excerpt_count} 条、每条最多 "
            f"{excerpt_length} 字符"
        )
    if not limits:
        return ""
    return (
        "\n输出前逐字段检查字符数和条目数。不要把一条事实写成完整日记或"
        "事件流水；statement 只保留可供后续对话使用的当前事实，历史演变放入"
        "必要的简短 notes 或由既有时间线保存。硬性上限如下：\n- " + "\n- ".join(limits)
    )


def _normalize_evidence_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.translate(EVIDENCE_QUOTE_TRANSLATION)
    return " ".join(normalized.split())


def _sanitize_evidence_excerpts(
    patch: dict[str, Any], diary_text: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sanitized = copy.deepcopy(patch)
    discarded: list[dict[str, Any]] = []
    for list_name in ("added_facts", "updated_facts", "removed_facts"):
        for operation in sanitized[list_name]:
            if (
                list_name == "removed_facts"
                and operation.get("removal_basis")
                in {"duplicate_or_invalid", "stale_for_current_context"}
                and operation.get("evidence_type") == "background_timeline"
            ):
                continue
            supported: list[str] = []
            for excerpt in operation["evidence_excerpts"]:
                if excerpt in diary_text:
                    supported.append(excerpt)
                    continue
                discarded.append(
                    {
                        "fact_id": operation["fact_id"],
                        "operation_list": list_name,
                        "excerpt": excerpt,
                        "reason": "excerpt_not_exact_after_safe_quote_repair",
                    }
                )
            operation["evidence_excerpts"] = supported
    return sanitized, discarded


def _prepare_evidence_excerpts(
    patch: dict[str, Any], diary_text: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Repair quote glyphs to source text, then retain only exact excerpts."""

    repaired, repairs = delta_runner._repair_quote_normalized_evidence(
        patch, diary_text
    )
    sanitized, discarded = _sanitize_evidence_excerpts(repaired, diary_text)
    return sanitized, discarded, repairs


def _objective_operation_errors(
    patch: dict[str, Any],
    *,
    current_day: date,
    previous: dict[str, Any],
    diary_text: str,
) -> dict[str, list[str]]:
    previous_map = legacy._categorized_items_by_id(previous)
    errors: dict[str, list[str]] = {}
    # Keep the cloud path subject to the same neutral, atomic operation rules as
    # the local isolated runner.  The cloud-specific checks below add normalized
    # evidence handling and per-operation quarantine.
    for error in delta_runner._validate_delta(
        patch,
        current_day=current_day,
        previous=previous,
        diary_text=diary_text,
        evidence_is_supported=lambda excerpt: (
            _normalize_evidence_text(excerpt) in _normalize_evidence_text(diary_text)
        ),
    ):
        fact_id, separator, _rest = error.partition(":")
        key = fact_id if separator else "$"
        errors.setdefault(key, []).append(error)
    seen: dict[str, str] = {}
    for list_name, must_exist in (
        ("added_facts", False),
        ("updated_facts", True),
        ("removed_facts", True),
    ):
        for operation in patch[list_name]:
            fact_id = operation["fact_id"]
            item_errors = errors.setdefault(fact_id, [])
            if fact_id in seen:
                item_errors.append(f"fact_id also appears in {seen[fact_id]}")
            else:
                seen[fact_id] = list_name
            exists = fact_id in previous_map
            if exists is not must_exist:
                item_errors.append(
                    "fact_id must already exist"
                    if must_exist
                    else "fact_id must be new"
                )
            category = operation["category"]
            if (
                exists
                and list_name == "removed_facts"
                and previous_map[fact_id][0] != category
            ):
                item_errors.append("category differs from previous background")
            timeline_removal = list_name == "removed_facts" and operation.get(
                "removal_basis"
            ) in {"duplicate_or_invalid", "stale_for_current_context"}
            if timeline_removal:
                if operation.get("evidence_type") != "background_timeline":
                    item_errors.append(
                        "timeline removal must use background_timeline evidence"
                    )
                if operation.get("evidence_excerpts"):
                    item_errors.append(
                        "timeline removal must use an empty evidence array"
                    )
            else:
                if not operation["evidence_excerpts"]:
                    item_errors.append("no traceable evidence excerpt remains")
                if operation.get("evidence_type") == "background_timeline":
                    item_errors.append(
                        "current-diary operation cannot use background_timeline"
                    )
                diary_normalized = _normalize_evidence_text(diary_text)
                for excerpt in operation["evidence_excerpts"]:
                    if _normalize_evidence_text(excerpt) not in diary_normalized:
                        item_errors.append("evidence excerpt is not verbatim")
            if list_name != "removed_facts":
                supersedes = operation["supersedes_fact_ids"]
                if any(item not in previous_map for item in supersedes):
                    item_errors.append("supersedes an unknown fact_id")
                if exists:
                    prior_category, prior_item = previous_map[fact_id]
                    if (
                        delta_runner._change_payload(prior_item, prior_category)
                        == operation
                    ):
                        item_errors.append("unchanged fact emitted as update")
            if not item_errors:
                errors.pop(fact_id, None)
    if patch.get("source_date") != current_day.isoformat():
        errors.setdefault("$", []).append("source_date differs from current day")
    return errors


def _prompt_rule_flags(patch: dict[str, Any]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for list_name in ("added_facts", "updated_facts"):
        for operation in patch.get(list_name, []):
            statement = operation.get("statement", "")
            fact_id = operation.get("fact_id")
            if not statement.strip().startswith("用户"):
                flags.append(
                    {"fact_id": fact_id, "flag": "statement_not_user_prefixed"}
                )
            if (
                operation.get("evidence_type") == "diary_interpretation"
                and operation.get("status") != "uncertain"
            ):
                flags.append(
                    {
                        "fact_id": fact_id,
                        "flag": "interpretation_not_marked_uncertain",
                    }
                )
    return flags


def _quarantine_objective_failures(
    patch: dict[str, Any], operation_errors: dict[str, list[str]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    applied = copy.deepcopy(patch)
    quarantined: list[dict[str, Any]] = []
    failed_ids = {fact_id for fact_id in operation_errors if fact_id != "$"}
    for list_name in ("added_facts", "updated_facts", "removed_facts"):
        kept: list[dict[str, Any]] = []
        for operation in applied[list_name]:
            fact_id = operation["fact_id"]
            if fact_id not in failed_ids:
                kept.append(operation)
                continue
            quarantined.append(
                {
                    "fact_id": fact_id,
                    "operation_list": list_name,
                    "proposed_operation": copy.deepcopy(operation),
                    "objective_errors": operation_errors[fact_id],
                }
            )
        applied[list_name] = kept
    applied["summary"] = delta_runner._deterministic_delta_summary(applied)
    return applied, quarantined


def _usage_from_response(response_path: Path) -> dict[str, Any]:
    if not response_path.is_file():
        return {}
    response = legacy._read_json(response_path)
    usage = response.get("usage")
    result = copy.deepcopy(usage) if isinstance(usage, dict) else {}
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        result["finish_reason"] = choices[0].get("finish_reason")
    elif "stop_reason" in response:
        result["finish_reason"] = response["stop_reason"]
    return result


def _tool_arguments_from_response(
    value: dict[str, Any], schema: dict[str, Any] | None = None
) -> str:
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices:
        raise DailyChildEventProviderError("provider_choices_missing")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise DailyChildEventProviderError("provider_message_missing")
    if choices[0].get("finish_reason") == "length":
        raise DailyChildEventProviderError("provider_output_truncated_before_tool_call")
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise DailyChildEventProviderError("provider_tool_call_missing_or_multiple")
    function = tool_calls[0].get("function")
    if not isinstance(function, dict):
        raise DailyChildEventProviderError("provider_tool_function_missing")
    if function.get("name") != "submit_layer2_delta":
        raise DailyChildEventProviderError("provider_tool_name_mismatch")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            argument_object = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise DailyChildEventProviderError(
                "provider_tool_arguments_not_json"
            ) from exc
    elif isinstance(arguments, dict):
        argument_object = arguments
    else:
        raise DailyChildEventProviderError("provider_tool_arguments_missing")
    delta_json = argument_object.get("delta_json")
    if isinstance(delta_json, str) and delta_json.strip():
        return delta_json
    required_delta_keys = (
        set(schema.get("required", []))
        if isinstance(schema, dict) and isinstance(schema.get("required"), list)
        else {
            "schema_version",
            "source_date",
            "added_facts",
            "updated_facts",
            "removed_facts",
            "summary",
        }
    )
    if set(argument_object) == required_delta_keys:
        return json.dumps(argument_object, ensure_ascii=False)
    raise DailyChildEventProviderError("provider_delta_json_missing")


def _generate_cloud_delta_once(
    *,
    system_prompt: str,
    task_prompt: str,
    attempt_dir: Path,
    settings: DailyChildEventGenerationConfig,
    schema: dict[str, Any],
    deepseek_reasoning_effort: str = "high",
) -> str:
    payload, request_url, transport = _cloud_request(
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        settings=settings,
        schema=schema,
        deepseek_reasoning_effort=deepseek_reasoning_effort,
    )
    legacy._write_json_new(
        attempt_dir / "request_metadata.json",
        {
            "llm_provider": settings.llm_provider,
            "model": settings.model,
            "base_url": _safe_base_url(settings.base_url),
            "temperature": settings.temperature,
            "max_output_tokens": settings.max_output_tokens,
            "timeout_seconds": settings.timeout_seconds,
            "structured_output_transport": transport,
            "request_endpoint": _safe_base_url(request_url),
            "deepseek_reasoning_effort": (
                deepseek_reasoning_effort
                if settings.llm_provider == "deepseek_llm"
                else None
            ),
            "request_payload": payload,
            "credential_written": False,
        },
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.llm_api_key}",
    }
    if settings.organization_id:
        headers["OpenAI-Organization"] = settings.organization_id
    if settings.project_id:
        headers["OpenAI-Project"] = settings.project_id
    if claude_transport.enabled(settings):
        headers = claude_transport.headers(settings)
    response = (
        claude_transport.post(settings, payload, attempt_dir / "response.json")
        if claude_transport.enabled(settings)
        else provider_adapter._post_json(
            request_url,
            payload,
            headers,
            settings.timeout_seconds,
            attempt_dir / "response.json",
            proxy_url=settings.proxy_url,
        )
    )
    arguments = (
        json.dumps(claude_transport.parse(response), ensure_ascii=False)
        if claude_transport.enabled(settings)
        else _tool_arguments_from_response(response, schema)
    )
    legacy._write_text_new(attempt_dir / "response.txt", arguments)
    return arguments


def _call_with_retries(
    *,
    current_day: date,
    system_prompt: str,
    task_prompt: str,
    attempts_dir: Path,
    settings: DailyChildEventGenerationConfig,
    schema: dict[str, Any],
    deepseek_reasoning_effort: str = "high",
) -> tuple[str, Path, int, float]:
    errors: list[dict[str, Any]] = []
    started = time.monotonic()
    claude_enabled = claude_transport.enabled(settings)
    repair_base: dict[str, Any] | None = None
    repair_fields: list[str] = []
    repair_violations: list[dict[str, Any]] = []
    full_retry_violations: list[dict[str, Any]] = []
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        attempt_dir = attempts_dir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        legacy._write_json_new(
            attempt_dir / "attempt.json",
            {
                "attempt": attempt,
                "started_at": legacy._iso_now(),
                "date": current_day.isoformat(),
            },
        )
        try:
            request_system_prompt = system_prompt
            request_task_prompt = task_prompt
            request_settings = settings
            if claude_enabled:
                if repair_base is not None:
                    request_system_prompt = (
                        "你是结构化数据局部纠错器。只修复输入指定的顶层字段，"
                        "并且只返回这些字段组成的 YAML 对象。不得返回完整台账或"
                        "其他字段；不得新增事实、证据或推断。除了 violations 指出的"
                        "结构问题外，不要改变原意；优先删除多余证据或压缩冗长表述。"
                        "输出会由程序替换回原台账并重新执行完整校验。"
                    )
                    request_task_prompt = (
                        "<schema_repair>\n"
                        + json.dumps(
                            {
                                "replace_fields": repair_fields,
                                "violations": repair_violations,
                                "previous_values": {
                                    key: repair_base.get(key) for key in repair_fields
                                },
                                "field_schemas": {
                                    key: schema["properties"][key]
                                    for key in repair_fields
                                },
                            },
                            ensure_ascii=False,
                        )
                        + "\n</schema_repair>"
                    )
                    request_settings = settings.model_copy(
                        update={
                            "temperature": 0.0,
                            "max_output_tokens": min(settings.max_output_tokens, 16384),
                        }
                    )
                else:
                    request_system_prompt += _claude_delta_limit_instructions(schema)
                    if full_retry_violations:
                        request_system_prompt += (
                            "\n上次完整对象存在根级结构错误。请重新输出完整对象，"
                            "所有 schema 键名必须逐字使用 ASCII 原名；不得使用"
                            "形似 Unicode 字符，不得遗漏或增加顶层键。"
                        )
                        request_task_prompt += (
                            "\n\n<root_schema_retry>\n"
                            + json.dumps(
                                {"violations": full_retry_violations},
                                ensure_ascii=True,
                            )
                            + "\n</root_schema_retry>"
                        )
                    request_settings = settings.model_copy(
                        update={
                            "max_output_tokens": min(settings.max_output_tokens, 32768)
                        }
                    )
            text = _generate_cloud_delta_once(
                system_prompt=request_system_prompt,
                task_prompt=request_task_prompt,
                attempt_dir=attempt_dir,
                settings=request_settings,
                schema=schema,
                deepseek_reasoning_effort=deepseek_reasoning_effort,
            )
            if claude_enabled:
                try:
                    value = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise claude_transport.ClaudeOutputError(
                        "claude_delta_not_json"
                    ) from exc
                if not isinstance(value, dict):
                    raise claude_transport.ClaudeOutputError(
                        "claude_delta_root_not_object"
                    )
                if repair_base is not None:
                    returned_fields = set(value)
                    complete_fields = set(schema.get("required", []))
                    if returned_fields == set(repair_fields):
                        replacements = value
                    elif complete_fields and returned_fields == complete_fields:
                        replacements = {field: value[field] for field in repair_fields}
                    else:
                        raise claude_transport.ClaudeOutputError(
                            "claude_delta_repair_fields_mismatch"
                        )
                    value = {**repair_base, **replacements}
                schema_failures = list(Draft202012Validator(schema).iter_errors(value))
                if schema_failures:
                    all_violations = []
                    for error in schema_failures:
                        violation = {
                            "path": list(error.absolute_path),
                            "rule": error.validator,
                            "limit": error.validator_value,
                        }
                        if not error.absolute_path and error.validator in {
                            "required",
                            "additionalProperties",
                        }:
                            violation["message"] = error.message
                        all_violations.append(violation)
                    fields = sorted(
                        {
                            str(error.absolute_path[0])
                            for error in schema_failures
                            if error.absolute_path
                        }
                    )
                    if (
                        fields
                        and all(error.absolute_path for error in schema_failures)
                        and all(
                            field in schema.get("properties", {}) for field in fields
                        )
                    ):
                        repair_base = value
                        repair_fields = fields
                        repair_violations = all_violations
                        full_retry_violations = []
                    else:
                        repair_base = None
                        repair_fields = []
                        repair_violations = []
                        full_retry_violations = all_violations
                    raise claude_transport.ClaudeOutputError(
                        "claude_delta_schema_invalid:"
                        + json.dumps(all_violations, ensure_ascii=True)
                    )
                text = json.dumps(value, ensure_ascii=False)
        except (
            DailyChildEventProviderError,
            claude_transport.ClaudeOutputError,
        ) as exc:
            error = {
                "attempt": attempt,
                "failed_at": legacy._iso_now(),
                "error": str(exc),
            }
            errors.append(error)
            legacy._write_json_new(attempt_dir / "error.json", error)
            print(
                f"[api-failure] {current_day} attempt {attempt}/{RETRY_ATTEMPTS}: {exc}",
                flush=True,
            )
            if attempt < RETRY_ATTEMPTS:
                time.sleep(min(2 * attempt, 5))
            continue
        elapsed = time.monotonic() - started
        return text, attempt_dir, attempt, elapsed
    raise ExperimentError(
        "API failed three attempts; stop required: "
        + json.dumps(errors, ensure_ascii=False)
    )


def _load_previous(run_dir: Path, manifest: list[ManifestDay]) -> dict[str, Any]:
    snapshots = sorted((run_dir / "snapshots").glob("20??-??-??"))
    if snapshots:
        return legacy._read_json(snapshots[-1] / "background.json")
    return delta_runner._empty_background(manifest[0].day - timedelta(days=1))


def _next_day_attempts_dir(run_dir: Path, current_day: date) -> Path:
    day_dir = run_dir / "attempts" / current_day.isoformat()
    if not day_dir.exists():
        return day_dir
    for resume_index in range(2, 100):
        candidate = day_dir / f"resume_{resume_index:02d}"
        if not candidate.exists():
            return candidate
    raise ExperimentError(
        f"Too many preserved resume attempts for {current_day.isoformat()}"
    )


def _next_model_failure_path(run_dir: Path, current_day: date) -> Path:
    failure_dir = run_dir / "model_failures"
    base = failure_dir / f"{current_day.isoformat()}.json"
    if not base.exists():
        return base
    for resume_index in range(2, 100):
        candidate = failure_dir / (
            f"{current_day.isoformat()}.resume_{resume_index:02d}.json"
        )
        if not candidate.exists():
            return candidate
    raise ExperimentError(
        f"Too many preserved model failures for {current_day.isoformat()}"
    )


def _save_carry_forward(
    run_dir: Path,
    current_day: date,
    previous: dict[str, Any],
) -> dict[str, Any]:
    snapshot_dir = run_dir / "snapshots" / current_day.isoformat()
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    carried = delta_runner._carry_forward(previous, current_day)
    legacy._write_json_new(snapshot_dir / "background.json", carried)
    legacy._write_text_new(
        snapshot_dir / "background.md",
        delta_runner._render_markdown(carried, "cloud_neutral_carry_forward"),
    )
    model_facing.write_model_facing_artifacts(snapshot_dir, carried)
    legacy._write_json_new(
        snapshot_dir / "metrics.json",
        {
            "date": current_day.isoformat(),
            "model_called": False,
            "generation_mode": "deterministic_carry_forward",
            "completed_at": legacy._iso_now(),
        },
    )
    return carried


def run_experiment(args: argparse.Namespace) -> None:
    root = args.experiment_root.resolve()
    seal = legacy._read_json(root / "PREPARATION_SEAL.json")
    if seal.get("runner_sha256") != legacy._sha256_file(Path(__file__).resolve()):
        raise ExperimentError("Runner changed after preparation seal")
    source_hash_errors = _runtime_source_hash_errors(seal.get("runtime_source_sha256"))
    if source_hash_errors:
        raise ExperimentError(
            "Runtime dependency changed after preparation seal: "
            + "; ".join(source_hash_errors)
        )
    preflight(args, print_result=False)
    config = legacy._read_json(root / "experiment_config.json")
    settings, provider_safe = _load_cloud_settings(
        args.config_path.resolve(),
        args.provider_name,
        args.model_override,
        args.deepseek_max_output_tokens,
    )
    manifest = _manifest_days(root)
    run_dir = root / "runs" / f"run_{args.run_index:02d}"
    snapshots_dir = run_dir / "snapshots"
    snapshots_dir.mkdir(exist_ok=True)
    existing = sorted(snapshots_dir.glob("20??-??-??"))
    if existing and not args.resume:
        raise ExperimentError("Snapshots already exist; use --resume")
    if (run_dir / "RUN_COMPLETE.json").exists():
        raise ExperimentError("This independent run is already complete")

    system_prompt = (root / "frozen_assets" / NEUTRAL_PROMPT_SOURCE.name).read_text(
        encoding="utf-8"
    )
    schema = legacy._read_json(root / "frozen_assets" / DELTA_SCHEMA_SOURCE.name)
    available_days = {
        item.day.isoformat() for item in manifest if item.status == "available"
    }
    previous = _load_previous(run_dir, manifest)
    event_log = run_dir / "run_events.jsonl"
    legacy._append_jsonl(
        event_log,
        {
            "event": "run_started" if not existing else "run_resumed",
            "at": legacy._iso_now(),
            "run_index": args.run_index,
            "model": provider_safe["model"],
            "existing_snapshot_count": len(existing),
            "stop_after": args.stop_after.isoformat() if args.stop_after else None,
        },
    )

    for manifest_day in manifest:
        current_day = manifest_day.day
        snapshot_dir = snapshots_dir / current_day.isoformat()
        if snapshot_dir.exists():
            previous = legacy._read_json(snapshot_dir / "background.json")
            print(f"[skip] {current_day}: snapshot exists", flush=True)
            if args.stop_after == current_day:
                break
            continue
        if manifest_day.status != "available":
            previous = _save_carry_forward(run_dir, current_day, previous)
            print(f"[carry] {current_day}: no diary", flush=True)
            if args.stop_after == current_day:
                break
            continue

        diary_path = root / str(manifest_day.frozen_path)
        diary_text = diary_path.read_text(encoding="utf-8")
        task_prompt = _build_task_prompt(
            current_day=current_day,
            previous=previous,
            diary_text=diary_text,
            schema=schema,
        )
        request_metadata = {
            "model": provider_safe["model"],
            "provider": provider_safe["llm_provider"],
            "temperature": settings.temperature,
            "max_output_tokens": settings.max_output_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_prompt},
            ],
            "structured_output_transport": _transport_for_provider(
                settings.llm_provider
            ),
            "deepseek_reasoning_effort": (
                args.deepseek_reasoning_effort
                if settings.llm_provider == "deepseek_llm"
                else None
            ),
            "credential_written": False,
        }
        attempts_dir = _next_day_attempts_dir(run_dir, current_day)
        attempts_dir.mkdir(parents=True, exist_ok=False)
        print(f"[start] run_{args.run_index:02d} {current_day}", flush=True)
        try:
            text, successful_attempt, attempt_count, elapsed = _call_with_retries(
                current_day=current_day,
                system_prompt=system_prompt,
                task_prompt=task_prompt,
                attempts_dir=attempts_dir,
                settings=settings,
                schema=schema,
                deepseek_reasoning_effort=args.deepseek_reasoning_effort,
            )
        except ExperimentError as exc:
            pause_dir = run_dir / "api_pauses"
            pause_dir.mkdir(parents=True, exist_ok=True)
            pause_path = pause_dir / f"PAUSED_API_{current_day.isoformat()}.json"
            legacy._write_json_new(
                pause_path,
                {
                    "status": "paused_after_three_api_failures",
                    "date": current_day.isoformat(),
                    "paused_at": legacy._iso_now(),
                    "error": str(exc),
                    "next_action": "STOP. Retry only after API connectivity returns.",
                },
            )
            legacy._append_jsonl(
                event_log,
                {
                    "event": "api_paused",
                    "at": legacy._iso_now(),
                    "date": current_day.isoformat(),
                    "pause_path": str(pause_path),
                },
            )
            raise

        try:
            patch = json.loads(text)
        except json.JSONDecodeError as exc:
            failure_path = _next_model_failure_path(run_dir, current_day)
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            legacy._write_json_new(
                failure_path,
                {
                    "failure_kind": "model_output_not_json",
                    "date": current_day.isoformat(),
                    "error": str(exc),
                    "raw_text": text,
                },
            )
            raise ExperimentError(
                f"Model output is not JSON; preserved at {failure_path}"
            ) from exc
        if not isinstance(patch, dict):
            raise ExperimentError("Model JSON root is not an object")
        schema_errors = _schema_errors(patch, schema)
        if schema_errors:
            failure_path = _next_model_failure_path(run_dir, current_day)
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            legacy._write_json_new(
                failure_path,
                {
                    "failure_kind": "model_output_schema_invalid",
                    "date": current_day.isoformat(),
                    "schema_errors": schema_errors,
                    "patch": patch,
                },
            )
            raise ExperimentError(
                f"Model output failed JSON Schema; preserved at {failure_path}"
            )

        evidence_sanitized_patch, discarded_evidence, evidence_repairs = (
            _prepare_evidence_excerpts(patch, diary_text)
        )
        objective_errors = _objective_operation_errors(
            evidence_sanitized_patch,
            current_day=current_day,
            previous=previous,
            diary_text=diary_text,
        )
        if "$" in objective_errors:
            raise ExperimentError(
                "Patch has an unmergeable root error: "
                + "; ".join(objective_errors["$"])
            )
        applied_patch, quarantined = _quarantine_objective_failures(
            evidence_sanitized_patch, objective_errors
        )
        background, merge_audit = delta_runner._merge_delta(
            previous, applied_patch, current_day
        )
        background_errors = delta_runner._validate_background(
            background,
            expected_day=current_day,
            available_days=available_days,
            previous=previous,
        )
        if background_errors:
            raise ExperimentError(
                "Merged background failed validation: " + "; ".join(background_errors)
            )

        snapshot_dir.mkdir(parents=True, exist_ok=False)
        legacy._write_json_new(snapshot_dir / "request.json", request_metadata)
        legacy._write_text_new(snapshot_dir / "raw_response.txt", text)
        response_json = successful_attempt / "response.json"
        raw_response_artifact: dict[str, Any] = {"tool_arguments_text": text}
        if response_json.is_file():
            shutil.copy2(response_json, snapshot_dir / "provider_response.json")
            raw_response_artifact["provider_response"] = legacy._read_json(
                response_json
            )
        legacy._write_json_new(
            snapshot_dir / "raw_response.json", raw_response_artifact
        )
        legacy._write_json_new(snapshot_dir / "patch.json", patch)
        legacy._write_json_new(snapshot_dir / "applied_patch.json", applied_patch)
        merge_audit.update(
            {
                "run_index": args.run_index,
                "provider": provider_safe,
                "model_patch_sha256": delta_runner._sha256_json(patch),
                "evidence_prepared_patch_sha256": delta_runner._sha256_json(
                    evidence_sanitized_patch
                ),
                "evidence_repairs": evidence_repairs,
                "quarantined_operations": quarantined,
                "discarded_evidence_excerpts": discarded_evidence,
                "semantic_filtering_applied": False,
            }
        )
        legacy._write_json_new(snapshot_dir / "merge_audit.json", merge_audit)
        legacy._write_json_new(snapshot_dir / "background.json", background)
        legacy._write_text_new(
            snapshot_dir / "background.md",
            delta_runner._render_markdown(background, "cloud_neutral_delta"),
        )
        model_facing.write_model_facing_artifacts(snapshot_dir, background)
        usage = _usage_from_response(response_json)
        metrics = {
            "date": current_day.isoformat(),
            "model_called": True,
            "run_index": args.run_index,
            "model": provider_safe["model"],
            "temperature": settings.temperature,
            "elapsed_wall_seconds": round(elapsed, 3),
            "api_attempt_count": attempt_count,
            "usage": usage,
            "model_operation_counts": {
                "added": len(patch["added_facts"]),
                "updated": len(patch["updated_facts"]),
                "removed": len(patch["removed_facts"]),
            },
            "applied_operation_counts": {
                "added": len(applied_patch["added_facts"]),
                "updated": len(applied_patch["updated_facts"]),
                "removed": len(applied_patch["removed_facts"]),
            },
            "objective_quarantine_count": len(quarantined),
            "discarded_evidence_excerpt_count": len(discarded_evidence),
            "evidence_repair_count": len(evidence_repairs),
            "prompt_rule_flags": _prompt_rule_flags(patch),
            "semantic_filtering_applied": False,
            "result_fact_count": len(legacy._items_by_id(background)),
            "completed_at": legacy._iso_now(),
        }
        legacy._write_json_new(snapshot_dir / "metrics.json", metrics)
        previous = background
        legacy._append_jsonl(
            event_log,
            {
                "event": "day_completed",
                "at": legacy._iso_now(),
                "date": current_day.isoformat(),
                "elapsed_wall_seconds": round(elapsed, 3),
                "api_attempt_count": attempt_count,
                "objective_quarantine_count": len(quarantined),
                "result_fact_count": metrics["result_fact_count"],
            },
        )
        print(
            f"[done] run_{args.run_index:02d} {current_day}: "
            f"{elapsed:.1f}s, facts={metrics['result_fact_count']}, "
            f"quarantined={len(quarantined)}",
            flush=True,
        )
        if args.stop_after == current_day:
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = (
                checkpoint_dir / f"PAUSED_AFTER_{current_day.isoformat()}.json"
            )
            legacy._write_json_new(
                checkpoint_path,
                {
                    "status": "paused_for_human_review",
                    "run_index": args.run_index,
                    "through_date": current_day.isoformat(),
                    "paused_at": legacy._iso_now(),
                    "next_action": "Review raw patches before resume.",
                },
            )
            print(f"PAUSED_AFTER: {current_day}", flush=True)
            return

    final_day = manifest[-1].day
    if (snapshots_dir / final_day.isoformat() / "background.json").is_file():
        legacy._write_json_new(
            run_dir / "RUN_COMPLETE.json",
            {
                "status": "complete",
                "run_index": args.run_index,
                "completed_at": legacy._iso_now(),
                "through_date": final_day.isoformat(),
                "model": config["provider"]["model"],
                "next_action": "Human semantic review required.",
            },
        )
        print(f"RUN_COMPLETE: run_{args.run_index:02d}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--provider-name",
        choices=sorted(SUPPORTED_CONFIG_PROVIDERS),
        help=(
            "Read this existing llm_configs block without changing conf.yaml; "
            "by default use the provider selected by basic_memory_agent."
        ),
    )
    parser.add_argument(
        "--deepseek-reasoning-effort",
        choices=("low", "high", "max"),
        default="high",
        help=(
            "DeepSeek thinking effort frozen for this experiment; current V4 "
            "maps low to high and xhigh to max."
        ),
    )
    parser.add_argument(
        "--model-override",
        help="Experiment-only model name; never edits conf.yaml.",
    )
    parser.add_argument(
        "--deepseek-max-output-tokens",
        type=int,
        default=DEEPSEEK_THINKING_MAX_OUTPUT_TOKENS,
        help="Experiment-only DeepSeek output budget frozen at prepare time.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--source-diaries", type=Path, default=DEFAULT_SOURCE_DIARIES
    )
    prepare_parser.add_argument(
        "--start-date", type=date.fromisoformat, default=DEFAULT_START_DATE
    )
    prepare_parser.add_argument(
        "--end-date", type=date.fromisoformat, default=DEFAULT_END_DATE
    )
    prepare_parser.set_defaults(handler=prepare_experiment)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.set_defaults(handler=preflight)

    seal_parser = subparsers.add_parser("seal")
    seal_parser.set_defaults(handler=seal_experiment)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--run-index", type=int, choices=range(1, RUN_COUNT + 1), required=True
    )
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--stop-after", type=date.fromisoformat)
    run_parser.set_defaults(handler=run_experiment)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        args.handler(args)
    except ExperimentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
