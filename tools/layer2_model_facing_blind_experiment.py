"""Run one isolated blind model-facing Layer-2 projection with DeepSeek.

The runner reads a frozen background snapshot and the live credential only at
request time.  It never edits the desktop-pet runtime, chat history, diary, or
configuration.  The human reference answer is intentionally not an input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

import layer2_background_experiment as legacy
import layer2_cloud_background_experiment as cloud_runner
from open_llm_vtuber.memory import daily_child_event_providers as provider_adapter
from open_llm_vtuber.memory.daily_child_event_providers import (
    DailyChildEventProviderError,
)


TOOLS_DIR = Path(__file__).resolve().parent
PROMPT_PATH = (
    TOOLS_DIR.parent
    / "src"
    / "open_llm_vtuber"
    / "memory"
    / "prompts"
    / "layer2_model_facing_blind_v1.txt"
)
SCHEMA_PATH = TOOLS_DIR / "layer2_model_facing_blind_schema.json"
TOOL_NAME = "submit_layer2_model_facing_overview"
SCHEMA_VERSION = "1.0-model-facing-overview-blind"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_MAX_OUTPUT_TOKENS = 65536
RETRY_ATTEMPTS = 3
DATE_TOKEN_RE = re.compile(
    r"(?:20\d{2}[年./-]\d{1,2}(?:[月./-]\d{1,2}日?)?|\d{1,2}月\d{1,2}日)"
)


class BlindProjectionError(RuntimeError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _compact_facts(background: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for section in background["sections"]:
        for item in section["items"]:
            result.append(
                {
                    "fact_id": item["fact_id"],
                    "category": section["category"],
                    "statement": item["statement"],
                    "status": item["status"],
                    "subject_scope": item["subject_scope"],
                    "temporal_class": item["temporal_class"],
                    "context_bucket": item["context_bucket"],
                    "expected_end_on": item["expected_end_on"],
                    "review_after": item["review_after"],
                    "first_observed_on": item["first_observed_on"],
                    "last_updated_on": item["last_updated_on"],
                }
            )
    return result


def _build_task_prompt(
    *, review_date: date, background: dict[str, Any], schema: dict[str, Any]
) -> str:
    facts = _compact_facts(background)
    return (
        f"审查日期：{review_date.isoformat()}\n"
        f"输入事实数：{len(facts)}\n"
        "以下事实账本是唯一允许使用的信息：\n"
        "<background_facts>\n"
        f"{json.dumps(facts, ensure_ascii=False, indent=2)}\n"
        "</background_facts>\n\n"
        "<output_json_schema>\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n"
        "</output_json_schema>\n\n"
        f"as_of_date必须为{review_date.isoformat()}。通过指定工具提交一次完整对象。"
    )


def _neutrality_audit(
    system_prompt: str, schema_text: str, background: dict[str, Any]
) -> dict[str, Any]:
    prompt_and_schema = system_prompt + "\n" + schema_text
    copied_statements = []
    copied_fact_ids = []
    for fact in _compact_facts(background):
        statement = fact["statement"].strip()
        if len(statement) >= 16 and statement in prompt_and_schema:
            copied_statements.append(fact["fact_id"])
        if fact["fact_id"] in prompt_and_schema:
            copied_fact_ids.append(fact["fact_id"])
    return {
        "human_reference_supplied": False,
        "personal_fact_selection_rules_present": False,
        "copied_statement_fact_ids": sorted(copied_statements),
        "copied_fact_ids": sorted(copied_fact_ids),
        "passed": not copied_statements and not copied_fact_ids,
    }


def _request_payload(
    *,
    system_prompt: str,
    task_prompt: str,
    settings: Any,
    schema: dict[str, Any],
    reasoning_effort: str,
) -> tuple[dict[str, Any], str]:
    request_url = cloud_runner._request_url_for_settings(settings)
    payload = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ],
        "thinking": {"type": "enabled"},
        "reasoning_effort": reasoning_effort,
        "max_tokens": settings.max_output_tokens,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": TOOL_NAME,
                    "description": "提交当前个人背景的模型阅读版与生命周期审查建议。",
                    "strict": True,
                    "parameters": cloud_runner._deepseek_strict_schema(schema),
                },
            }
        ],
    }
    return payload, request_url


def _tool_arguments(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise DailyChildEventProviderError("provider_choices_missing")
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        raise DailyChildEventProviderError("provider_output_truncated_before_tool_call")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise DailyChildEventProviderError("provider_message_missing")
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise DailyChildEventProviderError("provider_tool_call_missing_or_multiple")
    function = tool_calls[0].get("function")
    if not isinstance(function, dict) or function.get("name") != TOOL_NAME:
        raise DailyChildEventProviderError("provider_tool_name_mismatch")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            value = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise DailyChildEventProviderError(
                "provider_tool_arguments_not_json"
            ) from exc
    elif isinstance(arguments, dict):
        value = arguments
    else:
        raise DailyChildEventProviderError("provider_tool_arguments_missing")
    if not isinstance(value, dict):
        raise DailyChildEventProviderError("provider_tool_arguments_not_object")
    return value


def _schema_errors(value: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(schema)
    return [
        f"{'/'.join(str(part) for part in error.absolute_path) or '$'}: {error.message}"
        for error in sorted(
            validator.iter_errors(value), key=lambda item: list(item.path)
        )
    ]


def _semantic_errors(
    value: dict[str, Any], background: dict[str, Any], review_date: date
) -> list[str]:
    facts = {item["fact_id"]: item for item in _compact_facts(background)}
    errors: list[str] = []
    if value.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if value.get("as_of_date") != review_date.isoformat():
        errors.append("as_of_date mismatch")
    section_categories: list[str] = []
    for section in value.get("overview_sections", []):
        category = section.get("category")
        if isinstance(category, str):
            section_categories.append(category)
        for bullet in section.get("bullets", []):
            for fact_id in bullet.get("source_fact_ids", []):
                fact = facts.get(fact_id)
                if fact is None:
                    errors.append(f"unknown overview source fact_id: {fact_id}")
                elif fact.get("context_bucket") != "current":
                    errors.append(f"overview cites non-current fact_id: {fact_id}")
    duplicates = [
        category for category, count in Counter(section_categories).items() if count > 1
    ]
    for category in sorted(duplicates):
        errors.append(f"duplicate overview section category: {category}")
    lifecycle_ids: list[str] = []
    for operation in value.get("lifecycle_changes", []):
        fact_id = operation.get("fact_id")
        if not isinstance(fact_id, str):
            continue
        lifecycle_ids.append(fact_id)
        fact = facts.get(fact_id)
        if fact is None:
            errors.append(f"unknown lifecycle fact_id: {fact_id}")
        elif fact.get("context_bucket") != "current":
            errors.append(f"lifecycle change targets non-current fact_id: {fact_id}")
    for fact_id, count in Counter(lifecycle_ids).items():
        if count > 1:
            errors.append(f"duplicate lifecycle fact_id: {fact_id}")
    return sorted(set(errors))


def _call_once(
    *,
    attempt_dir: Path,
    system_prompt: str,
    task_prompt: str,
    settings: Any,
    schema: dict[str, Any],
    reasoning_effort: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, request_url = _request_payload(
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        settings=settings,
        schema=schema,
        reasoning_effort=reasoning_effort,
    )
    _write_json(
        attempt_dir / "request_metadata.json",
        {
            "model": settings.model,
            "provider": settings.llm_provider,
            "base_url": cloud_runner._safe_base_url(settings.base_url),
            "request_endpoint": cloud_runner._safe_base_url(request_url),
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": settings.max_output_tokens,
            "structured_output_transport": "deepseek_beta_strict_tool_thinking",
            "credential_written": False,
            "request_payload": payload,
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
    response = provider_adapter._post_json(
        request_url,
        payload,
        headers,
        settings.timeout_seconds,
        attempt_dir / "response.json",
        proxy_url=settings.proxy_url,
    )
    result = _tool_arguments(response)
    _write_json(attempt_dir / "tool_arguments.json", result)
    return result, response


def _call_with_retries(
    *,
    attempts_dir: Path,
    system_prompt: str,
    task_prompt: str,
    settings: Any,
    schema: dict[str, Any],
    background: dict[str, Any],
    review_date: date,
    reasoning_effort: str,
) -> tuple[dict[str, Any], dict[str, Any], Path, int, float]:
    started = time.monotonic()
    failures: list[dict[str, Any]] = []
    for attempt_number in range(1, RETRY_ATTEMPTS + 1):
        attempt_dir = attempts_dir / f"attempt_{attempt_number:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        _write_json(
            attempt_dir / "attempt.json",
            {
                "attempt": attempt_number,
                "review_date": review_date.isoformat(),
                "started_at": legacy._iso_now(),
            },
        )
        try:
            result, response = _call_once(
                attempt_dir=attempt_dir,
                system_prompt=system_prompt,
                task_prompt=task_prompt,
                settings=settings,
                schema=schema,
                reasoning_effort=reasoning_effort,
            )
            hard_errors = _schema_errors(result, schema) + _semantic_errors(
                result, background, review_date
            )
            if hard_errors:
                raise BlindProjectionError("; ".join(hard_errors))
        except (DailyChildEventProviderError, BlindProjectionError) as exc:
            failure = {
                "attempt": attempt_number,
                "failed_at": legacy._iso_now(),
                "error": f"{type(exc).__name__}:{exc}",
            }
            failures.append(failure)
            _write_json(attempt_dir / "error.json", failure)
            if attempt_number < RETRY_ATTEMPTS:
                time.sleep(min(attempt_number * 2, 5))
            continue
        return (
            result,
            response,
            attempt_dir,
            attempt_number,
            time.monotonic() - started,
        )
    raise BlindProjectionError(
        "blind projection failed three attempts: "
        + json.dumps(failures, ensure_ascii=False)
    )


def _render_overview(value: dict[str, Any], *, include_sources: bool) -> str:
    chunks: list[str] = []
    for section in value["overview_sections"]:
        chunks.append(f"### {section['category']}\n")
        for bullet in section["bullets"]:
            chunks.append(f"- {bullet['text']}\n")
            if include_sources:
                chunks.append(
                    "  <!-- source_fact_ids: "
                    + ", ".join(bullet["source_fact_ids"])
                    + " -->\n"
                )
        chunks.append("\n")
    return "".join(chunks).rstrip() + "\n"


def _metrics(
    *,
    value: dict[str, Any],
    overview: str,
    background: dict[str, Any],
    response: dict[str, Any],
    attempt_count: int,
    elapsed: float,
) -> dict[str, Any]:
    bullets = [
        bullet
        for section in value["overview_sections"]
        for bullet in section["bullets"]
    ]
    source_ids = {
        fact_id for bullet in bullets for fact_id in bullet["source_fact_ids"]
    }
    current_count = sum(
        1 for fact in _compact_facts(background) if fact["context_bucket"] == "current"
    )
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    return {
        "overview_bullet_count": len(bullets),
        "overview_character_count": len(overview.rstrip("\r\n")),
        "overview_line_count": len(overview.splitlines()),
        "overview_sha256": _sha256_bytes(overview.encode("utf-8")),
        "exact_date_token_count": len(DATE_TOKEN_RE.findall(overview)),
        "unique_source_fact_count": len(source_ids),
        "current_input_fact_count": current_count,
        "source_fact_coverage_ratio": round(len(source_ids) / current_count, 4),
        "lifecycle_change_count": len(value["lifecycle_changes"]),
        "lifecycle_action_counts": dict(
            sorted(
                Counter(item["action"] for item in value["lifecycle_changes"]).items()
            )
        ),
        "api_attempt_count": attempt_count,
        "elapsed_wall_seconds": round(elapsed, 3),
        "usage": usage,
    }


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    input_path = args.input_background.resolve()
    background = json.loads(input_path.read_text(encoding="utf-8"))
    prompt_text = PROMPT_PATH.read_text(encoding="utf-8")
    schema_text = SCHEMA_PATH.read_text(encoding="utf-8")
    schema = json.loads(schema_text)
    neutrality = _neutrality_audit(prompt_text, schema_text, background)
    if not neutrality["passed"]:
        _write_json(output_dir / "neutrality_failure.json", neutrality)
        raise BlindProjectionError(
            "neutral prompt contains frozen-input answer leakage"
        )

    shutil.copy2(input_path, output_dir / "input_background.json")
    shutil.copy2(PROMPT_PATH, output_dir / "system_prompt.txt")
    shutil.copy2(SCHEMA_PATH, output_dir / "output_schema.json")
    _write_json(output_dir / "neutrality_audit.json", neutrality)
    task_prompt = _build_task_prompt(
        review_date=args.review_date,
        background=background,
        schema=schema,
    )
    (output_dir / "task_prompt.txt").write_text(
        task_prompt, encoding="utf-8", newline="\n"
    )
    _write_json(
        output_dir / "seal.json",
        {
            "created_at": legacy._iso_now(),
            "input_background_sha256": _sha256_file(
                output_dir / "input_background.json"
            ),
            "system_prompt_sha256": _sha256_file(output_dir / "system_prompt.txt"),
            "task_prompt_sha256": _sha256_file(output_dir / "task_prompt.txt"),
            "output_schema_sha256": _sha256_file(output_dir / "output_schema.json"),
            "human_reference_supplied": False,
        },
    )

    settings, safe_provider = cloud_runner._load_cloud_settings(
        args.config_path.resolve(),
        "deepseek_llm",
        args.model,
        args.max_output_tokens,
    )
    _write_json(
        output_dir / "experiment_config.json",
        {
            "review_date": args.review_date.isoformat(),
            "provider": safe_provider,
            "reasoning_effort": args.reasoning_effort,
            "retry_attempts": RETRY_ATTEMPTS,
            "input_background": str(input_path),
            "formal_runtime_modified": False,
        },
    )
    attempts_dir = output_dir / "attempts"
    attempts_dir.mkdir()
    result, response, successful_attempt, attempt_count, elapsed = _call_with_retries(
        attempts_dir=attempts_dir,
        system_prompt=prompt_text,
        task_prompt=task_prompt,
        settings=settings,
        schema=schema,
        background=background,
        review_date=args.review_date,
        reasoning_effort=args.reasoning_effort,
    )
    _write_json(output_dir / "result.json", result)
    overview = _render_overview(result, include_sources=False)
    overview_with_sources = _render_overview(result, include_sources=True)
    (output_dir / "model_facing_overview.md").write_text(
        overview, encoding="utf-8", newline="\n"
    )
    (output_dir / "projection_with_sources.md").write_text(
        overview_with_sources, encoding="utf-8", newline="\n"
    )
    _write_json(output_dir / "lifecycle_changes.json", result["lifecycle_changes"])
    metrics = _metrics(
        value=result,
        overview=overview,
        background=background,
        response=response,
        attempt_count=attempt_count,
        elapsed=elapsed,
    )
    metrics["successful_attempt_dir"] = str(successful_attempt)
    _write_json(output_dir / "metrics.json", metrics)
    (output_dir / "manual_review.md").write_text(
        "# DeepSeek 模型侧个人背景盲测人工审查\n\n"
        "本文件只记录审查结论，不会反馈给本轮模型请求。\n\n"
        "- 生命周期：是否仍把已结束事项描述为正在进行？\n"
        "- 当前价值：是否遗漏身份、留学、当前项目、关系承诺、边界和负面约束？\n"
        "- 日期：是否只保留了真正改变当前含义的日期？\n"
        "- 合并：是否用最新状态替代了逐日过程？\n"
        "- 忠实度：是否存在无依据推断、主体颠倒或凭空新增？\n"
        "- 长度：是否适合作为每轮直接注入背景？\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_json(
        output_dir / "RUN_COMPLETE.json",
        {
            "completed_at": legacy._iso_now(),
            "status": "complete",
            "api_attempt_count": attempt_count,
            "model_facing_overview_sha256": metrics["overview_sha256"],
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-background", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-date", type=date.fromisoformat, required=True)
    parser.add_argument("--config-path", type=Path, default=Path("conf.yaml"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "high", "max"),
        default=DEFAULT_REASONING_EFFORT,
    )
    parser.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS
    )
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
