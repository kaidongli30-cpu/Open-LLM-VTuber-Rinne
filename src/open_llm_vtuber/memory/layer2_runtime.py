"""Daily cloud update and atomic publication for Rinne Layer-2 memory.

The fact ledger still reuses the frozen evaluation helpers. The model-facing
view is a rolling document: the last actually saved view is the base, the first
model call authorizes factual changes, and the second call can only propose
per-item wording operations. Invalid individual operations are ignored, while a
complete second-call failure falls back to the verified first-call facts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterator

from jsonschema import Draft202012Validator

from ..config_manager import read_yaml, validate_config
from ..config_manager.layer2_memory import Layer2MemoryGenerationConfig
from ..data_paths import resolve_character_history_root
from . import daily_child_event_providers as provider_adapter
from .daily_child_event_providers import DailyChildEventProviderError
from .layer2_context import (
    CURRENT_CONTEXT_NAME,
    CURRENT_POINTER_NAME,
    LAYER2_DIRECTORY_NAME,
    format_layer2_context,
    load_current_layer2_context,
    load_layer2_publication,
    sha256_file,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = REPOSITORY_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import layer2_background_delta_experiment as delta_runner  # noqa: E402
import layer2_background_experiment as legacy  # noqa: E402
import layer2_cloud_background_experiment as cloud_runner  # noqa: E402
import layer2_model_facing_blind_experiment as projection_runner  # noqa: E402


PROMPT_ROOT = Path(__file__).resolve().parent / "prompts"
LEDGER_PROMPT_PATH = PROMPT_ROOT / "layer2_background_cloud_neutral_v1.txt"
PROJECTION_PROMPT_PATH = PROMPT_ROOT / "layer2_model_facing_blind_v1.txt"
ROLLING_PROJECTION_PROMPT_PATH = (
    PROMPT_ROOT / "layer2_model_facing_rolling_patch_v1.txt"
)
LEDGER_SCHEMA_PATH = TOOLS_DIR / "layer2_background_delta_schema.json"
PROJECTION_SCHEMA_PATH = TOOLS_DIR / "layer2_model_facing_blind_schema.json"
ROLLING_PROJECTION_SCHEMA_PATH = (
    TOOLS_DIR / "layer2_model_facing_rolling_patch_schema.json"
)
POINTER_SCHEMA_VERSION = "1.0-layer2-current-pointer"
PUBLICATION_SCHEMA_VERSION = "1.0-layer2-runtime-publication"
ROLLING_PROJECTION_SCHEMA_VERSION = "1.0-model-facing-rolling-patch"
ROLLING_PROJECTION_TOOL_NAME = "submit_layer2_model_facing_patch"


class Layer2RuntimeError(RuntimeError):
    pass


class Layer2RunAlreadyActive(Layer2RuntimeError):
    pass


@dataclass(frozen=True)
class Layer2WorkerLaunch:
    process: subprocess.Popen[bytes]
    memory_day: str
    result_path: Path


def _operation_count(patch: dict[str, Any]) -> int:
    """Return the total number of top-level ledger delta operations."""

    return sum(
        len(patch.get(field, [])) if isinstance(patch.get(field), list) else 0
        for field in ("added_facts", "updated_facts", "removed_facts")
    )


def _is_only_operation_limit_error(
    objective_errors: dict[str, list[str]],
) -> bool:
    """Allow one rewrite only for the cross-field total-operation ceiling."""

    if set(objective_errors) != {"$"}:
        return False
    root_errors = objective_errors.get("$", [])
    if len(root_errors) != 1:
        return False
    message = root_errors[0]
    return message.startswith("delta contains ") and message.endswith(
        f"maximum is {delta_runner.MAX_TOTAL_OPERATIONS}"
    )


def _build_operation_limit_repair_prompt(
    *,
    task_prompt: str,
    rejected_patch: dict[str, Any],
    violation: str,
) -> str:
    """Request a full replacement while preserving the frozen evidence input."""

    repair_instruction = {
        "violation": violation,
        "required_action": (
            "重新取舍并输出一个完整替代对象；added_facts、updated_facts、"
            "removed_facts 三个数组合计不得超过 "
            f"{delta_runner.MAX_TOTAL_OPERATIONS} 项。"
        ),
        "prohibited_actions": [
            "不得只返回被删除的一项",
            "不得要求调用方静默截断",
            "不得改变证据忠实性、日期或其他 schema 约束",
        ],
        "rejected_candidate": rejected_patch,
    }
    return (
        task_prompt
        + "\n\n<operation_limit_repair>\n"
        + json.dumps(repair_instruction, ensure_ascii=False)
        + "\n</operation_limit_repair>"
    )


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_json(temporary, value)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(value)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Layer2RuntimeError(f"json_root_not_object:{path}")
    return value


def _manual_source_id(category: str, text: str) -> str:
    digest = hashlib.sha256(f"{category}\0{text}".encode("utf-8")).hexdigest()[:16]
    return f"manual-{digest}"


def _extract_model_facing_overview(context: str) -> str:
    start_marker = "<current_user_background>"
    end_marker = "</current_user_background>"
    start = context.find(start_marker)
    end = context.rfind(end_marker)
    if start >= 0 and end > start:
        return context[start + len(start_marker) : end].strip()
    return context.strip()


def _parse_overview_sections(overview: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_section: dict[str, Any] | None = None
    current_bullet: dict[str, Any] | None = None
    categories = set(legacy.CATEGORIES)
    for raw_line in overview.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("### "):
            category = stripped[4:].strip()
            if category not in categories:
                raise Layer2RuntimeError(f"rolling_base_unknown_category:{category}")
            current_section = {"category": category, "bullets": []}
            sections.append(current_section)
            current_bullet = None
            continue
        if stripped.startswith("- "):
            if current_section is None:
                raise Layer2RuntimeError("rolling_base_bullet_before_section")
            text = stripped[2:].strip()
            if not text:
                raise Layer2RuntimeError("rolling_base_empty_bullet")
            current_bullet = {"text": text, "source_fact_ids": []}
            current_section["bullets"].append(current_bullet)
            continue
        if current_bullet is None:
            raise Layer2RuntimeError("rolling_base_unrecognized_content")
        current_bullet["text"] += "\n" + stripped
    if not sections or not any(section["bullets"] for section in sections):
        raise Layer2RuntimeError("rolling_base_has_no_bullets")
    return sections


def _projection_sections(
    value: dict[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(value, dict):
        return result
    for section in value.get("overview_sections", []):
        if not isinstance(section, dict):
            continue
        category = section.get("category")
        bullets = section.get("bullets")
        if isinstance(category, str) and isinstance(bullets, list):
            result[category] = [
                bullet for bullet in bullets if isinstance(bullet, dict)
            ]
    return result


def _attach_previous_sources(
    sections: list[dict[str, Any]], previous_projection: dict[str, Any] | None
) -> None:
    """Map the prior publication's provenance onto the final saved wording.

    The editable file may contain user rewrites or inserted bullets. Equal text is
    matched exactly; one-for-one replacements keep the old source mapping. Any
    genuinely new bullet receives a stable local source id and is still carried
    forward even though it did not originate in the structured ledger.
    """

    prior_by_category = _projection_sections(previous_projection)
    for section in sections:
        category = section["category"]
        current = section["bullets"]
        prior = prior_by_category.get(category, [])
        old_texts = [str(item.get("text", "")) for item in prior]
        new_texts = [str(item["text"]) for item in current]
        matcher = SequenceMatcher(None, old_texts, new_texts, autojunk=False)
        for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
            if tag == "equal":
                pairs = zip(range(old_start, old_end), range(new_start, new_end))
            elif tag == "replace" and (old_end - old_start) == (new_end - new_start):
                pairs = zip(range(old_start, old_end), range(new_start, new_end))
            else:
                pairs = ()
            for old_index, new_index in pairs:
                source_ids = prior[old_index].get("source_fact_ids", [])
                if isinstance(source_ids, list):
                    current[new_index]["source_fact_ids"] = [
                        item for item in source_ids if isinstance(item, str)
                    ]
        for bullet in current:
            if not bullet["source_fact_ids"]:
                bullet["source_fact_ids"] = [
                    _manual_source_id(category, bullet["text"])
                ]


def _number_overview_bullets(
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    numbered: list[dict[str, Any]] = []
    counter = 1
    for section in sections:
        for bullet in section["bullets"]:
            bullet["bullet_id"] = f"b{counter:04d}"
            numbered.append(
                {
                    "bullet_id": bullet["bullet_id"],
                    "category": section["category"],
                    "text": bullet["text"],
                    "source_fact_ids": copy.deepcopy(bullet["source_fact_ids"]),
                }
            )
            counter += 1
    return numbered


def _render_overview_sections(sections: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for section in sections:
        bullets = section["bullets"]
        if not bullets:
            continue
        chunks.append(f"### {section['category']}\n")
        for bullet in bullets:
            chunks.append(f"- {bullet['text']}\n")
        chunks.append("\n")
    return "".join(chunks).rstrip() + "\n"


def _overview_projection_result(
    *, sections: list[dict[str, Any]], as_of_date: str, summary: str
) -> dict[str, Any]:
    return {
        "schema_version": "1.1-model-facing-overview-rolling",
        "as_of_date": as_of_date,
        "overview_sections": [
            {
                "category": section["category"],
                "bullets": [
                    {
                        "text": bullet["text"],
                        "source_fact_ids": copy.deepcopy(bullet["source_fact_ids"]),
                    }
                    for bullet in section["bullets"]
                ],
            }
            for section in sections
            if section["bullets"]
        ],
        "lifecycle_changes": [],
        "summary": summary,
    }


def _schema_errors(value: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(schema)
    return [
        f"{'/'.join(str(part) for part in error.absolute_path) or '$'}: {error.message}"
        for error in sorted(
            validator.iter_errors(value), key=lambda item: list(item.path)
        )
    ]


def _fact_map(background: dict[str, Any]) -> dict[str, tuple[str, dict[str, Any]]]:
    return legacy._categorized_items_by_id(background)


def _authorized_projection_changes(
    *,
    previous: dict[str, Any],
    current: dict[str, Any],
    applied_patch: dict[str, Any],
) -> list[dict[str, Any]]:
    previous_map = _fact_map(previous)
    current_map = _fact_map(current)
    changes: list[dict[str, Any]] = []
    seen: set[str] = set()

    def append_change(
        fact_id: str,
        change_type: str,
        operation: dict[str, Any],
    ) -> None:
        if fact_id in seen:
            return
        seen.add(fact_id)
        prior = previous_map.get(fact_id)
        after = current_map.get(fact_id)
        changes.append(
            {
                "fact_id": fact_id,
                "change_type": change_type,
                "before_category": prior[0] if prior else None,
                "before_statement": prior[1].get("statement") if prior else None,
                "after_category": after[0] if after else None,
                "after_statement": after[1].get("statement") if after else None,
                "today_evidence_excerpts": copy.deepcopy(
                    operation.get("evidence_excerpts", [])
                ),
            }
        )

    for operation in applied_patch.get("added_facts", []):
        append_change(operation["fact_id"], "added", operation)
        for fact_id in operation.get("supersedes_fact_ids", []):
            append_change(fact_id, "superseded", operation)
    for operation in applied_patch.get("updated_facts", []):
        append_change(operation["fact_id"], "updated", operation)
        for fact_id in operation.get("supersedes_fact_ids", []):
            append_change(fact_id, "superseded", operation)
    for operation in applied_patch.get("removed_facts", []):
        append_change(operation["fact_id"], "removed", operation)
    return changes


def _build_rolling_projection_task_prompt(
    *,
    current_day: date,
    numbered_bullets: list[dict[str, Any]],
    authorized_changes: list[dict[str, Any]],
    schema: dict[str, Any],
) -> str:
    return (
        f"处理日期：{current_day.isoformat()}。\n"
        "前一天最后实际保存的第二层条目如下。它们是本次局部更新的正文基础：\n"
        "<previous_final_layer2_bullets>\n"
        f"{json.dumps(numbered_bullets, ensure_ascii=False, indent=2)}\n"
        "</previous_final_layer2_bullets>\n\n"
        "第一次调用依据今日日记确认的可用变化如下。只有这些 fact_id 可以授权修改：\n"
        "<authorized_changes>\n"
        f"{json.dumps(authorized_changes, ensure_ascii=False, indent=2)}\n"
        "</authorized_changes>\n\n"
        "<output_json_schema>\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n"
        "</output_json_schema>\n\n"
        f"schema_version必须为{ROLLING_PROJECTION_SCHEMA_VERSION}，"
        f"as_of_date必须为{current_day.isoformat()}。只输出JSON对象。"
    )


def _filter_rolling_operations(
    *,
    value: dict[str, Any],
    numbered_bullets: list[dict[str, Any]],
    authorized_changes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets = {item["bullet_id"]: item for item in numbered_bullets}
    allowed_ids = {item["fact_id"] for item in authorized_changes}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    used_targets: set[str] = set()
    for index, operation in enumerate(value.get("operations", [])):
        reasons: list[str] = []
        action = operation.get("action")
        target_id = operation.get("target_bullet_id")
        text = operation.get("text")
        category = operation.get("category")
        source_ids = operation.get("source_fact_ids", [])
        if not isinstance(source_ids, list) or not source_ids:
            reasons.append("missing_authorized_source_fact_id")
        elif any(fact_id not in allowed_ids for fact_id in source_ids):
            reasons.append("source_fact_id_not_authorized")
        if action == "add":
            if target_id != "":
                reasons.append("add_must_not_target_existing_bullet")
            if not isinstance(text, str) or not text.strip():
                reasons.append("add_requires_text")
        elif action in {"update", "remove"}:
            target = targets.get(target_id)
            if target is None:
                reasons.append("target_bullet_not_found")
            elif target.get("category") != category:
                reasons.append("target_category_mismatch")
            elif any(
                not fact_id.startswith("manual-")
                for fact_id in target.get("source_fact_ids", [])
            ) and not set(source_ids).intersection(target["source_fact_ids"]):
                reasons.append("target_not_linked_to_authorized_fact")
            if target_id in used_targets:
                reasons.append("target_bullet_modified_more_than_once")
            if action == "update" and (not isinstance(text, str) or not text.strip()):
                reasons.append("update_requires_text")
            if action == "remove" and text != "":
                reasons.append("remove_text_must_be_empty")
        else:
            reasons.append("unknown_action")
        if reasons:
            rejected.append(
                {
                    "operation_index": index,
                    "operation": copy.deepcopy(operation),
                    "reasons": reasons,
                }
            )
            continue
        if action in {"update", "remove"}:
            used_targets.add(target_id)
        accepted.append(copy.deepcopy(operation))
    return accepted, rejected


def _apply_rolling_operations(
    *,
    sections: list[dict[str, Any]],
    operations: list[dict[str, Any]],
) -> None:
    locations: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    by_category = {section["category"]: section for section in sections}
    for section in sections:
        for bullet in section["bullets"]:
            locations[bullet["bullet_id"]] = (section, bullet)
    for operation in operations:
        action = operation["action"]
        source_ids = copy.deepcopy(operation["source_fact_ids"])
        if action == "add":
            section = by_category.get(operation["category"])
            if section is None:
                section = {"category": operation["category"], "bullets": []}
                sections.append(section)
                by_category[operation["category"]] = section
            section["bullets"].append(
                {
                    "text": operation["text"].strip(),
                    "source_fact_ids": source_ids,
                }
            )
            continue
        section, bullet = locations[operation["target_bullet_id"]]
        if action == "update":
            bullet["text"] = operation["text"].strip()
            bullet["source_fact_ids"] = source_ids
        else:
            section["bullets"].remove(bullet)


def _apply_projection_fallbacks(
    *,
    sections: list[dict[str, Any]],
    authorized_changes: list[dict[str, Any]],
    applied_source_ids: set[str],
) -> list[dict[str, Any]]:
    """Keep today's update moving if the wording model omits valid changes."""

    fallback_operations: list[dict[str, Any]] = []
    by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    by_category = {section["category"]: section for section in sections}
    for section in sections:
        for bullet in section["bullets"]:
            for fact_id in bullet["source_fact_ids"]:
                by_id.setdefault(fact_id, (section, bullet))
    for change in authorized_changes:
        fact_id = change["fact_id"]
        if fact_id in applied_source_ids:
            continue
        change_type = change["change_type"]
        location = by_id.get(fact_id)
        if change_type in {"removed", "superseded"}:
            if location is not None:
                section, bullet = location
                if bullet["source_fact_ids"] == [fact_id]:
                    section["bullets"].remove(bullet)
                    fallback_operations.append({"action": "remove", "fact_id": fact_id})
            continue
        statement = change.get("after_statement")
        category = change.get("after_category")
        if not isinstance(statement, str) or not statement.strip():
            continue
        if location is not None:
            _section, bullet = location
            if bullet["source_fact_ids"] == [fact_id]:
                bullet["text"] = statement.strip()
                fallback_operations.append({"action": "update", "fact_id": fact_id})
                continue
        section = by_category.get(category)
        if section is None and isinstance(category, str):
            section = {"category": category, "bullets": []}
            sections.append(section)
            by_category[category] = section
        if section is not None and all(
            bullet["text"] != statement.strip() for bullet in section["bullets"]
        ):
            section["bullets"].append(
                {"text": statement.strip(), "source_fact_ids": [fact_id]}
            )
            fallback_operations.append({"action": "add", "fact_id": fact_id})
    return fallback_operations


def _available_evidence_days(background: dict[str, Any], current_day: date) -> set[str]:
    result = {current_day.isoformat()}
    for section in background.get("sections", []):
        for item in section.get("items", []):
            result.update(
                value
                for value in item.get("source_dates", [])
                if isinstance(value, str)
            )
    return result


@contextmanager
def _daily_lock(layer2_root: Path, memory_day: str) -> Iterator[None]:
    lock_path = layer2_root / "locks" / f"{memory_day}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists() and time.time() - lock_path.stat().st_mtime > 12 * 3600:
        lock_path.unlink()
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise Layer2RunAlreadyActive(f"worker_already_running:{memory_day}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def load_layer2_settings(
    config_path: str | Path = "conf.yaml",
) -> Layer2MemoryGenerationConfig:
    config = validate_config(read_yaml(str(config_path)))
    return config.character_config.layer2_memory_generation


def layer2_publication_status(
    memory_day: str,
    history_root: str | Path | None = None,
) -> str:
    target_day = date.fromisoformat(memory_day)
    history = resolve_character_history_root(history_root).resolve()
    loaded = load_layer2_publication(history)
    status = loaded.diagnostics.get("status")
    if status == "missing":
        return "missing"
    if status != "loaded":
        return "invalid"
    published_day = date.fromisoformat(str(loaded.diagnostics["as_of_date"]))
    if published_day > target_day:
        return "current"
    if published_day < target_day:
        return "missing"
    diary = history / "diaries" / f"diary_{memory_day}.txt"
    if not diary.is_file() or diary.stat().st_size == 0:
        return "stale"
    if sha256_file(diary) != loaded.diagnostics.get("diary_sha256"):
        return "stale"
    return "current"


def _settings_for_cloud(
    config_path: Path, settings: Layer2MemoryGenerationConfig
) -> tuple[Any, dict[str, Any]]:
    cloud_settings, safe = cloud_runner._load_cloud_settings(
        config_path,
        settings.provider_config_name,
        settings.model,
        settings.max_output_tokens,
    )
    cloud_settings = cloud_settings.model_copy(
        update={
            "temperature": settings.temperature,
            "timeout_seconds": settings.timeout_seconds,
            "max_output_tokens": settings.max_output_tokens,
        }
    )
    safe = {
        **safe,
        "temperature": settings.temperature,
        "timeout_seconds": settings.timeout_seconds,
        "max_output_tokens": settings.max_output_tokens,
        "reasoning_effort": settings.reasoning_effort,
        "api_key_present": True,
    }
    return cloud_settings, safe


def _projection_tool_arguments(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise DailyChildEventProviderError("provider_choices_missing")
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        raise DailyChildEventProviderError("provider_output_truncated_before_tool_call")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise DailyChildEventProviderError("provider_message_missing")
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise DailyChildEventProviderError("provider_tool_call_missing_or_multiple")
    function = calls[0].get("function")
    if (
        not isinstance(function, dict)
        or function.get("name") != projection_runner.TOOL_NAME
    ):
        raise DailyChildEventProviderError("provider_tool_name_mismatch")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise DailyChildEventProviderError(
                "provider_tool_arguments_not_json"
            ) from exc
    if not isinstance(arguments, dict):
        raise DailyChildEventProviderError("provider_tool_arguments_missing")
    overview_json = arguments.get("overview_json")
    if isinstance(overview_json, str):
        try:
            value = json.loads(overview_json)
        except json.JSONDecodeError as exc:
            raise DailyChildEventProviderError(
                "provider_overview_json_invalid"
            ) from exc
    else:
        value = arguments
    if not isinstance(value, dict):
        raise DailyChildEventProviderError("provider_overview_not_object")
    return value


def _call_openai_projection_with_retries(
    *,
    attempts_dir: Path,
    system_prompt: str,
    task_prompt: str,
    settings: Any,
    schema: dict[str, Any],
    background: dict[str, Any],
    review_date: date,
) -> tuple[dict[str, Any], dict[str, Any], Path, int, float]:
    started = time.monotonic()
    failures: list[dict[str, Any]] = []
    repair_base = None
    repair_fields: list[str] = []
    repair_violations: list[dict[str, Any]] = []
    for attempt in range(1, 4):
        attempt_dir = attempts_dir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        attempt_settings = settings
        request_url = provider_adapter._append_endpoint(
            settings.base_url, "/chat/completions"
        )
        payload = {
            "model": settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_prompt},
            ],
            "temperature": settings.temperature,
            "max_tokens": settings.max_output_tokens,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": projection_runner.TOOL_NAME,
                        "description": "提交当前个人背景的模型阅读版与生命周期审查建议。",
                        "parameters": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["overview_json"],
                            "properties": {
                                "overview_json": {
                                    "type": "string",
                                    "description": "严格符合任务中output_json_schema的完整JSON对象文本",
                                }
                            },
                        },
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": projection_runner.TOOL_NAME},
            },
        }
        if cloud_runner.claude_transport.enabled(settings):
            attempt_settings = copy.copy(settings)
            attempt_settings.max_output_tokens = min(
                settings.max_output_tokens,
                16384 if repair_base is not None else 32768,
            )
            attempt_settings.timeout_seconds = max(
                settings.timeout_seconds,
                300.0 if repair_base is not None else 600.0,
            )
            bullet_limit = (
                schema.get("$defs", {})
                .get("overview_section", {})
                .get("properties", {})
                .get("bullets", {})
                .get("maxItems")
            )
            claude_prompt = system_prompt
            if isinstance(bullet_limit, int):
                claude_prompt += (
                    f"\n输出前逐组计数：每个 overview_sections 分组的 bullets "
                    f"最多 {bullet_limit} 条。相关内容请合并概括，并保留对应 "
                    "source_fact_ids；不要把每条历史事实都单独列为一条。"
                    "同时遵守每条文本和来源数量限制。"
                )
            lifecycle_limit = (
                schema.get("properties", {})
                .get("lifecycle_changes", {})
                .get("maxItems")
            )
            summary_limit = (
                schema.get("properties", {}).get("summary", {}).get("maxLength")
            )
            if isinstance(lifecycle_limit, int):
                claude_prompt += (
                    f"\nlifecycle_changes 最多 {lifecycle_limit} 条，"
                    "同一个 fact_id 只能出现一次；若候选更多，只保留最需要"
                    "处理的项目，不得用两个动作指向同一事实。"
                )
            if isinstance(summary_limit, int):
                claude_prompt += (
                    f"\nsummary 不得超过 {summary_limit} 个字符，"
                    f"请以不超过 {summary_limit // 2} 个字符为目标，留出余量。"
                )
            request_task = task_prompt
            if repair_base is not None:
                claude_prompt += (
                    "\n本次是局部纠错：只返回指定字段的 YAML 对象，"
                    "程序会将这些字段替换回上一份完整结果后重新校验。"
                    "不要返回其他字段；保留有依据的内容，合并概括以满足上限，"
                    "存在 maxLength 上限的文本请以该上限的一半为目标长度，"
                    "给字符计数留出余量，不必用满上限。"
                    "不可凭空增加事实或来源。完整 schema 的必填字段规则"
                    "用于合并后的结果，本次回复只需指定字段。"
                )
                request_task += "\n局部纠错输入：\n" + json.dumps(
                    {
                        "replace_fields": repair_fields,
                        "violations": repair_violations,
                        "previous_values": {
                            key: repair_base.get(key) for key in repair_fields
                        },
                        "field_schemas": {
                            key: schema["properties"][key] for key in repair_fields
                        },
                    },
                    ensure_ascii=False,
                )
            payload, request_url, _ = cloud_runner.claude_transport.request(
                attempt_settings, claude_prompt, request_task
            )
        _write_json(
            attempt_dir / "request_metadata.json",
            {
                "attempt": attempt,
                "model": settings.model,
                "provider": settings.llm_provider,
                "base_url": cloud_runner._safe_base_url(settings.base_url),
                "request_endpoint": cloud_runner._safe_base_url(request_url),
                "structured_output_transport": (
                    "anthropic_schema_validated_yaml"
                    if cloud_runner.claude_transport.enabled(settings)
                    else "forced_function_call"
                ),
                "request_payload": payload,
                "timeout_seconds": attempt_settings.timeout_seconds,
                "credential_written": False,
            },
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.llm_api_key}",
        }
        if cloud_runner.claude_transport.enabled(settings):
            headers = cloud_runner.claude_transport.headers(attempt_settings)
        try:
            response = (
                cloud_runner.claude_transport.post(
                    attempt_settings, payload, attempt_dir / "response.json"
                )
                if cloud_runner.claude_transport.enabled(settings)
                else provider_adapter._post_json(
                    request_url,
                    payload,
                    headers,
                    settings.timeout_seconds,
                    attempt_dir / "response.json",
                    proxy_url=settings.proxy_url,
                )
            )
            value = (
                cloud_runner.claude_transport.parse(response)
                if cloud_runner.claude_transport.enabled(settings)
                else _projection_tool_arguments(response)
            )
            if repair_base is not None:
                if set(value) != set(repair_fields):
                    raise Layer2RuntimeError("projection_repair_fields_mismatch")
                value = {**repair_base, **value}
            schema_failures = list(Draft202012Validator(schema).iter_errors(value))
            semantic_failures = projection_runner._semantic_errors(
                value, background, review_date
            )
            if cloud_runner.claude_transport.enabled(settings):
                fields = {str(error.path[0]) for error in schema_failures if error.path}
                semantic_field_rules = (
                    ("overview ", "overview_sections"),
                    ("overview section", "overview_sections"),
                    ("lifecycle ", "lifecycle_changes"),
                    ("schema_version", "schema_version"),
                    ("as_of_date", "as_of_date"),
                )
                semantic_repairs: list[dict[str, Any]] = []
                for semantic_error in semantic_failures:
                    for marker, field in semantic_field_rules:
                        if marker in semantic_error:
                            fields.add(field)
                            semantic_repairs.append(
                                {
                                    "path": [field],
                                    "rule": "semantic",
                                    "message": semantic_error,
                                }
                            )
                            break
                fields = sorted(fields)
                if (
                    fields
                    and all(error.path for error in schema_failures)
                    and all(key in schema.get("properties", {}) for key in fields)
                ):
                    repair_base = value
                    repair_fields = fields
                    repair_violations = [
                        {
                            "path": list(error.path),
                            "rule": error.validator,
                            "limit": error.validator_value,
                        }
                        for error in schema_failures
                    ] + semantic_repairs
                else:
                    repair_base = None
                    repair_fields = []
                    repair_violations = []
            errors = projection_runner._schema_errors(value, schema)
            errors.extend(semantic_failures)
            if errors:
                raise Layer2RuntimeError("; ".join(errors))
        except (
            DailyChildEventProviderError,
            Layer2RuntimeError,
            cloud_runner.claude_transport.ClaudeOutputError,
        ) as exc:
            failure = {
                "attempt": attempt,
                "failed_at": _iso_now(),
                "error": f"{type(exc).__name__}:{exc}",
            }
            failures.append(failure)
            _write_json(attempt_dir / "error.json", failure)
            if attempt < 3:
                time.sleep(min(attempt * 2, 5))
            continue
        _write_json(attempt_dir / "tool_arguments.json", value)
        return value, response, attempt_dir, attempt, time.monotonic() - started
    raise Layer2RuntimeError(
        "projection_failed_three_attempts:" + json.dumps(failures, ensure_ascii=False)
    )


def _rolling_tool_arguments(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise DailyChildEventProviderError("provider_choices_missing")
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        raise DailyChildEventProviderError("provider_output_truncated_before_tool_call")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise DailyChildEventProviderError("provider_message_missing")
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise DailyChildEventProviderError("provider_tool_call_missing_or_multiple")
    function = calls[0].get("function")
    if (
        not isinstance(function, dict)
        or function.get("name") != ROLLING_PROJECTION_TOOL_NAME
    ):
        raise DailyChildEventProviderError("provider_tool_name_mismatch")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise DailyChildEventProviderError(
                "provider_tool_arguments_not_json"
            ) from exc
    if not isinstance(arguments, dict):
        raise DailyChildEventProviderError("provider_tool_arguments_missing")
    patch_json = arguments.get("patch_json")
    if isinstance(patch_json, str):
        try:
            value = json.loads(patch_json)
        except json.JSONDecodeError as exc:
            raise DailyChildEventProviderError("provider_patch_json_invalid") from exc
    else:
        value = arguments
    if not isinstance(value, dict):
        raise DailyChildEventProviderError("provider_patch_not_object")
    return value


def _call_rolling_patch_with_retries(
    *,
    attempts_dir: Path,
    system_prompt: str,
    task_prompt: str,
    settings: Any,
    schema: dict[str, Any],
    current_day: date,
    reasoning_effort: str,
) -> tuple[dict[str, Any], dict[str, Any], Path, int, float]:
    started = time.monotonic()
    failures: list[dict[str, Any]] = []
    for attempt in range(1, 4):
        attempt_dir = attempts_dir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        attempt_settings = settings
        is_claude = cloud_runner.claude_transport.enabled(settings)
        if is_claude:
            attempt_settings = copy.copy(settings)
            attempt_settings.max_output_tokens = min(settings.max_output_tokens, 16384)
            attempt_settings.timeout_seconds = max(settings.timeout_seconds, 300.0)
            payload, request_url, _transport = cloud_runner.claude_transport.request(
                attempt_settings, system_prompt, task_prompt
            )
        elif settings.llm_provider == "deepseek_llm":
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
                            "name": ROLLING_PROJECTION_TOOL_NAME,
                            "description": "提交第二层模型阅读版的局部滚动修改。",
                            "strict": True,
                            "parameters": cloud_runner._deepseek_strict_schema(schema),
                        },
                    }
                ],
            }
        else:
            request_url = provider_adapter._append_endpoint(
                settings.base_url, "/chat/completions"
            )
            payload = {
                "model": settings.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task_prompt},
                ],
                "temperature": settings.temperature,
                "max_tokens": settings.max_output_tokens,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": ROLLING_PROJECTION_TOOL_NAME,
                            "description": "提交第二层模型阅读版的局部滚动修改。",
                            "parameters": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["patch_json"],
                                "properties": {
                                    "patch_json": {
                                        "type": "string",
                                        "description": (
                                            "严格符合任务中output_json_schema的"
                                            "完整JSON对象文本"
                                        ),
                                    }
                                },
                            },
                        },
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": ROLLING_PROJECTION_TOOL_NAME},
                },
            }
        _write_json(
            attempt_dir / "request_metadata.json",
            {
                "attempt": attempt,
                "model": settings.model,
                "provider": settings.llm_provider,
                "base_url": cloud_runner._safe_base_url(settings.base_url),
                "request_endpoint": cloud_runner._safe_base_url(request_url),
                "structured_output_transport": (
                    "anthropic_schema_validated_yaml"
                    if is_claude
                    else "forced_rolling_patch_tool"
                ),
                "request_payload": payload,
                "timeout_seconds": attempt_settings.timeout_seconds,
                "credential_written": False,
            },
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.llm_api_key}",
        }
        organization_id = getattr(settings, "organization_id", None)
        project_id = getattr(settings, "project_id", None)
        if organization_id:
            headers["OpenAI-Organization"] = organization_id
        if project_id:
            headers["OpenAI-Project"] = project_id
        if is_claude:
            headers = cloud_runner.claude_transport.headers(attempt_settings)
        try:
            response = (
                cloud_runner.claude_transport.post(
                    attempt_settings, payload, attempt_dir / "response.json"
                )
                if is_claude
                else provider_adapter._post_json(
                    request_url,
                    payload,
                    headers,
                    settings.timeout_seconds,
                    attempt_dir / "response.json",
                    proxy_url=settings.proxy_url,
                )
            )
            value = (
                cloud_runner.claude_transport.parse(response)
                if is_claude
                else _rolling_tool_arguments(response)
            )
            errors = _schema_errors(value, schema)
            if value.get("as_of_date") != current_day.isoformat():
                errors.append("as_of_date mismatch")
            if errors:
                raise Layer2RuntimeError("; ".join(errors))
        except (
            DailyChildEventProviderError,
            Layer2RuntimeError,
            cloud_runner.claude_transport.ClaudeOutputError,
        ) as exc:
            failure = {
                "attempt": attempt,
                "failed_at": _iso_now(),
                "error": f"{type(exc).__name__}:{exc}",
            }
            failures.append(failure)
            _write_json(attempt_dir / "error.json", failure)
            if attempt < 3:
                time.sleep(min(attempt * 2, 5))
            continue
        _write_json(attempt_dir / "tool_arguments.json", value)
        return value, response, attempt_dir, attempt, time.monotonic() - started
    raise Layer2RuntimeError(
        "rolling_projection_failed_three_attempts:"
        + json.dumps(failures, ensure_ascii=False)
    )


def _run_projection(
    *,
    run_dir: Path,
    background: dict[str, Any],
    current_day: date,
    cloud_settings: Any,
    settings: Layer2MemoryGenerationConfig,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    prompt = PROJECTION_PROMPT_PATH.read_text(encoding="utf-8")
    schema_text = PROJECTION_SCHEMA_PATH.read_text(encoding="utf-8")
    schema = json.loads(schema_text)
    neutrality = projection_runner._neutrality_audit(prompt, schema_text, background)
    if not neutrality["passed"]:
        raise Layer2RuntimeError("projection_prompt_neutrality_failed")
    task_prompt = projection_runner._build_task_prompt(
        review_date=current_day,
        background=background,
        schema=schema,
    )
    projection_dir = run_dir / "projection"
    attempts_dir = projection_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=False)
    _write_json(projection_dir / "neutrality_audit.json", neutrality)
    (projection_dir / "task_prompt.txt").write_text(
        task_prompt, encoding="utf-8", newline="\n"
    )
    if cloud_settings.llm_provider == "deepseek_llm":
        result, response, successful_attempt, attempt_count, elapsed = (
            projection_runner._call_with_retries(
                attempts_dir=attempts_dir,
                system_prompt=prompt,
                task_prompt=task_prompt,
                settings=cloud_settings,
                schema=schema,
                background=background,
                review_date=current_day,
                reasoning_effort=settings.reasoning_effort,
            )
        )
    else:
        result, response, successful_attempt, attempt_count, elapsed = (
            _call_openai_projection_with_retries(
                attempts_dir=attempts_dir,
                system_prompt=prompt,
                task_prompt=task_prompt,
                settings=cloud_settings,
                schema=schema,
                background=background,
                review_date=current_day,
            )
        )
    overview = projection_runner._render_overview(result, include_sources=False)
    overview_with_sources = projection_runner._render_overview(
        result, include_sources=True
    )
    metrics = projection_runner._metrics(
        value=result,
        overview=overview,
        background=background,
        response=response,
        attempt_count=attempt_count,
        elapsed=elapsed,
    )
    metrics["successful_attempt_dir"] = str(successful_attempt)
    _write_json(projection_dir / "result.json", result)
    _write_json(projection_dir / "metrics.json", metrics)
    (projection_dir / "model_facing_overview.md").write_text(
        overview, encoding="utf-8", newline="\n"
    )
    (projection_dir / "projection_with_sources.md").write_text(
        overview_with_sources, encoding="utf-8", newline="\n"
    )
    return result, overview, metrics


def _run_rolling_projection(
    *,
    run_dir: Path,
    base_sections: list[dict[str, Any]],
    authorized_changes: list[dict[str, Any]],
    current_day: date,
    cloud_settings: Any,
    settings: Layer2MemoryGenerationConfig,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Apply a model-authored patch to the last saved overview.

    The wording model can propose operations only. The runtime filters each
    operation independently, preserves every untouched baseline bullet, and
    deterministically carries valid first-call facts when the second call omits
    them or fails altogether.
    """

    prompt = ROLLING_PROJECTION_PROMPT_PATH.read_text(encoding="utf-8")
    schema = _read_json(ROLLING_PROJECTION_SCHEMA_PATH)
    sections = copy.deepcopy(base_sections)
    numbered_bullets = _number_overview_bullets(sections)
    task_prompt = _build_rolling_projection_task_prompt(
        current_day=current_day,
        numbered_bullets=numbered_bullets,
        authorized_changes=authorized_changes,
        schema=schema,
    )
    projection_dir = run_dir / "projection"
    attempts_dir = projection_dir / "attempts"
    projection_dir.mkdir(parents=True, exist_ok=False)
    (projection_dir / "previous_final_layer2.md").write_text(
        _render_overview_sections(sections), encoding="utf-8", newline="\n"
    )
    _write_json(projection_dir / "previous_final_bullets.json", numbered_bullets)
    _write_json(projection_dir / "authorized_changes.json", authorized_changes)
    (projection_dir / "task_prompt.txt").write_text(
        task_prompt, encoding="utf-8", newline="\n"
    )

    patch_value: dict[str, Any] = {
        "schema_version": ROLLING_PROJECTION_SCHEMA_VERSION,
        "as_of_date": current_day.isoformat(),
        "operations": [],
        "summary": "第一次调用没有产生需要写入模型阅读版的变化。",
    }
    response: dict[str, Any] = {}
    successful_attempt: Path | None = None
    attempt_count = 0
    elapsed = 0.0
    projection_failure: str | None = None
    if authorized_changes:
        attempts_dir.mkdir(parents=True, exist_ok=False)
        try:
            (
                patch_value,
                response,
                successful_attempt,
                attempt_count,
                elapsed,
            ) = _call_rolling_patch_with_retries(
                attempts_dir=attempts_dir,
                system_prompt=prompt,
                task_prompt=task_prompt,
                settings=cloud_settings,
                schema=schema,
                current_day=current_day,
                reasoning_effort=settings.reasoning_effort,
            )
        except Layer2RuntimeError as exc:
            projection_failure = str(exc)
            attempt_count = len(list(attempts_dir.glob("attempt_*")))
            patch_value = {
                "schema_version": ROLLING_PROJECTION_SCHEMA_VERSION,
                "as_of_date": current_day.isoformat(),
                "operations": [],
                "summary": "第二次调用失败，程序改用第一次调用事实作最低限度更新。",
            }

    accepted, rejected = _filter_rolling_operations(
        value=patch_value,
        numbered_bullets=numbered_bullets,
        authorized_changes=authorized_changes,
    )
    _apply_rolling_operations(sections=sections, operations=accepted)
    applied_source_ids = {
        fact_id for operation in accepted for fact_id in operation["source_fact_ids"]
    }
    fallbacks = _apply_projection_fallbacks(
        sections=sections,
        authorized_changes=authorized_changes,
        applied_source_ids=applied_source_ids,
    )
    overview = _render_overview_sections(sections)
    result = _overview_projection_result(
        sections=sections,
        as_of_date=current_day.isoformat(),
        summary=patch_value["summary"],
    )
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    metrics = {
        "mode": "rolling_patch",
        "api_attempt_count": attempt_count,
        "elapsed_wall_seconds": round(elapsed, 3),
        "successful_attempt_dir": (
            str(successful_attempt) if successful_attempt is not None else None
        ),
        "authorized_change_count": len(authorized_changes),
        "proposed_operation_count": len(patch_value["operations"]),
        "accepted_operation_count": len(accepted),
        "rejected_operation_count": len(rejected),
        "fallback_operation_count": len(fallbacks),
        "projection_failure": projection_failure,
        "overview_bullet_count": sum(len(section["bullets"]) for section in sections),
        "overview_character_count": len(overview.rstrip("\r\n")),
        "exact_date_token_count": len(
            projection_runner.DATE_TOKEN_RE.findall(overview)
        ),
        "usage": copy.deepcopy(usage),
    }
    _write_json(projection_dir / "patch_result.json", patch_value)
    _write_json(projection_dir / "accepted_operations.json", accepted)
    _write_json(projection_dir / "rejected_operations.json", rejected)
    _write_json(projection_dir / "fallback_operations.json", fallbacks)
    _write_json(projection_dir / "result.json", result)
    _write_json(projection_dir / "metrics.json", metrics)
    (projection_dir / "model_facing_overview.md").write_text(
        overview, encoding="utf-8", newline="\n"
    )
    return result, overview, metrics


def _publish(
    *,
    layer2_root: Path,
    run_id: str,
    current_day: date,
    diary_hash: str,
    background: dict[str, Any],
    projection_result: dict[str, Any],
    overview: str,
    provider_safe: dict[str, Any],
    ledger_metrics: dict[str, Any],
    projection_metrics: dict[str, Any],
) -> Path:
    publications = layer2_root / "publications"
    publications.mkdir(parents=True, exist_ok=True)
    staging = publications / f".staging-{run_id}"
    target = publications / f"{current_day.isoformat()}_{run_id}"
    pointer_path = layer2_root / CURRENT_POINTER_NAME
    previous_pointer = pointer_path.read_bytes() if pointer_path.is_file() else None
    if staging.exists() or target.exists():
        raise Layer2RuntimeError("publication_run_id_conflict")
    staging.mkdir()
    try:
        _write_json(staging / "background.json", background)
        _write_json(staging / "projection_result.json", projection_result)
        (staging / "model_facing_overview.md").write_text(
            overview, encoding="utf-8", newline="\n"
        )
        (staging / "model_context_preview.md").write_text(
            format_layer2_context(current_day.isoformat(), overview) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _write_json(
            staging / "publication_metrics.json",
            {"ledger": ledger_metrics, "projection": projection_metrics},
        )
        files = {
            "background": {
                "name": "background.json",
                "sha256": sha256_file(staging / "background.json"),
            },
            "model_facing_overview": {
                "name": "model_facing_overview.md",
                "sha256": sha256_file(staging / "model_facing_overview.md"),
            },
            "projection_result": {
                "name": "projection_result.json",
                "sha256": sha256_file(staging / "projection_result.json"),
            },
            "model_context_preview": {
                "name": "model_context_preview.md",
                "sha256": sha256_file(staging / "model_context_preview.md"),
            },
            "publication_metrics": {
                "name": "publication_metrics.json",
                "sha256": sha256_file(staging / "publication_metrics.json"),
            },
        }
        manifest = {
            "schema_version": PUBLICATION_SCHEMA_VERSION,
            "status": "published",
            "published_at": _iso_now(),
            "as_of_date": current_day.isoformat(),
            "diary_sha256": diary_hash,
            "provider": provider_safe,
            "prompt_sha256": {
                "ledger": sha256_file(LEDGER_PROMPT_PATH),
                "projection": sha256_file(ROLLING_PROJECTION_PROMPT_PATH),
            },
            "schema_sha256": {
                "ledger": sha256_file(LEDGER_SCHEMA_PATH),
                "projection": sha256_file(ROLLING_PROJECTION_SCHEMA_PATH),
            },
            "files": files,
        }
        _write_json(staging / "manifest.json", manifest)
        staging.rename(target)
        pointer = {
            "schema_version": POINTER_SCHEMA_VERSION,
            "updated_at": _iso_now(),
            "as_of_date": current_day.isoformat(),
            "publication": target.relative_to(layer2_root).as_posix(),
            "manifest_sha256": sha256_file(target / "manifest.json"),
        }
        _write_json_atomic(pointer_path, pointer)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    loaded = load_layer2_publication(layer2_root.parent)
    if loaded.diagnostics.get("status") != "loaded":
        if previous_pointer is None:
            try:
                pointer_path.unlink()
            except FileNotFoundError:
                pass
        else:
            _write_bytes_atomic(pointer_path, previous_pointer)
        raise Layer2RuntimeError("post_publication_verification_failed")
    try:
        _write_bytes_atomic(
            layer2_root / CURRENT_CONTEXT_NAME,
            (loaded.context.rstrip() + "\n").encode("utf-8"),
        )
    except Exception:
        if previous_pointer is None:
            try:
                pointer_path.unlink()
            except FileNotFoundError:
                pass
        else:
            _write_bytes_atomic(pointer_path, previous_pointer)
        raise
    return target


def _prune_retained_runtime(
    layer2_root: Path, current_day: date, retention_days: int
) -> dict[str, list[str]]:
    cutoff = current_day - timedelta(days=retention_days - 1)
    current = load_layer2_publication(layer2_root.parent)
    current_publication = current.diagnostics.get("publication")
    removed: dict[str, list[str]] = {
        "runs": [],
        "publications": [],
        "worker_results": [],
    }
    for parent_name in ("runs", "worker_results"):
        parent = layer2_root / parent_name
        if not parent.is_dir():
            continue
        for candidate in parent.iterdir():
            try:
                candidate_day = date.fromisoformat(candidate.name)
            except ValueError:
                continue
            if candidate.is_dir() and candidate_day < cutoff:
                shutil.rmtree(candidate)
                removed[parent_name].append(str(candidate))
    publications = layer2_root / "publications"
    if publications.is_dir():
        for candidate in publications.iterdir():
            if not candidate.is_dir() or candidate.name.startswith(".staging-"):
                continue
            try:
                candidate_day = date.fromisoformat(candidate.name[:10])
            except ValueError:
                continue
            if (
                candidate_day < cutoff
                and str(candidate.resolve()) != current_publication
            ):
                shutil.rmtree(candidate)
                removed["publications"].append(str(candidate))
    return removed


def run_daily_layer2_update(
    memory_day: str,
    history_root: str | Path | None = None,
    config_path: str | Path = "conf.yaml",
    generation_settings: Layer2MemoryGenerationConfig | None = None,
) -> dict[str, Any]:
    current_day = date.fromisoformat(memory_day)
    history = resolve_character_history_root(history_root).resolve()
    layer2_root = history / LAYER2_DIRECTORY_NAME
    config_path = Path(config_path).resolve()
    settings = generation_settings or load_layer2_settings(config_path)
    if not settings.enabled:
        return {"status": "disabled", "memory_day": memory_day}
    existing_status = layer2_publication_status(memory_day, history)
    if existing_status == "current":
        return {"status": "already_current", "memory_day": memory_day}
    if existing_status == "stale":
        raise Layer2RuntimeError("same_day_diary_changed_manual_review_required")
    if existing_status == "invalid":
        raise Layer2RuntimeError("current_layer2_publication_invalid")
    loaded = load_layer2_publication(history)
    first_publication = loaded.diagnostics.get("status") == "missing"
    if first_publication:
        previous = delta_runner._empty_background(current_day - timedelta(days=1))
    elif loaded.diagnostics.get("status") == "loaded" and loaded.background_path is not None:
        previous = _read_json(loaded.background_path)
        if date.fromisoformat(previous["as_of_date"]) >= current_day:
            raise Layer2RuntimeError("layer2_update_would_regress_publication_date")
    else:
        raise Layer2RuntimeError("current_layer2_publication_invalid")
    diary_path = history / "diaries" / f"diary_{memory_day}.txt"
    if not diary_path.is_file() or diary_path.stat().st_size == 0:
        raise Layer2RuntimeError(f"diary_missing_or_empty:{diary_path}")
    diary_text = diary_path.read_text(encoding="utf-8")
    diary_hash = sha256_file(diary_path)
    cloud_settings, provider_safe = _settings_for_cloud(config_path, settings)
    layer2_root.mkdir(parents=True, exist_ok=True)

    with _daily_lock(layer2_root, memory_day):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:10]}"
        run_dir = layer2_root / "runs" / memory_day / run_id
        ledger_dir = run_dir / "ledger"
        attempts_dir = ledger_dir / "attempts"
        attempts_dir.mkdir(parents=True, exist_ok=False)
        _write_json(
            run_dir / "run.json",
            {
                "status": "running",
                "started_at": _iso_now(),
                "memory_day": memory_day,
                "previous_as_of_date": previous.get("as_of_date"),
                "diary_sha256": diary_hash,
                "provider": provider_safe,
                "credential_written": False,
            },
        )
        ledger_prompt = LEDGER_PROMPT_PATH.read_text(encoding="utf-8")
        ledger_schema = _read_json(LEDGER_SCHEMA_PATH)
        task_prompt = cloud_runner._build_task_prompt(
            current_day=current_day,
            previous=previous,
            diary_text=diary_text,
            schema=ledger_schema,
        )
        (ledger_dir / "task_prompt.txt").write_text(
            task_prompt, encoding="utf-8", newline="\n"
        )
        try:
            text, successful_attempt, attempt_count, elapsed = (
                cloud_runner._call_with_retries(
                    current_day=current_day,
                    system_prompt=ledger_prompt,
                    task_prompt=task_prompt,
                    attempts_dir=attempts_dir,
                    settings=cloud_settings,
                    schema=ledger_schema,
                    deepseek_reasoning_effort=settings.reasoning_effort,
                )
            )
            patch = json.loads(text)
            if not isinstance(patch, dict):
                raise Layer2RuntimeError("ledger_patch_root_not_object")
            errors = _schema_errors(patch, ledger_schema)
            if errors:
                raise Layer2RuntimeError("ledger_schema_invalid:" + "; ".join(errors))
            prepared, discarded, repairs = cloud_runner._prepare_evidence_excerpts(
                patch, diary_text
            )
            objective_errors = cloud_runner._objective_operation_errors(
                prepared,
                current_day=current_day,
                previous=previous,
                diary_text=diary_text,
            )
            operation_limit_repair = None
            if _is_only_operation_limit_error(objective_errors):
                original_operation_count = _operation_count(patch)
                _write_json(ledger_dir / "rejected_over_limit_patch.json", patch)
                repair_dir = ledger_dir / "operation_limit_repair"
                repair_attempts_dir = repair_dir / "attempts"
                repair_attempts_dir.mkdir(parents=True, exist_ok=False)
                repair_task_prompt = _build_operation_limit_repair_prompt(
                    task_prompt=task_prompt,
                    rejected_patch=patch,
                    violation=objective_errors["$"][0],
                )
                (repair_dir / "task_prompt.txt").write_text(
                    repair_task_prompt,
                    encoding="utf-8",
                    newline="\n",
                )
                (
                    repair_text,
                    repair_attempt_dir,
                    repair_attempt_count,
                    repair_elapsed,
                ) = cloud_runner._call_with_retries(
                    current_day=current_day,
                    system_prompt=(
                        ledger_prompt + "\n\n上一份候选仅因跨字段操作总数超限而被拒绝。"
                        "必须重新输出完整对象并主动取舍；不得要求调用方截断。"
                    ),
                    task_prompt=repair_task_prompt,
                    attempts_dir=repair_attempts_dir,
                    settings=cloud_settings,
                    schema=ledger_schema,
                    deepseek_reasoning_effort=settings.reasoning_effort,
                )
                try:
                    repaired_patch = json.loads(repair_text)
                except json.JSONDecodeError as exc:
                    raise Layer2RuntimeError(
                        "ledger_operation_limit_repair_not_json"
                    ) from exc
                if not isinstance(repaired_patch, dict):
                    raise Layer2RuntimeError(
                        "ledger_operation_limit_repair_root_not_object"
                    )
                repair_schema_errors = _schema_errors(repaired_patch, ledger_schema)
                if repair_schema_errors:
                    raise Layer2RuntimeError(
                        "ledger_operation_limit_repair_schema_invalid:"
                        + "; ".join(repair_schema_errors)
                    )
                repaired_prepared, repaired_discarded, repaired_repairs = (
                    cloud_runner._prepare_evidence_excerpts(repaired_patch, diary_text)
                )
                repaired_objective_errors = cloud_runner._objective_operation_errors(
                    repaired_prepared,
                    current_day=current_day,
                    previous=previous,
                    diary_text=diary_text,
                )
                operation_limit_repair = {
                    "attempted": True,
                    "violation": objective_errors["$"][0],
                    "original_operation_count": original_operation_count,
                    "repaired_operation_count": _operation_count(repaired_patch),
                    "initial_successful_attempt_dir": str(successful_attempt),
                    "repair_successful_attempt_dir": str(repair_attempt_dir),
                    "repair_api_attempt_count": repair_attempt_count,
                    "repair_elapsed_wall_seconds": round(repair_elapsed, 3),
                }
                _write_json(repair_dir / "repair_audit.json", operation_limit_repair)
                patch = repaired_patch
                prepared = repaired_prepared
                discarded = repaired_discarded
                repairs = repaired_repairs
                objective_errors = repaired_objective_errors
                successful_attempt = repair_attempt_dir
                attempt_count += repair_attempt_count
                elapsed += repair_elapsed
            if "$" in objective_errors:
                raise Layer2RuntimeError(
                    "ledger_root_invariant_failed:" + "; ".join(objective_errors["$"])
                )
            applied, quarantined = cloud_runner._quarantine_objective_failures(
                prepared, objective_errors
            )
            background, merge_audit = delta_runner._merge_delta(
                previous, applied, current_day
            )
            background_errors = delta_runner._validate_background(
                background,
                expected_day=current_day,
                available_days=_available_evidence_days(background, current_day),
                previous=previous,
            )
            if background_errors:
                raise Layer2RuntimeError(
                    "merged_background_invalid:" + "; ".join(background_errors)
                )
            _write_json(ledger_dir / "patch.json", patch)
            _write_json(ledger_dir / "applied_patch.json", applied)
            _write_json(ledger_dir / "background.json", background)
            merge_audit.update(
                {
                    "provider": provider_safe,
                    "successful_attempt_dir": str(successful_attempt),
                    "evidence_repairs": repairs,
                    "discarded_evidence_excerpts": discarded,
                    "quarantined_operations": quarantined,
                    "semantic_filtering_applied": False,
                }
            )
            _write_json(ledger_dir / "merge_audit.json", merge_audit)
            ledger_metrics = {
                "elapsed_wall_seconds": round(elapsed, 3),
                "api_attempt_count": attempt_count,
                "model_operation_counts": {
                    "added": len(patch["added_facts"]),
                    "updated": len(patch["updated_facts"]),
                    "removed": len(patch["removed_facts"]),
                },
                "applied_operation_counts": {
                    "added": len(applied["added_facts"]),
                    "updated": len(applied["updated_facts"]),
                    "removed": len(applied["removed_facts"]),
                },
                "objective_quarantine_count": len(quarantined),
                "discarded_evidence_excerpt_count": len(discarded),
                "evidence_repair_count": len(repairs),
                "result_fact_count": len(legacy._items_by_id(background)),
            }
            if operation_limit_repair is not None:
                ledger_metrics["operation_limit_repair"] = operation_limit_repair
            _write_json(ledger_dir / "metrics.json", ledger_metrics)
            if first_publication:
                projection_result, overview, projection_metrics = _run_projection(
                    run_dir=run_dir,
                    background=background,
                    current_day=current_day,
                    cloud_settings=cloud_settings,
                    settings=settings,
                )
            else:
                current_context = load_current_layer2_context(history)
                base_sections = _parse_overview_sections(
                    _extract_model_facing_overview(current_context.context)
                )
                previous_projection_path = (
                    loaded.background_path.parent / "projection_result.json"
                )
                previous_projection = (
                    _read_json(previous_projection_path)
                    if previous_projection_path.is_file()
                    else None
                )
                _attach_previous_sources(base_sections, previous_projection)
                authorized_changes = _authorized_projection_changes(
                    previous=previous,
                    current=background,
                    applied_patch=applied,
                )
                _write_json(
                    run_dir / "rolling_base.json",
                    {
                        "context_source": current_context.diagnostics.get("context_source"),
                        "current_context_path": current_context.diagnostics.get(
                            "current_context_path"
                        ),
                        "character_count": len(current_context.context),
                    },
                )
                projection_result, overview, projection_metrics = _run_rolling_projection(
                    run_dir=run_dir,
                    base_sections=base_sections,
                    authorized_changes=authorized_changes,
                    current_day=current_day,
                    cloud_settings=cloud_settings,
                    settings=settings,
                )
            publication = _publish(
                layer2_root=layer2_root,
                run_id=run_id,
                current_day=current_day,
                diary_hash=diary_hash,
                background=background,
                projection_result=projection_result,
                overview=overview,
                provider_safe=provider_safe,
                ledger_metrics=ledger_metrics,
                projection_metrics=projection_metrics,
            )
            pruned = _prune_retained_runtime(
                layer2_root, current_day, settings.audit_retention_days
            )
            result = {
                "status": "published",
                "memory_day": memory_day,
                "publication": str(publication),
                "run_dir": str(run_dir),
                "ledger_api_attempt_count": attempt_count,
                "projection_api_attempt_count": projection_metrics["api_attempt_count"],
                "pruned": pruned,
            }
            _write_json(run_dir / "run.json", {**result, "completed_at": _iso_now()})
            return result
        except Exception as exc:
            _write_json(
                run_dir / "run.json",
                {
                    "status": "failed",
                    "memory_day": memory_day,
                    "failed_at": _iso_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "current_pointer_unchanged": True,
                },
            )
            raise


def launch_layer2_worker(
    memory_day: date,
    history_root: str | Path | None = None,
    config_path: str | Path = "conf.yaml",
) -> Layer2WorkerLaunch:
    history = resolve_character_history_root(history_root).resolve()
    layer2_root = history / LAYER2_DIRECTORY_NAME
    result_path = (
        layer2_root
        / "worker_results"
        / memory_day.isoformat()
        / (datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:10]}.json")
    )
    launcher_log = layer2_root / "launcher.log"
    launcher_log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-B",
        "-m",
        "src.open_llm_vtuber.memory.layer2_runtime",
        "--date",
        memory_day.isoformat(),
        "--history-root",
        str(history),
        "--config-path",
        str(Path(config_path).resolve()),
        "--result-path",
        str(result_path),
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with launcher_log.open("ab") as output:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
    return Layer2WorkerLaunch(
        process=process,
        memory_day=memory_day.isoformat(),
        result_path=result_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Daily Layer-2 memory worker")
    parser.add_argument("--date", required=True)
    parser.add_argument("--history-root")
    parser.add_argument("--config-path", default="conf.yaml")
    parser.add_argument("--result-path")
    arguments = parser.parse_args(argv)
    result_path = Path(arguments.result_path) if arguments.result_path else None
    try:
        result = run_daily_layer2_update(
            arguments.date,
            arguments.history_root,
            arguments.config_path,
        )
        exit_code = 0
    except Layer2RunAlreadyActive as exc:
        result = {
            "status": "already_running",
            "memory_day": arguments.date,
            "message": str(exc),
        }
        exit_code = 0
    except Exception as exc:
        result = {
            "status": "failed",
            "memory_day": arguments.date,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        exit_code = 1
    if result_path is not None:
        _write_json(result_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Layer2RunAlreadyActive",
    "Layer2RuntimeError",
    "Layer2WorkerLaunch",
    "layer2_publication_status",
    "launch_layer2_worker",
    "load_layer2_settings",
    "run_daily_layer2_update",
]
