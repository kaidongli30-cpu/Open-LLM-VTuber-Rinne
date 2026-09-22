"""Isolated chronological replay harness for Rinne layer-two background tests.

This tool never edits the live desktop-pet configuration, chat history, diaries,
or production memory files. It freezes diary inputs into an external experiment
directory and runs exactly one explicitly selected Ollama model at a time.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.1"
DEFAULT_END_DATE = date.today() - timedelta(days=1)
DEFAULT_START_DATE = DEFAULT_END_DATE - timedelta(days=6)
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_EXPERIMENT_ROOT = Path("tmp/layer2-prompt-experiment")
DEFAULT_SOURCE_DIARIES = Path("chat_history/rinne_01/diaries")
ASSET_DIR = Path(__file__).resolve().parent
SCHEMA_SOURCE = ASSET_DIR / "layer2_background_schema.json"
PROMPT_SOURCE = ASSET_DIR / "layer2_background_system_prompt.txt"
HANDOFF_SOURCE = (
    ASSET_DIR.parent
    / "docs"
    / "layer2_delta_successor_model_luna_handoff_2026-08-16.md"
)

CATEGORIES = (
    "stable_identity",
    "current_life_state",
    "ongoing_projects_and_study",
    "future_plans_and_commitments",
    "preferences_and_habits",
    "important_people_and_relationships",
    "relationship_with_rinne",
    "values_and_support_preferences",
    "constraints_and_environment",
    "recent_changes",
    "historical_transitions",
    "unknowns_and_do_not_assume",
)

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "mistral_24b": {
        "order": 1,
        "ollama_model": "mistral-small3.2:24b",
        "think": None,
        "label": "Mistral Small 3.2 24B",
    },
    "qwen3_32b": {
        "order": 2,
        "ollama_model": "qwen3:32B",
        "think": True,
        "label": "Qwen3 32B",
    },
    "qwen35_a3b": {
        "order": 3,
        "ollama_model": "qwen3.5:35b-a3b",
        # This run is a bounded JSON extraction task.  With thinking enabled,
        # qwen3.5:35b-a3b consumed the entire 2048-token output budget in its
        # hidden reasoning and returned an empty message.content.  Keep the
        # setting model-specific so other model trials remain unchanged.
        "think": False,
        "label": "Qwen3.5 35B-A3B",
    },
    "qwen35_27b": {
        "order": 4,
        "ollama_model": "qwen3.5:27b",
        "think": True,
        "label": "Qwen3.5 27B",
    },
    "gemma3_27b": {
        "order": 5,
        "ollama_model": "gemma3:27b",
        "think": None,
        "label": "Gemma 3 27B",
    },
    "gpt_oss_20b": {
        "order": 6,
        "ollama_model": "gpt-oss:20b",
        "think": "high",
        "label": "GPT-OSS 20B",
    },
}


class ExperimentError(RuntimeError):
    """Raised when an experiment safety or validation rule fails."""


@dataclass(frozen=True)
class ManifestDay:
    day: date
    status: str
    frozen_path: str | None
    sha256: str | None
    byte_count: int
    character_count: int


def _iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _days(start: date, end: date) -> list[date]:
    if end < start:
        raise ExperimentError("end date cannot be earlier than start date")
    result: list[date] = []
    current = start
    while current <= end:
        result.append(current)
        current += timedelta(days=1)
    return result


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ExperimentError(f"Expected a JSON object: {path}")
    return value


def _write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise ExperimentError(f"Refusing to overwrite existing file: {path}") from exc


def _write_text_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise ExperimentError(f"Refusing to overwrite existing file: {path}") from exc


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"Ollama request failed: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ExperimentError("Ollama returned a non-object response")
    return decoded


def _get_json(url: str, timeout: int = 10) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"Ollama health check failed: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ExperimentError("Ollama returned a non-object health response")
    return decoded


def _git_head() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ASSET_DIR.parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _model_inventory(base_url: str) -> tuple[str, dict[str, dict[str, Any]]]:
    version = _get_json(f"{base_url}/api/version").get("version", "unknown")
    tags = _get_json(f"{base_url}/api/tags").get("models", [])
    inventory: dict[str, dict[str, Any]] = {}
    for item in tags:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        inventory[item["name"].casefold()] = item
    return str(version), inventory


def _empty_background(previous_date: date) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "as_of_date": previous_date.isoformat(),
        "current_overview": "尚无可用的用户背景。",
        "sections": [
            {"category": category, "items": []} for category in CATEGORIES
        ],
        "daily_update": {
            "source_date": None,
            "added_fact_ids": [],
            "updated_fact_ids": [],
            "superseded_fact_ids": [],
            "removed_fact_ids": [],
            "no_change": True,
            "summary": "实验开始前的空背景。",
        },
        "uncertainty_warnings": [],
    }


def _items_by_id(background: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for section in background.get("sections", []):
        for item in section.get("items", []):
            fact_id = item.get("fact_id")
            if isinstance(fact_id, str):
                if fact_id in result:
                    raise ExperimentError(f"Duplicate fact_id: {fact_id}")
                result[fact_id] = item
    return result


def _categorized_items_by_id(
    background: dict[str, Any],
) -> dict[str, tuple[str, dict[str, Any]]]:
    result: dict[str, tuple[str, dict[str, Any]]] = {}
    for section in background.get("sections", []):
        category = section.get("category")
        for item in section.get("items", []):
            fact_id = item.get("fact_id")
            if isinstance(fact_id, str):
                if fact_id in result:
                    raise ExperimentError(f"Duplicate fact_id: {fact_id}")
                result[fact_id] = (category, item)
    return result


def _reconcile_daily_update(
    background: dict[str, Any],
    previous: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deterministically repair bookkeeping without changing semantic facts."""
    reconciled = copy.deepcopy(background)
    daily = reconciled.get("daily_update")
    if not isinstance(daily, dict):
        return reconciled, {
            "applied": False,
            "reason": "daily_update is not an object",
        }

    required_lists = (
        "added_fact_ids",
        "updated_fact_ids",
        "superseded_fact_ids",
        "removed_fact_ids",
    )
    if any(not isinstance(daily.get(key), list) for key in required_lists):
        return reconciled, {
            "applied": False,
            "reason": "one or more daily_update change lists are not arrays",
        }

    old = _categorized_items_by_id(previous)
    new = _categorized_items_by_id(reconciled)
    added = set(new) - set(old)
    removed = set(old) - set(new)
    changed = {
        fact_id
        for fact_id in set(new) & set(old)
        if new[fact_id] != old[fact_id]
    }
    superseded = {
        fact_id
        for fact_id in changed
        if new[fact_id][1].get("status") == "superseded"
    }
    updated = changed - superseded
    computed = {
        "added_fact_ids": sorted(added),
        "updated_fact_ids": sorted(updated),
        "superseded_fact_ids": sorted(superseded),
        "removed_fact_ids": sorted(removed),
        "no_change": not added and not removed and not changed,
    }
    declared = {
        key: copy.deepcopy(daily.get(key))
        for key in (*required_lists, "no_change")
    }
    applied = declared != computed
    if applied:
        daily.update(computed)
    return reconciled, {
        "applied": applied,
        "reason": "program-computed diff between previous and current facts",
        "model_declared": declared,
        "program_computed": computed,
    }


def _validate_background(
    background: dict[str, Any],
    *,
    expected_day: date,
    available_days: set[str],
    previous: dict[str, Any] | None,
    missing_diary: bool = False,
) -> list[str]:
    errors: list[str] = []
    required_root = {
        "schema_version",
        "as_of_date",
        "current_overview",
        "sections",
        "daily_update",
        "uncertainty_warnings",
    }
    if set(background) != required_root:
        errors.append(
            "root keys differ: "
            f"missing={sorted(required_root - set(background))}, "
            f"extra={sorted(set(background) - required_root)}"
        )
    if background.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version is not {SCHEMA_VERSION}")
    if background.get("as_of_date") != expected_day.isoformat():
        errors.append("as_of_date does not match the processed day")
    sections = background.get("sections")
    if not isinstance(sections, list):
        errors.append("sections is not a list")
        sections = []
    categories = [
        section.get("category")
        for section in sections
        if isinstance(section, dict)
    ]
    if categories != list(CATEGORIES):
        errors.append("sections do not contain the twelve categories in fixed order")

    if not isinstance(background.get("current_overview"), str):
        errors.append("current_overview is not a string")
    warnings = background.get("uncertainty_warnings")
    if not isinstance(warnings, list) or any(not isinstance(x, str) for x in warnings):
        errors.append("uncertainty_warnings is not a string list")

    ids: set[str] = set()
    for section in sections:
        if not isinstance(section, dict) or not isinstance(section.get("items"), list):
            errors.append("a section or section.items has an invalid type")
            continue
        for item in section["items"]:
            if not isinstance(item, dict):
                errors.append("a background item is not an object")
                continue
            required_item_keys = {
                "fact_id",
                "statement",
                "status",
                "first_observed_on",
                "last_updated_on",
                "source_dates",
                "evidence_type",
                "evidence_excerpts",
                "supersedes_fact_ids",
                "notes",
            }
            if set(item) != required_item_keys:
                errors.append("a background item has missing or extra keys")
            fact_id = item.get("fact_id")
            if not isinstance(fact_id, str) or not fact_id:
                errors.append("a fact_id is missing")
                continue
            if fact_id in ids:
                errors.append(f"duplicate fact_id: {fact_id}")
            ids.add(fact_id)
            source_dates = item.get("source_dates")
            if not isinstance(source_dates, list) or not source_dates:
                errors.append(f"{fact_id}: source_dates is empty")
            else:
                for source_day in source_dates:
                    if source_day not in available_days:
                        errors.append(
                            f"{fact_id}: source date is unavailable or outside frozen input: "
                            f"{source_day}"
                        )
                    if isinstance(source_day, str) and source_day > expected_day.isoformat():
                        errors.append(f"{fact_id}: future source date {source_day}")
            for key in ("first_observed_on", "last_updated_on"):
                value = item.get(key)
                if not isinstance(value, str) or value > expected_day.isoformat():
                    errors.append(f"{fact_id}: invalid or future {key}")
            excerpts = item.get("evidence_excerpts")
            if not isinstance(excerpts, list) or not excerpts:
                errors.append(f"{fact_id}: evidence_excerpts is empty")
            elif any(not isinstance(excerpt, str) or not excerpt for excerpt in excerpts):
                errors.append(f"{fact_id}: evidence_excerpts contains invalid text")
            if item.get("status") not in {
                "active",
                "historical",
                "superseded",
                "uncertain",
            }:
                errors.append(f"{fact_id}: invalid status")
            if item.get("evidence_type") not in {
                "diary_explicit_user_statement",
                "diary_reported_event",
                "diary_interpretation",
            }:
                errors.append(f"{fact_id}: invalid evidence_type")
            if not isinstance(item.get("statement"), str) or not item["statement"]:
                errors.append(f"{fact_id}: statement is empty or invalid")
            if not isinstance(item.get("notes"), str):
                errors.append(f"{fact_id}: notes is not a string")
            supersedes = item.get("supersedes_fact_ids")
            if not isinstance(supersedes, list) or any(
                not isinstance(value, str) for value in supersedes
            ):
                errors.append(f"{fact_id}: supersedes_fact_ids is invalid")

    daily = background.get("daily_update")
    if not isinstance(daily, dict):
        errors.append("daily_update is not an object")
        daily = {}
    required_daily_keys = {
        "source_date",
        "added_fact_ids",
        "updated_fact_ids",
        "superseded_fact_ids",
        "removed_fact_ids",
        "no_change",
        "summary",
    }
    if set(daily) != required_daily_keys:
        errors.append("daily_update has missing or extra keys")
    expected_source = None if missing_diary else expected_day.isoformat()
    if daily.get("source_date") != expected_source:
        errors.append("daily_update.source_date is incorrect")
    change_keys = (
        "added_fact_ids",
        "updated_fact_ids",
        "superseded_fact_ids",
        "removed_fact_ids",
    )
    for key in change_keys:
        values = daily.get(key)
        if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
            errors.append(f"daily_update.{key} is not a string list")
        elif len(values) != len(set(values)):
            errors.append(f"daily_update.{key} contains duplicates")

    change_sets = [set(daily.get(key, [])) for key in change_keys]
    for left_index, left in enumerate(change_sets):
        for right in change_sets[left_index + 1 :]:
            if left & right:
                errors.append("daily_update change lists are not disjoint")

    if previous is not None:
        try:
            old = _categorized_items_by_id(previous)
            new = _categorized_items_by_id(background)
        except ExperimentError as exc:
            errors.append(str(exc))
        else:
            added = set(new) - set(old)
            removed = set(old) - set(new)
            changed = {
                fact_id
                for fact_id in set(new) & set(old)
                if new[fact_id] != old[fact_id]
            }
            declared_added = set(daily.get("added_fact_ids", []))
            declared_removed = set(daily.get("removed_fact_ids", []))
            declared_changed = set(daily.get("updated_fact_ids", [])) | set(
                daily.get("superseded_fact_ids", [])
            )
            if added != declared_added:
                errors.append(
                    f"added_fact_ids mismatch: actual={sorted(added)}, "
                    f"declared={sorted(declared_added)}"
                )
            if removed != declared_removed:
                errors.append(
                    f"removed_fact_ids mismatch: actual={sorted(removed)}, "
                    f"declared={sorted(declared_removed)}"
                )
            if changed != declared_changed:
                errors.append(
                    f"changed fact ids mismatch: actual={sorted(changed)}, "
                    f"declared={sorted(declared_changed)}"
                )
            actual_no_change = not added and not removed and not changed
            if actual_no_change and (
                background.get("current_overview") != previous.get("current_overview")
                or background.get("uncertainty_warnings")
                != previous.get("uncertainty_warnings")
            ):
                errors.append(
                    "overview or uncertainty warnings changed without a fact change"
                )
            if daily.get("no_change") is not actual_no_change:
                errors.append(
                    f"no_change mismatch: actual={actual_no_change}, "
                    f"declared={daily.get('no_change')}"
                )
    if missing_diary and previous is not None:
        old = _items_by_id(previous)
        new = _items_by_id(background)
        if old != new:
            errors.append("missing-diary carry-forward changed facts")
    return errors


def _render_markdown(background: dict[str, Any], generation_mode: str) -> str:
    lines = [
        f"# 用户背景快照：{background['as_of_date']}",
        "",
        f"- 生成方式：`{generation_mode}`",
        f"- 当日来源：`{background['daily_update']['source_date']}`",
        f"- 是否无变化：`{background['daily_update']['no_change']}`",
        "",
        "## 当前概览",
        "",
        background["current_overview"],
        "",
    ]
    for section in background["sections"]:
        lines.extend([f"## {section['category']}", ""])
        if not section["items"]:
            lines.extend(["（空）", ""])
            continue
        for item in section["items"]:
            sources = ", ".join(item["source_dates"])
            evidence = " / ".join(item["evidence_excerpts"])
            lines.extend(
                [
                    f"- **{item['fact_id']}** [{item['status']}] {item['statement']}",
                    f"  - 首次/更新：{item['first_observed_on']} / {item['last_updated_on']}",
                    f"  - 来源：{sources}；类型：{item['evidence_type']}",
                    f"  - 证据摘录：{evidence}",
                    f"  - 替代：{', '.join(item['supersedes_fact_ids']) or '无'}",
                    f"  - 备注：{item['notes'] or '无'}",
                ]
            )
        lines.append("")
    update = background["daily_update"]
    lines.extend(
        [
            "## 当日变更",
            "",
            f"- 新增：{', '.join(update['added_fact_ids']) or '无'}",
            f"- 更新：{', '.join(update['updated_fact_ids']) or '无'}",
            f"- 失效/替代：{', '.join(update['superseded_fact_ids']) or '无'}",
            f"- 删除：{', '.join(update['removed_fact_ids']) or '无'}",
            f"- 摘要：{update['summary']}",
            "",
            "## 不确定与禁止推断",
            "",
        ]
    )
    warnings = background["uncertainty_warnings"]
    lines.extend([f"- {item}" for item in warnings] or ["（空）"])
    return "\n".join(lines) + "\n"


def _build_user_prompt(
    *, current_day: date, previous: dict[str, Any], diary_text: str
) -> str:
    previous_json = json.dumps(previous, ensure_ascii=False, indent=2)
    return (
        f"现在处理的唯一日期是 {current_day.isoformat()}。\n"
        "请根据上一日完整背景与今日唯一日记，输出截至今日的完整新背景。\n"
        "今日之前背景中没有出现、今日日记也没有提供的内容，一律不得新增。\n"
        "上一日背景如下：\n"
        "<previous_background>\n"
        f"{previous_json}\n"
        "</previous_background>\n\n"
        "今日日记如下：\n"
        "<today_diary>\n"
        f"{diary_text}\n"
        "</today_diary>\n\n"
        "请严格输出符合JSON Schema的对象。as_of_date和daily_update.source_date都必须为"
        f"{current_day.isoformat()}。"
    )


def _carry_forward(previous: dict[str, Any], current_day: date) -> dict[str, Any]:
    carried = copy.deepcopy(previous)
    carried["as_of_date"] = current_day.isoformat()
    carried["daily_update"] = {
        "source_date": None,
        "added_fact_ids": [],
        "updated_fact_ids": [],
        "superseded_fact_ids": [],
        "removed_fact_ids": [],
        "no_change": True,
        "summary": "当日没有日记输入；程序确定性沿用上一日背景，未调用模型。",
    }
    return carried


def _load_manifest(root: Path) -> list[ManifestDay]:
    manifest = _read_json(root / "input_manifest.json")
    days: list[ManifestDay] = []
    for item in manifest.get("days", []):
        days.append(
            ManifestDay(
                day=date.fromisoformat(item["date"]),
                status=item["status"],
                frozen_path=item.get("frozen_path"),
                sha256=item.get("sha256"),
                byte_count=int(item.get("byte_count", 0)),
                character_count=int(item.get("character_count", 0)),
            )
        )
    start = date.fromisoformat(manifest["start_date"])
    end = date.fromisoformat(manifest["end_date"])
    if [item.day for item in days] != _days(start, end):
        raise ExperimentError("Manifest does not contain the exact frozen date sequence")
    return days


def prepare_experiment(args: argparse.Namespace) -> None:
    root = args.experiment_root.resolve()
    source = args.source_diaries.resolve()
    if root.exists():
        raise ExperimentError(f"Experiment root already exists; refusing overwrite: {root}")
    if not source.is_dir():
        raise ExperimentError(f"Diary source directory not found: {source}")
    if not SCHEMA_SOURCE.is_file() or not PROMPT_SOURCE.is_file():
        raise ExperimentError("Experiment schema or prompt asset is missing")

    version, inventory = _model_inventory(args.ollama_url)
    missing_models = [
        spec["ollama_model"]
        for spec in MODEL_SPECS.values()
        if spec["ollama_model"].casefold() not in inventory
    ]
    if missing_models:
        raise ExperimentError(f"Required Ollama models are missing: {missing_models}")

    start_date: date = args.start_date
    end_date: date = args.end_date
    experiment_days = _days(start_date, end_date)
    root.mkdir(parents=True, exist_ok=False)
    frozen_dir = root / "frozen_inputs" / "diaries"
    assets_dir = root / "frozen_assets"
    frozen_dir.mkdir(parents=True)
    assets_dir.mkdir(parents=True)
    shutil.copy2(SCHEMA_SOURCE, assets_dir / SCHEMA_SOURCE.name)
    shutil.copy2(PROMPT_SOURCE, assets_dir / PROMPT_SOURCE.name)

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
        text = data.decode("utf-8")
        destination = frozen_dir / filename
        destination.write_bytes(data)
        manifest_days.append(
            {
                "date": current_day.isoformat(),
                "status": "available",
                "source_path": str(source_path),
                "frozen_path": str(destination.relative_to(root)),
                "sha256": _sha256_bytes(data),
                "byte_count": len(data),
                "character_count": len(text),
            }
        )

    model_records: dict[str, Any] = {}
    for key, spec in MODEL_SPECS.items():
        item = inventory[spec["ollama_model"].casefold()]
        model_records[key] = {
            **spec,
            "digest": item.get("digest"),
            "size": item.get("size"),
        }
        (root / "outputs" / key).mkdir(parents=True)
        review_dir = root / "reviews" / key
        review_dir.mkdir(parents=True)
        _write_text_new(
            review_dir / "REVIEW_TEMPLATE.md",
            _review_template(key, spec["label"], experiment_days),
        )

    manifest = {
        "experiment": "Rinne layer-two Experiment B chronological diary replay",
        "created_at": _iso_now(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "calendar_day_count": len(experiment_days),
        "available_diary_count": sum(
            item["status"] == "available" for item in manifest_days
        ),
        "missing_diary_count": sum(
            item["status"] == "missing" for item in manifest_days
        ),
        "days": manifest_days,
    }
    _write_json_new(root / "input_manifest.json", manifest)
    config = {
        "experiment": "layer2_experiment_b",
        "run_kind": args.run_kind,
        "created_at": _iso_now(),
        "private_data": True,
        "ollama_url": args.ollama_url,
        "ollama_version": version,
        "script_git_head": _git_head(),
        "schema_sha256": _sha256_file(assets_dir / SCHEMA_SOURCE.name),
        "prompt_sha256": _sha256_file(assets_dir / PROMPT_SOURCE.name),
        "generation_options": {
            "temperature": 0.0,
            "seed": 20260815,
            "num_ctx": 32768,
            "num_predict": 8192,
        },
        "request_timeout_seconds": 3600,
        "keep_alive_during_run": "30m",
        "models": model_records,
        "rule": "Run one model only; stop after RUN_COMPLETE and review before next.",
    }
    _write_json_new(root / "experiment_config.json", config)
    _write_text_new(
        root / "README.md",
        _experiment_readme(root, start_date, end_date, manifest_days, args.run_kind),
    )
    print(f"Prepared isolated experiment at {root}")
    print(
        f"Frozen {manifest['available_diary_count']} diaries across "
        f"{manifest['calendar_day_count']} calendar days; "
        f"missing={manifest['missing_diary_count']}"
    )


def _review_template(
    model_key: str, label: str, experiment_days: list[date]
) -> str:
    rows = "\n".join(
        f"| {day.isoformat()} | 未审查 |  |" for day in experiment_days
    )
    return (
        f"# {label}（{model_key}）逐日背景审查\n\n"
        "此文件只供人工审查，不会作为后续日期的模型输入。\n\n"
        "| 日期 | 结论 | 备注 |\n| --- | --- | --- |\n"
        f"{rows}\n\n"
        "## 严重错误\n\n- 无依据新增：\n- 当前/历史状态混淆：\n- 日记作者推测被写成用户事实：\n- 已有事实无故丢失或改写：\n\n"
        "## 总结\n\n- 完整性：\n- 准确性：\n- 状态维护：\n- 可作为下一阶段候选：\n"
    )


def _experiment_readme(
    root: Path,
    start_date: date,
    end_date: date,
    manifest_days: list[dict[str, Any]],
    run_kind: str,
) -> str:
    missing = [item["date"] for item in manifest_days if item["status"] == "missing"]
    missing_text = "、".join(missing) if missing else "无"
    return f"""# 凛祢第二层记忆实验B

此目录是私人、隔离、可续跑的模型评测工程，不属于Git仓库。

- 运行类型：{run_kind}
- 冻结范围：{start_date.isoformat()} 至 {end_date.isoformat()}
- 输入：只使用 `frozen_inputs/diaries/` 中的日记
- 缺失日期：{missing_text}（不调用模型，确定性沿用前一日背景）
- 输出：`outputs/<model_key>/snapshots/<日期>/`
- 原则：一次只运行一个模型；完成后立即停止并人工审查
- 不修改桌宠代码、配置、原始聊天、原始日记或正式记忆

实验根目录：`{root}`
"""


def preflight(args: argparse.Namespace, *, print_result: bool = True) -> dict[str, Any]:
    root = args.experiment_root.resolve()
    config = _read_json(root / "experiment_config.json")
    manifest = _load_manifest(root)
    schema_path = root / "frozen_assets" / SCHEMA_SOURCE.name
    prompt_path = root / "frozen_assets" / PROMPT_SOURCE.name
    errors: list[str] = []
    if _sha256_file(schema_path) != config.get("schema_sha256"):
        errors.append("frozen schema hash changed")
    if _sha256_file(prompt_path) != config.get("prompt_sha256"):
        errors.append("frozen prompt hash changed")
    for item in manifest:
        if item.status != "available":
            continue
        path = root / str(item.frozen_path)
        if not path.is_file() or _sha256_file(path) != item.sha256:
            errors.append(f"frozen diary changed or missing: {item.day}")
    version, inventory = _model_inventory(config["ollama_url"])
    model_status: dict[str, Any] = {}
    for key, spec in config["models"].items():
        current = inventory.get(spec["ollama_model"].casefold())
        ok = current is not None and current.get("digest") == spec.get("digest")
        model_status[key] = {
            "model": spec["ollama_model"],
            "available_with_frozen_digest": ok,
        }
        if not ok:
            errors.append(f"model missing or digest changed: {spec['ollama_model']}")
    result = {
        "ok": not errors,
        "checked_at": _iso_now(),
        "ollama_version_current": version,
        "ollama_version_prepared": config.get("ollama_version"),
        "errors": errors,
        "models": model_status,
    }
    if print_result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise ExperimentError("Preflight failed: " + "; ".join(errors))
    return result


def _snapshot_path(model_dir: Path, current_day: date) -> Path:
    return model_dir / "snapshots" / current_day.isoformat()


def _load_previous_snapshot(
    model_dir: Path, current_day: date, first_day: date
) -> dict[str, Any] | None:
    if current_day == first_day:
        return None
    previous_path = _snapshot_path(model_dir, current_day - timedelta(days=1))
    previous_file = previous_path / "background.json"
    if not previous_file.is_file():
        raise ExperimentError(
            f"Previous calendar-day snapshot is missing: {previous_file}"
        )
    return _read_json(previous_file)


class _Heartbeat:
    def __init__(self, label: str, interval_seconds: int = 30) -> None:
        self.label = label
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        started = time.monotonic()
        while not self._stop.wait(self.interval_seconds):
            elapsed = int(time.monotonic() - started)
            print(f"[heartbeat] {self.label}: waiting {elapsed}s", flush=True)

    def __enter__(self) -> _Heartbeat:
        self._thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def _save_snapshot(
    *,
    model_dir: Path,
    current_day: date,
    background: dict[str, Any],
    generation_mode: str,
    request_payload: dict[str, Any] | None,
    raw_response: dict[str, Any] | None,
    metrics: dict[str, Any],
) -> None:
    final_dir = _snapshot_path(model_dir, current_day)
    if final_dir.exists():
        raise ExperimentError(f"Refusing to overwrite snapshot: {final_dir}")
    snapshots_dir = final_dir.parent
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = snapshots_dir / f".tmp-{current_day.isoformat()}-{uuid.uuid4().hex}"
    temp_dir.mkdir()
    try:
        _write_json_new(temp_dir / "background.json", background)
        _write_text_new(
            temp_dir / "background.md",
            _render_markdown(background, generation_mode),
        )
        _write_json_new(temp_dir / "metrics.json", metrics)
        if request_payload is not None:
            _write_json_new(temp_dir / "request.json", request_payload)
        if raw_response is not None:
            _write_json_new(temp_dir / "raw_response.json", raw_response)
        temp_dir.rename(final_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _save_failure(
    model_dir: Path,
    current_day: date,
    payload: dict[str, Any],
) -> Path:
    failure_dir = model_dir / "failures"
    failure_dir.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        path = failure_dir / f"{current_day.isoformat()}_attempt_{attempt:03d}.json"
        if not path.exists():
            _write_json_new(path, payload)
            return path
        attempt += 1


def _unload_model(base_url: str, model: str) -> None:
    try:
        _post_json(
            f"{base_url}/api/generate",
            {"model": model, "keep_alive": 0},
            timeout=120,
        )
        print(f"Unloaded model: {model}", flush=True)
    except ExperimentError as exc:
        print(f"Warning: failed to unload {model}: {exc}", file=sys.stderr, flush=True)


def run_model(args: argparse.Namespace) -> None:
    root = args.experiment_root.resolve()
    preflight(args, print_result=False)
    config = _read_json(root / "experiment_config.json")
    manifest = _load_manifest(root)
    if args.model_key not in config["models"]:
        raise ExperimentError(f"Unknown model key: {args.model_key}")
    spec = config["models"][args.model_key]
    model_dir = root / "outputs" / args.model_key
    complete_path = model_dir / "RUN_COMPLETE.json"
    if complete_path.exists():
        raise ExperimentError(f"Model run is already complete: {complete_path}")
    existing = [
        item.day for item in manifest if _snapshot_path(model_dir, item.day).exists()
    ]
    if existing and not args.resume:
        raise ExperimentError(
            "Snapshots already exist. Use --resume to continue without overwriting."
        )
    event_log = model_dir / "run_events.jsonl"
    _append_jsonl(
        event_log,
        {
            "event": "run_started" if not existing else "run_resumed",
            "at": _iso_now(),
            "model_key": args.model_key,
            "model": spec["ollama_model"],
            "existing_snapshot_count": len(existing),
        },
    )

    schema = _read_json(root / "frozen_assets" / SCHEMA_SOURCE.name)
    system_prompt = (root / "frozen_assets" / PROMPT_SOURCE.name).read_text(
        encoding="utf-8"
    )
    available_days = {
        item.day.isoformat() for item in manifest if item.status == "available"
    }
    started_monotonic = time.monotonic()
    completed_this_run = 0
    model = spec["ollama_model"]
    base_url = config["ollama_url"]
    try:
        first_day = manifest[0].day
        for manifest_day in manifest:
            current_day = manifest_day.day
            snapshot_dir = _snapshot_path(model_dir, current_day)
            if snapshot_dir.exists():
                print(f"[skip] {current_day}: snapshot already exists", flush=True)
                continue
            previous = _load_previous_snapshot(model_dir, current_day, first_day)
            previous_for_prompt = previous or _empty_background(
                current_day - timedelta(days=1)
            )
            if manifest_day.status == "missing":
                if previous is None:
                    raise ExperimentError("Cannot carry forward a missing first day")
                background = _carry_forward(previous, current_day)
                errors = _validate_background(
                    background,
                    expected_day=current_day,
                    available_days=available_days,
                    previous=previous,
                    missing_diary=True,
                )
                if errors:
                    raise ExperimentError(
                        f"Carry-forward validation failed on {current_day}: {errors}"
                    )
                _save_snapshot(
                    model_dir=model_dir,
                    current_day=current_day,
                    background=background,
                    generation_mode="carry_forward_missing_diary",
                    request_payload=None,
                    raw_response=None,
                    metrics={
                        "date": current_day.isoformat(),
                        "generation_mode": "carry_forward_missing_diary",
                        "model_called": False,
                        "created_at": _iso_now(),
                    },
                )
                completed_this_run += 1
                print(f"[carry] {current_day}: no diary; model not called", flush=True)
                continue

            diary_path = root / str(manifest_day.frozen_path)
            if _sha256_file(diary_path) != manifest_day.sha256:
                raise ExperimentError(f"Frozen diary hash changed: {diary_path}")
            diary_text = diary_path.read_text(encoding="utf-8")
            user_prompt = _build_user_prompt(
                current_day=current_day,
                previous=previous_for_prompt,
                diary_text=diary_text,
            )
            request_payload: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "format": schema,
                "options": config["generation_options"],
                "keep_alive": config["keep_alive_during_run"],
            }
            if spec.get("think") is not None:
                request_payload["think"] = spec["think"]
            print(f"[start] {current_day}: {model}", flush=True)
            call_started = time.monotonic()
            raw_response: dict[str, Any] | None = None
            reconciliation: dict[str, Any] | None = None
            try:
                with _Heartbeat(f"{model} {current_day}"):
                    raw_response = _post_json(
                        f"{base_url}/api/chat",
                        request_payload,
                        timeout=int(config["request_timeout_seconds"]),
                    )
                content = raw_response.get("message", {}).get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ExperimentError("Ollama response has empty message.content")
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ExperimentError("Structured output is not a JSON object")
                parsed, reconciliation = _reconcile_daily_update(
                    parsed, previous_for_prompt
                )
                errors = _validate_background(
                    parsed,
                    expected_day=current_day,
                    available_days=available_days,
                    previous=previous_for_prompt,
                )
                if errors:
                    raise ExperimentError("; ".join(errors))
            except (ExperimentError, json.JSONDecodeError) as exc:
                failure_path = _save_failure(
                    model_dir,
                    current_day,
                    {
                        "date": current_day.isoformat(),
                        "failed_at": _iso_now(),
                        "model": model,
                        "error": str(exc),
                        "elapsed_seconds": round(time.monotonic() - call_started, 3),
                        "request": request_payload,
                        "raw_response": raw_response,
                        "bookkeeping_reconciliation": reconciliation,
                    },
                )
                _append_jsonl(
                    event_log,
                    {
                        "event": "day_failed",
                        "at": _iso_now(),
                        "date": current_day.isoformat(),
                        "error": str(exc),
                        "failure_path": str(failure_path),
                    },
                )
                raise ExperimentError(
                    f"Stopped on {current_day}; failure preserved at {failure_path}: {exc}"
                ) from exc

            elapsed = time.monotonic() - call_started
            generation_mode = (
                "ollama_structured_output_bookkeeping_reconciled"
                if reconciliation and reconciliation.get("applied")
                else "ollama_structured_output"
            )
            metrics = {
                "date": current_day.isoformat(),
                "generation_mode": generation_mode,
                "model_called": True,
                "model": model,
                "model_digest": spec["digest"],
                "think_setting": spec.get("think"),
                "elapsed_wall_seconds": round(elapsed, 3),
                "ollama_total_seconds": round(
                    float(raw_response.get("total_duration", 0)) / 1_000_000_000, 3
                ),
                "ollama_load_seconds": round(
                    float(raw_response.get("load_duration", 0)) / 1_000_000_000, 3
                ),
                "prompt_eval_count": raw_response.get("prompt_eval_count"),
                "prompt_eval_seconds": round(
                    float(raw_response.get("prompt_eval_duration", 0))
                    / 1_000_000_000,
                    3,
                ),
                "eval_count": raw_response.get("eval_count"),
                "eval_seconds": round(
                    float(raw_response.get("eval_duration", 0)) / 1_000_000_000,
                    3,
                ),
                "thinking_character_count": len(
                    str(raw_response.get("message", {}).get("thinking", ""))
                ),
                "content_character_count": len(
                    str(raw_response.get("message", {}).get("content", ""))
                ),
                "bookkeeping_reconciliation": reconciliation,
                "completed_at": _iso_now(),
            }
            _save_snapshot(
                model_dir=model_dir,
                current_day=current_day,
                background=parsed,
                generation_mode=generation_mode,
                request_payload=request_payload,
                raw_response=raw_response,
                metrics=metrics,
            )
            completed_this_run += 1
            _append_jsonl(
                event_log,
                {
                    "event": "day_completed",
                    "at": _iso_now(),
                    "date": current_day.isoformat(),
                    "elapsed_wall_seconds": round(elapsed, 3),
                    "fact_count": len(_items_by_id(parsed)),
                    "bookkeeping_reconciled": bool(
                        reconciliation and reconciliation.get("applied")
                    ),
                },
            )
            if reconciliation and reconciliation.get("applied"):
                print(
                    f"[reconcile] {current_day}: daily_update bookkeeping corrected",
                    flush=True,
                )
            print(
                f"[done] {current_day}: {elapsed:.1f}s, "
                f"facts={len(_items_by_id(parsed))}",
                flush=True,
            )

        validation = validate_model_output(root, args.model_key)
        complete = {
            "status": "complete",
            "completed_at": _iso_now(),
            "model_key": args.model_key,
            "model": model,
            "calendar_snapshots": validation["snapshot_count"],
            "model_generated_snapshots": validation["model_generated_count"],
            "carry_forward_snapshots": validation["carry_forward_count"],
            "total_wall_seconds_this_process": round(
                time.monotonic() - started_monotonic, 3
            ),
            "completed_this_process": completed_this_run,
            "validation": validation,
            "next_action": "STOP. Review this model before running another model.",
        }
        _write_json_new(complete_path, complete)
        _write_text_new(model_dir / "RUN_REPORT.md", _run_report(root, args.model_key))
        _append_jsonl(event_log, {"event": "run_completed", **complete})
        print(f"MODEL_COMPLETE: {args.model_key}", flush=True)
        print(f"Review output at: {model_dir}", flush=True)
        print("STOP NOW. Do not run the next model before human review.", flush=True)
    finally:
        _unload_model(base_url, model)


def repair_failure(args: argparse.Namespace) -> None:
    """Promote a preserved response only when bookkeeping is the sole defect."""
    root = args.experiment_root.resolve()
    preflight(args, print_result=False)
    config = _read_json(root / "experiment_config.json")
    manifest = _load_manifest(root)
    manifest_by_day = {item.day: item for item in manifest}
    current_day = args.date
    manifest_day = manifest_by_day.get(current_day)
    if manifest_day is None:
        raise ExperimentError(f"Date is outside the frozen manifest: {current_day}")
    if manifest_day.status != "available":
        raise ExperimentError(f"Cannot repair a missing-diary day: {current_day}")
    if args.model_key not in config["models"]:
        raise ExperimentError(f"Unknown model key: {args.model_key}")

    spec = config["models"][args.model_key]
    model_dir = root / "outputs" / args.model_key
    if _snapshot_path(model_dir, current_day).exists():
        raise ExperimentError(f"Snapshot already exists for {current_day}")
    failure_dir = model_dir / "failures"
    if args.attempt is None:
        candidates = sorted(
            failure_dir.glob(f"{current_day.isoformat()}_attempt_*.json")
        )
        if not candidates:
            raise ExperimentError(f"No preserved failure found for {current_day}")
        failure_path = candidates[-1]
    else:
        failure_path = failure_dir / (
            f"{current_day.isoformat()}_attempt_{args.attempt:03d}.json"
        )
        if not failure_path.is_file():
            raise ExperimentError(f"Preserved failure not found: {failure_path}")

    failure = _read_json(failure_path)
    if failure.get("date") != current_day.isoformat():
        raise ExperimentError("Failure date does not match requested repair date")
    if failure.get("model") != spec["ollama_model"]:
        raise ExperimentError("Failure model does not match requested model key")
    raw_response = failure.get("raw_response")
    if not isinstance(raw_response, dict):
        raise ExperimentError("Preserved failure has no raw Ollama response")
    content = raw_response.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ExperimentError("Preserved response has empty message.content")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ExperimentError(f"Preserved response is invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ExperimentError("Preserved structured output is not a JSON object")

    first_day = manifest[0].day
    previous = _load_previous_snapshot(model_dir, current_day, first_day)
    previous_for_diff = previous or _empty_background(
        current_day - timedelta(days=1)
    )
    available_days = {
        item.day.isoformat() for item in manifest if item.status == "available"
    }
    before_errors = _validate_background(
        parsed,
        expected_day=current_day,
        available_days=available_days,
        previous=previous_for_diff,
    )
    bookkeeping_prefixes = (
        "added_fact_ids mismatch:",
        "removed_fact_ids mismatch:",
        "changed fact ids mismatch:",
        "no_change mismatch:",
    )
    non_bookkeeping_errors = [
        error
        for error in before_errors
        if not error.startswith(bookkeeping_prefixes)
    ]
    if not before_errors:
        raise ExperimentError("Preserved response has no validation defect to repair")
    if non_bookkeeping_errors:
        raise ExperimentError(
            "Refusing semantic or structural repair: "
            + "; ".join(non_bookkeeping_errors)
        )

    repaired, reconciliation = _reconcile_daily_update(
        parsed, previous_for_diff
    )
    if not reconciliation.get("applied"):
        raise ExperimentError("Bookkeeping reconciliation made no change")
    after_errors = _validate_background(
        repaired,
        expected_day=current_day,
        available_days=available_days,
        previous=previous_for_diff,
    )
    if after_errors:
        raise ExperimentError(
            "Repaired response still fails validation: " + "; ".join(after_errors)
        )

    elapsed = float(failure.get("elapsed_seconds", 0))
    repair_audit = {
        "source_failure_path": str(failure_path),
        "source_failure_sha256": _sha256_file(failure_path),
        "original_validation_errors": before_errors,
        "reconciliation": reconciliation,
        "repaired_at": _iso_now(),
        "runner_git_head": _git_head(),
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
    }
    metrics = {
        "date": current_day.isoformat(),
        "generation_mode": "ollama_structured_output_bookkeeping_repaired",
        "model_called": True,
        "model": spec["ollama_model"],
        "model_digest": spec["digest"],
        "think_setting": spec.get("think"),
        "elapsed_wall_seconds": elapsed,
        "ollama_total_seconds": round(
            float(raw_response.get("total_duration", 0)) / 1_000_000_000, 3
        ),
        "ollama_load_seconds": round(
            float(raw_response.get("load_duration", 0)) / 1_000_000_000, 3
        ),
        "prompt_eval_count": raw_response.get("prompt_eval_count"),
        "prompt_eval_seconds": round(
            float(raw_response.get("prompt_eval_duration", 0)) / 1_000_000_000,
            3,
        ),
        "eval_count": raw_response.get("eval_count"),
        "eval_seconds": round(
            float(raw_response.get("eval_duration", 0)) / 1_000_000_000,
            3,
        ),
        "thinking_character_count": len(
            str(raw_response.get("message", {}).get("thinking", ""))
        ),
        "content_character_count": len(content),
        "bookkeeping_reconciliation": reconciliation,
        "repair_audit": repair_audit,
        "completed_at": _iso_now(),
    }
    request_payload = failure.get("request")
    if request_payload is not None and not isinstance(request_payload, dict):
        raise ExperimentError("Preserved request is not a JSON object")
    _save_snapshot(
        model_dir=model_dir,
        current_day=current_day,
        background=repaired,
        generation_mode="ollama_structured_output_bookkeeping_repaired",
        request_payload=request_payload,
        raw_response=raw_response,
        metrics=metrics,
    )
    event = {
        "event": "day_repaired_from_preserved_failure",
        "at": _iso_now(),
        "date": current_day.isoformat(),
        "source_failure_path": str(failure_path),
        "bookkeeping_reconciliation": reconciliation,
        "fact_count": len(_items_by_id(repaired)),
    }
    _append_jsonl(model_dir / "run_events.jsonl", event)
    print(json.dumps(event, ensure_ascii=False, indent=2))


def validate_model_output(root: Path, model_key: str) -> dict[str, Any]:
    manifest = _load_manifest(root)
    available_days = {
        item.day.isoformat() for item in manifest if item.status == "available"
    }
    model_dir = root / "outputs" / model_key
    first_day = manifest[0].day
    previous: dict[str, Any] | None = _empty_background(
        first_day - timedelta(days=1)
    )
    model_generated = 0
    carried = 0
    total_facts: list[int] = []
    total_seconds = 0.0
    errors: list[str] = []
    for manifest_day in manifest:
        snapshot_dir = _snapshot_path(model_dir, manifest_day.day)
        background_path = snapshot_dir / "background.json"
        metrics_path = snapshot_dir / "metrics.json"
        if not background_path.is_file() or not metrics_path.is_file():
            errors.append(f"missing snapshot files: {manifest_day.day}")
            break
        background = _read_json(background_path)
        metrics = _read_json(metrics_path)
        day_errors = _validate_background(
            background,
            expected_day=manifest_day.day,
            available_days=available_days,
            previous=previous,
            missing_diary=manifest_day.status == "missing",
        )
        errors.extend(f"{manifest_day.day}: {item}" for item in day_errors)
        if metrics.get("model_called"):
            model_generated += 1
            total_seconds += float(metrics.get("elapsed_wall_seconds", 0))
        else:
            carried += 1
        total_facts.append(len(_items_by_id(background)))
        previous = background
    result = {
        "ok": not errors and len(total_facts) == len(manifest),
        "snapshot_count": len(total_facts),
        "model_generated_count": model_generated,
        "carry_forward_count": carried,
        "final_fact_count": total_facts[-1] if total_facts else 0,
        "max_fact_count": max(total_facts, default=0),
        "model_call_wall_seconds": round(total_seconds, 3),
        "errors": errors,
    }
    if errors:
        raise ExperimentError("Output validation failed: " + "; ".join(errors))
    return result


def validate_command(args: argparse.Namespace) -> None:
    root = args.experiment_root.resolve()
    preflight(args, print_result=False)
    result = validate_model_output(root, args.model_key)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def seal_experiment(args: argparse.Namespace) -> None:
    """Seal a prepared root after the runner itself has been committed."""
    root = args.experiment_root.resolve()
    preflight_result = preflight(args, print_result=False)
    if not HANDOFF_SOURCE.is_file():
        raise ExperimentError(f"Missing Luna handoff source: {HANDOFF_SOURCE}")

    handoff_text = HANDOFF_SOURCE.read_text(encoding="utf-8")
    handoff_destination = root / "LUNA_HANDOFF.md"
    _write_text_new(handoff_destination, handoff_text)
    seal = {
        "sealed_at": _iso_now(),
        "authoritative_repository_head": _git_head(),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "luna_handoff_path": str(handoff_destination),
        "luna_handoff_sha256": _sha256_file(handoff_destination),
        "preflight": preflight_result,
        "note": (
            "This post-commit seal is authoritative. Any repository head saved "
            "during prepare records the pre-commit preparation state."
        ),
    }
    _write_json_new(root / "PREPARATION_SEAL.json", seal)
    print(json.dumps(seal, ensure_ascii=False, indent=2))


def _run_report(root: Path, model_key: str) -> str:
    config = _read_json(root / "experiment_config.json")
    model_dir = root / "outputs" / model_key
    spec = config["models"][model_key]
    lines = [
        f"# {spec['label']} 逐日运行报告",
        "",
        f"- 模型标签：`{spec['ollama_model']}`",
        f"- 模型摘要：`{spec['digest']}`",
        "- 结论：仅表示运行与结构校验完成，不表示背景内容质量合格。",
        "",
        "| 日期 | 方式 | 事实数 | 新增 | 更新/替代 | 耗时秒 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for manifest_day in _load_manifest(root):
        current_day = manifest_day.day
        snapshot = _snapshot_path(model_dir, current_day)
        background = _read_json(snapshot / "background.json")
        metrics = _read_json(snapshot / "metrics.json")
        update = background["daily_update"]
        changed = len(update["updated_fact_ids"]) + len(
            update["superseded_fact_ids"]
        )
        lines.append(
            f"| {current_day.isoformat()} | {metrics['generation_mode']} | "
            f"{len(_items_by_id(background))} | {len(update['added_fact_ids'])} | "
            f"{changed} | {metrics.get('elapsed_wall_seconds', 0)} |"
        )
    lines.extend(
        [
            "",
            "下一步：停止运行其他模型，审查 `snapshots/` 与对应 `reviews/` 模板。",
            "",
        ]
    )
    return "\n".join(lines)


def list_models_command(_args: argparse.Namespace) -> None:
    for key, spec in sorted(MODEL_SPECS.items(), key=lambda item: item[1]["order"]):
        print(f"{spec['order']}. {key}: {spec['ollama_model']} ({spec['label']})")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
    )
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--source-diaries",
        type=Path,
        default=DEFAULT_SOURCE_DIARIES,
    )
    prepare_parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=DEFAULT_START_DATE,
    )
    prepare_parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=DEFAULT_END_DATE,
    )
    prepare_parser.add_argument(
        "--run-kind",
        choices=("prompt_pilot", "official_comparison"),
        default="prompt_pilot",
    )
    prepare_parser.set_defaults(handler=prepare_experiment)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.set_defaults(handler=preflight)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model-key", required=True, choices=MODEL_SPECS)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.set_defaults(handler=run_model)

    repair_parser = subparsers.add_parser("repair-failure")
    repair_parser.add_argument("--model-key", required=True, choices=MODEL_SPECS)
    repair_parser.add_argument("--date", required=True, type=date.fromisoformat)
    repair_parser.add_argument("--attempt", type=int)
    repair_parser.set_defaults(handler=repair_failure)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--model-key", required=True, choices=MODEL_SPECS)
    validate_parser.set_defaults(handler=validate_command)

    seal_parser = subparsers.add_parser("seal")
    seal_parser.set_defaults(handler=seal_experiment)

    list_parser = subparsers.add_parser("list-models")
    list_parser.set_defaults(handler=list_models_command)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        args.handler(args)
    except ExperimentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
