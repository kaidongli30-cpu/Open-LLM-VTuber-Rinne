"""Generate and publish one day's child events with a configured LLM.

This module is deliberately separate from the live conversation pipeline.  The
backend may launch it after yesterday's diary is available, but event results
are not injected into chat.  A date is published only after strict validation;
failures remain in the local run directory for a later retry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from ..config_manager.daily_child_event import DailyChildEventGenerationConfig
from .daily_child_event_providers import (
    DailyChildEventGenerationResult,
    default_daily_child_event_settings,
    generate_with_provider,
    load_daily_child_event_settings,
)


EVENT_DIRECTORY_NAME = "\u4e8b\u4ef6TXT"
MOJIBAKE_EVENT_DIRECTORY_NAME = "\u6d5c\u5b29\u6b22TXT"
MANIFEST_FILENAME = "\u53d1\u5e03\u6e05\u5355.json"
PROMPT_VERSION = "v2.3-neutral"
SCHEMA_VERSION = 2
DEFAULT_WORK_ROOT = Path(r".\rinne_daily_child_event_pipeline")
PROMPT_ROOT = Path(__file__).with_name("prompts")
SYSTEM_PROMPT_PATH = PROMPT_ROOT / "child_event_system_v23.txt"
TASK_PROMPT_PATH = PROMPT_ROOT / "child_event_task_v23.txt"
CHECK_FIELDS = (
    "source_fact",
    "subject_check",
    "time_check",
    "location_check",
    "salience_check",
    "inference_check",
)
MEMORY_FIELDS = (
    "type",
    "title",
    "description",
    "why_independent",
    *CHECK_FIELDS,
)
DISCARDED_FIELDS = ("detail", "decision", "target_title", "reason")


class ChildEventError(RuntimeError):
    """Base error for a daily child-event run."""


class ValidationError(ChildEventError):
    """The model response or staged publication failed validation."""


class PublicationConflict(ChildEventError):
    """A date already contains data that this worker must not overwrite."""


class RunAlreadyActive(ChildEventError):
    """Another worker is already processing the same date."""


@dataclass(frozen=True)
class ChildEventWorkerLaunch:
    """Process handle plus its durable machine-readable result path."""

    process: subprocess.Popen[bytes]
    memory_day: str
    result_path: Path


def _string_schema(*, enum: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string"}
    if enum is not None:
        schema["enum"] = enum
    return schema


def event_json_schema(memory_day: str) -> dict[str, Any]:
    memory_properties = {
        field: _string_schema(
            enum=["\u4e8b\u4ef6", "\u77ac\u95f4"] if field == "type" else None
        )
        for field in MEMORY_FIELDS
    }
    discarded_properties = {
        field: _string_schema(
            enum=["merged", "routine_background", "future_plan", "not_salient"]
            if field == "decision"
            else None
        )
        for field in DISCARDED_FIELDS
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "memory_day",
            "day_richness",
            "memories",
            "merged_or_discarded",
        ],
        "properties": {
            "memory_day": {"type": "string", "enum": [memory_day]},
            "day_richness": {"type": "string"},
            "memories": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(MEMORY_FIELDS),
                    "properties": memory_properties,
                },
            },
            "merged_or_discarded": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(DISCARDED_FIELDS),
                    "properties": discarded_properties,
                },
            },
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary basename short.  Repeating a long Chinese event title
    # here can cross the legacy Windows MAX_PATH limit even when the final path
    # itself is valid.
    temporary = path.with_name(f".{uuid.uuid4().hex[:12]}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{timestamp} {message}\n")


def _load_prompts(memory_day: str, diary_text: str) -> tuple[str, str]:
    system = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    task = TASK_PROMPT_PATH.read_text(encoding="utf-8")
    task = task.replace("{MEMORY_DAY}", memory_day).replace("{DIARY_TEXT}", diary_text)
    return system, task


def generate_with_ollama(
    memory_day: str,
    system_prompt: str,
    task_prompt: str,
    run_dir: Path,
) -> str:
    """Compatibility wrapper for callers that explicitly want default Ollama."""

    result = generate_with_provider(
        memory_day,
        system_prompt,
        task_prompt,
        run_dir,
        default_daily_child_event_settings(),
        event_json_schema(memory_day),
    )
    return result.text


def _require_exact_fields(
    value: dict[str, Any], required: tuple[str, ...], label: str
) -> None:
    actual = set(value)
    expected = set(required)
    if actual != expected:
        raise ValidationError(
            f"{label}_fields_mismatch:missing={sorted(expected - actual)}:"
            f"extra={sorted(actual - expected)}"
        )


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label}_must_be_nonempty_string")
    return value.strip()


def validate_model_response(text: str, memory_day: str) -> dict[str, Any]:
    """Parse JSON without heuristic repairs and validate every published field."""

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"invalid_json:line={exc.lineno}:column={exc.colno}:{exc.msg}"
        ) from exc
    if not isinstance(raw, dict):
        raise ValidationError("root_must_be_object")
    _require_exact_fields(
        raw,
        ("memory_day", "day_richness", "memories", "merged_or_discarded"),
        "root",
    )
    if raw["memory_day"] != memory_day:
        raise ValidationError(
            f"memory_day_mismatch:{raw['memory_day']!r}!={memory_day!r}"
        )
    day_richness = _require_nonempty_string(raw["day_richness"], "day_richness")
    memories = raw["memories"]
    if not isinstance(memories, list) or not 1 <= len(memories) <= 5:
        raise ValidationError("memories_count_must_be_1_to_5")
    normalized_memories: list[dict[str, str]] = []
    for index, item in enumerate(memories, 1):
        if not isinstance(item, dict):
            raise ValidationError(f"memory_{index}_must_be_object")
        _require_exact_fields(item, MEMORY_FIELDS, f"memory_{index}")
        normalized = {
            field: _require_nonempty_string(item[field], f"memory_{index}.{field}")
            for field in MEMORY_FIELDS
        }
        if normalized["type"] not in {"\u4e8b\u4ef6", "\u77ac\u95f4"}:
            raise ValidationError(f"memory_{index}.type_invalid")
        normalized_memories.append(normalized)
    discarded = raw["merged_or_discarded"]
    if not isinstance(discarded, list) or len(discarded) > 8:
        raise ValidationError("merged_or_discarded_must_be_list_max_8")
    normalized_discarded: list[dict[str, str]] = []
    for index, item in enumerate(discarded, 1):
        if not isinstance(item, dict):
            raise ValidationError(f"discarded_{index}_must_be_object")
        _require_exact_fields(item, DISCARDED_FIELDS, f"discarded_{index}")
        normalized: dict[str, str] = {}
        for field in DISCARDED_FIELDS:
            value = item[field]
            if not isinstance(value, str):
                raise ValidationError(f"discarded_{index}.{field}_must_be_string")
            normalized[field] = value.strip()
        if not normalized["detail"] or not normalized["reason"]:
            raise ValidationError(f"discarded_{index}_detail_and_reason_required")
        if normalized["decision"] not in {
            "merged",
            "routine_background",
            "future_plan",
            "not_salient",
        }:
            raise ValidationError(f"discarded_{index}.decision_invalid")
        normalized_discarded.append(normalized)
    return {
        "memory_day": memory_day,
        "day_richness": day_richness,
        "memories": normalized_memories,
        "merged_or_discarded": normalized_discarded,
    }


def _safe_title(title: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]', "，", title).strip().rstrip(".")
    if not cleaned:
        raise ValidationError("empty_filename_after_sanitization")
    if len(cleaned) > 80:
        suffix = hashlib.sha256(title.encode("utf-8")).hexdigest()[:8]
        cleaned = f"{cleaned[:70]}_{suffix}"
    return cleaned


def _render_event(memory_day: str, item: dict[str, str]) -> str:
    label_date = date.fromisoformat(memory_day)
    date_label = f"{label_date.year}年{label_date.month}月{label_date.day}日"
    return "\n".join(
        [
            f"记忆类型：{item['type']}",
            f"事件名称：{item['title']}",
            f"时间：{date_label}",
            "",
            item["description"],
            "",
            "来源：",
            f"日记：chat_history/rinne_01/diaries/diary_{memory_day}.txt",
            "",
        ]
    )


def _render_review(
    memory_day: str,
    normalized: dict[str, Any],
    elapsed_seconds: float,
    generation_metadata: dict[str, Any] | None = None,
) -> str:
    label_date = date.fromisoformat(memory_day)
    date_label = f"{label_date.year}年{label_date.month}月{label_date.day}日"
    metadata = generation_metadata or {
        "llm_provider": "ollama_llm",
        "model": "mistral-small3.2:24b",
    }
    provider = str(metadata.get("llm_provider", "unknown_provider"))
    model = str(metadata.get("model", "unknown_model"))
    lines = [
        f"# {date_label} 事件整理：{provider} / {model} + v2.3",
        "",
        "本文件由每日子事件后台程序确定性渲染，供人工抽查。",
        "",
        f"- 模型接口：{provider}",
        f"- 模型：{model}",
        f"- 日记：chat_history/rinne_01/diaries/diary_{memory_day}.txt",
        f"- 语义调用耗时：{elapsed_seconds:.3f} 秒",
        f"- 记忆数量：{len(normalized['memories'])}",
        "- 程序状态：parsed",
        "- 机械问题：无",
        "",
    ]
    for index, item in enumerate(normalized["memories"], 1):
        lines.extend(
            [
                f"### {index}. {item['title']}",
                "",
                f"类型：{item['type']}",
                "",
                item["description"],
                "",
                f"独立理由：{item['why_independent']}",
                "",
                "核对字段：",
                *[f"- {field}：{item[field]}" for field in CHECK_FIELDS],
                "",
            ]
        )
    lines.extend(["### 模型主动合并或舍弃的主要细节", ""])
    if normalized["merged_or_discarded"]:
        for item in normalized["merged_or_discarded"]:
            lines.append(f"- {item['detail']}：{item['decision']}；{item['reason']}")
    else:
        lines.append("- 未提供。")
    lines.append("")
    return "\n".join(lines)


def _manifest_is_complete(target: Path, diary_hash: str | None = None) -> bool:
    manifest_path = target / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "published":
            return False
        if diary_hash is not None and manifest.get("diary_sha256") != diary_hash:
            return False
        event_dir = target / EVENT_DIRECTORY_NAME
        if not event_dir.is_dir():
            return False
        if {item.name for item in target.iterdir() if item.is_dir()} != {
            EVENT_DIRECTORY_NAME
        }:
            return False
        expected = manifest.get("event_files")
        if not isinstance(expected, list) or not expected:
            return False
        actual_files = sorted(event_dir.glob("*.txt"), key=lambda item: item.name)
        if len(actual_files) != len(expected):
            return False
        expected_by_name = {
            item["name"]: item["sha256"]
            for item in expected
            if isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and isinstance(item.get("sha256"), str)
        }
        if set(expected_by_name) != {item.name for item in actual_files}:
            return False
        return all(
            sha256_file(item) == expected_by_name[item.name] for item in actual_files
        )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
        return False


def _verify_event_directory_name(path: Path) -> None:
    if path.name != EVENT_DIRECTORY_NAME:
        raise ValidationError(f"wrong_event_directory_name:{path.name!r}")
    if tuple(ord(character) for character in path.name[:2]) != (0x4E8B, 0x4EF6):
        raise ValidationError("event_directory_unicode_codepoints_invalid")


def _stage_publication(
    staging: Path,
    memory_day: str,
    diary_path: Path,
    normalized: dict[str, Any],
    elapsed_seconds: float,
    generation_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event_dir = staging / EVENT_DIRECTORY_NAME
    event_dir.mkdir(parents=True, exist_ok=False)
    _verify_event_directory_name(event_dir)
    event_files: list[dict[str, str]] = []
    seen_names: set[str] = set()
    child_ids: list[str] = []
    for index, item in enumerate(normalized["memories"], 1):
        filename = f"{memory_day}_{_safe_title(item['title'])}.txt"
        if filename in seen_names:
            raise ValidationError(f"event_filename_collision:{filename}")
        seen_names.add(filename)
        path = event_dir / filename
        _write_text(path, _render_event(memory_day, item))
        event_files.append({"name": filename, "sha256": sha256_file(path)})
        child_ids.append(f"{memory_day}_{index}")
    review_path = staging / f"审阅_{memory_day}.md"
    _write_text(
        review_path,
        _render_review(
            memory_day, normalized, elapsed_seconds, generation_metadata
        ),
    )
    metadata = generation_metadata or {
        "llm_provider": "ollama_llm",
        "model": "mistral-small3.2:24b",
        "base_url": "",
        "temperature": 0.1,
        "max_output_tokens": 8192,
        "options": {},
    }
    manifest = {
        "status": "published",
        "schema_version": SCHEMA_VERSION,
        "memory_day": memory_day,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "llm_provider": metadata.get("llm_provider", "unknown_provider"),
        "model": metadata.get("model", "unknown_model"),
        "model_quantization": metadata.get("model_quantization"),
        "base_url": metadata.get("base_url", ""),
        "options": metadata.get("options", {}),
        "temperature": metadata.get("temperature"),
        "max_output_tokens": metadata.get("max_output_tokens"),
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": sha256_file(SYSTEM_PROMPT_PATH),
        "task_prompt_sha256": sha256_file(TASK_PROMPT_PATH),
        "diary_sha256": sha256_file(diary_path),
        "event_count": len(event_files),
        "child_ids": child_ids,
        "event_files": event_files,
        "review_file": {
            "name": review_path.name,
            "sha256": sha256_file(review_path),
        },
    }
    _write_json(staging / MANIFEST_FILENAME, manifest)
    if len(list(event_dir.glob("*.txt"))) != len(normalized["memories"]):
        raise ValidationError("staged_event_count_mismatch")
    return manifest


@contextmanager
def _daily_lock(work_root: Path, memory_day: str) -> Iterator[None]:
    lock_path = work_root / "locks" / f"{memory_day}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists() and time.time() - lock_path.stat().st_mtime > 12 * 3600:
        lock_path.unlink()
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RunAlreadyActive(f"worker_already_running:{memory_day}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


Generator = Callable[[str, str, str, Path], str]


def _coerce_generation_result(
    value: str | DailyChildEventGenerationResult,
) -> DailyChildEventGenerationResult:
    if isinstance(value, DailyChildEventGenerationResult):
        return value
    if isinstance(value, str):
        return DailyChildEventGenerationResult(
            text=value,
            metadata={
                "llm_provider": "custom_generator",
                "model": "custom_generator",
                "base_url": "",
                "temperature": None,
                "max_output_tokens": None,
                "options": {},
            },
        )
    raise ChildEventError("generator_return_type_invalid")


def daily_child_event_publication_status(
    memory_day: str,
    history_root: str | Path = Path("chat_history/rinne_01"),
) -> str:
    """Classify an existing publication without starting the local model."""

    date.fromisoformat(memory_day)
    history = Path(history_root).resolve()
    target = history / "events" / "child_events" / memory_day
    if not target.exists():
        return "missing"
    if not target.is_dir() or not _manifest_is_complete(target):
        return "invalid"
    diary_path = history / "diaries" / f"diary_{memory_day}.txt"
    if not diary_path.is_file() or diary_path.stat().st_size == 0:
        return "stale"
    manifest = json.loads(
        (target / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    if manifest.get("diary_sha256") != sha256_file(diary_path):
        return "stale"
    return "current"


def is_daily_child_event_published(
    memory_day: str,
    history_root: str | Path = Path("chat_history/rinne_01"),
) -> bool:
    """Return whether this diary revision already has a complete publication."""

    return daily_child_event_publication_status(memory_day, history_root) == "current"


def run_daily_child_event(
    memory_day: str,
    history_root: str | Path = Path("chat_history/rinne_01"),
    work_root: str | Path = DEFAULT_WORK_ROOT,
    generator: Generator | None = None,
    generation_settings: DailyChildEventGenerationConfig | None = None,
) -> dict[str, Any]:
    """Generate yesterday's events and atomically publish one complete date."""

    date.fromisoformat(memory_day)
    history = Path(history_root).resolve()
    work = Path(work_root).resolve()
    diary_path = history / "diaries" / f"diary_{memory_day}.txt"
    if not diary_path.is_file() or diary_path.stat().st_size == 0:
        raise ChildEventError(f"diary_missing_or_empty:{diary_path}")
    diary_hash = sha256_file(diary_path)
    events_root = history / "events" / "child_events"
    target = events_root / memory_day
    if target.exists():
        if _manifest_is_complete(target, diary_hash):
            return {"status": "already_published", "memory_day": memory_day}
        raise PublicationConflict(f"existing_date_without_valid_manifest:{target}")

    with _daily_lock(work, memory_day):
        if target.exists():
            if _manifest_is_complete(target, diary_hash):
                return {"status": "already_published", "memory_day": memory_day}
            raise PublicationConflict(f"existing_date_without_valid_manifest:{target}")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
        run_dir = work / "runs" / memory_day / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        log_path = run_dir / "run.log"
        settings = generation_settings or default_daily_child_event_settings()
        if not settings.enabled:
            _append_log(log_path, "SKIPPED daily_child_event_generation=disabled")
            return {"status": "disabled", "memory_day": memory_day}
        _append_log(
            log_path,
            "START "
            f"memory_day={memory_day} llm_provider={settings.llm_provider} "
            f"model={settings.model}",
        )
        diary_text = diary_path.read_text(encoding="utf-8")
        system_prompt, task_prompt = _load_prompts(memory_day, diary_text)
        started = time.monotonic()
        try:
            if generator is None:
                generation_result = generate_with_provider(
                    memory_day,
                    system_prompt,
                    task_prompt,
                    run_dir,
                    settings,
                    event_json_schema(memory_day),
                )
            else:
                generation_result = _coerce_generation_result(
                    generator(memory_day, system_prompt, task_prompt, run_dir)
                )
            response_text = generation_result.text
            elapsed = time.monotonic() - started
            if not (run_dir / "response.txt").exists():
                _write_text(run_dir / "response.txt", response_text)
            normalized = validate_model_response(response_text, memory_day)
            _write_json(run_dir / "normalized_result.json", normalized)
            staging_root = events_root / ".child_event_staging"
            staging = staging_root / f"{memory_day}_{uuid.uuid4().hex}"
            staging.mkdir(parents=True, exist_ok=False)
            try:
                manifest = _stage_publication(
                    staging,
                    memory_day,
                    diary_path,
                    normalized,
                    elapsed,
                    generation_result.metadata,
                )
                events_root.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise PublicationConflict(f"date_appeared_during_run:{target}")
                staging.rename(target)
            except Exception:
                if staging.exists():
                    shutil.rmtree(staging)
                raise
            finally:
                try:
                    staging_root.rmdir()
                except OSError:
                    pass
            if not _manifest_is_complete(target, diary_hash):
                raise ValidationError("post_publish_verification_failed")
            _append_log(
                log_path,
                f"PUBLISHED memory_day={memory_day} events={manifest['event_count']}",
            )
            return {
                "status": "published",
                "memory_day": memory_day,
                "event_count": manifest["event_count"],
                "target": str(target),
                "run_dir": str(run_dir),
            }
        except Exception as exc:
            _append_log(log_path, f"FAILED {type(exc).__name__}:{exc}")
            raise


def launch_daily_child_event_worker(
    memory_day: date,
    history_root: str | Path = Path("chat_history/rinne_01"),
    work_root: str | Path = DEFAULT_WORK_ROOT,
    config_path: str | Path = "conf.yaml",
) -> ChildEventWorkerLaunch:
    """Start the daily worker without blocking backend startup."""

    repository_root = Path(__file__).resolve().parents[3]
    history = Path(history_root).resolve()
    work = Path(work_root).resolve()
    launcher_log = work / "launcher.log"
    launcher_log.parent.mkdir(parents=True, exist_ok=True)
    result_path = (
        work
        / "worker_results"
        / memory_day.isoformat()
        / (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + f"_{uuid.uuid4().hex[:10]}.json"
        )
    )
    command = [
        sys.executable,
        "-B",
        "-m",
        "src.open_llm_vtuber.memory.daily_child_events",
        "--date",
        memory_day.isoformat(),
        "--history-root",
        str(history),
        "--work-root",
        str(work),
        "--config-path",
        str(Path(config_path).resolve()),
        "--result-path",
        str(result_path),
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with launcher_log.open("ab") as output:
        process = subprocess.Popen(
            command,
            cwd=repository_root,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
    return ChildEventWorkerLaunch(
        process=process,
        memory_day=memory_day.isoformat(),
        result_path=result_path,
    )


def audit_mojibake_directories(
    events_root: str | Path, *, remove_exact_duplicates: bool = False
) -> dict[str, Any]:
    """Find mojibake directories and optionally delete only exact duplicates."""

    root = Path(events_root).resolve()
    records: list[dict[str, Any]] = []
    if not root.is_dir():
        raise ChildEventError(f"events_root_missing:{root}")
    for day_dir in sorted(root.iterdir(), key=lambda item: item.name):
        if not day_dir.is_dir() or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day_dir.name):
            continue
        bad = day_dir / MOJIBAKE_EVENT_DIRECTORY_NAME
        if not bad.exists():
            continue
        good = day_dir / EVENT_DIRECTORY_NAME
        record: dict[str, Any] = {
            "memory_day": day_dir.name,
            "mojibake_path": str(bad),
            "status": "not_exact_duplicate",
        }
        if not bad.is_dir() or not good.is_dir():
            record["reason"] = "normal_or_mojibake_path_is_not_directory"
            records.append(record)
            continue
        if any(item.is_dir() for item in bad.iterdir()):
            record["reason"] = "mojibake_directory_contains_subdirectory"
            records.append(record)
            continue
        bad_files = sorted(
            (item for item in bad.iterdir() if item.is_file()),
            key=lambda item: item.name,
        )
        good_files = sorted(
            (item for item in good.iterdir() if item.is_file()),
            key=lambda item: item.name,
        )
        if {item.name for item in bad_files} != {item.name for item in good_files}:
            record["reason"] = "filename_sets_differ"
            records.append(record)
            continue
        differing = [
            item.name
            for item in bad_files
            if sha256_file(item) != sha256_file(good / item.name)
        ]
        if differing:
            record["reason"] = "file_hashes_differ"
            record["differing_files"] = differing
            records.append(record)
            continue
        record["file_count"] = len(bad_files)
        record["status"] = "exact_duplicate"
        if remove_exact_duplicates:
            resolved_bad = bad.resolve()
            if (
                resolved_bad.parent != day_dir.resolve()
                or resolved_bad.name != MOJIBAKE_EVENT_DIRECTORY_NAME
            ):
                raise ChildEventError(f"unsafe_delete_target:{resolved_bad}")
            for item in bad_files:
                item.unlink()
            bad.rmdir()
            record["status"] = "removed_exact_duplicate"
        records.append(record)
    return {
        "events_root": str(root),
        "remove_exact_duplicates": remove_exact_duplicates,
        "directories_found": len(records),
        "files_in_exact_duplicates": sum(
            int(item.get("file_count", 0)) for item in records
        ),
        "records": records,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Daily configured-LLM child-event worker"
    )
    parser.add_argument("--history-root", default="chat_history/rinne_01")
    parser.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT))
    parser.add_argument("--config-path", default="conf.yaml")
    parser.add_argument("--result-path")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date")
    group.add_argument("--audit-mojibake", action="store_true")
    group.add_argument("--remove-mojibake-duplicates", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    history = Path(arguments.history_root)
    work = Path(arguments.work_root)
    result_path = Path(arguments.result_path) if arguments.result_path else None
    try:
        if arguments.date:
            settings = load_daily_child_event_settings(arguments.config_path)
            if not settings.enabled:
                result = {
                    "status": "disabled",
                    "memory_day": arguments.date,
                }
            else:
                result = run_daily_child_event(
                    arguments.date,
                    history,
                    work,
                    generation_settings=settings,
                )
        else:
            result = audit_mojibake_directories(
                history / "events" / "child_events",
                remove_exact_duplicates=arguments.remove_mojibake_duplicates,
            )
            audit_dir = work / "directory_audits"
            audit_dir.mkdir(parents=True, exist_ok=True)
            mode = "remove" if arguments.remove_mojibake_duplicates else "audit"
            audit_path = audit_dir / (
                datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{mode}.json"
            )
            _write_json(audit_path, result)
            result["audit_path"] = str(audit_path)
        if result_path is not None:
            _write_json(result_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except RunAlreadyActive as exc:
        result = {
            "status": "already_running",
            "memory_day": arguments.date,
            "message": str(exc),
        }
        if result_path is not None:
            _write_json(result_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        result = {
            "status": "failed",
            "memory_day": arguments.date,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if result_path is not None:
            try:
                _write_json(result_path, result)
            except Exception as result_exc:
                print(
                    "FAILED writing worker result: "
                    f"{type(result_exc).__name__}: {result_exc}",
                    file=sys.stderr,
                )
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
