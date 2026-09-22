"""Generate and validate compact model-facing versions of approved diaries."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from . import layer2_claude_yaml


PROMPT_PATH = Path(__file__).parent / "prompts" / "recent_diary_compaction_v1.txt"
TOOL_NAME = "submit_recent_diary_context"
SCHEMA_VERSION = 1
COMPACTION_DIRECTORY_NAME = "recent_diary_context"
MAX_ITEM_COUNT = 15
MAX_MEMORY_TEXT_CHARS = 260
MAX_SUMMARY_CHARS = 1800


class RecentDiaryCompactionError(RuntimeError):
    """A compact diary could not be generated or safely validated."""


@dataclass(frozen=True)
class RecentDiaryCompaction:
    memory_day: str
    summary: str
    items: tuple[dict[str, Any], ...]
    source_character_count: int
    summary_character_count: int
    model: str
    elapsed_seconds: float

    @property
    def compression_ratio(self) -> float:
        if self.source_character_count <= 0:
            return 1.0
        return self.summary_character_count / self.source_character_count


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _evidence_key(value: str) -> str:
    """Keep source words/numbers exact while ignoring whitespace and punctuation."""

    return "".join(character.casefold() for character in value if character.isalnum())


def target_summary_character_count(source_character_count: int) -> int:
    """Give the model a real compression budget instead of a vague brevity request."""

    return min(MAX_SUMMARY_CHARS, max(480, int(source_character_count * 0.52)))


def _evidence_is_grounded(quote: str, diary_text: str) -> bool:
    quote_key = _evidence_key(quote)
    source_key = _evidence_key(diary_text)
    if len(quote_key) >= 4 and quote_key in source_key:
        return True
    segments = [
        _evidence_key(part)
        for part in re.split(r"[。！？!?；;，,\n]+", quote)
        if _evidence_key(part)
    ]
    return bool(segments) and all(
        len(segment) >= 4 and segment in source_key for segment in segments
    )


def _tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "提交有逐字证据支持的近期日记上下文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_day": {
                        "type": "string",
                        "pattern": r"^\d{4}-\d{2}-\d{2}$",
                    },
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_ITEM_COUNT,
                        "items": {
                            "type": "object",
                            "properties": {
                                "memory_text": {
                                    "type": "string",
                                    "minLength": 8,
                                    "maxLength": MAX_MEMORY_TEXT_CHARS,
                                },
                                "evidence_quotes": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 4,
                                    "items": {
                                        "type": "string",
                                        "minLength": 4,
                                        "maxLength": 100,
                                    },
                                },
                            },
                            "required": ["memory_text", "evidence_quotes"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["memory_day", "items"],
                "additionalProperties": False,
            },
        },
    }


def _extract_tool_arguments(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RecentDiaryCompactionError("provider_choices_missing")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RecentDiaryCompactionError("provider_message_missing")
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise RecentDiaryCompactionError("provider_tool_call_missing_or_multiple")
    function = tool_calls[0].get("function")
    if not isinstance(function, dict) or function.get("name") != TOOL_NAME:
        raise RecentDiaryCompactionError("provider_tool_name_mismatch")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise RecentDiaryCompactionError("provider_arguments_not_json") from exc
    if not isinstance(arguments, dict):
        raise RecentDiaryCompactionError("provider_arguments_missing")
    return arguments


def _parse_json_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        prefix = cleaned[:240].replace("\r", "\\r").replace("\n", "\\n")
        suffix = cleaned[-160:].replace("\r", "\\r").replace("\n", "\\n")
        raise RecentDiaryCompactionError(
            "provider_content_not_json:"
            f"{exc.msg}@{exc.pos}:length={len(cleaned)}:"
            f"prefix={prefix!r}:suffix={suffix!r}"
        ) from exc
    if not isinstance(value, dict):
        raise RecentDiaryCompactionError("provider_content_root_not_object")
    return value


def _stream_response_text(response: requests.Response) -> str:
    chunks: list[str] = []
    finish_reasons: list[str] = []
    saw_done = False
    for raw_line in response.iter_lines(decode_unicode=False, delimiter=b"\n"):
        try:
            line = (raw_line or b"").decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RecentDiaryCompactionError("provider_stream_not_utf8") from exc
        line = line.rstrip("\r")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        if data == "[DONE]":
            saw_done = True
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError as exc:
            raise RecentDiaryCompactionError("provider_stream_event_not_json") from exc
        choices = event.get("choices") if isinstance(event, dict) else None
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason:
                finish_reasons.append(finish_reason.casefold())
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str) and content:
                chunks.append(content)
    observed = set(finish_reasons)
    if observed.intersection({"length", "max_tokens", "max_output_tokens"}):
        raise RecentDiaryCompactionError("provider_output_truncated")
    if not saw_done and not observed.intersection({"stop", "end_turn", "stop_sequence"}):
        raise RecentDiaryCompactionError("provider_stream_incomplete")
    text = "".join(chunks).strip()
    if not text:
        raise RecentDiaryCompactionError("provider_content_missing")
    return text


def _stream_response_tool_arguments(response: requests.Response) -> dict[str, Any]:
    """Accumulate one forced OpenAI-compatible tool call from an SSE stream."""

    accumulated: dict[int, dict[str, str]] = {}
    finish_reasons: list[str] = []
    saw_done = False
    for raw_line in response.iter_lines(decode_unicode=False, delimiter=b"\n"):
        try:
            line = (raw_line or b"").decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RecentDiaryCompactionError("provider_stream_not_utf8") from exc
        line = line.rstrip("\r")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        if data == "[DONE]":
            saw_done = True
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError as exc:
            raise RecentDiaryCompactionError("provider_stream_event_not_json") from exc
        choices = event.get("choices") if isinstance(event, dict) else None
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason:
                finish_reasons.append(finish_reason.casefold())
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            for call in delta.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                index = call.get("index", 0)
                if not isinstance(index, int):
                    index = 0
                current = accumulated.setdefault(
                    index, {"name": "", "arguments": ""}
                )
                function = call.get("function")
                if not isinstance(function, dict):
                    continue
                name = function.get("name")
                if isinstance(name, str) and name:
                    current["name"] = name
                arguments = function.get("arguments")
                if isinstance(arguments, str) and arguments:
                    current["arguments"] += arguments

    observed = set(finish_reasons)
    if observed.intersection({"length", "max_tokens", "max_output_tokens"}):
        raise RecentDiaryCompactionError("provider_output_truncated")
    if not saw_done and not observed.intersection(
        {"stop", "end_turn", "stop_sequence", "tool_calls"}
    ):
        raise RecentDiaryCompactionError("provider_stream_incomplete")
    if len(accumulated) != 1:
        raise RecentDiaryCompactionError("provider_tool_call_missing_or_multiple")
    tool_call = next(iter(accumulated.values()))
    if tool_call["name"] != TOOL_NAME:
        raise RecentDiaryCompactionError("provider_tool_name_mismatch")
    try:
        arguments = json.loads(tool_call["arguments"])
    except json.JSONDecodeError as exc:
        raise RecentDiaryCompactionError("provider_arguments_not_json") from exc
    if not isinstance(arguments, dict):
        raise RecentDiaryCompactionError("provider_arguments_missing")
    return arguments


def _render_summary(memory_day: str, items: list[dict[str, Any]]) -> str:
    lines = [f"【近期记忆｜{memory_day}】"]
    lines.extend(f"- {item['memory_text'].strip()}" for item in items)
    return "\n".join(lines)


def _native_claude_settings(
    *, api_key: str, api_url: str, model: str, timeout_seconds: float
) -> SimpleNamespace | None:
    """Return APINebula native-Claude settings for its proven YAML transport."""

    parsed = urlsplit(api_url)
    if not model.casefold().startswith("claude-") or parsed.hostname != "apinebula.ai":
        return None
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")]
    base_url = urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")
    return SimpleNamespace(
        llm_provider="openai_compatible_llm",
        llm_api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.0,
        max_output_tokens=5000,
        timeout_seconds=timeout_seconds,
    )


def _request_native_claude_yaml(
    *,
    settings: SimpleNamespace,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    payload, url, _transport = layer2_claude_yaml.request(
        settings, system_prompt, user_prompt
    )
    try:
        with requests.post(
            url,
            headers=layer2_claude_yaml.headers(settings),
            json=payload,
            timeout=(20, settings.timeout_seconds),
            stream=True,
        ) as response:
            if response.status_code >= 400:
                raise RecentDiaryCompactionError(
                    f"provider_http_{response.status_code}"
                )
            assembled = layer2_claude_yaml.assemble_stream(
                response.iter_lines(chunk_size=1),
                deadline_seconds=settings.timeout_seconds,
            )
        return layer2_claude_yaml.parse(assembled)
    except layer2_claude_yaml.ClaudeOutputError as exc:
        raise RecentDiaryCompactionError(str(exc)) from exc


def validate_compaction(
    value: dict[str, Any],
    *,
    memory_day: str,
    diary_text: str,
    model: str,
    elapsed_seconds: float = 0.0,
) -> RecentDiaryCompaction:
    """Reject unsupported, oversized, duplicated, or malformed model output."""

    date.fromisoformat(memory_day)
    if value.get("memory_day") != memory_day:
        raise RecentDiaryCompactionError("memory_day_mismatch")
    items = value.get("items")
    if not isinstance(items, list):
        raise RecentDiaryCompactionError("items_not_array")
    if not 1 <= len(items) <= MAX_ITEM_COUNT:
        raise RecentDiaryCompactionError(f"item_count_invalid:{len(items)}")

    checked: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise RecentDiaryCompactionError(f"item_{index}_not_object")
        memory_text = item.get("memory_text")
        quotes = item.get("evidence_quotes")
        if (
            not isinstance(memory_text, str)
            or not 8 <= len(memory_text.strip()) <= MAX_MEMORY_TEXT_CHARS
        ):
            raise RecentDiaryCompactionError(f"item_{index}_text_invalid")
        normalized_text = "".join(memory_text.split())
        if normalized_text in seen_texts:
            raise RecentDiaryCompactionError(f"item_{index}_duplicate")
        seen_texts.add(normalized_text)
        if not isinstance(quotes, list) or not 1 <= len(quotes) <= 4:
            raise RecentDiaryCompactionError(f"item_{index}_evidence_invalid")
        checked_quotes: list[str] = []
        for quote in quotes:
            if not isinstance(quote, str) or len(_evidence_key(quote)) < 4:
                continue
            cleaned = quote.strip()
            if cleaned not in diary_text and not _evidence_is_grounded(
                cleaned, diary_text
            ):
                continue
            checked_quotes.append(cleaned)
        if not checked_quotes:
            raise RecentDiaryCompactionError(f"item_{index}_has_no_grounded_evidence")
        checked.append(
            {"memory_text": memory_text.strip(), "evidence_quotes": checked_quotes}
        )

    summary = _render_summary(memory_day, checked)
    target_character_count = target_summary_character_count(len(diary_text))
    if len(summary) > target_character_count:
        raise RecentDiaryCompactionError(
            f"summary_over_target:{len(summary)}>{target_character_count}"
        )
    if len(diary_text) >= 900 and len(summary) >= len(diary_text):
        raise RecentDiaryCompactionError("summary_did_not_compress")
    return RecentDiaryCompaction(
        memory_day=memory_day,
        summary=summary,
        items=tuple(checked),
        source_character_count=len(diary_text),
        summary_character_count=len(summary),
        model=model,
        elapsed_seconds=elapsed_seconds,
    )


def request_compaction(
    *,
    memory_day: str,
    diary_text: str,
    api_key: str,
    api_url: str,
    model: str,
    timeout_seconds: float = 180.0,
    max_attempts: int = 2,
    diagnostic_output_dir: str | Path | None = None,
) -> RecentDiaryCompaction:
    """Call the configured diary API without logging credentials or raw payloads."""

    date.fromisoformat(memory_day)
    if not api_key.strip():
        raise RecentDiaryCompactionError("api_key_missing")
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    user_prompt = (
        f"<memory_day>{memory_day}</memory_day>\n"
        "<summary_character_budget>"
        f"{target_summary_character_count(len(diary_text))}"
        "</summary_character_budget>\n"
        "<approved_diary>\n"
        f"{diary_text.strip()}\n"
        "</approved_diary>"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 5000,
        "stream": True,
        "tools": [_tool_schema()],
        "tool_choice": {
            "type": "function",
            "function": {"name": TOOL_NAME},
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    retryable_statuses = {408, 429, 500, 502, 503, 504}
    native_settings = _native_claude_settings(
        api_key=api_key,
        api_url=api_url,
        model=model,
        timeout_seconds=timeout_seconds,
    )
    failures: list[str] = []
    current_user_prompt = user_prompt
    rejected_arguments: dict[str, Any] | None = None
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        try:
            rejected_arguments = None
            if native_settings is not None:
                arguments = _request_native_claude_yaml(
                    settings=native_settings,
                    system_prompt=system_prompt,
                    user_prompt=current_user_prompt,
                )
                rejected_arguments = arguments
                if diagnostic_output_dir is not None:
                    diagnostic_path = Path(diagnostic_output_dir)
                    diagnostic_path.mkdir(parents=True, exist_ok=True)
                    (diagnostic_path / f"{memory_day}_attempt_{attempt}.json").write_text(
                        json.dumps(arguments, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                return validate_compaction(
                    arguments,
                    memory_day=memory_day,
                    diary_text=diary_text,
                    model=model,
                    elapsed_seconds=time.perf_counter() - started,
                )
            request_payload = {
                **payload,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": current_user_prompt},
                ],
            }
            response = requests.post(
                api_url,
                headers=headers,
                json=request_payload,
                timeout=(20, timeout_seconds),
                stream=True,
            )
            if response.status_code >= 400:
                failures.append(f"attempt_{attempt}:http_{response.status_code}")
                if response.status_code in retryable_statuses and attempt < max_attempts:
                    time.sleep(2)
                    continue
                raise RecentDiaryCompactionError(f"provider_http_{response.status_code}")
            response.encoding = "utf-8"
            arguments = _stream_response_tool_arguments(response)
            rejected_arguments = arguments
            if diagnostic_output_dir is not None:
                diagnostic_path = Path(diagnostic_output_dir)
                diagnostic_path.mkdir(parents=True, exist_ok=True)
                (diagnostic_path / f"{memory_day}_attempt_{attempt}.json").write_text(
                    json.dumps(arguments, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            return validate_compaction(
                arguments,
                memory_day=memory_day,
                diary_text=diary_text,
                model=model,
                elapsed_seconds=time.perf_counter() - started,
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            failures.append(f"attempt_{attempt}:{type(exc).__name__}")
            if attempt < max_attempts:
                time.sleep(2)
                continue
            raise RecentDiaryCompactionError("provider_connection_failed") from exc
        except RecentDiaryCompactionError as exc:
            failures.append(f"attempt_{attempt}:{exc}")
            if attempt < max_attempts and not str(exc).startswith("provider_http_4"):
                if rejected_arguments is not None:
                    current_user_prompt = (
                        user_prompt
                        + "\n\n<rejected_candidate>\n"
                        + json.dumps(rejected_arguments, ensure_ascii=False)
                        + "\n</rejected_candidate>\n"
                        + f"<rejection_reason>{exc}</rejection_reason>\n"
                        + "上一版未通过后台硬校验。只输出修正后的完整对象："
                        + "把同一事件合并并删掉过程性细节，严格压到字符预算内；"
                        + "每条memory_text不超过140字；亲密事件也只保留工具、次数、"
                        + "当事人的选择与结果/事后照顾，不复述动作过程；"
                        + "每条只能有1至2段短证据，"
                        + "每段证据只从原日记连续摘取6至40个文字数字，不得改写。"
                    )
                time.sleep(1)
                continue
            raise RecentDiaryCompactionError(";".join(failures)) from exc
    raise RecentDiaryCompactionError(";".join(failures) or "provider_failed")


def publication_paths(history_root: str | Path, memory_day: str) -> tuple[Path, Path]:
    date.fromisoformat(memory_day)
    root = Path(history_root) / COMPACTION_DIRECTORY_NAME
    return root / f"diary_{memory_day}.md", root / f"diary_{memory_day}.json"


def publish_compaction(
    history_root: str | Path,
    diary_path: str | Path,
    compaction: RecentDiaryCompaction,
) -> tuple[Path, Path]:
    """Atomically publish a summary bound to the current diary bytes."""

    diary = Path(diary_path)
    diary_text = diary.read_text(encoding="utf-8").strip()
    if len(diary_text) != compaction.source_character_count:
        raise RecentDiaryCompactionError("source_changed_before_publication")
    summary_path, manifest_path = publication_paths(
        history_root, compaction.memory_day
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "published",
        "schema_version": SCHEMA_VERSION,
        "memory_day": compaction.memory_day,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": compaction.model,
        "diary_file": diary.name,
        "diary_sha256": _sha256_text(diary_text),
        "summary_file": summary_path.name,
        "summary_sha256": _sha256_text(compaction.summary),
        "source_character_count": compaction.source_character_count,
        "summary_character_count": compaction.summary_character_count,
        "compression_ratio": round(compaction.compression_ratio, 6),
        "elapsed_seconds": round(compaction.elapsed_seconds, 3),
        "items": list(compaction.items),
    }
    nonce = uuid.uuid4().hex[:12]
    temporary_summary = summary_path.with_name(f".{summary_path.name}.{nonce}.tmp")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.{nonce}.tmp")
    temporary_summary.write_text(
        compaction.summary + "\n", encoding="utf-8", newline="\n"
    )
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary_summary, summary_path)
    os.replace(temporary_manifest, manifest_path)
    return summary_path, manifest_path


def load_matching_compaction(
    history_root: str | Path,
    diary_path: str | Path,
    memory_day: str,
) -> RecentDiaryCompaction | None:
    """Load only a complete summary whose hashes still match its source diary."""

    diary = Path(diary_path)
    summary_path, manifest_path = publication_paths(history_root, memory_day)
    if not diary.is_file() or not summary_path.is_file() or not manifest_path.is_file():
        return None
    try:
        diary_text = diary.read_text(encoding="utf-8").strip()
        summary = summary_path.read_text(encoding="utf-8").strip()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return None
        expected = {
            "status": "published",
            "schema_version": SCHEMA_VERSION,
            "memory_day": memory_day,
            "diary_file": diary.name,
            "diary_sha256": _sha256_text(diary_text),
            "summary_file": summary_path.name,
            "summary_sha256": _sha256_text(summary),
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            return None
        items = manifest.get("items")
        value = {"memory_day": memory_day, "items": items}
        validated = validate_compaction(
            value,
            memory_day=memory_day,
            diary_text=diary_text,
            model=str(manifest.get("model") or "unknown"),
            elapsed_seconds=float(manifest.get("elapsed_seconds") or 0.0),
        )
        if validated.summary != summary:
            return None
        return validated
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        return None


__all__ = [
    "RecentDiaryCompaction",
    "RecentDiaryCompactionError",
    "load_matching_compaction",
    "publication_paths",
    "publish_compaction",
    "request_compaction",
    "target_summary_character_count",
    "validate_compaction",
]
