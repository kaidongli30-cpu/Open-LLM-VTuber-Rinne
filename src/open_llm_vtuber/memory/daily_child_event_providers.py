"""Provider adapters for the independent daily child-event worker.

The adapters return untrusted model text.  Publication and schema validation
remain in ``daily_child_events`` so every provider follows the same safety
path.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..config_manager.daily_child_event import DailyChildEventGenerationConfig


DEFAULT_CONFIG_PATH = Path("conf.yaml")


class DailyChildEventProviderError(RuntimeError):
    """A provider request failed before event validation or publication."""


@dataclass(frozen=True)
class DailyChildEventGenerationResult:
    """Untrusted provider text plus safe, non-secret call metadata."""

    text: str
    metadata: dict[str, Any]


def default_daily_child_event_settings() -> DailyChildEventGenerationConfig:
    """Return the backwards-compatible local Ollama configuration."""

    return DailyChildEventGenerationConfig()


def load_daily_child_event_settings(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> DailyChildEventGenerationConfig:
    """Load only the daily-event section without logging the full config.

    The worker is a child process and should not validate the whole application
    configuration.  Missing sections intentionally use the safe legacy
    default, which keeps older user configurations runnable.
    """

    path = Path(config_path)
    if not path.is_file():
        return default_daily_child_event_settings()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        section = (
            raw.get("character_config", {}).get("daily_child_event_generation", {})
        )
        if not isinstance(section, dict):
            raise ValueError("daily_child_event_generation must be a mapping")
        return DailyChildEventGenerationConfig.model_validate(section)
    except Exception as exc:
        raise DailyChildEventProviderError(
            f"invalid_daily_child_event_config:{type(exc).__name__}"
        ) from exc


def _safe_base_url(value: str) -> str:
    """Return a log-safe endpoint without query strings or fragments."""

    parsed = urllib.parse.urlsplit(str(value or "").strip())
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "")
    )


def _append_endpoint(base_url: str, suffix: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        raise DailyChildEventProviderError("provider_base_url_is_empty")
    if base.endswith("/v1"):
        return f"{base}{suffix}"
    return f"{base}/v1{suffix}"


def _ollama_generate_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(str(base_url or "").strip())
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3].rstrip("/")
    if path.endswith("/api"):
        path = f"{path}/generate"
    else:
        path = f"{path}/api/generate"
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, path, "", "")
    )


def _metadata(settings: DailyChildEventGenerationConfig) -> dict[str, Any]:
    return {
        "llm_provider": settings.llm_provider,
        "model": settings.model,
        "base_url": _safe_base_url(settings.base_url),
        "temperature": settings.temperature,
        "max_output_tokens": settings.max_output_tokens,
        "timeout_seconds": settings.timeout_seconds,
        "options": {
            "top_p": settings.top_p,
            "top_k": settings.top_k,
            "presence_penalty": settings.presence_penalty,
            "num_ctx": settings.num_ctx,
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _write_request_metadata(
    run_dir: Path,
    settings: DailyChildEventGenerationConfig,
    schema: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    metadata = _metadata(settings)
    metadata.update(
        {
            "prompt_version": "v2.3-neutral",
            "format": schema,
            "request_kind": "daily_child_event_generation",
        }
    )
    # Keep the request shape useful for diagnosis while never writing the API
    # key or authorization headers to disk.
    metadata["request_payload"] = {
        key: value
        for key, value in payload.items()
        if key not in {"headers", "llm_api_key", "api_key"}
    }
    _write_json(run_dir / "request_metadata.json", metadata)


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    response_path: Path,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise DailyChildEventProviderError(
            f"provider_http_error:{exc.code}"
        ) from exc
    except (
        urllib.error.URLError,
        http.client.IncompleteRead,
        TimeoutError,
        OSError,
    ) as exc:
        raise DailyChildEventProviderError(
            f"provider_connection_error:{type(exc).__name__}"
        ) from exc
    if not raw.strip():
        raise DailyChildEventProviderError("provider_empty_response")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DailyChildEventProviderError("provider_response_not_json") from exc
    if not isinstance(value, dict):
        raise DailyChildEventProviderError("provider_response_root_not_object")
    _write_json(response_path, value)
    return value


def _text_from_openai_response(value: dict[str, Any]) -> str:
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices:
        raise DailyChildEventProviderError("provider_choices_missing")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise DailyChildEventProviderError("provider_message_missing")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "".join(fragments)
    raise DailyChildEventProviderError("provider_content_missing")


def _text_from_claude_response(value: dict[str, Any]) -> str:
    content = value.get("content")
    if not isinstance(content, list):
        raise DailyChildEventProviderError("provider_content_missing")
    fragments = [
        item.get("text", "")
        for item in content
        if isinstance(item, dict)
        and item.get("type") == "text"
        and isinstance(item.get("text"), str)
    ]
    return "".join(fragments)


def _ollama_keep_alive(value: float) -> int | str:
    if value < 0:
        return -1
    return f"{int(value)}s"


def _generate_with_ollama(
    memory_day: str,
    system_prompt: str,
    task_prompt: str,
    run_dir: Path,
    settings: DailyChildEventGenerationConfig,
    schema: dict[str, Any],
) -> DailyChildEventGenerationResult:
    options = {
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "top_k": settings.top_k,
        "presence_penalty": settings.presence_penalty,
        "num_ctx": settings.num_ctx,
        "num_predict": settings.max_output_tokens,
    }
    payload = {
        "model": settings.model,
        "system": system_prompt,
        "prompt": task_prompt,
        "stream": True,
        "think": False,
        "keep_alive": _ollama_keep_alive(settings.keep_alive),
        "format": schema,
        "options": options,
    }
    _write_request_metadata(run_dir, settings, schema, payload)
    request = urllib.request.Request(
        _ollama_generate_url(settings.base_url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    fragments: list[str] = []
    final_chunk: dict[str, Any] | None = None
    stream_path = run_dir / "response.stream.jsonl"
    try:
        with (
            urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response,
            stream_path.open("w", encoding="utf-8", newline="") as stream,
        ):
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="strict").strip()
                if not line:
                    continue
                stream.write(line + "\n")
                stream.flush()
                chunk = json.loads(line)
                if "error" in chunk:
                    raise DailyChildEventProviderError(
                        "ollama_stream_error:provider_returned_error"
                    )
                fragment = chunk.get("response")
                if isinstance(fragment, str):
                    fragments.append(fragment)
                if chunk.get("done") is True:
                    final_chunk = chunk
    except urllib.error.HTTPError as exc:
        raise DailyChildEventProviderError(
            f"provider_http_error:{exc.code}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DailyChildEventProviderError(
            f"provider_connection_error:{type(exc).__name__}"
        ) from exc
    finally:
        if settings.unload_at_exit:
            _try_unload_ollama(settings)
    if final_chunk is None:
        raise DailyChildEventProviderError("ollama_stream_incomplete")
    if final_chunk.get("done_reason") == "length":
        raise DailyChildEventProviderError("ollama_output_truncated")
    text = "".join(fragments)
    if not text.strip():
        raise DailyChildEventProviderError("provider_empty_response")
    _write_text(run_dir / "response.txt", text)
    _write_json(run_dir / "final_chunk.json", final_chunk)
    return DailyChildEventGenerationResult(text=text, metadata=_metadata(settings))


def _try_unload_ollama(settings: DailyChildEventGenerationConfig) -> None:
    payload = {
        "model": settings.model,
        "prompt": "",
        "keep_alive": 0,
        "stream": False,
    }
    request = urllib.request.Request(
        _ollama_generate_url(settings.base_url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=min(settings.timeout_seconds, 10)):
            pass
    except Exception:
        # Unloading is cleanup only.  It must never replace a generation
        # result or turn a successful publication into a failed one.
        return


def _generate_with_openai_compatible(
    memory_day: str,
    system_prompt: str,
    task_prompt: str,
    run_dir: Path,
    settings: DailyChildEventGenerationConfig,
    schema: dict[str, Any],
) -> DailyChildEventGenerationResult:
    del memory_day
    payload = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ],
        "temperature": settings.temperature,
        "max_tokens": settings.max_output_tokens,
        "response_format": {"type": "json_object"},
    }
    _write_request_metadata(run_dir, settings, schema, payload)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.llm_api_key}",
    }
    if settings.organization_id:
        headers["OpenAI-Organization"] = settings.organization_id
    if settings.project_id:
        headers["OpenAI-Project"] = settings.project_id
    value = _post_json(
        _append_endpoint(settings.base_url, "/chat/completions"),
        payload,
        headers,
        settings.timeout_seconds,
        run_dir / "response.json",
    )
    text = _text_from_openai_response(value)
    if not text.strip():
        raise DailyChildEventProviderError("provider_empty_response")
    _write_text(run_dir / "response.txt", text)
    return DailyChildEventGenerationResult(text=text, metadata=_metadata(settings))


def _generate_with_claude(
    memory_day: str,
    system_prompt: str,
    task_prompt: str,
    run_dir: Path,
    settings: DailyChildEventGenerationConfig,
    schema: dict[str, Any],
) -> DailyChildEventGenerationResult:
    del memory_day
    payload = {
        "model": settings.model,
        "system": system_prompt,
        "messages": [{"role": "user", "content": task_prompt}],
        "temperature": settings.temperature,
        "max_tokens": settings.max_output_tokens,
    }
    _write_request_metadata(run_dir, settings, schema, payload)
    headers = {
        "Content-Type": "application/json",
        "x-api-key": settings.llm_api_key,
        "anthropic-version": "2023-06-01",
    }
    value = _post_json(
        _append_endpoint(settings.base_url, "/messages"),
        payload,
        headers,
        settings.timeout_seconds,
        run_dir / "response.json",
    )
    text = _text_from_claude_response(value)
    if not text.strip():
        raise DailyChildEventProviderError("provider_empty_response")
    _write_text(run_dir / "response.txt", text)
    return DailyChildEventGenerationResult(text=text, metadata=_metadata(settings))


def generate_with_provider(
    memory_day: str,
    system_prompt: str,
    task_prompt: str,
    run_dir: Path,
    settings: DailyChildEventGenerationConfig,
    schema: dict[str, Any],
) -> DailyChildEventGenerationResult:
    """Call the selected provider exactly once and return untrusted text."""

    if settings.llm_provider == "ollama_llm":
        return _generate_with_ollama(
            memory_day, system_prompt, task_prompt, run_dir, settings, schema
        )
    if settings.llm_provider == "claude_llm":
        return _generate_with_claude(
            memory_day, system_prompt, task_prompt, run_dir, settings, schema
        )
    if settings.llm_provider in {
        "lmstudio_llm",
        "openai_compatible_llm",
        "openai_llm",
        "gemini_llm",
        "zhipu_llm",
        "deepseek_llm",
        "groq_llm",
        "mistral_llm",
    }:
        return _generate_with_openai_compatible(
            memory_day, system_prompt, task_prompt, run_dir, settings, schema
        )
    raise DailyChildEventProviderError(
        f"unsupported_daily_child_event_provider:{settings.llm_provider}"
    )
