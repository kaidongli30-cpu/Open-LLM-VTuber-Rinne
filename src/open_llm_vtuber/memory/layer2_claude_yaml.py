"""APINebula Claude structured text transport; local JSON schemas stay authoritative."""

import re
import json
import time
from urllib.parse import urlsplit

import requests
import yaml


class ClaudeOutputError(ValueError):
    pass


_NATURAL_LANGUAGE_SCALAR = re.compile(
    r"^(?P<indent>\s*)(?P<key>text|reason|summary|statement|notes): '(?P<value>.*)$"
)


def _repair_unclosed_natural_language_scalars(text):
    """Convert an unclosed quoted prose scalar to a literal block.

    Claude occasionally starts a YAML single-quoted ``reason``/``text`` value
    and then uses an ordinary apostrophe in the prose without closing the YAML
    scalar.  The transformation is deliberately narrow and character
    preserving: it applies only to known prose fields whose line has no closing
    quote, and removes only the opening YAML delimiter.
    """

    changed = 0
    output = []
    for line in text.splitlines():
        match = _NATURAL_LANGUAGE_SCALAR.fullmatch(line)
        if match is None or match.group("value").endswith("'"):
            output.append(line)
            continue
        indent = match.group("indent")
        output.extend(
            (
                f"{indent}{match.group('key')}: |-",
                f"{indent}  {match.group('value')}",
            )
        )
        changed += 1
    return "\n".join(output), changed


class _UniqueSafeLoader(yaml.SafeLoader):
    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise ClaudeOutputError("yaml_alias_not_allowed")
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ClaudeOutputError("yaml_duplicate_or_nonstring_key")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


# Dates must remain strings, exactly as required by the existing JSON schemas.
_UniqueSafeLoader.yaml_implicit_resolvers = {
    key: [(tag, regex) for tag, regex in values if tag != "tag:yaml.org,2002:timestamp"]
    for key, values in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def enabled(settings):
    return (
        settings.llm_provider == "openai_compatible_llm"
        and settings.model.lower().startswith("claude-")
        and urlsplit(settings.base_url).hostname == "apinebula.ai"
    )


def headers(settings):
    return {
        "Content-Type": "application/json",
        "x-api-key": settings.llm_api_key,
        "anthropic-version": "2023-06-01",
    }


def request(settings, system_prompt, task_prompt):
    instruction = (
        "\n\n【接口输出格式，优先于任务中的 JSON 输出字样】\n"
        "请只返回一个 YAML 对象，不要工具调用、解释、代码围栏。"
        "任务中的 output_json_schema 仍严格规定字段、类型、必填项和内容；"
        "但用 YAML 表达这个对象，而不是 JSON，也不要把对象或数组编码成字符串。"
        "数组使用 YAML 列表，空数组用 []，空值用 null，布尔值用 true/false。"
        "日期是字符串。statement、notes、summary、证据摘录等自然语言字符串"
        "请用 YAML 的 |- 多行文本形式，保持原文引号和文字，不做改写或转义。"
        "禁止重复键、别名和自定义标签。输出完整后结束。"
    )
    base = settings.base_url.rstrip("/")
    url = base + ("/messages" if base.endswith("/v1") else "/v1/messages")
    return (
        {
            "model": settings.model,
            "temperature": settings.temperature,
            "max_tokens": settings.max_output_tokens,
            "stream": True,
            "system": system_prompt + instruction,
            "messages": [{"role": "user", "content": task_prompt}],
        },
        url,
        "anthropic_schema_validated_yaml",
    )


def parse(response):
    if response.get("stop_reason") != "end_turn":
        raise ClaudeOutputError("claude_output_not_complete")
    blocks = response.get("content")
    if not isinstance(blocks, list) or not blocks:
        raise ClaudeOutputError("claude_text_missing")
    if any(not isinstance(b, dict) or b.get("type") != "text" for b in blocks):
        raise ClaudeOutputError("claude_unexpected_content_block")
    text = "\n".join(b.get("text", "") for b in blocks).strip()
    fenced = re.fullmatch(r"```(?:yaml|yml)\s*\n(.*)\n```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        value = yaml.load(text, Loader=_UniqueSafeLoader)
    except yaml.MarkedYAMLError as first_error:
        repaired_text, repair_count = _repair_unclosed_natural_language_scalars(
            text
        )
        if not repair_count:
            raise ClaudeOutputError("claude_invalid_yaml") from first_error
        try:
            value = yaml.load(repaired_text, Loader=_UniqueSafeLoader)
        except (yaml.YAMLError, RecursionError) as exc:
            raise ClaudeOutputError("claude_invalid_yaml") from exc
    except (yaml.YAMLError, RecursionError) as exc:
        raise ClaudeOutputError("claude_invalid_yaml") from exc
    if not isinstance(value, dict):
        raise ClaudeOutputError("claude_output_not_object")
    return value


def assemble_stream(lines, *, deadline_seconds=None):
    """Require a complete native Messages stream, including message_stop."""
    started = time.monotonic()
    response = None
    blocks = {}
    closed = set()
    stopped = False
    data = []

    def events():
        for raw in lines:
            if (
                deadline_seconds is not None
                and time.monotonic() - started > deadline_seconds
            ):
                raise ClaudeOutputError("claude_stream_total_timeout")
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if not line:
                if data:
                    yield json.loads("\n".join(data))
                    data.clear()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
        if data:
            yield json.loads("\n".join(data))

    try:
        for event in events():
            kind = event.get("type")
            if kind == "error":
                raise ClaudeOutputError("claude_stream_error")
            if kind == "message_start":
                if response is not None:
                    raise ClaudeOutputError("claude_duplicate_message_start")
                response = event["message"]
                response["content"] = []
            elif kind == "content_block_start":
                index = event["index"]
                block = event["content_block"]
                if index in blocks or block.get("type") != "text":
                    raise ClaudeOutputError("claude_unexpected_stream_block")
                blocks[index] = {"type": "text", "text": block.get("text", "")}
            elif kind == "content_block_delta":
                index = event["index"]
                if (
                    index not in blocks
                    or index in closed
                    or event["delta"].get("type") != "text_delta"
                ):
                    raise ClaudeOutputError("claude_invalid_stream_delta")
                blocks[index]["text"] += event["delta"]["text"]
            elif kind == "content_block_stop":
                closed.add(event["index"])
            elif kind == "message_delta":
                if response is None:
                    raise ClaudeOutputError("claude_message_start_missing")
                response.update(event.get("delta", {}))
                response.setdefault("usage", {}).update(event.get("usage", {}))
            elif kind == "message_stop":
                stopped = True
                break
    except ClaudeOutputError:
        raise
    except (ValueError, KeyError, TypeError, UnicodeError) as exc:
        raise ClaudeOutputError("claude_invalid_stream") from exc
    if not stopped or response is None or not blocks or set(blocks) != closed:
        raise ClaudeOutputError("claude_incomplete_stream")
    response["content"] = [blocks[index] for index in sorted(blocks)]
    return response


def post(settings, payload, response_path):
    _, url, _ = request(settings, "", "")
    started = time.monotonic()
    try:
        with requests.post(
            url,
            headers=headers(settings),
            json={**payload, "stream": True},
            stream=True,
            timeout=(
                min(settings.timeout_seconds, 30.0),
                min(settings.timeout_seconds, 60.0),
            ),
        ) as response:
            if response.status_code != 200:
                raise ClaudeOutputError(f"claude_http_error:{response.status_code}")
            remaining = max(
                0.001, settings.timeout_seconds - (time.monotonic() - started)
            )
            value = assemble_stream(
                response.iter_lines(chunk_size=1), deadline_seconds=remaining
            )
    except requests.RequestException as exc:
        raise ClaudeOutputError(
            f"claude_stream_connection_error:{type(exc).__name__}"
        ) from exc
    response_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return value
