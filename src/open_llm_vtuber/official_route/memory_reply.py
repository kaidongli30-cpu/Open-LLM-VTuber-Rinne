"""Decode memory JSON; tolerate only literal quotes inside final official_text.

No model call, inferred fields, label changes, or guessed text. Ambiguous item
boundaries and all other malformed structure are rejected.
"""

import json
import re


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_memory_field")
        result[key] = value
    return result


def _literal_quotes(value):
    result, escaped = [], False
    for char in value:
        if char == '"' and not escaped:
            result.append("\\")
        result.append(char)
        escaped = char == "\\" and not escaped
    return "".join(result)


def parse_memory_reply(raw):
    try:
        return json.loads(raw, object_pairs_hook=_unique)
    except json.JSONDecodeError:
        pass
    header = re.match(r'^\s*\{\s*"items"\s*:\s*\[\s*', raw)
    if not header:
        raise ValueError("invalid_memory_envelope")
    position, items = header.end(), []
    while position < len(raw):
        start = re.match(r'\{\s*"id"\s*:', raw[position:])
        if not start:
            raise ValueError("invalid_memory_item")
        marker = re.search(r'"official_text"\s*:\s*"', raw[position:])
        if not marker:
            raise ValueError("missing_memory_text")
        text_start = position + marker.end()
        prefix = raw[position : position + marker.start()]
        # The schema prefix itself must be valid, flat and unique. Only this
        # final string field is eligible for literal-quote recovery.
        item = json.loads(prefix + '"official_text":""}', object_pairs_hook=_unique)
        allowed = {
            "id",
            "label",
            "confidence",
            "reason",
            "line",
            "action",
            "official_label",
            "official_text",
        }
        if set(item) - allowed or any(
            isinstance(v, (dict, list)) for v in item.values()
        ):
            raise ValueError("ambiguous_memory_prefix")
        end = re.search(r'"\s*}\s*(?=,\s*\{\s*"id"\s*:|\]\s*}\s*$)', raw[text_start:])
        if not end:
            raise ValueError("unclosed_memory_text")
        value = raw[text_start : text_start + end.start()]
        # A structural-looking item/key embedded in malformed text cannot be
        # distinguished safely from a result item. Do not guess its boundary.
        if re.search(r'"[^"\r\n]*"\s*:', value) or re.search(r'"\s*}\s*[,\]]', value):
            raise ValueError("ambiguous_memory_text")
        item["official_text"] = json.loads('"' + _literal_quotes(value) + '"')
        items.append(item)
        position = text_start + end.end()
        tail = raw[position:]
        if re.fullmatch(r"\s*\]\s*}\s*", tail):
            return {"items": items}
        separator = re.match(r"\s*,\s*", tail)
        if not separator:
            raise ValueError("invalid_memory_separator")
        position += separator.end()
    raise ValueError("incomplete_memory_reply")
