"""Safe metadata-only diagnostics for image data URLs."""

from __future__ import annotations

import base64
import binascii
import re
import struct
from typing import Any


_DATA_URL = re.compile(
    r"^data:(?P<mime>image/[a-zA-Z0-9.+-]+);base64,(?P<payload>.*)$",
    re.DOTALL,
)
_JPEG_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


def _read_png_dimensions(raw: bytes) -> tuple[int, int] | None:
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", raw[16:24])
    return (width, height) if width > 0 and height > 0 else None


def _read_jpeg_dimensions(raw: bytes) -> tuple[int, int] | None:
    if len(raw) < 4 or raw[:2] != b"\xff\xd8":
        return None

    index = 2
    while index + 4 <= len(raw):
        if raw[index] != 0xFF:
            index += 1
            continue
        while index < len(raw) and raw[index] == 0xFF:
            index += 1
        if index >= len(raw):
            break

        marker = raw[index]
        index += 1
        if marker in {0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > len(raw):
            break
        segment_length = int.from_bytes(raw[index : index + 2], "big")
        if segment_length < 2 or index + segment_length > len(raw):
            break
        if marker in _JPEG_SOF_MARKERS and segment_length >= 7:
            height = int.from_bytes(raw[index + 3 : index + 5], "big")
            width = int.from_bytes(raw[index + 5 : index + 7], "big")
            return (width, height) if width > 0 and height > 0 else None
        index += segment_length
    return None


def inspect_image_payload(image: Any) -> dict[str, Any]:
    """Return dimensions and byte size without retaining or logging image data."""

    diagnostic: dict[str, Any] = {
        "source": image.get("source") if isinstance(image, dict) else None,
        "declared_mime": image.get("mime_type") if isinstance(image, dict) else None,
        "encoded_chars": 0,
        "decoded_bytes": 0,
        "width": None,
        "height": None,
        "status": "invalid",
    }
    if not isinstance(image, dict):
        return diagnostic
    data = image.get("data")
    if not isinstance(data, str):
        return diagnostic

    match = _DATA_URL.match(data)
    if not match:
        return diagnostic
    diagnostic["actual_mime"] = match.group("mime").lower()
    payload = match.group("payload")
    diagnostic["encoded_chars"] = len(payload)
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        diagnostic["status"] = "invalid_base64"
        return diagnostic

    diagnostic["decoded_bytes"] = len(raw)
    dimensions = _read_png_dimensions(raw) or _read_jpeg_dimensions(raw)
    if dimensions is None:
        diagnostic["status"] = "unsupported_or_invalid_image"
        return diagnostic
    diagnostic["width"], diagnostic["height"] = dimensions
    diagnostic["status"] = "ok"
    return diagnostic


__all__ = ["inspect_image_payload"]
