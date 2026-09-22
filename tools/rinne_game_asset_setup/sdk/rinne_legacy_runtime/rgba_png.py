from __future__ import annotations

import struct
import zlib


PNG_MAX_DIMENSION = 8_192
PNG_MAX_PIXELS = 16_777_216


def encode_rgba_png(
    width: int,
    height: int,
    rgba: bytes | bytearray | memoryview,
    *,
    flip_y: bool = False,
) -> bytes:
    if (
        width <= 0
        or height <= 0
        or width > PNG_MAX_DIMENSION
        or height > PNG_MAX_DIMENSION
        or width * height > PNG_MAX_PIXELS
    ):
        raise ValueError(
            "PNG dimensions exceed the safe output limit "
            f"({PNG_MAX_DIMENSION} per side, {PNG_MAX_PIXELS} pixels)"
        )
    pixels = memoryview(rgba).cast("B")
    expected = width * height * 4
    if len(pixels) != expected:
        raise ValueError(f"RGBA payload length {len(pixels)} != {expected}")

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    row_indices = range(height - 1, -1, -1) if flip_y else range(height)
    scanlines = b"".join(
        b"\x00" + pixels[row * width * 4 : (row + 1) * width * 4]
        for row in row_indices
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + chunk(b"IEND", b"")
    )


__all__ = ["encode_rgba_png"]
