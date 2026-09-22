from __future__ import annotations

import struct
from dataclasses import dataclass

from .checked_binary import BinaryBoundsError, CheckedBinary


TEXTURE_WRAPPER_SIZE = 16
TEXTURE_MAX_DIMENSION = 8_192
TEXTURE_MAX_PIXELS = 16_777_216
TEXTURE_MAX_RGBA_BYTES = TEXTURE_MAX_PIXELS * 4
TEXTURE_MAX_COMPRESSED_BYTES = 80 * 1024 * 1024
CELL_LZ77_HEADER_SIZE = 16


@dataclass(frozen=True)
class MotionPortraitTexture:
    width: int
    height: int
    rgba: bytes
    wrapper_word_0: int
    wrapper_word_4: int
    compressed_size: int
    token_count: int
    data_offset: int


def decompress_cell_lz77(
    data: bytes | bytearray | memoryview,
    *,
    expected_size: int | None = None,
) -> tuple[bytes, int, int]:
    byte_view = memoryview(data).cast("B")
    if len(byte_view) > TEXTURE_MAX_COMPRESSED_BYTES:
        raise BinaryBoundsError(
            "cell LZ77 input size "
            f"{len(byte_view)} exceeds {TEXTURE_MAX_COMPRESSED_BYTES}"
        )
    stable_data = data if isinstance(data, bytes) else bytes(byte_view)
    reader = CheckedBinary(stable_data)
    reader.span("cell LZ77 header", 0, CELL_LZ77_HEADER_SIZE)
    if reader.bytes_at("cell LZ77 signature", 0, 4) != b"LZ77":
        raise ValueError("missing cell LZ77 signature")

    raw_size = reader.u32("cell LZ77 raw size", 4)
    token_count = reader.u32("cell LZ77 token count", 8)
    data_offset = reader.u32("cell LZ77 data offset", 12)
    if raw_size > TEXTURE_MAX_RGBA_BYTES:
        raise BinaryBoundsError(
            f"cell LZ77 raw size {raw_size} exceeds {TEXTURE_MAX_RGBA_BYTES}"
        )
    if token_count > raw_size:
        raise BinaryBoundsError(
            f"cell LZ77 token count {token_count} exceeds raw size {raw_size}"
        )
    if expected_size is not None and raw_size != expected_size:
        raise ValueError(f"cell LZ77 raw size {raw_size} != expected {expected_size}")

    flag_bytes = (token_count + 7) // 8
    expected_offset = CELL_LZ77_HEADER_SIZE + flag_bytes
    if data_offset != expected_offset:
        raise ValueError(
            f"cell LZ77 data offset {data_offset} != expected {expected_offset}"
        )
    reader.span("cell LZ77 flags", CELL_LZ77_HEADER_SIZE, flag_bytes)
    reader.span("cell LZ77 payload", data_offset, reader.size - data_offset)

    flags = memoryview(stable_data)[CELL_LZ77_HEADER_SIZE:data_offset]
    payload = memoryview(stable_data)[data_offset:]
    output = bytearray()
    payload_position = 0

    for token_index in range(token_count):
        flag = flags[token_index // 8]
        is_back_reference = flag & (0x80 >> (token_index % 8))
        if not is_back_reference:
            if payload_position >= len(payload):
                raise BinaryBoundsError(
                    f"cell LZ77 literal {token_index} exceeds compressed payload"
                )
            if len(output) >= raw_size:
                raise BinaryBoundsError("cell LZ77 output exceeds declared raw size")
            output.append(payload[payload_position])
            payload_position += 1
            continue

        if payload_position + 2 > len(payload):
            raise BinaryBoundsError(
                f"cell LZ77 back-reference {token_index} exceeds compressed payload"
            )
        distance = payload[payload_position]
        length = payload[payload_position + 1] + 3
        payload_position += 2
        if distance == 0 or distance > len(output):
            raise ValueError(
                f"invalid cell LZ77 distance {distance} at token {token_index}"
            )
        if len(output) + length > raw_size:
            raise BinaryBoundsError("cell LZ77 output exceeds declared raw size")
        for _index in range(length):
            output.append(output[-distance])

    if len(output) != raw_size:
        raise ValueError(f"cell LZ77 decoded {len(output)} bytes, expected {raw_size}")
    if payload_position != len(payload):
        raise ValueError(
            "cell LZ77 compressed payload has an unconsumed suffix: "
            f"{len(payload) - payload_position} bytes"
        )
    return bytes(output), token_count, data_offset


def decode_motionportrait_texture(
    data: bytes | bytearray | memoryview,
) -> MotionPortraitTexture:
    byte_view = memoryview(data).cast("B")
    maximum_container_size = TEXTURE_WRAPPER_SIZE + TEXTURE_MAX_COMPRESSED_BYTES
    if len(byte_view) > maximum_container_size:
        raise BinaryBoundsError(
            f"texture input size {len(byte_view)} exceeds {maximum_container_size}"
        )
    stable_data = data if isinstance(data, bytes) else bytes(byte_view)
    reader = CheckedBinary(stable_data)
    reader.span("texture wrapper", 0, TEXTURE_WRAPPER_SIZE)
    wrapper_word_0 = reader.u32("texture wrapper word 0", 0)
    wrapper_word_4 = reader.u32("texture wrapper word 4", 4)
    compressed_size = reader.u32("texture compressed size", 8)
    width, height = struct.unpack(
        "<HH", reader.bytes_at("texture dimensions", 12, 4)
    )
    if compressed_size != reader.size - TEXTURE_WRAPPER_SIZE:
        raise ValueError(
            f"texture compressed size {compressed_size} != payload size "
            f"{reader.size - TEXTURE_WRAPPER_SIZE}"
        )
    if (
        width <= 0
        or height <= 0
        or width > TEXTURE_MAX_DIMENSION
        or height > TEXTURE_MAX_DIMENSION
    ):
        raise BinaryBoundsError(
            "texture dimensions must be within 1.."
            f"{TEXTURE_MAX_DIMENSION}: {width}x{height}"
        )
    rgba_size = width * height * 4
    if width * height > TEXTURE_MAX_PIXELS:
        raise BinaryBoundsError(
            f"texture pixel count {width * height} exceeds {TEXTURE_MAX_PIXELS}"
        )

    rgba, token_count, data_offset = decompress_cell_lz77(
        reader.bytes_at(
            "texture compressed payload", TEXTURE_WRAPPER_SIZE, compressed_size
        ),
        expected_size=rgba_size,
    )
    return MotionPortraitTexture(
        width=width,
        height=height,
        rgba=rgba,
        wrapper_word_0=wrapper_word_0,
        wrapper_word_4=wrapper_word_4,
        compressed_size=compressed_size,
        token_count=token_count,
        data_offset=data_offset,
    )


__all__ = [
    "MotionPortraitTexture",
    "decode_motionportrait_texture",
    "decompress_cell_lz77",
]
