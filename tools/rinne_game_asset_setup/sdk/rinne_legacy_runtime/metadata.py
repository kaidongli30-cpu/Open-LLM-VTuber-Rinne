from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

from .checked_binary import BinaryBoundsError


TEXT_METADATA_MAX_BYTES = 4_096
LAYER_NAME_MAX_BYTES = 255
LAYER_RECORD_MAX_COUNT = 4_096
LAYER_GROUP_MARKERS = frozenset((0xFD, 0xFE, 0xFF))
_INTEGER_PATTERN = re.compile(r"[+]?[0-9]+\Z")
_SIGNED_INTEGER_PATTERN = re.compile(r"[+-]?[0-9]+\Z")
_FLOAT_PATTERN = re.compile(
    r"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?)|(?:\.[0-9]+))(?:[eE][+-]?[0-9]+)?\Z"
)
_LAYER_NAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]+\Z")


@dataclass(frozen=True)
class MotionPortraitScreen:
    first: int
    second: int
    third: int
    fourth: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class MotionPortraitConfig:
    grid_width: int
    grid_height: int
    pair: tuple[float, float]
    integer: int
    quad: tuple[float, float, float, float]
    flag: int
    scalar_a: float
    scalar_b: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MotionPortraitLayerName:
    offset: int
    group_marker: int
    layer_id: int
    name: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _decode_small_ascii(data: bytes | bytearray | memoryview, label: str) -> str:
    byte_view = memoryview(data).cast("B")
    if len(byte_view) > TEXT_METADATA_MAX_BYTES:
        raise BinaryBoundsError(
            f"{label} size {len(byte_view)} exceeds {TEXT_METADATA_MAX_BYTES}"
        )
    try:
        return bytes(byte_view).decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not ASCII") from exc


def _parse_integer(token: str, label: str, *, signed: bool = True) -> int:
    pattern = _SIGNED_INTEGER_PATTERN if signed else _INTEGER_PATTERN
    if pattern.fullmatch(token) is None:
        raise ValueError(f"{label} is not a decimal integer: {token!r}")
    value = int(token, 10)
    if abs(value) > 1_000_000_000:
        raise BinaryBoundsError(f"{label} exceeds the integer safety limit")
    return value


def _parse_float(token: str, label: str) -> float:
    if _FLOAT_PATTERN.fullmatch(token) is None:
        raise ValueError(f"{label} is not a decimal float: {token!r}")
    value = float(token)
    if not math.isfinite(value) or abs(value) > 1_000_000:
        raise BinaryBoundsError(f"{label} exceeds the finite float safety limit")
    return value


def parse_screen_text(
    data: bytes | bytearray | memoryview,
) -> MotionPortraitScreen:
    text = _decode_small_ascii(data, "screen.txt")
    tokens = text.split(",")
    if len(tokens) != 4:
        raise ValueError("screen.txt must contain exactly four comma-separated integers")
    values = tuple(
        _parse_integer(token, f"screen.txt field {index}")
        for index, token in enumerate(tokens)
    )
    return MotionPortraitScreen(*values)


def parse_config_text(
    data: bytes | bytearray | memoryview,
) -> MotionPortraitConfig:
    text = _decode_small_ascii(data, "Config.txt")
    lines = text.splitlines()
    if len(lines) != 7 or any(not line for line in lines):
        raise ValueError("Config.txt must contain exactly seven non-empty lines")
    fields = [line.split() for line in lines]
    expected_counts = (2, 2, 1, 4, 1, 1, 1)
    for index, (tokens, expected) in enumerate(zip(fields, expected_counts, strict=True)):
        if len(tokens) != expected or any(not token for token in tokens):
            raise ValueError(
                f"Config.txt line {index + 1} must contain {expected} fields"
            )

    grid_width = _parse_integer(fields[0][0], "Config.txt grid width", signed=False)
    grid_height = _parse_integer(fields[0][1], "Config.txt grid height", signed=False)
    if not (10 <= grid_width <= 80 and 10 <= grid_height <= 80):
        raise BinaryBoundsError("Config.txt grid dimensions must be within 10..80")
    pair = tuple(
        _parse_float(token, f"Config.txt pair {index}")
        for index, token in enumerate(fields[1])
    )
    integer = _parse_integer(fields[2][0], "Config.txt integer")
    quad = tuple(
        _parse_float(token, f"Config.txt quad {index}")
        for index, token in enumerate(fields[3])
    )
    flag = _parse_integer(fields[4][0], "Config.txt flag")
    scalar_a = _parse_float(fields[5][0], "Config.txt scalar A")
    scalar_b = _parse_float(fields[6][0], "Config.txt scalar B")
    return MotionPortraitConfig(
        grid_width=grid_width,
        grid_height=grid_height,
        pair=(pair[0], pair[1]),
        integer=integer,
        quad=(quad[0], quad[1], quad[2], quad[3]),
        flag=flag,
        scalar_a=scalar_a,
        scalar_b=scalar_b,
    )


def parse_layer_names(
    data: bytes | bytearray | memoryview,
) -> tuple[MotionPortraitLayerName, ...]:
    """Parse the delimiter convention shared by the four verified Rinne samples.

    FD, FE and FF are treated as reserved group delimiters for this scoped
    compatibility parser. This is not claimed as a universal MotionPortrait
    constraint without additional consumer evidence.
    """

    byte_view = memoryview(data).cast("B")
    if len(byte_view) > TEXT_METADATA_MAX_BYTES:
        raise BinaryBoundsError(
            f"layername.bin size {len(byte_view)} exceeds {TEXT_METADATA_MAX_BYTES}"
        )
    stable_data = bytes(byte_view)
    position = 0
    group_marker: int | None = None
    records: list[MotionPortraitLayerName] = []

    while position < len(stable_data):
        if stable_data[position] in LAYER_GROUP_MARKERS:
            group_marker = stable_data[position]
            position += 1
            if position >= len(stable_data):
                raise BinaryBoundsError("layername.bin ends after a group marker")
        if group_marker is None:
            raise ValueError("layername.bin must begin with a group marker")
        if len(records) >= LAYER_RECORD_MAX_COUNT:
            raise BinaryBoundsError("layername.bin record count exceeds safety limit")

        record_offset = position
        layer_id = stable_data[position]
        if layer_id in LAYER_GROUP_MARKERS:
            raise ValueError("layername.bin contains an empty group")
        position += 1
        terminator = stable_data.find(b"\x00", position)
        if terminator < 0:
            raise BinaryBoundsError("layername.bin contains an unterminated name")
        if terminator == position or terminator - position > LAYER_NAME_MAX_BYTES:
            raise BinaryBoundsError("layername.bin contains an invalid name length")
        try:
            name = stable_data[position:terminator].decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("layername.bin contains a non-ASCII name") from exc
        if _LAYER_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError(f"layername.bin contains an unsafe name: {name!r}")
        if name in {".", ".."}:
            raise ValueError(f"layername.bin contains a reserved path name: {name!r}")
        records.append(
            MotionPortraitLayerName(
                offset=record_offset,
                group_marker=group_marker,
                layer_id=layer_id,
                name=name,
            )
        )
        position = terminator + 1

    if not records:
        raise ValueError("layername.bin contains no records")
    return tuple(records)


__all__ = [
    "MotionPortraitConfig",
    "MotionPortraitLayerName",
    "MotionPortraitScreen",
    "parse_config_text",
    "parse_layer_names",
    "parse_screen_text",
]
