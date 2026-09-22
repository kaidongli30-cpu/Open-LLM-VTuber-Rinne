from __future__ import annotations

import struct
from dataclasses import dataclass

from .checked_binary import BinaryBoundsError


STSC_MAGIC = b"STSC"
STSC_HEADER_SIZE = 0x3C
STSC_VERSION = 7
STSC_STRING_POOL_MARKER = 0x0E
RINNE_PORTRAIT_ID_MIN = 60_101
RINNE_PORTRAIT_ID_MAX = 60_115
RINNE_PORTRAIT_ARGUMENT_PREFIX = b"\x62\x00"


@dataclass(frozen=True)
class StscString:
    offset: int
    raw: bytes

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        return self.raw.decode(encoding, errors)


@dataclass(frozen=True)
class StscStringReference:
    code_offset: int
    string: StscString


@dataclass(frozen=True)
class RinnePortraitUse:
    command_offset: int
    portrait_id: int


@dataclass(frozen=True)
class StscScript:
    name: str
    version: int
    string_pool_offset: int
    code: bytes
    strings: tuple[StscString, ...]

    @property
    def strings_by_offset(self) -> dict[int, StscString]:
        return {entry.offset: entry for entry in self.strings}

    def candidate_string_references(
        self,
        start: int,
        end: int,
    ) -> tuple[StscStringReference, ...]:
        """Find exact string-pool pointers in a bounded bytecode interval.

        The complete STSC opcode table is not claimed here. The returned
        locations are deliberately named *candidates*: they are unaligned
        little-endian values that exactly address a parsed string boundary.
        This is sufficient for reproducible local evidence scans without
        pretending that every surrounding opcode has already been decoded.
        """

        if start < 0 or end < start or end > len(self.code):
            raise BinaryBoundsError("STSC reference interval is outside bytecode")
        strings_by_offset = self.strings_by_offset
        references: list[StscStringReference] = []
        for code_offset in range(start, max(start, end - 3)):
            value = struct.unpack_from("<I", self.code, code_offset)[0]
            entry = strings_by_offset.get(value)
            if entry is not None:
                references.append(
                    StscStringReference(code_offset=code_offset, string=entry)
                )
        return tuple(references)


def _parse_script_name(data: bytes) -> str:
    raw = data[12:44].split(b"\0", 1)[0]
    if not raw:
        raise ValueError("STSC script name is empty")
    try:
        return raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("STSC script name is not ASCII") from exc


def _parse_string_pool(data: bytes, offset: int) -> tuple[StscString, ...]:
    strings: list[StscString] = []
    cursor = offset
    while cursor < len(data):
        terminator = data.find(b"\0", cursor)
        if terminator < 0:
            raise BinaryBoundsError("STSC string pool is not null terminated")
        if terminator > cursor:
            strings.append(StscString(offset=cursor, raw=data[cursor:terminator]))
        cursor = terminator + 1
    return tuple(strings)


def parse_stsc(data: bytes) -> StscScript:
    """Parse the bounded STSC container facts used by the evidence scanner."""

    if not isinstance(data, bytes):
        raise TypeError("STSC input must be bytes")
    if len(data) < STSC_HEADER_SIZE + 5:
        raise BinaryBoundsError("STSC input is smaller than its fixed header")
    if data[:4] != STSC_MAGIC:
        raise ValueError("not an STSC script")
    header_size, version = struct.unpack_from("<II", data, 4)
    if header_size != STSC_HEADER_SIZE:
        raise ValueError(f"unsupported STSC header size: {header_size}")
    if version != STSC_VERSION:
        raise ValueError(f"unsupported STSC version: {version}")
    if data[STSC_HEADER_SIZE] != STSC_STRING_POOL_MARKER:
        raise ValueError("STSC string-pool marker is missing")
    string_pool_offset = struct.unpack_from("<I", data, STSC_HEADER_SIZE + 1)[0]
    if string_pool_offset < STSC_HEADER_SIZE + 5 or string_pool_offset > len(data):
        raise BinaryBoundsError("STSC string-pool offset is outside the file")
    return StscScript(
        name=_parse_script_name(data),
        version=version,
        string_pool_offset=string_pool_offset,
        code=data[:string_pool_offset],
        strings=_parse_string_pool(data, string_pool_offset),
    )


def find_rinne_portrait_uses(script: StscScript) -> tuple[RinnePortraitUse, ...]:
    """Find the game-observed integer arguments for MP060101..MP060115.

    ``62 00 <uint32 portrait id>`` is treated as an observed argument
    signature, not as a guessed name for opcode 0x62. Its ordered results are
    independently identical in the Japanese and English script archives.
    """

    if not isinstance(script, StscScript):
        raise TypeError("portrait scan requires a parsed STSC script")
    uses: list[RinnePortraitUse] = []
    cursor = STSC_HEADER_SIZE
    while cursor + 6 <= len(script.code):
        command_offset = script.code.find(RINNE_PORTRAIT_ARGUMENT_PREFIX, cursor)
        if command_offset < 0 or command_offset + 6 > len(script.code):
            break
        portrait_id = struct.unpack_from("<I", script.code, command_offset + 2)[0]
        if RINNE_PORTRAIT_ID_MIN <= portrait_id <= RINNE_PORTRAIT_ID_MAX:
            uses.append(
                RinnePortraitUse(
                    command_offset=command_offset,
                    portrait_id=portrait_id,
                )
            )
        cursor = command_offset + 2
    return tuple(uses)


__all__ = [
    "RINNE_PORTRAIT_ARGUMENT_PREFIX",
    "RINNE_PORTRAIT_ID_MAX",
    "RINNE_PORTRAIT_ID_MIN",
    "RinnePortraitUse",
    "STSC_HEADER_SIZE",
    "STSC_MAGIC",
    "STSC_STRING_POOL_MARKER",
    "STSC_VERSION",
    "StscScript",
    "StscString",
    "StscStringReference",
    "find_rinne_portrait_uses",
    "parse_stsc",
]
