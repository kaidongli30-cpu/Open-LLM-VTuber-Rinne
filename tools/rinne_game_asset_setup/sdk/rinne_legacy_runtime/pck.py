from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from .checked_binary import BinaryBoundsError


PCK_FILENAME_TAG = b"Filename" + b" " * 12
PCK_PACK_TAG = b"Pack" + b" " * 16
PCK_MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024
PCK_MAX_ENTRY_COUNT = 65_536
PCK_MAX_MEMBER_BYTES = 512 * 1024 * 1024
PCK_MAX_NAME_BYTES = 1_024
PCK_PACK_SCAN_BYTES = 256


@dataclass(frozen=True)
class PckEntry:
    name: str
    offset: int
    size: int


class PckArchive:
    """Bounded read-only access to members in the game's PCK container."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.entries = self._read_index()
        self.by_name = {entry.name: entry for entry in self.entries}

    def _read_exact(self, handle, size: int, *, name: str) -> bytes:
        data = handle.read(size)
        if len(data) != size:
            raise BinaryBoundsError(f"PCK {name} is truncated")
        return data

    def _read_index(self) -> tuple[PckEntry, ...]:
        file_size = self.path.stat().st_size
        if file_size < 28 or file_size > PCK_MAX_FILE_BYTES:
            raise BinaryBoundsError(
                f"PCK size {file_size} is outside 28..{PCK_MAX_FILE_BYTES}"
            )
        with self.path.open("rb") as handle:
            if self._read_exact(handle, 20, name="filename tag") != PCK_FILENAME_TAG:
                raise ValueError("not a supported PCK filename section")
            name_section_hint, first_relative = struct.unpack(
                "<II", self._read_exact(handle, 8, name="filename header")
            )
            if first_relative == 0 or first_relative % 4:
                raise ValueError("invalid PCK filename offset table")
            count = first_relative // 4
            if count <= 0 or count > PCK_MAX_ENTRY_COUNT:
                raise BinaryBoundsError(
                    f"PCK entry count {count} exceeds {PCK_MAX_ENTRY_COUNT}"
                )
            remaining_offsets = self._read_exact(
                handle, (count - 1) * 4, name="filename offsets"
            )
            relative_offsets = (first_relative,) + (
                struct.unpack(f"<{count - 1}I", remaining_offsets)
                if count > 1
                else ()
            )

            names: list[str] = []
            seen_names: set[str] = set()
            for entry_index, relative in enumerate(relative_offsets):
                name_offset = 24 + relative
                if name_offset < 24 + count * 4 or name_offset >= file_size:
                    raise BinaryBoundsError(
                        f"PCK filename {entry_index} offset is outside the name table"
                    )
                handle.seek(name_offset)
                raw = bytearray()
                for _ in range(PCK_MAX_NAME_BYTES + 1):
                    byte = handle.read(1)
                    if not byte:
                        raise BinaryBoundsError(
                            f"PCK filename {entry_index} is truncated"
                        )
                    if byte == b"\0":
                        break
                    raw.extend(byte)
                else:
                    raise BinaryBoundsError(
                        f"PCK filename {entry_index} exceeds {PCK_MAX_NAME_BYTES} bytes"
                    )
                if not raw:
                    raise ValueError(f"PCK filename {entry_index} is empty")
                try:
                    name = raw.decode("ascii")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        f"PCK filename {entry_index} is not ASCII"
                    ) from exc
                if name in seen_names:
                    raise ValueError(f"duplicate PCK member name: {name}")
                names.append(name)
                seen_names.add(name)

            if name_section_hint >= file_size:
                raise BinaryBoundsError("PCK pack-section hint exceeds file size")
            scan_start = max(0, name_section_hint - 64)
            handle.seek(scan_start)
            scan = handle.read(min(PCK_PACK_SCAN_BYTES, file_size - scan_start))
            relative_pack_position = scan.find(PCK_PACK_TAG)
            if relative_pack_position < 0:
                raise ValueError("PCK pack index was not found near its hint")
            pack_position = scan_start + relative_pack_position
            handle.seek(pack_position + len(PCK_PACK_TAG))
            pack_section_size, pack_count = struct.unpack(
                "<II", self._read_exact(handle, 8, name="pack header")
            )
            if pack_count != count:
                raise ValueError(
                    f"PCK filename/pack count mismatch: {count} != {pack_count}"
                )
            minimum_pack_size = len(PCK_PACK_TAG) + 8 + count * 8
            if pack_section_size < minimum_pack_size:
                raise ValueError("PCK pack section is smaller than its record table")
            if pack_position + pack_section_size > file_size:
                raise BinaryBoundsError("PCK pack section exceeds file size")
            records = self._read_exact(handle, count * 8, name="pack records")

        entries: list[PckEntry] = []
        for entry_index, name in enumerate(names):
            offset, size = struct.unpack_from("<II", records, entry_index * 8)
            if size > PCK_MAX_MEMBER_BYTES:
                raise BinaryBoundsError(
                    f"PCK member {name!r} exceeds {PCK_MAX_MEMBER_BYTES} bytes"
                )
            if offset < pack_position + pack_section_size or offset + size > file_size:
                raise BinaryBoundsError(
                    f"PCK member {name!r} range exceeds the data section"
                )
            entries.append(PckEntry(name=name, offset=offset, size=size))
        return tuple(entries)

    def read(self, name: str) -> bytes:
        try:
            entry = self.by_name[name]
        except KeyError as exc:
            raise KeyError(f"PCK member not found: {name}") from exc
        with self.path.open("rb") as handle:
            handle.seek(entry.offset)
            return self._read_exact(handle, entry.size, name=f"member {name!r}")


__all__ = [
    "PCK_FILENAME_TAG",
    "PCK_MAX_ENTRY_COUNT",
    "PCK_MAX_FILE_BYTES",
    "PCK_MAX_MEMBER_BYTES",
    "PCK_MAX_NAME_BYTES",
    "PCK_PACK_TAG",
    "PckArchive",
    "PckEntry",
]
