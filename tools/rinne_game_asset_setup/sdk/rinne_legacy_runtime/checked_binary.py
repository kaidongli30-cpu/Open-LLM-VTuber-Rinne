from __future__ import annotations

import struct
from dataclasses import dataclass


MAX_U64 = (1 << 64) - 1


class BinaryBoundsError(ValueError):
    """Raised when a declared binary region cannot fit in its input."""


@dataclass(frozen=True)
class Span:
    name: str
    offset: int
    length: int

    @property
    def end(self) -> int:
        return self.offset + self.length


class CheckedBinary:
    """Bounds-checked little-endian reader used by the reference parser.

    Python integers do not overflow, but the runtime will later be ported to a
    fixed-width Windows implementation. These checks therefore enforce unsigned
    64-bit arithmetic now, so malformed counts cannot become unsafe later.
    """

    def __init__(self, data: bytes | bytearray | memoryview) -> None:
        self._data = memoryview(data).cast("B")

    @property
    def size(self) -> int:
        return len(self._data)

    def span(self, name: str, offset: int, length: int) -> Span:
        if offset < 0 or length < 0:
            raise BinaryBoundsError(
                f"{name}: negative offset or length ({offset}, {length})"
            )
        if offset > MAX_U64 or length > MAX_U64 or offset + length > MAX_U64:
            raise BinaryBoundsError(f"{name}: unsigned 64-bit range overflow")
        end = offset + length
        if end > self.size:
            raise BinaryBoundsError(
                f"{name}: range 0x{offset:x}..0x{end:x} exceeds "
                f"file size 0x{self.size:x}"
            )
        return Span(name=name, offset=offset, length=length)

    def array_span(
        self,
        name: str,
        offset: int,
        count: int,
        stride: int,
        *,
        max_count: int,
    ) -> Span:
        if count < 0 or stride < 0:
            raise BinaryBoundsError(
                f"{name}: negative count or stride ({count}, {stride})"
            )
        if count > max_count:
            raise BinaryBoundsError(
                f"{name}: count {count} exceeds safety limit {max_count}"
            )
        if count and stride > MAX_U64 // count:
            raise BinaryBoundsError(f"{name}: count * stride overflows u64")
        return self.span(name, offset, count * stride)

    def bytes_at(self, name: str, offset: int, length: int) -> bytes:
        region = self.span(name, offset, length)
        return bytes(self._data[region.offset : region.end])

    def u32(self, name: str, offset: int) -> int:
        raw = self.bytes_at(name, offset, 4)
        return struct.unpack("<I", raw)[0]

    def i32(self, name: str, offset: int) -> int:
        raw = self.bytes_at(name, offset, 4)
        return struct.unpack("<i", raw)[0]

    def align_up(self, name: str, offset: int, alignment: int) -> int:
        if alignment <= 0 or alignment & (alignment - 1):
            raise BinaryBoundsError(
                f"{name}: alignment must be a positive power of two"
            )
        if offset < 0 or offset > MAX_U64 - (alignment - 1):
            raise BinaryBoundsError(f"{name}: alignment arithmetic overflows u64")
        aligned = (offset + alignment - 1) & ~(alignment - 1)
        self.span(name, aligned, 0)
        return aligned

    @staticmethod
    def strict_padded_length(name: str, length: int, alignment: int = 16) -> int:
        """Return the format's strict next-block storage length.

        Unlike ordinary alignment, an already aligned payload consumes one
        additional block. The game parser uses this rule for variable arrays.
        """

        if alignment <= 0 or alignment & (alignment - 1):
            raise BinaryBoundsError(
                f"{name}: alignment must be a positive power of two"
            )
        if length < 0:
            raise BinaryBoundsError(f"{name}: negative payload length {length}")
        if length > MAX_U64 - alignment:
            raise BinaryBoundsError(f"{name}: strict padding overflows u64")
        padded = length + alignment - (length & (alignment - 1))
        if padded > MAX_U64:
            raise BinaryBoundsError(f"{name}: strict padding overflows u64")
        return padded
