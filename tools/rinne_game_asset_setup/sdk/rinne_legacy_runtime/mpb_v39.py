from __future__ import annotations

from dataclasses import asdict, dataclass

from .checked_binary import BinaryBoundsError, CheckedBinary, Span


ITOM_MAGIC = b"ITOM"
VERSION_OFFSET = 0x10
V39_VERSION = 39
V39_FIXED_BLOCK_OFFSET = 0x80
V39_FIXED_BLOCK_LENGTH = 0x1490
V39_GAP_OFFSET = 0x1510
V39_PRIMARY_HEADER_OFFSET = 0x1520
V39_PRIMARY_HEADER_LENGTH = 0x40
V39_PRIMARY_RECORDS_OFFSET = 0x1560
V39_PRIMARY_SOURCE_STRIDE = 0x210
V39_PRIMARY_MAX_COUNT = 4096
V39_STRUCTURAL_MAX_COUNT = 1_000_000
V39_MAX_FILE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class MpbV39PrimaryLayout:
    file_size: int
    version: int
    fixed_block: Span
    uninterpreted_gap: Span
    primary_header: Span
    primary_record_count: int
    primary_source_stride: int
    primary_records: Span
    next_offset: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StructuralSection:
    name: str
    offset: int
    storage_length: int
    payload_length: int

    @property
    def end(self) -> int:
        return self.offset + self.storage_length


@dataclass(frozen=True)
class MpbV39Layout:
    primary: MpbV39PrimaryLayout
    sections: tuple[StructuralSection, ...]
    observed_counts: dict[str, int]
    final_flag: int
    tail_offset: int | None
    tail_length: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def summary_dict(self) -> dict[str, object]:
        return {
            "file_size": self.primary.file_size,
            "version": self.primary.version,
            "section_count": len(self.sections),
            "observed_counts": self.observed_counts,
            "final_flag": self.final_flag,
            "tail_offset": self.tail_offset,
            "tail_length": self.tail_length,
        }


class _SectionCursor:
    def __init__(self, reader: CheckedBinary, offset: int) -> None:
        self.reader = reader
        self.offset = offset
        self.sections: list[StructuralSection] = []

    def take(
        self,
        name: str,
        storage_length: int,
        *,
        payload_length: int | None = None,
    ) -> StructuralSection:
        if payload_length is None:
            payload_length = storage_length
        if payload_length < 0 or payload_length > storage_length:
            raise BinaryBoundsError(
                f"{name}: payload length {payload_length} is invalid for "
                f"storage length {storage_length}"
            )
        checked = self.reader.span(name, self.offset, storage_length)
        section = StructuralSection(
            name=name,
            offset=checked.offset,
            storage_length=checked.length,
            payload_length=payload_length,
        )
        self.sections.append(section)
        self.offset = checked.end
        return section

    def count_slot(self, name: str, *, max_count: int = V39_STRUCTURAL_MAX_COUNT) -> int:
        value = self.reader.u32(name, self.offset)
        if value > max_count:
            raise BinaryBoundsError(
                f"{name}: count {value} exceeds safety limit {max_count}"
            )
        self.take(name, 0x10, payload_length=4)
        return value

    def fixed_array(
        self,
        name: str,
        count: int,
        stride: int,
        *,
        max_count: int = V39_STRUCTURAL_MAX_COUNT,
    ) -> StructuralSection:
        checked = self.reader.array_span(
            name, self.offset, count, stride, max_count=max_count
        )
        return self.take(name, checked.length)

    def strict_array(
        self,
        name: str,
        count: int,
        item_size: int,
        *,
        max_count: int = V39_STRUCTURAL_MAX_COUNT,
    ) -> StructuralSection:
        payload = self.reader.array_span(
            name, self.offset, count, item_size, max_count=max_count
        )
        storage_length = self.reader.strict_padded_length(name, payload.length)
        return self.take(name, storage_length, payload_length=payload.length)


def walk_v39_primary(data: bytes | bytearray | memoryview) -> MpbV39PrimaryLayout:
    """Walk the first confirmed counted record region of an ITOM v39 MPB.

    This intentionally stops at the first unverified boundary. It does not name
    the records as mesh, layer, or animation data because those semantics are
    not yet proven.
    """

    reader = CheckedBinary(data)
    magic = reader.bytes_at("ITOM magic", 0, 4)
    if magic != ITOM_MAGIC:
        raise ValueError(f"unexpected magic: {magic!r}; expected {ITOM_MAGIC!r}")

    version = reader.u32("format version", VERSION_OFFSET)
    if version != V39_VERSION:
        raise ValueError(
            f"unsupported MPB version {version}; this walker is specific to v39"
        )

    fixed_block = reader.span(
        "v39 fixed block", V39_FIXED_BLOCK_OFFSET, V39_FIXED_BLOCK_LENGTH
    )
    uninterpreted_gap = reader.span(
        "uninterpreted bytes before first record header",
        V39_GAP_OFFSET,
        V39_PRIMARY_HEADER_OFFSET - V39_GAP_OFFSET,
    )
    primary_header = reader.span(
        "first record header", V39_PRIMARY_HEADER_OFFSET, V39_PRIMARY_HEADER_LENGTH
    )
    record_count = reader.u32(
        "first record count", V39_PRIMARY_HEADER_OFFSET
    )
    primary_records = reader.array_span(
        "first source record array",
        V39_PRIMARY_RECORDS_OFFSET,
        record_count,
        V39_PRIMARY_SOURCE_STRIDE,
        max_count=V39_PRIMARY_MAX_COUNT,
    )

    return MpbV39PrimaryLayout(
        file_size=reader.size,
        version=version,
        fixed_block=fixed_block,
        uninterpreted_gap=uninterpreted_gap,
        primary_header=primary_header,
        primary_record_count=record_count,
        primary_source_stride=V39_PRIMARY_SOURCE_STRIDE,
        primary_records=primary_records,
        next_offset=primary_records.end,
    )


def walk_v39_to_tail(data: bytes | bytearray | memoryview) -> MpbV39Layout:
    """Walk every confirmed v39 region up to the handed-off tail payload.

    Section names describe parser structure only. They deliberately do not
    assign mesh, UV, layer, or animation semantics that have not been proven.
    """

    primary = walk_v39_primary(data)
    reader = CheckedBinary(data)
    cursor = _SectionCursor(reader, primary.next_offset)
    observed: dict[str, int] = {
        "primary_record_count": primary.primary_record_count
    }

    cursor.fixed_array(
        "secondary source record array",
        primary.primary_record_count,
        0x1D0,
        max_count=V39_PRIMARY_MAX_COUNT,
    )

    group_a_count = cursor.count_slot("group A count")
    observed["group_a_count"] = group_a_count
    cursor.fixed_array("group A records", group_a_count, 0x60)

    group_b_count = cursor.count_slot("group B count")
    observed["group_b_count"] = group_b_count
    cursor.fixed_array("group B records", group_b_count, 0x40)

    cursor.take("version 5 slot", 0x10, payload_length=4)
    cursor.take("version 36 fixed block", 0x60)
    cursor.take("version 8 fixed block and gap", 0x70, payload_length=0x60)
    cursor.take("version 22 slots", 0x20, payload_length=8)
    cursor.take("version 38 slot", 0x10, payload_length=4)
    cursor.take("version 39 slot", 0x10, payload_length=4)
    cursor.take("version 19 slots", 0x40, payload_length=16)

    optional_count = cursor.count_slot("optional record count")
    observed["optional_record_count"] = optional_count
    cursor.fixed_array("optional records", optional_count, 0xA0)
    cursor.take("version 23 slots", 0x20, payload_length=8)

    variable_header_offset = cursor.offset
    variable_format = reader.u32("variable group format", variable_header_offset)
    variable_count = reader.u32(
        "variable outer record count", variable_header_offset + 0x10
    )
    if variable_count > 4096:
        raise BinaryBoundsError(
            "variable outer record count: count "
            f"{variable_count} exceeds safety limit 4096"
        )
    cursor.take("variable group header", 0x20, payload_length=8)
    observed["variable_group_format"] = variable_format
    observed["variable_outer_record_count"] = variable_count

    for record_index in range(variable_count):
        prefix = cursor.offset
        c4 = reader.u32(f"variable record {record_index} c4", prefix + 0x10)
        c8 = reader.u32(f"variable record {record_index} c8", prefix + 0x60)
        if c4 > V39_STRUCTURAL_MAX_COUNT or c8 > V39_STRUCTURAL_MAX_COUNT:
            raise BinaryBoundsError(
                f"variable record {record_index}: count exceeds safety limit"
            )
        cursor.take(f"variable record {record_index} prefix", 0x70, payload_length=28)
        cursor.strict_array(f"variable record {record_index} payload 8", c8, 8)

        c2a = cursor.count_slot(f"variable record {record_index} payload 2A count")
        if c2a != 0:
            cursor.strict_array(f"variable record {record_index} payload 2A", c2a, 2)

        c2b = cursor.count_slot(f"variable record {record_index} payload 2B count")
        if c2b != 0:
            cursor.strict_array(f"variable record {record_index} payload 2B", c2b, 2)

        cursor.take(f"variable record {record_index} two slots", 0x20, payload_length=8)
        if c4 != 0:
            for payload_index in range(4):
                cursor.strict_array(
                    f"variable record {record_index} payload 4[{payload_index}]",
                    c4,
                    4,
                )

        cursor.take(
            f"variable record {record_index} opaque blob A",
            0x70,
            payload_length=0x65,
        )
        cursor.take(
            f"variable record {record_index} opaque blob B",
            0x70,
            payload_length=0x65,
        )
        observed[f"variable_record_{record_index}_c4"] = c4
        observed[f"variable_record_{record_index}_c8"] = c8
        observed[f"variable_record_{record_index}_c2a"] = c2a
        observed[f"variable_record_{record_index}_c2b"] = c2b

    if variable_count > 0:
        for record_index in range(3):
            header_offset = cursor.offset
            d2a = reader.i32(
                f"fixed triplet record {record_index} d2A", header_offset + 0x20
            )
            cursor.take(
                f"fixed triplet record {record_index} three slots",
                0x30,
                payload_length=12,
            )
            if d2a > 0:
                cursor.strict_array(
                    f"fixed triplet record {record_index} payload 2A", d2a, 2
                )

            d2b_offset = cursor.offset
            d2b = reader.i32(
                f"fixed triplet record {record_index} d2B", d2b_offset
            )
            cursor.take(
                f"fixed triplet record {record_index} payload 2B count",
                0x10,
                payload_length=4,
            )
            if d2b > 0:
                cursor.strict_array(
                    f"fixed triplet record {record_index} payload 2B", d2b, 2
                )

            five_slots_offset = cursor.offset
            d4 = reader.u32(
                f"fixed triplet record {record_index} d4", five_slots_offset + 0x40
            )
            if d4 > V39_STRUCTURAL_MAX_COUNT:
                raise BinaryBoundsError(
                    f"fixed triplet record {record_index} d4: count {d4} "
                    f"exceeds safety limit {V39_STRUCTURAL_MAX_COUNT}"
                )
            cursor.take(
                f"fixed triplet record {record_index} five slots",
                0x50,
                payload_length=20,
            )
            if d4 != 0:
                for payload_index in range(5):
                    cursor.strict_array(
                        f"fixed triplet record {record_index} payload 4[{payload_index}]",
                        d4,
                        4,
                    )
            observed[f"fixed_triplet_{record_index}_d2a"] = d2a
            observed[f"fixed_triplet_{record_index}_d2b"] = d2b
            observed[f"fixed_triplet_{record_index}_d4"] = d4

    tail_rows = cursor.count_slot("version 17 tail row count", max_count=4096)
    observed["tail_row_count"] = tail_rows
    for row_index in range(tail_rows):
        cursor.strict_array(
            f"version 17 tail row {row_index}",
            primary.primary_record_count,
            1,
            max_count=V39_PRIMARY_MAX_COUNT,
        )

    final_flag = reader.u32("final tail flag", cursor.offset)
    cursor.take("final tail flag", 0x10, payload_length=4)
    tail_offset = cursor.offset if final_flag == 1 else None
    tail_length = reader.size - tail_offset if tail_offset is not None else 0
    if tail_offset is not None:
        reader.span("handed-off tail", tail_offset, tail_length)

    return MpbV39Layout(
        primary=primary,
        sections=tuple(cursor.sections),
        observed_counts=observed,
        final_flag=final_flag,
        tail_offset=tail_offset,
        tail_length=tail_length,
    )


__all__ = [
    "BinaryBoundsError",
    "MpbV39PrimaryLayout",
    "MpbV39Layout",
    "V39_PRIMARY_RECORDS_OFFSET",
    "V39_PRIMARY_SOURCE_STRIDE",
    "walk_v39_primary",
    "walk_v39_to_tail",
]
