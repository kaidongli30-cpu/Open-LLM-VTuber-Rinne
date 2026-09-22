from __future__ import annotations

from dataclasses import asdict, dataclass

from .checked_binary import BinaryBoundsError, CheckedBinary
from .mpb_v39 import MpbV39Layout, StructuralSection, walk_v39_to_tail


TAIL_TYPE_SENTINEL = 0x83
TAIL_MAX_TYPE_ID = 0x83
TAIL_GROUP_COUNT_SAFETY_LIMIT = 0x84


@dataclass(frozen=True)
class TailTypeRecord:
    source_index: int
    type_id: int
    outer_index: int
    key_08: int
    key_10: int
    key_14: int
    synthetic: bool = False


@dataclass(frozen=True)
class TailBlock:
    name: str
    offset: int
    length: int
    element_count: int
    element_size: int

    @property
    def end(self) -> int:
        return self.offset + self.length


@dataclass(frozen=True)
class MpbV39HandedTailLayout:
    tail_offset: int
    first_word: int
    second_word: int
    outer_counts: tuple[int, ...]
    type_records: tuple[TailTypeRecord, ...]
    map_25bc_by_type: dict[int, int]
    map_2650_by_type: dict[int, int]
    map_26d8_by_type: dict[int, int]
    trailing_group_ids: tuple[int, ...]
    blocks: tuple[TailBlock, ...]
    end_offset: int
    file_size: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def summary_dict(self) -> dict[str, object]:
        return {
            "tail_offset": self.tail_offset,
            "first_word": self.first_word,
            "second_word": self.second_word,
            "outer_counts": self.outer_counts,
            "source_type_record_count": sum(
                1 for record in self.type_records if not record.synthetic
            ),
            "effective_type_record_count": len(self.type_records),
            "map_25bc_category_count": len(set(self.map_25bc_by_type.values())),
            "map_2650_category_count": len(set(self.map_2650_by_type.values())),
            "map_26d8_category_count": len(set(self.map_26d8_by_type.values())),
            "trailing_group_ids": self.trailing_group_ids,
            "block_count": len(self.blocks),
            "end_offset": self.end_offset,
            "file_size": self.file_size,
            "remaining_bytes": self.file_size - self.end_offset,
        }


class _TailCursor:
    def __init__(self, reader: CheckedBinary, offset: int) -> None:
        self.reader = reader
        self.offset = offset
        self.blocks: list[TailBlock] = []

    def take(self, name: str, element_count: int, element_size: int) -> TailBlock:
        span = self.reader.array_span(
            name,
            self.offset,
            element_count,
            element_size,
            max_count=1_000_000_000,
        )
        block = TailBlock(
            name=name,
            offset=span.offset,
            length=span.length,
            element_count=element_count,
            element_size=element_size,
        )
        self.blocks.append(block)
        self.offset = span.end
        return block


def _assign_simple_categories(
    records: tuple[TailTypeRecord, ...], key_name: str
) -> tuple[dict[int, int], tuple[TailTypeRecord, ...]]:
    representatives: list[TailTypeRecord] = []
    mapping: dict[int, int] = {}
    for record in records:
        key_value = getattr(record, key_name)
        category = next(
            (
                index
                for index, representative in enumerate(representatives)
                if representative.outer_index == record.outer_index
                and getattr(representative, key_name) == key_value
            ),
            None,
        )
        if category is None:
            category = len(representatives)
            representatives.append(record)
        mapping[record.type_id] = category
    return mapping, tuple(representatives)


def _assign_25bc_categories(
    records: tuple[TailTypeRecord, ...]
) -> tuple[dict[int, int], tuple[TailTypeRecord, ...]]:
    """Mirror the v39 grouping at game RVA 0x6770 without naming its semantics."""

    representatives: list[TailTypeRecord] = []
    mapping: dict[int, int] = {}
    for record in records:
        category: int | None = None
        for index, representative in enumerate(representatives):
            if (
                representative.outer_index != record.outer_index
                or representative.key_10 != record.key_10
            ):
                continue
            if record.outer_index == 0 and record.key_10 == 0:
                if record.type_id != 0 and representative.type_id != 0:
                    category = index
                    break
                continue
            category = index
            break

        if category is None:
            category = len(representatives)
            representatives.append(record)
        mapping[record.type_id] = category
    return mapping, tuple(representatives)


def _outer_for_type(
    records: tuple[TailTypeRecord, ...], type_id: int, outer_count: int
) -> int | None:
    for record in records:
        if record.type_id == type_id:
            return record.outer_index
    return 0 if not records and outer_count > 0 else None


def parse_v39_handed_tail(
    data: bytes | bytearray | memoryview,
    *,
    tail_offset: int,
    primary_record_count: int,
    source_type_records: tuple[TailTypeRecord, ...],
    outer_counts: tuple[int, ...],
) -> MpbV39HandedTailLayout:
    """Parse the handed-off v39 payload using the game's consumer copy lengths.

    Names tied to object offsets (25bc, 2650 and 26d8) are intentionally
    structural. The consumer proves the byte boundaries but does not by itself
    prove that a block is a mesh, UV table, mask or another rendering concept.
    """

    reader = CheckedBinary(data)
    if not outer_counts:
        raise BinaryBoundsError("handed tail: no variable outer records")
    if primary_record_count < 0 or primary_record_count > 4096:
        raise BinaryBoundsError(
            "handed tail: primary record count exceeds safety limit"
        )
    if len(source_type_records) > TAIL_MAX_TYPE_ID:
        raise BinaryBoundsError(
            "handed tail: source type record count exceeds safety limit"
        )
    if len(outer_counts) > 0x100:
        raise BinaryBoundsError(
            "handed tail: variable outer record count exceeds byte-sized mapping"
        )
    if any(count < 0 or count > 1_000_000 for count in outer_counts):
        raise BinaryBoundsError(
            "handed tail: variable record count exceeds safety limit"
        )

    seen_types: set[int] = set()
    for record in source_type_records:
        if record.synthetic:
            raise ValueError("source type records must not include synthetic records")
        if record.type_id < 0 or record.type_id > TAIL_MAX_TYPE_ID:
            raise BinaryBoundsError(
                f"handed tail: type id {record.type_id} exceeds {TAIL_MAX_TYPE_ID}"
            )
        if record.type_id in seen_types:
            raise ValueError(f"handed tail: duplicate type id {record.type_id}")
        if record.outer_index < 0 or record.outer_index >= len(outer_counts):
            raise BinaryBoundsError(
                f"handed tail: type {record.type_id} outer index "
                f"{record.outer_index} exceeds outer record count {len(outer_counts)}"
            )
        if record.outer_index > 0xFF or any(
            key < 0 or key > 0xFF
            for key in (record.key_08, record.key_10, record.key_14)
        ):
            raise BinaryBoundsError(
                f"handed tail: type {record.type_id} has a non-byte mapping key"
            )
        seen_types.add(record.type_id)

    effective_records = source_type_records
    if source_type_records:
        if TAIL_TYPE_SENTINEL in seen_types:
            raise ValueError("handed tail: source records contain reserved type 0x83")
        effective_records += (
            TailTypeRecord(
                source_index=len(source_type_records),
                type_id=TAIL_TYPE_SENTINEL,
                outer_index=0,
                key_08=0,
                key_10=1,
                key_14=0,
                synthetic=True,
            ),
        )

    map_25bc, representatives_25bc = _assign_25bc_categories(effective_records)
    map_2650, representatives_2650 = _assign_simple_categories(
        effective_records, "key_08"
    )
    map_26d8, representatives_26d8 = _assign_simple_categories(
        effective_records, "key_14"
    )

    cursor = _TailCursor(reader, tail_offset)
    first_word = reader.u32("handed tail first word", cursor.offset)
    second_word = reader.u32("handed tail second word", cursor.offset + 4)
    cursor.take("handed tail header", 2, 4)

    for type_id in (0, 2):
        outer_index = _outer_for_type(
            effective_records, type_id, len(outer_counts)
        )
        element_count = outer_counts[outer_index] if outer_index is not None else 0
        cursor.take(f"type {type_id} four-byte block", element_count, 4)

    for category, representative in enumerate(representatives_25bc):
        element_count = primary_record_count * outer_counts[representative.outer_index]
        cursor.take(f"25bc category {category} eight-byte block", element_count, 8)

    for category, representative in enumerate(representatives_26d8):
        cursor.take(
            f"26d8 category {category} four-byte block",
            outer_counts[representative.outer_index],
            4,
        )

    group_count = reader.u32("trailing group count", cursor.offset)
    # The game accepts a signed count without this upper bound. The stricter
    # limit is a compatibility-runtime safety policy for untrusted files.
    if group_count > TAIL_GROUP_COUNT_SAFETY_LIMIT:
        raise BinaryBoundsError(
            f"trailing group count {group_count} exceeds "
            f"{TAIL_GROUP_COUNT_SAFETY_LIMIT}"
        )
    cursor.take("trailing group count", 1, 4)

    trailing_group_ids: list[int] = []
    for group_index in range(group_count):
        group_id = reader.u32(f"trailing group {group_index} id", cursor.offset)
        cursor.take(f"trailing group {group_index} id", 1, 4)
        if group_id >= len(representatives_2650):
            raise BinaryBoundsError(
                f"trailing group {group_index}: id {group_id} has no 2650 category"
            )
        representative = representatives_2650[group_id]
        cursor.take(
            f"trailing group {group_index} four-byte block",
            outer_counts[representative.outer_index],
            4,
        )
        trailing_group_ids.append(group_id)

    if cursor.offset != reader.size:
        raise BinaryBoundsError(
            "handed tail consumer did not end at file boundary: "
            f"0x{cursor.offset:x} != 0x{reader.size:x}"
        )

    return MpbV39HandedTailLayout(
        tail_offset=tail_offset,
        first_word=first_word,
        second_word=second_word,
        outer_counts=outer_counts,
        type_records=effective_records,
        map_25bc_by_type=map_25bc,
        map_2650_by_type=map_2650,
        map_26d8_by_type=map_26d8,
        trailing_group_ids=tuple(trailing_group_ids),
        blocks=tuple(cursor.blocks),
        end_offset=cursor.offset,
        file_size=reader.size,
    )


def _section(layout: MpbV39Layout, name: str) -> StructuralSection:
    matches = [section for section in layout.sections if section.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one section named {name!r}, found {len(matches)}")
    return matches[0]


def walk_v39_handed_tail(
    data: bytes | bytearray | memoryview,
) -> MpbV39HandedTailLayout:
    """Walk an entire v39 MPB and fully consume its handed-off tail."""

    layout = walk_v39_to_tail(data)
    if layout.tail_offset is None:
        raise ValueError("MPB v39 does not expose a handed-off tail")
    reader = CheckedBinary(data)

    group_a = _section(layout, "group A records")
    group_a_count = layout.observed_counts["group_a_count"]
    if group_a_count > TAIL_MAX_TYPE_ID:
        raise BinaryBoundsError(
            "handed tail: source type record count exceeds safety limit"
        )
    source_type_records = tuple(
        TailTypeRecord(
            source_index=index,
            type_id=reader.u32(f"group A record {index} type", offset + 4),
            outer_index=reader.u32(f"group A record {index} outer", offset + 0x0C),
            key_08=reader.u32(f"group A record {index} key 08", offset + 0x08),
            key_10=reader.u32(f"group A record {index} key 10", offset + 0x10),
            key_14=reader.u32(f"group A record {index} key 14", offset + 0x14),
        )
        for index in range(group_a_count)
        for offset in (group_a.offset + index * 0x60,)
    )

    variable_count = layout.observed_counts["variable_outer_record_count"]
    if variable_count > 0x100:
        raise BinaryBoundsError(
            "handed tail: variable outer record count exceeds byte-sized mapping"
        )
    outer_counts = tuple(
        reader.u32(
            f"variable record {index} count at 60",
            _section(layout, f"variable record {index} prefix").offset + 0x60,
        )
        for index in range(variable_count)
    )

    return parse_v39_handed_tail(
        data,
        tail_offset=layout.tail_offset,
        primary_record_count=layout.primary.primary_record_count,
        source_type_records=source_type_records,
        outer_counts=outer_counts,
    )


__all__ = [
    "MpbV39HandedTailLayout",
    "TailBlock",
    "TailTypeRecord",
    "parse_v39_handed_tail",
    "walk_v39_handed_tail",
]
