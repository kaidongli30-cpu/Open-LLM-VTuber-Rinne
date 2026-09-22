from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from .checked_binary import BinaryBoundsError, CheckedBinary
from .mpb_v39 import MpbV39Layout, StructuralSection, walk_v39_to_tail
from .mpb_v39_renderer import MpbV39UvRectangle


V39_DRAW_SOURCE_STRIDE = 0x60
V39_ATLAS_SOURCE_STRIDE = 0x40
V39_GENERIC_ATLAS_ID_BIAS = 0x1E
V39_DEFAULT_COMPOSITOR_TRIGGER_TYPE = 2
V39_ABSENT_PACKAGED_ATLAS_COORDINATE = -1.0


def _f32(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError as exc:
        raise BinaryBoundsError("atlas float32 arithmetic overflow") from exc


def resolve_v39_runtime_atlas_rectangle(
    rectangle: MpbV39UvRectangle,
) -> MpbV39UvRectangle:
    """Apply the packaged-atlas Y rewrite at game RVA 0x4AE1..0x4B14.

    Version-39 group-B records store atlas rectangles in visually upright
    coordinates while ``tex_all.tex`` rows are uploaded unchanged.  The game
    relocates the rectangle to the uploaded row order without mirroring the
    image inside it: ``(first_y, second_y)`` becomes
    ``(1-second_y, 1-first_y)``.  Records whose first atlas X is the -1
    sentinel skip the rewrite.
    """

    rectangle.validate()
    if rectangle.first_x == V39_ABSENT_PACKAGED_ATLAS_COORDINATE:
        return rectangle
    resolved = MpbV39UvRectangle(
        rectangle.first_x,
        _f32(_f32(1.0) - _f32(rectangle.second_y)),
        rectangle.second_x,
        _f32(_f32(1.0) - _f32(rectangle.first_y)),
    )
    resolved.validate()
    return resolved


@dataclass(frozen=True)
class MpbV39DrawRecord:
    file_index: int
    source_offset: int
    enabled_word: int
    type_id: int
    key_08: int
    outer_index: int
    key_10: int
    key_14: int
    initial_opacity: float
    blend_mode: int = 0
    compositor_trigger_a_word: int = 0
    compositor_trigger_b_word: int = 0

    @property
    def enabled(self) -> bool:
        return bool(self.enabled_word & 0xFF)

    @property
    def triggers_compositor_a(self) -> bool:
        return self.compositor_trigger_a_word != 0

    @property
    def triggers_compositor_b(self) -> bool:
        return self.compositor_trigger_b_word != 0


@dataclass(frozen=True)
class MpbV39CompositorTriggers:
    """Final delayed-draw trigger types selected by the v39 constructor.

    The original runtime initializes both values to type 2, then walks source
    draw records in file order. Every nonzero source ``+0x24``/``+0x28`` word
    replaces the corresponding value, so the last flagged record wins.
    ``a`` and ``b`` remain role-neutral until their complete render-target
    operations have been reproduced.
    """

    a_type_id: int
    b_type_id: int


@dataclass(frozen=True)
class MpbV39AtlasRecord:
    file_index: int
    source_offset: int
    record_id: int
    source_rectangle: MpbV39UvRectangle
    atlas_rectangle: MpbV39UvRectangle
    fallback_rectangle: MpbV39UvRectangle

    @property
    def runtime_atlas_rectangle(self) -> MpbV39UvRectangle:
        return resolve_v39_runtime_atlas_rectangle(self.atlas_rectangle)

    def validate(self) -> None:
        self.source_rectangle.validate(source=True)
        self.atlas_rectangle.validate()
        self.fallback_rectangle.validate()


@dataclass(frozen=True)
class MpbV39AtlasBinding:
    draw_order: int
    type_id: int
    outer_index: int
    atlas_record_id: int
    source_rectangle: MpbV39UvRectangle
    atlas_rectangle: MpbV39UvRectangle
    fallback_rectangle: MpbV39UvRectangle
    initially_enabled: bool
    initial_opacity: float

    @property
    def runtime_atlas_rectangle(self) -> MpbV39UvRectangle:
        return resolve_v39_runtime_atlas_rectangle(self.atlas_rectangle)


@dataclass(frozen=True)
class MpbV39AtlasSemantics:
    draw_records: tuple[MpbV39DrawRecord, ...]
    atlas_records: tuple[MpbV39AtlasRecord, ...]
    generic_bindings: tuple[MpbV39AtlasBinding, ...]

    @property
    def compositor_triggers(self) -> MpbV39CompositorTriggers:
        a_type_id = V39_DEFAULT_COMPOSITOR_TRIGGER_TYPE
        b_type_id = V39_DEFAULT_COMPOSITOR_TRIGGER_TYPE
        for record in self.draw_records:
            if record.triggers_compositor_a:
                a_type_id = record.type_id
            if record.triggers_compositor_b:
                b_type_id = record.type_id
        return MpbV39CompositorTriggers(a_type_id=a_type_id, b_type_id=b_type_id)

    def summary_dict(self) -> dict[str, object]:
        return {
            "draw_record_count": len(self.draw_records),
            "atlas_record_count": len(self.atlas_records),
            "generic_binding_count": len(self.generic_bindings),
            "compositor_triggers": vars(self.compositor_triggers),
            "draw_blend_modes": [
                {
                    "draw_order": record.file_index,
                    "type_id": record.type_id,
                    "blend_mode": record.blend_mode,
                }
                for record in self.draw_records
            ],
            "generic_bindings": [
                {
                    "draw_order": binding.draw_order,
                    "type_id": binding.type_id,
                    "outer_index": binding.outer_index,
                    "atlas_record_id": binding.atlas_record_id,
                    "source_rectangle": vars(binding.source_rectangle),
                    "atlas_rectangle": vars(binding.atlas_rectangle),
                    "runtime_atlas_rectangle": vars(
                        binding.runtime_atlas_rectangle
                    ),
                    "fallback_rectangle": vars(binding.fallback_rectangle),
                    "initially_enabled": binding.initially_enabled,
                    "initial_opacity": binding.initial_opacity,
                }
                for binding in self.generic_bindings
            ],
        }


def _section(layout: MpbV39Layout, name: str) -> StructuralSection:
    matches = [section for section in layout.sections if section.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one section named {name!r}, found {len(matches)}")
    return matches[0]


def _rectangle(data: bytes, offset: int) -> MpbV39UvRectangle:
    return MpbV39UvRectangle(*struct.unpack_from("<ffff", data, offset))


def parse_v39_atlas_semantics(
    data: bytes | bytearray | memoryview,
) -> MpbV39AtlasSemantics:
    """Parse renderer-proven v39 draw records and atlas rectangle records.

    The generic type-to-atlas rule mirrors the call setup at game RVA
    0x13200..0x13236. Types 0 through 2 use separate renderer paths and are
    deliberately excluded from ``generic_bindings``.
    """

    stable_data = data if isinstance(data, bytes) else bytes(data)
    layout = walk_v39_to_tail(stable_data)
    reader = CheckedBinary(stable_data)
    draw_section = _section(layout, "group A records")
    atlas_section = _section(layout, "group B records")
    draw_count = layout.observed_counts["group_a_count"]
    atlas_count = layout.observed_counts["group_b_count"]
    reader.array_span(
        "v39 draw records",
        draw_section.offset,
        draw_count,
        V39_DRAW_SOURCE_STRIDE,
        max_count=4096,
    )
    reader.array_span(
        "v39 atlas records",
        atlas_section.offset,
        atlas_count,
        V39_ATLAS_SOURCE_STRIDE,
        max_count=4096,
    )

    draw_records: list[MpbV39DrawRecord] = []
    seen_types: set[int] = set()
    for file_index in range(draw_count):
        offset = draw_section.offset + file_index * V39_DRAW_SOURCE_STRIDE
        fields = struct.unpack_from("<IIIIII", stable_data, offset)
        opacity = struct.unpack_from("<f", stable_data, offset + 0x20)[0]
        blend_mode = struct.unpack_from("<I", stable_data, offset + 0x30)[0]
        compositor_trigger_a_word, compositor_trigger_b_word = struct.unpack_from(
            "<II", stable_data, offset + 0x24
        )
        record = MpbV39DrawRecord(
            file_index=file_index,
            source_offset=offset,
            enabled_word=fields[0],
            type_id=fields[1],
            key_08=fields[2],
            outer_index=fields[3],
            key_10=fields[4],
            key_14=fields[5],
            initial_opacity=opacity,
            blend_mode=blend_mode,
            compositor_trigger_a_word=compositor_trigger_a_word,
            compositor_trigger_b_word=compositor_trigger_b_word,
        )
        if record.type_id in seen_types:
            raise BinaryBoundsError(f"duplicate v39 draw type {record.type_id}")
        if not math.isfinite(record.initial_opacity):
            raise BinaryBoundsError(
                f"v39 draw type {record.type_id} has non-finite opacity"
            )
        seen_types.add(record.type_id)
        draw_records.append(record)

    atlas_records: list[MpbV39AtlasRecord] = []
    atlas_by_id: dict[int, MpbV39AtlasRecord] = {}
    for file_index in range(atlas_count):
        offset = atlas_section.offset + file_index * V39_ATLAS_SOURCE_STRIDE
        record_id = struct.unpack_from("<H", stable_data, offset)[0]
        if record_id in atlas_by_id:
            raise BinaryBoundsError(f"duplicate v39 atlas record id {record_id}")
        record = MpbV39AtlasRecord(
            file_index=file_index,
            source_offset=offset,
            record_id=record_id,
            source_rectangle=_rectangle(stable_data, offset + 0x04),
            atlas_rectangle=_rectangle(stable_data, offset + 0x14),
            fallback_rectangle=_rectangle(stable_data, offset + 0x24),
        )
        record.validate()
        atlas_by_id[record_id] = record
        atlas_records.append(record)

    generic_bindings: list[MpbV39AtlasBinding] = []
    for draw_record in draw_records:
        if draw_record.type_id <= 2:
            continue
        atlas_record_id = draw_record.type_id + V39_GENERIC_ATLAS_ID_BIAS
        atlas_record = atlas_by_id.get(atlas_record_id)
        if atlas_record is None:
            raise BinaryBoundsError(
                f"v39 draw type {draw_record.type_id} is missing atlas record "
                f"{atlas_record_id}"
            )
        generic_bindings.append(
            MpbV39AtlasBinding(
                draw_order=draw_record.file_index,
                type_id=draw_record.type_id,
                outer_index=draw_record.outer_index,
                atlas_record_id=atlas_record_id,
                source_rectangle=atlas_record.source_rectangle,
                atlas_rectangle=atlas_record.atlas_rectangle,
                fallback_rectangle=atlas_record.fallback_rectangle,
                initially_enabled=draw_record.enabled,
                initial_opacity=draw_record.initial_opacity,
            )
        )

    return MpbV39AtlasSemantics(
        draw_records=tuple(draw_records),
        atlas_records=tuple(atlas_records),
        generic_bindings=tuple(generic_bindings),
    )


__all__ = [
    "MpbV39AtlasBinding",
    "MpbV39CompositorTriggers",
    "MpbV39AtlasRecord",
    "MpbV39AtlasSemantics",
    "MpbV39DrawRecord",
    "V39_DEFAULT_COMPOSITOR_TRIGGER_TYPE",
    "V39_ABSENT_PACKAGED_ATLAS_COORDINATE",
    "parse_v39_atlas_semantics",
    "resolve_v39_runtime_atlas_rectangle",
]
