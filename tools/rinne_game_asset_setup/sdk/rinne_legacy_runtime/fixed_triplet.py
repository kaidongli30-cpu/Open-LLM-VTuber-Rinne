from __future__ import annotations

import struct
from dataclasses import dataclass

from .checked_binary import BinaryBoundsError
from .mpb_v39 import MpbV39Layout, StructuralSection, walk_v39_to_tail
from .mpb_v39_renderer import UInt16View


V39_FIXED_TRIPLET_RECORD_COUNT = 3


@dataclass(frozen=True)
class MpbV39FixedTripletMeshView:
    """One indexed helper mesh loaded by game RVA 0x57B0.

    ``surface_vertex_ids`` selects vertices from the main face surface. The
    second uint16 payload is an index buffer over that compact selected-vertex
    array; it is not another list of main-surface vertex IDs.
    """

    record_index: int
    slot_0: int
    slot_1: int
    surface_vertex_ids: UInt16View
    triangle_indices: UInt16View
    auxiliary_slots: tuple[int, int, int, int, int]

    def validate(self, *, surface_vertex_count: int) -> None:
        if self.record_index < 0:
            raise BinaryBoundsError("fixed-triplet record index must not be negative")
        if surface_vertex_count <= 0:
            raise BinaryBoundsError("fixed-triplet surface vertex count must be positive")
        self.surface_vertex_ids.require_below(surface_vertex_count)
        self.triangle_indices.require_below(len(self.surface_vertex_ids))
        if len(self.triangle_indices) % 3:
            raise BinaryBoundsError(
                f"fixed-triplet record {self.record_index}: triangle index count "
                f"{len(self.triangle_indices)} is not divisible by 3"
            )


def _section(layout: MpbV39Layout, name: str) -> StructuralSection:
    matches = tuple(section for section in layout.sections if section.name == name)
    if len(matches) != 1:
        raise ValueError(f"expected one section named {name!r}, found {len(matches)}")
    return matches[0]


def _optional_payload_offset(
    layout: MpbV39Layout,
    *,
    payload_name: str,
    following_section_name: str,
) -> int:
    matches = tuple(
        section for section in layout.sections if section.name == payload_name
    )
    if len(matches) == 1:
        return matches[0].offset
    if len(matches) > 1:
        raise ValueError(f"expected at most one section named {payload_name!r}")
    return _section(layout, following_section_name).offset


def parse_v39_fixed_triplet_meshes(
    data: bytes | bytearray | memoryview,
    *,
    surface_vertex_count: int | None = None,
) -> tuple[MpbV39FixedTripletMeshView, ...]:
    """Expose the three compact indexed meshes from a v39 MPB.

    The section walker proves that these records exist only when the variable
    mesh group is present. Their eye-mask use is assigned by the higher-level
    Rinne compatibility path, not by this structural parser.
    """

    stable_data = data if isinstance(data, bytes) else bytes(data)
    layout = walk_v39_to_tail(stable_data)
    if layout.observed_counts["variable_outer_record_count"] == 0:
        return ()

    records: list[MpbV39FixedTripletMeshView] = []
    for record_index in range(V39_FIXED_TRIPLET_RECORD_COUNT):
        prefix = f"fixed triplet record {record_index}"
        slots = _section(layout, f"{prefix} three slots")
        slot_0 = struct.unpack_from("<I", stable_data, slots.offset)[0]
        slot_1 = struct.unpack_from("<I", stable_data, slots.offset + 0x10)[0]
        surface_id_count = layout.observed_counts[
            f"fixed_triplet_{record_index}_d2a"
        ]
        triangle_index_count = layout.observed_counts[
            f"fixed_triplet_{record_index}_d2b"
        ]
        surface_id_offset = _optional_payload_offset(
            layout,
            payload_name=f"{prefix} payload 2A",
            following_section_name=f"{prefix} payload 2B count",
        )
        triangle_index_offset = _optional_payload_offset(
            layout,
            payload_name=f"{prefix} payload 2B",
            following_section_name=f"{prefix} five slots",
        )
        auxiliary = _section(layout, f"{prefix} five slots")
        record = MpbV39FixedTripletMeshView(
            record_index=record_index,
            slot_0=slot_0,
            slot_1=slot_1,
            surface_vertex_ids=UInt16View.create(
                stable_data,
                name=f"{prefix} surface vertex IDs",
                offset=surface_id_offset,
                count=surface_id_count,
            ),
            triangle_indices=UInt16View.create(
                stable_data,
                name=f"{prefix} triangle indices",
                offset=triangle_index_offset,
                count=triangle_index_count,
            ),
            auxiliary_slots=tuple(
                struct.unpack_from("<I", stable_data, auxiliary.offset + index * 0x10)[0]
                for index in range(5)
            ),
        )
        if surface_vertex_count is not None:
            record.validate(surface_vertex_count=surface_vertex_count)
        records.append(record)
    return tuple(records)


__all__ = [
    "MpbV39FixedTripletMeshView",
    "V39_FIXED_TRIPLET_RECORD_COUNT",
    "parse_v39_fixed_triplet_meshes",
]
