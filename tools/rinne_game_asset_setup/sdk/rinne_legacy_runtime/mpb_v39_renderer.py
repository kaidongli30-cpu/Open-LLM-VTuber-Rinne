from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Iterator

from .checked_binary import BinaryBoundsError, CheckedBinary
from .mpb_v39 import MpbV39Layout, StructuralSection, walk_v39_to_tail
from .mpb_v39_tail import TailBlock, walk_v39_handed_tail


RENDERER_DIMENSION_SAFETY_LIMIT = 16_384
V39_DRAW_DEPTH_UPPER_FILE_OFFSET = 0x14D0
V39_DRAW_DEPTH_LOWER_FILE_OFFSET = 0x14D8
V39_DRAW_DEPTH_RANGE = 0.1899999976158142
V39_DRAW_DEPTH_FACTOR = 0.699999988079071
V39_DRAW_DEPTH_DIVISOR = 255.0
V39_DRAW_DEPTH_MULTIPLIER = 0.0625


def _stable_bytes(data: bytes | bytearray | memoryview) -> bytes:
    return data if isinstance(data, bytes) else bytes(data)


def _f32(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError as exc:
        raise BinaryBoundsError("renderer float32 arithmetic overflow") from exc


def compute_v39_draw_depth_offset(draw_order: int, depth_scale: float) -> float:
    """Mirror RVA 0xF817..0xF839 for one ordinary draw record."""

    if draw_order < 0:
        raise BinaryBoundsError("draw order must not be negative")
    if not math.isfinite(depth_scale):
        raise BinaryBoundsError("draw depth scale must be finite")
    product = _f32(_f32(float(draw_order)) * _f32(depth_scale))
    quotient = _f32(product / V39_DRAW_DEPTH_DIVISOR)
    return _f32(quotient * V39_DRAW_DEPTH_MULTIPLIER)


@dataclass(frozen=True)
class Float32ScalarView:
    name: str
    offset: int
    count: int
    _data: memoryview = field(repr=False, compare=False)
    _finite_checked: bool = field(
        default=False, init=False, repr=False, compare=False
    )

    @classmethod
    def create(
        cls,
        data: bytes | bytearray | memoryview,
        *,
        name: str,
        offset: int,
        count: int,
    ) -> "Float32ScalarView":
        stable_data = _stable_bytes(data)
        reader = CheckedBinary(stable_data)
        reader.array_span(name, offset, count, 4, max_count=1_000_000_000)
        return cls(name, offset, count, memoryview(stable_data).cast("B"))

    def __len__(self) -> int:
        return self.count

    def at(self, index: int) -> float:
        if index < 0 or index >= self.count:
            raise IndexError(f"{self.name}: float index {index} out of range")
        return struct.unpack_from("<f", self._data, self.offset + index * 4)[0]

    def values(self) -> Iterator[float]:
        for index in range(self.count):
            yield self.at(index)

    def require_finite(self) -> None:
        if self._finite_checked:
            return
        for index, value in enumerate(self.values()):
            if not math.isfinite(value):
                raise BinaryBoundsError(
                    f"{self.name}: non-finite float at index {index}"
                )
        # ``create`` owns an immutable bytes snapshot, so this validation remains
        # true for the lifetime of the view.  The private cache prevents every
        # animation frame from rescanning multi-megabyte MPB arrays.
        object.__setattr__(self, "_finite_checked", True)


@dataclass(frozen=True)
class Float32PairView:
    name: str
    offset: int
    count: int
    _data: memoryview = field(repr=False, compare=False)
    _finite_checked: bool = field(
        default=False, init=False, repr=False, compare=False
    )

    @classmethod
    def create(
        cls,
        data: bytes | bytearray | memoryview,
        *,
        name: str,
        offset: int,
        count: int,
    ) -> "Float32PairView":
        stable_data = _stable_bytes(data)
        reader = CheckedBinary(stable_data)
        reader.array_span(name, offset, count, 8, max_count=1_000_000_000)
        return cls(name, offset, count, memoryview(stable_data).cast("B"))

    def __len__(self) -> int:
        return self.count

    def at(self, index: int) -> tuple[float, float]:
        if index < 0 or index >= self.count:
            raise IndexError(f"{self.name}: pair index {index} out of range")
        return struct.unpack_from("<ff", self._data, self.offset + index * 8)

    def values(self) -> Iterator[tuple[float, float]]:
        for index in range(self.count):
            yield self.at(index)

    def require_finite(self) -> None:
        if self._finite_checked:
            return
        for index, pair in enumerate(self.values()):
            if not all(math.isfinite(value) for value in pair):
                raise BinaryBoundsError(
                    f"{self.name}: non-finite float pair at index {index}"
                )
        object.__setattr__(self, "_finite_checked", True)


@dataclass(frozen=True)
class UInt16View:
    name: str
    offset: int
    count: int
    _data: memoryview = field(repr=False, compare=False)
    _checked_upper_bound: int | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @classmethod
    def create(
        cls,
        data: bytes | bytearray | memoryview,
        *,
        name: str,
        offset: int,
        count: int,
    ) -> "UInt16View":
        stable_data = _stable_bytes(data)
        reader = CheckedBinary(stable_data)
        reader.array_span(name, offset, count, 2, max_count=1_000_000_000)
        return cls(name, offset, count, memoryview(stable_data).cast("B"))

    def __len__(self) -> int:
        return self.count

    def at(self, index: int) -> int:
        if index < 0 or index >= self.count:
            raise IndexError(f"{self.name}: uint16 index {index} out of range")
        return struct.unpack_from("<H", self._data, self.offset + index * 2)[0]

    def values(self) -> Iterator[int]:
        for index in range(self.count):
            yield self.at(index)

    def require_below(self, upper_bound: int) -> None:
        if upper_bound < 0:
            raise ValueError("upper bound must not be negative")
        if (
            self._checked_upper_bound is not None
            and upper_bound >= self._checked_upper_bound
        ):
            return
        for index, value in enumerate(self.values()):
            if value >= upper_bound:
                raise BinaryBoundsError(
                    f"{self.name}: value {value} at index {index} exceeds "
                    f"vertex count {upper_bound}"
                )
        object.__setattr__(self, "_checked_upper_bound", upper_bound)


@dataclass(frozen=True)
class MpbV39MeshRecordView:
    outer_index: int
    coordinate_source_xy: Float32PairView
    morph_vertex_ids: UInt16View
    triangle_indices: UInt16View
    grid_cell_columns: int | None = None
    grid_cell_rows: int | None = None
    _validated: bool = field(default=False, init=False, repr=False, compare=False)

    def validate(self) -> None:
        if self._validated:
            return
        self.coordinate_source_xy.require_finite()
        self.morph_vertex_ids.require_below(len(self.coordinate_source_xy))
        self.triangle_indices.require_below(len(self.coordinate_source_xy))
        if len(self.triangle_indices) % 3:
            raise BinaryBoundsError(
                f"mesh record {self.outer_index}: triangle index count "
                f"{len(self.triangle_indices)} is not divisible by 3"
            )
        dimensions = (self.grid_cell_columns, self.grid_cell_rows)
        if (dimensions[0] is None) != (dimensions[1] is None):
            raise BinaryBoundsError(
                f"mesh record {self.outer_index}: both grid dimensions are required"
            )
        if dimensions[0] is not None and dimensions[1] is not None:
            columns, rows = dimensions
            if (
                columns <= 0
                or rows <= 0
                or columns > RENDERER_DIMENSION_SAFETY_LIMIT
                or rows > RENDERER_DIMENSION_SAFETY_LIMIT
            ):
                raise BinaryBoundsError(
                    f"mesh record {self.outer_index}: grid dimensions must be "
                    "within the renderer safety limit"
                )
            grid_vertex_count = (columns + 1) * (rows + 1)
            if grid_vertex_count > len(self.coordinate_source_xy):
                raise BinaryBoundsError(
                    f"mesh record {self.outer_index}: grid requires "
                    f"{grid_vertex_count} coordinates but only "
                    f"{len(self.coordinate_source_xy)} are available"
                )
        object.__setattr__(self, "_validated", True)


@dataclass(frozen=True)
class MpbV39UvRectangle:
    first_x: float
    first_y: float
    second_x: float
    second_y: float

    def validate(self, *, source: bool = False) -> None:
        values = (self.first_x, self.first_y, self.second_x, self.second_y)
        if not all(math.isfinite(value) for value in values):
            raise BinaryBoundsError("UV rectangle contains a non-finite float")
        if source and (
            self.first_x == self.second_x or self.first_y == self.second_y
        ):
            raise BinaryBoundsError("source UV rectangle has a zero-length axis")


def map_v39_uv_pair(
    pair: tuple[float, float],
    *,
    source: MpbV39UvRectangle,
    target: MpbV39UvRectangle,
    clamp: bool,
) -> tuple[float, float]:
    """Mirror the two-axis mapping at game RVA 0x1a940..0x1aad3."""

    source.validate(source=True)
    target.validate()
    u, v = pair
    if not math.isfinite(u) or not math.isfinite(v):
        raise BinaryBoundsError("UV source pair contains a non-finite float")

    def map_axis(
        value: float,
        source_first: float,
        source_second: float,
        target_first: float,
        target_second: float,
    ) -> float:
        if clamp and source_first > value:
            return target_first
        if clamp and value > source_second:
            return target_second
        return target_first + (
            (value - source_first)
            / (source_second - source_first)
            * (target_second - target_first)
        )

    return (
        map_axis(u, source.first_x, source.second_x, target.first_x, target.second_x),
        map_axis(v, source.first_y, source.second_y, target.first_y, target.second_y),
    )


def map_v39_uv_view(
    view: Float32PairView,
    *,
    source: MpbV39UvRectangle,
    target: MpbV39UvRectangle,
    clamp: bool,
) -> tuple[tuple[float, float], ...]:
    view.require_finite()
    return tuple(
        map_v39_uv_pair(pair, source=source, target=target, clamp=clamp)
        for pair in view.values()
    )


@dataclass(frozen=True)
class MpbV39RendererViews:
    mask_width: int
    mask_height: int
    primary_morph_count: int
    mesh_records: tuple[MpbV39MeshRecordView, ...]
    morph_delta_categories: tuple[Float32PairView, ...]
    blend_weight_categories: tuple[Float32ScalarView, ...]
    depth_groups: tuple[tuple[int, Float32ScalarView], ...]
    type_to_outer: dict[int, int]
    type_to_morph_category: dict[int, int]
    type_to_blend_category: dict[int, int]
    type_to_depth_category: dict[int, int]
    expression_ease_flags: tuple[int, ...] = ()
    draw_depth_scale: float | None = None

    def validate(self) -> None:
        if (
            self.mask_width <= 0
            or self.mask_height <= 0
            or self.mask_width > RENDERER_DIMENSION_SAFETY_LIMIT
            or self.mask_height > RENDERER_DIMENSION_SAFETY_LIMIT
        ):
            raise BinaryBoundsError(
                "renderer mask dimensions must be within 1.."
                f"{RENDERER_DIMENSION_SAFETY_LIMIT}"
            )
        for expected_outer_index, record in enumerate(self.mesh_records):
            if record.outer_index != expected_outer_index:
                raise BinaryBoundsError(
                    "renderer mesh records must be ordered by outer index: "
                    f"expected {expected_outer_index}, got {record.outer_index}"
                )
            record.validate()
        for view in self.morph_delta_categories:
            view.require_finite()
        for view in self.blend_weight_categories:
            view.require_finite()
        for _group_id, view in self.depth_groups:
            view.require_finite()
        if self.expression_ease_flags and len(self.expression_ease_flags) != (
            self.primary_morph_count
        ):
            raise BinaryBoundsError(
                "expression ease-flag count does not match primary morph count"
            )
        if self.draw_depth_scale is not None and not math.isfinite(
            self.draw_depth_scale
        ):
            raise BinaryBoundsError("renderer draw depth scale must be finite")

        self._validate_category_lengths(
            "morph delta",
            self.type_to_morph_category,
            self.morph_delta_categories,
            multiplier=self.primary_morph_count,
        )
        self._validate_category_lengths(
            "blend weight",
            self.type_to_blend_category,
            self.blend_weight_categories,
        )
        depth_group_ids = tuple(group_id for group_id, _view in self.depth_groups)
        if len(set(depth_group_ids)) != len(depth_group_ids):
            raise BinaryBoundsError("renderer depth group IDs must be unique")
        expected_depth_group_ids = set(self.type_to_depth_category.values())
        actual_depth_group_ids = set(depth_group_ids)
        if actual_depth_group_ids != expected_depth_group_ids:
            missing = sorted(expected_depth_group_ids - actual_depth_group_ids)
            extra = sorted(actual_depth_group_ids - expected_depth_group_ids)
            raise BinaryBoundsError(
                "renderer depth groups do not cover the type mapping: "
                f"missing={missing}, extra={extra}"
            )
        for group_id, view in self.depth_groups:
            expected = self._category_vertex_count(
                "depth", self.type_to_depth_category, group_id
            )
            if len(view) != expected:
                raise BinaryBoundsError(
                    f"depth group {group_id}: count {len(view)} does not match "
                    f"mesh vertex count {expected}"
                )

    def _category_vertex_count(
        self, label: str, mapping: dict[int, int], category: int
    ) -> int:
        missing_types = {
            type_id
            for type_id, mapped_category in mapping.items()
            if mapped_category == category and type_id not in self.type_to_outer
        }
        if missing_types:
            raise BinaryBoundsError(
                f"{label} category {category}: missing outer mapping for types "
                f"{sorted(missing_types)}"
            )
        outer_indices = {
            self.type_to_outer[type_id]
            for type_id, mapped_category in mapping.items()
            if mapped_category == category
        }
        if len(outer_indices) != 1:
            raise BinaryBoundsError(
                f"{label} category {category}: expected one outer mesh, found "
                f"{sorted(outer_indices)}"
            )
        outer_index = next(iter(outer_indices))
        if outer_index < 0 or outer_index >= len(self.mesh_records):
            raise BinaryBoundsError(
                f"{label} category {category}: outer mesh {outer_index} is missing"
            )
        return len(self.mesh_records[outer_index].coordinate_source_xy)

    def _validate_category_lengths(
        self,
        label: str,
        mapping: dict[int, int],
        views: tuple[Float32PairView, ...] | tuple[Float32ScalarView, ...],
        *,
        multiplier: int = 1,
    ) -> None:
        categories = set(mapping.values())
        if categories != set(range(len(views))):
            raise BinaryBoundsError(
                f"{label} categories are not contiguous: {sorted(categories)}"
            )
        for category, view in enumerate(views):
            expected = self._category_vertex_count(label, mapping, category)
            expected *= multiplier
            if len(view) != expected:
                raise BinaryBoundsError(
                    f"{label} category {category}: count {len(view)} does not "
                    f"match expected {expected}"
                )

    def summary_dict(self) -> dict[str, object]:
        return {
            "mask_dimensions": [self.mask_width, self.mask_height],
            "primary_morph_count": self.primary_morph_count,
            "expression_ease_flag_count": len(self.expression_ease_flags),
            "mesh_records": [
                {
                    "outer_index": record.outer_index,
                    "vertex_count": len(record.coordinate_source_xy),
                    "morph_vertex_id_count": len(record.morph_vertex_ids),
                    "triangle_index_count": len(record.triangle_indices),
                    "triangle_count": len(record.triangle_indices) // 3,
                    "grid_cell_columns": record.grid_cell_columns,
                    "grid_cell_rows": record.grid_cell_rows,
                }
                for record in self.mesh_records
            ],
            "morph_delta_category_count": len(self.morph_delta_categories),
            "blend_weight_category_count": len(self.blend_weight_categories),
            "depth_group_ids": [group_id for group_id, _view in self.depth_groups],
            "draw_depth_scale": self.draw_depth_scale,
        }


def _section(layout: MpbV39Layout, name: str) -> StructuralSection:
    matches = [section for section in layout.sections if section.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one section named {name!r}, found {len(matches)}")
    return matches[0]


def _optional_payload_offset(
    layout: MpbV39Layout, payload_name: str, count_slot_name: str
) -> int:
    matches = [section for section in layout.sections if section.name == payload_name]
    if len(matches) == 1:
        return matches[0].offset
    if len(matches) > 1:
        raise ValueError(f"expected at most one section named {payload_name!r}")
    return _section(layout, count_slot_name).end


def build_v39_renderer_views(
    data: bytes | bytearray | memoryview,
    *,
    validate: bool = True,
) -> MpbV39RendererViews:
    """Expose only renderer types proven by the game's read instructions.

    The source mesh records contain float32 XY parameter coordinates which the
    renderer maps into a separate 8-byte-per-vertex UV stream, plus uint16
    vertex/index tables. The handed tail contributes float32 XY morph deltas,
    float32 blend weights and float32 depth values. Layer semantics are not
    inferred here.
    """

    stable_data = _stable_bytes(data)
    structural = walk_v39_to_tail(stable_data)
    tail = walk_v39_handed_tail(stable_data)
    reader = CheckedBinary(stable_data)
    outer_count = structural.observed_counts["variable_outer_record_count"]

    mesh_records: list[MpbV39MeshRecordView] = []
    for outer_index in range(outer_count):
        prefix_section = _section(
            structural, f"variable record {outer_index} prefix"
        )
        vertex_count = structural.observed_counts[f"variable_record_{outer_index}_c8"]
        morph_id_count = structural.observed_counts[
            f"variable_record_{outer_index}_c2a"
        ]
        triangle_index_count = structural.observed_counts[
            f"variable_record_{outer_index}_c2b"
        ]
        base_section = _section(
            structural, f"variable record {outer_index} payload 8"
        )
        morph_id_offset = _optional_payload_offset(
            structural,
            f"variable record {outer_index} payload 2A",
            f"variable record {outer_index} payload 2A count",
        )
        triangle_offset = _optional_payload_offset(
            structural,
            f"variable record {outer_index} payload 2B",
            f"variable record {outer_index} payload 2B count",
        )
        mesh_records.append(
            MpbV39MeshRecordView(
                outer_index=outer_index,
                coordinate_source_xy=Float32PairView.create(
                    stable_data,
                    name=f"mesh record {outer_index} coordinate-source XY",
                    offset=base_section.offset,
                    count=vertex_count,
                ),
                morph_vertex_ids=UInt16View.create(
                    stable_data,
                    name=f"mesh record {outer_index} morph vertex IDs",
                    offset=morph_id_offset,
                    count=morph_id_count,
                ),
                triangle_indices=UInt16View.create(
                    stable_data,
                    name=f"mesh record {outer_index} triangle indices",
                    offset=triangle_offset,
                    count=triangle_index_count,
                ),
                grid_cell_columns=struct.unpack_from(
                    "<I", stable_data, prefix_section.offset + 0x20
                )[0],
                grid_cell_rows=struct.unpack_from(
                    "<I", stable_data, prefix_section.offset + 0x30
                )[0],
            )
        )

    morph_blocks = sorted(
        (block for block in tail.blocks if block.name.startswith("25bc category ")),
        key=lambda block: int(block.name.split()[2]),
    )
    blend_blocks = sorted(
        (block for block in tail.blocks if block.name.startswith("26d8 category ")),
        key=lambda block: int(block.name.split()[2]),
    )
    trailing_depth_blocks = [
        block
        for block in tail.blocks
        if block.name.startswith("trailing group ")
        and block.name.endswith(" four-byte block")
    ]

    # RVA 0x5E68/0x5E79 aliases the type-0/type-2 depth accessors as the
    # dedicated +0x2644/+0x2640 pointers. RVA 0x7260 then copies the two
    # leading tail blocks into those arrays before copying the ordinary
    # trailing groups. Preserve that last-write-wins order when a category is
    # shared by a special and trailing record.
    effective_depth_blocks: dict[int, TailBlock] = {}
    for type_id in (0, 2):
        group_id = tail.map_2650_by_type.get(type_id)
        if group_id is None:
            continue
        matches = [
            block
            for block in tail.blocks
            if block.name == f"type {type_id} four-byte block"
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one type {type_id} depth block, found {len(matches)}"
            )
        effective_depth_blocks[group_id] = matches[0]
    for group_id, block in zip(
        tail.trailing_group_ids, trailing_depth_blocks, strict=True
    ):
        effective_depth_blocks[group_id] = block

    type_to_outer = {record.type_id: record.outer_index for record in tail.type_records}
    reader.span("v39 draw-depth upper", V39_DRAW_DEPTH_UPPER_FILE_OFFSET, 4)
    reader.span("v39 draw-depth lower", V39_DRAW_DEPTH_LOWER_FILE_OFFSET, 4)
    depth_upper = struct.unpack_from(
        "<f", stable_data, V39_DRAW_DEPTH_UPPER_FILE_OFFSET
    )[0]
    depth_lower = struct.unpack_from(
        "<f", stable_data, V39_DRAW_DEPTH_LOWER_FILE_OFFSET
    )[0]
    draw_depth_scale = _f32(
        _f32(_f32(depth_upper - depth_lower) / V39_DRAW_DEPTH_RANGE)
        * V39_DRAW_DEPTH_FACTOR
    )
    renderer = MpbV39RendererViews(
        mask_width=tail.first_word,
        mask_height=tail.second_word,
        primary_morph_count=structural.primary.primary_record_count,
        mesh_records=tuple(mesh_records),
        morph_delta_categories=tuple(
            Float32PairView.create(
                stable_data,
                name=f"morph delta category {category}",
                offset=block.offset,
                count=block.element_count,
            )
            for category, block in enumerate(morph_blocks)
        ),
        blend_weight_categories=tuple(
            Float32ScalarView.create(
                stable_data,
                name=f"blend weight category {category}",
                offset=block.offset,
                count=block.element_count,
            )
            for category, block in enumerate(blend_blocks)
        ),
        depth_groups=tuple(
            (
                group_id,
                Float32ScalarView.create(
                    stable_data,
                    name=f"depth group {group_id}",
                    offset=block.offset,
                    count=block.element_count,
                ),
            )
            for group_id, block in sorted(effective_depth_blocks.items())
        ),
        type_to_outer=type_to_outer,
        type_to_morph_category=dict(tail.map_25bc_by_type),
        type_to_blend_category=dict(tail.map_26d8_by_type),
        type_to_depth_category=dict(tail.map_2650_by_type),
        expression_ease_flags=tuple(
            struct.unpack_from(
                "<I",
                stable_data,
                structural.primary.primary_records.offset
                + index * structural.primary.primary_source_stride
                + 4,
            )[0]
            for index in range(structural.primary.primary_record_count)
        ),
        draw_depth_scale=draw_depth_scale,
    )
    if validate:
        renderer.validate()
    return renderer


__all__ = [
    "Float32PairView",
    "Float32ScalarView",
    "MpbV39MeshRecordView",
    "MpbV39RendererViews",
    "MpbV39UvRectangle",
    "UInt16View",
    "V39_DRAW_DEPTH_FACTOR",
    "V39_DRAW_DEPTH_DIVISOR",
    "V39_DRAW_DEPTH_LOWER_FILE_OFFSET",
    "V39_DRAW_DEPTH_RANGE",
    "V39_DRAW_DEPTH_MULTIPLIER",
    "V39_DRAW_DEPTH_UPPER_FILE_OFFSET",
    "build_v39_renderer_views",
    "compute_v39_draw_depth_offset",
    "map_v39_uv_pair",
    "map_v39_uv_view",
]
