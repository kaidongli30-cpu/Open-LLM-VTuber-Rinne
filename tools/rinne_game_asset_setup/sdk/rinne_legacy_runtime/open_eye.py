from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .checked_binary import BinaryBoundsError, CheckedBinary
from .fixed_triplet import (
    MpbV39FixedTripletMeshView,
    parse_v39_fixed_triplet_meshes,
)
from .mpb_v39_atlas import MpbV39AtlasRecord, MpbV39AtlasSemantics
from .mpb_v39_renderer import MpbV39RendererViews, MpbV39UvRectangle
from .neutral_type0 import project_rinne_legacy_vertex
from .pupil_deformation import (
    RinneLegacyPupilProfile,
    apply_rinne_legacy_pupil_deformation_xy,
)
from .special_morph import apply_v39_special_base_morph_xy


V39_RIGHT_EYE_CENTER_OFFSET = 0x14D0
V39_LEFT_EYE_CENTER_OFFSET = 0x14D8
RINNE_OPEN_EYE_X_LOWER_RADIUS = 0.125
RINNE_OPEN_EYE_Y_LOWER_RADIUS = 0.0625
RINNE_OPEN_EYE_PARAMETER_SCALE = 4.0
RINNE_OPEN_EYE_SIDE_TO_FIXED_RECORD = (2, 1)
RINNE_OPEN_EYE_SIDE_TO_ATLAS_RECORD = (14, 15)
RINNE_OPEN_EYE_POSITION_TYPE_ID = 1
RINNE_OPEN_EYE_DEPTH_TYPE_ID = 0


def _f32(value: float) -> float:
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise BinaryBoundsError("open-eye value is outside float32 range") from exc
    if not math.isfinite(result):
        raise BinaryBoundsError("open-eye value must be finite")
    return result


def _sub(left: float, right: float) -> float:
    return _f32(_f32(left) - _f32(right))


def _mul(left: float, right: float) -> float:
    return _f32(_f32(left) * _f32(right))


def _div(numerator: float, denominator: float) -> float:
    denominator = _f32(denominator)
    if denominator == 0.0:
        raise BinaryBoundsError("open-eye UV source axis has zero length")
    return _f32(_f32(numerator) / denominator)


def _add(left: float, right: float) -> float:
    return _f32(_f32(left) + _f32(right))


@dataclass(frozen=True)
class RinneOpenEyeLayout:
    centers_xy: tuple[tuple[float, float], tuple[float, float]]
    fixed_meshes: tuple[MpbV39FixedTripletMeshView, ...]


@dataclass(frozen=True)
class RinneNeutralOpenEyeMesh:
    side: int
    fixed_record_index: int
    atlas_record_id: int
    positions_xyz: tuple[tuple[float, float, float], ...]
    game_uvs: tuple[tuple[float, float], ...]
    triangle_indices: tuple[int, ...]
    opacity: float
    texture_filter: Literal["linear"]


def parse_v39_rinne_open_eye_layout(
    data: bytes | bytearray | memoryview,
    *,
    surface_vertex_count: int,
) -> RinneOpenEyeLayout:
    """Read v39 eye-window centers and the compact helper meshes.

    The four fixed-block floats are copied to runtime fields +0x1C4C through
    +0x1C58 by the game's v39 loader. RVA 0x13FC0 then uses them to build the
    two side-specific UV streams.
    """

    stable_data = data if isinstance(data, bytes) else bytes(data)
    reader = CheckedBinary(stable_data)

    def center(name: str, offset: int) -> tuple[float, float]:
        values = struct.unpack("<ff", reader.bytes_at(name, offset, 8))
        if not all(math.isfinite(value) for value in values):
            raise BinaryBoundsError(f"{name}: center contains a non-finite float")
        return values

    fixed_meshes = parse_v39_fixed_triplet_meshes(
        stable_data, surface_vertex_count=surface_vertex_count
    )
    if len(fixed_meshes) != 3:
        raise BinaryBoundsError("Rinne open-eye layout requires three fixed meshes")
    return RinneOpenEyeLayout(
        centers_xy=(
            center("right eye center", V39_RIGHT_EYE_CENTER_OFFSET),
            center("left eye center", V39_LEFT_EYE_CENTER_OFFSET),
        ),
        fixed_meshes=fixed_meshes,
    )


def _record_by_id(
    records: Sequence[MpbV39AtlasRecord], record_id: int
) -> MpbV39AtlasRecord:
    matches = tuple(record for record in records if record.record_id == record_id)
    if len(matches) != 1:
        raise BinaryBoundsError(
            f"open-eye atlas record {record_id} must occur exactly once"
        )
    matches[0].validate()
    return matches[0]


def _depths_for_type(renderer: MpbV39RendererViews, type_id: int):
    category = renderer.type_to_depth_category.get(type_id)
    matches = tuple(
        view for group_id, view in renderer.depth_groups if group_id == category
    )
    if category is None or len(matches) != 1:
        raise BinaryBoundsError(f"open-eye depth for type {type_id} is missing")
    return matches[0]


def _map_open_eye_axis(
    value: float,
    source_first: float,
    source_second: float,
    target_first: float,
    target_second: float,
) -> float:
    value = _f32(value)
    source_first = _f32(source_first)
    source_second = _f32(source_second)
    target_first = _f32(target_first)
    target_second = _f32(target_second)
    if source_first > value:
        return target_first
    if value > source_second:
        return target_second
    return _add(
        target_first,
        _mul(
            _div(_sub(value, source_first), _sub(source_second, source_first)),
            _sub(target_second, target_first),
        ),
    )


def _map_open_eye_uv(
    pair: tuple[float, float],
    *,
    source: MpbV39UvRectangle,
    target: MpbV39UvRectangle,
) -> tuple[float, float]:
    source.validate(source=True)
    target.validate()
    return (
        _map_open_eye_axis(
            pair[0],
            source.first_x,
            source.second_x,
            target.first_x,
            target.second_x,
        ),
        _map_open_eye_axis(
            pair[1],
            source.first_y,
            source.second_y,
            target.first_y,
            target.second_y,
        ),
    )


def build_rinne_neutral_open_eye_meshes(
    renderer: MpbV39RendererViews,
    atlas: MpbV39AtlasSemantics,
    layout: RinneOpenEyeLayout,
    *,
    expression_weights: Sequence[float] | None = None,
    pupil_profile: RinneLegacyPupilProfile | None = None,
    right_pupil_position: Sequence[float] = (0.0, 0.0),
    left_pupil_position: Sequence[float] = (0.0, 0.0),
) -> tuple[RinneNeutralOpenEyeMesh, RinneNeutralOpenEyeMesh]:
    """Rebuild the two open-eye draws submitted by RVA 0x18B20.

    v39 uses type-1 current XY positions, type-0 depths, fixed-triplet records
    2/1 and group-B atlas records 14/15 for sides 0/1 respectively.
    """

    renderer.validate()
    if len(layout.centers_xy) != 2 or len(layout.fixed_meshes) != 3:
        raise BinaryBoundsError("open-eye layout is incomplete")

    position_outer = renderer.type_to_outer.get(RINNE_OPEN_EYE_POSITION_TYPE_ID)
    depth_outer = renderer.type_to_outer.get(RINNE_OPEN_EYE_DEPTH_TYPE_ID)
    if position_outer is None or depth_outer is None or position_outer != depth_outer:
        raise BinaryBoundsError("open-eye type-1/type-0 surface mapping does not match")
    if position_outer < 0 or position_outer >= len(renderer.mesh_records):
        raise BinaryBoundsError("open-eye face surface mesh is missing")
    surface = renderer.mesh_records[position_outer]
    surface_vertex_count = len(surface.coordinate_source_xy)
    for fixed_mesh in layout.fixed_meshes:
        fixed_mesh.validate(surface_vertex_count=surface_vertex_count)

    raw_expression_weights = (
        (0.0,) * renderer.primary_morph_count
        if expression_weights is None
        else expression_weights
    )
    current_positions = apply_v39_special_base_morph_xy(
        renderer, RINNE_OPEN_EYE_POSITION_TYPE_ID, raw_expression_weights
    )
    if pupil_profile is not None:
        current_positions = apply_rinne_legacy_pupil_deformation_xy(
            tuple(surface.coordinate_source_xy.values()),
            current_positions,
            pupil_profile,
            right_position_xy=right_pupil_position,
            left_position_xy=left_pupil_position,
        )
    elif tuple(right_pupil_position) != (0.0, 0.0) or tuple(
        left_pupil_position
    ) != (0.0, 0.0):
        raise BinaryBoundsError("open-eye pupil controls require a pupil profile")
    depths = _depths_for_type(renderer, RINNE_OPEN_EYE_DEPTH_TYPE_ID)
    if len(current_positions) != surface_vertex_count or len(depths) != (
        surface_vertex_count
    ):
        raise BinaryBoundsError("open-eye face position/depth counts do not match")

    type_one_draws = tuple(
        draw for draw in atlas.draw_records if draw.type_id == 1
    )
    if len(type_one_draws) != 1:
        raise BinaryBoundsError("open-eye type-1 draw must occur exactly once")
    type_one_draw = type_one_draws[0]
    if not type_one_draw.enabled:
        raise BinaryBoundsError("open-eye type-1 draw is not initially enabled")
    opacity = type_one_draw.initial_opacity
    if not math.isfinite(opacity) or opacity < 0.0 or opacity > 1.0:
        raise BinaryBoundsError("open-eye opacity must be within 0..1")

    result: list[RinneNeutralOpenEyeMesh] = []
    for side in (0, 1):
        fixed_record_index = RINNE_OPEN_EYE_SIDE_TO_FIXED_RECORD[side]
        atlas_record_id = RINNE_OPEN_EYE_SIDE_TO_ATLAS_RECORD[side]
        fixed_mesh = layout.fixed_meshes[fixed_record_index]
        atlas_record = _record_by_id(atlas.atlas_records, atlas_record_id)
        center_x, center_y = layout.centers_xy[side]
        origin_x = _sub(center_x, RINNE_OPEN_EYE_X_LOWER_RADIUS)
        origin_y = _sub(center_y, RINNE_OPEN_EYE_Y_LOWER_RADIUS)
        positions: list[tuple[float, float, float]] = []
        game_uvs: list[tuple[float, float]] = []
        for surface_vertex_id in fixed_mesh.surface_vertex_ids.values():
            x, y = current_positions[surface_vertex_id]
            positions.append(
                project_rinne_legacy_vertex(
                    x,
                    y,
                    depths.at(surface_vertex_id),
                    depth_offset=0.0,
                )
            )
            source_x, source_y = surface.coordinate_source_xy.at(surface_vertex_id)
            window_pair = (
                _mul(_sub(source_x, origin_x), RINNE_OPEN_EYE_PARAMETER_SCALE),
                _mul(_sub(source_y, origin_y), RINNE_OPEN_EYE_PARAMETER_SCALE),
            )
            game_uvs.append(
                _map_open_eye_uv(
                    window_pair,
                    source=atlas_record.source_rectangle,
                    target=atlas_record.runtime_atlas_rectangle,
                )
            )
        result.append(
            RinneNeutralOpenEyeMesh(
                side=side,
                fixed_record_index=fixed_record_index,
                atlas_record_id=atlas_record_id,
                positions_xyz=tuple(positions),
                game_uvs=tuple(game_uvs),
                triangle_indices=tuple(fixed_mesh.triangle_indices.values()),
                opacity=opacity,
                texture_filter="linear",
            )
        )
    return result[0], result[1]


__all__ = [
    "RINNE_OPEN_EYE_DEPTH_TYPE_ID",
    "RINNE_OPEN_EYE_PARAMETER_SCALE",
    "RINNE_OPEN_EYE_POSITION_TYPE_ID",
    "RINNE_OPEN_EYE_SIDE_TO_ATLAS_RECORD",
    "RINNE_OPEN_EYE_SIDE_TO_FIXED_RECORD",
    "RINNE_OPEN_EYE_X_LOWER_RADIUS",
    "RINNE_OPEN_EYE_Y_LOWER_RADIUS",
    "RinneNeutralOpenEyeMesh",
    "RinneOpenEyeLayout",
    "V39_LEFT_EYE_CENTER_OFFSET",
    "V39_RIGHT_EYE_CENTER_OFFSET",
    "build_rinne_neutral_open_eye_meshes",
    "parse_v39_rinne_open_eye_layout",
]
