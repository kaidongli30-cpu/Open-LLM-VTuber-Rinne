from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Sequence

from .checked_binary import BinaryBoundsError
from .mpb_v39_atlas import MpbV39AtlasRecord
from .mpb_v39_renderer import MpbV39MeshRecordView
from .special_eye import (
    RINNE_SPECIAL_EYE_COLUMNS,
    RINNE_SPECIAL_EYE_POINT_COUNT,
    RinneSpecialEyePreprojection,
    RinneSpecialEyeProfile,
    project_rinne_special_eye_surface,
)


RINNE_TYPE2_EYE_STRIP_ATLAS_RECORD_ID = 2
RINNE_TYPE2_EYE_STRIP_ROWS = 2
RINNE_TYPE2_EYE_STRIP_VERTEX_COUNT = (
    RINNE_SPECIAL_EYE_COLUMNS * RINNE_TYPE2_EYE_STRIP_ROWS
)


def _f32(value: float) -> float:
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise BinaryBoundsError("type-2 eye-strip value is outside float32 range") from exc
    if not math.isfinite(result):
        raise BinaryBoundsError("type-2 eye-strip value must be finite")
    return result


def _add(left: float, right: float) -> float:
    return _f32(_f32(left) + _f32(right))


def _sub(left: float, right: float) -> float:
    return _f32(_f32(left) - _f32(right))


def _mul(left: float, right: float) -> float:
    return _f32(_f32(left) * _f32(right))


def _div(left: float, right: float) -> float:
    if right == 0.0:
        raise BinaryBoundsError("type-2 eye-strip division by zero")
    return _f32(_f32(left) / _f32(right))


def _unit(value: float, *, name: str) -> float:
    value = _f32(value)
    if value < 0.0 or value > 1.0:
        raise BinaryBoundsError(f"{name} must be within 0..1")
    return value


def _ease_unit(value: float) -> float:
    """Reproduce RVA 0x1AF0's clamped sine ease."""

    value = _f32(value)
    if value >= 1.0:
        return 1.0
    if value <= 0.0:
        return 0.0
    degrees = _sub(_mul(value, 180.0), 90.0)
    if degrees == 0.0:
        sine = 0.0
    else:
        radians = _div(_mul(degrees, 3.1415927410125732), 180.0)
        sine = _f32(math.sin(float(radians)))
    return _add(_mul(sine, 0.5), 0.5)


def _topology() -> tuple[int, ...]:
    indices: list[int] = []
    for column in range(RINNE_SPECIAL_EYE_COLUMNS - 1):
        top_left = column
        bottom_left = RINNE_SPECIAL_EYE_COLUMNS + column
        top_right = top_left + 1
        bottom_right = bottom_left + 1
        indices.extend(
            (
                top_left,
                bottom_left,
                top_right,
                top_right,
                bottom_left,
                bottom_right,
            )
        )
    return tuple(indices)


RINNE_TYPE2_EYE_STRIP_TRIANGLE_INDICES = _topology()


@dataclass(frozen=True)
class RinneType2EyeStripMesh:
    side: int
    deformation_u: float
    opacity: float
    positions_xyz: tuple[tuple[float, float, float], ...]
    game_uvs: tuple[tuple[float, float], ...]
    triangle_indices: tuple[int, ...]


def build_rinne_type2_eye_strip_preprojection(
    profile: RinneSpecialEyeProfile,
    *,
    side: int,
    deformation_a: float,
    deformation_c: float,
) -> RinneSpecialEyePreprojection:
    """Build the two-row type-2 eye strip before face-surface projection."""

    if side not in (0, 1):
        raise BinaryBoundsError("type-2 eye-strip side must be 0 or 1")
    points = profile.points_for_side(side)
    if len(points) != RINNE_SPECIAL_EYE_POINT_COUNT:
        raise BinaryBoundsError("type-2 eye strip requires 22 profile points")
    deformation_a = _unit(deformation_a, name="type-2 deformation A")
    deformation_c = _unit(deformation_c, name="type-2 deformation C")
    deformation_u = _mul(
        _add(
            _mul(_sub(1.0, deformation_c), deformation_a),
            deformation_c,
        ),
        0.949999988079071,
    )
    upper = points[:RINNE_SPECIAL_EYE_COLUMNS]
    lower = points[RINNE_SPECIAL_EYE_COLUMNS:]
    center = RINNE_SPECIAL_EYE_COLUMNS // 2
    center_y_span = _sub(lower[center][1], upper[center][1])
    first_offset = _mul(center_y_span, 0.20000000298023224)
    second_offset = _mul(center_y_span, 0.4000000059604645)

    first_row: list[tuple[float, float]] = []
    second_row: list[tuple[float, float]] = []
    for upper_point, lower_point in zip(upper, lower, strict=True):
        x = _add(
            upper_point[0],
            _mul(_sub(lower_point[0], upper_point[0]), deformation_u),
        )
        first_y = _add(upper_point[1], first_offset)
        first_y = _add(
            first_y,
            _mul(
                _sub(_sub(lower_point[1], upper_point[1]), first_offset),
                deformation_u,
            ),
        )
        second_y = _add(
            _add(
                upper_point[1],
                _mul(_sub(lower_point[1], upper_point[1]), deformation_u),
            ),
            second_offset,
        )
        first_row.append((x, first_y))
        second_row.append((x, second_y))
    return RinneSpecialEyePreprojection(
        side=side,
        deformation_u=deformation_u,
        positions_xy=tuple((*first_row, *second_row)),
    )


def _record_two(records: Sequence[MpbV39AtlasRecord]) -> MpbV39AtlasRecord:
    matches = tuple(
        record
        for record in records
        if record.record_id == RINNE_TYPE2_EYE_STRIP_ATLAS_RECORD_ID
    )
    if len(matches) != 1:
        raise BinaryBoundsError("type-2 eye strip requires one atlas record 2")
    record = matches[0]
    record.validate()
    target = record.atlas_rectangle
    if (
        target.first_x,
        target.first_y,
        target.second_x,
        target.second_y,
    ) == (-1.0,) * 4:
        raise BinaryBoundsError("type-2 eye-strip atlas target is absent")
    return record


def build_rinne_type2_eye_strip_mesh(
    profile: RinneSpecialEyeProfile,
    atlas_records: Sequence[MpbV39AtlasRecord],
    surface_mesh: MpbV39MeshRecordView,
    surface_positions_xyz: Sequence[Sequence[float]],
    *,
    side: int,
    deformation_a: float,
    deformation_c: float,
    intensity: float,
    runtime_opacity: float,
) -> RinneType2EyeStripMesh:
    """Reproduce the Rinne branch of RVA 0x190D0 for one eye side."""

    intensity = _unit(intensity, name="type-2 intensity")
    runtime_opacity = _unit(runtime_opacity, name="type-2 runtime opacity")
    preprojection = build_rinne_type2_eye_strip_preprojection(
        profile,
        side=side,
        deformation_a=deformation_a,
        deformation_c=deformation_c,
    )
    deformation_u = preprojection.deformation_u
    projection = project_rinne_special_eye_surface(
        preprojection, surface_mesh, surface_positions_xyz
    )

    target = _record_two(atlas_records).runtime_atlas_rectangle
    upper_v = _add(
        target.first_y,
        _mul(_sub(target.second_y, target.first_y), 0.75),
    )
    lower_v = _add(
        target.first_y,
        _mul(
            _sub(target.second_y, target.first_y),
            0.5199999809265137,
        ),
    )
    row_u = tuple(
        _add(
            target.first_x,
            _mul(
                _div(float(column), float(RINNE_SPECIAL_EYE_COLUMNS)),
                _sub(target.second_x, target.first_x),
            ),
        )
        for column in range(RINNE_SPECIAL_EYE_COLUMNS)
    )
    game_uvs = tuple((u, upper_v) for u in row_u) + tuple(
        (u, lower_v) for u in row_u
    )
    opacity = _mul(
        _mul(
            _ease_unit(deformation_u),
            _mul(intensity, 0.800000011920929),
        ),
        runtime_opacity,
    )
    return RinneType2EyeStripMesh(
        side=side,
        deformation_u=deformation_u,
        opacity=opacity,
        positions_xyz=projection.normalized_positions_xyz,
        game_uvs=game_uvs,
        triangle_indices=RINNE_TYPE2_EYE_STRIP_TRIANGLE_INDICES,
    )


__all__ = [
    "RINNE_TYPE2_EYE_STRIP_ATLAS_RECORD_ID",
    "RINNE_TYPE2_EYE_STRIP_ROWS",
    "RINNE_TYPE2_EYE_STRIP_TRIANGLE_INDICES",
    "RINNE_TYPE2_EYE_STRIP_VERTEX_COUNT",
    "RinneType2EyeStripMesh",
    "build_rinne_type2_eye_strip_preprojection",
    "build_rinne_type2_eye_strip_mesh",
]
