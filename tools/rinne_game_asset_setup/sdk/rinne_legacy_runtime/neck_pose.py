from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass, replace

from .checked_binary import BinaryBoundsError, CheckedBinary
from .neutral_type0 import (
    RINNE_LEGACY_DEPTH_BASE,
    RINNE_LEGACY_POSITION_BIAS,
    RINNE_LEGACY_POSITION_SCALE,
    RINNE_LEGACY_PROJECTION_XY_SCALE,
    RINNE_LEGACY_PROJECTION_Z_SCALE,
)
from .textured_mesh import TexturedDepthMesh

V39_NECK_TARGET_LEFT_UPPER_OFFSET = 0xAAC
V39_NECK_TARGET_UPPER_Y_OFFSET = 0xAB0
V39_NECK_TARGET_RIGHT_UPPER_OFFSET = 0xAB4
V39_NECK_TARGET_LEFT_LOWER_OFFSET = 0xACC
V39_NECK_TARGET_LOWER_Y_OFFSET = 0xAD0
V39_NECK_TARGET_RIGHT_LOWER_OFFSET = 0xAD4
V39_NECK_DEPTH_UPPER_OFFSET = 0x14D0
V39_NECK_DEPTH_LOWER_OFFSET = 0x14D8

RINNE_LEGACY_NECK_SOURCE_LEFT_UPPER = 0.4
RINNE_LEGACY_NECK_SOURCE_RIGHT_UPPER = 0.6
RINNE_LEGACY_NECK_SOURCE_LEFT_LOWER = 0.42
RINNE_LEGACY_NECK_SOURCE_RIGHT_LOWER = 0.58
RINNE_LEGACY_NECK_SOURCE_UPPER_Y = 0.53
RINNE_LEGACY_NECK_SOURCE_LOWER_Y = 0.31
RINNE_LEGACY_NECK_SOURCE_CENTER_X = 0.5
RINNE_LEGACY_NECK_SOURCE_PIVOT_Y = 1.0
RINNE_LEGACY_NECK_DEPTH_DIVISOR = 0.19
RINNE_LEGACY_NECK_DEPTH_SCALE = 0.7


def _f32(value: float) -> float:
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise BinaryBoundsError("neck-pose value is outside float32 range") from exc
    if not math.isfinite(result):
        raise BinaryBoundsError("neck-pose value must be finite")
    return result


def _add(left: float, right: float) -> float:
    return _f32(_f32(left) + _f32(right))


def _sub(left: float, right: float) -> float:
    return _f32(_f32(left) - _f32(right))


def _mul(left: float, right: float) -> float:
    return _f32(_f32(left) * _f32(right))


def _div(numerator: float, denominator: float, *, label: str) -> float:
    denominator = _f32(denominator)
    if denominator == 0.0:
        raise BinaryBoundsError(f"neck-pose {label} denominator is zero")
    return _f32(_f32(numerator) / denominator)


def _triple(name: str, values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise BinaryBoundsError(f"{name} must contain exactly three values")
    try:
        return tuple(_f32(float(value)) for value in values)  # type: ignore[return-value]
    except (TypeError, ValueError) as exc:
        raise BinaryBoundsError(f"{name} values must be finite float32") from exc


@dataclass(frozen=True)
class RinneLegacyNeckProfile:
    """Resource-derived rotation pivot used by game RVA 0x1DC20."""

    pivot_xyz: tuple[float, float, float]

    def validate(self) -> None:
        _triple("neck pivot", self.pivot_xyz)


@dataclass(frozen=True)
class RinneLegacyNeckMatrix:
    """Row-vector 4x4 model-view matrix emitted by game RVA 0x1DC20."""

    rows: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]

    def transform_model_vertex(
        self, vertex_xyz: Sequence[float]
    ) -> tuple[float, float, float]:
        x, y, z = _triple("neck model vertex", vertex_xyz)
        row0, row1, row2, row3 = self.rows
        return (
            _add(
                _add(_mul(x, row0[0]), _mul(y, row1[0])),
                _add(_mul(z, row2[0]), row3[0]),
            ),
            _add(
                _add(_mul(x, row0[1]), _mul(y, row1[1])),
                _add(_mul(z, row2[1]), row3[1]),
            ),
            _add(
                _add(_mul(x, row0[2]), _mul(y, row1[2])),
                _add(_mul(z, row2[2]), row3[2]),
            ),
        )


def _read_f32(reader: CheckedBinary, name: str, offset: int) -> float:
    return _f32(struct.unpack("<f", reader.bytes_at(name, offset, 4))[0])


def parse_v39_rinne_neck_profile(
    data: bytes | bytearray | memoryview,
) -> RinneLegacyNeckProfile:
    """Reproduce the resource-to-pivot calculations at RVAs 0x1DB6/0x12E10."""

    reader = CheckedBinary(data if isinstance(data, bytes) else bytes(data))
    target_left_upper = _read_f32(
        reader, "neck target left upper", V39_NECK_TARGET_LEFT_UPPER_OFFSET
    )
    target_upper_y = _read_f32(
        reader, "neck target upper Y", V39_NECK_TARGET_UPPER_Y_OFFSET
    )
    target_right_upper = _read_f32(
        reader, "neck target right upper", V39_NECK_TARGET_RIGHT_UPPER_OFFSET
    )
    target_left_lower = _read_f32(
        reader, "neck target left lower", V39_NECK_TARGET_LEFT_LOWER_OFFSET
    )
    target_lower_y = _read_f32(
        reader, "neck target lower Y", V39_NECK_TARGET_LOWER_Y_OFFSET
    )
    target_right_lower = _read_f32(
        reader, "neck target right lower", V39_NECK_TARGET_RIGHT_LOWER_OFFSET
    )

    vertical_ratio = _div(
        _sub(
            RINNE_LEGACY_NECK_SOURCE_PIVOT_Y,
            RINNE_LEGACY_NECK_SOURCE_LOWER_Y,
        ),
        _sub(
            RINNE_LEGACY_NECK_SOURCE_UPPER_Y,
            RINNE_LEGACY_NECK_SOURCE_LOWER_Y,
        ),
        label="vertical calibration",
    )
    pivot_y = _add(
        target_lower_y,
        _mul(vertical_ratio, _sub(target_upper_y, target_lower_y)),
    )

    if vertical_ratio >= 0.0:
        source_left = _add(
            RINNE_LEGACY_NECK_SOURCE_LEFT_LOWER,
            _mul(
                vertical_ratio,
                _sub(
                    RINNE_LEGACY_NECK_SOURCE_LEFT_UPPER,
                    RINNE_LEGACY_NECK_SOURCE_LEFT_LOWER,
                ),
            ),
        )
        source_right = _add(
            RINNE_LEGACY_NECK_SOURCE_RIGHT_LOWER,
            _mul(
                vertical_ratio,
                _sub(
                    RINNE_LEGACY_NECK_SOURCE_RIGHT_UPPER,
                    RINNE_LEGACY_NECK_SOURCE_RIGHT_LOWER,
                ),
            ),
        )
        target_left = _add(
            target_left_lower,
            _mul(vertical_ratio, _sub(target_left_upper, target_left_lower)),
        )
        target_right = _add(
            target_right_lower,
            _mul(vertical_ratio, _sub(target_right_upper, target_right_lower)),
        )
    else:
        source_left = _f32(RINNE_LEGACY_NECK_SOURCE_LEFT_LOWER)
        source_right = _f32(RINNE_LEGACY_NECK_SOURCE_RIGHT_LOWER)
        target_left = target_left_lower
        target_right = target_right_lower

    horizontal_ratio = _div(
        _sub(RINNE_LEGACY_NECK_SOURCE_CENTER_X, source_left),
        _sub(source_right, source_left),
        label="horizontal calibration",
    )
    pivot_x = _add(
        target_left,
        _mul(horizontal_ratio, _sub(target_right, target_left)),
    )
    pivot_z = _f32(
        -_mul(
            _div(
                _sub(
                    _read_f32(reader, "neck depth upper", V39_NECK_DEPTH_UPPER_OFFSET),
                    _read_f32(reader, "neck depth lower", V39_NECK_DEPTH_LOWER_OFFSET),
                ),
                RINNE_LEGACY_NECK_DEPTH_DIVISOR,
                label="depth calibration",
            ),
            RINNE_LEGACY_NECK_DEPTH_SCALE,
        )
    )
    profile = RinneLegacyNeckProfile(
        pivot_xyz=(
            _sub(
                _mul(pivot_x, RINNE_LEGACY_POSITION_SCALE), RINNE_LEGACY_POSITION_BIAS
            ),
            _sub(
                _mul(pivot_y, RINNE_LEGACY_POSITION_SCALE), RINNE_LEGACY_POSITION_BIAS
            ),
            pivot_z,
        )
    )
    profile.validate()
    return profile


def _sin_degrees(value: float) -> float:
    value = _f32(value)
    if value == 0.0:
        return 0.0
    radians = _div(_mul(value, math.pi), 180.0, label="rotation radians")
    return _f32(math.sin(float(radians)))


def _cos_degrees(value: float) -> float:
    radians = _div(_mul(_f32(value), math.pi), 180.0, label="rotation radians")
    return _f32(math.cos(float(radians)))


def build_rinne_legacy_neck_matrix(
    profile: RinneLegacyNeckProfile,
    *,
    rotation_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    translation_xyz: Sequence[float] = (0.0, 0.0, 0.0),
) -> RinneLegacyNeckMatrix:
    """Build the exact row-vector X-then-Y-then-Z matrix used by the game."""

    if not isinstance(profile, RinneLegacyNeckProfile):
        raise BinaryBoundsError("neck matrix requires a Rinne neck profile")
    profile.validate()
    rx, ry, rz = _triple("neck rotation", rotation_xyz)
    tx, ty, tz = _triple("neck translation", translation_xyz)
    px, py, pz = profile.pivot_xyz

    sx, sy, sz = _sin_degrees(rx), _sin_degrees(ry), _sin_degrees(rz)
    cx, cy, cz = _cos_degrees(rx), _cos_degrees(ry), _cos_degrees(rz)

    m00 = _mul(cz, cy)
    m01 = _f32(-_mul(cy, sz))
    m02 = sy
    m10 = _add(_mul(_mul(sy, sx), cz), _mul(cx, sz))
    m11 = _sub(_mul(cx, cz), _mul(_mul(sy, sx), sz))
    m12 = _f32(-_mul(cy, sx))
    m20 = _sub(_mul(sz, sx), _mul(_mul(cx, sy), cz))
    m21 = _add(_mul(_mul(cx, sy), sz), _mul(cz, sx))
    m22 = _mul(cy, cx)

    translated_z = _sub(tz, RINNE_LEGACY_DEPTH_BASE)
    out_tx = _sub(
        _add(px, tx),
        _add(_add(_mul(px, m00), _mul(py, m10)), _mul(pz, m20)),
    )
    out_ty = _sub(
        _add(py, ty),
        _add(_add(_mul(px, m01), _mul(py, m11)), _mul(pz, m21)),
    )
    out_tz = _sub(
        _add(pz, translated_z),
        _add(_add(_mul(px, m02), _mul(py, m12)), _mul(pz, m22)),
    )
    return RinneLegacyNeckMatrix(
        rows=(
            (m00, m01, m02, 0.0),
            (m10, m11, m12, 0.0),
            (m20, m21, m22, 0.0),
            (out_tx, out_ty, out_tz, 1.0),
        )
    )


def transform_rinne_legacy_projected_positions(
    positions_xyz: Sequence[Sequence[float]],
    profile: RinneLegacyNeckProfile,
    *,
    rotation_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    translation_xyz: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[tuple[float, float, float], ...]:
    """Apply the neck matrix to already projected CPU-rasterizer vertices."""

    rotation = _triple("neck rotation", rotation_xyz)
    translation = _triple("neck translation", translation_xyz)
    stable = tuple(_triple("neck projected vertex", item) for item in positions_xyz)
    if rotation == (0.0, 0.0, 0.0) and translation == (0.0, 0.0, 0.0):
        return stable

    matrix = build_rinne_legacy_neck_matrix(
        profile, rotation_xyz=rotation, translation_xyz=translation
    )
    output: list[tuple[float, float, float]] = []
    for projected_x, projected_y, projected_z in stable:
        model_vertex = (
            _sub(
                _mul(projected_x, RINNE_LEGACY_POSITION_SCALE),
                RINNE_LEGACY_POSITION_BIAS,
            ),
            _sub(
                _mul(projected_y, RINNE_LEGACY_POSITION_SCALE),
                RINNE_LEGACY_POSITION_BIAS,
            ),
            _add(
                _div(
                    projected_z,
                    RINNE_LEGACY_PROJECTION_Z_SCALE,
                    label="projected depth",
                ),
                RINNE_LEGACY_DEPTH_BASE,
            ),
        )
        x, y, z = matrix.transform_model_vertex(model_vertex)
        output.append(
            (
                _mul(_add(_mul(x, RINNE_LEGACY_PROJECTION_XY_SCALE), 1.0), 0.5),
                _mul(_add(_mul(y, RINNE_LEGACY_PROJECTION_XY_SCALE), 1.0), 0.5),
                _mul(z, RINNE_LEGACY_PROJECTION_Z_SCALE),
            )
        )
    return tuple(output)


def transform_rinne_legacy_meshes(
    meshes: Sequence[TexturedDepthMesh],
    profile: RinneLegacyNeckProfile,
    *,
    rotation_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    translation_xyz: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[TexturedDepthMesh, ...]:
    """Apply one global legacy neck pose to every submitted draw mesh."""

    rotation = _triple("neck rotation", rotation_xyz)
    translation = _triple("neck translation", translation_xyz)
    stable = tuple(meshes)
    if rotation == (0.0, 0.0, 0.0) and translation == (0.0, 0.0, 0.0):
        return stable
    return tuple(
        replace(
            mesh,
            positions_xyz=transform_rinne_legacy_projected_positions(
                mesh.positions_xyz,
                profile,
                rotation_xyz=rotation,
                translation_xyz=translation,
            ),
        )
        for mesh in stable
    )


__all__ = [
    "RINNE_LEGACY_NECK_DEPTH_DIVISOR",
    "RINNE_LEGACY_NECK_DEPTH_SCALE",
    "V39_NECK_DEPTH_LOWER_OFFSET",
    "V39_NECK_DEPTH_UPPER_OFFSET",
    "V39_NECK_TARGET_LEFT_LOWER_OFFSET",
    "V39_NECK_TARGET_LEFT_UPPER_OFFSET",
    "V39_NECK_TARGET_LOWER_Y_OFFSET",
    "V39_NECK_TARGET_RIGHT_LOWER_OFFSET",
    "V39_NECK_TARGET_RIGHT_UPPER_OFFSET",
    "V39_NECK_TARGET_UPPER_Y_OFFSET",
    "RinneLegacyNeckMatrix",
    "RinneLegacyNeckProfile",
    "build_rinne_legacy_neck_matrix",
    "parse_v39_rinne_neck_profile",
    "transform_rinne_legacy_meshes",
    "transform_rinne_legacy_projected_positions",
]
