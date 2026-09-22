from __future__ import annotations

import math
import struct
from collections.abc import Callable
from dataclasses import dataclass

from .checked_binary import BinaryBoundsError


TYPE0_GRID_COLUMNS = 9
TYPE0_GRID_ROWS = 5
TYPE0_GRID_VERTEX_COUNT = TYPE0_GRID_COLUMNS * TYPE0_GRID_ROWS
TYPE0_GRID_INDEX_COUNT = 192
TYPE0_GRID_TRIANGLE_COUNT = TYPE0_GRID_INDEX_COUNT // 3


def _f32(value: float) -> float:
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise BinaryBoundsError("type0 grid value is outside float32 range") from exc
    if not math.isfinite(result):
        raise BinaryBoundsError("type0 grid value must be finite")
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
        raise BinaryBoundsError(f"type0 grid {label} denominator is zero")
    return _f32(_f32(numerator) / denominator)


@dataclass(frozen=True)
class LegacyType0GridTopology:
    initial_positions: tuple[tuple[float, float, float], ...]
    uvs: tuple[tuple[float, float], ...]
    triangle_indices: tuple[int, ...]


@dataclass(frozen=True)
class LegacyType0RuntimeInputs:
    f32c4: float
    f32bc: float
    f32d4: float
    f3314: float
    f3304: float
    f32d0: float
    f32c0: float
    f32c8: float
    f32b8: float
    f3310: float
    f3300: float
    f3308: float
    f32f8: float
    f1c84: float
    f1c88: float
    f1c5c: float
    f1c60: float
    f25a4: float
    f3428: float


@dataclass(frozen=True)
class LegacyType0GroupBRect:
    b1: float
    b2: float
    b3: float
    b4: float


@dataclass(frozen=True)
class LegacyType0GridFrame:
    group_b_index: int
    buffer_selector: int
    positions: tuple[tuple[float, float, float], ...]
    uvs: tuple[tuple[float, float], ...]
    triangle_indices: tuple[int, ...]
    draw_color_w: float


def build_legacy_type0_grid_topology() -> LegacyType0GridTopology:
    positions: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    for row in range(TYPE0_GRID_ROWS):
        v = _f32(row / 4.0)
        for column in range(TYPE0_GRID_COLUMNS):
            u = _f32(column / 8.0)
            positions.append(
                (_sub(_mul(2.0, u), 1.0), _sub(1.0, _mul(2.0, v)), 0.0)
            )
            uvs.append((u, v))

    indices: list[int] = []
    for row in range(TYPE0_GRID_ROWS - 1):
        for column in range(TYPE0_GRID_COLUMNS - 1):
            top_left = row * TYPE0_GRID_COLUMNS + column
            bottom_left = (row + 1) * TYPE0_GRID_COLUMNS + column
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
    if len(positions) != TYPE0_GRID_VERTEX_COUNT or len(indices) != (
        TYPE0_GRID_INDEX_COUNT
    ):
        raise AssertionError("internal type0 grid topology mismatch")
    return LegacyType0GridTopology(tuple(positions), tuple(uvs), tuple(indices))


TYPE0_GRID_TOPOLOGY = build_legacy_type0_grid_topology()


def build_legacy_type0_grid_frame(
    runtime: LegacyType0RuntimeInputs,
    rectangle: LegacyType0GroupBRect,
    *,
    group_b_index: int,
    height_sampler: Callable[[float, float], float],
) -> LegacyType0GridFrame:
    """Build one of the two RVA 0x19370 CPU staging grids."""

    if group_b_index not in (3, 4):
        raise BinaryBoundsError("type0 grid requires group-B index 3 or 4")
    values = tuple(vars(runtime).values()) + tuple(vars(rectangle).values())
    if any(not math.isfinite(float(value)) for value in values):
        raise BinaryBoundsError("type0 grid inputs must be finite")

    midpoint = _mul(_add(runtime.f32c4, runtime.f32bc), 0.5)
    t = _div(
        _sub(midpoint, runtime.f32d4),
        _sub(runtime.f32c4, runtime.f32d4),
        label="t",
    )
    p = _add(runtime.f3314, _mul(t, _sub(runtime.f3304, runtime.f3314)))

    if t >= 0.0:
        a = _add(runtime.f32d0, _mul(t, _sub(runtime.f32c0, runtime.f32d0)))
        b = _add(runtime.f32c8, _mul(t, _sub(runtime.f32b8, runtime.f32c8)))
        c0 = _add(runtime.f3310, _mul(t, _sub(runtime.f3300, runtime.f3310)))
        d = _add(runtime.f3308, _mul(t, _sub(runtime.f32f8, runtime.f3308)))
    else:
        a, b, c0, d = (
            _f32(runtime.f32d0),
            _f32(runtime.f32c8),
            _f32(runtime.f3310),
            _f32(runtime.f3308),
        )

    sx = _mul(
        runtime.f1c84,
        _div(_sub(d, c0), _sub(b, a), label="horizontal scale"),
    )
    sy = _mul(
        runtime.f1c88,
        _div(
            _sub(runtime.f3304, runtime.f3314),
            _sub(runtime.f32c4, runtime.f32d4),
            label="vertical scale",
        ),
    )
    base_x = _sub(_mul(4.0, runtime.f1c5c), 2.0)
    base_y = _sub(_mul(4.0, runtime.f1c60), 2.0)
    half_sx = _mul(sx, 0.5)
    half_sy = _mul(sy, 0.5)

    if group_b_index == 3:
        y_offset = _sub(_mul(runtime.f25a4, p), 0.005)
        z_offset = _mul(-0.08, runtime.f3428)
        color_w = 1.0
        buffer_selector = 0x12
    else:
        y_offset = 0.0
        z_offset = _mul(-0.02, runtime.f3428)
        color_w = z_offset
        buffer_selector = 0x11

    positions: list[tuple[float, float, float]] = []
    for row in range(TYPE0_GRID_ROWS):
        row_fraction = _f32(row / 4.0)
        rect_y = _add(
            _sub(1.0, rectangle.b4),
            _mul(_sub(rectangle.b4, rectangle.b2), row_fraction),
        )
        for column in range(TYPE0_GRID_COLUMNS):
            column_fraction = _f32(column / 8.0)
            rect_x = _add(
                rectangle.b1,
                _mul(_sub(rectangle.b3, rectangle.b1), column_fraction),
            )
            x = _add(_sub(_mul(rect_x, sx), half_sx), base_x)
            y = _add(
                _add(_sub(half_sy, _mul(rect_y, sy)), base_y),
                y_offset,
            )
            sample_u = _add(0.5, _mul(0.25, x))
            sample_v = _add(0.5, _mul(0.25, y))
            sampled = float(height_sampler(sample_u, sample_v))
            if not math.isfinite(sampled):
                raise BinaryBoundsError("type0 height sampler returned non-finite")
            z = _add(sampled, z_offset)
            positions.append((x, y, z))

    return LegacyType0GridFrame(
        group_b_index=group_b_index,
        buffer_selector=buffer_selector,
        positions=tuple(positions),
        uvs=TYPE0_GRID_TOPOLOGY.uvs,
        triangle_indices=TYPE0_GRID_TOPOLOGY.triangle_indices,
        draw_color_w=_f32(color_w),
    )


__all__ = [
    "TYPE0_GRID_COLUMNS",
    "TYPE0_GRID_INDEX_COUNT",
    "TYPE0_GRID_ROWS",
    "TYPE0_GRID_TOPOLOGY",
    "TYPE0_GRID_TRIANGLE_COUNT",
    "TYPE0_GRID_VERTEX_COUNT",
    "LegacyType0GridFrame",
    "LegacyType0GridTopology",
    "LegacyType0GroupBRect",
    "LegacyType0RuntimeInputs",
    "build_legacy_type0_grid_frame",
    "build_legacy_type0_grid_topology",
]
