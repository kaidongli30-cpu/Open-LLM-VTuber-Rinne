from __future__ import annotations

import math
import struct
from collections.abc import Sequence

from .checked_binary import BinaryBoundsError
from .mpb_v39_renderer import Float32PairView, MpbV39MeshRecordView


V39_EXPRESSION_ACTIVE_THRESHOLD = 0.01


def _f32(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError as exc:
        raise BinaryBoundsError(
            "expression morph float32 arithmetic overflow"
        ) from exc


def prepare_v39_expression_weights(
    raw_values: Sequence[float], ease_flags: Sequence[int]
) -> tuple[float, ...]:
    """Apply the primary-record flag-controlled easing from RVA 0x15C90."""

    if len(raw_values) != len(ease_flags):
        raise BinaryBoundsError(
            "expression value count does not match ease-flag count"
        )
    prepared: list[float] = []
    for index, (raw_value, flag) in enumerate(
        zip(raw_values, ease_flags, strict=True)
    ):
        value = float(raw_value)
        if not math.isfinite(value):
            raise BinaryBoundsError(
                f"expression value {index} must be finite"
            )
        value = _f32(value)
        if flag == 1:
            if value <= 0.0:
                value = 0.0
            elif value >= 1.0:
                value = 1.0
            else:
                value = _f32((1.0 - math.cos(math.pi * value)) * 0.5)
        prepared.append(value)
    return tuple(prepared)


def apply_v39_expression_morph_xy(
    mesh: MpbV39MeshRecordView,
    morph_deltas: Float32PairView,
    expression_weights: Sequence[float],
) -> tuple[tuple[float, float], ...]:
    """Apply the v39 expression delta loop from game RVA 0x16592..0x167A2.

    Deltas are vertex-major and expression-minor. Only the mesh's listed morph
    vertex IDs are updated, and weights below the game's float32 0.01 constant
    are skipped. Arithmetic is rounded through float32 after each SSE-style
    multiply and add.
    """

    mesh.validate()
    morph_deltas.require_finite()
    raw_weights = tuple(float(weight) for weight in expression_weights)
    if any(not math.isfinite(weight) for weight in raw_weights):
        raise BinaryBoundsError("expression morph weights must be finite")
    weights = tuple(_f32(weight) for weight in raw_weights)
    if not weights:
        raise BinaryBoundsError("expression morph requires at least one weight")
    vertex_count = len(mesh.coordinate_source_xy)
    expected_delta_count = vertex_count * len(weights)
    if len(morph_deltas) != expected_delta_count:
        raise BinaryBoundsError(
            f"expression morph delta count {len(morph_deltas)} does not equal "
            f"vertex count {vertex_count} * weight count {len(weights)}"
        )

    threshold = _f32(V39_EXPRESSION_ACTIVE_THRESHOLD)
    active_weights = tuple(
        (index, weight)
        for index, weight in enumerate(weights)
        if weight >= threshold
    )
    base_positions = tuple(mesh.coordinate_source_xy.values())
    output = list(base_positions)
    for vertex_id in mesh.morph_vertex_ids.values():
        x, y = base_positions[vertex_id]
        row_start = vertex_id * len(weights)
        for expression_index, weight in active_weights:
            delta_x, delta_y = morph_deltas.at(row_start + expression_index)
            x = _f32(x + _f32(weight * delta_x))
            y = _f32(y + _f32(weight * delta_y))
        output[vertex_id] = (x, y)
    return tuple(output)


__all__ = [
    "V39_EXPRESSION_ACTIVE_THRESHOLD",
    "apply_v39_expression_morph_xy",
    "prepare_v39_expression_weights",
]
