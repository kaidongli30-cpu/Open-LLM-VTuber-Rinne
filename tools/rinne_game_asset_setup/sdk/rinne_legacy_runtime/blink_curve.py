from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from .checked_binary import BinaryBoundsError


LEGACY_BLINK_CURVE_SIZE = 1024
LEGACY_BLINK_TYPE_COUNT = 3


def _f32(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise BinaryBoundsError("blink value is outside float32 range") from exc


LEGACY_BLINK_BASE_WEIGHTS = tuple(
    _f32(value) for value in (1.0, 0.7, 0.49, 0.343)
)
_LEGACY_PI_F32 = _f32(3.1415927410125732)
_LEGACY_DECAY_DIVISOR_F32 = _f32(0.30000001192092896)
_LEGACY_HALF_BLINK_SCALE_F32 = _f32(0.4000000059604645)
_LEGACY_DOUBLE_FIRST_SCALE_F32 = _f32(0.699999988079071)
_LEGACY_DOUBLE_SECOND_SCALE_F32 = _f32(0.800000011920929)


def _ease01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    value = _f32(value)
    degrees = _f32(_f32(value * _f32(180.0)) - _f32(90.0))
    radians = _f32(_f32(degrees * _LEGACY_PI_F32) / _f32(180.0))
    sine = _f32(math.sin(radians))
    return _f32(_f32(sine * _f32(0.5)) + _f32(0.5))


def _decay(index: int) -> float:
    ratio = _f32(_f32(float(index)) / _f32(400.0))
    ratio = _f32(ratio / _LEGACY_DECAY_DIVISOR_F32)
    exponent = _f32(-_f32(ratio * ratio))
    return _f32(math.exp(exponent))


def _curve_value(blink_type: int, index: int) -> float:
    if blink_type == 0:
        if index < 80:
            return _ease01(_f32(_f32(float(index)) / _f32(80.0)))
        if index < 90:
            return 1.0
        if index < 490:
            return _decay(index - 90)
        return 0.0
    if blink_type == 1:
        if index < 100:
            value = _ease01(_f32(_f32(float(index)) / _f32(100.0)))
            return _f32(value * _LEGACY_HALF_BLINK_SCALE_F32)
        if index < 400:
            return _LEGACY_HALF_BLINK_SCALE_F32
        if index < 500:
            eased = _ease01(
                _f32(_f32(float(index - 400)) / _f32(100.0))
            )
            inverse = _f32(_f32(1.0) - eased)
            return _f32(inverse * _LEGACY_HALF_BLINK_SCALE_F32)
        return 0.0
    if blink_type == 2:
        if index < 80:
            eased = _ease01(_f32(_f32(float(index)) / _f32(80.0)))
            return _f32(eased * _LEGACY_DOUBLE_FIRST_SCALE_F32)
        if index < 160:
            eased = _ease01(
                _f32(_f32(float(index - 80)) / _f32(80.0))
            )
            inverse = _f32(_f32(1.0) - eased)
            return _f32(inverse * _LEGACY_DOUBLE_FIRST_SCALE_F32)
        if index < 240:
            eased = _ease01(
                _f32(_f32(float(index - 160)) / _f32(80.0))
            )
            return _f32(eased * _LEGACY_DOUBLE_SECOND_SCALE_F32)
        if index < 250:
            return _LEGACY_DOUBLE_SECOND_SCALE_F32
        if index < 650:
            return _f32(
                _decay(index - 250) * _LEGACY_DOUBLE_SECOND_SCALE_F32
            )
        return 0.0
    raise BinaryBoundsError(f"blink type {blink_type} is outside 0..2")


LEGACY_BLINK_CURVES = tuple(
    tuple(
        _f32(_curve_value(blink_type, index))
        for index in range(LEGACY_BLINK_CURVE_SIZE)
    )
    for blink_type in range(LEGACY_BLINK_TYPE_COUNT)
)


def _normalized_curve_index(blink_type: int, elapsed: int) -> int:
    if not isinstance(elapsed, int) or isinstance(elapsed, bool):
        raise BinaryBoundsError("blink elapsed time must be an integer")
    if blink_type == 1:
        if elapsed > 1000:
            elapsed -= 800
        elif elapsed > 200:
            return 200
    return min(LEGACY_BLINK_CURVE_SIZE - 1, max(0, elapsed))


@dataclass(frozen=True)
class LegacyBlinkGeometrySample:
    deformation: tuple[float, float, float, float]
    base_weights: tuple[float, float, float, float]


def sample_legacy_blink_geometry(
    blink_type: int,
    *,
    current_elapsed: int,
    previous_elapsed: int,
    single_blink_gain: float,
    floor: float,
) -> LegacyBlinkGeometrySample:
    """Sample the four blink geometry inputs from RVA 0xD8A0."""

    if (
        not isinstance(blink_type, int)
        or isinstance(blink_type, bool)
        or blink_type < 0
        or blink_type >= LEGACY_BLINK_TYPE_COUNT
    ):
        raise BinaryBoundsError(f"blink type {blink_type} is outside 0..2")
    if not math.isfinite(single_blink_gain) or not math.isfinite(floor):
        raise BinaryBoundsError("blink gain and floor must be finite")
    current = _normalized_curve_index(blink_type, current_elapsed)
    previous = _normalized_curve_index(blink_type, previous_elapsed)
    delta = current - previous
    gain = _f32(single_blink_gain) if blink_type == 0 else 1.0
    floor_value = _f32(floor)
    values: list[float] = []
    for sample_number in range(1, 5):
        curve_index = previous + math.trunc(sample_number * delta / 4.0)
        curve_value = LEGACY_BLINK_CURVES[blink_type][curve_index]
        values.append(max(floor_value, _f32(curve_value * gain)))
    return LegacyBlinkGeometrySample(
        deformation=tuple(values),  # type: ignore[arg-type]
        base_weights=LEGACY_BLINK_BASE_WEIGHTS,
    )


__all__ = [
    "LEGACY_BLINK_BASE_WEIGHTS",
    "LEGACY_BLINK_CURVES",
    "LEGACY_BLINK_CURVE_SIZE",
    "LEGACY_BLINK_TYPE_COUNT",
    "LegacyBlinkGeometrySample",
    "sample_legacy_blink_geometry",
]
