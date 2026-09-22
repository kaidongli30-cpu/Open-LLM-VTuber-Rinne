from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass

from .checked_binary import BinaryBoundsError, CheckedBinary


V39_RIGHT_PUPIL_CENTER_X_OFFSETS = (0x188, 0x1A0)
V39_RIGHT_PUPIL_CENTER_Y_OFFSETS = (0x19C, 0x194, 0x1AC, 0x1B4)
V39_LEFT_PUPIL_CENTER_X_OFFSETS = (0x2A4, 0x28C)
V39_LEFT_PUPIL_CENTER_Y_OFFSETS = (0x2A0, 0x298, 0x2B0, 0x2B8)
V39_RIGHT_PUPIL_RADIUS_OFFSET = 0x1500
V39_LEFT_PUPIL_RADIUS_OFFSET = 0x1504

RINNE_LEGACY_PUPIL_RADIUS_SCALE = 26.0
RINNE_LEGACY_PUPIL_DISTANCE_SCALE = 10.0
RINNE_LEGACY_PUPIL_INPUT_SCALE = 0.3
RINNE_LEGACY_PUPIL_FALLOFF_SCALE = 0.4


def _f32(value: float) -> float:
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise BinaryBoundsError("pupil-deformation value is outside float32 range") from exc
    if not math.isfinite(result):
        raise BinaryBoundsError("pupil-deformation value must be finite")
    return result


def _add(left: float, right: float) -> float:
    return _f32(_f32(left) + _f32(right))


def _sub(left: float, right: float) -> float:
    return _f32(_f32(left) - _f32(right))


def _mul(left: float, right: float) -> float:
    return _f32(_f32(left) * _f32(right))


def _div(numerator: float, denominator: float) -> float:
    denominator = _f32(denominator)
    if denominator == 0.0:
        raise BinaryBoundsError("pupil-deformation radius must not be zero")
    return _f32(_f32(numerator) / denominator)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise BinaryBoundsError("pupil-deformation mean requires values")
    total = _f32(values[0])
    for value in values[1:]:
        total = _add(total, value)
    return _mul(total, _div(1.0, float(len(values))))


@dataclass(frozen=True)
class RinneLegacyPupilProfile:
    right_center_xy: tuple[float, float]
    left_center_xy: tuple[float, float]
    right_radius: float
    left_radius: float

    def validate(self) -> None:
        values = (
            *self.right_center_xy,
            *self.left_center_xy,
            self.right_radius,
            self.left_radius,
        )
        if not all(math.isfinite(value) for value in values):
            raise BinaryBoundsError("pupil profile values must be finite")
        if len(self.right_center_xy) != 2 or len(self.left_center_xy) != 2:
            raise BinaryBoundsError("pupil profile centers must contain two values")
        if self.right_radius <= 0.0 or self.left_radius <= 0.0:
            raise BinaryBoundsError("pupil profile radii must be positive")


def parse_v39_rinne_pupil_profile(
    data: bytes | bytearray | memoryview,
) -> RinneLegacyPupilProfile:
    """Read the eye-landmark centers and radii consumed by RVAs 0x15D10/0x16410."""

    stable_data = data if isinstance(data, bytes) else bytes(data)
    reader = CheckedBinary(stable_data)

    def values(name: str, offsets: Sequence[int]) -> tuple[float, ...]:
        result: list[float] = []
        for index, offset in enumerate(offsets):
            raw = reader.bytes_at(f"{name} value {index}", offset, 4)
            result.append(struct.unpack("<f", raw)[0])
        return tuple(result)

    right_center = (
        _mean(values("right pupil center X", V39_RIGHT_PUPIL_CENTER_X_OFFSETS)),
        _mean(values("right pupil center Y", V39_RIGHT_PUPIL_CENTER_Y_OFFSETS)),
    )
    left_center = (
        _mean(values("left pupil center X", V39_LEFT_PUPIL_CENTER_X_OFFSETS)),
        _mean(values("left pupil center Y", V39_LEFT_PUPIL_CENTER_Y_OFFSETS)),
    )
    right_radius = _mul(
        struct.unpack(
            "<f",
            reader.bytes_at(
                "right pupil radius", V39_RIGHT_PUPIL_RADIUS_OFFSET, 4
            ),
        )[0],
        RINNE_LEGACY_PUPIL_RADIUS_SCALE,
    )
    left_radius = _mul(
        struct.unpack(
            "<f",
            reader.bytes_at("left pupil radius", V39_LEFT_PUPIL_RADIUS_OFFSET, 4),
        )[0],
        RINNE_LEGACY_PUPIL_RADIUS_SCALE,
    )
    profile = RinneLegacyPupilProfile(
        right_center_xy=right_center,
        left_center_xy=left_center,
        right_radius=right_radius,
        left_radius=left_radius,
    )
    profile.validate()
    return profile


def _sine_ease(value: float) -> float:
    """Mirror RVA 0x1AF0, including the float-to-double sine call."""

    value = _f32(value)
    if value >= 1.0:
        return 1.0
    if value <= 0.0:
        return 0.0
    degrees = _sub(_mul(value, 180.0), 90.0)
    radians = _div(_mul(degrees, math.pi), 180.0)
    sine = _f32(math.sin(float(radians)))
    return _add(_mul(sine, 0.5), 0.5)


def _distance_falloff(
    source_xy: tuple[float, float],
    center_xy: tuple[float, float],
    radius: float,
) -> float:
    delta_y = _div(_sub(source_xy[1], center_xy[1]), radius)
    delta_x = _div(_sub(source_xy[0], center_xy[0]), radius)
    squared = _add(_mul(delta_y, delta_y), _mul(delta_x, delta_x))
    distance = _mul(_f32(math.sqrt(float(squared))), RINNE_LEGACY_PUPIL_DISTANCE_SCALE)
    if distance >= 1.0:
        return 0.0
    eased = _mul(_sine_ease(_sub(1.0, distance)), 0.5)
    circle = _mul(
        _f32(math.sqrt(float(_sub(1.0, _mul(distance, distance))))),
        0.5,
    )
    return _mul(_add(eased, circle), RINNE_LEGACY_PUPIL_FALLOFF_SCALE)


def apply_rinne_legacy_pupil_deformation_xy(
    source_positions_xy: Sequence[tuple[float, float]],
    current_positions_xy: Sequence[tuple[float, float]],
    profile: RinneLegacyPupilProfile,
    *,
    right_position_xy: Sequence[float] = (0.0, 0.0),
    left_position_xy: Sequence[float] = (0.0, 0.0),
) -> tuple[tuple[float, float], ...]:
    """Apply the original radial right-then-left pupil vertex deformation."""

    if not isinstance(profile, RinneLegacyPupilProfile):
        raise BinaryBoundsError("pupil deformation requires a Rinne pupil profile")
    profile.validate()
    if len(source_positions_xy) != len(current_positions_xy):
        raise BinaryBoundsError("pupil source/current position counts do not match")

    def pair(name: str, raw: Sequence[float]) -> tuple[float, float]:
        if len(raw) != 2:
            raise BinaryBoundsError(f"{name} must contain exactly two values")
        try:
            return _f32(float(raw[0])), _f32(float(raw[1]))
        except (BinaryBoundsError, TypeError, ValueError) as exc:
            raise BinaryBoundsError(f"{name} values must be finite float32") from exc

    right = pair("right pupil position", right_position_xy)
    left = pair("left pupil position", left_position_xy)
    for source, current in zip(
        source_positions_xy, current_positions_xy, strict=True
    ):
        pair("pupil source position", source)
        pair("pupil current position", current)
    if right == (0.0, 0.0) and left == (0.0, 0.0):
        return tuple((float(x), float(y)) for x, y in current_positions_xy)

    right_scaled = tuple(
        _mul(value, RINNE_LEGACY_PUPIL_INPUT_SCALE) for value in right
    )
    left_scaled = tuple(
        _mul(value, RINNE_LEGACY_PUPIL_INPUT_SCALE) for value in left
    )
    output: list[tuple[float, float]] = []
    for source, current in zip(
        source_positions_xy, current_positions_xy, strict=True
    ):
        x, y = _f32(current[0]), _f32(current[1])
        for center, radius, movement in (
            (profile.right_center_xy, profile.right_radius, right_scaled),
            (profile.left_center_xy, profile.left_radius, left_scaled),
        ):
            weight = _distance_falloff(source, center, radius)
            x = _add(x, _mul(movement[0], weight))
            y = _add(y, _mul(movement[1], weight))
        output.append((x, y))
    return tuple(output)


__all__ = [
    "RINNE_LEGACY_PUPIL_DISTANCE_SCALE",
    "RINNE_LEGACY_PUPIL_FALLOFF_SCALE",
    "RINNE_LEGACY_PUPIL_INPUT_SCALE",
    "RINNE_LEGACY_PUPIL_RADIUS_SCALE",
    "RinneLegacyPupilProfile",
    "V39_LEFT_PUPIL_CENTER_X_OFFSETS",
    "V39_LEFT_PUPIL_CENTER_Y_OFFSETS",
    "V39_LEFT_PUPIL_RADIUS_OFFSET",
    "V39_RIGHT_PUPIL_CENTER_X_OFFSETS",
    "V39_RIGHT_PUPIL_CENTER_Y_OFFSETS",
    "V39_RIGHT_PUPIL_RADIUS_OFFSET",
    "apply_rinne_legacy_pupil_deformation_xy",
    "parse_v39_rinne_pupil_profile",
]
