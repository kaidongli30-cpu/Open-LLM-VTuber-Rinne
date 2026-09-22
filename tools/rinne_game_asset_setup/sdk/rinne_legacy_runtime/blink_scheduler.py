from __future__ import annotations

import math
import struct
from collections.abc import Callable
from dataclasses import dataclass

from .blink_curve import LegacyBlinkGeometrySample, sample_legacy_blink_geometry
from .checked_binary import BinaryBoundsError


LEGACY_BLINK_DEFAULT_ENABLED = 1
LEGACY_BLINK_DEFAULT_DURATION_FACTOR = 1.0
LEGACY_BLINK_DEFAULT_FREQUENCIES = (5, 1, 1)
LEGACY_BLINK_RAND_MAX = 32767
_UINT32_MAX = 0xFFFFFFFF
_INT32_MAX = 0x7FFFFFFF
_LEGACY_GAIN_RANGE_F32 = struct.unpack("<f", struct.pack("<f", 0.8))[0]
_LEGACY_GAIN_BASE_WITH_HALF_F32 = struct.unpack(
    "<f", struct.pack("<f", 0.5)
)[0]
_LEGACY_GAIN_BASE_WITHOUT_HALF_F32 = _LEGACY_GAIN_RANGE_F32


def _f32(value: float) -> float:
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise BinaryBoundsError("blink scheduler value is outside float32 range") from exc
    if not math.isfinite(result):
        raise BinaryBoundsError("blink scheduler value must be finite")
    return result


def _add(left: float, right: float) -> float:
    return _f32(_f32(left) + _f32(right))


def _mul(left: float, right: float) -> float:
    return _f32(_f32(left) * _f32(right))


def _checked_u32(value: int, *, label: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > _UINT32_MAX
    ):
        raise BinaryBoundsError(f"blink scheduler {label} must be uint32")
    return value


def _signed_u32_difference(left: int, right: int) -> int:
    value = (left - right) & _UINT32_MAX
    return value - 0x100000000 if value > _INT32_MAX else value


def _checked_frequencies(values: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(values, tuple) or len(values) != 3:
        raise BinaryBoundsError("blink scheduler requires three frequencies")
    checked: list[int] = []
    for value in values:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value > _INT32_MAX
        ):
            raise BinaryBoundsError(
                "blink scheduler frequencies must be non-negative int32"
            )
        checked.append(value)
    result = tuple(checked)
    if sum(result) <= 0 or sum(result) > _INT32_MAX:
        raise BinaryBoundsError(
            "blink scheduler frequency sum must be within 1..int32_max"
        )
    return result  # type: ignore[return-value]


def _next_random(random_int: Callable[[], int]) -> int:
    value = random_int()
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > LEGACY_BLINK_RAND_MAX
    ):
        raise BinaryBoundsError(
            f"blink scheduler random source must return 0..{LEGACY_BLINK_RAND_MAX}"
        )
    return value


@dataclass(frozen=True)
class LegacyBlinkSchedulerState:
    enabled: int
    elapsed: int
    anchor_time: int
    interval: int
    duration_factor: float
    frequencies: tuple[int, int, int]
    single_blink_gain: float
    blink_type: int
    geometry: LegacyBlinkGeometrySample


@dataclass(frozen=True)
class LegacyBlinkSchedulerStep:
    state: LegacyBlinkSchedulerState
    cycle_started: int
    cycle_gate: int
    half_blink_fourth_sample: float


def create_legacy_blink_scheduler_state(
    *,
    enabled: int = LEGACY_BLINK_DEFAULT_ENABLED,
    duration_factor: float = LEGACY_BLINK_DEFAULT_DURATION_FACTOR,
    frequencies: tuple[int, int, int] = LEGACY_BLINK_DEFAULT_FREQUENCIES,
) -> LegacyBlinkSchedulerState:
    """Create zero-time state with the nonzero defaults stored at RVA 0xB3A0."""

    enabled = _checked_u32(enabled, label="enabled")
    duration_factor = _f32(duration_factor)
    if duration_factor <= 0.0:
        raise BinaryBoundsError("blink scheduler duration factor must be positive")
    frequencies = _checked_frequencies(frequencies)
    return LegacyBlinkSchedulerState(
        enabled=enabled,
        elapsed=0,
        anchor_time=0,
        interval=0,
        duration_factor=duration_factor,
        frequencies=frequencies,
        single_blink_gain=0.0,
        blink_type=0,
        geometry=LegacyBlinkGeometrySample(
            deformation=(0.0, 0.0, 0.0, 0.0),
            base_weights=(0.0, 0.0, 0.0, 0.0),
        ),
    )


def _validate_state(state: LegacyBlinkSchedulerState) -> None:
    if not isinstance(state, LegacyBlinkSchedulerState):
        raise BinaryBoundsError("blink scheduler state has the wrong type")
    _checked_u32(state.enabled, label="enabled")
    _checked_u32(state.elapsed, label="elapsed")
    _checked_u32(state.anchor_time, label="anchor time")
    _checked_u32(state.interval, label="interval")
    duration_factor = _f32(state.duration_factor)
    if duration_factor <= 0.0:
        raise BinaryBoundsError("blink scheduler duration factor must be positive")
    _checked_frequencies(state.frequencies)
    if not math.isfinite(state.single_blink_gain):
        raise BinaryBoundsError("blink scheduler gain must be finite")
    if state.blink_type not in (0, 1, 2):
        raise BinaryBoundsError("blink scheduler type must be within 0..2")


def update_legacy_blink_scheduler(
    state: LegacyBlinkSchedulerState,
    *,
    previous_time: int,
    current_time: int,
    floor: float,
    random_int: Callable[[], int],
) -> LegacyBlinkSchedulerStep:
    """Advance the automatic blink state reconstructed from RVA 0xC070."""

    _validate_state(state)
    previous_time = _checked_u32(previous_time, label="previous time")
    current_time = _checked_u32(current_time, label="current time")
    if not callable(random_int):
        raise BinaryBoundsError("blink scheduler random source must be callable")

    anchor_time = state.anchor_time or current_time
    elapsed = (current_time - anchor_time) & _UINT32_MAX
    interval = state.interval
    single_blink_gain = _f32(state.single_blink_gain)
    blink_type = state.blink_type
    cycle_started = 0
    cycle_gate = 0

    if elapsed >= interval and state.enabled != 0:
        gain_random = _next_random(random_int)
        ratio = _f32(float(gain_random) / float(LEGACY_BLINK_RAND_MAX))
        gain_base = (
            _LEGACY_GAIN_BASE_WITHOUT_HALF_F32
            if state.frequencies[1] == 0
            else _LEGACY_GAIN_BASE_WITH_HALF_F32
        )
        single_blink_gain = _add(
            _mul(ratio, _LEGACY_GAIN_RANGE_F32), gain_base
        )
        if single_blink_gain > 1.0:
            single_blink_gain = 1.0

        anchor_time = current_time
        elapsed = 0

        interval_random = _next_random(random_int)
        interval_base = 30 * (40 + interval_random % 50)
        interval_float = _mul(float(interval_base), state.duration_factor)
        interval = math.trunc(interval_float)
        if interval < 0 or interval > _UINT32_MAX:
            raise BinaryBoundsError(
                "blink scheduler interval is outside the checked uint32 range"
            )

        selection_random = _next_random(random_int)
        selection = selection_random % sum(state.frequencies)
        first_end = state.frequencies[0]
        second_end = first_end + state.frequencies[1]
        if selection < first_end:
            blink_type = 0
        elif selection < second_end:
            blink_type = 1
        else:
            blink_type = 2
        cycle_gate = int(selection_random % 5 < 3)
        cycle_started = 1

    geometry = sample_legacy_blink_geometry(
        blink_type,
        current_elapsed=_signed_u32_difference(current_time, anchor_time),
        previous_elapsed=_signed_u32_difference(previous_time, anchor_time),
        single_blink_gain=single_blink_gain,
        floor=floor,
    )
    next_state = LegacyBlinkSchedulerState(
        enabled=state.enabled,
        elapsed=elapsed,
        anchor_time=anchor_time,
        interval=interval,
        duration_factor=_f32(state.duration_factor),
        frequencies=state.frequencies,
        single_blink_gain=single_blink_gain,
        blink_type=blink_type,
        geometry=geometry,
    )
    half_blink_fourth_sample = (
        geometry.deformation[3]
        if state.enabled == 1 and blink_type == 1
        else 0.0
    )
    return LegacyBlinkSchedulerStep(
        state=next_state,
        cycle_started=cycle_started,
        cycle_gate=cycle_gate,
        half_blink_fourth_sample=half_blink_fourth_sample,
    )


__all__ = [
    "LEGACY_BLINK_DEFAULT_DURATION_FACTOR",
    "LEGACY_BLINK_DEFAULT_ENABLED",
    "LEGACY_BLINK_DEFAULT_FREQUENCIES",
    "LEGACY_BLINK_RAND_MAX",
    "LegacyBlinkSchedulerState",
    "LegacyBlinkSchedulerStep",
    "create_legacy_blink_scheduler_state",
    "update_legacy_blink_scheduler",
]
