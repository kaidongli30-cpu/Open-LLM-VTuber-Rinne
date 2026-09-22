from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from .checked_binary import BinaryBoundsError


LEGACY_BREATH_DEFAULT_ENABLED = 1
LEGACY_BREATH_DEFAULT_DURATION_FACTOR = 1.0
LEGACY_BREATH_INITIAL_PHASE_DEGREES = -90.0
LEGACY_BREATH_PHASE_STEP_DEGREES = 3.5
LEGACY_BREATH_CLOCK_DIVISOR = 30.0
LEGACY_BREATH_FULL_CYCLE_DEGREES = 360.0
RINNE_BREATH_EXPRESSION_INDEX = 4
_UINT32_MAX = 0xFFFFFFFF


def _f32(value: float) -> float:
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise BinaryBoundsError(
            "breath scheduler value is outside float32 range"
        ) from exc
    if not math.isfinite(result):
        raise BinaryBoundsError("breath scheduler value must be finite")
    return result


def _add(left: float, right: float) -> float:
    return _f32(_f32(left) + _f32(right))


def _subtract(left: float, right: float) -> float:
    return _f32(_f32(left) - _f32(right))


def _multiply(left: float, right: float) -> float:
    return _f32(_f32(left) * _f32(right))


def _divide(left: float, right: float) -> float:
    divisor = _f32(right)
    if divisor == 0.0:
        raise BinaryBoundsError("breath scheduler divisor must be nonzero")
    return _f32(_f32(left) / divisor)


def _checked_u32(value: int, *, label: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > _UINT32_MAX
    ):
        raise BinaryBoundsError(f"breath scheduler {label} must be uint32")
    return value


def _sample_gain(phase_degrees: float) -> float:
    radians = _divide(_multiply(phase_degrees, math.pi), 180.0)
    sine = _f32(math.sin(radians))
    return _add(_multiply(sine, 0.5), 0.5)


@dataclass(frozen=True)
class LegacyBreathSchedulerState:
    enabled: int
    duration_factor: float
    phase_degrees: float
    expression_gain: float


@dataclass(frozen=True)
class LegacyBreathSchedulerStep:
    state: LegacyBreathSchedulerState
    sampled_phase_degrees: float
    expression_gain: float
    delta_time: int
    phase_advanced: int


def create_legacy_breath_scheduler_state(
    *,
    enabled: int = LEGACY_BREATH_DEFAULT_ENABLED,
    duration_factor: float = LEGACY_BREATH_DEFAULT_DURATION_FACTOR,
) -> LegacyBreathSchedulerState:
    """Create the automatic-breath state initialized at RVA 0xB3A0."""

    enabled = _checked_u32(enabled, label="enabled")
    duration_factor = _f32(duration_factor)
    if duration_factor <= 0.0:
        raise BinaryBoundsError("breath scheduler duration factor must be positive")
    phase_degrees = _f32(LEGACY_BREATH_INITIAL_PHASE_DEGREES)
    return LegacyBreathSchedulerState(
        enabled=enabled,
        duration_factor=duration_factor,
        phase_degrees=phase_degrees,
        expression_gain=_sample_gain(phase_degrees),
    )


def _validate_state(state: LegacyBreathSchedulerState) -> None:
    if not isinstance(state, LegacyBreathSchedulerState):
        raise BinaryBoundsError("breath scheduler state has the wrong type")
    _checked_u32(state.enabled, label="enabled")
    duration_factor = _f32(state.duration_factor)
    if duration_factor <= 0.0:
        raise BinaryBoundsError("breath scheduler duration factor must be positive")
    _f32(state.phase_degrees)
    _f32(state.expression_gain)


def update_legacy_breath_scheduler(
    state: LegacyBreathSchedulerState,
    *,
    previous_time: int,
    current_time: int,
) -> LegacyBreathSchedulerStep:
    """Advance the automatic-breath channel reconstructed from RVA 0xC4B0.

    The old controller writes the visible gain from the phase at function
    entry, then advances the stored phase for the following frame.  Its outer
    update gate runs only for a strictly increasing uint32 millisecond clock.
    """

    _validate_state(state)
    previous_time = _checked_u32(previous_time, label="previous time")
    current_time = _checked_u32(current_time, label="current time")

    sampled_phase = _f32(state.phase_degrees)
    expression_gain = _sample_gain(sampled_phase)
    next_phase = sampled_phase
    delta_time = 0
    phase_advanced = 0

    if current_time > previous_time:
        delta_time = current_time - previous_time
        if state.enabled == 1:
            delta_frames = _divide(_f32(float(delta_time)), LEGACY_BREATH_CLOCK_DIVISOR)
            phase_increment = _divide(
                _multiply(delta_frames, LEGACY_BREATH_PHASE_STEP_DEGREES),
                state.duration_factor,
            )
            next_phase = _add(next_phase, phase_increment)
            if next_phase >= LEGACY_BREATH_FULL_CYCLE_DEGREES:
                next_phase = _subtract(
                    next_phase, LEGACY_BREATH_FULL_CYCLE_DEGREES
                )
            phase_advanced = 1

    next_state = LegacyBreathSchedulerState(
        enabled=state.enabled,
        duration_factor=_f32(state.duration_factor),
        phase_degrees=next_phase,
        expression_gain=expression_gain,
    )
    return LegacyBreathSchedulerStep(
        state=next_state,
        sampled_phase_degrees=sampled_phase,
        expression_gain=expression_gain,
        delta_time=delta_time,
        phase_advanced=phase_advanced,
    )


__all__ = [
    "LEGACY_BREATH_CLOCK_DIVISOR",
    "LEGACY_BREATH_DEFAULT_DURATION_FACTOR",
    "LEGACY_BREATH_DEFAULT_ENABLED",
    "LEGACY_BREATH_FULL_CYCLE_DEGREES",
    "LEGACY_BREATH_INITIAL_PHASE_DEGREES",
    "LEGACY_BREATH_PHASE_STEP_DEGREES",
    "RINNE_BREATH_EXPRESSION_INDEX",
    "LegacyBreathSchedulerState",
    "LegacyBreathSchedulerStep",
    "create_legacy_breath_scheduler_state",
    "update_legacy_breath_scheduler",
]
