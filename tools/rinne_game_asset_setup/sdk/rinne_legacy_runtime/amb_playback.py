from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from .amb_v4 import AmbV4Animation, AmbV4FrameSample
from .checked_binary import BinaryBoundsError


AMB_V4_CLOCK_DIVISOR = 1000.0
_UINT32_MAX = 0xFFFFFFFF
_INT32_MAX = 0x7FFFFFFF


def _f32(value: float) -> float:
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise BinaryBoundsError("AMB playback value is outside float32 range") from exc
    if not math.isfinite(result):
        raise BinaryBoundsError("AMB playback value must be finite")
    return result


def _divide_f32(left: float, right: float) -> float:
    divisor = _f32(right)
    if divisor == 0.0:
        raise BinaryBoundsError("AMB playback divisor must be nonzero")
    return _f32(_f32(left) / divisor)


def _multiply_f32(left: float, right: float) -> float:
    return _f32(_f32(left) * _f32(right))


def _subtract_f32(left: float, right: float) -> float:
    return _f32(_f32(left) - _f32(right))


def _checked_u32(value: int, *, label: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > _UINT32_MAX
    ):
        raise BinaryBoundsError(f"AMB playback {label} must be uint32")
    return value


@dataclass(frozen=True)
class AmbV4PlaybackPosition:
    """The caller-controlled time and loop decision before channel sampling."""

    current_time: int
    start_time: int
    elapsed_time: int
    raw_frame_position: float
    raw_frame_index: int
    frame_index: int | None
    next_frame_index: int | None
    fraction: float
    loop: bool
    loop_count: int
    finished: bool


@dataclass(frozen=True)
class AmbV4PlaybackStep:
    position: AmbV4PlaybackPosition
    sample: AmbV4FrameSample | None


def resolve_amb_v4_playback_position(
    animation: AmbV4Animation,
    *,
    current_time: int,
    start_time: int,
    loop: bool,
) -> AmbV4PlaybackPosition:
    """Reproduce the time/frame branch at game RVA 0xE134..0xE2E4.

    Milliseconds use the original unsigned wraparound.  Looping is an explicit
    caller policy set by the game, not a demand encoded by the AMB file.
    """

    if not isinstance(animation, AmbV4Animation):
        raise BinaryBoundsError("AMB playback animation has the wrong type")
    current_time = _checked_u32(current_time, label="current time")
    start_time = _checked_u32(start_time, label="start time")
    if not isinstance(loop, bool):
        raise BinaryBoundsError("AMB playback loop flag must be boolean")

    elapsed_time = (current_time - start_time) & _UINT32_MAX
    raw_frame_position = _multiply_f32(
        _divide_f32(float(elapsed_time), AMB_V4_CLOCK_DIVISOR),
        animation.rate,
    )
    if raw_frame_position > _INT32_MAX:
        return AmbV4PlaybackPosition(
            current_time=current_time,
            start_time=start_time,
            elapsed_time=elapsed_time,
            raw_frame_position=raw_frame_position,
            raw_frame_index=-0x80000000,
            frame_index=None,
            next_frame_index=None,
            fraction=0.0,
            loop=loop,
            loop_count=0,
            finished=True,
        )

    raw_frame_index = math.trunc(raw_frame_position)
    fraction = _subtract_f32(raw_frame_position, float(raw_frame_index))
    frame_index = raw_frame_index
    loop_count = 0

    if loop:
        loop_start = animation.runtime_loop_start
        loop_end = animation.runtime_loop_end
        if (
            loop_start < 0
            or loop_end < loop_start
            or loop_end >= animation.frame_count
        ):
            raise BinaryBoundsError("AMB playback has invalid normalized loop bounds")
        if raw_frame_index > loop_end:
            loop_span = loop_end - loop_start + 1
            excess = raw_frame_index - loop_end - 1
            loop_count = excess // loop_span + 1
            frame_index = loop_start + excess % loop_span
        if frame_index > animation.frame_count - 2:
            return AmbV4PlaybackPosition(
                current_time=current_time,
                start_time=start_time,
                elapsed_time=elapsed_time,
                raw_frame_position=raw_frame_position,
                raw_frame_index=raw_frame_index,
                frame_index=None,
                next_frame_index=None,
                fraction=fraction,
                loop=loop,
                loop_count=loop_count,
                finished=True,
            )
        next_frame_index = min(frame_index + 1, loop_end)
    else:
        if raw_frame_index < 0 or raw_frame_index >= animation.frame_count - 2:
            return AmbV4PlaybackPosition(
                current_time=current_time,
                start_time=start_time,
                elapsed_time=elapsed_time,
                raw_frame_position=raw_frame_position,
                raw_frame_index=raw_frame_index,
                frame_index=None,
                next_frame_index=None,
                fraction=fraction,
                loop=loop,
                loop_count=0,
                finished=True,
            )
        next_frame_index = frame_index + 1

    return AmbV4PlaybackPosition(
        current_time=current_time,
        start_time=start_time,
        elapsed_time=elapsed_time,
        raw_frame_position=raw_frame_position,
        raw_frame_index=raw_frame_index,
        frame_index=frame_index,
        next_frame_index=next_frame_index,
        fraction=fraction,
        loop=loop,
        loop_count=loop_count,
        finished=False,
    )


def sample_amb_v4_at_time(
    animation: AmbV4Animation,
    *,
    current_time: int,
    start_time: int,
    loop: bool,
) -> AmbV4PlaybackStep:
    position = resolve_amb_v4_playback_position(
        animation,
        current_time=current_time,
        start_time=start_time,
        loop=loop,
    )
    if position.finished:
        return AmbV4PlaybackStep(position=position, sample=None)
    assert position.frame_index is not None
    assert position.next_frame_index is not None
    sample = animation.sample_frame_pair(
        position.frame_index,
        position.next_frame_index,
        position.fraction,
        frame_position=position.raw_frame_position,
    )
    return AmbV4PlaybackStep(position=position, sample=sample)


__all__ = [
    "AMB_V4_CLOCK_DIVISOR",
    "AmbV4PlaybackPosition",
    "AmbV4PlaybackStep",
    "resolve_amb_v4_playback_position",
    "sample_amb_v4_at_time",
]
