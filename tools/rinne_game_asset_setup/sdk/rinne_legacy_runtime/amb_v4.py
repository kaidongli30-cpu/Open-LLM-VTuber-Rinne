from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from .checked_binary import BinaryBoundsError, CheckedBinary
from .mpb_v39_renderer import Float32ScalarView


AMB_MAGIC = b"BMAM"
AMB_V4_VERSION = 4
AMB_V4_BASE_HEADER_SIZE = 0x28
AMB_V4_CONFIG_SIZE = 0x58
AMB_V4_FIXED_CHANNEL_COUNT = 29
AMB_MAX_CHANNELS = 4096
AMB_MAX_FRAMES = 1_000_000
AMB_MAX_FLOAT_VALUES = 16_777_216


def _f32(value: float) -> float:
    """Round one scalar exactly as an SSE single-precision operation."""

    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError("AMB sampling value is outside float32 range") from exc
    if not math.isfinite(result):
        raise ValueError("AMB sampling value must be finite")
    return result


def _add_f32(left: float, right: float) -> float:
    return _f32(_f32(left) + _f32(right))


def _subtract_f32(left: float, right: float) -> float:
    return _f32(_f32(left) - _f32(right))


def _multiply_f32(left: float, right: float) -> float:
    return _f32(_f32(left) * _f32(right))


@dataclass(frozen=True)
class AmbV4Config:
    """The game-proven 22-slot legacy unconscious-animation control block."""

    neck_enabled: tuple[int, int, int]
    neck_duration_factors: tuple[float, float, float]
    neck_max_rotations: tuple[float, float, float]
    blink_enabled: int
    blink_duration_factor: float
    blink_frequencies: tuple[int, int, int]
    pupil_enabled: int
    pupil_duration_factor: float
    pupil_x_max: float
    pupil_y_max: float
    expression_enabled: int
    breath_enabled: int
    expression_duration_factor: float
    breath_duration_factor: float

    def summary_dict(self) -> dict[str, object]:
        return {
            "neck_enabled": self.neck_enabled,
            "neck_duration_factors": self.neck_duration_factors,
            "neck_max_rotations": self.neck_max_rotations,
            "blink_enabled": self.blink_enabled,
            "blink_duration_factor": self.blink_duration_factor,
            "blink_frequencies": self.blink_frequencies,
            "pupil_enabled": self.pupil_enabled,
            "pupil_duration_factor": self.pupil_duration_factor,
            "pupil_x_max": self.pupil_x_max,
            "pupil_y_max": self.pupil_y_max,
            "expression_enabled": self.expression_enabled,
            "breath_enabled": self.breath_enabled,
            "expression_duration_factor": self.expression_duration_factor,
            "breath_duration_factor": self.breath_duration_factor,
        }


@dataclass(frozen=True)
class AmbV4ChannelGroups:
    core: range
    expression: range
    auxiliary: range
    trailing_triplet: range
    extended: range


@dataclass(frozen=True)
class AmbV4FrameSample:
    """One mechanically sampled BMAM v4 frame.

    The names deliberately stop at the consumer-proven channel groups. The
    first ten extended values are discrete selectors; every other value here
    is linearly interpolated. Higher-level names such as ``blink`` and
    ``mouth`` must only be added after a channel has been mapped to a render
    record.
    """

    frame_position: float
    frame_index: int
    next_frame_index: int
    fraction: float
    core: tuple[float, ...]
    expression: tuple[float, ...]
    auxiliary: tuple[float, ...]
    trailing_triplet: tuple[float, ...]
    extended_discrete: tuple[float, ...]
    extended_continuous: tuple[float, ...]

    @property
    def extended(self) -> tuple[float, ...]:
        return self.extended_discrete + self.extended_continuous

    @property
    def extended_selector_words(self) -> tuple[int, ...]:
        """Return the ten discrete fields as their consumer-read uint32 words."""

        return tuple(
            struct.unpack("<I", struct.pack("<f", value))[0]
            for value in self.extended_discrete
        )

    @property
    def channel_values(self) -> tuple[float, ...]:
        return (
            self.core
            + self.expression
            + self.auxiliary
            + self.trailing_triplet
            + self.extended
        )


@dataclass(frozen=True)
class AmbV4Animation:
    version: int
    rate: float
    channel_count: int
    expression_channel_count: int
    auxiliary_channel_count: int
    frame_count: int
    loop_start_raw: int
    loop_end_raw: int
    runtime_loop_start: int
    runtime_loop_end: int
    config_present: bool
    config_payload: bytes
    config: AmbV4Config | None
    data_offset: int
    channels: tuple[Float32ScalarView, ...]

    @property
    def channel_groups(self) -> AmbV4ChannelGroups:
        expression_start = 12
        auxiliary_start = expression_start + self.expression_channel_count
        triplet_start = auxiliary_start + self.auxiliary_channel_count
        extended_start = triplet_start + 3
        return AmbV4ChannelGroups(
            core=range(0, expression_start),
            expression=range(expression_start, auxiliary_start),
            auxiliary=range(auxiliary_start, triplet_start),
            trailing_triplet=range(triplet_start, extended_start),
            extended=range(extended_start, self.channel_count),
        )

    def value(self, channel_index: int, frame_index: int) -> float:
        if channel_index < 0 or channel_index >= self.channel_count:
            raise IndexError(f"AMB channel index {channel_index} out of range")
        return self.channels[channel_index].at(frame_index)

    def selector_words(self, frame_index: int) -> tuple[int, ...]:
        """Return one frame's ten discrete selector fields as uint32 words."""

        groups = self.channel_groups
        discrete_stop = groups.extended.start + 10
        if discrete_stop > groups.extended.stop:
            raise ValueError("AMB v4 extended group is shorter than 10 selectors")
        return tuple(
            struct.unpack(
                "<I",
                struct.pack("<f", self.value(channel_index, frame_index)),
            )[0]
            for channel_index in range(groups.extended.start, discrete_stop)
        )

    def sample_frame_position(self, frame_position: float) -> AmbV4FrameSample:
        """Sample a non-looped fractional frame position like RVA 0xE0A0.

        Playback and loop policy intentionally remain outside this method.
        Callers must supply a finite position in the closed range
        ``[0, frame_count - 1]``.
        """

        if not math.isfinite(frame_position):
            raise ValueError("AMB frame position must be finite")
        frame_position = _f32(frame_position)
        last_frame = self.frame_count - 1
        if frame_position < 0.0 or frame_position > last_frame:
            raise ValueError(
                f"AMB frame position {frame_position!r} outside [0, {last_frame}]"
            )

        frame_index = math.floor(frame_position)
        next_frame_index = min(frame_index + 1, last_frame)
        fraction = frame_position - frame_index
        if next_frame_index == frame_index:
            fraction = 0.0

        return self.sample_frame_pair(
            frame_index,
            next_frame_index,
            fraction,
            frame_position=frame_position,
        )

    def sample_frame_pair(
        self,
        frame_index: int,
        next_frame_index: int,
        fraction: float,
        *,
        frame_position: float | None = None,
    ) -> AmbV4FrameSample:
        """Sample an explicitly resolved pair using the game's float32 rules.

        Playback code uses this entry point after applying caller-controlled
        loop policy.  The legacy selector comparison is ``0.5 >= fraction``:
        an exact half frame therefore selects the following discrete value.
        """

        for label, index in (
            ("frame", frame_index),
            ("next frame", next_frame_index),
        ):
            if not isinstance(index, int) or isinstance(index, bool):
                raise ValueError(f"AMB {label} index must be an integer")
            if index < 0 or index >= self.frame_count:
                raise ValueError(f"AMB {label} index {index} out of range")
        fraction = _f32(fraction)
        if fraction < 0.0 or fraction >= 1.0:
            raise ValueError("AMB frame fraction must be within [0, 1)")
        if frame_position is None:
            frame_position = _add_f32(float(frame_index), fraction)
        else:
            frame_position = _f32(frame_position)

        current_weight = _subtract_f32(1.0, fraction)

        def linear(channel_index: int) -> float:
            current = self.value(channel_index, frame_index)
            following = self.value(channel_index, next_frame_index)
            return _add_f32(
                _multiply_f32(following, fraction),
                _multiply_f32(current, current_weight),
            )

        def linear_group(indices: range) -> tuple[float, ...]:
            return tuple(linear(channel_index) for channel_index in indices)

        groups = self.channel_groups
        discrete_stop = groups.extended.start + 10
        if discrete_stop > groups.extended.stop:
            raise ValueError("AMB v4 extended group is shorter than 10 selectors")
        selected_frame = next_frame_index if fraction >= 0.5 else frame_index
        extended_discrete = tuple(
            self.value(channel_index, selected_frame)
            for channel_index in range(groups.extended.start, discrete_stop)
        )

        return AmbV4FrameSample(
            frame_position=frame_position,
            frame_index=frame_index,
            next_frame_index=next_frame_index,
            fraction=fraction,
            core=linear_group(groups.core),
            expression=linear_group(groups.expression),
            auxiliary=linear_group(groups.auxiliary),
            trailing_triplet=linear_group(groups.trailing_triplet),
            extended_discrete=extended_discrete,
            extended_continuous=linear_group(
                range(discrete_stop, groups.extended.stop)
            ),
        )

    def summary_dict(self) -> dict[str, object]:
        groups = self.channel_groups
        return {
            "version": self.version,
            "rate": self.rate,
            "channel_count": self.channel_count,
            "expression_channel_count": self.expression_channel_count,
            "auxiliary_channel_count": self.auxiliary_channel_count,
            "frame_count": self.frame_count,
            "loop_start_raw": self.loop_start_raw,
            "loop_end_raw": self.loop_end_raw,
            "runtime_loop_start": self.runtime_loop_start,
            "runtime_loop_end": self.runtime_loop_end,
            "config_present": self.config_present,
            "config_size": len(self.config_payload),
            "config": None if self.config is None else self.config.summary_dict(),
            "data_offset": self.data_offset,
            "channel_groups": {
                "core": [groups.core.start, groups.core.stop],
                "expression": [groups.expression.start, groups.expression.stop],
                "auxiliary": [groups.auxiliary.start, groups.auxiliary.stop],
                "trailing_triplet": [
                    groups.trailing_triplet.start,
                    groups.trailing_triplet.stop,
                ],
                "extended": [groups.extended.start, groups.extended.stop],
            },
        }


def _as_i32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x1_0000_0000 if value & 0x8000_0000 else value


def _normalize_runtime_loop(
    *, frame_count: int, loop_start_raw: int, loop_end_raw: int
) -> tuple[int, int]:
    """Mirror the signed-int normalization at game RVA 0x2c7e..0x2ca4."""

    runtime_start = _as_i32(loop_start_raw)
    runtime_end = _as_i32(loop_end_raw - 1)
    if runtime_start > runtime_end:
        runtime_start = runtime_end

    cap = _as_i32(frame_count - 2)
    if cap < runtime_start:
        runtime_start = 0
    if cap < runtime_end:
        runtime_end = cap
    return runtime_start, runtime_end


def _parse_v4_config(payload: bytes) -> AmbV4Config:
    if len(payload) != AMB_V4_CONFIG_SIZE:
        raise BinaryBoundsError(
            f"AMB v4 config must be {AMB_V4_CONFIG_SIZE} bytes"
        )

    def integer(index: int) -> int:
        return struct.unpack_from("<I", payload, index * 4)[0]

    def scalar(index: int) -> float:
        value = struct.unpack_from("<f", payload, index * 4)[0]
        if not math.isfinite(value):
            raise BinaryBoundsError(
                f"AMB v4 config slot {index} contains a non-finite float"
            )
        return value

    return AmbV4Config(
        neck_enabled=(integer(0), integer(1), integer(2)),
        neck_duration_factors=(scalar(3), scalar(4), scalar(5)),
        neck_max_rotations=(scalar(6), scalar(7), scalar(8)),
        blink_enabled=integer(9),
        blink_duration_factor=scalar(10),
        blink_frequencies=(integer(11), integer(12), integer(13)),
        pupil_enabled=integer(14),
        pupil_duration_factor=scalar(15),
        pupil_x_max=scalar(16),
        pupil_y_max=scalar(17),
        expression_enabled=integer(18),
        breath_enabled=integer(19),
        expression_duration_factor=scalar(20),
        breath_duration_factor=scalar(21),
    )


def parse_amb_v4(
    data: bytes | bytearray | memoryview,
) -> AmbV4Animation:
    """Parse the game-proven little-endian BMAM v4 channel-major layout.

    Channel group boundaries mirror the v4 consumer at game RVA
    0x304a..0x3276. Semantic names inside those groups remain intentionally
    unassigned until their animation consumers are traced.
    """

    stable_data = data if isinstance(data, bytes) else bytes(data)
    reader = CheckedBinary(stable_data)
    reader.span("AMB v4 base header", 0, AMB_V4_BASE_HEADER_SIZE)
    if reader.bytes_at("AMB magic", 0, 4) != AMB_MAGIC:
        raise BinaryBoundsError("AMB input has the wrong BMAM magic")

    version = reader.u32("AMB version", 4)
    if version != AMB_V4_VERSION:
        raise BinaryBoundsError(
            f"AMB compatibility parser requires version {AMB_V4_VERSION}, "
            f"got {version}"
        )

    rate = struct.unpack_from("<f", stable_data, 8)[0]
    if not math.isfinite(rate) or rate <= 0.0:
        raise BinaryBoundsError("AMB rate must be finite and positive")

    channel_count = reader.u32("AMB channel count", 0x0C)
    expression_channel_count = reader.u32("AMB expression channel count", 0x10)
    auxiliary_channel_count = reader.u32("AMB auxiliary channel count", 0x14)
    frame_count = reader.u32("AMB frame count", 0x18)
    loop_start_raw = reader.u32("AMB raw loop start", 0x1C)
    loop_end_raw = reader.u32("AMB raw loop end", 0x20)
    config_word = reader.u32("AMB config-present word", 0x24)

    if channel_count == 0 or channel_count > AMB_MAX_CHANNELS:
        raise BinaryBoundsError(
            f"AMB channel count {channel_count} exceeds safety range"
        )
    if frame_count == 0 or frame_count > AMB_MAX_FRAMES:
        raise BinaryBoundsError(
            f"AMB frame count {frame_count} exceeds safety range"
        )
    expected_channels = (
        expression_channel_count
        + auxiliary_channel_count
        + AMB_V4_FIXED_CHANNEL_COUNT
    )
    if channel_count != expected_channels:
        raise BinaryBoundsError(
            f"AMB v4 channel count {channel_count} does not equal "
            f"expression {expression_channel_count} + auxiliary "
            f"{auxiliary_channel_count} + fixed {AMB_V4_FIXED_CHANNEL_COUNT}"
        )
    if channel_count > AMB_MAX_FLOAT_VALUES // frame_count:
        raise BinaryBoundsError("AMB channel/frame product exceeds safety limit")
    runtime_loop_start, runtime_loop_end = _normalize_runtime_loop(
        frame_count=frame_count,
        loop_start_raw=loop_start_raw,
        loop_end_raw=loop_end_raw,
    )

    config_present = config_word != 0
    if config_present:
        config_payload = reader.bytes_at(
            "AMB v4 config payload",
            AMB_V4_BASE_HEADER_SIZE,
            AMB_V4_CONFIG_SIZE,
        )
        data_offset = AMB_V4_BASE_HEADER_SIZE + AMB_V4_CONFIG_SIZE
        config = _parse_v4_config(config_payload)
    else:
        config_payload = b""
        config = None
        data_offset = AMB_V4_BASE_HEADER_SIZE

    value_count = channel_count * frame_count
    value_span = reader.array_span(
        "AMB channel-major float payload",
        data_offset,
        value_count,
        4,
        max_count=AMB_MAX_FLOAT_VALUES,
    )
    if value_span.end != reader.size:
        raise BinaryBoundsError(
            "AMB v4 payload does not consume the complete input: "
            f"0x{value_span.end:x} != 0x{reader.size:x}"
        )

    channels = tuple(
        Float32ScalarView.create(
            stable_data,
            name=f"AMB channel {channel_index}",
            offset=data_offset + channel_index * frame_count * 4,
            count=frame_count,
        )
        for channel_index in range(channel_count)
    )
    for channel in channels:
        channel.require_finite()

    return AmbV4Animation(
        version=version,
        rate=rate,
        channel_count=channel_count,
        expression_channel_count=expression_channel_count,
        auxiliary_channel_count=auxiliary_channel_count,
        frame_count=frame_count,
        loop_start_raw=loop_start_raw,
        loop_end_raw=loop_end_raw,
        runtime_loop_start=runtime_loop_start,
        runtime_loop_end=runtime_loop_end,
        config_present=config_present,
        config_payload=config_payload,
        config=config,
        data_offset=data_offset,
        channels=channels,
    )


__all__ = [
    "AmbV4Animation",
    "AmbV4ChannelGroups",
    "AmbV4Config",
    "AmbV4FrameSample",
    "parse_amb_v4",
]
