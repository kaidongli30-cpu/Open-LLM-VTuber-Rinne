from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from .amb_playback import AmbV4PlaybackStep, sample_amb_v4_at_time
from .amb_v4 import AmbV4FrameSample
from .checked_binary import BinaryBoundsError
from .eye_controls import (
    RINNE_LEFT_EYE_CLOSE_CORE_INDEX,
    RINNE_RIGHT_EYE_CLOSE_CORE_INDEX,
)
from .frame_renderer import (
    RinneLegacyFrame,
    RinneLegacyFrameControls,
    RinneLegacyFrameRenderer,
)
from .speech import RINNE_SPEECH_EXPRESSION_INDICES


RINNE_AMB_EYE_BASELINE_GAIN_INDEX = 1
RINNE_AMB_POSE_BASELINE_GAIN_INDEX = 2
RINNE_AMB_EXPRESSION_BASELINE_GAIN_INDEX = 3
RINNE_AMB_PUPIL_BASELINE_GAIN_INDEX = 0
RINNE_AMB_PUPIL_X_SCALE = 0.1
RINNE_AMB_PUPIL_Y_SCALE = 0.05


def _f32(value: float) -> float:
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise BinaryBoundsError("AMB runtime value is outside float32 range") from exc
    if not math.isfinite(result):
        raise BinaryBoundsError("AMB runtime value must be finite")
    return result


def _add_f32(left: float, right: float) -> float:
    return _f32(_f32(left) + _f32(right))


def _multiply_f32(left: float, right: float) -> float:
    return _f32(_f32(left) * _f32(right))


def _clamp_unit_f32(value: float) -> float:
    value = _f32(value)
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _compose_baseline_vector(
    baseline: Sequence[float],
    animated: Sequence[float],
    baseline_gain: float,
    *,
    name: str,
) -> tuple[float, ...]:
    baseline_values = tuple(_f32(float(value)) for value in baseline)
    animated_values = tuple(_f32(float(value)) for value in animated)
    if len(baseline_values) != len(animated_values):
        raise BinaryBoundsError(f"{name} baseline and AMB sizes do not match")
    gain = _f32(baseline_gain)
    return tuple(
        _add_f32(value, _multiply_f32(base, gain))
        for base, value in zip(baseline_values, animated_values, strict=True)
    )


def compose_rinne_amb_neck_rotation(
    baseline: Sequence[float], sample: AmbV4FrameSample
) -> tuple[float, float, float]:
    """Compose core 0..2 over the persistent neck-rotation vector."""

    if not isinstance(sample, AmbV4FrameSample):
        raise BinaryBoundsError("AMB neck-rotation sample has the wrong type")
    if len(sample.core) < 3:
        raise BinaryBoundsError("AMB sample is missing neck-rotation controls")
    if len(sample.extended_continuous) <= RINNE_AMB_POSE_BASELINE_GAIN_INDEX:
        raise BinaryBoundsError("AMB sample is missing pose baseline gain")
    result = _compose_baseline_vector(
        baseline,
        sample.core[0:3],
        sample.extended_continuous[RINNE_AMB_POSE_BASELINE_GAIN_INDEX],
        name="neck rotation",
    )
    if len(result) != 3:
        raise BinaryBoundsError("neck rotation must contain exactly 3 values")
    return result  # type: ignore[return-value]


def compose_rinne_amb_neck_translation(
    baseline: Sequence[float], sample: AmbV4FrameSample
) -> tuple[float, float, float]:
    """Compose core 3..5 over the persistent neck-translation vector."""

    if not isinstance(sample, AmbV4FrameSample):
        raise BinaryBoundsError("AMB neck-translation sample has the wrong type")
    if len(sample.core) < 6:
        raise BinaryBoundsError("AMB sample is missing neck-translation controls")
    if len(sample.extended_continuous) <= RINNE_AMB_POSE_BASELINE_GAIN_INDEX:
        raise BinaryBoundsError("AMB sample is missing pose baseline gain")
    result = _compose_baseline_vector(
        baseline,
        sample.core[3:6],
        sample.extended_continuous[RINNE_AMB_POSE_BASELINE_GAIN_INDEX],
        name="neck translation",
    )
    if len(result) != 3:
        raise BinaryBoundsError("neck translation must contain exactly 3 values")
    return result  # type: ignore[return-value]


def compose_rinne_amb_pupil_positions(
    baseline_right: Sequence[float],
    baseline_left: Sequence[float],
    sample: AmbV4FrameSample,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compose scaled core 6..9 over right/left direct pupil positions."""

    if not isinstance(sample, AmbV4FrameSample):
        raise BinaryBoundsError("AMB pupil sample has the wrong type")
    if len(sample.core) < 10:
        raise BinaryBoundsError("AMB sample is missing bilateral pupil controls")
    if len(sample.extended_continuous) <= RINNE_AMB_PUPIL_BASELINE_GAIN_INDEX:
        raise BinaryBoundsError("AMB sample is missing pupil baseline gain")
    gain = sample.extended_continuous[RINNE_AMB_PUPIL_BASELINE_GAIN_INDEX]
    animated = (
        (
            _multiply_f32(sample.core[6], RINNE_AMB_PUPIL_X_SCALE),
            _multiply_f32(sample.core[7], RINNE_AMB_PUPIL_Y_SCALE),
        ),
        (
            _multiply_f32(sample.core[8], RINNE_AMB_PUPIL_X_SCALE),
            _multiply_f32(sample.core[9], RINNE_AMB_PUPIL_Y_SCALE),
        ),
    )
    right = _compose_baseline_vector(
        baseline_right, animated[0], gain, name="right pupil position"
    )
    left = _compose_baseline_vector(
        baseline_left, animated[1], gain, name="left pupil position"
    )
    if len(right) != 2 or len(left) != 2:
        raise BinaryBoundsError("pupil positions must contain exactly 2 values")
    return right, left  # type: ignore[return-value]


def compose_rinne_amb_expression_weights(
    baseline: Sequence[float],
    sample: AmbV4FrameSample,
) -> tuple[float, ...]:
    """Compose AMB expression curves over the persistent expression vector."""

    if not isinstance(sample, AmbV4FrameSample):
        raise BinaryBoundsError("AMB expression sample has the wrong type")
    baseline_values = tuple(_f32(float(value)) for value in baseline)
    if len(baseline_values) != len(sample.expression):
        raise BinaryBoundsError(
            "AMB expression count does not match renderer baseline: "
            f"{len(sample.expression)} != {len(baseline_values)}"
        )
    if len(sample.extended_continuous) <= RINNE_AMB_EXPRESSION_BASELINE_GAIN_INDEX:
        raise BinaryBoundsError("AMB sample is missing expression baseline gain")

    baseline_gain = _f32(
        sample.extended_continuous[
            RINNE_AMB_EXPRESSION_BASELINE_GAIN_INDEX
        ]
    )
    speech_indices = set(RINNE_SPEECH_EXPRESSION_INDICES)
    return tuple(
        _add_f32(
            animated,
            (
                baseline_values[index]
                if index in speech_indices
                else _multiply_f32(baseline_values[index], baseline_gain)
            ),
        )
        for index, animated in enumerate(sample.expression)
    )


def compose_rinne_amb_eye_closes(
    baseline_right: float,
    baseline_left: float,
    sample: AmbV4FrameSample,
) -> tuple[float, float]:
    """Compose the consumer-proven bilateral AMB eye-close controls."""

    if not isinstance(sample, AmbV4FrameSample):
        raise BinaryBoundsError("AMB eye-close sample has the wrong type")
    if len(sample.core) <= RINNE_LEFT_EYE_CLOSE_CORE_INDEX:
        raise BinaryBoundsError("AMB sample is missing bilateral eye-close controls")
    if len(sample.extended_continuous) <= RINNE_AMB_EYE_BASELINE_GAIN_INDEX:
        raise BinaryBoundsError("AMB sample is missing eye-close baseline gain")

    baseline_gain = _f32(
        sample.extended_continuous[RINNE_AMB_EYE_BASELINE_GAIN_INDEX]
    )
    return (
        _clamp_unit_f32(
            _add_f32(
                sample.core[RINNE_RIGHT_EYE_CLOSE_CORE_INDEX],
                _multiply_f32(baseline_right, baseline_gain),
            )
        ),
        _clamp_unit_f32(
            _add_f32(
                sample.core[RINNE_LEFT_EYE_CLOSE_CORE_INDEX],
                _multiply_f32(baseline_left, baseline_gain),
            )
        ),
    )


@dataclass(frozen=True)
class RinneLegacyAmbRuntimeStep:
    current_time: int
    playback_step: AmbV4PlaybackStep
    controls: RinneLegacyFrameControls | None


@dataclass(frozen=True)
class RinneLegacyAmbRenderedFrame:
    step: RinneLegacyAmbRuntimeStep
    frame: RinneLegacyFrame


class RinneLegacyAmbRuntime:
    """Timed pose, expression, eye, and render-record adapter for ``001.amb``."""

    def __init__(
        self,
        renderer: RinneLegacyFrameRenderer,
        *,
        start_time: int = 0,
        loop: bool = True,
    ) -> None:
        if not isinstance(renderer, RinneLegacyFrameRenderer):
            raise BinaryBoundsError("AMB runtime renderer must be RinneLegacyFrameRenderer")
        if renderer.assets.animation.expression_channel_count != (
            renderer.expression_weight_count
        ):
            raise BinaryBoundsError(
                "AMB runtime expression count does not match the MPB renderer"
            )
        if renderer.assets.animation.auxiliary_channel_count != len(
            renderer.assets.atlas.draw_records
        ):
            raise BinaryBoundsError(
                "AMB runtime auxiliary count does not match the MPB draw records"
            )
        animation = renderer.assets.animation
        for frame_index in range(animation.frame_count):
            selectors = animation.selector_words(frame_index)
            if any(selectors):
                raise BinaryBoundsError(
                    "AMB runtime contains a nonzero render resource selector"
                )
        self.renderer = renderer
        self.start_time = self._checked_time(start_time)
        if not isinstance(loop, bool):
            raise BinaryBoundsError("AMB runtime loop flag must be boolean")
        self.loop = loop

    @staticmethod
    def _checked_time(value: int) -> int:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value > 0xFFFFFFFF
        ):
            raise BinaryBoundsError("AMB runtime start time must be uint32")
        return value

    @classmethod
    def from_pck(
        cls,
        path: Path | str,
        *,
        start_time: int = 0,
        loop: bool = True,
    ) -> RinneLegacyAmbRuntime:
        return cls(
            RinneLegacyFrameRenderer.from_pck(path),
            start_time=start_time,
            loop=loop,
        )

    def reset(self, *, start_time: int, loop: bool | None = None) -> None:
        self.start_time = self._checked_time(start_time)
        if loop is not None:
            if not isinstance(loop, bool):
                raise BinaryBoundsError("AMB runtime loop flag must be boolean")
            self.loop = loop

    def advance(
        self,
        current_time: int,
        controls: RinneLegacyFrameControls | None = None,
    ) -> RinneLegacyAmbRuntimeStep:
        base_controls = RinneLegacyFrameControls() if controls is None else controls
        if not isinstance(base_controls, RinneLegacyFrameControls):
            raise BinaryBoundsError("AMB runtime controls must be RinneLegacyFrameControls")
        playback_step = sample_amb_v4_at_time(
            self.renderer.assets.animation,
            current_time=current_time,
            start_time=self.start_time,
            loop=self.loop,
        )
        sample = playback_step.sample
        if sample is None:
            return RinneLegacyAmbRuntimeStep(
                current_time=current_time,
                playback_step=playback_step,
                controls=None,
            )

        expression_count = self.renderer.expression_weight_count
        baseline_expression = (
            (0.0,) * expression_count
            if base_controls.expression_weights is None
            else base_controls.expression_weights
        )
        expression_weights = compose_rinne_amb_expression_weights(
            baseline_expression,
            sample,
        )
        baseline_right, baseline_left = base_controls.resolve_eye_closes()
        right_eye_close, left_eye_close = compose_rinne_amb_eye_closes(
            baseline_right,
            baseline_left,
            sample,
        )
        (
            baseline_neck_rotation,
            baseline_neck_translation,
            baseline_right_pupil,
            baseline_left_pupil,
        ) = base_controls.resolve_core_pose()
        neck_rotation = compose_rinne_amb_neck_rotation(
            baseline_neck_rotation, sample
        )
        neck_translation = compose_rinne_amb_neck_translation(
            baseline_neck_translation, sample
        )
        right_pupil_position, left_pupil_position = (
            compose_rinne_amb_pupil_positions(
                baseline_right_pupil,
                baseline_left_pupil,
                sample,
            )
        )
        resolved_controls = replace(
            base_controls,
            expression_weights=expression_weights,
            right_eye_close=right_eye_close,
            left_eye_close=left_eye_close,
            neck_rotation=neck_rotation,
            neck_translation=neck_translation,
            right_pupil_position=right_pupil_position,
            left_pupil_position=left_pupil_position,
            render_record_opacities=sample.auxiliary,
            render_resource_selectors=sample.extended_selector_words,
        )
        return RinneLegacyAmbRuntimeStep(
            current_time=current_time,
            playback_step=playback_step,
            controls=resolved_controls,
        )

    def render_at(
        self,
        current_time: int,
        controls: RinneLegacyFrameControls | None = None,
        *,
        width: int = 512,
        height: int = 512,
        include_diagnostics: bool = False,
    ) -> RinneLegacyAmbRenderedFrame:
        step = self.advance(current_time, controls)
        if step.controls is None:
            raise BinaryBoundsError(
                "AMB animation has finished; the caller must select a transition"
            )
        frame = self.renderer.render_frame(
            step.controls,
            width=width,
            height=height,
            include_diagnostics=include_diagnostics,
        )
        return RinneLegacyAmbRenderedFrame(step=step, frame=frame)


__all__ = [
    "RINNE_AMB_EXPRESSION_BASELINE_GAIN_INDEX",
    "RINNE_AMB_EYE_BASELINE_GAIN_INDEX",
    "RINNE_AMB_POSE_BASELINE_GAIN_INDEX",
    "RINNE_AMB_PUPIL_BASELINE_GAIN_INDEX",
    "RINNE_AMB_PUPIL_X_SCALE",
    "RINNE_AMB_PUPIL_Y_SCALE",
    "RinneLegacyAmbRenderedFrame",
    "RinneLegacyAmbRuntime",
    "RinneLegacyAmbRuntimeStep",
    "compose_rinne_amb_expression_weights",
    "compose_rinne_amb_eye_closes",
    "compose_rinne_amb_neck_rotation",
    "compose_rinne_amb_neck_translation",
    "compose_rinne_amb_pupil_positions",
]
