from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from .breath_scheduler import (
    RINNE_BREATH_EXPRESSION_INDEX,
    LegacyBreathSchedulerState,
    LegacyBreathSchedulerStep,
    create_legacy_breath_scheduler_state,
    update_legacy_breath_scheduler,
)
from .checked_binary import BinaryBoundsError
from .frame_renderer import (
    RinneLegacyFrame,
    RinneLegacyFrameControls,
    RinneLegacyFrameRenderer,
)


def _f32(value: float) -> float:
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise BinaryBoundsError(
            "breath expression value is outside float32 range"
        ) from exc
    if not math.isfinite(result):
        raise BinaryBoundsError("breath expression value must be finite")
    return result


def apply_breath_expression_gain(
    expression_values: Sequence[float],
    breath_gain: float,
    *,
    expression_index: int = RINNE_BREATH_EXPRESSION_INDEX,
) -> tuple[float, ...]:
    """Add the automatic breath value to its proven expression slot."""

    output = [_f32(float(value)) for value in expression_values]
    if not isinstance(expression_index, int) or isinstance(expression_index, bool):
        raise BinaryBoundsError("breath expression index must be an integer")
    if expression_index < 0 or expression_index >= len(output):
        raise BinaryBoundsError(
            f"breath expression index {expression_index} outside expression array"
        )
    output[expression_index] = _f32(
        output[expression_index] + _f32(breath_gain)
    )
    return tuple(output)


@dataclass(frozen=True)
class RinneLegacyBreathRuntimeStep:
    current_time: int
    scheduler_step: LegacyBreathSchedulerStep
    controls: RinneLegacyFrameControls


@dataclass(frozen=True)
class RinneLegacyBreathRenderedFrame:
    step: RinneLegacyBreathRuntimeStep
    frame: RinneLegacyFrame


class RinneLegacyBreathRuntime:
    """Stateful automatic-breath adapter over the deterministic frame renderer."""

    def __init__(self, renderer: RinneLegacyFrameRenderer) -> None:
        if not isinstance(renderer, RinneLegacyFrameRenderer):
            raise BinaryBoundsError(
                "breath runtime renderer must be RinneLegacyFrameRenderer"
            )
        self.renderer = renderer
        self._state = self._create_initial_state()
        self._previous_time: int | None = None

    @classmethod
    def from_pck(cls, path: Path | str) -> RinneLegacyBreathRuntime:
        return cls(RinneLegacyFrameRenderer.from_pck(path))

    @property
    def state(self) -> LegacyBreathSchedulerState:
        return self._state

    @property
    def previous_time(self) -> int | None:
        return self._previous_time

    def _create_initial_state(self) -> LegacyBreathSchedulerState:
        config = self.renderer.assets.auto_animation_config
        return create_legacy_breath_scheduler_state(
            enabled=config.breath_enabled,
            duration_factor=config.breath_duration_factor,
        )

    def reset(self) -> None:
        self._state = self._create_initial_state()
        self._previous_time = None

    def advance(
        self,
        current_time: int,
        controls: RinneLegacyFrameControls | None = None,
    ) -> RinneLegacyBreathRuntimeStep:
        base_controls = RinneLegacyFrameControls() if controls is None else controls
        if not isinstance(base_controls, RinneLegacyFrameControls):
            raise BinaryBoundsError(
                "breath runtime controls must be RinneLegacyFrameControls"
            )
        previous_time = (
            current_time if self._previous_time is None else self._previous_time
        )
        scheduler_step = update_legacy_breath_scheduler(
            self._state,
            previous_time=previous_time,
            current_time=current_time,
        )

        expression_count = self.renderer.expression_weight_count
        raw_expression_values = (
            (0.0,) * expression_count
            if base_controls.expression_weights is None
            else base_controls.expression_weights
        )
        if len(raw_expression_values) != expression_count:
            raise BinaryBoundsError(
                "breath runtime expression weight count does not match renderer: "
                f"{len(raw_expression_values)} != {expression_count}"
            )
        expression_weights = apply_breath_expression_gain(
            raw_expression_values,
            scheduler_step.expression_gain,
        )
        resolved_controls = replace(
            base_controls,
            expression_weights=expression_weights,
        )

        self._state = scheduler_step.state
        if self._previous_time is None or current_time > self._previous_time:
            self._previous_time = current_time
        return RinneLegacyBreathRuntimeStep(
            current_time=current_time,
            scheduler_step=scheduler_step,
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
    ) -> RinneLegacyBreathRenderedFrame:
        step = self.advance(current_time, controls)
        frame = self.renderer.render_frame(
            step.controls,
            width=width,
            height=height,
            include_diagnostics=include_diagnostics,
        )
        return RinneLegacyBreathRenderedFrame(step=step, frame=frame)


__all__ = [
    "RinneLegacyBreathRenderedFrame",
    "RinneLegacyBreathRuntime",
    "RinneLegacyBreathRuntimeStep",
    "apply_breath_expression_gain",
]
