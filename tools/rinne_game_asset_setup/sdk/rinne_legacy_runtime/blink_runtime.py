from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from .blink_scheduler import (
    LegacyBlinkSchedulerState,
    LegacyBlinkSchedulerStep,
    create_legacy_blink_scheduler_state,
    update_legacy_blink_scheduler,
)
from .checked_binary import BinaryBoundsError
from .frame_renderer import (
    RinneLegacyFrame,
    RinneLegacyFrameControls,
    RinneLegacyFrameRenderer,
)


@dataclass(frozen=True)
class RinneLegacyBlinkRuntimeStep:
    current_time: int
    scheduler_step: LegacyBlinkSchedulerStep
    controls: RinneLegacyFrameControls


@dataclass(frozen=True)
class RinneLegacyBlinkRenderedFrame:
    step: RinneLegacyBlinkRuntimeStep
    frame: RinneLegacyFrame


class RinneLegacyBlinkRuntime:
    """Stateful automatic-blink adapter over the deterministic frame renderer."""

    def __init__(
        self,
        renderer: RinneLegacyFrameRenderer,
        *,
        random_int: Callable[[], int],
        floor: float = 0.0,
    ) -> None:
        if not isinstance(renderer, RinneLegacyFrameRenderer):
            raise BinaryBoundsError(
                "blink runtime renderer must be RinneLegacyFrameRenderer"
            )
        if not callable(random_int):
            raise BinaryBoundsError("blink runtime random source must be callable")
        if not math.isfinite(floor):
            raise BinaryBoundsError("blink runtime floor must be finite")
        self.renderer = renderer
        self.random_int = random_int
        self.floor = float(floor)
        self._state = self._create_initial_state()
        self._previous_time: int | None = None

    @classmethod
    def from_pck(
        cls,
        path: Path | str,
        *,
        random_int: Callable[[], int],
        floor: float = 0.0,
    ) -> RinneLegacyBlinkRuntime:
        return cls(
            RinneLegacyFrameRenderer.from_pck(path),
            random_int=random_int,
            floor=floor,
        )

    @property
    def state(self) -> LegacyBlinkSchedulerState:
        return self._state

    @property
    def previous_time(self) -> int | None:
        return self._previous_time

    def _create_initial_state(self) -> LegacyBlinkSchedulerState:
        config = self.renderer.assets.auto_animation_config
        return create_legacy_blink_scheduler_state(
            enabled=config.blink_enabled,
            duration_factor=config.blink_duration_factor,
            frequencies=config.blink_frequencies,
        )

    def reset(self) -> None:
        self._state = self._create_initial_state()
        self._previous_time = None

    def advance(
        self,
        current_time: int,
        controls: RinneLegacyFrameControls | None = None,
    ) -> RinneLegacyBlinkRuntimeStep:
        base_controls = RinneLegacyFrameControls() if controls is None else controls
        if not isinstance(base_controls, RinneLegacyFrameControls):
            raise BinaryBoundsError(
                "blink runtime controls must be RinneLegacyFrameControls"
            )
        previous_time = (
            current_time if self._previous_time is None else self._previous_time
        )
        scheduler_step = update_legacy_blink_scheduler(
            self._state,
            previous_time=previous_time,
            current_time=current_time,
            floor=self.floor,
            random_int=self.random_int,
        )
        resolved_controls = replace(
            base_controls,
            blink_geometry=scheduler_step.state.geometry,
        )
        self._state = scheduler_step.state
        self._previous_time = current_time
        return RinneLegacyBlinkRuntimeStep(
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
    ) -> RinneLegacyBlinkRenderedFrame:
        step = self.advance(current_time, controls)
        frame = self.renderer.render_frame(
            step.controls,
            width=width,
            height=height,
            include_diagnostics=include_diagnostics,
        )
        return RinneLegacyBlinkRenderedFrame(step=step, frame=frame)


__all__ = [
    "RinneLegacyBlinkRenderedFrame",
    "RinneLegacyBlinkRuntime",
    "RinneLegacyBlinkRuntimeStep",
]
