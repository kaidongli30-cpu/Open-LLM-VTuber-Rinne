from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .amb_runtime import RinneLegacyAmbRuntime, RinneLegacyAmbRuntimeStep
from .blink_runtime import RinneLegacyBlinkRuntime, RinneLegacyBlinkRuntimeStep
from .breath_runtime import RinneLegacyBreathRuntime, RinneLegacyBreathRuntimeStep
from .checked_binary import BinaryBoundsError
from .frame_renderer import (
    RinneLegacyFrame,
    RinneLegacyFrameControls,
    RinneLegacyFrameRenderer,
)


@dataclass(frozen=True)
class RinneLegacyMotionRuntimeStep:
    current_time: int
    breath_step: RinneLegacyBreathRuntimeStep
    amb_step: RinneLegacyAmbRuntimeStep
    blink_step: RinneLegacyBlinkRuntimeStep | None
    controls: RinneLegacyFrameControls | None


@dataclass(frozen=True)
class RinneLegacyMotionRenderedFrame:
    step: RinneLegacyMotionRuntimeStep
    frame: RinneLegacyFrame


class RinneLegacyMotionRuntime:
    """Compose breath, packaged AMB, automatic blink, and live speech.

    Composition follows the old control dependencies: breathing first changes
    its proven expression slot, AMB then applies its baseline gains and pose,
    automatic blink supplies the four delayed eyelid samples, and the frame
    renderer finally adds the independent live mouth gains.
    """

    def __init__(
        self,
        renderer: RinneLegacyFrameRenderer,
        *,
        random_int: Callable[[], int],
        start_time: int = 0,
        loop: bool = True,
        blink_floor: float = 0.0,
    ) -> None:
        if not isinstance(renderer, RinneLegacyFrameRenderer):
            raise BinaryBoundsError(
                "motion runtime renderer must be RinneLegacyFrameRenderer"
            )
        self._checked_time(start_time)
        if not isinstance(loop, bool):
            raise BinaryBoundsError("motion runtime loop flag must be boolean")
        self.renderer = renderer
        self.breath = RinneLegacyBreathRuntime(renderer)
        self.amb = RinneLegacyAmbRuntime(
            renderer,
            start_time=start_time,
            loop=loop,
        )
        self.blink = RinneLegacyBlinkRuntime(
            renderer,
            random_int=random_int,
            floor=blink_floor,
        )

    @staticmethod
    def _checked_time(value: int) -> int:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value > 0xFFFFFFFF
        ):
            raise BinaryBoundsError("motion runtime time must be uint32")
        return value

    @classmethod
    def from_pck(
        cls,
        path: Path | str,
        *,
        random_int: Callable[[], int],
        start_time: int = 0,
        loop: bool = True,
        blink_floor: float = 0.0,
    ) -> RinneLegacyMotionRuntime:
        return cls(
            RinneLegacyFrameRenderer.from_pck(path),
            random_int=random_int,
            start_time=start_time,
            loop=loop,
            blink_floor=blink_floor,
        )

    def reset(self, *, start_time: int, loop: bool | None = None) -> None:
        self._checked_time(start_time)
        if loop is not None and not isinstance(loop, bool):
            raise BinaryBoundsError("motion runtime loop flag must be boolean")
        self.amb.reset(start_time=start_time, loop=loop)
        self.breath.reset()
        self.blink.reset()

    def advance(
        self,
        current_time: int,
        controls: RinneLegacyFrameControls | None = None,
    ) -> RinneLegacyMotionRuntimeStep:
        current_time = self._checked_time(current_time)
        base_controls = RinneLegacyFrameControls() if controls is None else controls
        if not isinstance(base_controls, RinneLegacyFrameControls):
            raise BinaryBoundsError(
                "motion runtime controls must be RinneLegacyFrameControls"
            )

        breath_step = self.breath.advance(current_time, base_controls)
        amb_step = self.amb.advance(current_time, breath_step.controls)
        if amb_step.controls is None:
            return RinneLegacyMotionRuntimeStep(
                current_time=current_time,
                breath_step=breath_step,
                amb_step=amb_step,
                blink_step=None,
                controls=None,
            )
        blink_step = self.blink.advance(current_time, amb_step.controls)
        return RinneLegacyMotionRuntimeStep(
            current_time=current_time,
            breath_step=breath_step,
            amb_step=amb_step,
            blink_step=blink_step,
            controls=blink_step.controls,
        )

    def render_at(
        self,
        current_time: int,
        controls: RinneLegacyFrameControls | None = None,
        *,
        width: int = 512,
        height: int = 512,
        include_diagnostics: bool = False,
    ) -> RinneLegacyMotionRenderedFrame:
        step = self.advance(current_time, controls)
        if step.controls is None:
            raise BinaryBoundsError(
                "packaged AMB has finished; the caller must select a transition"
            )
        frame = self.renderer.render_frame(
            step.controls,
            width=width,
            height=height,
            include_diagnostics=include_diagnostics,
        )
        return RinneLegacyMotionRenderedFrame(step=step, frame=frame)


__all__ = [
    "RinneLegacyMotionRenderedFrame",
    "RinneLegacyMotionRuntime",
    "RinneLegacyMotionRuntimeStep",
]
