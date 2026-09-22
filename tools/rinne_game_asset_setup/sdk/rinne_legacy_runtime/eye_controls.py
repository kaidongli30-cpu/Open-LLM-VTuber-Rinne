from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from .amb_v4 import AmbV4FrameSample
from .checked_binary import BinaryBoundsError


RINNE_RIGHT_EYE_CLOSE_CORE_INDEX = 10
RINNE_LEFT_EYE_CLOSE_CORE_INDEX = 11


def _f32(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise BinaryBoundsError("eye-close value is outside float32 range") from exc


def _clamp_eye_close(value: float, *, side: str) -> float:
    if not math.isfinite(value):
        raise BinaryBoundsError(f"{side} eye-close value must be finite")
    value = _f32(value)
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


@dataclass(frozen=True)
class RinneEyeCloseSample:
    """Consumer-proven bilateral eye-close controls; 0=open and 1=closed."""

    right: float
    left: float


def extract_rinne_eye_close(
    frame: AmbV4FrameSample,
) -> RinneEyeCloseSample:
    """Extract and clamp core[10]/core[11] like the game's eye-close setter.

    The special-eye draw bridge consumes these clamped values as the default
    diffuse alpha for the paired eyelid-overlay meshes.
    """

    if len(frame.core) <= RINNE_LEFT_EYE_CLOSE_CORE_INDEX:
        raise BinaryBoundsError("AMB frame does not expose 12 core controls")
    return RinneEyeCloseSample(
        right=_clamp_eye_close(
            frame.core[RINNE_RIGHT_EYE_CLOSE_CORE_INDEX], side="right"
        ),
        left=_clamp_eye_close(
            frame.core[RINNE_LEFT_EYE_CLOSE_CORE_INDEX], side="left"
        ),
    )


__all__ = [
    "RINNE_LEFT_EYE_CLOSE_CORE_INDEX",
    "RINNE_RIGHT_EYE_CLOSE_CORE_INDEX",
    "RinneEyeCloseSample",
    "extract_rinne_eye_close",
]
