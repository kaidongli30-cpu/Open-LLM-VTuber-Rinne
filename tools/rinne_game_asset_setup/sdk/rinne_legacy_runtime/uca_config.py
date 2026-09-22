from __future__ import annotations

import math
import struct

from .amb_v4 import AmbV4Config
from .checked_binary import BinaryBoundsError, CheckedBinary


FACE_UCA_CONFIG_SIZE = 0x54
FACE_UCA_FORCED_BREATH_ENABLE = 1


def parse_face_uca_config(
    data: bytes | bytearray | memoryview,
) -> AmbV4Config:
    """Decode the big-endian face.uca.bin block loaded at RVA 0x99DD1."""

    stable_data = data if isinstance(data, bytes) else bytes(data)
    reader = CheckedBinary(stable_data)
    reader.span("face UCA config", 0, FACE_UCA_CONFIG_SIZE)
    if reader.size != FACE_UCA_CONFIG_SIZE:
        raise BinaryBoundsError(
            f"face UCA config must be exactly {FACE_UCA_CONFIG_SIZE} bytes"
        )

    def integer(offset: int) -> int:
        return struct.unpack_from(">I", stable_data, offset)[0]

    def scalar(offset: int) -> float:
        value = struct.unpack_from(">f", stable_data, offset)[0]
        if not math.isfinite(value):
            raise BinaryBoundsError(
                f"face UCA config offset 0x{offset:x} is non-finite"
            )
        return value

    return AmbV4Config(
        neck_enabled=(integer(0x00), integer(0x04), integer(0x08)),
        neck_duration_factors=(scalar(0x0C), scalar(0x10), scalar(0x14)),
        neck_max_rotations=(scalar(0x18), scalar(0x1C), scalar(0x20)),
        blink_enabled=integer(0x24),
        blink_duration_factor=scalar(0x28),
        blink_frequencies=(integer(0x2C), integer(0x30), integer(0x34)),
        pupil_enabled=integer(0x38),
        pupil_duration_factor=scalar(0x3C),
        pupil_x_max=scalar(0x40),
        pupil_y_max=scalar(0x44),
        expression_enabled=integer(0x48),
        breath_enabled=FACE_UCA_FORCED_BREATH_ENABLE,
        expression_duration_factor=scalar(0x4C),
        breath_duration_factor=scalar(0x50),
    )


__all__ = [
    "FACE_UCA_CONFIG_SIZE",
    "FACE_UCA_FORCED_BREATH_ENABLE",
    "parse_face_uca_config",
]
