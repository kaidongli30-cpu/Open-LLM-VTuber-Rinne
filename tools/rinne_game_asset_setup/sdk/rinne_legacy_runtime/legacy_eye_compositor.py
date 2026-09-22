from __future__ import annotations

from .checked_binary import BinaryBoundsError
from .rgba_png import PNG_MAX_DIMENSION, PNG_MAX_PIXELS


def _checked_rgba_length(
    rgba: bytes | bytearray | memoryview,
    *,
    width: int,
    height: int,
    name: str,
) -> memoryview:
    if (
        width <= 0
        or height <= 0
        or width > PNG_MAX_DIMENSION
        or height > PNG_MAX_DIMENSION
        or width * height > PNG_MAX_PIXELS
    ):
        raise BinaryBoundsError("RGBA compositor dimensions exceed safety limit")
    view = memoryview(rgba).cast("B")
    expected = width * height * 4
    if len(view) != expected:
        raise BinaryBoundsError(f"{name} RGBA length {len(view)} != {expected}")
    return view


def combine_legacy_eye_buffers_rgba(
    eye_rgba: bytes | bytearray | memoryview,
    mask_rgba: bytes | bytearray | memoryview,
    *,
    width: int,
    height: int,
) -> bytes:
    """Reproduce the old renderer's embedded two-texture eye pixel shader.

    Pixel shader 16 samples the mask from texture 1 and the eye color from
    texture 0. Its output RGB is the eye RGB; output alpha is
    ``eye_alpha * (1 - mask_alpha)``. Both inputs and the result are RGBA8.
    """

    eye = _checked_rgba_length(
        eye_rgba, width=width, height=height, name="eye buffer"
    )
    mask = _checked_rgba_length(
        mask_rgba, width=width, height=height, name="mask buffer"
    )
    output = bytearray(len(eye))
    for offset in range(0, len(output), 4):
        output[offset : offset + 3] = eye[offset : offset + 3]
        output[offset + 3] = (
            eye[offset + 3] * (255 - mask[offset + 3]) + 127
        ) // 255
    return bytes(output)


def source_over_rgba_buffers(
    destination_rgba: bytes | bytearray | memoryview,
    source_rgba: bytes | bytearray | memoryview,
    *,
    width: int,
    height: int,
) -> bytes:
    """Composite one straight-alpha RGBA8 buffer over another."""

    destination = _checked_rgba_length(
        destination_rgba, width=width, height=height, name="destination"
    )
    source = _checked_rgba_length(
        source_rgba, width=width, height=height, name="source"
    )
    output = bytearray(destination)
    for offset in range(0, len(output), 4):
        source_alpha = source[offset + 3]
        if source_alpha == 0:
            continue
        destination_alpha = output[offset + 3]
        inverse_source_alpha = 255 - source_alpha
        output_alpha_scaled = (
            source_alpha * 255 + destination_alpha * inverse_source_alpha
        )
        if output_alpha_scaled == 0:
            continue
        for channel in range(3):
            numerator = (
                source[offset + channel] * source_alpha * 255
                + output[offset + channel]
                * destination_alpha
                * inverse_source_alpha
            )
            output[offset + channel] = (
                numerator + output_alpha_scaled // 2
            ) // output_alpha_scaled
        output[offset + 3] = (output_alpha_scaled + 127) // 255
    return bytes(output)


def blend_legacy_mode0_rgba_buffers(
    destination_rgba: bytes | bytearray | memoryview,
    source_rgba: bytes | bytearray | memoryview,
    *,
    width: int,
    height: int,
) -> bytes:
    """Apply the old renderer's blend mode 0 at an RGBA8 boundary.

    RVA ``0x9C7C0`` selects source-alpha/inverse-source-alpha for RGB, but
    one/one for alpha. The blend operation is ADD for both channels. Unlike
    conventional straight-alpha source-over, RGB is not divided by the output
    alpha and destination alpha does not attenuate destination RGB.
    """

    destination = _checked_rgba_length(
        destination_rgba, width=width, height=height, name="destination"
    )
    source = _checked_rgba_length(
        source_rgba, width=width, height=height, name="source"
    )
    output = bytearray(destination)
    for offset in range(0, len(output), 4):
        source_alpha = source[offset + 3]
        if source_alpha == 0:
            continue
        inverse_source_alpha = 255 - source_alpha
        for channel in range(3):
            output[offset + channel] = (
                source[offset + channel] * source_alpha
                + output[offset + channel] * inverse_source_alpha
                + 127
            ) // 255
        output[offset + 3] = min(255, source_alpha + output[offset + 3])
    return bytes(output)


__all__ = [
    "blend_legacy_mode0_rgba_buffers",
    "combine_legacy_eye_buffers_rgba",
    "source_over_rgba_buffers",
]
