from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .checked_binary import BinaryBoundsError

RINNE_WEBGL_MAX_DIFFERENT_PIXEL_RATIO = 0.01
RINNE_WEBGL_MAX_MEAN_ABSOLUTE_ERROR = 0.1
RINNE_WEBGL_MAX_ALPHA_DIFFERENT_RATIO = 0.001
RINNE_WEBGL_MAX_ALPHA_MASK_DIAGNOSTIC_SAMPLES = 32


@dataclass(frozen=True)
class RinneWebGlFrameComparison:
    width: int
    height: int
    cpu_sha256: str
    webgl_sha256: str
    byte_exact: bool
    different_bytes: int
    different_pixels: int
    different_pixel_ratio: float
    mean_absolute_error: float
    maximum_absolute_error: int
    alpha_different_pixels: int
    alpha_different_ratio: float
    alpha_mask_symmetric_difference: int
    alpha_mask_difference_samples: tuple[tuple[int, int, int, int], ...]
    cpu_nonzero_alpha: int
    webgl_nonzero_alpha: int
    accepted: bool

    def summary_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "cpu_sha256": self.cpu_sha256,
            "webgl_sha256": self.webgl_sha256,
            "byte_exact": self.byte_exact,
            "different_bytes": self.different_bytes,
            "different_pixels": self.different_pixels,
            "different_pixel_ratio": self.different_pixel_ratio,
            "mean_absolute_error": self.mean_absolute_error,
            "maximum_absolute_error": self.maximum_absolute_error,
            "alpha_different_pixels": self.alpha_different_pixels,
            "alpha_different_ratio": self.alpha_different_ratio,
            "alpha_mask_symmetric_difference": (self.alpha_mask_symmetric_difference),
            "alpha_mask_difference_samples": [
                {
                    "x": x,
                    "y": y,
                    "cpu_alpha": cpu_alpha,
                    "webgl_alpha": webgl_alpha,
                }
                for x, y, cpu_alpha, webgl_alpha in self.alpha_mask_difference_samples
            ],
            "cpu_nonzero_alpha": self.cpu_nonzero_alpha,
            "webgl_nonzero_alpha": self.webgl_nonzero_alpha,
            "limits": {
                "maximum_different_pixel_ratio": (
                    RINNE_WEBGL_MAX_DIFFERENT_PIXEL_RATIO
                ),
                "maximum_mean_absolute_error": (RINNE_WEBGL_MAX_MEAN_ABSOLUTE_ERROR),
                "maximum_alpha_different_ratio": (
                    RINNE_WEBGL_MAX_ALPHA_DIFFERENT_RATIO
                ),
                "required_alpha_mask_symmetric_difference": 0,
            },
            "accepted": self.accepted,
        }


def compare_rinne_webgl_frame(
    cpu_rgba: bytes,
    webgl_rgba: bytes,
    *,
    width: int,
    height: int,
) -> RinneWebGlFrameComparison:
    if width <= 0 or height <= 0:
        raise BinaryBoundsError("WebGL comparison dimensions must be positive")
    expected = width * height * 4
    if len(cpu_rgba) != expected or len(webgl_rgba) != expected:
        raise BinaryBoundsError(
            "WebGL comparison RGBA length does not match dimensions"
        )
    absolute_differences = tuple(
        abs(cpu_value - webgl_value)
        for cpu_value, webgl_value in zip(cpu_rgba, webgl_rgba, strict=True)
    )
    pixel_count = width * height
    different_pixels = 0
    alpha_different_pixels = 0
    alpha_mask_symmetric_difference = 0
    alpha_mask_difference_samples: list[tuple[int, int, int, int]] = []
    cpu_nonzero_alpha = 0
    webgl_nonzero_alpha = 0
    for pixel in range(pixel_count):
        offset = pixel * 4
        if cpu_rgba[offset : offset + 4] != webgl_rgba[offset : offset + 4]:
            different_pixels += 1
        cpu_alpha = cpu_rgba[offset + 3]
        webgl_alpha = webgl_rgba[offset + 3]
        if cpu_alpha != webgl_alpha:
            alpha_different_pixels += 1
        cpu_visible = cpu_alpha != 0
        webgl_visible = webgl_alpha != 0
        cpu_nonzero_alpha += int(cpu_visible)
        webgl_nonzero_alpha += int(webgl_visible)
        alpha_mask_symmetric_difference += int(cpu_visible != webgl_visible)
        if (
            cpu_visible != webgl_visible
            and len(alpha_mask_difference_samples)
            < RINNE_WEBGL_MAX_ALPHA_MASK_DIAGNOSTIC_SAMPLES
        ):
            alpha_mask_difference_samples.append(
                (pixel % width, pixel // width, cpu_alpha, webgl_alpha)
            )
    different_pixel_ratio = different_pixels / pixel_count
    mean_absolute_error = sum(absolute_differences) / expected
    alpha_different_ratio = alpha_different_pixels / pixel_count
    accepted = (
        alpha_mask_symmetric_difference == 0
        and different_pixel_ratio <= RINNE_WEBGL_MAX_DIFFERENT_PIXEL_RATIO
        and mean_absolute_error <= RINNE_WEBGL_MAX_MEAN_ABSOLUTE_ERROR
        and alpha_different_ratio <= RINNE_WEBGL_MAX_ALPHA_DIFFERENT_RATIO
    )
    return RinneWebGlFrameComparison(
        width=width,
        height=height,
        cpu_sha256=hashlib.sha256(cpu_rgba).hexdigest(),
        webgl_sha256=hashlib.sha256(webgl_rgba).hexdigest(),
        byte_exact=cpu_rgba == webgl_rgba,
        different_bytes=sum(value != 0 for value in absolute_differences),
        different_pixels=different_pixels,
        different_pixel_ratio=different_pixel_ratio,
        mean_absolute_error=mean_absolute_error,
        maximum_absolute_error=max(absolute_differences, default=0),
        alpha_different_pixels=alpha_different_pixels,
        alpha_different_ratio=alpha_different_ratio,
        alpha_mask_symmetric_difference=alpha_mask_symmetric_difference,
        alpha_mask_difference_samples=tuple(alpha_mask_difference_samples),
        cpu_nonzero_alpha=cpu_nonzero_alpha,
        webgl_nonzero_alpha=webgl_nonzero_alpha,
        accepted=accepted,
    )


def build_rinne_webgl_difference_rgba(
    cpu_rgba: bytes,
    webgl_rgba: bytes,
    *,
    width: int,
    height: int,
    amplification: int = 4,
) -> bytes:
    expected = width * height * 4
    if len(cpu_rgba) != expected or len(webgl_rgba) != expected:
        raise BinaryBoundsError(
            "WebGL difference RGBA length does not match dimensions"
        )
    if not isinstance(amplification, int) or not 1 <= amplification <= 255:
        raise BinaryBoundsError("WebGL difference amplification must be 1..255")
    output = bytearray(expected)
    for offset in range(0, expected, 4):
        changed = False
        for channel in range(3):
            difference = abs(cpu_rgba[offset + channel] - webgl_rgba[offset + channel])
            output[offset + channel] = min(255, difference * amplification)
            changed = changed or difference != 0
        alpha_difference = abs(cpu_rgba[offset + 3] - webgl_rgba[offset + 3])
        output[offset + 3] = 255 if changed or alpha_difference else 0
    return bytes(output)


__all__ = [
    "RINNE_WEBGL_MAX_ALPHA_DIFFERENT_RATIO",
    "RINNE_WEBGL_MAX_ALPHA_MASK_DIAGNOSTIC_SAMPLES",
    "RINNE_WEBGL_MAX_DIFFERENT_PIXEL_RATIO",
    "RINNE_WEBGL_MAX_MEAN_ABSOLUTE_ERROR",
    "RinneWebGlFrameComparison",
    "build_rinne_webgl_difference_rgba",
    "compare_rinne_webgl_frame",
]
