from __future__ import annotations

import math
from collections.abc import Sequence


RINNE_SPEECH_EXPRESSION_INDICES = (5, 6, 7)


def apply_speech_expression_gains(
    expression_values: Sequence[float],
    *,
    speech_scale: float,
    speech_gains: Sequence[float],
    expression_indices: Sequence[int] = RINNE_SPEECH_EXPRESSION_INDICES,
) -> tuple[float, ...]:
    """Apply the game-proven three-slot speech composition formula.

    Indices are processed in order, matching RVA 0xDF09..0xDF4B. Repeated
    indices are intentionally not rejected because the original setter checks
    only bounds, not uniqueness.
    """

    if len(speech_gains) != 3:
        raise ValueError("speech gain input must contain exactly three values")
    if len(expression_indices) != 3:
        raise ValueError("speech expression indices must contain exactly three values")
    if not math.isfinite(speech_scale):
        raise ValueError("speech scale must be finite")

    output = [float(value) for value in expression_values]
    if any(not math.isfinite(value) for value in output):
        raise ValueError("expression values must be finite")

    for position, (index, gain) in enumerate(
        zip(expression_indices, speech_gains, strict=True)
    ):
        if not isinstance(index, int):
            raise TypeError(f"speech expression index {position} must be an integer")
        if index < 0 or index >= len(output):
            raise IndexError(
                f"speech expression index {index} outside expression array"
            )
        if not math.isfinite(gain):
            raise ValueError(f"speech gain {position} must be finite")
        output[index] = output[index] * speech_scale + gain

    return tuple(output)


__all__ = [
    "RINNE_SPEECH_EXPRESSION_INDICES",
    "apply_speech_expression_gains",
]
