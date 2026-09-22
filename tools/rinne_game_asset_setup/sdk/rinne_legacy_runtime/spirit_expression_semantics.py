from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpiritExpressionSemantic:
    """Final visual contract for one user-approved spirit-dress label."""

    label: str
    category: str
    native_base_portrait_id: int
    required_cues: tuple[str, ...]
    discriminator: str
    mouth_gain_percent: int = 100
    mouth_override_portrait_id: int | None = None
    overlay_id: str | None = None

    def __post_init__(self) -> None:
        if not self.label or self.label != self.label.casefold():
            raise ValueError("spirit expression label must be nonempty and case-folded")
        if not self.category:
            raise ValueError("spirit expression category is empty")
        if not 160_101 <= self.native_base_portrait_id <= 160_107:
            raise ValueError("spirit expression base must be MP160101..MP160107")
        if not self.required_cues or any(not cue for cue in self.required_cues):
            raise ValueError("spirit expression visible cues are empty")
        if not self.discriminator:
            raise ValueError("spirit expression discriminator is empty")
        if self.mouth_gain_percent not in {68, 100}:
            raise ValueError("spirit expression mouth gain must be 68 or 100")
        if (
            self.mouth_override_portrait_id is not None
            and not 160_101 <= self.mouth_override_portrait_id <= 160_107
        ):
            raise ValueError("spirit expression mouth override is outside spirit family")
        if self.overlay_id not in {
            None,
            "awkward-embarrassed-sweat",
            "flustered-blush",
            "shy-blush",
        }:
            raise ValueError("spirit expression overlay is outside approved cues")


SPIRIT_EXPRESSION_SEMANTICS = (
    SpiritExpressionSemantic("thinking", "cognition", 160_104,
        ("quiet open eyes", "small pensive mouth"),
        "user-approved untouched MP160104"),
    SpiritExpressionSemantic("neutral", "calm-positive", 160_101,
        ("relaxed open eyes", "small neutral mouth"),
        "user-approved untouched MP160101", 68),
    SpiritExpressionSemantic("confused", "cognition", 160_102,
        ("mildly uncertain eyes", "restrained downward mouth"),
        "intentionally shares MP160102 with worried and uneasy"),
    SpiritExpressionSemantic("gentle", "calm-positive", 160_106,
        ("relaxed closed eyes", "subtle native smile"),
        "approved closed-eye smile", 100, 160_107),
    SpiritExpressionSemantic("awkward", "self-conscious", 160_106,
        ("closed eyes", "subtle native smile", "original sweat cue"),
        "approved to share one visual with embarrassed", 100, 160_107,
        "awkward-embarrassed-sweat"),
    SpiritExpressionSemantic("happy", "calm-positive", 160_101,
        ("relaxed open eyes", "subtle native smile"),
        "approved open-eye smile", 68, 160_107),
    SpiritExpressionSemantic("worried", "distress", 160_102,
        ("mildly uncertain eyes", "restrained downward mouth"),
        "user-approved untouched MP160102"),
    SpiritExpressionSemantic("surprised", "high-arousal", 160_105,
        ("alert open eyes", "tense small mouth"),
        "user-approved untouched MP160105"),
    SpiritExpressionSemantic("dissatisfaction", "distress", 160_103,
        ("serious open eyes", "tense downward mouth"),
        "approved to share one visual with angry"),
    SpiritExpressionSemantic("flustered", "high-arousal", 160_105,
        ("alert open eyes", "natural medium blush", "tense small mouth"),
        "approved surprised base with MP060110-derived blush", 100, None,
        "flustered-blush"),
    SpiritExpressionSemantic("sad", "distress", 160_106,
        ("closed downcast eyes", "small downward mouth"),
        "user-approved untouched MP160106"),
    SpiritExpressionSemantic("embarrassed", "self-conscious", 160_106,
        ("closed eyes", "subtle native smile", "original sweat cue"),
        "approved to share one visual with awkward", 100, 160_107,
        "awkward-embarrassed-sweat"),
    SpiritExpressionSemantic("shy", "self-conscious", 160_101,
        ("calm open eyes", "subtle native smile", "strong natural blush"),
        "approved red-face smile without the original pout", 100, 160_107,
        "shy-blush"),
    SpiritExpressionSemantic("uneasy", "distress", 160_102,
        ("mildly uncertain eyes", "restrained downward mouth"),
        "approved to share MP160102 with worried and confused"),
    SpiritExpressionSemantic("angry", "distress", 160_103,
        ("serious open eyes", "tense downward mouth"),
        "user-approved untouched MP160103"),
)

SPIRIT_EXPRESSION_LABELS = tuple(item.label for item in SPIRIT_EXPRESSION_SEMANTICS)
SPIRIT_EXPRESSION_SEMANTICS_BY_LABEL = {
    item.label: item for item in SPIRIT_EXPRESSION_SEMANTICS
}

__all__ = [
    "SPIRIT_EXPRESSION_LABELS",
    "SPIRIT_EXPRESSION_SEMANTICS",
    "SPIRIT_EXPRESSION_SEMANTICS_BY_LABEL",
    "SpiritExpressionSemantic",
]
