from __future__ import annotations

from dataclasses import dataclass


_RINNE_PORTRAIT_FAMILIES = range(601, 605)


def _is_supported_rinne_portrait_id(portrait_id: object) -> bool:
    return (
        isinstance(portrait_id, int)
        and not isinstance(portrait_id, bool)
        and portrait_id // 100 in _RINNE_PORTRAIT_FAMILIES
        and 1 <= portrait_id % 100 <= 15
    )


@dataclass(frozen=True)
class RinnePortraitProfile:
    """User-approved desktop semantic label for one canonical PCK.

    ``compatibility_label`` and ``tags`` are intentionally not presented as
    official game metadata. The numeric portrait ID remains canonical.
    """

    portrait_id: int
    compatibility_label: str
    tags: tuple[str, ...]
    game_usage_count: int
    has_closed_eye_layer: bool = False
    has_sweat_layer: bool = False

    def __post_init__(self) -> None:
        if not _is_supported_rinne_portrait_id(self.portrait_id):
            raise ValueError(
                "Rinne portrait profile id must be a 601xx..604xx family id with suffix 01..15"
            )
        if not self.compatibility_label:
            raise ValueError("Rinne portrait compatibility label is empty")
        if not self.tags or any(not tag or tag != tag.casefold() for tag in self.tags):
            raise ValueError("Rinne portrait tags must be nonempty and case-folded")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("Rinne portrait tags must be unique")
        if (
            not isinstance(self.game_usage_count, int)
            or isinstance(self.game_usage_count, bool)
            or self.game_usage_count < 0
        ):
            raise ValueError("Rinne portrait game usage count must be nonnegative")

    @property
    def filename(self) -> str:
        return f"MP{self.portrait_id:06d}.pck"


RINNE_PORTRAIT_PROFILES = (
    RinnePortraitProfile(60_101, "thinking", ("thinking",), 14),
    RinnePortraitProfile(60_102, "neutral", ("neutral",), 114),
    RinnePortraitProfile(60_103, "confused", ("confused",), 47),
    RinnePortraitProfile(60_104, "gentle", ("gentle",), 61),
    RinnePortraitProfile(
        60_105,
        "awkward",
        ("awkward",),
        41,
        has_closed_eye_layer=True,
    ),
    RinnePortraitProfile(
        60_106,
        "happy",
        ("happy",),
        37,
        has_closed_eye_layer=True,
    ),
    RinnePortraitProfile(60_107, "worried", ("worried",), 26),
    RinnePortraitProfile(
        60_108,
        "surprised",
        ("surprised", "surprise"),
        27,
    ),
    RinnePortraitProfile(
        60_109,
        "dissatisfaction",
        ("dissatisfaction",),
        1,
        has_closed_eye_layer=True,
        has_sweat_layer=True,
    ),
    RinnePortraitProfile(60_110, "flustered", ("flustered",), 10),
    RinnePortraitProfile(60_111, "sad", ("sad",), 15),
    RinnePortraitProfile(
        60_112,
        "embarrassed",
        ("embarrassed",),
        29,
        has_closed_eye_layer=True,
        has_sweat_layer=True,
    ),
    RinnePortraitProfile(60_113, "shy", ("shy",), 13),
    RinnePortraitProfile(
        60_114,
        "uneasy",
        ("uneasy",),
        14,
        has_sweat_layer=True,
    ),
    RinnePortraitProfile(60_115, "angry", ("angry",), 9),
)
_COMPATIBILITY_LABELS = tuple(
    (profile.compatibility_label, profile.tags) for profile in RINNE_PORTRAIT_PROFILES
)
_SECOND_OUTFIT_USAGE_COUNTS = (20, 109, 26, 59, 21, 43, 19, 6, 0, 3, 12, 9, 4, 5, 11)
_THIRD_OUTFIT_USAGE_COUNTS = (10, 29, 9, 11, 11, 29, 13, 3, 1, 2, 10, 0, 7, 2, 4)
_FOURTH_OUTFIT_USAGE_COUNTS = (8, 18, 3, 3, 11, 3, 1, 1, 0, 0, 1, 3, 2, 2, 0)
RINNE_SECOND_OUTFIT_PORTRAIT_PROFILES = tuple(
    RinnePortraitProfile(
        60_200 + suffix,
        label,
        tags,
        _SECOND_OUTFIT_USAGE_COUNTS[suffix - 1],
        has_closed_eye_layer=suffix in {5, 6, 9, 12},
        has_sweat_layer=suffix in {9, 12, 14},
    )
    for suffix, (label, tags) in enumerate(_COMPATIBILITY_LABELS, start=1)
)
RINNE_THIRD_OUTFIT_PORTRAIT_PROFILES = tuple(
    RinnePortraitProfile(
        60_300 + suffix,
        label,
        tags,
        _THIRD_OUTFIT_USAGE_COUNTS[suffix - 1],
        has_closed_eye_layer=suffix in {5, 6, 9, 12},
        has_sweat_layer=suffix in {9, 12, 14},
    )
    for suffix, (label, tags) in enumerate(_COMPATIBILITY_LABELS, start=1)
)
RINNE_FOURTH_OUTFIT_PORTRAIT_PROFILES = tuple(
    RinnePortraitProfile(
        60_400 + suffix,
        label,
        tags,
        _FOURTH_OUTFIT_USAGE_COUNTS[suffix - 1],
        has_closed_eye_layer=suffix in {5, 6, 9, 12},
        has_sweat_layer=suffix in {9, 12, 14},
    )
    for suffix, (label, tags) in enumerate(_COMPATIBILITY_LABELS, start=1)
)
RINNE_OUTFIT_PORTRAIT_PROFILES = {
    1: RINNE_PORTRAIT_PROFILES,
    2: RINNE_SECOND_OUTFIT_PORTRAIT_PROFILES,
    3: RINNE_THIRD_OUTFIT_PORTRAIT_PROFILES,
    4: RINNE_FOURTH_OUTFIT_PORTRAIT_PROFILES,
}
_RINNE_PORTRAIT_PROFILES_BY_ID = {
    profile.portrait_id: profile
    for family in RINNE_OUTFIT_PORTRAIT_PROFILES.values()
    for profile in family
}


def get_rinne_portrait_profile(portrait_id: int) -> RinnePortraitProfile:
    if not isinstance(portrait_id, int) or isinstance(portrait_id, bool):
        raise TypeError("Rinne portrait id must be an integer")
    try:
        return _RINNE_PORTRAIT_PROFILES_BY_ID[portrait_id]
    except KeyError as exc:
        raise KeyError(f"unknown Rinne portrait id: {portrait_id}") from exc


def get_rinne_outfit_portrait_profiles(
    outfit_number: int,
) -> tuple[RinnePortraitProfile, ...]:
    if not isinstance(outfit_number, int) or isinstance(outfit_number, bool):
        raise TypeError("Rinne outfit number must be an integer")
    try:
        return RINNE_OUTFIT_PORTRAIT_PROFILES[outfit_number]
    except KeyError as exc:
        raise KeyError(f"unknown Rinne outfit number: {outfit_number}") from exc


def find_rinne_portraits_by_tag(tag: str) -> tuple[RinnePortraitProfile, ...]:
    if not isinstance(tag, str):
        raise TypeError("Rinne portrait tag must be a string")
    normalized = tag.strip().casefold().replace("-", "_").replace(" ", "_")
    if not normalized:
        raise ValueError("Rinne portrait tag is empty")
    return tuple(
        profile for profile in RINNE_PORTRAIT_PROFILES if normalized in profile.tags
    )


def select_rinne_portrait(tag: str, *, variant: int = 0) -> RinnePortraitProfile:
    matches = find_rinne_portraits_by_tag(tag)
    if not matches:
        raise KeyError(f"no Rinne portrait has compatibility tag: {tag}")
    if not isinstance(variant, int) or isinstance(variant, bool):
        raise TypeError("Rinne portrait variant must be an integer")
    if variant < 0 or variant >= len(matches):
        raise IndexError(
            f"Rinne portrait variant {variant} outside 0..{len(matches) - 1}"
        )
    return matches[variant]


def find_rinne_outfit_portraits_by_tag(
    outfit_number: int,
    tag: str,
) -> tuple[RinnePortraitProfile, ...]:
    if not isinstance(tag, str):
        raise TypeError("Rinne portrait tag must be a string")
    normalized = tag.strip().casefold().replace("-", "_").replace(" ", "_")
    if not normalized:
        raise ValueError("Rinne portrait tag is empty")
    return tuple(
        profile
        for profile in get_rinne_outfit_portrait_profiles(outfit_number)
        if normalized in profile.tags
    )


def select_rinne_outfit_portrait(
    outfit_number: int,
    tag: str,
    *,
    variant: int = 0,
) -> RinnePortraitProfile:
    matches = find_rinne_outfit_portraits_by_tag(outfit_number, tag)
    if not matches:
        raise KeyError(
            f"no Rinne outfit {outfit_number} portrait has compatibility tag: {tag}"
        )
    if not isinstance(variant, int) or isinstance(variant, bool):
        raise TypeError("Rinne portrait variant must be an integer")
    if variant < 0 or variant >= len(matches):
        raise IndexError(
            f"Rinne portrait variant {variant} outside 0..{len(matches) - 1}"
        )
    return matches[variant]


__all__ = [
    "RINNE_PORTRAIT_PROFILES",
    "RINNE_OUTFIT_PORTRAIT_PROFILES",
    "RINNE_SECOND_OUTFIT_PORTRAIT_PROFILES",
    "RINNE_THIRD_OUTFIT_PORTRAIT_PROFILES",
    "RINNE_FOURTH_OUTFIT_PORTRAIT_PROFILES",
    "RinnePortraitProfile",
    "find_rinne_portraits_by_tag",
    "find_rinne_outfit_portraits_by_tag",
    "get_rinne_outfit_portrait_profiles",
    "get_rinne_portrait_profile",
    "select_rinne_portrait",
    "select_rinne_outfit_portrait",
]
