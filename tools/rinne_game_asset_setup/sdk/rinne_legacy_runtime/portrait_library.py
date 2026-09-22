from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .asset_manifest import (
    RinnePckVerification,
    verify_rinne_pck_directory,
)
from .checked_binary import BinaryBoundsError
from .frame_renderer import RinneLegacyFrameRenderer
from .motion_runtime import RinneLegacyMotionRuntime
from .portrait_catalog import RinnePortraitProfile, select_rinne_portrait


@dataclass(frozen=True)
class RinnePortraitSelection:
    profile: RinnePortraitProfile
    verification: RinnePckVerification
    renderer: RinneLegacyFrameRenderer


class RinnePortraitLibrary:
    """Lazy renderer cache over an already verified local PCK collection."""

    def __init__(
        self,
        verifications: Sequence[RinnePckVerification],
        *,
        cache_capacity: int = 2,
    ) -> None:
        items = tuple(verifications)
        if not items or any(
            not isinstance(item, RinnePckVerification) for item in items
        ):
            raise BinaryBoundsError(
                "portrait library requires verified Rinne PCK records"
            )
        if (
            not isinstance(cache_capacity, int)
            or isinstance(cache_capacity, bool)
            or cache_capacity <= 0
        ):
            raise BinaryBoundsError("portrait cache capacity must be positive")
        by_id = {item.spec.portrait_id: item for item in items}
        if len(by_id) != len(items):
            raise BinaryBoundsError("portrait library contains duplicate ids")
        for item in items:
            if item.path.name.casefold() != item.spec.filename.casefold():
                raise BinaryBoundsError(
                    "verified portrait path does not match its manifest filename"
                )
        self._verifications = by_id
        self.cache_capacity = cache_capacity
        self._renderers: OrderedDict[int, RinneLegacyFrameRenderer] = OrderedDict()

    @classmethod
    def from_directory(
        cls,
        path: Path | str,
        *,
        cache_capacity: int = 2,
    ) -> RinnePortraitLibrary:
        """Hash-check all fifteen originals before exposing any renderer."""

        return cls(
            verify_rinne_pck_directory(path),
            cache_capacity=cache_capacity,
        )

    @property
    def verified_portrait_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._verifications))

    @property
    def cached_portrait_ids(self) -> tuple[int, ...]:
        return tuple(self._renderers)

    def verification(self, portrait_id: int) -> RinnePckVerification:
        try:
            return self._verifications[portrait_id]
        except KeyError as exc:
            raise KeyError(f"Rinne portrait {portrait_id} is not verified") from exc

    def renderer(self, portrait_id: int) -> RinneLegacyFrameRenderer:
        verification = self.verification(portrait_id)
        cached = self._renderers.pop(portrait_id, None)
        if cached is not None:
            self._renderers[portrait_id] = cached
            return cached
        renderer = RinneLegacyFrameRenderer.from_pck(verification.path)
        self._renderers[portrait_id] = renderer
        while len(self._renderers) > self.cache_capacity:
            self._renderers.popitem(last=False)
        return renderer

    def select(self, tag: str, *, variant: int = 0) -> RinnePortraitSelection:
        profile = select_rinne_portrait(tag, variant=variant)
        verification = self.verification(profile.portrait_id)
        return RinnePortraitSelection(
            profile=profile,
            verification=verification,
            renderer=self.renderer(profile.portrait_id),
        )

    def create_motion_runtime(
        self,
        portrait_id: int,
        *,
        random_int: Callable[[], int],
        start_time: int = 0,
        loop: bool = True,
        blink_floor: float = 0.0,
    ) -> RinneLegacyMotionRuntime:
        return RinneLegacyMotionRuntime(
            self.renderer(portrait_id),
            random_int=random_int,
            start_time=start_time,
            loop=loop,
            blink_floor=blink_floor,
        )


__all__ = [
    "RinnePortraitLibrary",
    "RinnePortraitSelection",
]
