from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from .checked_binary import BinaryBoundsError
from .mpb_v39_atlas import MpbV39AtlasRecord, MpbV39AtlasSemantics
from .neutral_type0 import (
    RINNE_LEGACY_DEPTH_BASE,
    project_rinne_legacy_clip_vertex,
)
from .open_eye import RinneOpenEyeLayout


RINNE_NEUTRAL_EYE_SPRITE_RECORDS = ((8, 6), (9, 7))
RINNE_NEUTRAL_EYE_SPRITE_TRIANGLES = (0, 2, 1, 1, 2, 3)


def _f32(value: float) -> float:
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise BinaryBoundsError(
            "neutral eye-sprite value is outside float32 range"
        ) from exc
    if not math.isfinite(result):
        raise BinaryBoundsError("neutral eye-sprite value must be finite")
    return result


def _sub(left: float, right: float) -> float:
    return _f32(_f32(left) - _f32(right))


def _add(left: float, right: float) -> float:
    return _f32(_f32(left) + _f32(right))


def _mul(left: float, right: float) -> float:
    return _f32(_f32(left) * _f32(right))


@dataclass(frozen=True)
class RinneNeutralEyeSpriteDraw:
    side: int
    layer_index: int
    atlas_record_id: int
    positions_xyz: tuple[tuple[float, float, float], ...]
    game_uvs: tuple[tuple[float, float], ...]
    triangle_indices: tuple[int, ...]
    opacity: float


def _record_by_id(
    records: tuple[MpbV39AtlasRecord, ...], record_id: int
) -> MpbV39AtlasRecord:
    matches = tuple(record for record in records if record.record_id == record_id)
    if len(matches) != 1:
        raise BinaryBoundsError(
            f"neutral eye-sprite atlas record {record_id} must occur exactly once"
        )
    record = matches[0]
    record.validate()
    target = record.atlas_rectangle
    if (
        target.first_x,
        target.first_y,
        target.second_x,
        target.second_y,
    ) == (-1.0,) * 4:
        raise BinaryBoundsError(
            f"neutral eye-sprite atlas record {record_id} has no packaged target"
        )
    return record


def build_rinne_neutral_eye_sprite_draws(
    atlas: MpbV39AtlasSemantics,
    layout: RinneOpenEyeLayout,
    *,
    opacity: float = 1.0,
) -> tuple[
    RinneNeutralEyeSpriteDraw,
    RinneNeutralEyeSpriteDraw,
    RinneNeutralEyeSpriteDraw,
    RinneNeutralEyeSpriteDraw,
]:
    """Rebuild the packaged neutral eye sprites drawn by RVA 0x16BC0.

    RVA 0x14E40 initializes five one-cell helpers per side. The normal Rinne
    branch later draws helper 0 with records 8/9 and helper 2 with records 6/7;
    records 10/11 have no tex_all target and are not emitted here. Each helper
    uses its group-B source rectangle relative to 0.5 as local geometry, then
    translates by the side center copied to runtime +0x1230/+0x1228.
    """

    if len(layout.centers_xy) != 2:
        raise BinaryBoundsError("neutral eye-sprite layout requires two centers")
    if not math.isfinite(opacity) or opacity < 0.0 or opacity > 1.0:
        raise BinaryBoundsError("neutral eye-sprite opacity must be within 0..1")

    draws: list[RinneNeutralEyeSpriteDraw] = []
    for side, record_ids in enumerate(RINNE_NEUTRAL_EYE_SPRITE_RECORDS):
        center_x, center_y = layout.centers_xy[side]
        if not math.isfinite(center_x) or not math.isfinite(center_y):
            raise BinaryBoundsError("neutral eye-sprite center must be finite")
        center_clip_x = _sub(_mul(center_x, 4.0), 2.0)
        center_clip_y = _sub(_mul(center_y, 4.0), 2.0)
        for layer_index, record_id in enumerate(record_ids):
            record = _record_by_id(atlas.atlas_records, record_id)
            source = record.source_rectangle
            target = record.runtime_atlas_rectangle
            local = (
                (_sub(source.first_x, 0.5), _sub(source.second_y, 0.5)),
                (_sub(source.second_x, 0.5), _sub(source.second_y, 0.5)),
                (_sub(source.first_x, 0.5), _sub(source.first_y, 0.5)),
                (_sub(source.second_x, 0.5), _sub(source.first_y, 0.5)),
            )
            positions = tuple(
                project_rinne_legacy_clip_vertex(
                    _add(x, center_clip_x),
                    _add(y, center_clip_y),
                    _f32(-RINNE_LEGACY_DEPTH_BASE),
                )
                for x, y in local
            )
            game_uvs = (
                (target.first_x, target.second_y),
                (target.second_x, target.second_y),
                (target.first_x, target.first_y),
                (target.second_x, target.first_y),
            )
            draws.append(
                RinneNeutralEyeSpriteDraw(
                    side=side,
                    layer_index=layer_index,
                    atlas_record_id=record_id,
                    positions_xyz=positions,
                    game_uvs=game_uvs,
                    triangle_indices=RINNE_NEUTRAL_EYE_SPRITE_TRIANGLES,
                    opacity=opacity,
                )
            )
    return draws[0], draws[1], draws[2], draws[3]


__all__ = [
    "RINNE_NEUTRAL_EYE_SPRITE_RECORDS",
    "RINNE_NEUTRAL_EYE_SPRITE_TRIANGLES",
    "RinneNeutralEyeSpriteDraw",
    "build_rinne_neutral_eye_sprite_draws",
]
