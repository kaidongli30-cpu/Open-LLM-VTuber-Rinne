from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass

from .checked_binary import BinaryBoundsError
from .mpb_v39_atlas import MpbV39AtlasSemantics
from .mpb_v39_renderer import MpbV39RendererViews, map_v39_uv_view
from .pupil_deformation import (
    RinneLegacyPupilProfile,
    apply_rinne_legacy_pupil_deformation_xy,
)
from .rgba_png import encode_rgba_png
from .special_morph import apply_v39_special_base_morph_xy
from .texture import MotionPortraitTexture
from .textured_mesh import rasterize_textured_mesh_rgba


RINNE_NEUTRAL_TYPE0_ID = 0
RINNE_NEUTRAL_TYPE0_ATLAS_RECORD_ID = 0
RINNE_LEGACY_POSITION_SCALE = 4.0
RINNE_LEGACY_POSITION_BIAS = 2.0
RINNE_LEGACY_DEPTH_BASE = 9.021416664123535
RINNE_LEGACY_PROJECTION_XY_SCALE = 0.5
RINNE_LEGACY_PROJECTION_Z_SCALE = 0.01


def _f32(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError as exc:
        raise BinaryBoundsError("neutral type-0 float32 arithmetic overflow") from exc


@dataclass(frozen=True)
class RinneNeutralType0Mesh:
    """One asset-free description of the neutral type-0 base draw."""

    positions_xyz: tuple[tuple[float, float, float], ...]
    game_uvs: tuple[tuple[float, float], ...]
    triangle_indices: tuple[int, ...]
    opacity: float


def project_rinne_legacy_clip_vertex(
    clip_x: float,
    clip_y: float,
    clip_z: float,
) -> tuple[float, float, float]:
    """Apply the fixed RVA-0x9CCAE projection and CPU viewport mapping."""

    if not all(math.isfinite(value) for value in (clip_x, clip_y, clip_z)):
        raise BinaryBoundsError("legacy clip vertex must be finite")
    projected_x = _f32(_f32(clip_x) * RINNE_LEGACY_PROJECTION_XY_SCALE)
    projected_y = _f32(_f32(clip_y) * RINNE_LEGACY_PROJECTION_XY_SCALE)
    projected_z = _f32(_f32(clip_z) * RINNE_LEGACY_PROJECTION_Z_SCALE)
    normalized_x = _f32(_f32(projected_x + 1.0) * 0.5)
    normalized_y = _f32(_f32(projected_y + 1.0) * 0.5)
    return normalized_x, normalized_y, projected_z


def project_rinne_legacy_vertex(
    x: float,
    y: float,
    depth: float,
    *,
    depth_offset: float,
) -> tuple[float, float, float]:
    """Convert one identity-transform RVA-0x17970 vertex for CPU rasterizing."""

    if not all(math.isfinite(value) for value in (x, y, depth, depth_offset)):
        raise BinaryBoundsError("legacy projected vertex must be finite")
    clip_x = _f32(
        _f32(x * RINNE_LEGACY_POSITION_SCALE) - RINNE_LEGACY_POSITION_BIAS
    )
    clip_y = _f32(
        _f32(y * RINNE_LEGACY_POSITION_SCALE) - RINNE_LEGACY_POSITION_BIAS
    )
    clip_z = _f32(_f32(depth - RINNE_LEGACY_DEPTH_BASE) + depth_offset)
    return project_rinne_legacy_clip_vertex(clip_x, clip_y, clip_z)


def build_rinne_neutral_type0_mesh(
    renderer: MpbV39RendererViews,
    atlas: MpbV39AtlasSemantics,
    *,
    expression_weights: Sequence[float] | None = None,
    pupil_profile: RinneLegacyPupilProfile | None = None,
    right_pupil_position: Sequence[float] = (0.0, 0.0),
    left_pupil_position: Sequence[float] = (0.0, 0.0),
) -> RinneNeutralType0Mesh:
    """Build the identity-transform type-0 base draw.

    Game RVA 0x13199..0x131BA maps type 0 through group-B atlas record 0.
    RVA 0x17B30..0x17B93 expands XY from 0..1 to clip space with
    ``value * 4 - 2`` and combines the type depth with the fixed Z base. The
    returned XY is converted from clip space to the CPU rasterizer's normalized
    bottom-up coordinates. The returned UVs use the game loader's packaged
    atlas-rectangle rewrite and therefore address decoded TEX rows directly.
    """

    renderer.validate()
    type_zero_draws = tuple(
        record
        for record in atlas.draw_records
        if record.type_id == RINNE_NEUTRAL_TYPE0_ID
    )
    if len(type_zero_draws) != 1:
        raise BinaryBoundsError(
            f"neutral type-0 draw count must be one, got {len(type_zero_draws)}"
        )
    draw = type_zero_draws[0]
    if not draw.enabled:
        raise BinaryBoundsError("neutral type-0 draw is not initially enabled")
    if (
        not math.isfinite(draw.initial_opacity)
        or draw.initial_opacity < 0.0
        or draw.initial_opacity > 1.0
    ):
        raise BinaryBoundsError("neutral type-0 opacity must be within 0..1")

    atlas_records = tuple(
        record
        for record in atlas.atlas_records
        if record.record_id == RINNE_NEUTRAL_TYPE0_ATLAS_RECORD_ID
    )
    if len(atlas_records) != 1:
        raise BinaryBoundsError(
            "neutral type-0 atlas record 0 must occur exactly once"
        )
    atlas_record = atlas_records[0]
    atlas_record.validate()

    outer_index = renderer.type_to_outer.get(RINNE_NEUTRAL_TYPE0_ID)
    if outer_index is None or outer_index != draw.outer_index:
        raise BinaryBoundsError(
            "neutral type-0 draw outer mesh does not match renderer mapping"
        )
    if outer_index < 0 or outer_index >= len(renderer.mesh_records):
        raise BinaryBoundsError("neutral type-0 outer mesh is missing")
    mesh = renderer.mesh_records[outer_index]

    depth_category = renderer.type_to_depth_category.get(RINNE_NEUTRAL_TYPE0_ID)
    depth_matches = tuple(
        view for group_id, view in renderer.depth_groups if group_id == depth_category
    )
    if depth_category is None or len(depth_matches) != 1:
        raise BinaryBoundsError("neutral type-0 depth array is missing")
    depths = depth_matches[0]
    if len(depths) != len(mesh.coordinate_source_xy):
        raise BinaryBoundsError(
            "neutral type-0 depth count does not match mesh vertex count"
        )

    raw_weights = (
        (0.0,) * renderer.primary_morph_count
        if expression_weights is None
        else expression_weights
    )
    parameter_positions = apply_v39_special_base_morph_xy(
        renderer, RINNE_NEUTRAL_TYPE0_ID, raw_weights
    )
    if pupil_profile is not None:
        parameter_positions = apply_rinne_legacy_pupil_deformation_xy(
            tuple(mesh.coordinate_source_xy.values()),
            parameter_positions,
            pupil_profile,
            right_position_xy=right_pupil_position,
            left_position_xy=left_pupil_position,
        )
    elif tuple(right_pupil_position) != (0.0, 0.0) or tuple(
        left_pupil_position
    ) != (0.0, 0.0):
        raise BinaryBoundsError("type-0 pupil controls require a pupil profile")
    positions_xyz: list[tuple[float, float, float]] = []
    for index, (x, y) in enumerate(parameter_positions):
        positions_xyz.append(
            project_rinne_legacy_vertex(
                x, y, depths.at(index), depth_offset=0.0
            )
        )

    game_uvs = map_v39_uv_view(
        mesh.coordinate_source_xy,
        source=atlas_record.source_rectangle,
        target=atlas_record.runtime_atlas_rectangle,
        clamp=True,
    )
    return RinneNeutralType0Mesh(
        positions_xyz=tuple(positions_xyz),
        game_uvs=game_uvs,
        triangle_indices=tuple(mesh.triangle_indices.values()),
        opacity=draw.initial_opacity,
    )


def rasterize_rinne_neutral_type0_rgba(
    texture: MotionPortraitTexture,
    mesh: RinneNeutralType0Mesh,
    *,
    width: int,
    height: int,
) -> bytes:
    """Rasterize the type-0 base with the loader-resolved atlas coordinates."""

    return rasterize_textured_mesh_rgba(
        texture,
        tuple((x, y) for x, y, _z in mesh.positions_xyz),
        mesh.game_uvs,
        mesh.triangle_indices,
        width=width,
        height=height,
        opacity=mesh.opacity,
    )


def render_rinne_neutral_type0_preview(
    texture: MotionPortraitTexture,
    mesh: RinneNeutralType0Mesh,
    *,
    width: int,
    height: int,
) -> bytes:
    return encode_rgba_png(
        width,
        height,
        rasterize_rinne_neutral_type0_rgba(
            texture, mesh, width=width, height=height
        ),
    )


__all__ = [
    "RINNE_LEGACY_DEPTH_BASE",
    "RINNE_LEGACY_POSITION_BIAS",
    "RINNE_LEGACY_POSITION_SCALE",
    "RINNE_LEGACY_PROJECTION_XY_SCALE",
    "RINNE_LEGACY_PROJECTION_Z_SCALE",
    "RINNE_NEUTRAL_TYPE0_ATLAS_RECORD_ID",
    "RINNE_NEUTRAL_TYPE0_ID",
    "RinneNeutralType0Mesh",
    "build_rinne_neutral_type0_mesh",
    "project_rinne_legacy_clip_vertex",
    "project_rinne_legacy_vertex",
    "rasterize_rinne_neutral_type0_rgba",
    "render_rinne_neutral_type0_preview",
]
