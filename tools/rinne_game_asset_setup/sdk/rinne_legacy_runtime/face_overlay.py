from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .checked_binary import BinaryBoundsError
from .expression_morph import (
    apply_v39_expression_morph_xy,
    prepare_v39_expression_weights,
)
from .mpb_v39_atlas import MpbV39AtlasSemantics
from .mpb_v39_renderer import MpbV39RendererViews, map_v39_uv_view
from .neutral_type0 import project_rinne_legacy_vertex
from .open_eye import RinneOpenEyeLayout
from .pupil_deformation import (
    RinneLegacyPupilProfile,
    apply_rinne_legacy_pupil_deformation_xy,
)


RINNE_NEUTRAL_FACE_OVERLAY_TYPE_ID = 0x83
RINNE_NEUTRAL_FACE_OVERLAY_FIXED_RECORD = 0
RINNE_NEUTRAL_FACE_OVERLAY_ATLAS_RECORD = 1


@dataclass(frozen=True)
class RinneNeutralFaceOverlayMesh:
    positions_xyz: tuple[tuple[float, float, float], ...]
    game_uvs: tuple[tuple[float, float], ...]
    triangle_indices: tuple[int, ...]
    opacity: float


def _depths_for_type(renderer: MpbV39RendererViews, type_id: int):
    category = renderer.type_to_depth_category.get(type_id)
    matches = tuple(
        view for group_id, view in renderer.depth_groups if group_id == category
    )
    if category is None or len(matches) != 1:
        raise BinaryBoundsError(f"face-overlay depth for type {type_id} is missing")
    return matches[0]


def build_rinne_neutral_face_overlay_mesh(
    renderer: MpbV39RendererViews,
    atlas: MpbV39AtlasSemantics,
    layout: RinneOpenEyeLayout,
    *,
    opacity: float = 1.0,
    expression_weights: Sequence[float] | None = None,
    pupil_profile: RinneLegacyPupilProfile | None = None,
    right_pupil_position: Sequence[float] = (0.0, 0.0),
    left_pupil_position: Sequence[float] = (0.0, 0.0),
) -> RinneNeutralFaceOverlayMesh:
    """Rebuild the fixed-record-0 draw submitted by RVA 0x181C0.

    On v39 the function selects runtime type 0x83 positions/depths. Its mesh
    buffer was initialized at RVA 0x13320 from fixed-triplet record 0 and
    group-B atlas record 1.
    """

    renderer.validate()
    if not math.isfinite(opacity) or opacity < 0.0 or opacity > 1.0:
        raise BinaryBoundsError("face-overlay opacity must be within 0..1")
    if len(layout.fixed_meshes) != 3:
        raise BinaryBoundsError("face-overlay layout requires three fixed meshes")

    outer_index = renderer.type_to_outer.get(RINNE_NEUTRAL_FACE_OVERLAY_TYPE_ID)
    category = renderer.type_to_morph_category.get(
        RINNE_NEUTRAL_FACE_OVERLAY_TYPE_ID
    )
    if outer_index is None or outer_index < 0 or outer_index >= len(
        renderer.mesh_records
    ):
        raise BinaryBoundsError("face-overlay type 0x83 surface mesh is missing")
    if category is None or category < 0 or category >= len(
        renderer.morph_delta_categories
    ):
        raise BinaryBoundsError("face-overlay type 0x83 morph category is missing")
    surface = renderer.mesh_records[outer_index]
    raw_expression_weights = (
        (0.0,) * renderer.primary_morph_count
        if expression_weights is None
        else expression_weights
    )
    prepared_expression_weights = prepare_v39_expression_weights(
        raw_expression_weights, renderer.expression_ease_flags
    )
    positions_xy = apply_v39_expression_morph_xy(
        surface,
        renderer.morph_delta_categories[category],
        prepared_expression_weights,
    )
    if pupil_profile is not None:
        positions_xy = apply_rinne_legacy_pupil_deformation_xy(
            tuple(surface.coordinate_source_xy.values()),
            positions_xy,
            pupil_profile,
            right_position_xy=right_pupil_position,
            left_position_xy=left_pupil_position,
        )
    elif tuple(right_pupil_position) != (0.0, 0.0) or tuple(
        left_pupil_position
    ) != (0.0, 0.0):
        raise BinaryBoundsError("face-overlay pupil controls require a pupil profile")
    depths = _depths_for_type(renderer, RINNE_NEUTRAL_FACE_OVERLAY_TYPE_ID)
    if len(positions_xy) != len(surface.coordinate_source_xy) or len(depths) != len(
        positions_xy
    ):
        raise BinaryBoundsError("face-overlay position/depth counts do not match")

    fixed_mesh = layout.fixed_meshes[RINNE_NEUTRAL_FACE_OVERLAY_FIXED_RECORD]
    fixed_mesh.validate(surface_vertex_count=len(positions_xy))
    atlas_matches = tuple(
        record
        for record in atlas.atlas_records
        if record.record_id == RINNE_NEUTRAL_FACE_OVERLAY_ATLAS_RECORD
    )
    if len(atlas_matches) != 1:
        raise BinaryBoundsError("face-overlay atlas record 1 must occur exactly once")
    atlas_record = atlas_matches[0]
    atlas_record.validate()

    all_game_uvs = map_v39_uv_view(
        surface.coordinate_source_xy,
        source=atlas_record.source_rectangle,
        target=atlas_record.runtime_atlas_rectangle,
        clamp=True,
    )
    positions: list[tuple[float, float, float]] = []
    game_uvs: list[tuple[float, float]] = []
    for surface_vertex_id in fixed_mesh.surface_vertex_ids.values():
        x, y = positions_xy[surface_vertex_id]
        positions.append(
            project_rinne_legacy_vertex(
                x,
                y,
                depths.at(surface_vertex_id),
                depth_offset=0.0,
            )
        )
        game_uvs.append(all_game_uvs[surface_vertex_id])

    return RinneNeutralFaceOverlayMesh(
        positions_xyz=tuple(positions),
        game_uvs=tuple(game_uvs),
        triangle_indices=tuple(fixed_mesh.triangle_indices.values()),
        opacity=opacity,
    )


__all__ = [
    "RINNE_NEUTRAL_FACE_OVERLAY_ATLAS_RECORD",
    "RINNE_NEUTRAL_FACE_OVERLAY_FIXED_RECORD",
    "RINNE_NEUTRAL_FACE_OVERLAY_TYPE_ID",
    "RinneNeutralFaceOverlayMesh",
    "build_rinne_neutral_face_overlay_mesh",
]
