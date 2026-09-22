from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .checked_binary import BinaryBoundsError
from .expression_morph import (
    apply_v39_expression_morph_xy,
    prepare_v39_expression_weights,
)
from .mpb_v39_atlas import MpbV39AtlasBinding, MpbV39AtlasSemantics
from .mpb_v39_renderer import (
    MpbV39MeshRecordView,
    MpbV39RendererViews,
    compute_v39_draw_depth_offset,
    map_v39_uv_view,
)
from .neutral_type0 import (
    build_rinne_neutral_type0_mesh,
    project_rinne_legacy_vertex,
)
from .pupil_deformation import (
    RinneLegacyPupilProfile,
    apply_rinne_legacy_pupil_deformation_xy,
)
from .rgba_png import encode_rgba_png
from .texture import MotionPortraitTexture
from .textured_mesh import (
    TexturedDepthMesh,
    rasterize_textured_depth_meshes_rgba,
)


RINNE_NEUTRAL_OMITTED_SPECIAL_TYPES = (1, 2)


@dataclass(frozen=True)
class RinneNeutralFrameDraw:
    type_id: int
    source_draw_order: int
    mesh: TexturedDepthMesh


@dataclass(frozen=True)
class RinneNeutralBaseFrame:
    draws: tuple[RinneNeutralFrameDraw, ...]
    omitted_enabled_special_types: tuple[int, ...]


def _depth_view_for_type(renderer: MpbV39RendererViews, type_id: int):
    category = renderer.type_to_depth_category.get(type_id)
    matches = tuple(
        view for group_id, view in renderer.depth_groups if group_id == category
    )
    if category is None or len(matches) != 1:
        raise BinaryBoundsError(f"neutral type {type_id}: depth array is missing")
    return matches[0]


def _generic_binding_for_type(
    atlas: MpbV39AtlasSemantics, type_id: int
) -> MpbV39AtlasBinding:
    matches = tuple(
        binding for binding in atlas.generic_bindings if binding.type_id == type_id
    )
    if len(matches) != 1:
        raise BinaryBoundsError(
            f"neutral type {type_id}: generic atlas binding count must be one"
        )
    return matches[0]


def _neutral_generic_positions(
    renderer: MpbV39RendererViews,
    type_id: int,
    mesh: MpbV39MeshRecordView,
    prepared_expression_weights: Sequence[float],
) -> tuple[tuple[float, float], ...]:
    category = renderer.type_to_morph_category.get(type_id)
    if category is None or category < 0 or category >= len(
        renderer.morph_delta_categories
    ):
        raise BinaryBoundsError(
            f"neutral type {type_id}: morph category is missing"
        )
    return apply_v39_expression_morph_xy(
        mesh,
        renderer.morph_delta_categories[category],
        prepared_expression_weights,
    )


def build_rinne_neutral_base_frame(
    renderer: MpbV39RendererViews,
    atlas: MpbV39AtlasSemantics,
    *,
    expression_weights: Sequence[float] | None = None,
    pupil_profile: RinneLegacyPupilProfile | None = None,
    right_pupil_position: Sequence[float] = (0.0, 0.0),
    left_pupil_position: Sequence[float] = (0.0, 0.0),
) -> RinneNeutralBaseFrame:
    """Build enabled submissions in group-A file order.

    Rinne's populated group-A branch dispatches type 0 from inside the same
    file-order loop as the ordinary draws. Types 1 and 2 use separate
    procedural paths and are reported as omissions instead of being silently
    treated as ordinary atlas meshes.
    """

    renderer.validate()
    if renderer.draw_depth_scale is None:
        raise BinaryBoundsError("neutral frame requires the proven draw depth scale")

    raw_expression_weights = (
        (0.0,) * renderer.primary_morph_count
        if expression_weights is None
        else expression_weights
    )
    prepared_expression_weights = prepare_v39_expression_weights(
        raw_expression_weights, renderer.expression_ease_flags
    )
    type_zero = build_rinne_neutral_type0_mesh(
        renderer,
        atlas,
        expression_weights=raw_expression_weights,
        pupil_profile=pupil_profile,
        right_pupil_position=right_pupil_position,
        left_pupil_position=left_pupil_position,
    )
    type_zero_mesh = TexturedDepthMesh(
        positions_xyz=type_zero.positions_xyz,
        uvs=type_zero.game_uvs,
        triangle_indices=type_zero.triangle_indices,
        opacity=type_zero.opacity,
    )
    draws: list[RinneNeutralFrameDraw] = []
    omitted: list[int] = []

    for draw in sorted(atlas.draw_records, key=lambda record: record.file_index):
        if not draw.enabled or draw.initial_opacity == 0.0:
            continue
        if draw.type_id == 0:
            draws.append(
                RinneNeutralFrameDraw(
                    type_id=0,
                    source_draw_order=draw.file_index,
                    mesh=type_zero_mesh,
                )
            )
            continue
        if draw.type_id in RINNE_NEUTRAL_OMITTED_SPECIAL_TYPES:
            omitted.append(draw.type_id)
            continue
        if draw.type_id < 0:
            raise BinaryBoundsError("neutral frame draw type must not be negative")
        binding = _generic_binding_for_type(atlas, draw.type_id)
        if binding.draw_order != draw.file_index or binding.outer_index != draw.outer_index:
            raise BinaryBoundsError(
                f"neutral type {draw.type_id}: atlas binding does not match draw record"
            )
        outer_index = renderer.type_to_outer.get(draw.type_id)
        if outer_index is None or outer_index != draw.outer_index:
            raise BinaryBoundsError(
                f"neutral type {draw.type_id}: renderer outer mapping mismatch"
            )
        if outer_index < 0 or outer_index >= len(renderer.mesh_records):
            raise BinaryBoundsError(
                f"neutral type {draw.type_id}: outer mesh is missing"
            )
        mesh = renderer.mesh_records[outer_index]
        parameter_positions = _neutral_generic_positions(
            renderer,
            draw.type_id,
            mesh,
            prepared_expression_weights,
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
            raise BinaryBoundsError(
                f"neutral type {draw.type_id}: pupil controls require a profile"
            )
        depths = _depth_view_for_type(renderer, draw.type_id)
        if len(depths) != len(parameter_positions):
            raise BinaryBoundsError(
                f"neutral type {draw.type_id}: depth count mismatch"
            )
        depth_offset = compute_v39_draw_depth_offset(
            draw.file_index, renderer.draw_depth_scale
        )
        positions_xyz = tuple(
            project_rinne_legacy_vertex(
                x, y, depths.at(index), depth_offset=depth_offset
            )
            for index, (x, y) in enumerate(parameter_positions)
        )
        game_uvs = map_v39_uv_view(
            mesh.coordinate_source_xy,
            source=binding.source_rectangle,
            target=binding.runtime_atlas_rectangle,
            clamp=True,
        )
        draws.append(
            RinneNeutralFrameDraw(
                type_id=draw.type_id,
                source_draw_order=draw.file_index,
                mesh=TexturedDepthMesh(
                    positions_xyz=positions_xyz,
                    uvs=game_uvs,
                    triangle_indices=tuple(mesh.triangle_indices.values()),
                    opacity=draw.initial_opacity,
                ),
            )
        )

    if sum(draw.type_id == 0 for draw in draws) != 1:
        raise BinaryBoundsError("neutral frame requires one enabled type-0 draw")

    return RinneNeutralBaseFrame(
        draws=tuple(draws),
        omitted_enabled_special_types=tuple(sorted(set(omitted))),
    )


def rasterize_rinne_neutral_base_frame_rgba(
    texture: MotionPortraitTexture,
    frame: RinneNeutralBaseFrame,
    *,
    width: int,
    height: int,
    depth_compare: Literal["disabled", "less_equal", "greater_equal"],
) -> bytes:
    return rasterize_textured_depth_meshes_rgba(
        texture,
        tuple(draw.mesh for draw in frame.draws),
        width=width,
        height=height,
        depth_compare=depth_compare,
    )


def render_rinne_neutral_base_frame_preview(
    texture: MotionPortraitTexture,
    frame: RinneNeutralBaseFrame,
    *,
    width: int,
    height: int,
    depth_compare: Literal["disabled", "less_equal", "greater_equal"],
) -> bytes:
    return encode_rgba_png(
        width,
        height,
        rasterize_rinne_neutral_base_frame_rgba(
            texture,
            frame,
            width=width,
            height=height,
            depth_compare=depth_compare,
        ),
    )


__all__ = [
    "RINNE_NEUTRAL_OMITTED_SPECIAL_TYPES",
    "RinneNeutralBaseFrame",
    "RinneNeutralFrameDraw",
    "build_rinne_neutral_base_frame",
    "rasterize_rinne_neutral_base_frame_rgba",
    "render_rinne_neutral_base_frame_preview",
]
