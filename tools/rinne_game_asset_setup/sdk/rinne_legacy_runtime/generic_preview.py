from __future__ import annotations

from collections.abc import Callable, Sequence

from .checked_binary import BinaryBoundsError
from .expression_morph import (
    apply_v39_expression_morph_xy,
    prepare_v39_expression_weights,
)
from .mpb_v39_atlas import MpbV39AtlasBinding
from .mpb_v39_renderer import (
    MpbV39MeshRecordView,
    MpbV39RendererViews,
    map_v39_uv_view,
)
from .rgba_png import encode_rgba_png
from .texture import MotionPortraitTexture
from .textured_mesh import (
    TEXTURED_MESH_MAX_VERTICES,
    rasterize_textured_mesh_rgba,
)


GENERIC_PREVIEW_MAX_BINDINGS = 256


def _rasterize_atlas_bindings_rgba(
    texture: MotionPortraitTexture,
    meshes: Sequence[MpbV39MeshRecordView],
    bindings: Sequence[MpbV39AtlasBinding],
    *,
    width: int,
    height: int,
    position_resolver: Callable[
        [MpbV39AtlasBinding, MpbV39MeshRecordView],
        Sequence[tuple[float, float]],
    ],
) -> bytes:
    if len(bindings) > GENERIC_PREVIEW_MAX_BINDINGS:
        raise BinaryBoundsError(
            f"generic binding count {len(bindings)} exceeds "
            f"{GENERIC_PREVIEW_MAX_BINDINGS}"
        )

    ordered_bindings = sorted(bindings, key=lambda binding: binding.draw_order)
    if any(binding.draw_order < 0 for binding in ordered_bindings):
        raise BinaryBoundsError("generic binding draw order must not be negative")
    if len({binding.draw_order for binding in ordered_bindings}) != len(
        ordered_bindings
    ):
        raise BinaryBoundsError("generic binding draw order must be unique")

    positions: list[tuple[float, float]] = []
    uvs: list[tuple[float, float]] = []
    triangle_indices: list[int] = []
    for binding in ordered_bindings:
        if binding.type_id <= 2:
            raise BinaryBoundsError(
                f"special draw type {binding.type_id} is not supported by "
                "the generic atlas preview"
            )
        if not binding.initially_enabled or binding.initial_opacity == 0.0:
            continue
        if binding.initial_opacity != 1.0:
            raise BinaryBoundsError(
                f"generic type {binding.type_id}: preview only supports "
                f"initial opacity 1.0, got {binding.initial_opacity}"
            )
        if binding.outer_index < 0 or binding.outer_index >= len(meshes):
            raise BinaryBoundsError(
                f"generic type {binding.type_id}: outer mesh "
                f"{binding.outer_index} is missing"
            )
        mesh = meshes[binding.outer_index]
        mesh.validate()
        binding_positions = tuple(position_resolver(binding, mesh))
        if len(binding_positions) != len(mesh.coordinate_source_xy):
            raise BinaryBoundsError(
                f"generic type {binding.type_id}: resolved position count "
                f"{len(binding_positions)} does not match mesh vertex count "
                f"{len(mesh.coordinate_source_xy)}"
            )
        next_vertex_count = len(positions) + len(binding_positions)
        if next_vertex_count > TEXTURED_MESH_MAX_VERTICES:
            raise BinaryBoundsError(
                "combined generic preview vertex count exceeds safety limit"
            )
        next_index_count = len(triangle_indices) + len(mesh.triangle_indices)
        if next_index_count > TEXTURED_MESH_MAX_VERTICES * 6:
            raise BinaryBoundsError(
                "combined generic preview index count exceeds safety limit"
            )
        vertex_offset = len(positions)
        positions.extend(binding_positions)
        game_uvs = map_v39_uv_view(
            mesh.coordinate_source_xy,
            source=binding.source_rectangle,
            target=binding.runtime_atlas_rectangle,
            clamp=True,
        )
        uvs.extend(game_uvs)
        triangle_indices.extend(
            vertex_offset + index for index in mesh.triangle_indices.values()
        )

    return rasterize_textured_mesh_rgba(
        texture,
        positions,
        uvs,
        triangle_indices,
        width=width,
        height=height,
    )


def rasterize_initial_generic_atlas_preview_rgba(
    texture: MotionPortraitTexture,
    meshes: Sequence[MpbV39MeshRecordView],
    bindings: Sequence[MpbV39AtlasBinding],
    *,
    width: int,
    height: int,
) -> bytes:
    """Rasterize the initially visible generic type>2 layers into RGBA.

    This is a bounded reconstruction gate, not the final character renderer.
    It excludes special types 0 through 2 and runtime morph/blend state.
    """

    return _rasterize_atlas_bindings_rgba(
        texture,
        meshes,
        bindings,
        width=width,
        height=height,
        position_resolver=lambda _binding, mesh: mesh.coordinate_source_xy.values(),
    )


def rasterize_expression_generic_atlas_preview_rgba(
    texture: MotionPortraitTexture,
    renderer: MpbV39RendererViews,
    bindings: Sequence[MpbV39AtlasBinding],
    expression_weights: Sequence[float],
    *,
    width: int,
    height: int,
) -> bytes:
    """Rasterize generic layers after the consumer-proven expression morph."""

    renderer.validate()
    if len(expression_weights) != renderer.primary_morph_count:
        raise BinaryBoundsError(
            f"expression weight count {len(expression_weights)} does not match "
            f"MPB morph count {renderer.primary_morph_count}"
        )
    if len(renderer.expression_ease_flags) != renderer.primary_morph_count:
        raise BinaryBoundsError(
            "renderer does not expose one expression ease flag per morph"
        )
    prepared_weights = prepare_v39_expression_weights(
        expression_weights, renderer.expression_ease_flags
    )
    position_cache: dict[tuple[int, int], tuple[tuple[float, float], ...]] = {}

    def resolve(
        binding: MpbV39AtlasBinding, mesh: MpbV39MeshRecordView
    ) -> tuple[tuple[float, float], ...]:
        expected_outer = renderer.type_to_outer.get(binding.type_id)
        if expected_outer is None or expected_outer != binding.outer_index:
            raise BinaryBoundsError(
                f"generic type {binding.type_id}: binding outer mesh "
                f"{binding.outer_index} does not match renderer mapping "
                f"{expected_outer}"
            )
        category = renderer.type_to_morph_category.get(binding.type_id)
        if category is None or category < 0 or category >= len(
            renderer.morph_delta_categories
        ):
            raise BinaryBoundsError(
                f"generic type {binding.type_id}: morph category is missing"
            )
        cache_key = (mesh.outer_index, category)
        positions = position_cache.get(cache_key)
        if positions is None:
            positions = apply_v39_expression_morph_xy(
                mesh,
                renderer.morph_delta_categories[category],
                prepared_weights,
            )
            position_cache[cache_key] = positions
        return positions

    return _rasterize_atlas_bindings_rgba(
        texture,
        renderer.mesh_records,
        bindings,
        width=width,
        height=height,
        position_resolver=resolve,
    )


def render_initial_generic_atlas_preview(
    texture: MotionPortraitTexture,
    meshes: Sequence[MpbV39MeshRecordView],
    bindings: Sequence[MpbV39AtlasBinding],
    *,
    width: int,
    height: int,
) -> bytes:
    rgba = rasterize_initial_generic_atlas_preview_rgba(
        texture,
        meshes,
        bindings,
        width=width,
        height=height,
    )
    return encode_rgba_png(width, height, rgba)


def render_expression_generic_atlas_preview(
    texture: MotionPortraitTexture,
    renderer: MpbV39RendererViews,
    bindings: Sequence[MpbV39AtlasBinding],
    expression_weights: Sequence[float],
    *,
    width: int,
    height: int,
) -> bytes:
    rgba = rasterize_expression_generic_atlas_preview_rgba(
        texture,
        renderer,
        bindings,
        expression_weights,
        width=width,
        height=height,
    )
    return encode_rgba_png(width, height, rgba)


__all__ = [
    "rasterize_initial_generic_atlas_preview_rgba",
    "rasterize_expression_generic_atlas_preview_rgba",
    "render_expression_generic_atlas_preview",
    "render_initial_generic_atlas_preview",
]
