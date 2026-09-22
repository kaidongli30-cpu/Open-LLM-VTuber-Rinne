from __future__ import annotations

from collections.abc import Sequence

from .checked_binary import BinaryBoundsError
from .expression_morph import (
    apply_v39_expression_morph_xy,
    prepare_v39_expression_weights,
)
from .mpb_v39_renderer import MpbV39MeshRecordView, MpbV39RendererViews


V39_BASE_MORPH_SPECIAL_TYPES = (0, 1)


def _special_base_morph_views(
    renderer: MpbV39RendererViews, type_id: int
) -> tuple[MpbV39MeshRecordView, int]:
    renderer.validate()
    if type_id not in V39_BASE_MORPH_SPECIAL_TYPES:
        raise BinaryBoundsError(
            f"special type {type_id} does not use the proven type0/type1 "
            "base-morph path"
        )
    outer_index = renderer.type_to_outer.get(type_id)
    category = renderer.type_to_morph_category.get(type_id)
    if outer_index is None or outer_index < 0 or outer_index >= len(
        renderer.mesh_records
    ):
        raise BinaryBoundsError(f"special type {type_id}: outer mesh is missing")
    if category is None or category < 0 or category >= len(
        renderer.morph_delta_categories
    ):
        raise BinaryBoundsError(
            f"special type {type_id}: morph category is missing"
        )
    mesh = renderer.mesh_records[outer_index]
    if mesh.outer_index != outer_index:
        raise BinaryBoundsError(
            f"special type {type_id}: mesh outer index does not match mapping"
        )
    return mesh, category


def inspect_v39_special_base_morph_slots(
    renderer: MpbV39RendererViews, type_id: int
) -> tuple[int, ...]:
    """Return expression indices with non-zero applied type0/type1 deltas."""

    mesh, category = _special_base_morph_views(renderer, type_id)
    deltas = renderer.morph_delta_categories[category]
    active: list[int] = []
    for expression_index in range(renderer.primary_morph_count):
        if any(
            deltas.at(vertex_id * renderer.primary_morph_count + expression_index)
            != (0.0, 0.0)
            for vertex_id in mesh.morph_vertex_ids.values()
        ):
            active.append(expression_index)
    return tuple(active)


def apply_v39_special_base_morph_xy(
    renderer: MpbV39RendererViews,
    type_id: int,
    raw_expression_weights: Sequence[float],
) -> tuple[tuple[float, float], ...]:
    """Apply the proven shared base morph before the unresolved special draw.

    This function deliberately stops before the type0/type1 procedural grids,
    depth projection, texture selection, and compositor.
    """

    mesh, category = _special_base_morph_views(renderer, type_id)
    if len(raw_expression_weights) != renderer.primary_morph_count:
        raise BinaryBoundsError(
            "special base-morph expression count does not match MPB"
        )
    if len(renderer.expression_ease_flags) != renderer.primary_morph_count:
        raise BinaryBoundsError(
            "renderer does not expose one expression ease flag per morph"
        )
    prepared = prepare_v39_expression_weights(
        raw_expression_weights, renderer.expression_ease_flags
    )
    return apply_v39_expression_morph_xy(
        mesh, renderer.morph_delta_categories[category], prepared
    )


__all__ = [
    "V39_BASE_MORPH_SPECIAL_TYPES",
    "apply_v39_special_base_morph_xy",
    "inspect_v39_special_base_morph_slots",
]
