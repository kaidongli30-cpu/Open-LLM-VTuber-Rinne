from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .amb_v4 import AmbV4Animation, AmbV4FrameSample
from .checked_binary import BinaryBoundsError
from .metadata import MotionPortraitLayerName
from .mpb_v39_atlas import MpbV39AtlasSemantics


V39_RENDER_LAYER_NAME_MARKER = 0xFE
V39_SPECIAL_DRAW_TYPE_MAX = 2


@dataclass(frozen=True)
class AmbV4RenderRecordBinding:
    """Consumer-proven link from one AMB value to one runtime draw record."""

    auxiliary_index: int
    channel_index: int
    draw_order: int
    type_id: int
    outer_index: int
    layer_name: str | None
    initially_enabled: bool
    initial_opacity: float

    def value_from(self, sample: AmbV4FrameSample) -> float:
        if (
            self.auxiliary_index < 0
            or self.auxiliary_index >= len(sample.auxiliary)
        ):
            raise ValueError(
                f"AMB sample has no auxiliary value {self.auxiliary_index}"
            )
        return sample.auxiliary[self.auxiliary_index]


def bind_v39_amb_render_records(
    animation: AmbV4Animation,
    atlas: MpbV39AtlasSemantics,
    layer_names: Sequence[MotionPortraitLayerName],
) -> tuple[AmbV4RenderRecordBinding, ...]:
    """Bind v4 auxiliary channels to v39 draw records in proven list order.

    The game copies group-A records into the runtime draw list in file order,
    then the AMB consumer writes one continuous value to each record using the
    same zero-based index.  Types 0 through 2 use unnamed special draw paths;
    ordinary types are checked against the 0xFE layer-name group.
    """

    draw_records = atlas.draw_records
    if animation.auxiliary_channel_count != len(draw_records):
        raise BinaryBoundsError(
            "AMB auxiliary count does not match MPB draw-record count: "
            f"{animation.auxiliary_channel_count} != {len(draw_records)}"
        )
    if tuple(record.file_index for record in draw_records) != tuple(range(len(draw_records))):
        raise BinaryBoundsError("MPB draw records are not in contiguous file order")
    if len({record.type_id for record in draw_records}) != len(draw_records):
        raise BinaryBoundsError("MPB draw-record types are not unique")

    names_by_type: dict[int, MotionPortraitLayerName] = {}
    for layer in layer_names:
        if layer.group_marker != V39_RENDER_LAYER_NAME_MARKER:
            continue
        if layer.layer_id in names_by_type:
            raise BinaryBoundsError(
                f"duplicate 0xFE layer name for draw type {layer.layer_id}"
            )
        names_by_type[layer.layer_id] = layer

    ordinary_types = {
        record.type_id
        for record in draw_records
        if record.type_id > V39_SPECIAL_DRAW_TYPE_MAX
    }
    if set(names_by_type) != ordinary_types:
        missing = sorted(ordinary_types - set(names_by_type))
        extra = sorted(set(names_by_type) - ordinary_types)
        raise BinaryBoundsError(
            "0xFE layer names do not match ordinary draw types: "
            f"missing={missing}, extra={extra}"
        )

    auxiliary_start = animation.channel_groups.auxiliary.start
    return tuple(
        AmbV4RenderRecordBinding(
            auxiliary_index=index,
            channel_index=auxiliary_start + index,
            draw_order=record.file_index,
            type_id=record.type_id,
            outer_index=record.outer_index,
            layer_name=(
                None
                if record.type_id <= V39_SPECIAL_DRAW_TYPE_MAX
                else names_by_type[record.type_id].name
            ),
            initially_enabled=record.enabled,
            initial_opacity=record.initial_opacity,
        )
        for index, record in enumerate(draw_records)
    )


__all__ = [
    "AmbV4RenderRecordBinding",
    "V39_RENDER_LAYER_NAME_MARKER",
    "V39_SPECIAL_DRAW_TYPE_MAX",
    "bind_v39_amb_render_records",
]
