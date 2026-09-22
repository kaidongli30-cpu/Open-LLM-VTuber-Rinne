from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from .amb_v4 import AmbV4Animation, AmbV4Config, parse_amb_v4
from .blink_curve import (
    LEGACY_BLINK_BASE_WEIGHTS,
    LegacyBlinkGeometrySample,
    sample_legacy_blink_geometry,
)
from .checked_binary import BinaryBoundsError
from .eye_sprites import build_rinne_neutral_eye_sprite_draws
from .face_overlay import build_rinne_neutral_face_overlay_mesh
from .legacy_eye_compositor import (
    blend_legacy_mode0_rgba_buffers,
    combine_legacy_eye_buffers_rgba,
)
from .mpb_v39_atlas import MpbV39AtlasSemantics, parse_v39_atlas_semantics
from .mpb_v39_renderer import MpbV39RendererViews, build_v39_renderer_views
from .neck_pose import (
    RinneLegacyNeckProfile,
    parse_v39_rinne_neck_profile,
    transform_rinne_legacy_meshes,
)
from .neutral_frame import build_rinne_neutral_base_frame
from .neutral_type0 import project_rinne_legacy_vertex
from .open_eye import (
    RinneOpenEyeLayout,
    build_rinne_neutral_open_eye_meshes,
    parse_v39_rinne_open_eye_layout,
)
from .pck import PckArchive
from .pupil_deformation import (
    RinneLegacyPupilProfile,
    apply_rinne_legacy_pupil_deformation_xy,
    parse_v39_rinne_pupil_profile,
)
from .rgba_png import encode_rgba_png
from .speech import apply_speech_expression_gains
from .special_eye import (
    RinneSpecialEyeDrawState,
    RinneSpecialEyeProfile,
    build_rinne_special_eye_draw_mesh,
    parse_v39_rinne_special_eye_profile,
    resolve_rinne_legacy_blur_eye_draw_states,
    resolve_rinne_special_eye_draw_state,
)
from .special_morph import apply_v39_special_base_morph_xy
from .texture import MotionPortraitTexture, decode_motionportrait_texture
from .textured_mesh import TexturedDepthMesh, rasterize_textured_depth_meshes_rgba
from .type2_eye_strip import build_rinne_type2_eye_strip_mesh
from .uca_config import parse_face_uca_config


RINNE_NEUTRAL_BLINK_GEOMETRY = LegacyBlinkGeometrySample(
    deformation=(0.0, 0.0, 0.0, 0.0),
    base_weights=LEGACY_BLINK_BASE_WEIGHTS,
)


def _validate_finite_sequence(
    name: str,
    values: Sequence[float],
    *,
    count: int,
) -> tuple[float, ...]:
    if len(values) != count:
        raise BinaryBoundsError(f"{name} must contain exactly {count} values")
    resolved = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in resolved):
        raise BinaryBoundsError(f"{name} values must be finite")
    return resolved


def _validate_blink_geometry(
    geometry: LegacyBlinkGeometrySample,
) -> LegacyBlinkGeometrySample:
    if not isinstance(geometry, LegacyBlinkGeometrySample):
        raise BinaryBoundsError("blink geometry must be a LegacyBlinkGeometrySample")
    deformation = _validate_finite_sequence(
        "blink deformation", geometry.deformation, count=4
    )
    base_weights = _validate_finite_sequence(
        "blink base weights", geometry.base_weights, count=4
    )
    return LegacyBlinkGeometrySample(
        deformation=deformation,  # type: ignore[arg-type]
        base_weights=base_weights,  # type: ignore[arg-type]
    )


def _f32(value: float) -> float:
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise BinaryBoundsError("frame opacity is outside float32 range") from exc
    if not math.isfinite(result):
        raise BinaryBoundsError("frame opacity must be finite")
    return result


def _type0_effective_opacity(runtime_opacity: float) -> float:
    """Apply the type-0 record's executable-proven fourth-power curve."""

    value = _f32(runtime_opacity)
    squared = _f32(value * value)
    cubed = _f32(squared * value)
    return _f32(cubed * value)


@dataclass(frozen=True)
class RinneLegacyFrameControls:
    """One deterministic frame's public animation inputs.

    ``blink_geometry`` contains the old renderer's four delayed samples.
    ``right_eye_close`` and ``left_eye_close`` optionally override the older
    shared ``eye_close`` input without breaking existing callers.
    ``neck_rotation``, ``neck_translation``, and the bilateral pupil positions
    preserve the old direct-animation API's core pose values.  They are
    accepted here before the corresponding geometry stages so timed AMB
    playback does not discard consumer-proven channels.
    ``mouth_gains`` are the three independently proven speech inputs mapped to
    expression slots 5, 6, and 7.  Ordinary expression weights remain separate
    so speech can be composed over an emotion without destroying it.
    """

    blink_geometry: LegacyBlinkGeometrySample = RINNE_NEUTRAL_BLINK_GEOMETRY
    eye_close: float = 0.0
    right_eye_close: float | None = None
    left_eye_close: float | None = None
    neck_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    neck_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_pupil_position: tuple[float, float] = (0.0, 0.0)
    left_pupil_position: tuple[float, float] = (0.0, 0.0)
    type2_intensity: float = 1.0
    render_record_opacities: tuple[float, ...] | None = None
    render_resource_selectors: tuple[int, ...] | None = None
    expression_weights: tuple[float, ...] | None = None
    mouth_scale: float = 1.0
    mouth_gains: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def timed_blink(
        cls,
        blink_type: int,
        *,
        current_elapsed: int,
        previous_elapsed: int,
        single_blink_gain: float = 1.0,
        floor: float = 0.0,
        eye_close: float = 0.0,
        right_eye_close: float | None = None,
        left_eye_close: float | None = None,
        neck_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        neck_translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        right_pupil_position: tuple[float, float] = (0.0, 0.0),
        left_pupil_position: tuple[float, float] = (0.0, 0.0),
        type2_intensity: float = 1.0,
        render_record_opacities: tuple[float, ...] | None = None,
        render_resource_selectors: tuple[int, ...] | None = None,
        expression_weights: tuple[float, ...] | None = None,
        mouth_scale: float = 1.0,
        mouth_gains: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> RinneLegacyFrameControls:
        geometry = sample_legacy_blink_geometry(
            blink_type,
            current_elapsed=current_elapsed,
            previous_elapsed=previous_elapsed,
            single_blink_gain=single_blink_gain,
            floor=floor,
        )
        return cls(
            blink_geometry=geometry,
            eye_close=eye_close,
            right_eye_close=right_eye_close,
            left_eye_close=left_eye_close,
            neck_rotation=neck_rotation,
            neck_translation=neck_translation,
            right_pupil_position=right_pupil_position,
            left_pupil_position=left_pupil_position,
            type2_intensity=type2_intensity,
            render_record_opacities=render_record_opacities,
            render_resource_selectors=render_resource_selectors,
            expression_weights=expression_weights,
            mouth_scale=mouth_scale,
            mouth_gains=mouth_gains,
        )

    def resolve_expression_weights(self, count: int) -> tuple[float, ...]:
        if count <= 0:
            raise BinaryBoundsError("expression weight count must be positive")
        raw = (
            (0.0,) * count
            if self.expression_weights is None
            else _validate_finite_sequence(
                "expression weights", self.expression_weights, count=count
            )
        )
        if not math.isfinite(self.mouth_scale):
            raise BinaryBoundsError("mouth scale must be finite")
        mouth_gains = _validate_finite_sequence(
            "mouth gains", self.mouth_gains, count=3
        )
        try:
            return apply_speech_expression_gains(
                raw,
                speech_scale=self.mouth_scale,
                speech_gains=mouth_gains,
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise BinaryBoundsError(f"invalid mouth controls: {exc}") from exc

    def validated_blink_geometry(self) -> LegacyBlinkGeometrySample:
        self.resolve_eye_closes()
        self.resolve_core_pose()
        self.resolve_render_resource_selectors()
        if not math.isfinite(self.type2_intensity):
            raise BinaryBoundsError("type-2 intensity must be finite")
        return _validate_blink_geometry(self.blink_geometry)

    def resolve_render_record_opacities(
        self, initial_opacities: Sequence[float]
    ) -> tuple[float, ...]:
        """Return one checked live opacity for every MPB draw record."""

        initial = tuple(initial_opacities)
        if not initial:
            raise BinaryBoundsError("render record opacity list must not be empty")
        raw = (
            initial
            if self.render_record_opacities is None
            else self.render_record_opacities
        )
        values = _validate_finite_sequence(
            "render record opacities", raw, count=len(initial)
        )
        if any(value < 0.0 or value > 1.0 for value in values):
            raise BinaryBoundsError("render record opacities must be within 0..1")
        return tuple(_f32(value) for value in values)

    def resolve_render_resource_selectors(self) -> tuple[int, ...]:
        """Validate the ten resource selectors used by the Rinne renderer.

        All fifteen target PCKs keep these selector words at zero for every
        frame. Zero retains each MPB's constructor-selected default resource.
        A nonzero value is rejected until its alternate resource table is
        implemented instead of being silently rendered with the wrong mesh.
        """

        raw = (
            (0,) * 10
            if self.render_resource_selectors is None
            else self.render_resource_selectors
        )
        if len(raw) != 10:
            raise BinaryBoundsError(
                "render resource selectors must contain exactly 10 values"
            )
        values: list[int] = []
        for value in raw:
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                or value > 0xFFFFFFFF
            ):
                raise BinaryBoundsError("render resource selectors must be uint32")
            values.append(value)
        if any(values):
            raise BinaryBoundsError(
                "nonzero render resource selectors are not supported by the "
                "Rinne default-resource path"
            )
        return tuple(values)

    def resolve_eye_closes(self) -> tuple[float, float]:
        """Return right/left values, falling back to the legacy shared input."""

        if not math.isfinite(self.eye_close):
            raise BinaryBoundsError("eye close input must be finite")
        resolved: list[float] = []
        for label, value in (
            ("right", self.right_eye_close),
            ("left", self.left_eye_close),
        ):
            candidate = self.eye_close if value is None else value
            if not math.isfinite(candidate):
                raise BinaryBoundsError(f"{label} eye close input must be finite")
            resolved.append(float(candidate))
        return resolved[0], resolved[1]

    def resolve_core_pose(
        self,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float],
        tuple[float, float],
    ]:
        """Return finite neck and bilateral pupil controls in legacy order."""

        return (
            _validate_finite_sequence(
                "neck rotation", self.neck_rotation, count=3
            ),
            _validate_finite_sequence(
                "neck translation", self.neck_translation, count=3
            ),
            _validate_finite_sequence(
                "right pupil position", self.right_pupil_position, count=2
            ),
            _validate_finite_sequence(
                "left pupil position", self.left_pupil_position, count=2
            ),
        )


@dataclass(frozen=True)
class RinneLegacyAssets:
    """Parsed resources cached from one read-only Rinne PCK."""

    pck_path: Path
    mpb: bytes
    texture: MotionPortraitTexture
    renderer: MpbV39RendererViews
    atlas: MpbV39AtlasSemantics
    open_eye_layout: RinneOpenEyeLayout
    special_eye_profile: RinneSpecialEyeProfile
    pupil_profile: RinneLegacyPupilProfile
    neck_profile: RinneLegacyNeckProfile
    auto_animation_config: AmbV4Config
    animation: AmbV4Animation
    compositor_trigger_type: int
    eye_sprite_meshes: tuple[TexturedDepthMesh, ...]
    mpb_sha256: str
    texture_container_sha256: str
    auto_animation_config_sha256: str
    animation_sha256: str

    @classmethod
    def from_pck(cls, path: Path | str) -> RinneLegacyAssets:
        pck_path = Path(path).resolve()
        archive = PckArchive(pck_path)
        mpb = archive.read("face.mpb")
        texture_container = archive.read("tex_all.tex")
        auto_animation_container = archive.read("face.uca.bin")
        animation_container = archive.read("001.amb")
        texture = decode_motionportrait_texture(texture_container)
        auto_animation_config = parse_face_uca_config(auto_animation_container)
        animation = parse_amb_v4(animation_container)
        renderer = build_v39_renderer_views(mpb)
        atlas = parse_v39_atlas_semantics(mpb)
        triggers = atlas.compositor_triggers
        # The four MP06 outfit families schedule both delayed compositor paths
        # after the same draw type.  MP1601 spirit-dress portraits retain the
        # two original, distinct trigger positions.  This renderer's recovered
        # eye pass is a combined final overlay, so submit it at the later B
        # trigger; matching-trigger portraits keep their established behavior.
        compositor_trigger_type = triggers.b_type_id
        layout = parse_v39_rinne_open_eye_layout(
            mpb,
            surface_vertex_count=len(
                renderer.mesh_records[0].coordinate_source_xy
            ),
        )
        profile = parse_v39_rinne_special_eye_profile(mpb)
        pupil_profile = parse_v39_rinne_pupil_profile(mpb)
        neck_profile = parse_v39_rinne_neck_profile(mpb)
        eye_sprite_meshes = tuple(
            TexturedDepthMesh(
                positions_xyz=draw.positions_xyz,
                uvs=draw.game_uvs,
                triangle_indices=draw.triangle_indices,
                opacity=draw.opacity,
            )
            for draw in build_rinne_neutral_eye_sprite_draws(atlas, layout)
        )
        return cls(
            pck_path=pck_path,
            mpb=mpb,
            texture=texture,
            renderer=renderer,
            atlas=atlas,
            open_eye_layout=layout,
            special_eye_profile=profile,
            pupil_profile=pupil_profile,
            neck_profile=neck_profile,
            auto_animation_config=auto_animation_config,
            animation=animation,
            compositor_trigger_type=compositor_trigger_type,
            eye_sprite_meshes=eye_sprite_meshes,
            mpb_sha256=hashlib.sha256(mpb).hexdigest(),
            texture_container_sha256=hashlib.sha256(texture_container).hexdigest(),
            auto_animation_config_sha256=hashlib.sha256(
                auto_animation_container
            ).hexdigest(),
            animation_sha256=hashlib.sha256(animation_container).hexdigest(),
        )


@dataclass(frozen=True)
class RinneLegacyFrame:
    width: int
    height: int
    rgba: bytes
    blink_geometry: LegacyBlinkGeometrySample
    expression_weights: tuple[float, ...]
    compositor_trigger_type: int
    diagnostics: tuple[tuple[str, bytes], ...] = ()

    def __post_init__(self) -> None:
        expected = self.width * self.height * 4
        if self.width <= 0 or self.height <= 0 or len(self.rgba) != expected:
            raise BinaryBoundsError("frame RGBA dimensions or payload are invalid")
        names: set[str] = set()
        for name, rgba in self.diagnostics:
            if not name or name in names:
                raise BinaryBoundsError("diagnostic buffer names must be unique")
            if len(rgba) != expected:
                raise BinaryBoundsError(
                    f"diagnostic RGBA length for {name!r} does not match frame"
                )
            names.add(name)

    def png_bytes(self) -> bytes:
        return encode_rgba_png(self.width, self.height, self.rgba)

    def diagnostic_rgba(self, name: str) -> bytes:
        for diagnostic_name, rgba in self.diagnostics:
            if diagnostic_name == name:
                return rgba
        raise KeyError(f"diagnostic buffer not found: {name}")


@dataclass(frozen=True)
class RinneLegacyDrawPlan:
    """GPU-ready mesh batches and compositor passes for one resolved frame."""

    width: int
    height: int
    texture: MotionPortraitTexture
    blink_geometry: LegacyBlinkGeometrySample
    expression_weights: tuple[float, ...]
    compositor_trigger_type: int
    eye_buffer_meshes: tuple[TexturedDepthMesh, ...]
    main_eye_surface_meshes: tuple[TexturedDepthMesh, ...]
    face_overlay_meshes: tuple[TexturedDepthMesh, ...]
    special_mask_meshes: tuple[TexturedDepthMesh, ...]
    mask_buffer_meshes: tuple[TexturedDepthMesh, ...]
    type2_eye_strip_meshes: tuple[TexturedDepthMesh, ...]
    before_trigger_meshes: tuple[TexturedDepthMesh, ...]
    final_eye_overlay_meshes: tuple[TexturedDepthMesh, ...]
    after_trigger_meshes: tuple[TexturedDepthMesh, ...]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise BinaryBoundsError("draw-plan dimensions must be positive")
        if not isinstance(self.texture, MotionPortraitTexture):
            raise BinaryBoundsError("draw plan requires a decoded texture")
        if not isinstance(self.blink_geometry, LegacyBlinkGeometrySample):
            raise BinaryBoundsError("draw plan requires checked blink geometry")
        if not self.expression_weights:
            raise BinaryBoundsError("draw plan requires expression weights")


def _rasterize(
    texture: MotionPortraitTexture,
    meshes: Sequence[TexturedDepthMesh],
    *,
    width: int,
    height: int,
) -> bytes:
    return rasterize_textured_depth_meshes_rgba(
        texture,
        tuple(meshes),
        width=width,
        height=height,
        depth_compare="disabled",
        color_blend="legacy_mode0",
    )


def _build_target1_mask_meshes(
    type_zero_mesh: TexturedDepthMesh,
    mask_meshes: tuple[TexturedDepthMesh, ...],
) -> tuple[TexturedDepthMesh, ...]:
    """Reproduce the target-1 linear/clamp sampler reuse."""

    return (replace(type_zero_mesh, texture_filter="linear"), *mask_meshes)


def rasterize_rinne_legacy_draw_plan(
    plan: RinneLegacyDrawPlan,
    *,
    include_diagnostics: bool = False,
) -> RinneLegacyFrame:
    """Execute a resolved draw plan through the deterministic CPU reference."""

    if not isinstance(plan, RinneLegacyDrawPlan):
        raise BinaryBoundsError("draw-plan rasterizer requires RinneLegacyDrawPlan")
    texture = plan.texture
    width = plan.width
    height = plan.height
    eye_buffer = _rasterize(
        texture,
        plan.eye_buffer_meshes,
        width=width,
        height=height,
    )
    mask_buffer = _rasterize(
        texture,
        plan.mask_buffer_meshes,
        width=width,
        height=height,
    )
    clipped_eye = combine_legacy_eye_buffers_rgba(
        eye_buffer,
        mask_buffer,
        width=width,
        height=height,
    )
    before_rgba = _rasterize(
        texture,
        plan.before_trigger_meshes,
        width=width,
        height=height,
    )
    after_rgba = _rasterize(
        texture,
        plan.after_trigger_meshes,
        width=width,
        height=height,
    )
    final_eye_rgba = _rasterize(
        texture,
        plan.final_eye_overlay_meshes,
        width=width,
        height=height,
    )
    rgba = blend_legacy_mode0_rgba_buffers(
        before_rgba,
        clipped_eye,
        width=width,
        height=height,
    )
    rgba = blend_legacy_mode0_rgba_buffers(
        rgba,
        final_eye_rgba,
        width=width,
        height=height,
    )
    rgba = blend_legacy_mode0_rgba_buffers(
        rgba,
        after_rgba,
        width=width,
        height=height,
    )

    diagnostics: tuple[tuple[str, bytes], ...] = ()
    if include_diagnostics:
        diagnostics = (
            ("01_eye_buffer.png", eye_buffer),
            (
                "02_main_eye_surface.png",
                _rasterize(
                    texture,
                    plan.main_eye_surface_meshes,
                    width=width,
                    height=height,
                ),
            ),
            (
                "03_face_overlay.png",
                _rasterize(
                    texture,
                    plan.face_overlay_meshes,
                    width=width,
                    height=height,
                ),
            ),
            (
                "04_special_mask_only.png",
                _rasterize(
                    texture,
                    plan.special_mask_meshes,
                    width=width,
                    height=height,
                ),
            ),
            ("05_mask_buffer.png", mask_buffer),
            ("06_shader16_clipped_eye.png", clipped_eye),
            (
                "07_type2_eye_strip_only.png",
                _rasterize(
                    texture,
                    plan.type2_eye_strip_meshes,
                    width=width,
                    height=height,
                ),
            ),
            ("08_before_trigger.png", before_rgba),
            ("09_final_eye_overlay.png", final_eye_rgba),
            ("10_after_trigger.png", after_rgba),
        )
    return RinneLegacyFrame(
        width=width,
        height=height,
        rgba=rgba,
        blink_geometry=plan.blink_geometry,
        expression_weights=plan.expression_weights,
        compositor_trigger_type=plan.compositor_trigger_type,
        diagnostics=diagnostics,
    )


def _resolve_target1_special_eye_states(
    *,
    deformation_a: float,
    deformation_c: float | None = None,
    right_eye_close: float | None = None,
    left_eye_close: float | None = None,
    runtime_opacity: float = 1.0,
) -> tuple[RinneSpecialEyeDrawState, RinneSpecialEyeDrawState]:
    if right_eye_close is None:
        right_eye_close = deformation_c
    if left_eye_close is None:
        left_eye_close = deformation_c
    if right_eye_close is None or left_eye_close is None:
        raise BinaryBoundsError("target-1 special eyes require eye-close inputs")
    eye_closes = (right_eye_close, left_eye_close)
    return tuple(
        resolve_rinne_special_eye_draw_state(
            side=side,
            draw_mode=3,
            deformation_a=deformation_a,
            deformation_c=eye_closes[side],
            base_alpha=1.0,
            runtime_opacity=runtime_opacity,
        )
        for side in (0, 1)
    )  # type: ignore[return-value]


@dataclass(frozen=True)
class RinneLegacyFrameRenderer:
    assets: RinneLegacyAssets

    @classmethod
    def from_pck(cls, path: Path | str) -> RinneLegacyFrameRenderer:
        return cls(RinneLegacyAssets.from_pck(path))

    @property
    def expression_weight_count(self) -> int:
        return self.assets.renderer.primary_morph_count

    def render_rgba(
        self,
        controls: RinneLegacyFrameControls | None = None,
        *,
        width: int = 512,
        height: int = 512,
    ) -> bytes:
        """Render one transparent RGBA8 frame without filesystem output."""

        return self.render_frame(
            controls,
            width=width,
            height=height,
            include_diagnostics=False,
        ).rgba

    def _surface_xyz(
        self,
        expression_weights: tuple[float, ...],
        *,
        right_pupil_position: tuple[float, float],
        left_pupil_position: tuple[float, float],
    ) -> tuple[tuple[float, float, float], ...]:
        renderer = self.assets.renderer
        surface_xy = apply_v39_special_base_morph_xy(
            renderer, 1, expression_weights
        )
        surface_mesh = renderer.mesh_records[renderer.type_to_outer[1]]
        surface_xy = apply_rinne_legacy_pupil_deformation_xy(
            tuple(surface_mesh.coordinate_source_xy.values()),
            surface_xy,
            self.assets.pupil_profile,
            right_position_xy=right_pupil_position,
            left_position_xy=left_pupil_position,
        )
        depth_category = renderer.type_to_depth_category[0]
        depth_matches = tuple(
            view
            for group_id, view in renderer.depth_groups
            if group_id == depth_category
        )
        if len(depth_matches) != 1:
            raise BinaryBoundsError("neutral special-eye depth array is missing")
        depths = depth_matches[0]
        return tuple(
            (x, y, depths.at(index))
            for index, (x, y) in enumerate(surface_xy)
        )

    def build_surface_positions_xyz(
        self,
        controls: RinneLegacyFrameControls | None = None,
    ) -> tuple[tuple[float, float, float], ...]:
        """Build the current unposed type-1 face surface for GPU helpers.

        This bounded API avoids rebuilding the complete draw submission when
        a resident GPU runtime only needs the expression/pupil surface used by
        the nonlinear eyelid projection. Blink and eye-close values deform the
        projected helper meshes later; neck pose remains a final global stage.
        """

        controls = RinneLegacyFrameControls() if controls is None else controls
        if not isinstance(controls, RinneLegacyFrameControls):
            raise BinaryBoundsError(
                "surface positions require RinneLegacyFrameControls"
            )
        expression_weights = controls.resolve_expression_weights(
            self.expression_weight_count
        )
        (
            _neck_rotation,
            _neck_translation,
            right_pupil_position,
            left_pupil_position,
        ) = controls.resolve_core_pose()
        return self._surface_xyz(
            expression_weights,
            right_pupil_position=right_pupil_position,
            left_pupil_position=left_pupil_position,
        )

    def _special_eye_meshes_from_states(
        self,
        states: Sequence[RinneSpecialEyeDrawState],
        *,
        surface_xyz: tuple[tuple[float, float, float], ...],
    ) -> tuple[TexturedDepthMesh, ...]:
        renderer = self.assets.renderer
        surface_mesh = renderer.mesh_records[renderer.type_to_outer[1]]
        meshes: list[TexturedDepthMesh] = []
        for state in states:
            draw = build_rinne_special_eye_draw_mesh(
                self.assets.special_eye_profile,
                self.assets.atlas.atlas_records,
                surface_mesh,
                surface_xyz,
                state,
            )
            meshes.append(
                TexturedDepthMesh(
                    positions_xyz=tuple(
                        project_rinne_legacy_vertex(x, y, z, depth_offset=0.0)
                        for x, y, z in draw.positions_xyz
                    ),
                    uvs=draw.uvs,
                    triangle_indices=draw.triangle_indices,
                    opacity=draw.state.opacity,
                    vertex_opacities=draw.vertex_opacities,
                    texture_filter="linear",
                )
            )
        return tuple(meshes)

    def build_draw_plan(
        self,
        controls: RinneLegacyFrameControls | None = None,
        *,
        width: int = 512,
        height: int = 512,
    ) -> RinneLegacyDrawPlan:
        assets = self.assets
        controls = RinneLegacyFrameControls() if controls is None else controls
        if not isinstance(controls, RinneLegacyFrameControls):
            raise BinaryBoundsError("frame controls must be RinneLegacyFrameControls")
        geometry = controls.validated_blink_geometry()
        right_eye_close, left_eye_close = controls.resolve_eye_closes()
        (
            neck_rotation,
            neck_translation,
            right_pupil_position,
            left_pupil_position,
        ) = controls.resolve_core_pose()
        eye_closes = (right_eye_close, left_eye_close)
        expression_weights = controls.resolve_expression_weights(
            assets.renderer.primary_morph_count
        )
        draw_records = tuple(
            sorted(assets.atlas.draw_records, key=lambda item: item.file_index)
        )
        if tuple(record.file_index for record in draw_records) != tuple(
            range(len(draw_records))
        ):
            raise BinaryBoundsError("draw records must have contiguous file indices")
        render_record_opacities = controls.resolve_render_record_opacities(
            tuple(record.initial_opacity for record in draw_records)
        )
        records_by_type = {record.type_id: record for record in draw_records}
        if len(records_by_type) != len(draw_records):
            raise BinaryBoundsError("draw record types must be unique")
        try:
            type0_record = records_by_type[0]
            type1_record = records_by_type[1]
            type2_record = records_by_type[2]
        except KeyError as exc:
            raise BinaryBoundsError(
                "special draw records 0, 1, and 2 are required"
            ) from exc
        type0_opacity = _type0_effective_opacity(
            render_record_opacities[type0_record.file_index]
        )
        type1_opacity = render_record_opacities[type1_record.file_index]
        type2_opacity = render_record_opacities[type2_record.file_index]
        primary_deformation = geometry.deformation[0]
        primary_base_alpha = geometry.base_weights[0]
        surface_xyz = self._surface_xyz(
            expression_weights,
            right_pupil_position=right_pupil_position,
            left_pupil_position=left_pupil_position,
        )

        open_eye_meshes = tuple(
            TexturedDepthMesh(
                positions_xyz=draw.positions_xyz,
                uvs=draw.game_uvs,
                triangle_indices=draw.triangle_indices,
                opacity=type1_opacity,
                texture_filter=draw.texture_filter,
            )
            for draw in build_rinne_neutral_open_eye_meshes(
                assets.renderer,
                assets.atlas,
                assets.open_eye_layout,
                expression_weights=expression_weights,
                pupil_profile=assets.pupil_profile,
                right_pupil_position=right_pupil_position,
                left_pupil_position=left_pupil_position,
            )
        )
        main_eye_surface_meshes = self._special_eye_meshes_from_states(
            resolve_rinne_legacy_blur_eye_draw_states(
                geometry,
                geometry,
                right_eye_close=right_eye_close,
                left_eye_close=left_eye_close,
                runtime_opacity=type0_opacity,
            ),
            surface_xyz=surface_xyz,
        )
        mask_meshes = self._special_eye_meshes_from_states(
            _resolve_target1_special_eye_states(
                deformation_a=primary_deformation,
                right_eye_close=right_eye_close,
                left_eye_close=left_eye_close,
                runtime_opacity=type0_opacity,
            ),
            surface_xyz=surface_xyz,
        )
        final_eye_meshes = self._special_eye_meshes_from_states(
            tuple(
                resolve_rinne_special_eye_draw_state(
                    side=side,
                    draw_mode=0,
                    deformation_a=primary_deformation,
                    deformation_c=eye_closes[side],
                    base_alpha=primary_base_alpha,
                    runtime_opacity=type0_opacity,
                )
                for side in (0, 1)
            ),
            surface_xyz=surface_xyz,
        )
        base = build_rinne_neutral_base_frame(
            assets.renderer,
            assets.atlas,
            expression_weights=expression_weights,
            pupil_profile=assets.pupil_profile,
            right_pupil_position=right_pupil_position,
            left_pupil_position=left_pupil_position,
        )
        type_zero_meshes = tuple(
            replace(draw.mesh, opacity=type0_opacity)
            for draw in base.draws
            if draw.type_id == 0
        )
        if len(type_zero_meshes) != 1:
            raise BinaryBoundsError("neutral delayed-eye type-0 base is missing")

        face_overlay_draw = build_rinne_neutral_face_overlay_mesh(
            assets.renderer,
            assets.atlas,
            assets.open_eye_layout,
            expression_weights=expression_weights,
            pupil_profile=assets.pupil_profile,
            right_pupil_position=right_pupil_position,
            left_pupil_position=left_pupil_position,
            opacity=type0_opacity,
        )
        face_overlay_mesh = TexturedDepthMesh(
            positions_xyz=face_overlay_draw.positions_xyz,
            uvs=face_overlay_draw.game_uvs,
            triangle_indices=face_overlay_draw.triangle_indices,
            opacity=face_overlay_draw.opacity,
        )
        surface_mesh = assets.renderer.mesh_records[
            assets.renderer.type_to_outer[1]
        ]
        type2_meshes = tuple(
            TexturedDepthMesh(
                positions_xyz=strip.positions_xyz,
                uvs=strip.game_uvs,
                triangle_indices=strip.triangle_indices,
                opacity=strip.opacity,
            )
            for strip in (
                build_rinne_type2_eye_strip_mesh(
                    assets.special_eye_profile,
                    assets.atlas.atlas_records,
                    surface_mesh,
                    surface_xyz,
                    side=side,
                    deformation_a=primary_deformation,
                    deformation_c=eye_closes[side],
                    intensity=controls.type2_intensity,
                    runtime_opacity=type2_opacity,
                )
                for side in (0, 1)
            )
        )

        def posed(
            meshes: Sequence[TexturedDepthMesh],
        ) -> tuple[TexturedDepthMesh, ...]:
            return transform_rinne_legacy_meshes(
                meshes,
                assets.neck_profile,
                rotation_xyz=neck_rotation,
                translation_xyz=neck_translation,
            )

        posed_eye_sprite_meshes = posed(
            tuple(
                replace(mesh, opacity=type0_opacity)
                for mesh in assets.eye_sprite_meshes
            )
        )
        open_eye_meshes = posed(open_eye_meshes)
        main_eye_surface_meshes = posed(main_eye_surface_meshes)
        mask_meshes = posed(mask_meshes)
        final_eye_meshes = posed(final_eye_meshes)
        face_overlay_mesh = posed((face_overlay_mesh,))[0]
        type2_meshes = posed(type2_meshes)
        runtime_base_meshes = tuple(
            replace(
                draw.mesh,
                opacity=(
                    type0_opacity
                    if draw.type_id == 0
                    else render_record_opacities[draw.source_draw_order]
                ),
            )
            for draw in base.draws
        )
        posed_base_by_order = {
            draw.source_draw_order: mesh
            for draw, mesh in zip(
                base.draws,
                posed(runtime_base_meshes),
                strict=True,
            )
        }
        type_zero_meshes = tuple(
            posed_base_by_order[draw.source_draw_order]
            for draw in base.draws
            if draw.type_id == 0
        )

        before_trigger: list[TexturedDepthMesh] = []
        after_trigger: list[TexturedDepthMesh] = []
        current = before_trigger
        trigger_seen = False
        for record in draw_records:
            if (
                not record.enabled
                or render_record_opacities[record.file_index] <= 0.0
            ):
                continue
            if record.type_id == 0:
                current.extend(posed_eye_sprite_meshes)
                current.append(posed_base_by_order[record.file_index])
                current.extend(main_eye_surface_meshes)
                current.append(face_overlay_mesh)
            elif record.type_id == 1:
                current.extend(open_eye_meshes)
            elif record.type_id == 2:
                current.extend(type2_meshes)
            else:
                current.append(posed_base_by_order[record.file_index])

            if record.type_id == assets.compositor_trigger_type:
                if trigger_seen:
                    raise BinaryBoundsError("neutral delayed-eye trigger occurs twice")
                trigger_seen = True
                current = after_trigger
        if not trigger_seen:
            raise BinaryBoundsError("neutral delayed-eye trigger was not submitted")

        return RinneLegacyDrawPlan(
            width=width,
            height=height,
            texture=assets.texture,
            blink_geometry=geometry,
            expression_weights=expression_weights,
            compositor_trigger_type=assets.compositor_trigger_type,
            eye_buffer_meshes=posed_eye_sprite_meshes,
            main_eye_surface_meshes=main_eye_surface_meshes,
            face_overlay_meshes=(face_overlay_mesh,),
            special_mask_meshes=mask_meshes,
            mask_buffer_meshes=_build_target1_mask_meshes(
                type_zero_meshes[0], mask_meshes
            ),
            type2_eye_strip_meshes=type2_meshes,
            before_trigger_meshes=tuple(before_trigger),
            final_eye_overlay_meshes=final_eye_meshes,
            after_trigger_meshes=tuple(after_trigger),
        )

    def render_frame(
        self,
        controls: RinneLegacyFrameControls | None = None,
        *,
        width: int = 512,
        height: int = 512,
        include_diagnostics: bool = False,
    ) -> RinneLegacyFrame:
        plan = self.build_draw_plan(controls, width=width, height=height)
        return rasterize_rinne_legacy_draw_plan(
            plan,
            include_diagnostics=include_diagnostics,
        )


__all__ = [
    "RINNE_NEUTRAL_BLINK_GEOMETRY",
    "RinneLegacyAssets",
    "RinneLegacyDrawPlan",
    "RinneLegacyFrame",
    "RinneLegacyFrameControls",
    "RinneLegacyFrameRenderer",
    "rasterize_rinne_legacy_draw_plan",
]
