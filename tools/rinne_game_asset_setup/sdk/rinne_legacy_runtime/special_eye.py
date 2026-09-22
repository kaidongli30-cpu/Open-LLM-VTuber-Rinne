from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass

from .blink_curve import LegacyBlinkGeometrySample
from .checked_binary import BinaryBoundsError, CheckedBinary
from .mpb_v39 import ITOM_MAGIC, VERSION_OFFSET, V39_VERSION
from .mpb_v39_atlas import MpbV39AtlasRecord
from .mpb_v39_renderer import MpbV39MeshRecordView


RINNE_SPECIAL_EYE_POINT_COUNT = 22
RINNE_SPECIAL_EYE_COLUMNS = RINNE_SPECIAL_EYE_POINT_COUNT // 2
RINNE_SPECIAL_EYE_ROWS = 8
RINNE_SPECIAL_EYE_VERTEX_COUNT = (
    RINNE_SPECIAL_EYE_COLUMNS * RINNE_SPECIAL_EYE_ROWS
)
RINNE_SPECIAL_EYE_INDEX_COUNT = (
    (RINNE_SPECIAL_EYE_COLUMNS - 1)
    * (RINNE_SPECIAL_EYE_ROWS - 1)
    * 6
)
RINNE_SPECIAL_EYE_TRIANGLE_COUNT = RINNE_SPECIAL_EYE_INDEX_COUNT // 3
RINNE_LEGACY_BLUR_EYE_SAMPLE_COUNT = 4

V39_FACE_FLAGS_OFFSET = 0x60
V39_SPECIAL_POINT_MODE_OFFSET = 0x70
V39_RIGHT_EYE_POINT_COUNT_OFFSET = 0x12C8
V39_RIGHT_EYE_POINTS_OFFSET = 0x12CC
V39_RIGHT_EYE_SHAPE_CONSTANTS_OFFSET = 0x138C
V39_LEFT_EYE_POINT_COUNT_OFFSET = 0x13CC
V39_LEFT_EYE_POINTS_OFFSET = 0x13D0
V39_LEFT_EYE_SHAPE_CONSTANTS_OFFSET = 0x1490


def _f32(value: float) -> float:
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise BinaryBoundsError(
            "special-eye value is outside float32 range"
        ) from exc
    if not math.isfinite(result):
        raise BinaryBoundsError("special-eye value must be finite")
    return result


def _add(left: float, right: float) -> float:
    return _f32(_f32(left) + _f32(right))


def _sub(left: float, right: float) -> float:
    return _f32(_f32(left) - _f32(right))


def _mul(left: float, right: float) -> float:
    return _f32(_f32(left) * _f32(right))


def _div(numerator: float, denominator: float) -> float:
    denominator = _f32(denominator)
    if denominator == 0.0:
        raise BinaryBoundsError("special-eye interpolation denominator is zero")
    return _f32(_f32(numerator) / denominator)


def _read_f32(reader: CheckedBinary, name: str, offset: int) -> float:
    value = struct.unpack("<f", reader.bytes_at(name, offset, 4))[0]
    if not math.isfinite(value):
        raise BinaryBoundsError(f"{name}: float must be finite")
    return value


def _read_points(
    reader: CheckedBinary,
    *,
    name: str,
    offset: int,
    count: int,
) -> tuple[tuple[float, float], ...]:
    reader.array_span(name, offset, count, 8, max_count=RINNE_SPECIAL_EYE_POINT_COUNT)
    points: list[tuple[float, float]] = []
    for index in range(count):
        point_offset = offset + index * 8
        points.append(
            (
                _read_f32(reader, f"{name} {index} x", point_offset),
                _read_f32(reader, f"{name} {index} y", point_offset + 4),
            )
        )
    return tuple(points)


@dataclass(frozen=True)
class RinneSpecialEyeProfile:
    face_flags: int
    special_point_mode: int
    right_points: tuple[tuple[float, float], ...]
    left_points: tuple[tuple[float, float], ...]
    right_shape_constants: tuple[float, float]
    left_shape_constants: tuple[float, float]

    def points_for_side(self, side: int) -> tuple[tuple[float, float], ...]:
        if side == 0:
            return self.right_points
        if side == 1:
            return self.left_points
        raise BinaryBoundsError("special-eye side must be 0 (right) or 1 (left)")


@dataclass(frozen=True)
class RinneSpecialEyePreprojection:
    side: int
    deformation_u: float
    positions_xy: tuple[tuple[float, float], ...]

    @property
    def columns(self) -> int:
        return RINNE_SPECIAL_EYE_COLUMNS

    @property
    def rows(self) -> int:
        return RINNE_SPECIAL_EYE_ROWS


@dataclass(frozen=True)
class RinneSpecialEyeTopology:
    initial_positions: tuple[tuple[float, float, float], ...]
    initial_uvs: tuple[tuple[float, float], ...]
    vertex_opacities: tuple[float, ...]
    triangle_indices: tuple[int, ...]


@dataclass(frozen=True)
class RinneSpecialEyeProjection:
    side: int
    normalized_positions_xyz: tuple[tuple[float, float, float], ...]
    clip_positions_xyz: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class RinneSpecialEyeAtlasUvs:
    side: int
    record_0_uvs: tuple[tuple[float, float], ...]
    record_1_uvs: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class RinneSpecialEyeDrawState:
    side: int
    draw_mode: int
    atlas_record_id: int
    deformation_a: float
    deformation_c: float
    opacity: float


@dataclass(frozen=True)
class RinneSpecialEyeDrawMesh:
    state: RinneSpecialEyeDrawState
    positions_xyz: tuple[tuple[float, float, float], ...]
    uvs: tuple[tuple[float, float], ...]
    vertex_opacities: tuple[float, ...]
    triangle_indices: tuple[int, ...]


def build_rinne_special_eye_topology() -> RinneSpecialEyeTopology:
    """Reproduce the fixed 10-by-7-cell helper mesh from RVA 0x9BED0."""

    positions: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    vertex_opacities: list[float] = []
    for row in range(RINNE_SPECIAL_EYE_ROWS):
        v = _div(float(row), float(RINNE_SPECIAL_EYE_ROWS - 1))
        for column in range(RINNE_SPECIAL_EYE_COLUMNS):
            u = _div(float(column), float(RINNE_SPECIAL_EYE_COLUMNS - 1))
            positions.append(
                (_sub(_mul(2.0, u), 1.0), _sub(1.0, _mul(2.0, v)), 0.0)
            )
            uvs.append((u, v))
            vertex_opacities.append(
                0.0
                if column in (0, RINNE_SPECIAL_EYE_COLUMNS - 1)
                or row == RINNE_SPECIAL_EYE_ROWS - 1
                else 1.0
            )

    indices: list[int] = []
    for row in range(RINNE_SPECIAL_EYE_ROWS - 1):
        for column in range(RINNE_SPECIAL_EYE_COLUMNS - 1):
            top_left = row * RINNE_SPECIAL_EYE_COLUMNS + column
            bottom_left = (row + 1) * RINNE_SPECIAL_EYE_COLUMNS + column
            top_right = top_left + 1
            bottom_right = bottom_left + 1
            indices.extend(
                (
                    top_left,
                    bottom_left,
                    top_right,
                    top_right,
                    bottom_left,
                    bottom_right,
                )
            )
    if len(positions) != RINNE_SPECIAL_EYE_VERTEX_COUNT or len(indices) != (
        RINNE_SPECIAL_EYE_INDEX_COUNT
    ):
        raise AssertionError("internal special-eye topology mismatch")
    return RinneSpecialEyeTopology(
        tuple(positions), tuple(uvs), tuple(vertex_opacities), tuple(indices)
    )


RINNE_SPECIAL_EYE_TOPOLOGY = build_rinne_special_eye_topology()


def _map_special_eye_atlas_pair(
    pair: tuple[float, float], record: MpbV39AtlasRecord
) -> tuple[float, float]:
    """Map and clamp one pair in RVA 0x1ADF0's float32 operation order."""

    record.validate()

    def map_axis(
        value: float,
        source_first: float,
        source_second: float,
        target_first: float,
        target_second: float,
    ) -> float:
        value = _f32(value)
        source_first = _f32(source_first)
        source_second = _f32(source_second)
        target_first = _f32(target_first)
        target_second = _f32(target_second)
        if source_first > value:
            return target_first
        if value > source_second:
            return target_second
        fraction = _div(
            _sub(value, source_first), _sub(source_second, source_first)
        )
        return _add(
            _mul(fraction, _sub(target_second, target_first)), target_first
        )

    source = record.source_rectangle
    target = record.runtime_atlas_rectangle
    return (
        map_axis(
            pair[0],
            source.first_x,
            source.second_x,
            target.first_x,
            target.second_x,
        ),
        map_axis(
            pair[1],
            source.first_y,
            source.second_y,
            target.first_y,
            target.second_y,
        ),
    )


def build_rinne_special_eye_atlas_uvs(
    profile: RinneSpecialEyeProfile,
    atlas_records: Sequence[MpbV39AtlasRecord],
    *,
    side: int,
) -> RinneSpecialEyeAtlasUvs:
    """Build the two tex_all.tex UV streams uploaded by RVA 0x1ADF0.

    The verified Rinne packages use group-B record IDs zero and one for the
    paired helper meshes. This bounded compatibility path deliberately maps
    into their atlas rectangles; it does not emulate the unused standalone
    texture/fallback resource selector.
    """

    by_id: dict[int, MpbV39AtlasRecord] = {}
    for record in atlas_records:
        if record.record_id in by_id:
            raise BinaryBoundsError(
                f"duplicate special-eye atlas record id {record.record_id}"
            )
        by_id[record.record_id] = record
    try:
        record_0 = by_id[0]
        record_1 = by_id[1]
    except KeyError as exc:
        raise BinaryBoundsError(
            "special-eye atlas requires group-B record IDs 0 and 1"
        ) from exc

    neutral = build_rinne_special_eye_preprojection(
        profile,
        side=side,
        deformation_a=0.0,
        deformation_c=0.0,
    )
    record_0_uvs = tuple(
        _map_special_eye_atlas_pair(pair, record_0)
        for pair in neutral.positions_xy
    )
    record_1_uvs = tuple(
        _map_special_eye_atlas_pair(pair, record_1)
        for pair in neutral.positions_xy
    )
    if (
        len(record_0_uvs) != RINNE_SPECIAL_EYE_VERTEX_COUNT
        or len(record_1_uvs) != RINNE_SPECIAL_EYE_VERTEX_COUNT
    ):
        raise AssertionError("internal special-eye UV count mismatch")
    return RinneSpecialEyeAtlasUvs(
        side=side,
        record_0_uvs=record_0_uvs,
        record_1_uvs=record_1_uvs,
    )


def _clamp_special_eye_unit(value: float, *, name: str) -> float:
    value = _f32(value)
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def resolve_rinne_special_eye_draw_state(
    *,
    side: int,
    draw_mode: int,
    deformation_a: float,
    deformation_c: float,
    base_alpha: float,
    runtime_opacity: float,
) -> RinneSpecialEyeDrawState:
    """Resolve the special-eye buffer and color alpha used by RVA 0x184A0.

    Draw mode zero selects the helper backed by group-B record 1; the Rinne
    calls with modes one through three select record 0. The caller's companion
    geometry weight supplies the default diffuse alpha, with the game's 0.9
    factor for modes below two. A runtime opacity other than exactly float32
    one overrides that default. The independent eye-close control is consumed
    as ``deformation_c`` and must not be confused with this alpha input.
    """

    if side not in (0, 1):
        raise BinaryBoundsError("special-eye side must be 0 (right) or 1 (left)")
    if (
        not isinstance(draw_mode, int)
        or isinstance(draw_mode, bool)
        or draw_mode < 0
        or draw_mode > 3
    ):
        raise BinaryBoundsError("special-eye draw mode must be within 0..3")
    deformation_a = _clamp_special_eye_unit(
        deformation_a, name="deformation A"
    )
    deformation_c = _clamp_special_eye_unit(
        deformation_c, name="deformation C"
    )
    base_alpha = _clamp_special_eye_unit(base_alpha, name="base alpha")
    runtime_opacity = _f32(runtime_opacity)
    if runtime_opacity < 0.0 or runtime_opacity > 1.0:
        raise BinaryBoundsError("special-eye runtime opacity must be within 0..1")

    opacity = base_alpha
    if draw_mode < 2:
        opacity = _mul(opacity, 0.8999999761581421)
    if runtime_opacity != 1.0:
        opacity = runtime_opacity
    return RinneSpecialEyeDrawState(
        side=side,
        draw_mode=draw_mode,
        atlas_record_id=1 if draw_mode == 0 else 0,
        deformation_a=deformation_a,
        deformation_c=deformation_c,
        opacity=opacity,
    )


def resolve_rinne_legacy_blur_eye_draw_states(
    right_geometry: LegacyBlinkGeometrySample,
    left_geometry: LegacyBlinkGeometrySample,
    *,
    right_eye_close: float,
    left_eye_close: float,
    runtime_opacity: float,
) -> tuple[RinneSpecialEyeDrawState, ...]:
    """Resolve the eight mode-1 eye submissions used by this game.

    The game's startup passes a null renderer configuration at RVA 0x9B0FF,
    so RVA 0xFFD0 enables the global eyelid-blur branch. RVA
    0xF352..0xF491 then walks four geometry/alpha samples and submits the
    right eye followed by the left eye for each sample. The Rinne MPBs have
    face flags zero, so the optional RVA-0x18E40 companion branch is skipped.

    Keeping the two geometry samples separate preserves the executable's
    bilateral runtime layout even when an automatic blink feeds identical
    values to both sides.
    """

    samples = (right_geometry, left_geometry)
    for side, geometry in enumerate(samples):
        if len(geometry.deformation) != RINNE_LEGACY_BLUR_EYE_SAMPLE_COUNT:
            raise BinaryBoundsError(
                f"blur-eye side {side} requires exactly four deformations"
            )
        if len(geometry.base_weights) != RINNE_LEGACY_BLUR_EYE_SAMPLE_COUNT:
            raise BinaryBoundsError(
                f"blur-eye side {side} requires exactly four base weights"
            )

    eye_closes = (right_eye_close, left_eye_close)
    states: list[RinneSpecialEyeDrawState] = []
    for sample_index in range(RINNE_LEGACY_BLUR_EYE_SAMPLE_COUNT):
        for side, geometry in enumerate(samples):
            states.append(
                resolve_rinne_special_eye_draw_state(
                    side=side,
                    draw_mode=1,
                    deformation_a=geometry.deformation[sample_index],
                    deformation_c=eye_closes[side],
                    base_alpha=geometry.base_weights[sample_index],
                    runtime_opacity=runtime_opacity,
                )
            )
    return tuple(states)


def build_rinne_special_eye_draw_mesh(
    profile: RinneSpecialEyeProfile,
    atlas_records: Sequence[MpbV39AtlasRecord],
    surface_mesh: MpbV39MeshRecordView,
    surface_positions_xyz: Sequence[Sequence[float]],
    state: RinneSpecialEyeDrawState,
) -> RinneSpecialEyeDrawMesh:
    """Join the proven staging, projection, UV, topology, and draw selection."""

    verified_state = resolve_rinne_special_eye_draw_state(
        side=state.side,
        draw_mode=state.draw_mode,
        deformation_a=state.deformation_a,
        deformation_c=state.deformation_c,
        base_alpha=state.opacity,
        runtime_opacity=state.opacity,
    )
    if verified_state != state:
        raise BinaryBoundsError("special-eye draw state is not canonical")
    preprojection = build_rinne_special_eye_preprojection(
        profile,
        side=state.side,
        deformation_a=state.deformation_a,
        deformation_c=state.deformation_c,
    )
    projection = project_rinne_special_eye_surface(
        preprojection, surface_mesh, surface_positions_xyz
    )
    atlas_uvs = build_rinne_special_eye_atlas_uvs(
        profile, atlas_records, side=state.side
    )
    uvs = (
        atlas_uvs.record_1_uvs
        if state.atlas_record_id == 1
        else atlas_uvs.record_0_uvs
    )
    return RinneSpecialEyeDrawMesh(
        state=state,
        positions_xyz=projection.normalized_positions_xyz,
        uvs=uvs,
        vertex_opacities=RINNE_SPECIAL_EYE_TOPOLOGY.vertex_opacities,
        triangle_indices=RINNE_SPECIAL_EYE_TOPOLOGY.triangle_indices,
    )


def parse_v39_rinne_special_eye_profile(
    data: bytes | bytearray | memoryview,
) -> RinneSpecialEyeProfile:
    """Read the Rinne-proven 22-point eye profiles from a v39 face.mpb.

    This parser intentionally accepts only the branch observed in every local
    MP060101..MP060115 sample. The alternate ten-point reduction and face-flag
    shape-constant branch remain unsupported until they have their own assets
    and consumer evidence.
    """

    reader = CheckedBinary(data)
    magic = reader.bytes_at("ITOM magic", 0, 4)
    if magic != ITOM_MAGIC:
        raise ValueError(f"unexpected magic: {magic!r}; expected {ITOM_MAGIC!r}")
    version = reader.u32("format version", VERSION_OFFSET)
    if version != V39_VERSION:
        raise ValueError(
            f"unsupported MPB version {version}; special-eye parser requires v39"
        )

    face_flags = reader.u32("v39 face flags", V39_FACE_FLAGS_OFFSET)
    if face_flags != 0:
        raise BinaryBoundsError(
            "unsupported special-eye face flags; Rinne samples require zero"
        )
    special_point_mode = reader.u32(
        "v39 special point mode", V39_SPECIAL_POINT_MODE_OFFSET
    )
    if special_point_mode != 0:
        raise BinaryBoundsError(
            "unsupported special-eye point mode; Rinne samples require zero"
        )

    right_count = reader.u32(
        "right special-eye point count", V39_RIGHT_EYE_POINT_COUNT_OFFSET
    )
    left_count = reader.u32(
        "left special-eye point count", V39_LEFT_EYE_POINT_COUNT_OFFSET
    )
    if right_count != RINNE_SPECIAL_EYE_POINT_COUNT:
        raise BinaryBoundsError(
            "right special-eye point count must be 22 for the Rinne path; "
            f"got {right_count}"
        )
    if left_count != RINNE_SPECIAL_EYE_POINT_COUNT:
        raise BinaryBoundsError(
            "left special-eye point count must be 22 for the Rinne path; "
            f"got {left_count}"
        )

    right_points = _read_points(
        reader,
        name="right special-eye points",
        offset=V39_RIGHT_EYE_POINTS_OFFSET,
        count=right_count,
    )
    left_points = _read_points(
        reader,
        name="left special-eye points",
        offset=V39_LEFT_EYE_POINTS_OFFSET,
        count=left_count,
    )
    right_shape_constants = (
        _read_f32(
            reader,
            "right special-eye shape constant 0",
            V39_RIGHT_EYE_SHAPE_CONSTANTS_OFFSET,
        ),
        _read_f32(
            reader,
            "right special-eye shape constant 1",
            V39_RIGHT_EYE_SHAPE_CONSTANTS_OFFSET + 4,
        ),
    )
    left_shape_constants = (
        _read_f32(
            reader,
            "left special-eye shape constant 0",
            V39_LEFT_EYE_SHAPE_CONSTANTS_OFFSET,
        ),
        _read_f32(
            reader,
            "left special-eye shape constant 1",
            V39_LEFT_EYE_SHAPE_CONSTANTS_OFFSET + 4,
        ),
    )
    return RinneSpecialEyeProfile(
        face_flags=face_flags,
        special_point_mode=special_point_mode,
        right_points=right_points,
        left_points=left_points,
        right_shape_constants=right_shape_constants,
        left_shape_constants=left_shape_constants,
    )


def build_rinne_special_eye_preprojection(
    profile: RinneSpecialEyeProfile,
    *,
    side: int,
    deformation_a: float,
    deformation_c: float,
) -> RinneSpecialEyePreprojection:
    """Reproduce RVA 0x184A0's Rinne-only XY staging before RVA 0x1A1D0.

    The 22 source points are two eleven-point contours. Eight interpolated or
    extrapolated rows are emitted in row-major order. Face-surface projection,
    Z generation, UV/index buffers, texture choice, and draw state are later
    stages and are deliberately not claimed here.
    """

    if profile.face_flags != 0 or profile.special_point_mode != 0:
        raise BinaryBoundsError(
            "special-eye profile is outside the verified Rinne branch"
        )
    points = profile.points_for_side(side)
    if len(points) != RINNE_SPECIAL_EYE_POINT_COUNT:
        raise BinaryBoundsError("special-eye profile must contain exactly 22 points")
    if any(not math.isfinite(value) for point in points for value in point):
        raise BinaryBoundsError("special-eye source points must be finite")

    deformation_a = _f32(deformation_a)
    deformation_c = _f32(deformation_c)
    deformation_u = _add(
        _mul(_sub(1.0, deformation_c), deformation_a),
        deformation_c,
    )
    narrow_weight = _mul(deformation_u, 0.05)
    main_weight = _mul(deformation_u, 0.95)

    upper = points[:RINNE_SPECIAL_EYE_COLUMNS]
    lower = points[RINNE_SPECIAL_EYE_COLUMNS:]
    positions: list[tuple[float, float]] = []

    intermediate_span = _add(_sub(main_weight, 0.1), 0.12)
    for row in range(RINNE_SPECIAL_EYE_ROWS - 3):
        row_fraction = _div(float(row), float(RINNE_SPECIAL_EYE_ROWS - 2))
        weight = _sub(_mul(intermediate_span, row_fraction), 0.12)
        for upper_point, lower_point in zip(upper, lower, strict=True):
            positions.append(
                (
                    _add(
                        upper_point[0],
                        _mul(_sub(lower_point[0], upper_point[0]), weight),
                    ),
                    _add(
                        upper_point[1],
                        _mul(_sub(lower_point[1], upper_point[1]), weight),
                    ),
                )
            )

    for upper_point, lower_point in zip(upper, lower, strict=True):
        positions.append(
            (
                _add(
                    upper_point[0],
                    _mul(_sub(lower_point[0], upper_point[0]), main_weight),
                ),
                _add(
                    upper_point[1],
                    _mul(_sub(lower_point[1], upper_point[1]), main_weight),
                ),
            )
        )

    for upper_point, lower_point in zip(upper, lower, strict=True):
        positions.append(
            (
                _add(
                    lower_point[0],
                    _mul(_sub(upper_point[0], lower_point[0]), narrow_weight),
                ),
                _add(
                    lower_point[1],
                    _mul(_sub(upper_point[1], lower_point[1]), narrow_weight),
                ),
            )
        )

    for upper_point, lower_point in zip(upper, lower, strict=True):
        positions.append(
            (
                _add(
                    lower_point[0],
                    _mul(_sub(lower_point[0], upper_point[0]), 0.4),
                ),
                _add(
                    lower_point[1],
                    _mul(_sub(lower_point[1], upper_point[1]), 0.4),
                ),
            )
        )

    if len(positions) != RINNE_SPECIAL_EYE_VERTEX_COUNT:
        raise AssertionError("internal special-eye vertex-count mismatch")
    return RinneSpecialEyePreprojection(
        side=side,
        deformation_u=deformation_u,
        positions_xy=tuple(positions),
    )


def _find_surface_x_cell(mesh: MpbV39MeshRecordView, value: float) -> int:
    columns = mesh.grid_cell_columns
    if columns is None:
        raise BinaryBoundsError("special-eye surface grid has no column count")
    clamped = min(max(_f32(value), 0.0), 1.0)
    index = 0
    while index < columns:
        if mesh.coordinate_source_xy.at(index)[0] > clamped:
            break
        index += 1
    cell = index - 1
    if cell < 0 or cell >= columns:
        raise BinaryBoundsError("special-eye X lies outside a valid surface cell")
    return cell


def _find_surface_y_cell(mesh: MpbV39MeshRecordView, value: float) -> int:
    columns = mesh.grid_cell_columns
    rows = mesh.grid_cell_rows
    if columns is None or rows is None:
        raise BinaryBoundsError("special-eye surface grid has no row count")
    stride = columns + 1
    clamped = min(max(_f32(value), 0.0), 1.0)
    row = 0
    while row < rows:
        if clamped > mesh.coordinate_source_xy.at(row * stride)[1]:
            break
        row += 1
    cell = row - 1
    if cell < 0 or cell >= rows:
        raise BinaryBoundsError("special-eye Y lies outside a valid surface cell")
    return cell


def project_rinne_special_eye_surface(
    preprojection: RinneSpecialEyePreprojection,
    surface_mesh: MpbV39MeshRecordView,
    surface_positions_xyz: Sequence[Sequence[float]],
) -> RinneSpecialEyeProjection:
    """Apply RVA 0x1A1D0 and the following 4*x-2, 4*y-2 transform.

    ``surface_positions_xyz`` is the caller's current type-0 face surface.
    Keeping it explicit prevents this bounded function from inventing the
    still-separate expression/depth state that supplies those runtime values.
    """

    surface_mesh.validate()
    columns = surface_mesh.grid_cell_columns
    rows = surface_mesh.grid_cell_rows
    if columns is None or rows is None:
        raise BinaryBoundsError("special-eye projection requires grid dimensions")
    grid_vertex_count = (columns + 1) * (rows + 1)
    if len(surface_positions_xyz) < grid_vertex_count:
        raise BinaryBoundsError(
            "special-eye surface position count is smaller than its grid"
        )
    for column in range(columns):
        left_x = surface_mesh.coordinate_source_xy.at(column)[0]
        right_x = surface_mesh.coordinate_source_xy.at(column + 1)[0]
        if not left_x < right_x:
            raise BinaryBoundsError(
                "special-eye surface first-row X coordinates must increase"
            )
    stride = columns + 1
    for row in range(rows):
        top_y = surface_mesh.coordinate_source_xy.at(row * stride)[1]
        bottom_y = surface_mesh.coordinate_source_xy.at((row + 1) * stride)[1]
        if not top_y > bottom_y:
            raise BinaryBoundsError(
                "special-eye surface first-column Y coordinates must decrease"
            )
    stable_surface: list[tuple[float, float, float]] = []
    for vertex_index, vertex in enumerate(surface_positions_xyz):
        if len(vertex) != 3:
            raise BinaryBoundsError(
                f"special-eye surface vertex {vertex_index} must contain XYZ"
            )
        stable_surface.append(tuple(_f32(value) for value in vertex))

    normalized: list[tuple[float, float, float]] = []
    clip: list[tuple[float, float, float]] = []
    for input_x, input_y in preprojection.positions_xy:
        input_x = _f32(input_x)
        input_y = _f32(input_y)
        cell_x = _find_surface_x_cell(surface_mesh, input_x)
        cell_y = _find_surface_y_cell(surface_mesh, input_y)
        top_left_index = cell_y * stride + cell_x
        top_right_index = top_left_index + 1
        bottom_left_index = (cell_y + 1) * stride + cell_x
        bottom_right_index = bottom_left_index + 1

        source_top_left = surface_mesh.coordinate_source_xy.at(top_left_index)
        source_top_right = surface_mesh.coordinate_source_xy.at(top_right_index)
        source_bottom_left = surface_mesh.coordinate_source_xy.at(bottom_left_index)
        current_top_left = stable_surface[top_left_index]
        current_top_right = stable_surface[top_right_index]
        current_bottom_left = stable_surface[bottom_left_index]
        current_bottom_right = stable_surface[bottom_right_index]

        fraction_x = _div(
            _sub(input_x, source_top_left[0]),
            _sub(source_top_right[0], source_top_left[0]),
        )
        fraction_y = _div(
            _sub(input_y, source_bottom_left[1]),
            _sub(source_top_left[1], source_bottom_left[1]),
        )
        one_minus_x = _sub(1.0, fraction_x)
        one_minus_y = _sub(1.0, fraction_y)
        weight_top_right = _mul(fraction_y, fraction_x)
        weight_top_left = _mul(one_minus_x, fraction_y)
        weight_bottom_left = _mul(one_minus_y, one_minus_x)
        weight_bottom_right = _mul(one_minus_y, fraction_x)

        delta_top_left_x = _sub(current_top_left[0], source_top_left[0])
        delta_top_right_x = _sub(current_top_right[0], source_top_right[0])
        delta_bottom_left_x = _sub(
            current_bottom_left[0], source_top_left[0]
        )
        delta_bottom_right_x = _sub(
            current_bottom_right[0], source_top_right[0]
        )
        projected_x = _mul(weight_top_right, delta_top_right_x)
        projected_x = _add(
            projected_x, _mul(weight_top_left, delta_top_left_x)
        )
        projected_x = _add(
            projected_x, _mul(weight_bottom_left, delta_bottom_left_x)
        )
        projected_x = _add(
            projected_x, _mul(weight_bottom_right, delta_bottom_right_x)
        )
        projected_x = _add(projected_x, input_x)

        delta_top_left_y = _sub(current_top_left[1], source_top_left[1])
        delta_top_right_y = _sub(current_top_right[1], source_top_left[1])
        delta_bottom_left_y = _sub(
            current_bottom_left[1], source_bottom_left[1]
        )
        delta_bottom_right_y = _sub(
            current_bottom_right[1], source_bottom_left[1]
        )
        projected_y = _mul(weight_top_right, delta_top_right_y)
        projected_y = _add(
            projected_y, _mul(weight_top_left, delta_top_left_y)
        )
        projected_y = _add(
            projected_y, _mul(weight_bottom_left, delta_bottom_left_y)
        )
        projected_y = _add(
            projected_y, _mul(weight_bottom_right, delta_bottom_right_y)
        )
        projected_y = _add(projected_y, input_y)

        projected_z = _mul(current_top_left[2], weight_top_left)
        projected_z = _add(
            projected_z, _mul(current_top_right[2], weight_top_right)
        )
        projected_z = _add(
            projected_z, _mul(current_bottom_left[2], weight_bottom_left)
        )
        projected_z = _add(
            projected_z, _mul(current_bottom_right[2], weight_bottom_right)
        )
        normalized_position = (projected_x, projected_y, projected_z)
        normalized.append(normalized_position)
        clip.append(
            (
                _sub(_mul(projected_x, 4.0), 2.0),
                _sub(_mul(projected_y, 4.0), 2.0),
                projected_z,
            )
        )

    return RinneSpecialEyeProjection(
        side=preprojection.side,
        normalized_positions_xyz=tuple(normalized),
        clip_positions_xyz=tuple(clip),
    )


__all__ = [
    "RINNE_LEGACY_BLUR_EYE_SAMPLE_COUNT",
    "RINNE_SPECIAL_EYE_COLUMNS",
    "RINNE_SPECIAL_EYE_INDEX_COUNT",
    "RINNE_SPECIAL_EYE_POINT_COUNT",
    "RINNE_SPECIAL_EYE_ROWS",
    "RINNE_SPECIAL_EYE_TOPOLOGY",
    "RINNE_SPECIAL_EYE_TRIANGLE_COUNT",
    "RINNE_SPECIAL_EYE_VERTEX_COUNT",
    "RinneSpecialEyeAtlasUvs",
    "RinneSpecialEyeDrawMesh",
    "RinneSpecialEyeDrawState",
    "RinneSpecialEyePreprojection",
    "RinneSpecialEyeProfile",
    "RinneSpecialEyeProjection",
    "RinneSpecialEyeTopology",
    "build_rinne_special_eye_atlas_uvs",
    "build_rinne_special_eye_draw_mesh",
    "build_rinne_special_eye_preprojection",
    "build_rinne_special_eye_topology",
    "parse_v39_rinne_special_eye_profile",
    "project_rinne_special_eye_surface",
    "resolve_rinne_legacy_blur_eye_draw_states",
    "resolve_rinne_special_eye_draw_state",
]
