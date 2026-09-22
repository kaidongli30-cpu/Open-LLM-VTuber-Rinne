from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .blink_curve import LEGACY_BLINK_BASE_WEIGHTS, LegacyBlinkGeometrySample
from .checked_binary import BinaryBoundsError
from .frame_renderer import (
    RinneLegacyDrawPlan,
    RinneLegacyFrameControls,
    RinneLegacyFrameRenderer,
)
from .gpu_bundle import (
    RINNE_GPU_GEOMETRY_FILENAME,
    RINNE_GPU_MANIFEST_FILENAME,
    RINNE_GPU_TEXTURE_FILENAME,
)
from .gpu_linear_dynamics import (
    RINNE_GPU_DEFORMATION_FILENAME,
    RINNE_GPU_DYNAMIC_MANIFEST_FILENAME,
    RinneGpuLinearDynamicBundle,
    flatten_rinne_gpu_positions,
    reconstruct_rinne_gpu_linear_positions,
    verify_rinne_gpu_linear_dynamic_bundle,
)
from .gpu_topology import (
    RINNE_GPU_DYNAMIC_PASS_FIELDS,
    verify_rinne_gpu_topology_compatible,
)
from .io_safety import write_new_bytes
from .neutral_type0 import project_rinne_legacy_vertex
from .special_eye import (
    RINNE_SPECIAL_EYE_VERTEX_COUNT,
    build_rinne_special_eye_preprojection,
    project_rinne_special_eye_surface,
)
from .type2_eye_strip import (
    RINNE_TYPE2_EYE_STRIP_VERTEX_COUNT,
    build_rinne_type2_eye_strip_mesh,
    build_rinne_type2_eye_strip_preprojection,
)


RINNE_GPU_EYE_DYNAMIC_FORMAT = "rinne-legacy-gpu-eye-dynamics"
RINNE_GPU_EYE_DYNAMIC_VERSION = 1
RINNE_GPU_EYE_MANIFEST_FILENAME = "eye-dynamic-manifest.json"
RINNE_GPU_EYE_DEFORMATION_FILENAME = "eye-deformation.rgba32f"
RINNE_GPU_EYE_SAMPLE_COUNT = 129
RINNE_GPU_EYE_MAX_SAMPLE_COUNT = 512
RINNE_GPU_EYE_TEXTURE_WIDTH = 1024
RINNE_GPU_EYE_MAX_BYTES = 64 * 1024 * 1024
RINNE_GPU_EYE_MANIFEST_MAX_BYTES = 4 * 1024 * 1024
RINNE_GPU_EYE_RECONSTRUCTION_TOLERANCE = 2.0e-6
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CAPABILITIES = [
    "blink_deformation",
    "bilateral_eye_close",
    "type2_intensity",
]
_UNSUPPORTED_CONTROLS = [
    "neck_pose",
    "render_record_opacity",
    "amb_playback",
]
_GROUPS = (
    ("special_eye_right", "special_eye", 0, RINNE_SPECIAL_EYE_VERTEX_COUNT),
    ("special_eye_left", "special_eye", 1, RINNE_SPECIAL_EYE_VERTEX_COUNT),
    (
        "type2_eye_strip_right",
        "type2_eye_strip",
        0,
        RINNE_TYPE2_EYE_STRIP_VERTEX_COUNT,
    ),
    (
        "type2_eye_strip_left",
        "type2_eye_strip",
        1,
        RINNE_TYPE2_EYE_STRIP_VERTEX_COUNT,
    ),
)
_ACCEPTANCE_BLINK_DEFORMATION = (0.8, 0.55, 0.25, 0.05)
_ACCEPTANCE_RIGHT_EYE_CLOSE = 0.2
_ACCEPTANCE_LEFT_EYE_CLOSE = 0.65
_ACCEPTANCE_TYPE2_INTENSITY = 0.7


@dataclass(frozen=True)
class RinneGpuEyeDynamicBundle:
    linear_bundle: RinneGpuLinearDynamicBundle
    eye_manifest_json: bytes
    eye_deformation_rgba32f: bytes

    def __post_init__(self) -> None:
        verify_rinne_gpu_eye_dynamic_bundle(self)

    def eye_manifest(self) -> dict[str, Any]:
        decoded = json.loads(self.eye_manifest_json)
        if not isinstance(decoded, dict):
            raise BinaryBoundsError("GPU eye manifest root must be an object")
        return decoded


@dataclass(frozen=True)
class RinneGpuEyeReconstruction:
    component_count: int
    maximum_absolute_error: float
    mean_absolute_error: float
    components_above_tolerance: int
    tolerance: float
    accepted: bool


def _f32(value: float) -> float:
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise BinaryBoundsError("GPU eye value is outside float32 range") from exc
    if not math.isfinite(result):
        raise BinaryBoundsError("GPU eye value must be finite")
    return result


def _unit(value: float, label: str) -> float:
    result = _f32(float(value))
    if result < 0.0 or result > 1.0:
        raise BinaryBoundsError(f"GPU eye {label} must be within 0..1")
    return result


def _group_descriptors() -> tuple[dict[str, object], ...]:
    result = []
    cursor = 0
    for index, (name, kind, side, vertex_count) in enumerate(_GROUPS):
        result.append(
            {
                "index": index,
                "name": name,
                "kind": kind,
                "eye_side": side,
                "vertex_offset": cursor,
                "vertex_count": vertex_count,
            }
        )
        cursor += vertex_count
    return tuple(result)


def _eye_sample_positions(
    renderer: RinneLegacyFrameRenderer,
) -> tuple[float, ...]:
    """Add face-grid crossing points to the uniform deformation grid."""

    assets = renderer.assets
    surface_mesh = assets.renderer.mesh_records[
        assets.renderer.type_to_outer[1]
    ]
    columns = surface_mesh.grid_cell_columns
    rows = surface_mesh.grid_cell_rows
    if columns is None or rows is None:
        raise BinaryBoundsError("GPU eye lookup requires a face-surface grid")
    stride = columns + 1
    x_boundaries = tuple(
        surface_mesh.coordinate_source_xy.at(index)[0]
        for index in range(1, columns)
    )
    y_boundaries = tuple(
        surface_mesh.coordinate_source_xy.at(row * stride)[1]
        for row in range(1, rows)
    )
    endpoints = []
    for side in (0, 1):
        endpoints.append(
            (
                build_rinne_special_eye_preprojection(
                    assets.special_eye_profile,
                    side=side,
                    deformation_a=0.0,
                    deformation_c=0.0,
                ),
                build_rinne_special_eye_preprojection(
                    assets.special_eye_profile,
                    side=side,
                    deformation_a=1.0,
                    deformation_c=0.0,
                ),
            )
        )
        endpoints.append(
            (
                build_rinne_type2_eye_strip_preprojection(
                    assets.special_eye_profile,
                    side=side,
                    deformation_a=0.0,
                    deformation_c=0.0,
                ),
                build_rinne_type2_eye_strip_preprojection(
                    assets.special_eye_profile,
                    side=side,
                    deformation_a=1.0,
                    deformation_c=0.0,
                ),
            )
        )
    values = {
        index / (RINNE_GPU_EYE_SAMPLE_COUNT - 1)
        for index in range(RINNE_GPU_EYE_SAMPLE_COUNT)
    }
    for start, end in endpoints:
        if len(start.positions_xy) != len(end.positions_xy):
            raise AssertionError("internal GPU eye endpoint count mismatch")
        for start_point, end_point in zip(
            start.positions_xy, end.positions_xy, strict=True
        ):
            for axis, boundaries in ((0, x_boundaries), (1, y_boundaries)):
                first = start_point[axis]
                delta = end_point[axis] - first
                if delta == 0.0:
                    continue
                for boundary in boundaries:
                    crossing = (boundary - first) / delta
                    if 0.0 < crossing < 1.0:
                        values.add(_f32(crossing))
    result = tuple(sorted(values))
    if len(result) > RINNE_GPU_EYE_MAX_SAMPLE_COUNT:
        raise BinaryBoundsError(
            "GPU eye adaptive sample count exceeds safety limit: "
            f"{len(result)} > {RINNE_GPU_EYE_MAX_SAMPLE_COUNT}"
        )
    if result[0] != 0.0 or result[-1] != 1.0 or any(
        left >= right for left, right in zip(result, result[1:], strict=False)
    ):
        raise AssertionError("internal GPU eye sample positions are not canonical")
    return result


def _surface_control_states(
    renderer: RinneLegacyFrameRenderer,
    control_names: Sequence[str],
) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    expression_count = renderer.expression_weight_count
    surfaces = [renderer.build_surface_positions_xyz()]
    for name in control_names:
        if name.startswith("expression_"):
            try:
                expression_index = int(name.removeprefix("expression_"))
            except ValueError as exc:
                raise BinaryBoundsError(
                    "GPU eye expression control name is invalid"
                ) from exc
            if expression_index < 0 or expression_index >= expression_count:
                raise BinaryBoundsError("GPU eye expression index is invalid")
            expression = tuple(
                1.0 if index == expression_index else 0.0
                for index in range(expression_count)
            )
            controls = RinneLegacyFrameControls(expression_weights=expression)
        elif name == "right_pupil_x":
            controls = RinneLegacyFrameControls(
                right_pupil_position=(1.0, 0.0)
            )
        elif name == "right_pupil_y":
            controls = RinneLegacyFrameControls(
                right_pupil_position=(0.0, 1.0)
            )
        elif name == "left_pupil_x":
            controls = RinneLegacyFrameControls(
                left_pupil_position=(1.0, 0.0)
            )
        elif name == "left_pupil_y":
            controls = RinneLegacyFrameControls(
                left_pupil_position=(0.0, 1.0)
            )
        else:
            raise BinaryBoundsError(f"GPU eye control name is unsupported: {name}")
        surfaces.append(renderer.build_surface_positions_xyz(controls))
    return tuple(surfaces)


def _positions_for_groups(
    renderer: RinneLegacyFrameRenderer,
    surface_positions_xyz: tuple[tuple[float, float, float], ...],
    deformation: float,
) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    assets = renderer.assets
    surface_mesh = assets.renderer.mesh_records[
        assets.renderer.type_to_outer[1]
    ]
    deformation = _unit(deformation, "lookup deformation")
    special_groups = []
    for side in (0, 1):
        preprojection = build_rinne_special_eye_preprojection(
            assets.special_eye_profile,
            side=side,
            deformation_a=deformation,
            deformation_c=0.0,
        )
        projection = project_rinne_special_eye_surface(
            preprojection,
            surface_mesh,
            surface_positions_xyz,
        )
        special_groups.append(
            tuple(
                project_rinne_legacy_vertex(x, y, z, depth_offset=0.0)
                for x, y, z in projection.normalized_positions_xyz
            )
        )
    type2_groups = tuple(
        build_rinne_type2_eye_strip_mesh(
            assets.special_eye_profile,
            assets.atlas.atlas_records,
            surface_mesh,
            surface_positions_xyz,
            side=side,
            deformation_a=deformation,
            deformation_c=0.0,
            intensity=1.0,
            runtime_opacity=1.0,
        ).positions_xyz
        for side in (0, 1)
    )
    return (*special_groups, *type2_groups)


def _reference_group_positions(
    reference: RinneLegacyDrawPlan,
) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    if len(reference.main_eye_surface_meshes) != 8:
        raise BinaryBoundsError("GPU eye reference requires eight blur-eye meshes")
    if len(reference.special_mask_meshes) != 2:
        raise BinaryBoundsError("GPU eye reference requires two mask-eye meshes")
    if len(reference.final_eye_overlay_meshes) != 2:
        raise BinaryBoundsError("GPU eye reference requires two final-eye meshes")
    if len(reference.type2_eye_strip_meshes) != 2:
        raise BinaryBoundsError("GPU eye reference requires two type-2 meshes")
    groups = (
        tuple(reference.main_eye_surface_meshes[0].positions_xyz),
        tuple(reference.main_eye_surface_meshes[1].positions_xyz),
        tuple(reference.type2_eye_strip_meshes[0].positions_xyz),
        tuple(reference.type2_eye_strip_meshes[1].positions_xyz),
    )
    for group, descriptor in zip(groups, _GROUPS, strict=True):
        if len(group) != descriptor[3]:
            raise BinaryBoundsError("GPU eye reference group vertex count changed")
    return groups


def _mesh_bindings(
    reference: RinneLegacyDrawPlan,
    linear_manifest: dict[str, Any],
    group_positions: tuple[tuple[tuple[float, float, float], ...], ...],
) -> tuple[dict[str, object], ...]:
    semantics: dict[int, tuple[int, int, str]] = {}

    def register(mesh: object, group: int, sample: int, opacity_mode: str) -> None:
        existing = semantics.get(id(mesh))
        value = (group, sample, opacity_mode)
        if existing is not None and existing != value:
            raise BinaryBoundsError("GPU eye mesh has conflicting semantics")
        semantics[id(mesh)] = value

    for index, mesh in enumerate(reference.main_eye_surface_meshes):
        register(
            mesh,
            index % 2,
            index // 2,
            "blur_sample_base_weight_0_9",
        )
    for side, mesh in enumerate(reference.special_mask_meshes):
        register(mesh, side, 0, "runtime_type0")
    for side, mesh in enumerate(reference.final_eye_overlay_meshes):
        register(mesh, side, 0, "primary_base_weight_0_9")
    for side, mesh in enumerate(reference.type2_eye_strip_meshes):
        register(mesh, 2 + side, 0, "eased_u_intensity_0_8")

    offsets = {
        (item["pass"], item["mesh_index"]): item
        for item in linear_manifest["deformation"]["mesh_vertex_offsets"]
    }
    bindings = []
    for pass_field in RINNE_GPU_DYNAMIC_PASS_FIELDS:
        pass_name = pass_field.removesuffix("_meshes")
        for mesh_index, mesh in enumerate(getattr(reference, pass_field)):
            semantic = semantics.get(id(mesh))
            if semantic is None:
                continue
            group, blink_sample, opacity_mode = semantic
            offset = offsets[(pass_name, mesh_index)]
            if tuple(mesh.positions_xyz) != group_positions[group]:
                raise BinaryBoundsError(
                    "GPU eye bound mesh does not match its neutral group"
                )
            bindings.append(
                {
                    "pass": pass_name,
                    "mesh_index": mesh_index,
                    "vertex_offset": offset["vertex_offset"],
                    "vertex_count": offset["vertex_count"],
                    "group_index": group,
                    "blink_sample_index": blink_sample,
                    "eye_side": _GROUPS[group][2],
                    "opacity_mode": opacity_mode,
                }
            )
    expected_modes = {
        "blur_sample_base_weight_0_9": 8,
        "runtime_type0": 2,
        "primary_base_weight_0_9": 2,
        "eased_u_intensity_0_8": 2,
    }
    actual_modes = {
        name: sum(binding["opacity_mode"] == name for binding in bindings)
        for name in expected_modes
    }
    if actual_modes != expected_modes:
        raise BinaryBoundsError("GPU eye mesh-binding coverage is incomplete")
    return tuple(bindings)


def build_rinne_gpu_eye_dynamic_bundle(
    linear_bundle: RinneGpuLinearDynamicBundle,
    renderer: RinneLegacyFrameRenderer,
    reference: RinneLegacyDrawPlan,
) -> RinneGpuEyeDynamicBundle:
    linear_manifest = verify_rinne_gpu_linear_dynamic_bundle(linear_bundle)
    if not isinstance(renderer, RinneLegacyFrameRenderer):
        raise TypeError("GPU eye bundle requires a frame renderer")
    if not isinstance(reference, RinneLegacyDrawPlan):
        raise TypeError("GPU eye bundle requires a reference draw plan")
    source = linear_bundle.base_bundle.manifest()["source"]
    assets = renderer.assets
    if source["parsed_member_sha256"] != {
        "face.mpb": assets.mpb_sha256,
        "tex_all.tex": assets.texture_container_sha256,
        "face.uca.bin": assets.auto_animation_config_sha256,
        "001.amb": assets.animation_sha256,
    }:
        raise BinaryBoundsError("GPU eye renderer does not match the base bundle")
    control_names = tuple(
        item["name"] for item in linear_manifest["deformation"]["controls"]
    )
    surfaces = _surface_control_states(renderer, control_names)
    basis_count = len(surfaces)
    if basis_count != len(control_names) + 1:
        raise AssertionError("internal GPU eye basis count mismatch")
    group_descriptors = _group_descriptors()
    group_vertex_count = sum(item["vertex_count"] for item in group_descriptors)
    reference_groups = _reference_group_positions(reference)
    direct_neutral = _positions_for_groups(renderer, surfaces[0], 0.0)
    if direct_neutral != reference_groups:
        raise BinaryBoundsError(
            "GPU eye direct neutral projection changed the reference plan"
        )
    bindings = _mesh_bindings(reference, linear_manifest, reference_groups)

    sample_positions = _eye_sample_positions(renderer)
    sample_count = len(sample_positions)
    used_texels = group_vertex_count * sample_count * basis_count
    texture_height = (
        used_texels + RINNE_GPU_EYE_TEXTURE_WIDTH - 1
    ) // RINNE_GPU_EYE_TEXTURE_WIDTH
    payload = bytearray(RINNE_GPU_EYE_TEXTURE_WIDTH * texture_height * 16)
    for sample_index, deformation in enumerate(sample_positions):
        positions_by_basis = tuple(
            _positions_for_groups(renderer, surface, deformation)
            for surface in surfaces
        )
        base_groups = positions_by_basis[0]
        group_vertex_offset = 0
        for group_index, base_group in enumerate(base_groups):
            static_group = reference_groups[group_index]
            for vertex_index, base_position in enumerate(base_group):
                global_group_vertex = group_vertex_offset + vertex_index
                for basis_index in range(basis_count):
                    if basis_index == 0:
                        values = tuple(
                            base_position[axis] - static_group[vertex_index][axis]
                            for axis in range(3)
                        )
                    else:
                        unit_position = positions_by_basis[basis_index][
                            group_index
                        ][vertex_index]
                        values = tuple(
                            unit_position[axis] - base_position[axis]
                            for axis in range(3)
                        )
                    texel = (
                        (
                            global_group_vertex * sample_count
                            + sample_index
                        )
                        * basis_count
                        + basis_index
                    )
                    struct.pack_into(
                        "<ffff",
                        payload,
                        texel * 16,
                        values[0],
                        values[1],
                        values[2],
                        0.0,
                    )
            group_vertex_offset += len(base_group)
    eye_deformation = bytes(payload)
    if len(eye_deformation) > RINNE_GPU_EYE_MAX_BYTES:
        raise BinaryBoundsError("GPU eye deformation table exceeds safety limit")
    manifest = {
        "format": RINNE_GPU_EYE_DYNAMIC_FORMAT,
        "version": RINNE_GPU_EYE_DYNAMIC_VERSION,
        "capabilities": _CAPABILITIES,
        "unsupported_controls": _UNSUPPORTED_CONTROLS,
        "linear_bundle": {
            "dynamic_manifest_sha256": hashlib.sha256(
                linear_bundle.dynamic_manifest_json
            ).hexdigest(),
            "deformation_sha256": hashlib.sha256(
                linear_bundle.deformation_rgba32f
            ).hexdigest(),
            "base_topology_sha256": linear_manifest["base_bundle"][
                "topology_sha256"
            ],
        },
        "lookup": {
            "file": RINNE_GPU_EYE_DEFORMATION_FILENAME,
            "format": "rgba32f_little_endian",
            "byte_length": len(eye_deformation),
            "sha256": hashlib.sha256(eye_deformation).hexdigest(),
            "texture_width": RINNE_GPU_EYE_TEXTURE_WIDTH,
            "texture_height": texture_height,
            "used_texel_count": used_texels,
            "sample_count": sample_count,
            "sample_positions": list(sample_positions),
            "sample_parameter": "combined_eye_deformation_u_0_to_1",
            "interpolation": "linear_between_declared_samples",
            "basis_count": basis_count,
            "linear_control_names": list(control_names),
            "group_vertex_count": group_vertex_count,
            "groups": list(group_descriptors),
            "mesh_bindings": list(bindings),
        },
        "reconstruction": {
            "maximum_position_error": RINNE_GPU_EYE_RECONSTRUCTION_TOLERANCE,
            "position_space": "projected_normalized_xyz_before_neck_pose",
        },
        "acceptance_sample": {
            "blink_deformation": list(_ACCEPTANCE_BLINK_DEFORMATION),
            "blink_base_weights": list(LEGACY_BLINK_BASE_WEIGHTS),
            "right_eye_close": _ACCEPTANCE_RIGHT_EYE_CLOSE,
            "left_eye_close": _ACCEPTANCE_LEFT_EYE_CLOSE,
            "type2_intensity": _ACCEPTANCE_TYPE2_INTENSITY,
        },
    }
    eye_manifest_json = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return RinneGpuEyeDynamicBundle(
        linear_bundle=linear_bundle,
        eye_manifest_json=eye_manifest_json,
        eye_deformation_rgba32f=eye_deformation,
    )


def _require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BinaryBoundsError(f"GPU eye {label} must be an object")
    return value


def verify_rinne_gpu_eye_dynamic_bundle(
    bundle: RinneGpuEyeDynamicBundle,
) -> dict[str, Any]:
    if not isinstance(bundle, RinneGpuEyeDynamicBundle):
        raise TypeError("GPU eye verification requires an eye bundle")
    linear = verify_rinne_gpu_linear_dynamic_bundle(bundle.linear_bundle)
    if not 0 < len(bundle.eye_manifest_json) <= RINNE_GPU_EYE_MANIFEST_MAX_BYTES:
        raise BinaryBoundsError("GPU eye manifest size exceeds safety limit")
    try:
        manifest = json.loads(bundle.eye_manifest_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BinaryBoundsError("GPU eye manifest is invalid JSON") from exc
    root = _require_object(manifest, "manifest")
    if (
        root.get("format") != RINNE_GPU_EYE_DYNAMIC_FORMAT
        or root.get("version") != RINNE_GPU_EYE_DYNAMIC_VERSION
        or root.get("capabilities") != _CAPABILITIES
        or root.get("unsupported_controls") != _UNSUPPORTED_CONTROLS
    ):
        raise BinaryBoundsError("GPU eye format or capability boundary is invalid")
    parent = _require_object(root.get("linear_bundle"), "linear bundle")
    if parent != {
        "dynamic_manifest_sha256": hashlib.sha256(
            bundle.linear_bundle.dynamic_manifest_json
        ).hexdigest(),
        "deformation_sha256": hashlib.sha256(
            bundle.linear_bundle.deformation_rgba32f
        ).hexdigest(),
        "base_topology_sha256": linear["base_bundle"]["topology_sha256"],
    }:
        raise BinaryBoundsError("GPU eye linear-bundle identity is invalid")
    lookup = _require_object(root.get("lookup"), "lookup")
    width = lookup.get("texture_width")
    height = lookup.get("texture_height")
    used_texels = lookup.get("used_texel_count")
    sample_count = lookup.get("sample_count")
    basis_count = lookup.get("basis_count")
    group_vertex_count = lookup.get("group_vertex_count")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in (
            width,
            height,
            used_texels,
            sample_count,
            basis_count,
            group_vertex_count,
        )
    ):
        raise BinaryBoundsError("GPU eye lookup dimensions are invalid")
    assert isinstance(width, int)
    assert isinstance(height, int)
    assert isinstance(used_texels, int)
    assert isinstance(sample_count, int)
    assert isinstance(basis_count, int)
    assert isinstance(group_vertex_count, int)
    if (
        lookup.get("file") != RINNE_GPU_EYE_DEFORMATION_FILENAME
        or lookup.get("format") != "rgba32f_little_endian"
        or width != RINNE_GPU_EYE_TEXTURE_WIDTH
        or sample_count < RINNE_GPU_EYE_SAMPLE_COUNT
        or sample_count > RINNE_GPU_EYE_MAX_SAMPLE_COUNT
        or lookup.get("sample_parameter")
        != "combined_eye_deformation_u_0_to_1"
        or lookup.get("interpolation") != "linear_between_declared_samples"
        or used_texels != group_vertex_count * sample_count * basis_count
        or lookup.get("byte_length") != len(bundle.eye_deformation_rgba32f)
        or len(bundle.eye_deformation_rgba32f) != width * height * 16
        or len(bundle.eye_deformation_rgba32f) > RINNE_GPU_EYE_MAX_BYTES
        or lookup.get("sha256")
        != hashlib.sha256(bundle.eye_deformation_rgba32f).hexdigest()
    ):
        raise BinaryBoundsError("GPU eye lookup bytes are inconsistent")
    sample_positions = lookup.get("sample_positions")
    if (
        not isinstance(sample_positions, list)
        or len(sample_positions) != sample_count
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or float(value) > 1.0
            for value in sample_positions
        )
        or sample_positions[0] != 0.0
        or sample_positions[-1] != 1.0
        or any(
            left >= right
            for left, right in zip(
                sample_positions, sample_positions[1:], strict=False
            )
        )
    ):
        raise BinaryBoundsError("GPU eye sample positions are invalid")
    control_names = tuple(
        item["name"] for item in linear["deformation"]["controls"]
    )
    if (
        basis_count != len(control_names) + 1
        or lookup.get("linear_control_names") != list(control_names)
    ):
        raise BinaryBoundsError("GPU eye lookup bases do not match linear controls")
    expected_groups = list(_group_descriptors())
    if lookup.get("groups") != expected_groups:
        raise BinaryBoundsError("GPU eye lookup groups are invalid")
    if group_vertex_count != sum(item["vertex_count"] for item in expected_groups):
        raise BinaryBoundsError("GPU eye group vertex count is invalid")
    offsets = {
        (item["pass"], item["mesh_index"]): item
        for item in linear["deformation"]["mesh_vertex_offsets"]
    }
    bindings = lookup.get("mesh_bindings")
    if not isinstance(bindings, list) or len(bindings) != 14:
        raise BinaryBoundsError("GPU eye mesh bindings are invalid")
    seen = set()
    modes = {}
    for item in bindings:
        binding = _require_object(item, "mesh binding")
        key = (binding.get("pass"), binding.get("mesh_index"))
        group_index = binding.get("group_index")
        sample_index = binding.get("blink_sample_index")
        opacity_mode = binding.get("opacity_mode")
        if (
            key not in offsets
            or key in seen
            or not isinstance(group_index, int)
            or isinstance(group_index, bool)
            or group_index < 0
            or group_index >= len(expected_groups)
            or not isinstance(sample_index, int)
            or isinstance(sample_index, bool)
            or sample_index < 0
            or sample_index > 3
            or binding.get("vertex_offset") != offsets[key]["vertex_offset"]
            or binding.get("vertex_count") != offsets[key]["vertex_count"]
            or binding.get("vertex_count")
            != expected_groups[group_index]["vertex_count"]
            or binding.get("eye_side")
            != expected_groups[group_index]["eye_side"]
            or opacity_mode
            not in (
                "blur_sample_base_weight_0_9",
                "runtime_type0",
                "primary_base_weight_0_9",
                "eased_u_intensity_0_8",
            )
        ):
            raise BinaryBoundsError("GPU eye mesh binding is inconsistent")
        seen.add(key)
        modes[opacity_mode] = modes.get(opacity_mode, 0) + 1
    if modes != {
        "blur_sample_base_weight_0_9": 8,
        "runtime_type0": 2,
        "primary_base_weight_0_9": 2,
        "eased_u_intensity_0_8": 2,
    }:
        raise BinaryBoundsError("GPU eye opacity binding coverage is invalid")
    for texel_index, values in enumerate(
        struct.iter_unpack("<ffff", bundle.eye_deformation_rgba32f)
    ):
        if any(not math.isfinite(value) for value in values):
            raise BinaryBoundsError(
                f"GPU eye lookup contains a non-finite texel at {texel_index}"
            )
        if values[3] != 0.0 or (
            texel_index >= used_texels and values != (0.0, 0.0, 0.0, 0.0)
        ):
            raise BinaryBoundsError("GPU eye lookup padding is not canonical")
    if root.get("reconstruction") != {
        "maximum_position_error": RINNE_GPU_EYE_RECONSTRUCTION_TOLERANCE,
        "position_space": "projected_normalized_xyz_before_neck_pose",
    }:
        raise BinaryBoundsError("GPU eye reconstruction contract is invalid")
    if root.get("acceptance_sample") != {
        "blink_deformation": list(_ACCEPTANCE_BLINK_DEFORMATION),
        "blink_base_weights": list(LEGACY_BLINK_BASE_WEIGHTS),
        "right_eye_close": _ACCEPTANCE_RIGHT_EYE_CLOSE,
        "left_eye_close": _ACCEPTANCE_LEFT_EYE_CLOSE,
        "type2_intensity": _ACCEPTANCE_TYPE2_INTENSITY,
    }:
        raise BinaryBoundsError("GPU eye acceptance sample is invalid")
    return root


def _lookup_delta(
    bundle: RinneGpuEyeDynamicBundle,
    *,
    group_vertex: int,
    sample_index: int,
    sample_count: int,
    basis_index: int,
    basis_count: int,
) -> tuple[float, float, float]:
    texel = (
        (group_vertex * sample_count + sample_index)
        * basis_count
        + basis_index
    )
    return struct.unpack_from("<fff", bundle.eye_deformation_rgba32f, texel * 16)


def reconstruct_rinne_gpu_eye_positions(
    bundle: RinneGpuEyeDynamicBundle,
    reference: RinneLegacyDrawPlan,
    linear_coefficients: Sequence[float],
    *,
    blink_deformation: Sequence[float],
    right_eye_close: float,
    left_eye_close: float,
) -> tuple[float, ...]:
    manifest = verify_rinne_gpu_eye_dynamic_bundle(bundle)
    if len(blink_deformation) != 4:
        raise BinaryBoundsError("GPU eye blink deformation requires four values")
    deformation = tuple(
        _unit(value, f"blink deformation {index}")
        for index, value in enumerate(blink_deformation)
    )
    eye_closes = (
        _unit(right_eye_close, "right close"),
        _unit(left_eye_close, "left close"),
    )
    coefficients = tuple(float(value) for value in linear_coefficients)
    lookup = manifest["lookup"]
    if len(coefficients) + 1 != lookup["basis_count"] or any(
        not math.isfinite(value) for value in coefficients
    ):
        raise BinaryBoundsError("GPU eye linear coefficients are invalid")
    output = list(
        reconstruct_rinne_gpu_linear_positions(
            bundle.linear_bundle,
            reference,
            coefficients,
        )
    )
    reference_positions = flatten_rinne_gpu_positions(reference)
    groups = lookup["groups"]
    sample_positions = tuple(float(value) for value in lookup["sample_positions"])
    sample_count = len(sample_positions)
    for binding in lookup["mesh_bindings"]:
        side = binding["eye_side"]
        blink_sample = binding["blink_sample_index"]
        combined = _f32(
            _f32(_f32(1.0 - eye_closes[side]) * deformation[blink_sample])
            + eye_closes[side]
        )
        lower = max(0, min(sample_count - 1, bisect_right(sample_positions, combined) - 1))
        upper = min(sample_count - 1, lower + 1)
        span = sample_positions[upper] - sample_positions[lower]
        fraction = 0.0 if span == 0.0 else (combined - sample_positions[lower]) / span
        group = groups[binding["group_index"]]
        for local_vertex in range(binding["vertex_count"]):
            group_vertex = group["vertex_offset"] + local_vertex
            destination = (binding["vertex_offset"] + local_vertex) * 3
            result = [reference_positions[destination + axis] for axis in range(3)]
            for basis_index, coefficient in enumerate((1.0, *coefficients)):
                if coefficient == 0.0:
                    continue
                lower_delta = _lookup_delta(
                    bundle,
                    group_vertex=group_vertex,
                    sample_index=lower,
                    sample_count=sample_count,
                    basis_index=basis_index,
                    basis_count=lookup["basis_count"],
                )
                upper_delta = _lookup_delta(
                    bundle,
                    group_vertex=group_vertex,
                    sample_index=upper,
                    sample_count=sample_count,
                    basis_index=basis_index,
                    basis_count=lookup["basis_count"],
                )
                for axis in range(3):
                    delta = lower_delta[axis] + fraction * (
                        upper_delta[axis] - lower_delta[axis]
                    )
                    result[axis] += coefficient * delta
            output[destination : destination + 3] = result
    return tuple(output)


def compare_rinne_gpu_eye_reconstruction(
    bundle: RinneGpuEyeDynamicBundle,
    reference: RinneLegacyDrawPlan,
    actual: RinneLegacyDrawPlan,
    linear_coefficients: Sequence[float],
    *,
    blink_deformation: Sequence[float],
    right_eye_close: float,
    left_eye_close: float,
) -> RinneGpuEyeReconstruction:
    verify_rinne_gpu_topology_compatible(reference, actual)
    predicted = reconstruct_rinne_gpu_eye_positions(
        bundle,
        reference,
        linear_coefficients,
        blink_deformation=blink_deformation,
        right_eye_close=right_eye_close,
        left_eye_close=left_eye_close,
    )
    actual_positions = flatten_rinne_gpu_positions(actual)
    differences = tuple(
        abs(actual_value - predicted_value)
        for actual_value, predicted_value in zip(
            actual_positions, predicted, strict=True
        )
    )
    tolerance = RINNE_GPU_EYE_RECONSTRUCTION_TOLERANCE
    components_above = sum(value > tolerance for value in differences)
    return RinneGpuEyeReconstruction(
        component_count=len(differences),
        maximum_absolute_error=max(differences, default=0.0),
        mean_absolute_error=(
            0.0 if not differences else sum(differences) / len(differences)
        ),
        components_above_tolerance=components_above,
        tolerance=tolerance,
        accepted=components_above == 0,
    )


def compare_rinne_gpu_eye_interpolation_matrix(
    bundle: RinneGpuEyeDynamicBundle,
    renderer: RinneLegacyFrameRenderer,
    surface_controls: RinneLegacyFrameControls,
    linear_coefficients: Sequence[float],
    *,
    deformation_values: Sequence[float] | None = None,
) -> RinneGpuEyeReconstruction:
    """Compare lookup interpolation with direct eye projection over many U values."""

    manifest = verify_rinne_gpu_eye_dynamic_bundle(bundle)
    if not isinstance(renderer, RinneLegacyFrameRenderer):
        raise TypeError("GPU eye interpolation matrix requires a frame renderer")
    if not isinstance(surface_controls, RinneLegacyFrameControls):
        raise TypeError("GPU eye interpolation matrix requires frame controls")
    coefficients = tuple(float(value) for value in linear_coefficients)
    lookup = manifest["lookup"]
    if len(coefficients) + 1 != lookup["basis_count"] or any(
        not math.isfinite(value) for value in coefficients
    ):
        raise BinaryBoundsError("GPU eye matrix coefficients are invalid")
    sample_positions = tuple(float(value) for value in lookup["sample_positions"])
    sample_count = len(sample_positions)
    values = (
        tuple(
            (left + right) * 0.5
            for left, right in zip(
                sample_positions, sample_positions[1:], strict=False
            )
        )
        if deformation_values is None
        else tuple(_unit(value, "matrix deformation") for value in deformation_values)
    )
    if not values:
        raise BinaryBoundsError("GPU eye interpolation matrix is empty")
    base_surface = renderer.build_surface_positions_xyz()
    current_surface = renderer.build_surface_positions_xyz(surface_controls)
    static_groups = _positions_for_groups(renderer, base_surface, 0.0)
    differences = []
    for deformation in values:
        actual_groups = _positions_for_groups(
            renderer, current_surface, deformation
        )
        lower = max(
            0,
            min(
                sample_count - 1,
                bisect_right(sample_positions, deformation) - 1,
            ),
        )
        upper = min(sample_count - 1, lower + 1)
        span = sample_positions[upper] - sample_positions[lower]
        fraction = (
            0.0
            if span == 0.0
            else (deformation - sample_positions[lower]) / span
        )
        for group_index, actual_group in enumerate(actual_groups):
            group = lookup["groups"][group_index]
            static_group = static_groups[group_index]
            for local_vertex, actual_position in enumerate(actual_group):
                group_vertex = group["vertex_offset"] + local_vertex
                predicted = [static_group[local_vertex][axis] for axis in range(3)]
                for basis_index, coefficient in enumerate((1.0, *coefficients)):
                    if coefficient == 0.0:
                        continue
                    lower_delta = _lookup_delta(
                        bundle,
                        group_vertex=group_vertex,
                        sample_index=lower,
                        sample_count=sample_count,
                        basis_index=basis_index,
                        basis_count=lookup["basis_count"],
                    )
                    upper_delta = _lookup_delta(
                        bundle,
                        group_vertex=group_vertex,
                        sample_index=upper,
                        sample_count=sample_count,
                        basis_index=basis_index,
                        basis_count=lookup["basis_count"],
                    )
                    for axis in range(3):
                        delta = lower_delta[axis] + fraction * (
                            upper_delta[axis] - lower_delta[axis]
                        )
                        predicted[axis] += coefficient * delta
                differences.extend(
                    abs(actual_position[axis] - predicted[axis])
                    for axis in range(3)
                )
    tolerance = RINNE_GPU_EYE_RECONSTRUCTION_TOLERANCE
    components_above = sum(value > tolerance for value in differences)
    return RinneGpuEyeReconstruction(
        component_count=len(differences),
        maximum_absolute_error=max(differences, default=0.0),
        mean_absolute_error=(
            0.0 if not differences else sum(differences) / len(differences)
        ),
        components_above_tolerance=components_above,
        tolerance=tolerance,
        accepted=components_above == 0,
    )


def write_rinne_gpu_eye_dynamic_bundle(
    output_directory: Path | str,
    bundle: RinneGpuEyeDynamicBundle,
) -> None:
    verify_rinne_gpu_eye_dynamic_bundle(bundle)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=False)
    base = bundle.linear_bundle.base_bundle
    for filename, payload in (
        (RINNE_GPU_MANIFEST_FILENAME, base.manifest_json),
        (RINNE_GPU_TEXTURE_FILENAME, base.texture_rgba8),
        (RINNE_GPU_GEOMETRY_FILENAME, base.geometry_binary),
        (
            RINNE_GPU_DYNAMIC_MANIFEST_FILENAME,
            bundle.linear_bundle.dynamic_manifest_json,
        ),
        (RINNE_GPU_DEFORMATION_FILENAME, bundle.linear_bundle.deformation_rgba32f),
        (RINNE_GPU_EYE_MANIFEST_FILENAME, bundle.eye_manifest_json),
        (RINNE_GPU_EYE_DEFORMATION_FILENAME, bundle.eye_deformation_rgba32f),
    ):
        write_new_bytes(destination / filename, payload)


def acceptance_rinne_gpu_eye_controls(
    linear_expression_weights: Sequence[float],
    *,
    right_pupil_position: Sequence[float],
    left_pupil_position: Sequence[float],
) -> RinneLegacyFrameControls:
    return RinneLegacyFrameControls(
        blink_geometry=LegacyBlinkGeometrySample(
            deformation=_ACCEPTANCE_BLINK_DEFORMATION,
            base_weights=LEGACY_BLINK_BASE_WEIGHTS,
        ),
        right_eye_close=_ACCEPTANCE_RIGHT_EYE_CLOSE,
        left_eye_close=_ACCEPTANCE_LEFT_EYE_CLOSE,
        right_pupil_position=tuple(right_pupil_position),  # type: ignore[arg-type]
        left_pupil_position=tuple(left_pupil_position),  # type: ignore[arg-type]
        type2_intensity=_ACCEPTANCE_TYPE2_INTENSITY,
        expression_weights=tuple(linear_expression_weights),
    )


__all__ = [
    "RINNE_GPU_EYE_DEFORMATION_FILENAME",
    "RINNE_GPU_EYE_DYNAMIC_FORMAT",
    "RINNE_GPU_EYE_DYNAMIC_VERSION",
    "RINNE_GPU_EYE_MANIFEST_FILENAME",
    "RINNE_GPU_EYE_MAX_SAMPLE_COUNT",
    "RINNE_GPU_EYE_RECONSTRUCTION_TOLERANCE",
    "RINNE_GPU_EYE_SAMPLE_COUNT",
    "RINNE_GPU_EYE_TEXTURE_WIDTH",
    "RinneGpuEyeDynamicBundle",
    "RinneGpuEyeReconstruction",
    "acceptance_rinne_gpu_eye_controls",
    "build_rinne_gpu_eye_dynamic_bundle",
    "compare_rinne_gpu_eye_reconstruction",
    "compare_rinne_gpu_eye_interpolation_matrix",
    "reconstruct_rinne_gpu_eye_positions",
    "verify_rinne_gpu_eye_dynamic_bundle",
    "write_rinne_gpu_eye_dynamic_bundle",
]
