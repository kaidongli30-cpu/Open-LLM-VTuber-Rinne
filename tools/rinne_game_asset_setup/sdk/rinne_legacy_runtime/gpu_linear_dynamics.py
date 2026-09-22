from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .checked_binary import BinaryBoundsError
from .expression_morph import prepare_v39_expression_weights
from .frame_renderer import (
    RinneLegacyDrawPlan,
    RinneLegacyFrameControls,
    RinneLegacyFrameRenderer,
)
from .gpu_bundle import (
    RINNE_GPU_GEOMETRY_FILENAME,
    RINNE_GPU_MANIFEST_FILENAME,
    RINNE_GPU_TEXTURE_FILENAME,
    RinneGpuBundle,
    verify_rinne_gpu_bundle,
)
from .gpu_topology import (
    RINNE_GPU_DYNAMIC_PASS_FIELDS,
    verify_rinne_gpu_topology_compatible,
    verify_rinne_gpu_topology_matrix,
)
from .io_safety import write_new_bytes


RINNE_GPU_LINEAR_DYNAMIC_FORMAT = "rinne-legacy-gpu-linear-dynamics"
RINNE_GPU_LINEAR_DYNAMIC_VERSION = 1
RINNE_GPU_DYNAMIC_MANIFEST_FILENAME = "dynamic-manifest.json"
RINNE_GPU_DEFORMATION_FILENAME = "deformation.rgba32f"
RINNE_GPU_DEFORMATION_TEXTURE_WIDTH = 1024
RINNE_GPU_LINEAR_RECONSTRUCTION_TOLERANCE = 2.0e-6
RINNE_GPU_DYNAMIC_MANIFEST_MAX_BYTES = 4 * 1024 * 1024
RINNE_GPU_LINEAR_MAX_BYTES = 64 * 1024 * 1024
RINNE_GPU_LINEAR_MAX_CONTROLS = 64
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_PUPIL_CONTROL_NAMES = (
    "right_pupil_x",
    "right_pupil_y",
    "left_pupil_x",
    "left_pupil_y",
)
_CAPABILITIES = [
    "expression_weights",
    "mouth_expression_slots_5_6_7",
    "breath_expression_slot_4",
    "bilateral_pupil_position",
]
_UNSUPPORTED_CONTROLS = [
    "blink_deformation",
    "bilateral_eye_close",
    "neck_pose",
    "render_record_opacity",
    "amb_playback",
]
_ACCEPTANCE_RIGHT_PUPIL_POSITION = (-0.6, 0.35)
_ACCEPTANCE_LEFT_PUPIL_POSITION = (0.45, -0.25)


def _acceptance_expression_weights(count: int) -> tuple[float, ...]:
    values = [0.0] * count
    for index, value in (
        (0, 0.25),
        (1, 0.4),
        (4, 0.37),
        (5, 0.16),
        (6, 0.28),
        (7, 0.44),
        (8, 0.2),
    ):
        if index < count:
            values[index] = value
    return tuple(values)


@dataclass(frozen=True)
class RinneGpuLinearControlSample:
    name: str
    coefficient_mode: str
    unit_positions_xyz: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.name.isascii():
            raise BinaryBoundsError("GPU linear control name must be ASCII")
        if self.coefficient_mode not in (
            "threshold_0_01",
            "cosine_ease_then_threshold_0_01",
            "identity",
        ):
            raise BinaryBoundsError("GPU linear coefficient mode is unsupported")
        if not self.unit_positions_xyz or any(
            not math.isfinite(value) for value in self.unit_positions_xyz
        ):
            raise BinaryBoundsError("GPU linear unit positions must be finite")


@dataclass(frozen=True)
class RinneGpuLinearDynamicBundle:
    base_bundle: RinneGpuBundle
    dynamic_manifest_json: bytes
    deformation_rgba32f: bytes

    def __post_init__(self) -> None:
        verify_rinne_gpu_linear_dynamic_bundle(self)

    def dynamic_manifest(self) -> dict[str, Any]:
        decoded = json.loads(self.dynamic_manifest_json)
        if not isinstance(decoded, dict):
            raise BinaryBoundsError("GPU dynamic manifest root must be an object")
        return decoded


@dataclass(frozen=True)
class RinneGpuLinearReconstruction:
    component_count: int
    maximum_absolute_error: float
    mean_absolute_error: float
    components_above_tolerance: int
    tolerance: float
    accepted: bool


def flatten_rinne_gpu_positions(plan: RinneLegacyDrawPlan) -> tuple[float, ...]:
    if not isinstance(plan, RinneLegacyDrawPlan):
        raise TypeError("GPU position flattening requires a draw plan")
    return tuple(
        component
        for pass_name in RINNE_GPU_DYNAMIC_PASS_FIELDS
        for mesh in getattr(plan, pass_name)
        for position in mesh.positions_xyz
        for component in position
    )


def _mesh_vertex_offsets(plan: RinneLegacyDrawPlan) -> tuple[dict[str, object], ...]:
    result: list[dict[str, object]] = []
    offset = 0
    for pass_name in RINNE_GPU_DYNAMIC_PASS_FIELDS:
        for mesh_index, mesh in enumerate(getattr(plan, pass_name)):
            result.append(
                {
                    "pass": pass_name.removesuffix("_meshes"),
                    "mesh_index": mesh_index,
                    "vertex_offset": offset,
                    "vertex_count": len(mesh.positions_xyz),
                }
            )
            offset += len(mesh.positions_xyz)
    return tuple(result)


def build_rinne_gpu_linear_control_samples(
    renderer: RinneLegacyFrameRenderer,
    reference: RinneLegacyDrawPlan,
    *,
    width: int,
    height: int,
) -> tuple[RinneGpuLinearControlSample, ...]:
    if not isinstance(renderer, RinneLegacyFrameRenderer):
        raise TypeError("GPU linear samples require a frame renderer")
    expression_count = renderer.expression_weight_count
    ease_flags = renderer.assets.renderer.expression_ease_flags
    if len(ease_flags) != expression_count:
        raise BinaryBoundsError("GPU linear expression ease flags are incomplete")
    samples: list[RinneGpuLinearControlSample] = []
    for expression_index, ease_flag in enumerate(ease_flags):
        if ease_flag not in (0, 1):
            raise BinaryBoundsError(
                "GPU linear expression ease flag must be zero or one"
            )
        plan = renderer.build_draw_plan(
            RinneLegacyFrameControls(
                expression_weights=tuple(
                    1.0 if index == expression_index else 0.0
                    for index in range(expression_count)
                )
            ),
            width=width,
            height=height,
        )
        verify_rinne_gpu_topology_compatible(reference, plan)
        samples.append(
            RinneGpuLinearControlSample(
                name=f"expression_{expression_index}",
                coefficient_mode=(
                    "cosine_ease_then_threshold_0_01"
                    if ease_flag == 1
                    else "threshold_0_01"
                ),
                unit_positions_xyz=flatten_rinne_gpu_positions(plan),
            )
        )
    pupil_inputs = (
        ((1.0, 0.0), (0.0, 0.0)),
        ((0.0, 1.0), (0.0, 0.0)),
        ((0.0, 0.0), (1.0, 0.0)),
        ((0.0, 0.0), (0.0, 1.0)),
    )
    for name, (right, left) in zip(
        _PUPIL_CONTROL_NAMES, pupil_inputs, strict=True
    ):
        plan = renderer.build_draw_plan(
            RinneLegacyFrameControls(
                right_pupil_position=right,
                left_pupil_position=left,
            ),
            width=width,
            height=height,
        )
        verify_rinne_gpu_topology_compatible(reference, plan)
        samples.append(
            RinneGpuLinearControlSample(
                name=name,
                coefficient_mode="identity",
                unit_positions_xyz=flatten_rinne_gpu_positions(plan),
            )
        )
    return tuple(samples)


def build_rinne_gpu_linear_dynamic_bundle(
    base_bundle: RinneGpuBundle,
    reference: RinneLegacyDrawPlan,
    samples: tuple[RinneGpuLinearControlSample, ...],
) -> RinneGpuLinearDynamicBundle:
    base_manifest = verify_rinne_gpu_bundle(base_bundle)
    if not isinstance(reference, RinneLegacyDrawPlan):
        raise TypeError("GPU linear bundle requires a reference draw plan")
    if not samples or len(samples) > RINNE_GPU_LINEAR_MAX_CONTROLS:
        raise BinaryBoundsError("GPU linear control count exceeds safety limit")
    if len({sample.name for sample in samples}) != len(samples):
        raise BinaryBoundsError("GPU linear control names must be unique")
    base_positions = flatten_rinne_gpu_positions(reference)
    if len(base_positions) % 3:
        raise AssertionError("internal GPU position component count mismatch")
    vertex_count = len(base_positions) // 3
    for sample in samples:
        if len(sample.unit_positions_xyz) != len(base_positions):
            raise BinaryBoundsError("GPU linear unit position count changed")

    control_count = len(samples)
    texel_count = vertex_count * control_count
    texture_width = RINNE_GPU_DEFORMATION_TEXTURE_WIDTH
    texture_height = (texel_count + texture_width - 1) // texture_width
    deformation = bytearray(texture_width * texture_height * 16)
    for vertex in range(vertex_count):
        position_offset = vertex * 3
        for control_index, sample in enumerate(samples):
            texel = vertex * control_count + control_index
            byte_offset = texel * 16
            struct.pack_into(
                "<ffff",
                deformation,
                byte_offset,
                sample.unit_positions_xyz[position_offset]
                - base_positions[position_offset],
                sample.unit_positions_xyz[position_offset + 1]
                - base_positions[position_offset + 1],
                sample.unit_positions_xyz[position_offset + 2]
                - base_positions[position_offset + 2],
                0.0,
            )
    deformation_bytes = bytes(deformation)
    if len(deformation_bytes) > RINNE_GPU_LINEAR_MAX_BYTES:
        raise BinaryBoundsError("GPU linear deformation table exceeds safety limit")
    topology = verify_rinne_gpu_topology_matrix(reference, (reference,))
    manifest = {
        "format": RINNE_GPU_LINEAR_DYNAMIC_FORMAT,
        "version": RINNE_GPU_LINEAR_DYNAMIC_VERSION,
        "capabilities": _CAPABILITIES,
        "unsupported_controls": _UNSUPPORTED_CONTROLS,
        "base_bundle": {
            "manifest_sha256": hashlib.sha256(
                base_bundle.manifest_json
            ).hexdigest(),
            "texture_sha256": base_manifest["texture"]["sha256"],
            "geometry_sha256": base_manifest["geometry"]["sha256"],
            "topology_sha256": topology.topology_sha256,
        },
        "deformation": {
            "file": RINNE_GPU_DEFORMATION_FILENAME,
            "format": "rgba32f_little_endian",
            "byte_length": len(deformation_bytes),
            "sha256": hashlib.sha256(deformation_bytes).hexdigest(),
            "texture_width": texture_width,
            "texture_height": texture_height,
            "used_texel_count": texel_count,
            "vertex_count": vertex_count,
            "control_count": control_count,
            "mesh_vertex_offsets": list(_mesh_vertex_offsets(reference)),
            "controls": [
                {
                    "index": index,
                    "name": sample.name,
                    "coefficient_mode": sample.coefficient_mode,
                }
                for index, sample in enumerate(samples)
            ],
        },
        "reconstruction": {
            "maximum_position_error": (
                RINNE_GPU_LINEAR_RECONSTRUCTION_TOLERANCE
            ),
            "position_space": "projected_normalized_xyz_before_neck_pose",
        },
        "acceptance_sample": {
            "raw_expression_weights": list(
                _acceptance_expression_weights(len(reference.expression_weights))
            ),
            "right_pupil_position": list(_ACCEPTANCE_RIGHT_PUPIL_POSITION),
            "left_pupil_position": list(_ACCEPTANCE_LEFT_PUPIL_POSITION),
        },
    }
    dynamic_manifest_json = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return RinneGpuLinearDynamicBundle(
        base_bundle=base_bundle,
        dynamic_manifest_json=dynamic_manifest_json,
        deformation_rgba32f=deformation_bytes,
    )


def resolve_rinne_gpu_linear_coefficients(
    expression_weights: Sequence[float],
    expression_ease_flags: Sequence[int],
    *,
    right_pupil_position: Sequence[float] = (0.0, 0.0),
    left_pupil_position: Sequence[float] = (0.0, 0.0),
) -> tuple[float, ...]:
    prepared = prepare_v39_expression_weights(
        expression_weights, expression_ease_flags
    )
    expression = tuple(value if value >= 0.01 else 0.0 for value in prepared)
    pupils: list[float] = []
    for label, values in (
        ("right pupil", right_pupil_position),
        ("left pupil", left_pupil_position),
    ):
        if len(values) != 2:
            raise BinaryBoundsError(f"GPU {label} requires exactly two values")
        for value in values:
            converted = float(value)
            if not math.isfinite(converted):
                raise BinaryBoundsError(f"GPU {label} values must be finite")
            pupils.append(converted)
    return (*expression, *pupils)


def reconstruct_rinne_gpu_linear_positions(
    bundle: RinneGpuLinearDynamicBundle,
    reference: RinneLegacyDrawPlan,
    coefficients: Sequence[float],
) -> tuple[float, ...]:
    manifest = verify_rinne_gpu_linear_dynamic_bundle(bundle)
    base = flatten_rinne_gpu_positions(reference)
    deformation = manifest["deformation"]
    control_count = deformation["control_count"]
    vertex_count = deformation["vertex_count"]
    if len(coefficients) != control_count:
        raise BinaryBoundsError("GPU linear coefficient count is invalid")
    stable_coefficients = tuple(float(value) for value in coefficients)
    if any(not math.isfinite(value) for value in stable_coefficients):
        raise BinaryBoundsError("GPU linear coefficients must be finite")
    if len(base) != vertex_count * 3:
        raise BinaryBoundsError("GPU linear reference vertex count changed")
    output = list(base)
    for vertex in range(vertex_count):
        destination = vertex * 3
        for control_index, coefficient in enumerate(stable_coefficients):
            if coefficient == 0.0:
                continue
            source = (vertex * control_count + control_index) * 16
            delta_x, delta_y, delta_z = struct.unpack_from(
                "<fff", bundle.deformation_rgba32f, source
            )
            output[destination] += coefficient * delta_x
            output[destination + 1] += coefficient * delta_y
            output[destination + 2] += coefficient * delta_z
    return tuple(output)


def compare_rinne_gpu_linear_reconstruction(
    bundle: RinneGpuLinearDynamicBundle,
    reference: RinneLegacyDrawPlan,
    actual: RinneLegacyDrawPlan,
    coefficients: Sequence[float],
) -> RinneGpuLinearReconstruction:
    verify_rinne_gpu_topology_compatible(reference, actual)
    predicted = reconstruct_rinne_gpu_linear_positions(
        bundle, reference, coefficients
    )
    actual_positions = flatten_rinne_gpu_positions(actual)
    differences = tuple(
        abs(actual_value - predicted_value)
        for actual_value, predicted_value in zip(
            actual_positions, predicted, strict=True
        )
    )
    tolerance = RINNE_GPU_LINEAR_RECONSTRUCTION_TOLERANCE
    components_above = sum(value > tolerance for value in differences)
    return RinneGpuLinearReconstruction(
        component_count=len(differences),
        maximum_absolute_error=max(differences, default=0.0),
        mean_absolute_error=(
            0.0 if not differences else sum(differences) / len(differences)
        ),
        components_above_tolerance=components_above,
        tolerance=tolerance,
        accepted=components_above == 0,
    )


def _require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BinaryBoundsError(f"GPU dynamic {label} must be an object")
    return value


def verify_rinne_gpu_linear_dynamic_bundle(
    bundle: RinneGpuLinearDynamicBundle,
) -> dict[str, Any]:
    if not isinstance(bundle, RinneGpuLinearDynamicBundle):
        raise TypeError("GPU linear verification requires a dynamic bundle")
    base = verify_rinne_gpu_bundle(bundle.base_bundle)
    if not 0 < len(bundle.dynamic_manifest_json) <= (
        RINNE_GPU_DYNAMIC_MANIFEST_MAX_BYTES
    ):
        raise BinaryBoundsError("GPU dynamic manifest size exceeds safety limit")
    try:
        manifest = json.loads(bundle.dynamic_manifest_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BinaryBoundsError("GPU dynamic manifest is invalid JSON") from exc
    root = _require_object(manifest, "manifest")
    if (
        root.get("format") != RINNE_GPU_LINEAR_DYNAMIC_FORMAT
        or root.get("version") != RINNE_GPU_LINEAR_DYNAMIC_VERSION
    ):
        raise BinaryBoundsError("GPU dynamic format or version is unsupported")
    if root.get("capabilities") != _CAPABILITIES:
        raise BinaryBoundsError("GPU linear capabilities are invalid")
    if root.get("unsupported_controls") != _UNSUPPORTED_CONTROLS:
        raise BinaryBoundsError("GPU linear unsupported-control boundary is invalid")
    base_reference = _require_object(root.get("base_bundle"), "base bundle")
    if (
        base_reference.get("manifest_sha256")
        != hashlib.sha256(bundle.base_bundle.manifest_json).hexdigest()
        or base_reference.get("texture_sha256") != base["texture"]["sha256"]
        or base_reference.get("geometry_sha256") != base["geometry"]["sha256"]
        or not isinstance(base_reference.get("topology_sha256"), str)
        or not _SHA256_PATTERN.fullmatch(base_reference["topology_sha256"])
    ):
        raise BinaryBoundsError("GPU dynamic base-bundle identity is invalid")
    deformation = _require_object(root.get("deformation"), "deformation")
    width = deformation.get("texture_width")
    height = deformation.get("texture_height")
    vertex_count = deformation.get("vertex_count")
    control_count = deformation.get("control_count")
    used_texels = deformation.get("used_texel_count")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in (width, height, vertex_count, control_count, used_texels)
    ):
        raise BinaryBoundsError("GPU dynamic deformation dimensions are invalid")
    assert isinstance(width, int)
    assert isinstance(height, int)
    assert isinstance(vertex_count, int)
    assert isinstance(control_count, int)
    assert isinstance(used_texels, int)
    if (
        deformation.get("file") != RINNE_GPU_DEFORMATION_FILENAME
        or deformation.get("format") != "rgba32f_little_endian"
        or width != RINNE_GPU_DEFORMATION_TEXTURE_WIDTH
        or control_count > RINNE_GPU_LINEAR_MAX_CONTROLS
        or used_texels != vertex_count * control_count
        or deformation.get("byte_length") != len(bundle.deformation_rgba32f)
        or len(bundle.deformation_rgba32f) != width * height * 16
        or len(bundle.deformation_rgba32f) > RINNE_GPU_LINEAR_MAX_BYTES
        or deformation.get("sha256")
        != hashlib.sha256(bundle.deformation_rgba32f).hexdigest()
    ):
        raise BinaryBoundsError("GPU dynamic deformation bytes are inconsistent")
    controls = deformation.get("controls")
    if not isinstance(controls, list) or len(controls) != control_count:
        raise BinaryBoundsError("GPU dynamic controls are invalid")
    names: set[str] = set()
    expression_count = len(base["reference"]["expression_weights"])
    expected_names = tuple(
        f"expression_{index}" for index in range(expression_count)
    ) + _PUPIL_CONTROL_NAMES
    if control_count != len(expected_names):
        raise BinaryBoundsError("GPU dynamic control count does not match source")
    for index, item in enumerate(controls):
        control = _require_object(item, "control")
        name = control.get("name")
        if (
            control.get("index") != index
            or not isinstance(name, str)
            or not name.isascii()
            or not name
            or name in names
            or name != expected_names[index]
            or control.get("coefficient_mode")
            not in (
                "threshold_0_01",
                "cosine_ease_then_threshold_0_01",
                "identity",
            )
        ):
            raise BinaryBoundsError("GPU dynamic control descriptor is invalid")
        mode = control["coefficient_mode"]
        if (
            index < expression_count
            and mode == "identity"
            or index >= expression_count
            and mode != "identity"
        ):
            raise BinaryBoundsError("GPU dynamic control coefficient mode is invalid")
        names.add(name)
    mesh_offsets = deformation.get("mesh_vertex_offsets")
    if not isinstance(mesh_offsets, list) or not mesh_offsets:
        raise BinaryBoundsError("GPU dynamic mesh offsets are invalid")
    base_meshes = tuple(
        (pass_item["name"], mesh)
        for pass_item in base["geometry"]["passes"]
        for mesh in pass_item["meshes"]
    )
    if len(mesh_offsets) != len(base_meshes):
        raise BinaryBoundsError("GPU dynamic mesh offsets changed base mesh count")
    vertex_cursor = 0
    for item, (base_pass, base_mesh) in zip(
        mesh_offsets, base_meshes, strict=True
    ):
        descriptor = _require_object(item, "mesh offset")
        count = descriptor.get("vertex_count")
        if (
            descriptor.get("vertex_offset") != vertex_cursor
            or descriptor.get("pass") != base_pass
            or descriptor.get("mesh_index") != base_mesh["mesh_index"]
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            or count != base_mesh["vertex_count"]
        ):
            raise BinaryBoundsError("GPU dynamic mesh offsets are not canonical")
        vertex_cursor += count
    if vertex_cursor != vertex_count:
        raise BinaryBoundsError("GPU dynamic mesh offsets do not cover vertices")
    for texel_index, values in enumerate(
        struct.iter_unpack("<ffff", bundle.deformation_rgba32f)
    ):
        if any(not math.isfinite(value) for value in values):
            raise BinaryBoundsError(
                f"GPU deformation contains a non-finite texel at {texel_index}"
            )
        if values[3] != 0.0 or (
            texel_index >= used_texels and values != (0.0, 0.0, 0.0, 0.0)
        ):
            raise BinaryBoundsError("GPU deformation padding is not canonical")
    if root.get("reconstruction") != {
        "maximum_position_error": RINNE_GPU_LINEAR_RECONSTRUCTION_TOLERANCE,
        "position_space": "projected_normalized_xyz_before_neck_pose",
    }:
        raise BinaryBoundsError("GPU dynamic reconstruction contract is invalid")
    if root.get("acceptance_sample") != {
        "raw_expression_weights": list(
            _acceptance_expression_weights(expression_count)
        ),
        "right_pupil_position": list(_ACCEPTANCE_RIGHT_PUPIL_POSITION),
        "left_pupil_position": list(_ACCEPTANCE_LEFT_PUPIL_POSITION),
    }:
        raise BinaryBoundsError("GPU dynamic acceptance sample is invalid")
    return root


def write_rinne_gpu_linear_dynamic_bundle(
    output_directory: Path | str,
    bundle: RinneGpuLinearDynamicBundle,
) -> None:
    verify_rinne_gpu_linear_dynamic_bundle(bundle)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=False)
    write_new_bytes(
        destination / RINNE_GPU_MANIFEST_FILENAME,
        bundle.base_bundle.manifest_json,
    )
    write_new_bytes(
        destination / RINNE_GPU_TEXTURE_FILENAME,
        bundle.base_bundle.texture_rgba8,
    )
    write_new_bytes(
        destination / RINNE_GPU_GEOMETRY_FILENAME,
        bundle.base_bundle.geometry_binary,
    )
    write_new_bytes(
        destination / RINNE_GPU_DYNAMIC_MANIFEST_FILENAME,
        bundle.dynamic_manifest_json,
    )
    write_new_bytes(
        destination / RINNE_GPU_DEFORMATION_FILENAME,
        bundle.deformation_rgba32f,
    )


__all__ = [
    "RINNE_GPU_DEFORMATION_FILENAME",
    "RINNE_GPU_DEFORMATION_TEXTURE_WIDTH",
    "RINNE_GPU_DYNAMIC_MANIFEST_MAX_BYTES",
    "RINNE_GPU_DYNAMIC_MANIFEST_FILENAME",
    "RINNE_GPU_LINEAR_DYNAMIC_FORMAT",
    "RINNE_GPU_LINEAR_DYNAMIC_VERSION",
    "RINNE_GPU_LINEAR_RECONSTRUCTION_TOLERANCE",
    "RinneGpuLinearControlSample",
    "RinneGpuLinearDynamicBundle",
    "RinneGpuLinearReconstruction",
    "build_rinne_gpu_linear_control_samples",
    "build_rinne_gpu_linear_dynamic_bundle",
    "compare_rinne_gpu_linear_reconstruction",
    "flatten_rinne_gpu_positions",
    "reconstruct_rinne_gpu_linear_positions",
    "resolve_rinne_gpu_linear_coefficients",
    "verify_rinne_gpu_linear_dynamic_bundle",
    "write_rinne_gpu_linear_dynamic_bundle",
]
