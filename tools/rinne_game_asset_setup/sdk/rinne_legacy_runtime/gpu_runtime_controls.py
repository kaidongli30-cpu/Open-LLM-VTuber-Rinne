from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

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
from .gpu_eye_dynamics import (
    RINNE_GPU_EYE_DEFORMATION_FILENAME,
    RINNE_GPU_EYE_MANIFEST_FILENAME,
    RinneGpuEyeDynamicBundle,
    verify_rinne_gpu_eye_dynamic_bundle,
)
from .gpu_linear_dynamics import (
    RINNE_GPU_DEFORMATION_FILENAME,
    RINNE_GPU_DYNAMIC_MANIFEST_FILENAME,
)
from .gpu_topology import (
    RINNE_GPU_DYNAMIC_PASS_FIELDS,
    verify_rinne_gpu_topology_compatible,
)
from .io_safety import write_new_bytes


RINNE_GPU_RUNTIME_CONTROL_FORMAT = "rinne-legacy-gpu-runtime-controls"
RINNE_GPU_RUNTIME_CONTROL_VERSION = 1
RINNE_GPU_RUNTIME_CONTROL_MANIFEST_FILENAME = "runtime-control-manifest.json"
RINNE_GPU_RUNTIME_CONTROL_MANIFEST_MAX_BYTES = 4 * 1024 * 1024
_CAPABILITIES = ["neck_pose", "render_record_opacity"]
_UNSUPPORTED_CONTROLS = ["amb_playback"]
_ACCEPTANCE_NECK_ROTATION = (2.0, -1.5, 1.0)
_ACCEPTANCE_NECK_TRANSLATION = (0.015, -0.01, 0.005)


@dataclass(frozen=True)
class RinneGpuRuntimeControlBundle:
    eye_bundle: RinneGpuEyeDynamicBundle
    runtime_manifest_json: bytes

    def __post_init__(self) -> None:
        verify_rinne_gpu_runtime_control_bundle(self)

    def runtime_manifest(self) -> dict[str, Any]:
        decoded = json.loads(self.runtime_manifest_json)
        if not isinstance(decoded, dict):
            raise BinaryBoundsError("GPU runtime manifest root must be an object")
        return decoded


def _f32(value: float) -> float:
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise BinaryBoundsError("GPU runtime float is outside float32 range") from exc
    if not math.isfinite(result):
        raise BinaryBoundsError("GPU runtime float must be finite")
    return result


def _acceptance_record_opacities(
    count: int,
    *,
    neutral_indices: tuple[int, ...] = (),
) -> tuple[float, ...]:
    neutral = frozenset(neutral_indices)
    return tuple(
        1.0 if index in neutral else _f32(0.8 + 0.05 * (index % 5))
        for index in range(count)
    )


def _pass_meshes(plan: RinneLegacyDrawPlan):
    for pass_field in RINNE_GPU_DYNAMIC_PASS_FIELDS:
        pass_name = pass_field.removesuffix("_meshes")
        for mesh_index, mesh in enumerate(getattr(plan, pass_field)):
            yield (pass_name, mesh_index), mesh


def _type2_pass_keys(reference: RinneLegacyDrawPlan) -> set[tuple[str, int]]:
    type2_ids = {id(mesh) for mesh in reference.type2_eye_strip_meshes}
    return {
        key
        for key, mesh in _pass_meshes(reference)
        if id(mesh) in type2_ids
    }


def _build_opacity_bindings(
    renderer: RinneLegacyFrameRenderer,
    reference: RinneLegacyDrawPlan,
) -> tuple[dict[str, object], ...]:
    records = tuple(
        sorted(renderer.assets.atlas.draw_records, key=lambda item: item.file_index)
    )
    defaults = tuple(record.initial_opacity for record in records)
    reference_meshes = dict(_pass_meshes(reference))
    assignments: dict[tuple[str, int], int] = {}
    for record_index, default in enumerate(defaults):
        values = list(defaults)
        values[record_index] = 0.25 if default == 0.5 else 0.5
        candidate = renderer.build_draw_plan(
            RinneLegacyFrameControls(render_record_opacities=tuple(values)),
            width=reference.width,
            height=reference.height,
        )
        verify_rinne_gpu_topology_compatible(reference, candidate)
        candidate_meshes = dict(_pass_meshes(candidate))
        for key, reference_mesh in reference_meshes.items():
            if candidate_meshes[key].opacity != reference_mesh.opacity:
                previous = assignments.get(key)
                if previous is not None and previous != record_index:
                    raise BinaryBoundsError(
                        "GPU runtime mesh responds to multiple opacity records"
                    )
                assignments[key] = record_index
    type2_records = tuple(record for record in records if record.type_id == 2)
    if len(type2_records) != 1:
        raise BinaryBoundsError("GPU runtime requires one type-2 draw record")
    for key in _type2_pass_keys(reference):
        assignments[key] = type2_records[0].file_index
    if set(assignments) != set(reference_meshes):
        missing = sorted(set(reference_meshes) - set(assignments))
        raise BinaryBoundsError(
            f"GPU runtime opacity binding coverage is incomplete: {missing}"
        )
    return tuple(
        {
            "pass": pass_name,
            "mesh_index": mesh_index,
            "record_index": assignments[(pass_name, mesh_index)],
            "curve": (
                "type0_fourth_power"
                if records[assignments[(pass_name, mesh_index)]].type_id == 0
                else "direct"
            ),
        }
        for pass_name, mesh_index in reference_meshes
    )


def build_rinne_gpu_runtime_control_bundle(
    eye_bundle: RinneGpuEyeDynamicBundle,
    renderer: RinneLegacyFrameRenderer,
    reference: RinneLegacyDrawPlan,
) -> RinneGpuRuntimeControlBundle:
    eye_manifest = verify_rinne_gpu_eye_dynamic_bundle(eye_bundle)
    if not isinstance(renderer, RinneLegacyFrameRenderer):
        raise TypeError("GPU runtime controls require a frame renderer")
    if not isinstance(reference, RinneLegacyDrawPlan):
        raise TypeError("GPU runtime controls require a reference draw plan")
    base_source = eye_bundle.linear_bundle.base_bundle.manifest()["source"]
    if base_source["parsed_member_sha256"]["face.mpb"] != (
        renderer.assets.mpb_sha256
    ):
        raise BinaryBoundsError("GPU runtime renderer does not match eye bundle")
    records = tuple(
        sorted(renderer.assets.atlas.draw_records, key=lambda item: item.file_index)
    )
    if tuple(record.file_index for record in records) != tuple(range(len(records))):
        raise BinaryBoundsError("GPU runtime draw records are not contiguous")
    opacity_bindings = _build_opacity_bindings(renderer, reference)
    manifest = {
        "format": RINNE_GPU_RUNTIME_CONTROL_FORMAT,
        "version": RINNE_GPU_RUNTIME_CONTROL_VERSION,
        "capabilities": _CAPABILITIES,
        "unsupported_controls": _UNSUPPORTED_CONTROLS,
        "eye_bundle": {
            "manifest_sha256": hashlib.sha256(
                eye_bundle.eye_manifest_json
            ).hexdigest(),
            "deformation_sha256": hashlib.sha256(
                eye_bundle.eye_deformation_rgba32f
            ).hexdigest(),
            "base_topology_sha256": eye_manifest["linear_bundle"][
                "base_topology_sha256"
            ],
        },
        "neck": {
            "pivot_xyz": list(renderer.assets.neck_profile.pivot_xyz),
            "matrix_convention": "row_vector_x_then_y_then_z_float32",
            "input_rotation_unit": "degrees",
            "input_translation_space": "legacy_model_xyz",
        },
        "opacity": {
            "record_count": len(records),
            "records": [
                {
                    "index": record.file_index,
                    "type_id": record.type_id,
                    "initial_opacity": record.initial_opacity,
                    "curve": (
                        "type0_fourth_power"
                        if record.type_id == 0
                        else "direct"
                    ),
                }
                for record in records
            ],
            "mesh_bindings": list(opacity_bindings),
        },
        "acceptance_sample": {
            "neck_rotation": list(_ACCEPTANCE_NECK_ROTATION),
            "neck_translation": list(_ACCEPTANCE_NECK_TRANSLATION),
            "record_opacities": list(
                _acceptance_record_opacities(
                    len(records),
                    neutral_indices=tuple(
                        record.file_index
                        for record in records
                        if record.type_id == 25
                    ),
                )
            ),
        },
    }
    runtime_manifest_json = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return RinneGpuRuntimeControlBundle(
        eye_bundle=eye_bundle,
        runtime_manifest_json=runtime_manifest_json,
    )


def _require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BinaryBoundsError(f"GPU runtime {label} must be an object")
    return value


def verify_rinne_gpu_runtime_control_bundle(
    bundle: RinneGpuRuntimeControlBundle,
) -> dict[str, Any]:
    if not isinstance(bundle, RinneGpuRuntimeControlBundle):
        raise TypeError("GPU runtime verification requires a runtime bundle")
    eye = verify_rinne_gpu_eye_dynamic_bundle(bundle.eye_bundle)
    if not 0 < len(bundle.runtime_manifest_json) <= (
        RINNE_GPU_RUNTIME_CONTROL_MANIFEST_MAX_BYTES
    ):
        raise BinaryBoundsError("GPU runtime manifest size exceeds safety limit")
    try:
        manifest = json.loads(bundle.runtime_manifest_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BinaryBoundsError("GPU runtime manifest is invalid JSON") from exc
    root = _require_object(manifest, "manifest")
    if (
        root.get("format") != RINNE_GPU_RUNTIME_CONTROL_FORMAT
        or root.get("version") != RINNE_GPU_RUNTIME_CONTROL_VERSION
        or root.get("capabilities") != _CAPABILITIES
        or root.get("unsupported_controls") != _UNSUPPORTED_CONTROLS
    ):
        raise BinaryBoundsError("GPU runtime format or capabilities are invalid")
    if root.get("eye_bundle") != {
        "manifest_sha256": hashlib.sha256(
            bundle.eye_bundle.eye_manifest_json
        ).hexdigest(),
        "deformation_sha256": hashlib.sha256(
            bundle.eye_bundle.eye_deformation_rgba32f
        ).hexdigest(),
        "base_topology_sha256": eye["linear_bundle"]["base_topology_sha256"],
    }:
        raise BinaryBoundsError("GPU runtime eye-bundle identity is invalid")
    neck = _require_object(root.get("neck"), "neck")
    pivot = neck.get("pivot_xyz")
    if (
        not isinstance(pivot, list)
        or len(pivot) != 3
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in pivot
        )
        or neck.get("matrix_convention")
        != "row_vector_x_then_y_then_z_float32"
        or neck.get("input_rotation_unit") != "degrees"
        or neck.get("input_translation_space") != "legacy_model_xyz"
    ):
        raise BinaryBoundsError("GPU runtime neck contract is invalid")
    opacity = _require_object(root.get("opacity"), "opacity")
    record_count = opacity.get("record_count")
    records = opacity.get("records")
    if (
        not isinstance(record_count, int)
        or isinstance(record_count, bool)
        or record_count <= 0
        or record_count > 256
        or not isinstance(records, list)
        or len(records) != record_count
    ):
        raise BinaryBoundsError("GPU runtime opacity records are invalid")
    type_ids: set[int] = set()
    for index, item in enumerate(records):
        record = _require_object(item, "opacity record")
        value = record.get("initial_opacity")
        type_id = record.get("type_id")
        if (
            record.get("index") != index
            or not isinstance(type_id, int)
            or isinstance(type_id, bool)
            or type_id in type_ids
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or float(value) > 1.0
            or record.get("curve")
            != (
                "type0_fourth_power"
                if record.get("type_id") == 0
                else "direct"
            )
        ):
            raise BinaryBoundsError("GPU runtime opacity record is inconsistent")
        type_ids.add(type_id)
    linear = bundle.eye_bundle.linear_bundle.dynamic_manifest()
    base_keys = {
        (item["pass"], item["mesh_index"])
        for item in linear["deformation"]["mesh_vertex_offsets"]
    }
    bindings = opacity.get("mesh_bindings")
    if not isinstance(bindings, list) or len(bindings) != len(base_keys):
        raise BinaryBoundsError("GPU runtime opacity bindings are invalid")
    seen = set()
    for item in bindings:
        binding = _require_object(item, "opacity binding")
        key = (binding.get("pass"), binding.get("mesh_index"))
        record_index = binding.get("record_index")
        if (
            key not in base_keys
            or key in seen
            or not isinstance(record_index, int)
            or isinstance(record_index, bool)
            or record_index < 0
            or record_index >= record_count
            or binding.get("curve") != records[record_index]["curve"]
        ):
            raise BinaryBoundsError("GPU runtime opacity binding is inconsistent")
        seen.add(key)
    if seen != base_keys:
        raise BinaryBoundsError("GPU runtime opacity binding coverage changed")
    acceptance = _require_object(root.get("acceptance_sample"), "acceptance")
    if acceptance != {
        "neck_rotation": list(_ACCEPTANCE_NECK_ROTATION),
        "neck_translation": list(_ACCEPTANCE_NECK_TRANSLATION),
        "record_opacities": list(
            _acceptance_record_opacities(
                record_count,
                neutral_indices=tuple(
                    record["index"] for record in records if record["type_id"] == 25
                ),
            )
        ),
    }:
        raise BinaryBoundsError("GPU runtime acceptance sample is invalid")
    return root


def acceptance_rinne_gpu_runtime_controls(
    base_controls: RinneLegacyFrameControls,
    runtime_bundle: RinneGpuRuntimeControlBundle,
) -> RinneLegacyFrameControls:
    if not isinstance(base_controls, RinneLegacyFrameControls):
        raise TypeError("GPU runtime acceptance requires frame controls")
    manifest = verify_rinne_gpu_runtime_control_bundle(runtime_bundle)
    sample = manifest["acceptance_sample"]
    return replace(
        base_controls,
        neck_rotation=tuple(sample["neck_rotation"]),
        neck_translation=tuple(sample["neck_translation"]),
        render_record_opacities=tuple(sample["record_opacities"]),
    )


def write_rinne_gpu_runtime_control_bundle(
    output_directory: Path | str,
    bundle: RinneGpuRuntimeControlBundle,
) -> None:
    verify_rinne_gpu_runtime_control_bundle(bundle)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=False)
    eye = bundle.eye_bundle
    linear = eye.linear_bundle
    base = linear.base_bundle
    for filename, payload in (
        (RINNE_GPU_MANIFEST_FILENAME, base.manifest_json),
        (RINNE_GPU_TEXTURE_FILENAME, base.texture_rgba8),
        (RINNE_GPU_GEOMETRY_FILENAME, base.geometry_binary),
        (RINNE_GPU_DYNAMIC_MANIFEST_FILENAME, linear.dynamic_manifest_json),
        (RINNE_GPU_DEFORMATION_FILENAME, linear.deformation_rgba32f),
        (RINNE_GPU_EYE_MANIFEST_FILENAME, eye.eye_manifest_json),
        (RINNE_GPU_EYE_DEFORMATION_FILENAME, eye.eye_deformation_rgba32f),
        (RINNE_GPU_RUNTIME_CONTROL_MANIFEST_FILENAME, bundle.runtime_manifest_json),
    ):
        write_new_bytes(destination / filename, payload)


__all__ = [
    "RINNE_GPU_RUNTIME_CONTROL_FORMAT",
    "RINNE_GPU_RUNTIME_CONTROL_MANIFEST_FILENAME",
    "RINNE_GPU_RUNTIME_CONTROL_VERSION",
    "RinneGpuRuntimeControlBundle",
    "acceptance_rinne_gpu_runtime_controls",
    "build_rinne_gpu_runtime_control_bundle",
    "verify_rinne_gpu_runtime_control_bundle",
    "write_rinne_gpu_runtime_control_bundle",
]
