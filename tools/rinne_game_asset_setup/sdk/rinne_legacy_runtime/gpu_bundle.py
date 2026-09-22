from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .asset_manifest import RinnePckVerification
from .checked_binary import BinaryBoundsError
from .frame_renderer import (
    RinneLegacyAssets,
    RinneLegacyDrawPlan,
    rasterize_rinne_legacy_draw_plan,
)
from .io_safety import write_new_bytes
from .textured_mesh import TexturedDepthMesh, _validate_depth_mesh


RINNE_GPU_BUNDLE_FORMAT = "rinne-legacy-gpu-bundle"
RINNE_GPU_BUNDLE_VERSION = 1
RINNE_GPU_MANIFEST_FILENAME = "manifest.json"
RINNE_GPU_TEXTURE_FILENAME = "texture.rgba8"
RINNE_GPU_GEOMETRY_FILENAME = "geometry.bin"
RINNE_GPU_MANIFEST_MAX_BYTES = 4 * 1024 * 1024
RINNE_GPU_TEXTURE_MAX_BYTES = 128 * 1024 * 1024
RINNE_GPU_GEOMETRY_MAX_BYTES = 256 * 1024 * 1024
RINNE_GPU_PASS_NAMES = (
    "eye_buffer",
    "mask_buffer",
    "before_trigger",
    "final_eye_overlay",
    "after_trigger",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _is_supported_rinne_portrait_id(portrait_id: object) -> bool:
    return (
        isinstance(portrait_id, int)
        and not isinstance(portrait_id, bool)
        and (
            (601 <= portrait_id // 100 <= 604 and 1 <= portrait_id % 100 <= 15)
            or (portrait_id // 100 == 1601 and 1 <= portrait_id % 100 <= 7)
        )
    )


@dataclass(frozen=True)
class RinneGpuSourceEvidence:
    portrait_id: int
    filename: str
    pck_size: int
    pck_sha256: str
    mpb_sha256: str
    texture_container_sha256: str
    auto_animation_config_sha256: str
    animation_sha256: str

    def __post_init__(self) -> None:
        if not _is_supported_rinne_portrait_id(self.portrait_id):
            raise BinaryBoundsError(
                "GPU source portrait id must be a 601xx..604xx family id with "
                "suffix 01..15 or a 1601xx spirit-dress id with suffix 01..07"
            )
        if self.filename != f"MP{self.portrait_id:06d}.pck":
            raise BinaryBoundsError("GPU source filename does not match portrait id")
        if not isinstance(self.pck_size, int) or self.pck_size <= 0:
            raise BinaryBoundsError("GPU source PCK size must be positive")
        for label, digest in (
            ("PCK", self.pck_sha256),
            ("MPB", self.mpb_sha256),
            ("texture container", self.texture_container_sha256),
            ("automatic animation config", self.auto_animation_config_sha256),
            ("animation", self.animation_sha256),
        ):
            if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
                raise BinaryBoundsError(
                    f"GPU source {label} SHA-256 must be lowercase hexadecimal"
                )

    @classmethod
    def from_verified_assets(
        cls,
        verification: RinnePckVerification,
        assets: RinneLegacyAssets,
    ) -> RinneGpuSourceEvidence:
        if not isinstance(verification, RinnePckVerification):
            raise TypeError("GPU source requires verified Rinne PCK evidence")
        if not isinstance(assets, RinneLegacyAssets):
            raise TypeError("GPU source requires parsed Rinne assets")
        if verification.path != assets.pck_path.resolve(strict=True):
            raise BinaryBoundsError(
                "verified PCK and parsed GPU assets do not refer to the same file"
            )
        return cls(
            portrait_id=verification.spec.portrait_id,
            filename=verification.spec.filename,
            pck_size=verification.spec.size,
            pck_sha256=verification.sha256,
            mpb_sha256=assets.mpb_sha256,
            texture_container_sha256=assets.texture_container_sha256,
            auto_animation_config_sha256=(assets.auto_animation_config_sha256),
            animation_sha256=assets.animation_sha256,
        )


@dataclass(frozen=True)
class RinneGpuBundle:
    manifest_json: bytes
    texture_rgba8: bytes
    geometry_binary: bytes

    def __post_init__(self) -> None:
        verify_rinne_gpu_bundle(self)

    def manifest(self) -> dict[str, Any]:
        decoded = json.loads(self.manifest_json)
        if not isinstance(decoded, dict):
            raise BinaryBoundsError("GPU bundle manifest root must be an object")
        return decoded


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _append_f32(buffer: bytearray, values: list[float]) -> dict[str, int | str]:
    offset = len(buffer)
    try:
        payload = struct.pack(f"<{len(values)}f", *values)
    except (OverflowError, struct.error) as exc:
        raise BinaryBoundsError("GPU float32 buffer contains an invalid value") from exc
    buffer.extend(payload)
    return {
        "byte_offset": offset,
        "byte_length": len(payload),
        "component_type": "float32",
        "component_count": len(values),
    }


def _append_u32(buffer: bytearray, values: list[int]) -> dict[str, int | str]:
    offset = len(buffer)
    try:
        payload = struct.pack(f"<{len(values)}I", *values)
    except struct.error as exc:
        raise BinaryBoundsError("GPU index buffer contains an invalid value") from exc
    buffer.extend(payload)
    return {
        "byte_offset": offset,
        "byte_length": len(payload),
        "component_type": "uint32",
        "component_count": len(values),
    }


def _pack_mesh(
    buffer: bytearray,
    mesh: TexturedDepthMesh,
    *,
    mesh_index: int,
) -> dict[str, Any]:
    _validate_depth_mesh(mesh)
    positions = [component for item in mesh.positions_xyz for component in item]
    uvs = [component for item in mesh.uvs for component in item]
    indices = [int(index) for index in mesh.triangle_indices]
    descriptor: dict[str, Any] = {
        "mesh_index": mesh_index,
        "vertex_count": len(mesh.positions_xyz),
        "index_count": len(mesh.triangle_indices),
        "triangle_count": len(mesh.triangle_indices) // 3,
        "opacity": mesh.opacity,
        "depth_write": mesh.depth_write,
        "texture_filter": mesh.texture_filter or "nearest",
        "positions": _append_f32(buffer, positions),
        "uvs": _append_f32(buffer, uvs),
        "indices": _append_u32(buffer, indices),
    }
    if mesh.vertex_opacities is not None:
        descriptor["vertex_opacities"] = _append_f32(
            buffer,
            [float(value) for value in mesh.vertex_opacities],
        )
    return descriptor


def _source_manifest(source: RinneGpuSourceEvidence) -> dict[str, Any]:
    return {
        "portrait_id": source.portrait_id,
        "filename": source.filename,
        "pck_size": source.pck_size,
        "pck_sha256": source.pck_sha256,
        "parsed_member_sha256": {
            "face.mpb": source.mpb_sha256,
            "tex_all.tex": source.texture_container_sha256,
            "face.uca.bin": source.auto_animation_config_sha256,
            "001.amb": source.animation_sha256,
        },
    }


def build_rinne_gpu_bundle(
    plan: RinneLegacyDrawPlan,
    source: RinneGpuSourceEvidence,
) -> RinneGpuBundle:
    """Build a deterministic local-only WebGL2 reference bundle.

    Version 1 intentionally contains one resolved draw plan.  Its declared
    capabilities prevent a consumer from mistaking it for the later dynamic
    control transport.
    """

    if not isinstance(plan, RinneLegacyDrawPlan):
        raise TypeError("GPU bundle requires a RinneLegacyDrawPlan")
    if not isinstance(source, RinneGpuSourceEvidence):
        raise TypeError("GPU bundle requires checked source evidence")
    texture = plan.texture
    expected_texture_bytes = texture.width * texture.height * 4
    if (
        texture.width <= 0
        or texture.height <= 0
        or len(texture.rgba) != expected_texture_bytes
        or expected_texture_bytes > RINNE_GPU_TEXTURE_MAX_BYTES
    ):
        raise BinaryBoundsError("GPU bundle texture dimensions or bytes are invalid")

    pass_meshes = (
        plan.eye_buffer_meshes,
        plan.mask_buffer_meshes,
        plan.before_trigger_meshes,
        plan.final_eye_overlay_meshes,
        plan.after_trigger_meshes,
    )
    geometry = bytearray()
    passes: list[dict[str, Any]] = []
    for pass_name, meshes in zip(RINNE_GPU_PASS_NAMES, pass_meshes, strict=True):
        passes.append(
            {
                "name": pass_name,
                "meshes": [
                    _pack_mesh(geometry, mesh, mesh_index=index)
                    for index, mesh in enumerate(meshes)
                ],
            }
        )
    geometry_binary = bytes(geometry)
    if len(geometry_binary) > RINNE_GPU_GEOMETRY_MAX_BYTES:
        raise BinaryBoundsError("GPU geometry exceeds bundle safety limit")
    reference_frame = rasterize_rinne_legacy_draw_plan(plan)
    manifest = {
        "format": RINNE_GPU_BUNDLE_FORMAT,
        "version": RINNE_GPU_BUNDLE_VERSION,
        "capabilities": [
            "resolved_static_draw_plan",
            "original_rgba8_atlas",
            "legacy_delayed_eye_pass_graph",
            "cpu_reference_hash",
        ],
        "dynamic_controls": [],
        "runtime_requirements": {
            "graphics_api": "WebGL2",
            "little_endian_buffers": True,
            "uint32_element_indices": True,
        },
        "source": _source_manifest(source),
        "reference": {
            "canvas_width": plan.width,
            "canvas_height": plan.height,
            "rgba_sha256": _sha256(reference_frame.rgba),
            "expression_weights": list(plan.expression_weights),
            "blink_deformation": list(plan.blink_geometry.deformation),
            "blink_base_weights": list(plan.blink_geometry.base_weights),
            "compositor_trigger_type": plan.compositor_trigger_type,
        },
        "texture": {
            "file": RINNE_GPU_TEXTURE_FILENAME,
            "width": texture.width,
            "height": texture.height,
            "format": "rgba8_unorm",
            "row_order": "top_to_bottom",
            "byte_length": len(texture.rgba),
            "sha256": _sha256(texture.rgba),
        },
        "geometry": {
            "file": RINNE_GPU_GEOMETRY_FILENAME,
            "byte_length": len(geometry_binary),
            "sha256": _sha256(geometry_binary),
            "passes": passes,
        },
        "compositor": {
            "draw_depth_compare": "disabled",
            "draw_blend": "legacy_mode0",
            "eye_clip": "eye_alpha_times_one_minus_mask_alpha",
            "final_order": [
                "before_trigger",
                "clipped_eye_buffer",
                "final_eye_overlay",
                "after_trigger",
            ],
        },
    }
    manifest_json = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return RinneGpuBundle(
        manifest_json=manifest_json,
        texture_rgba8=texture.rgba,
        geometry_binary=geometry_binary,
    )


def _require_manifest_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BinaryBoundsError(f"GPU bundle {label} must be an object")
    return value


def _verify_buffer_view(
    value: object,
    *,
    geometry_length: int,
    expected_component_type: str,
    expected_component_count: int,
) -> tuple[int, int, int]:
    view = _require_manifest_object(value, "buffer view")
    if view.get("component_type") != expected_component_type:
        raise BinaryBoundsError("GPU buffer view has the wrong component type")
    offset = view.get("byte_offset")
    length = view.get("byte_length")
    count = view.get("component_count")
    if not all(
        isinstance(item, int) and not isinstance(item, bool)
        for item in (offset, length, count)
    ):
        raise BinaryBoundsError("GPU buffer view bounds must be integers")
    assert isinstance(offset, int)
    assert isinstance(length, int)
    assert isinstance(count, int)
    if (
        offset < 0
        or length < 0
        or count != expected_component_count
        or length != count * 4
        or offset % 4
        or offset + length > geometry_length
    ):
        raise BinaryBoundsError("GPU buffer view exceeds geometry bounds")
    return offset, length, count


def _verify_source_manifest(value: object) -> None:
    source = _require_manifest_object(value, "source")
    portrait_id = source.get("portrait_id")
    filename = source.get("filename")
    pck_size = source.get("pck_size")
    if (
        not _is_supported_rinne_portrait_id(portrait_id)
        or filename != f"MP{portrait_id:06d}.pck"
        or not isinstance(pck_size, int)
        or isinstance(pck_size, bool)
        or pck_size <= 0
        or not isinstance(source.get("pck_sha256"), str)
        or not _SHA256_PATTERN.fullmatch(source["pck_sha256"])
    ):
        raise BinaryBoundsError("GPU source identity is invalid")
    member_hashes = _require_manifest_object(
        source.get("parsed_member_sha256"), "parsed member hashes"
    )
    expected_names = ("face.mpb", "tex_all.tex", "face.uca.bin", "001.amb")
    if set(member_hashes) != set(expected_names) or any(
        not isinstance(member_hashes[name], str)
        or not _SHA256_PATTERN.fullmatch(member_hashes[name])
        for name in expected_names
    ):
        raise BinaryBoundsError("GPU parsed member hashes are invalid")


def _verify_reference_manifest(value: object) -> None:
    reference = _require_manifest_object(value, "reference")
    width = reference.get("canvas_width")
    height = reference.get("canvas_height")
    trigger = reference.get("compositor_trigger_type")
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or width <= 0
        or not isinstance(height, int)
        or isinstance(height, bool)
        or height <= 0
        or not isinstance(trigger, int)
        or isinstance(trigger, bool)
        or trigger < 0
        or not isinstance(reference.get("rgba_sha256"), str)
        or not _SHA256_PATTERN.fullmatch(reference["rgba_sha256"])
    ):
        raise BinaryBoundsError("GPU reference identity is invalid")
    for label, expected_length in (
        ("expression_weights", None),
        ("blink_deformation", 4),
        ("blink_base_weights", 4),
    ):
        values = reference.get(label)
        if (
            not isinstance(values, list)
            or not values
            or (expected_length is not None and len(values) != expected_length)
            or any(
                not isinstance(item, (int, float))
                or isinstance(item, bool)
                or not math.isfinite(float(item))
                for item in values
            )
        ):
            raise BinaryBoundsError(f"GPU reference {label} is invalid")


def _verify_f32_payload(
    geometry: bytes,
    view: tuple[int, int, int],
    *,
    label: str,
) -> None:
    offset, length, _count = view
    payload = memoryview(geometry)[offset : offset + length]
    for index, (value,) in enumerate(struct.iter_unpack("<f", payload)):
        if not math.isfinite(value):
            raise BinaryBoundsError(f"GPU {label} contains non-finite float at {index}")


def _verify_u32_indices(
    geometry: bytes,
    view: tuple[int, int, int],
    *,
    vertex_count: int,
) -> None:
    offset, length, _count = view
    payload = memoryview(geometry)[offset : offset + length]
    for index, (value,) in enumerate(struct.iter_unpack("<I", payload)):
        if value >= vertex_count:
            raise BinaryBoundsError(
                f"GPU index {value} at {index} exceeds vertex count {vertex_count}"
            )


def verify_rinne_gpu_bundle(bundle: RinneGpuBundle) -> dict[str, Any]:
    if not isinstance(bundle, RinneGpuBundle):
        raise TypeError("GPU bundle verification requires RinneGpuBundle")
    if not 0 < len(bundle.manifest_json) <= RINNE_GPU_MANIFEST_MAX_BYTES:
        raise BinaryBoundsError("GPU manifest size exceeds safety limit")
    if len(bundle.texture_rgba8) > RINNE_GPU_TEXTURE_MAX_BYTES:
        raise BinaryBoundsError("GPU texture size exceeds safety limit")
    if len(bundle.geometry_binary) > RINNE_GPU_GEOMETRY_MAX_BYTES:
        raise BinaryBoundsError("GPU geometry size exceeds safety limit")
    try:
        manifest = json.loads(bundle.manifest_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BinaryBoundsError("GPU manifest is not valid UTF-8 JSON") from exc
    root = _require_manifest_object(manifest, "manifest root")
    if root.get("format") != RINNE_GPU_BUNDLE_FORMAT:
        raise BinaryBoundsError("GPU bundle format is unsupported")
    if root.get("version") != RINNE_GPU_BUNDLE_VERSION:
        raise BinaryBoundsError("GPU bundle version is unsupported")
    expected_capabilities = [
        "resolved_static_draw_plan",
        "original_rgba8_atlas",
        "legacy_delayed_eye_pass_graph",
        "cpu_reference_hash",
    ]
    if root.get("capabilities") != expected_capabilities:
        raise BinaryBoundsError("GPU bundle capabilities are invalid")
    if root.get("dynamic_controls") != []:
        raise BinaryBoundsError("GPU bundle v1 must not claim dynamic controls")
    if root.get("runtime_requirements") != {
        "graphics_api": "WebGL2",
        "little_endian_buffers": True,
        "uint32_element_indices": True,
    }:
        raise BinaryBoundsError("GPU runtime requirements are invalid")
    _verify_source_manifest(root.get("source"))
    _verify_reference_manifest(root.get("reference"))

    texture = _require_manifest_object(root.get("texture"), "texture")
    width = texture.get("width")
    height = texture.get("height")
    if not all(
        isinstance(item, int) and not isinstance(item, bool) and item > 0
        for item in (width, height)
    ):
        raise BinaryBoundsError("GPU texture dimensions must be positive integers")
    assert isinstance(width, int)
    assert isinstance(height, int)
    if (
        texture.get("file") != RINNE_GPU_TEXTURE_FILENAME
        or texture.get("format") != "rgba8_unorm"
        or texture.get("row_order") != "top_to_bottom"
        or texture.get("byte_length") != len(bundle.texture_rgba8)
        or len(bundle.texture_rgba8) != width * height * 4
        or texture.get("sha256") != _sha256(bundle.texture_rgba8)
    ):
        raise BinaryBoundsError("GPU texture metadata or hash does not match bytes")

    geometry = _require_manifest_object(root.get("geometry"), "geometry")
    if (
        geometry.get("file") != RINNE_GPU_GEOMETRY_FILENAME
        or geometry.get("byte_length") != len(bundle.geometry_binary)
        or geometry.get("sha256") != _sha256(bundle.geometry_binary)
    ):
        raise BinaryBoundsError("GPU geometry metadata or hash does not match bytes")
    passes = geometry.get("passes")
    if (
        not isinstance(passes, list)
        or tuple(
            item.get("name") if isinstance(item, dict) else None for item in passes
        )
        != RINNE_GPU_PASS_NAMES
    ):
        raise BinaryBoundsError("GPU geometry pass graph is invalid")
    geometry_cursor = 0
    total_meshes = 0
    for pass_item in passes:
        pass_object = _require_manifest_object(pass_item, "pass")
        meshes = pass_object.get("meshes")
        if not isinstance(meshes, list):
            raise BinaryBoundsError("GPU pass meshes must be an array")
        total_meshes += len(meshes)
        if total_meshes > 256:
            raise BinaryBoundsError("GPU bundle mesh count exceeds safety limit")
        for expected_mesh_index, item in enumerate(meshes):
            mesh = _require_manifest_object(item, "mesh")
            vertex_count = mesh.get("vertex_count")
            index_count = mesh.get("index_count")
            if not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (vertex_count, index_count)
            ):
                raise BinaryBoundsError("GPU mesh counts must be nonnegative integers")
            assert isinstance(vertex_count, int)
            assert isinstance(index_count, int)
            opacity = mesh.get("opacity")
            if (
                mesh.get("mesh_index") != expected_mesh_index
                or mesh.get("triangle_count") != index_count // 3
                or index_count % 3
                or not isinstance(opacity, (int, float))
                or isinstance(opacity, bool)
                or not math.isfinite(float(opacity))
                or not 0.0 <= float(opacity) <= 1.0
                or mesh.get("texture_filter") not in ("nearest", "linear")
                or not isinstance(mesh.get("depth_write"), bool)
            ):
                raise BinaryBoundsError("GPU mesh metadata is invalid")
            positions_view = _verify_buffer_view(
                mesh.get("positions"),
                geometry_length=len(bundle.geometry_binary),
                expected_component_type="float32",
                expected_component_count=vertex_count * 3,
            )
            uvs_view = _verify_buffer_view(
                mesh.get("uvs"),
                geometry_length=len(bundle.geometry_binary),
                expected_component_type="float32",
                expected_component_count=vertex_count * 2,
            )
            indices_view = _verify_buffer_view(
                mesh.get("indices"),
                geometry_length=len(bundle.geometry_binary),
                expected_component_type="uint32",
                expected_component_count=index_count,
            )
            views = [positions_view, uvs_view, indices_view]
            if "vertex_opacities" in mesh:
                views.append(
                    _verify_buffer_view(
                        mesh["vertex_opacities"],
                        geometry_length=len(bundle.geometry_binary),
                        expected_component_type="float32",
                        expected_component_count=vertex_count,
                    )
                )
            for view in views:
                if view[0] != geometry_cursor:
                    raise BinaryBoundsError(
                        "GPU buffer views must cover geometry in canonical order"
                    )
                geometry_cursor += view[1]
            _verify_f32_payload(
                bundle.geometry_binary,
                positions_view,
                label="positions",
            )
            _verify_f32_payload(
                bundle.geometry_binary,
                uvs_view,
                label="UVs",
            )
            _verify_u32_indices(
                bundle.geometry_binary,
                indices_view,
                vertex_count=vertex_count,
            )
            if "vertex_opacities" in mesh:
                _verify_f32_payload(
                    bundle.geometry_binary,
                    views[-1],
                    label="vertex opacities",
                )
    if geometry_cursor != len(bundle.geometry_binary):
        raise BinaryBoundsError("GPU geometry contains unreferenced trailing bytes")
    if root.get("compositor") != {
        "draw_depth_compare": "disabled",
        "draw_blend": "legacy_mode0",
        "eye_clip": "eye_alpha_times_one_minus_mask_alpha",
        "final_order": [
            "before_trigger",
            "clipped_eye_buffer",
            "final_eye_overlay",
            "after_trigger",
        ],
    }:
        raise BinaryBoundsError("GPU compositor graph is invalid")
    return root


def write_rinne_gpu_bundle(
    output_directory: Path | str, bundle: RinneGpuBundle
) -> None:
    verify_rinne_gpu_bundle(bundle)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=False)
    write_new_bytes(destination / RINNE_GPU_MANIFEST_FILENAME, bundle.manifest_json)
    write_new_bytes(destination / RINNE_GPU_TEXTURE_FILENAME, bundle.texture_rgba8)
    write_new_bytes(destination / RINNE_GPU_GEOMETRY_FILENAME, bundle.geometry_binary)


__all__ = [
    "RINNE_GPU_BUNDLE_FORMAT",
    "RINNE_GPU_BUNDLE_VERSION",
    "RINNE_GPU_GEOMETRY_FILENAME",
    "RINNE_GPU_MANIFEST_FILENAME",
    "RINNE_GPU_PASS_NAMES",
    "RINNE_GPU_TEXTURE_FILENAME",
    "RinneGpuBundle",
    "RinneGpuSourceEvidence",
    "build_rinne_gpu_bundle",
    "verify_rinne_gpu_bundle",
    "write_rinne_gpu_bundle",
]
