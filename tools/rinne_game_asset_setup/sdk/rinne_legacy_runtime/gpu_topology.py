from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

from .checked_binary import BinaryBoundsError
from .frame_renderer import RinneLegacyDrawPlan


RINNE_GPU_DYNAMIC_PASS_FIELDS = (
    "eye_buffer_meshes",
    "mask_buffer_meshes",
    "before_trigger_meshes",
    "final_eye_overlay_meshes",
    "after_trigger_meshes",
)


@dataclass(frozen=True)
class RinneGpuTopologyVerification:
    topology_sha256: str
    candidate_count: int
    pass_mesh_counts: tuple[tuple[str, int], ...]
    total_vertices: int
    total_triangles: int


def _topology_sha256(plan: RinneLegacyDrawPlan) -> str:
    digest = hashlib.sha256()
    digest.update(b"RINNE-GPU-TOPOLOGY-V1\0")
    digest.update(struct.pack("<III", plan.texture.width, plan.texture.height, 4))
    digest.update(hashlib.sha256(plan.texture.rgba).digest())
    digest.update(struct.pack("<I", plan.compositor_trigger_type))
    digest.update(struct.pack("<I", len(plan.expression_weights)))
    for pass_name in RINNE_GPU_DYNAMIC_PASS_FIELDS:
        digest.update(pass_name.encode("ascii") + b"\0")
        meshes = getattr(plan, pass_name)
        digest.update(struct.pack("<I", len(meshes)))
        for mesh in meshes:
            texture_filter = mesh.texture_filter or "nearest"
            digest.update(
                struct.pack(
                    "<IIBBB",
                    len(mesh.positions_xyz),
                    len(mesh.triangle_indices),
                    int(mesh.depth_write),
                    int(mesh.vertex_opacities is not None),
                    0 if texture_filter == "nearest" else 1,
                )
            )
            for u, v in mesh.uvs:
                if not math.isfinite(u) or not math.isfinite(v):
                    raise BinaryBoundsError("GPU topology contains a non-finite UV")
                digest.update(struct.pack("<dd", u, v))
            for index in mesh.triangle_indices:
                if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                    raise BinaryBoundsError("GPU topology contains an invalid index")
                digest.update(struct.pack("<I", index))
            if mesh.vertex_opacities is not None:
                digest.update(struct.pack("<I", len(mesh.vertex_opacities)))
    return digest.hexdigest()


def verify_rinne_gpu_topology_compatible(
    reference: RinneLegacyDrawPlan,
    candidate: RinneLegacyDrawPlan,
) -> None:
    if not isinstance(reference, RinneLegacyDrawPlan) or not isinstance(
        candidate, RinneLegacyDrawPlan
    ):
        raise TypeError("GPU topology verification requires draw plans")
    if (
        reference.texture.width != candidate.texture.width
        or reference.texture.height != candidate.texture.height
        or reference.texture.rgba != candidate.texture.rgba
    ):
        raise BinaryBoundsError("GPU topology candidate uses a different atlas")
    if reference.compositor_trigger_type != candidate.compositor_trigger_type:
        raise BinaryBoundsError("GPU topology compositor trigger changed")
    if len(reference.expression_weights) != len(candidate.expression_weights):
        raise BinaryBoundsError("GPU topology expression count changed")
    for pass_name in RINNE_GPU_DYNAMIC_PASS_FIELDS:
        reference_meshes = getattr(reference, pass_name)
        candidate_meshes = getattr(candidate, pass_name)
        if len(reference_meshes) != len(candidate_meshes):
            raise BinaryBoundsError(
                f"GPU topology {pass_name} mesh count changed: "
                f"{len(reference_meshes)} != {len(candidate_meshes)}"
            )
        for mesh_index, (reference_mesh, candidate_mesh) in enumerate(
            zip(reference_meshes, candidate_meshes, strict=True)
        ):
            label = f"GPU topology {pass_name}[{mesh_index}]"
            if len(reference_mesh.positions_xyz) != len(
                candidate_mesh.positions_xyz
            ):
                raise BinaryBoundsError(f"{label} vertex count changed")
            if tuple(reference_mesh.uvs) != tuple(candidate_mesh.uvs):
                raise BinaryBoundsError(f"{label} UVs changed")
            if tuple(reference_mesh.triangle_indices) != tuple(
                candidate_mesh.triangle_indices
            ):
                raise BinaryBoundsError(f"{label} triangle indices changed")
            if reference_mesh.depth_write != candidate_mesh.depth_write:
                raise BinaryBoundsError(f"{label} depth-write flag changed")
            if (reference_mesh.texture_filter or "nearest") != (
                candidate_mesh.texture_filter or "nearest"
            ):
                raise BinaryBoundsError(f"{label} texture filter changed")
            reference_opacity_count = (
                None
                if reference_mesh.vertex_opacities is None
                else len(reference_mesh.vertex_opacities)
            )
            candidate_opacity_count = (
                None
                if candidate_mesh.vertex_opacities is None
                else len(candidate_mesh.vertex_opacities)
            )
            if reference_opacity_count != candidate_opacity_count:
                raise BinaryBoundsError(
                    f"{label} vertex-opacity structure changed"
                )


def verify_rinne_gpu_topology_matrix(
    reference: RinneLegacyDrawPlan,
    candidates: tuple[RinneLegacyDrawPlan, ...],
) -> RinneGpuTopologyVerification:
    if not candidates:
        raise BinaryBoundsError("GPU topology matrix requires candidate plans")
    reference_hash = _topology_sha256(reference)
    for candidate in candidates:
        verify_rinne_gpu_topology_compatible(reference, candidate)
        if _topology_sha256(candidate) != reference_hash:
            raise BinaryBoundsError(
                "GPU topology hash changed despite structural verification"
            )
    pass_mesh_counts = tuple(
        (name, len(getattr(reference, name)))
        for name in RINNE_GPU_DYNAMIC_PASS_FIELDS
    )
    meshes = tuple(
        mesh
        for pass_name in RINNE_GPU_DYNAMIC_PASS_FIELDS
        for mesh in getattr(reference, pass_name)
    )
    return RinneGpuTopologyVerification(
        topology_sha256=reference_hash,
        candidate_count=len(candidates),
        pass_mesh_counts=pass_mesh_counts,
        total_vertices=sum(len(mesh.positions_xyz) for mesh in meshes),
        total_triangles=sum(len(mesh.triangle_indices) // 3 for mesh in meshes),
    )


__all__ = [
    "RINNE_GPU_DYNAMIC_PASS_FIELDS",
    "RinneGpuTopologyVerification",
    "verify_rinne_gpu_topology_compatible",
    "verify_rinne_gpu_topology_matrix",
]
