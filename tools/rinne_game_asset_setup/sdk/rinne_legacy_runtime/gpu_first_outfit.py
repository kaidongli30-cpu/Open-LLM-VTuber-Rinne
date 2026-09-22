from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .asset_manifest import RinnePckVerification
from .checked_binary import BinaryBoundsError
from .gpu_bundle import (
    RINNE_GPU_GEOMETRY_FILENAME,
    RINNE_GPU_MANIFEST_FILENAME,
    RINNE_GPU_TEXTURE_FILENAME,
)
from .gpu_eye_dynamics import (
    RINNE_GPU_EYE_DEFORMATION_FILENAME,
    RINNE_GPU_EYE_MANIFEST_FILENAME,
)
from .gpu_linear_dynamics import (
    RINNE_GPU_DEFORMATION_FILENAME,
    RINNE_GPU_DYNAMIC_MANIFEST_FILENAME,
)
from .gpu_runtime_controls import RINNE_GPU_RUNTIME_CONTROL_MANIFEST_FILENAME
from .gpu_timeline import (
    RINNE_GPU_TIMELINE_DATA_FILENAME,
    RINNE_GPU_TIMELINE_MANIFEST_FILENAME,
    RinneGpuTimelineBundle,
    verify_rinne_gpu_timeline_bundle,
    write_rinne_gpu_timeline_bundle,
)
from .io_safety import write_new_bytes
from .portrait_catalog import RINNE_PORTRAIT_PROFILES, RinnePortraitProfile

RINNE_GPU_FIRST_OUTFIT_FORMAT = "rinne-legacy-gpu-first-outfit"
RINNE_GPU_FIRST_OUTFIT_VERSION = 1
RINNE_GPU_FIRST_OUTFIT_MANIFEST_FILENAME = "first-outfit-manifest.json"
RINNE_GPU_FIRST_OUTFIT_MANIFEST_MAX_BYTES = 4 * 1024 * 1024
RINNE_GPU_FIRST_OUTFIT_DEFAULT_PORTRAIT_ID = 60_102
_BUNDLE_FILENAMES = (
    RINNE_GPU_MANIFEST_FILENAME,
    RINNE_GPU_TEXTURE_FILENAME,
    RINNE_GPU_GEOMETRY_FILENAME,
    RINNE_GPU_DYNAMIC_MANIFEST_FILENAME,
    RINNE_GPU_DEFORMATION_FILENAME,
    RINNE_GPU_EYE_MANIFEST_FILENAME,
    RINNE_GPU_EYE_DEFORMATION_FILENAME,
    RINNE_GPU_RUNTIME_CONTROL_MANIFEST_FILENAME,
    RINNE_GPU_TIMELINE_MANIFEST_FILENAME,
    RINNE_GPU_TIMELINE_DATA_FILENAME,
)


@dataclass(frozen=True)
class RinneGpuFirstOutfitArtifact:
    profile: RinnePortraitProfile
    verification: RinnePckVerification
    timeline_bundle: RinneGpuTimelineBundle

    @property
    def directory_name(self) -> str:
        return f"MP{self.profile.portrait_id:06d}"


def _bundle_payloads(bundle: RinneGpuTimelineBundle) -> dict[str, bytes]:
    runtime = bundle.runtime_bundle
    eye = runtime.eye_bundle
    linear = eye.linear_bundle
    base = linear.base_bundle
    return {
        RINNE_GPU_MANIFEST_FILENAME: base.manifest_json,
        RINNE_GPU_TEXTURE_FILENAME: base.texture_rgba8,
        RINNE_GPU_GEOMETRY_FILENAME: base.geometry_binary,
        RINNE_GPU_DYNAMIC_MANIFEST_FILENAME: linear.dynamic_manifest_json,
        RINNE_GPU_DEFORMATION_FILENAME: linear.deformation_rgba32f,
        RINNE_GPU_EYE_MANIFEST_FILENAME: eye.eye_manifest_json,
        RINNE_GPU_EYE_DEFORMATION_FILENAME: eye.eye_deformation_rgba32f,
        RINNE_GPU_RUNTIME_CONTROL_MANIFEST_FILENAME: runtime.runtime_manifest_json,
        RINNE_GPU_TIMELINE_MANIFEST_FILENAME: bundle.timeline_manifest_json,
        RINNE_GPU_TIMELINE_DATA_FILENAME: bundle.amb_channels_f32,
    }


def describe_rinne_gpu_first_outfit_artifact(
    artifact: RinneGpuFirstOutfitArtifact,
) -> dict[str, object]:
    if not isinstance(artifact, RinneGpuFirstOutfitArtifact):
        raise TypeError("first-outfit artifact has the wrong type")
    profile = artifact.profile
    verification = artifact.verification
    if verification.spec.portrait_id != profile.portrait_id:
        raise BinaryBoundsError("first-outfit profile and verification do not match")
    timeline = verify_rinne_gpu_timeline_bundle(artifact.timeline_bundle)
    base = artifact.timeline_bundle.runtime_bundle.eye_bundle.linear_bundle.base_bundle
    base_manifest = base.manifest()
    if (
        base_manifest["source"]["portrait_id"] != profile.portrait_id
        or base_manifest["source"]["pck_sha256"] != verification.sha256
    ):
        raise BinaryBoundsError("first-outfit source identity does not match bundle")
    payloads = _bundle_payloads(artifact.timeline_bundle)
    if tuple(payloads) != _BUNDLE_FILENAMES:
        raise AssertionError("first-outfit bundle file order changed")
    return {
        "portrait_id": profile.portrait_id,
        "directory": artifact.directory_name,
        "compatibility_label": profile.compatibility_label,
        "tags": list(profile.tags),
        "game_usage_count": profile.game_usage_count,
        "has_closed_eye_layer": profile.has_closed_eye_layer,
        "has_sweat_layer": profile.has_sweat_layer,
        "source": {
            "filename": verification.spec.filename,
            "byte_length": verification.spec.size,
            "sha256": verification.sha256,
        },
        "bundle_identity": {
            "base_topology_sha256": timeline["runtime_bundle"]["base_topology_sha256"],
            "timeline_manifest_sha256": hashlib.sha256(
                artifact.timeline_bundle.timeline_manifest_json
            ).hexdigest(),
        },
        "files": [
            {
                "name": name,
                "byte_length": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for name, payload in payloads.items()
        ],
    }


def build_rinne_gpu_first_outfit_manifest(
    artifacts: Sequence[RinneGpuFirstOutfitArtifact],
) -> bytes:
    items = tuple(artifacts)
    if len(items) != len(RINNE_PORTRAIT_PROFILES):
        raise BinaryBoundsError("first-outfit export requires all fifteen portraits")
    by_id = {item.profile.portrait_id: item for item in items}
    expected_ids = tuple(profile.portrait_id for profile in RINNE_PORTRAIT_PROFILES)
    if tuple(sorted(by_id)) != expected_ids or len(by_id) != len(items):
        raise BinaryBoundsError("first-outfit portraits are missing or duplicated")
    entries = [
        describe_rinne_gpu_first_outfit_artifact(by_id[portrait_id])
        for portrait_id in expected_ids
    ]
    return build_rinne_gpu_first_outfit_manifest_from_entries(entries)


def build_rinne_gpu_first_outfit_manifest_from_entries(
    entries: Sequence[dict[str, object]],
) -> bytes:
    entries = list(entries)
    expected_ids = [profile.portrait_id for profile in RINNE_PORTRAIT_PROFILES]
    if (
        len(entries) != len(expected_ids)
        or [entry.get("portrait_id") for entry in entries] != expected_ids
    ):
        raise BinaryBoundsError("first-outfit manifest entries are incomplete")
    topology_hashes = sorted(
        {entry["bundle_identity"]["base_topology_sha256"] for entry in entries}
    )
    manifest = {
        "format": RINNE_GPU_FIRST_OUTFIT_FORMAT,
        "version": RINNE_GPU_FIRST_OUTFIT_VERSION,
        "outfit_id": "rinne_mp0601_first_outfit",
        "portrait_id_range": [60_101, 60_115],
        "portrait_count": len(entries),
        "default_portrait_id": RINNE_GPU_FIRST_OUTFIT_DEFAULT_PORTRAIT_ID,
        "selection_contract": {
            "canonical_selector": "portrait_id",
            "tags_are_compatibility_labels": True,
            "switching_policy": "load_selected_bundle_then_reset_its_timeline",
            "resident_cache_recommendation": 2,
        },
        "topology": {
            "distinct_hashes": topology_hashes,
            "shared_across_all_portraits": len(topology_hashes) == 1,
        },
        "portraits": entries,
        "source_policy": {
            "original_pcks_remain_read_only": True,
            "bundle_is_local_only": True,
            "game_assets_are_not_stored_in_source_repository": True,
        },
    }
    payload = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    verify_rinne_gpu_first_outfit_manifest(payload)
    return payload


def refresh_rinne_gpu_first_outfit_manifest_metadata(payload: bytes) -> bytes:
    """Rebuild only user-approved portrait labels around unchanged bundle hashes."""

    if not isinstance(payload, bytes):
        raise TypeError("first-outfit manifest must be bytes")
    if not 0 < len(payload) <= RINNE_GPU_FIRST_OUTFIT_MANIFEST_MAX_BYTES:
        raise BinaryBoundsError("first-outfit manifest exceeds its size limit")
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BinaryBoundsError("first-outfit manifest is invalid JSON") from exc
    root = _require_object(manifest, "manifest")
    portraits = root.get("portraits")
    if not isinstance(portraits, list) or len(portraits) != len(
        RINNE_PORTRAIT_PROFILES
    ):
        raise BinaryBoundsError("first-outfit portrait list is invalid")
    refreshed: list[dict[str, object]] = []
    for profile, value in zip(RINNE_PORTRAIT_PROFILES, portraits, strict=True):
        entry = deepcopy(_require_object(value, "portrait"))
        if (
            entry.get("portrait_id") != profile.portrait_id
            or entry.get("directory") != f"MP{profile.portrait_id:06d}"
        ):
            raise BinaryBoundsError("first-outfit portrait order is invalid")
        entry["compatibility_label"] = profile.compatibility_label
        entry["tags"] = list(profile.tags)
        refreshed.append(entry)
    return build_rinne_gpu_first_outfit_manifest_from_entries(refreshed)


def _require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BinaryBoundsError(f"first-outfit {label} must be an object")
    return value


def verify_rinne_gpu_first_outfit_manifest(payload: bytes) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise TypeError("first-outfit manifest must be bytes")
    if not 0 < len(payload) <= RINNE_GPU_FIRST_OUTFIT_MANIFEST_MAX_BYTES:
        raise BinaryBoundsError("first-outfit manifest exceeds its size limit")
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BinaryBoundsError("first-outfit manifest is invalid JSON") from exc
    root = _require_object(manifest, "manifest")
    if (
        root.get("format") != RINNE_GPU_FIRST_OUTFIT_FORMAT
        or root.get("version") != RINNE_GPU_FIRST_OUTFIT_VERSION
        or root.get("outfit_id") != "rinne_mp0601_first_outfit"
        or root.get("portrait_id_range") != [60_101, 60_115]
        or root.get("portrait_count") != 15
        or root.get("default_portrait_id") != RINNE_GPU_FIRST_OUTFIT_DEFAULT_PORTRAIT_ID
        or root.get("selection_contract")
        != {
            "canonical_selector": "portrait_id",
            "tags_are_compatibility_labels": True,
            "switching_policy": "load_selected_bundle_then_reset_its_timeline",
            "resident_cache_recommendation": 2,
        }
        or root.get("source_policy")
        != {
            "original_pcks_remain_read_only": True,
            "bundle_is_local_only": True,
            "game_assets_are_not_stored_in_source_repository": True,
        }
    ):
        raise BinaryBoundsError("first-outfit manifest contract is invalid")
    portraits = root.get("portraits")
    if not isinstance(portraits, list) or len(portraits) != 15:
        raise BinaryBoundsError("first-outfit portrait list is invalid")
    topology_hashes: set[str] = set()
    for profile, value in zip(RINNE_PORTRAIT_PROFILES, portraits, strict=True):
        entry = _require_object(value, "portrait")
        source = _require_object(entry.get("source"), "source")
        identity = _require_object(entry.get("bundle_identity"), "identity")
        files = entry.get("files")
        if (
            entry.get("portrait_id") != profile.portrait_id
            or entry.get("directory") != f"MP{profile.portrait_id:06d}"
            or entry.get("compatibility_label") != profile.compatibility_label
            or entry.get("tags") != list(profile.tags)
            or entry.get("game_usage_count") != profile.game_usage_count
            or entry.get("has_closed_eye_layer") != profile.has_closed_eye_layer
            or entry.get("has_sweat_layer") != profile.has_sweat_layer
            or source.get("filename") != profile.filename
            or not isinstance(source.get("byte_length"), int)
            or isinstance(source.get("byte_length"), bool)
            or source["byte_length"] <= 0
            or not isinstance(source.get("sha256"), str)
            or len(source["sha256"]) != 64
            or not isinstance(identity.get("base_topology_sha256"), str)
            or len(identity["base_topology_sha256"]) != 64
            or not isinstance(identity.get("timeline_manifest_sha256"), str)
            or len(identity["timeline_manifest_sha256"]) != 64
            or not isinstance(files, list)
            or len(files) != len(_BUNDLE_FILENAMES)
        ):
            raise BinaryBoundsError("first-outfit portrait entry is invalid")
        topology_hashes.add(identity["base_topology_sha256"])
        for expected_name, file_value in zip(_BUNDLE_FILENAMES, files, strict=True):
            file_entry = _require_object(file_value, "file")
            path = PurePosixPath(file_entry.get("name", ""))
            byte_length = file_entry.get("byte_length")
            digest = file_entry.get("sha256")
            if (
                str(path) != expected_name
                or path.name != expected_name
                or not isinstance(byte_length, int)
                or isinstance(byte_length, bool)
                or byte_length <= 0
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise BinaryBoundsError("first-outfit file entry is invalid")
    topology = _require_object(root.get("topology"), "topology")
    if topology != {
        "distinct_hashes": sorted(topology_hashes),
        "shared_across_all_portraits": len(topology_hashes) == 1,
    }:
        raise BinaryBoundsError("first-outfit topology summary is invalid")
    return root


def verify_rinne_gpu_first_outfit_directory(
    directory: Path | str,
) -> dict[str, Any]:
    root = Path(directory)
    manifest_path = root / RINNE_GPU_FIRST_OUTFIT_MANIFEST_FILENAME
    manifest = verify_rinne_gpu_first_outfit_manifest(manifest_path.read_bytes())
    for portrait in manifest["portraits"]:
        portrait_directory = root / portrait["directory"]
        if portrait_directory.parent != root or not portrait_directory.is_dir():
            raise BinaryBoundsError("first-outfit portrait directory is invalid")
        for expected in portrait["files"]:
            path = portrait_directory / expected["name"]
            if path.parent != portrait_directory or not path.is_file():
                raise BinaryBoundsError("first-outfit bundle file is missing")
            payload = path.read_bytes()
            if (
                len(payload) != expected["byte_length"]
                or hashlib.sha256(payload).hexdigest() != expected["sha256"]
            ):
                raise BinaryBoundsError("first-outfit bundle file hash changed")
    return manifest


def write_rinne_gpu_first_outfit_bundle(
    output_directory: Path | str,
    artifacts: Sequence[RinneGpuFirstOutfitArtifact],
) -> None:
    manifest_bytes = build_rinne_gpu_first_outfit_manifest(artifacts)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=False)
    by_id = {item.profile.portrait_id: item for item in artifacts}
    for profile in RINNE_PORTRAIT_PROFILES:
        artifact = by_id[profile.portrait_id]
        write_rinne_gpu_timeline_bundle(
            destination / artifact.directory_name,
            artifact.timeline_bundle,
        )
    write_new_bytes(
        destination / RINNE_GPU_FIRST_OUTFIT_MANIFEST_FILENAME,
        manifest_bytes,
    )
    verify_rinne_gpu_first_outfit_directory(destination)


__all__ = [
    "RINNE_GPU_FIRST_OUTFIT_DEFAULT_PORTRAIT_ID",
    "RINNE_GPU_FIRST_OUTFIT_FORMAT",
    "RINNE_GPU_FIRST_OUTFIT_MANIFEST_FILENAME",
    "RINNE_GPU_FIRST_OUTFIT_VERSION",
    "RinneGpuFirstOutfitArtifact",
    "build_rinne_gpu_first_outfit_manifest",
    "build_rinne_gpu_first_outfit_manifest_from_entries",
    "describe_rinne_gpu_first_outfit_artifact",
    "verify_rinne_gpu_first_outfit_directory",
    "verify_rinne_gpu_first_outfit_manifest",
    "write_rinne_gpu_first_outfit_bundle",
]
