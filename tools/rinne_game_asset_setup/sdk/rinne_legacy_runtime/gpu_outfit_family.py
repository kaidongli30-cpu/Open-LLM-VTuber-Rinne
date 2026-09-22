from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .checked_binary import BinaryBoundsError
from .gpu_first_outfit import (
    RinneGpuFirstOutfitArtifact,
    _BUNDLE_FILENAMES,
    describe_rinne_gpu_first_outfit_artifact,
)
from .io_safety import write_new_bytes
from .portrait_catalog import get_rinne_outfit_portrait_profiles
from .gpu_timeline import write_rinne_gpu_timeline_bundle

RINNE_GPU_OUTFIT_FAMILY_FORMAT = "rinne-legacy-gpu-outfit-family"
RINNE_GPU_OUTFIT_FAMILY_VERSION = 1
RINNE_GPU_OUTFIT_FAMILY_MANIFEST_FILENAME = "outfit-manifest.json"
RINNE_GPU_OUTFIT_FAMILY_MANIFEST_MAX_BYTES = 4 * 1024 * 1024
RINNE_GPU_OUTFIT_BUNDLE_FILENAMES = _BUNDLE_FILENAMES
_OUTFIT_METADATA = {
    2: {
        "outfit_id": "rinne_mp0602_second_outfit",
        "display_name": "第二套衣服",
        "default_portrait_id": 60_202,
    },
    3: {
        "outfit_id": "rinne_mp0603_red_cardigan_casual",
        "display_name": "红色开衫与棕色半身裙便装",
        "default_portrait_id": 60_302,
    },
    4: {
        "outfit_id": "rinne_mp0604_dark_navy_winter_uniform",
        "display_name": "深蓝色冬季校服",
        "default_portrait_id": 60_402,
    },
}

RinneGpuOutfitArtifact = RinneGpuFirstOutfitArtifact


def _outfit_metadata(outfit_number: int) -> dict[str, object]:
    if not isinstance(outfit_number, int) or isinstance(outfit_number, bool):
        raise TypeError("Rinne outfit number must be an integer")
    try:
        return _OUTFIT_METADATA[outfit_number]
    except KeyError as exc:
        raise KeyError(
            f"unknown exported Rinne outfit number: {outfit_number}"
        ) from exc


def describe_rinne_gpu_outfit_artifact(
    artifact: RinneGpuOutfitArtifact,
) -> dict[str, object]:
    return describe_rinne_gpu_first_outfit_artifact(artifact)


def build_rinne_gpu_outfit_family_manifest(
    outfit_number: int,
    artifacts: Sequence[RinneGpuOutfitArtifact],
) -> bytes:
    profiles = get_rinne_outfit_portrait_profiles(outfit_number)
    items = tuple(artifacts)
    if len(items) != len(profiles):
        raise BinaryBoundsError("outfit-family export requires all fifteen portraits")
    by_id = {item.profile.portrait_id: item for item in items}
    expected_ids = tuple(profile.portrait_id for profile in profiles)
    if tuple(sorted(by_id)) != expected_ids or len(by_id) != len(items):
        raise BinaryBoundsError("outfit-family portraits are missing or duplicated")
    entries = [
        describe_rinne_gpu_outfit_artifact(by_id[portrait_id])
        for portrait_id in expected_ids
    ]
    return build_rinne_gpu_outfit_family_manifest_from_entries(
        outfit_number,
        entries,
    )


def build_rinne_gpu_outfit_family_manifest_from_entries(
    outfit_number: int,
    entries: Sequence[dict[str, object]],
) -> bytes:
    metadata = _outfit_metadata(outfit_number)
    profiles = get_rinne_outfit_portrait_profiles(outfit_number)
    values = list(entries)
    expected_ids = [profile.portrait_id for profile in profiles]
    if (
        len(values) != len(expected_ids)
        or [entry.get("portrait_id") for entry in values] != expected_ids
    ):
        raise BinaryBoundsError("outfit-family manifest entries are incomplete")
    topology_hashes = sorted(
        {entry["bundle_identity"]["base_topology_sha256"] for entry in values}
    )
    manifest = {
        "format": RINNE_GPU_OUTFIT_FAMILY_FORMAT,
        "version": RINNE_GPU_OUTFIT_FAMILY_VERSION,
        "outfit_number": outfit_number,
        "outfit_id": metadata["outfit_id"],
        "display_name": metadata["display_name"],
        "portrait_id_range": [expected_ids[0], expected_ids[-1]],
        "portrait_count": len(values),
        "default_portrait_id": metadata["default_portrait_id"],
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
        "portraits": values,
        "source_policy": {
            "original_pcks_remain_read_only": True,
            "bundle_is_local_only": True,
            "game_assets_are_not_stored_in_source_repository": True,
        },
    }
    payload = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    verify_rinne_gpu_outfit_family_manifest(payload)
    return payload


def _require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BinaryBoundsError(f"outfit-family {label} must be an object")
    return value


def verify_rinne_gpu_outfit_family_manifest(payload: bytes) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise TypeError("outfit-family manifest must be bytes")
    if not 0 < len(payload) <= RINNE_GPU_OUTFIT_FAMILY_MANIFEST_MAX_BYTES:
        raise BinaryBoundsError("outfit-family manifest exceeds its size limit")
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BinaryBoundsError("outfit-family manifest is invalid JSON") from exc
    root = _require_object(manifest, "manifest")
    outfit_number = root.get("outfit_number")
    metadata = _outfit_metadata(outfit_number)
    profiles = get_rinne_outfit_portrait_profiles(outfit_number)
    expected_ids = [profile.portrait_id for profile in profiles]
    if (
        root.get("format") != RINNE_GPU_OUTFIT_FAMILY_FORMAT
        or root.get("version") != RINNE_GPU_OUTFIT_FAMILY_VERSION
        or root.get("outfit_id") != metadata["outfit_id"]
        or root.get("display_name") != metadata["display_name"]
        or root.get("portrait_id_range") != [expected_ids[0], expected_ids[-1]]
        or root.get("portrait_count") != 15
        or root.get("default_portrait_id") != metadata["default_portrait_id"]
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
        raise BinaryBoundsError("outfit-family manifest contract is invalid")
    portraits = root.get("portraits")
    if not isinstance(portraits, list) or len(portraits) != 15:
        raise BinaryBoundsError("outfit-family portrait list is invalid")
    topology_hashes: set[str] = set()
    for profile, value in zip(profiles, portraits, strict=True):
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
            or len(files) != len(RINNE_GPU_OUTFIT_BUNDLE_FILENAMES)
        ):
            raise BinaryBoundsError("outfit-family portrait entry is invalid")
        topology_hashes.add(identity["base_topology_sha256"])
        for expected_name, file_value in zip(
            RINNE_GPU_OUTFIT_BUNDLE_FILENAMES,
            files,
            strict=True,
        ):
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
                raise BinaryBoundsError("outfit-family file entry is invalid")
    topology = _require_object(root.get("topology"), "topology")
    if topology != {
        "distinct_hashes": sorted(topology_hashes),
        "shared_across_all_portraits": len(topology_hashes) == 1,
    }:
        raise BinaryBoundsError("outfit-family topology summary is invalid")
    return root


def verify_rinne_gpu_outfit_family_directory(
    directory: Path | str,
) -> dict[str, Any]:
    root = Path(directory)
    manifest_path = root / RINNE_GPU_OUTFIT_FAMILY_MANIFEST_FILENAME
    manifest = verify_rinne_gpu_outfit_family_manifest(manifest_path.read_bytes())
    for portrait in manifest["portraits"]:
        portrait_directory = root / portrait["directory"]
        if portrait_directory.parent != root or not portrait_directory.is_dir():
            raise BinaryBoundsError("outfit-family portrait directory is invalid")
        for expected in portrait["files"]:
            path = portrait_directory / expected["name"]
            if path.parent != portrait_directory or not path.is_file():
                raise BinaryBoundsError("outfit-family bundle file is missing")
            payload = path.read_bytes()
            if (
                len(payload) != expected["byte_length"]
                or hashlib.sha256(payload).hexdigest() != expected["sha256"]
            ):
                raise BinaryBoundsError("outfit-family bundle file hash changed")
    return manifest


def write_rinne_gpu_outfit_family_bundle(
    output_directory: Path | str,
    outfit_number: int,
    artifacts: Sequence[RinneGpuOutfitArtifact],
) -> None:
    manifest_bytes = build_rinne_gpu_outfit_family_manifest(
        outfit_number,
        artifacts,
    )
    profiles = get_rinne_outfit_portrait_profiles(outfit_number)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=False)
    by_id = {item.profile.portrait_id: item for item in artifacts}
    for profile in profiles:
        artifact = by_id[profile.portrait_id]
        write_rinne_gpu_timeline_bundle(
            destination / artifact.directory_name,
            artifact.timeline_bundle,
        )
    write_new_bytes(
        destination / RINNE_GPU_OUTFIT_FAMILY_MANIFEST_FILENAME,
        manifest_bytes,
    )
    verify_rinne_gpu_outfit_family_directory(destination)


__all__ = [
    "RINNE_GPU_OUTFIT_BUNDLE_FILENAMES",
    "RINNE_GPU_OUTFIT_FAMILY_FORMAT",
    "RINNE_GPU_OUTFIT_FAMILY_MANIFEST_FILENAME",
    "RINNE_GPU_OUTFIT_FAMILY_VERSION",
    "RinneGpuOutfitArtifact",
    "build_rinne_gpu_outfit_family_manifest",
    "build_rinne_gpu_outfit_family_manifest_from_entries",
    "describe_rinne_gpu_outfit_artifact",
    "verify_rinne_gpu_outfit_family_directory",
    "verify_rinne_gpu_outfit_family_manifest",
    "write_rinne_gpu_outfit_family_bundle",
]
