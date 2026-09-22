from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

from .checked_binary import BinaryBoundsError
from .gpu_outfit_family import RINNE_GPU_OUTFIT_BUNDLE_FILENAMES
from .io_safety import write_new_bytes
from .spirit_expression_semantics import (
    SPIRIT_EXPRESSION_SEMANTICS,
    SpiritExpressionSemantic,
)

RINNE_SPIRIT_EXPRESSION_PREVIEW_FORMAT = "rinne-spirit-dress-expression-preview"
RINNE_SPIRIT_EXPRESSION_PREVIEW_VERSION = 3
RINNE_SPIRIT_EXPRESSION_PREVIEW_MANIFEST_FILENAME = "spirit-expression-preview.json"
RINNE_SPIRIT_EXPRESSION_PREVIEW_MANIFEST_MAX_BYTES = 4 * 1024 * 1024
RINNE_SPIRIT_NATIVE_FAMILY_MANIFEST_FILENAME = "spirit-dress-native-family.json"
RINNE_SPIRIT_PREVIEW_CANVAS_SIZE = 4096

_SHA256_LENGTH = 64
_NATIVE_IDS = tuple(range(160_101, 160_108))
_OVERLAY_FILES = (
    ("awkward-embarrassed-sweat", "awkward-embarrassed-sweat.png"),
    ("flustered-blush", "flustered-blush.png"),
    ("shy-blush", "shy-blush.png"),
)
_CLOSED_EYE_LABELS = {"gentle", "awkward", "sad", "embarrassed"}
_IDENTITY_TRANSFORM = {
    "source_center": [0.5, 0.5],
    "target_center": [0.5, 0.5],
    "scale": 1.0,
    "rotation_degrees": 0.0,
}
_MOUTH_MASK = {
    "role": "mouth",
    "shape": "soft_ellipse",
    "center": [2185 / 4096, 2267.5 / 4096],
    "radii": [115 / 4096, 67.5 / 4096],
    "feather": 0.27,
}


def _require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BinaryBoundsError(f"spirit-expression {label} must be an object")
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        return _require_object(json.loads(path.read_bytes()), label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BinaryBoundsError(f"spirit-expression {label} is unavailable") from exc


def _checked_native_family(root: Path) -> None:
    value = _read_json_object(
        root / RINNE_SPIRIT_NATIVE_FAMILY_MANIFEST_FILENAME,
        "native-family manifest",
    )
    portraits = value.get("portraits")
    if (
        value.get("format") != "rinne-spirit-dress-native-source-family"
        or value.get("version") != 1
        or value.get("portrait_count") != 7
        or not isinstance(portraits, list)
        or tuple(item.get("portrait_id") for item in portraits) != _NATIVE_IDS
    ):
        raise BinaryBoundsError("spirit-expression native-family identity is invalid")


def _describe_portrait(
    root: Path, portrait_id: int, source_family: str
) -> dict[str, object]:
    directory_name = f"MP{portrait_id:06d}"
    directory = root / directory_name
    if not directory.is_dir() or directory.parent != root:
        raise BinaryBoundsError("spirit-expression portrait is missing")
    files: list[dict[str, object]] = []
    for name in RINNE_GPU_OUTFIT_BUNDLE_FILENAMES:
        candidate = directory / name
        if not candidate.is_file() or candidate.parent != directory:
            raise BinaryBoundsError("spirit-expression runtime file is missing")
        payload = candidate.read_bytes()
        if not payload:
            raise BinaryBoundsError("spirit-expression runtime file is empty")
        files.append(
            {"name": name, "byte_length": len(payload), "sha256": _sha256(payload)}
        )
    return {
        "portrait_id": portrait_id,
        "directory": directory_name,
        "source_family": source_family,
        "files": files,
    }


def _expression_recipe(semantic: SpiritExpressionSemantic) -> dict[str, object]:
    mouth_layer = None
    if semantic.mouth_override_portrait_id is not None:
        mouth_layer = {
            "role": "mouth_override",
            "portrait_id": semantic.mouth_override_portrait_id,
            "transform": dict(_IDENTITY_TRANSFORM),
            "masks": [dict(_MOUTH_MASK)],
        }
    return {
        "label": semantic.label,
        "display_name": semantic.label,
        "category": semantic.category,
        "mode": "native-approved",
        "base_portrait_id": semantic.native_base_portrait_id,
        "mouth_gain_percent": semantic.mouth_gain_percent,
        "closed_eye_expression": semantic.label in _CLOSED_EYE_LABELS,
        "mouth_layer": mouth_layer,
        "overlay_id": semantic.overlay_id,
        "semantic_evidence": {
            "required_cues": list(semantic.required_cues),
            "discriminator": semantic.discriminator,
        },
    }


def _describe_overlay(root: Path, overlay_id: str, name: str) -> dict[str, object]:
    candidate = root / name
    if not candidate.is_file() or candidate.parent != root:
        raise BinaryBoundsError("spirit-expression approved overlay is missing")
    payload = candidate.read_bytes()
    if not payload:
        raise BinaryBoundsError("spirit-expression approved overlay is empty")
    return {
        "overlay_id": overlay_id,
        "file": f"overlays/{name}",
        "byte_length": len(payload),
        "sha256": _sha256(payload),
    }


def build_rinne_spirit_expression_preview_manifest(
    native_family_directory: Path | str,
    overlay_directory: Path | str,
) -> bytes:
    native_root = Path(native_family_directory).resolve()
    overlay_root = Path(overlay_directory).resolve()
    _checked_native_family(native_root)
    portraits = [
        *(_describe_portrait(native_root, item, "spirit-native") for item in _NATIVE_IDS),
    ]
    overlays = [
        _describe_overlay(overlay_root, overlay_id, name)
        for overlay_id, name in _OVERLAY_FILES
    ]
    expressions = [_expression_recipe(item) for item in SPIRIT_EXPRESSION_SEMANTICS]
    manifest = {
        "format": RINNE_SPIRIT_EXPRESSION_PREVIEW_FORMAT,
        "version": RINNE_SPIRIT_EXPRESSION_PREVIEW_VERSION,
        "stage": "approved-fifteen-native-runtime",
        "display_name": "灵装凛祢",
        "canvas": [4096, 4096],
        "portrait_count": len(portraits),
        "portraits": portraits,
        "overlay_count": len(overlays),
        "overlays": overlays,
        "expression_count": len(expressions),
        "complete_fifteen_label_set": True,
        "default_expression": "neutral",
        "expressions": expressions,
        "source_policy": {
            "original_pcks_remain_read_only": True,
            "desktop_pet_remains_untouched": True,
            "uses_prerendered_frame_sequence": False,
            "uses_video_playback": False,
            "preserves_spirit_red_irises": True,
            "uses_first_outfit_face_transplant": False,
            "approved_local_cues_only": True,
        },
    }
    payload = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    verify_rinne_spirit_expression_preview_manifest(payload)
    return payload


def _verify_file_entry(value: object, expected_name: str) -> dict[str, Any]:
    entry = _require_object(value, "file")
    path = PurePosixPath(entry.get("name", ""))
    if (
        str(path) != expected_name
        or path.name != expected_name
        or not isinstance(entry.get("byte_length"), int)
        or isinstance(entry.get("byte_length"), bool)
        or entry["byte_length"] <= 0
        or not isinstance(entry.get("sha256"), str)
        or len(entry["sha256"]) != _SHA256_LENGTH
    ):
        raise BinaryBoundsError("spirit-expression file entry is invalid")
    return entry


def verify_rinne_spirit_expression_preview_manifest(payload: bytes) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise TypeError("spirit-expression manifest must be bytes")
    if not 0 < len(payload) <= RINNE_SPIRIT_EXPRESSION_PREVIEW_MANIFEST_MAX_BYTES:
        raise BinaryBoundsError("spirit-expression manifest exceeds its size limit")
    try:
        root = _require_object(json.loads(payload), "manifest")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BinaryBoundsError("spirit-expression manifest is invalid JSON") from exc
    if (
        root.get("format") != RINNE_SPIRIT_EXPRESSION_PREVIEW_FORMAT
        or root.get("version") != RINNE_SPIRIT_EXPRESSION_PREVIEW_VERSION
        or root.get("stage") != "approved-fifteen-native-runtime"
        or root.get("display_name") != "灵装凛祢"
        or root.get("canvas") != [4096, 4096]
        or root.get("portrait_count") != 7
        or root.get("overlay_count") != 3
        or root.get("expression_count") != 15
        or root.get("complete_fifteen_label_set") is not True
        or root.get("default_expression") != "neutral"
        or root.get("source_policy") != {
            "original_pcks_remain_read_only": True,
            "desktop_pet_remains_untouched": True,
            "uses_prerendered_frame_sequence": False,
            "uses_video_playback": False,
            "preserves_spirit_red_irises": True,
            "uses_first_outfit_face_transplant": False,
            "approved_local_cues_only": True,
        }
    ):
        raise BinaryBoundsError("spirit-expression manifest contract is invalid")
    portraits = root.get("portraits")
    if not isinstance(portraits, list) or len(portraits) != 7:
        raise BinaryBoundsError("spirit-expression portrait list is invalid")
    for portrait_id, value in zip(_NATIVE_IDS, portraits, strict=True):
        entry = _require_object(value, "portrait")
        files = entry.get("files")
        if (
            entry.get("portrait_id") != portrait_id
            or entry.get("directory") != f"MP{portrait_id:06d}"
            or entry.get("source_family") != "spirit-native"
            or not isinstance(files, list)
            or len(files) != len(RINNE_GPU_OUTFIT_BUNDLE_FILENAMES)
        ):
            raise BinaryBoundsError("spirit-expression portrait entry is invalid")
        for name, file_entry in zip(
            RINNE_GPU_OUTFIT_BUNDLE_FILENAMES, files, strict=True
        ):
            _verify_file_entry(file_entry, name)
    overlays = root.get("overlays")
    if not isinstance(overlays, list) or len(overlays) != 3:
        raise BinaryBoundsError("spirit-expression overlay list is invalid")
    for (overlay_id, name), value in zip(_OVERLAY_FILES, overlays, strict=True):
        entry = _require_object(value, "overlay")
        path = PurePosixPath(entry.get("file", ""))
        if (
            entry.get("overlay_id") != overlay_id
            or str(path) != f"overlays/{name}"
            or not isinstance(entry.get("byte_length"), int)
            or isinstance(entry.get("byte_length"), bool)
            or entry["byte_length"] <= 0
            or not isinstance(entry.get("sha256"), str)
            or len(entry["sha256"]) != _SHA256_LENGTH
        ):
            raise BinaryBoundsError("spirit-expression overlay entry is invalid")
    expressions = root.get("expressions")
    if not isinstance(expressions, list) or len(expressions) != 15:
        raise BinaryBoundsError("spirit-expression expression list is invalid")
    for semantic, value in zip(
        SPIRIT_EXPRESSION_SEMANTICS, expressions, strict=True
    ):
        if value != _expression_recipe(semantic):
            raise BinaryBoundsError(
                f"spirit-expression {semantic.label} recipe is invalid"
            )
    return root


def verify_rinne_spirit_expression_preview_directory(
    directory: Path | str,
) -> dict[str, Any]:
    root = Path(directory).resolve()
    manifest = verify_rinne_spirit_expression_preview_manifest(
        (root / RINNE_SPIRIT_EXPRESSION_PREVIEW_MANIFEST_FILENAME).read_bytes()
    )
    for portrait in manifest["portraits"]:
        portrait_directory = root / portrait["directory"]
        if portrait_directory.parent != root or not portrait_directory.is_dir():
            raise BinaryBoundsError("spirit-expression portrait directory is invalid")
        for expected in portrait["files"]:
            path = portrait_directory / expected["name"]
            if path.parent != portrait_directory or not path.is_file():
                raise BinaryBoundsError("spirit-expression runtime file is missing")
            payload = path.read_bytes()
            if (
                len(payload) != expected["byte_length"]
                or _sha256(payload) != expected["sha256"]
            ):
                raise BinaryBoundsError("spirit-expression runtime file hash changed")
    for overlay in manifest["overlays"]:
        path = root.joinpath(*PurePosixPath(overlay["file"]).parts)
        if path.parent != root / "overlays" or not path.is_file():
            raise BinaryBoundsError("spirit-expression overlay file is missing")
        payload = path.read_bytes()
        if (
            len(payload) != overlay["byte_length"]
            or _sha256(payload) != overlay["sha256"]
        ):
            raise BinaryBoundsError("spirit-expression overlay file hash changed")
    return manifest


def write_rinne_spirit_expression_preview_directory(
    output_directory: Path | str,
    native_family_directory: Path | str,
    overlay_directory: Path | str,
) -> dict[str, Any]:
    destination = Path(output_directory).resolve()
    native_root = Path(native_family_directory).resolve()
    overlay_root = Path(overlay_directory).resolve()
    manifest_payload = build_rinne_spirit_expression_preview_manifest(
        native_root, overlay_root
    )
    manifest = verify_rinne_spirit_expression_preview_manifest(manifest_payload)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for portrait in manifest["portraits"]:
            portrait_destination = destination / portrait["directory"]
            portrait_destination.mkdir()
            portrait_source = native_root / portrait["directory"]
            for file_entry in portrait["files"]:
                os.link(
                    portrait_source / file_entry["name"],
                    portrait_destination / file_entry["name"],
                )
        overlay_destination = destination / "overlays"
        overlay_destination.mkdir()
        for overlay in manifest["overlays"]:
            name = PurePosixPath(overlay["file"]).name
            os.link(overlay_root / name, overlay_destination / name)
        write_new_bytes(
            destination / RINNE_SPIRIT_EXPRESSION_PREVIEW_MANIFEST_FILENAME,
            manifest_payload,
        )
        return verify_rinne_spirit_expression_preview_directory(destination)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


__all__ = [
    "RINNE_SPIRIT_EXPRESSION_PREVIEW_FORMAT",
    "RINNE_SPIRIT_EXPRESSION_PREVIEW_MANIFEST_FILENAME",
    "RINNE_SPIRIT_EXPRESSION_PREVIEW_VERSION",
    "build_rinne_spirit_expression_preview_manifest",
    "verify_rinne_spirit_expression_preview_directory",
    "verify_rinne_spirit_expression_preview_manifest",
    "write_rinne_spirit_expression_preview_directory",
]
