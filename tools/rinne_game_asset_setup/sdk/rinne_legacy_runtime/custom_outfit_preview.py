from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from .checked_binary import BinaryBoundsError
from .io_safety import write_new_bytes
from .spirit_expression_semantics import SPIRIT_EXPRESSION_LABELS


RINNE_CUSTOM_OUTFIT_PREVIEW_FORMAT = "rinne-custom-outfit-layered-preview"
RINNE_CUSTOM_OUTFIT_PREVIEW_VERSION = 1
RINNE_CUSTOM_OUTFIT_PREVIEW_MANIFEST_FILENAME = "custom-outfit-preview.json"
RINNE_CUSTOM_OUTFIT_PREVIEW_MANIFEST_MAX_BYTES = 1024 * 1024
RINNE_CUSTOM_OUTFIT_CANVAS_SIZE = 2048

_SHA256_LENGTH = 64
_ASSET_CONTRACT = (
    ("body", "layers/body.png"),
    ("native_head_mask", "layers/native-head-mask.png"),
    ("neutral_proof", "evidence/neutral-composite-proof.png"),
)


def _require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BinaryBoundsError(f"custom-outfit {label} must be an object")
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _describe_png(root: Path, role: str, relative: str) -> dict[str, object]:
    from PIL import Image

    path = root.joinpath(*PurePosixPath(relative).parts)
    if not path.is_file() or root not in path.parents:
        raise BinaryBoundsError(f"custom-outfit {role} PNG is missing")
    payload = path.read_bytes()
    if not payload:
        raise BinaryBoundsError(f"custom-outfit {role} PNG is empty")
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG":
                raise BinaryBoundsError(f"custom-outfit {role} must be PNG")
            size = list(image.size)
            mode = image.mode
    except OSError as exc:
        raise BinaryBoundsError(f"custom-outfit {role} PNG is invalid") from exc
    if size != [RINNE_CUSTOM_OUTFIT_CANVAS_SIZE] * 2 or mode != "RGBA":
        raise BinaryBoundsError(
            f"custom-outfit {role} must be 2048 x 2048 RGBA"
        )
    return {
        "role": role,
        "file": relative,
        "byte_length": len(payload),
        "sha256": _sha256(payload),
        "canvas": size,
        "mode": mode,
    }


def build_rinne_custom_outfit_preview_manifest(
    asset_directory: Path | str,
    *,
    outfit_id: str,
    display_name: str,
    source_reference_sha256: str,
) -> bytes:
    root = Path(asset_directory).resolve()
    if (
        not outfit_id
        or outfit_id != outfit_id.casefold()
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in outfit_id)
    ):
        raise BinaryBoundsError("custom-outfit id is invalid")
    if not display_name or display_name.strip() != display_name:
        raise BinaryBoundsError("custom-outfit display name is invalid")
    if (
        not isinstance(source_reference_sha256, str)
        or len(source_reference_sha256) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in source_reference_sha256)
    ):
        raise BinaryBoundsError("custom-outfit source reference hash is invalid")
    assets = [
        _describe_png(root, role, relative)
        for role, relative in _ASSET_CONTRACT
    ]
    manifest = {
        "format": RINNE_CUSTOM_OUTFIT_PREVIEW_FORMAT,
        "version": RINNE_CUSTOM_OUTFIT_PREVIEW_VERSION,
        "stage": "offline-user-acceptance-before-desktop-integration",
        "outfit_id": outfit_id,
        "display_name": display_name,
        "canvas": [RINNE_CUSTOM_OUTFIT_CANVAS_SIZE] * 2,
        "asset_count": len(assets),
        "assets": assets,
        "donor_runtime": {
            "outfit_number": 1,
            "portrait_id_range": [60_101, 60_115],
            "default_portrait_id": 60_102,
            "expression_labels": list(SPIRIT_EXPRESSION_LABELS),
            "shy_base_portrait_id": 60_113,
            "shy_mouth_portrait_id": 60_104,
        },
        "composition": {
            "layer_order": ["body", "native_head", "shy_mouth"],
            "native_head_mask_role": "native_head_mask",
            "body_breath": {
                "horizontal_motion_pixels": 0.0,
                "vertical_motion_pixels": 1.5,
                "scale_y_peak": 1.0015,
                "transform_origin": [0.5, 0.82],
            },
            "mouth_gain_percent": {
                "neutral": 68,
                "happy": 68,
                "default": 100,
            },
            "desktop_stable_required": True,
        },
        "provenance": {
            "source_reference_sha256": source_reference_sha256,
            "native_face_and_hair_are_runtime_donor": True,
            "generated_face_and_hair_are_not_delivered": True,
        },
        "source_policy": {
            "original_game_files_remain_read_only": True,
            "desktop_pet_remains_untouched": True,
            "uses_prerendered_frame_sequence": False,
            "uses_video_playback": False,
            "requires_user_acceptance_before_integration": True,
        },
    }
    payload = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    verify_rinne_custom_outfit_preview_manifest(payload)
    return payload


def verify_rinne_custom_outfit_preview_manifest(payload: bytes) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise TypeError("custom-outfit manifest must be bytes")
    if not 0 < len(payload) <= RINNE_CUSTOM_OUTFIT_PREVIEW_MANIFEST_MAX_BYTES:
        raise BinaryBoundsError("custom-outfit manifest exceeds its size limit")
    try:
        root = _require_object(json.loads(payload), "manifest")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BinaryBoundsError("custom-outfit manifest is invalid JSON") from exc
    if (
        root.get("format") != RINNE_CUSTOM_OUTFIT_PREVIEW_FORMAT
        or root.get("version") != RINNE_CUSTOM_OUTFIT_PREVIEW_VERSION
        or root.get("stage")
        != "offline-user-acceptance-before-desktop-integration"
        or root.get("canvas") != [RINNE_CUSTOM_OUTFIT_CANVAS_SIZE] * 2
        or root.get("asset_count") != len(_ASSET_CONTRACT)
        or root.get("source_policy")
        != {
            "original_game_files_remain_read_only": True,
            "desktop_pet_remains_untouched": True,
            "uses_prerendered_frame_sequence": False,
            "uses_video_playback": False,
            "requires_user_acceptance_before_integration": True,
        }
    ):
        raise BinaryBoundsError("custom-outfit manifest contract is invalid")
    if (
        not isinstance(root.get("outfit_id"), str)
        or not root["outfit_id"]
        or root["outfit_id"] != root["outfit_id"].casefold()
        or not isinstance(root.get("display_name"), str)
        or not root["display_name"]
    ):
        raise BinaryBoundsError("custom-outfit identity is invalid")
    assets = root.get("assets")
    if not isinstance(assets, list) or len(assets) != len(_ASSET_CONTRACT):
        raise BinaryBoundsError("custom-outfit asset list is invalid")
    for (role, relative), value in zip(_ASSET_CONTRACT, assets, strict=True):
        entry = _require_object(value, "asset")
        path = PurePosixPath(entry.get("file", ""))
        if (
            entry.get("role") != role
            or str(path) != relative
            or path.is_absolute()
            or ".." in path.parts
            or not isinstance(entry.get("byte_length"), int)
            or isinstance(entry.get("byte_length"), bool)
            or entry["byte_length"] <= 0
            or not isinstance(entry.get("sha256"), str)
            or len(entry["sha256"]) != _SHA256_LENGTH
            or entry.get("canvas") != [RINNE_CUSTOM_OUTFIT_CANVAS_SIZE] * 2
            or entry.get("mode") != "RGBA"
        ):
            raise BinaryBoundsError("custom-outfit asset entry is invalid")
    if root.get("donor_runtime") != {
        "outfit_number": 1,
        "portrait_id_range": [60_101, 60_115],
        "default_portrait_id": 60_102,
        "expression_labels": list(SPIRIT_EXPRESSION_LABELS),
        "shy_base_portrait_id": 60_113,
        "shy_mouth_portrait_id": 60_104,
    }:
        raise BinaryBoundsError("custom-outfit donor runtime is invalid")
    composition = root.get("composition")
    if composition != {
        "layer_order": ["body", "native_head", "shy_mouth"],
        "native_head_mask_role": "native_head_mask",
        "body_breath": {
            "horizontal_motion_pixels": 0.0,
            "vertical_motion_pixels": 1.5,
            "scale_y_peak": 1.0015,
            "transform_origin": [0.5, 0.82],
        },
        "mouth_gain_percent": {
            "neutral": 68,
            "happy": 68,
            "default": 100,
        },
        "desktop_stable_required": True,
    }:
        raise BinaryBoundsError("custom-outfit composition contract is invalid")
    provenance = _require_object(root.get("provenance"), "provenance")
    if (
        not isinstance(provenance.get("source_reference_sha256"), str)
        or len(provenance["source_reference_sha256"]) != _SHA256_LENGTH
        or provenance.get("native_face_and_hair_are_runtime_donor") is not True
        or provenance.get("generated_face_and_hair_are_not_delivered") is not True
    ):
        raise BinaryBoundsError("custom-outfit provenance is invalid")
    return root


def verify_rinne_custom_outfit_preview_directory(
    directory: Path | str,
) -> dict[str, Any]:
    root = Path(directory).resolve()
    manifest = verify_rinne_custom_outfit_preview_manifest(
        (root / RINNE_CUSTOM_OUTFIT_PREVIEW_MANIFEST_FILENAME).read_bytes()
    )
    for asset in manifest["assets"]:
        path = root.joinpath(*PurePosixPath(asset["file"]).parts)
        if not path.is_file() or root not in path.parents:
            raise BinaryBoundsError("custom-outfit asset file is missing")
        payload = path.read_bytes()
        if len(payload) != asset["byte_length"] or _sha256(payload) != asset["sha256"]:
            raise BinaryBoundsError("custom-outfit asset hash changed")
        _describe_png(root, asset["role"], asset["file"])
    return manifest


def write_rinne_custom_outfit_preview_manifest(
    directory: Path | str,
    *,
    outfit_id: str,
    display_name: str,
    source_reference_sha256: str,
) -> dict[str, Any]:
    root = Path(directory).resolve()
    payload = build_rinne_custom_outfit_preview_manifest(
        root,
        outfit_id=outfit_id,
        display_name=display_name,
        source_reference_sha256=source_reference_sha256,
    )
    write_new_bytes(root / RINNE_CUSTOM_OUTFIT_PREVIEW_MANIFEST_FILENAME, payload)
    return verify_rinne_custom_outfit_preview_directory(root)


__all__ = [
    "RINNE_CUSTOM_OUTFIT_CANVAS_SIZE",
    "RINNE_CUSTOM_OUTFIT_PREVIEW_FORMAT",
    "RINNE_CUSTOM_OUTFIT_PREVIEW_MANIFEST_FILENAME",
    "RINNE_CUSTOM_OUTFIT_PREVIEW_VERSION",
    "build_rinne_custom_outfit_preview_manifest",
    "verify_rinne_custom_outfit_preview_directory",
    "verify_rinne_custom_outfit_preview_manifest",
    "write_rinne_custom_outfit_preview_manifest",
]
