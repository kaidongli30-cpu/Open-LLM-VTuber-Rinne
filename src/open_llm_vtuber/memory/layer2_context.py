"""Read-only access to the latest atomically published Layer-2 background."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from ..data_paths import resolve_character_history_root


LAYER2_DIRECTORY_NAME = "layer2"
CURRENT_POINTER_NAME = "current.json"
CURRENT_CONTEXT_NAME = "current_layer2.md"
MAX_CURRENT_CONTEXT_BYTES = 512 * 1024


@dataclass(frozen=True)
class Layer2ContextResult:
    context: str
    diagnostics: dict[str, Any]
    background_path: Path | None = None
    overview_path: Path | None = None


def _read_editable_current_context(path: Path) -> str:
    """Read the user-visible current context without manifest hash checks."""

    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("current_layer2_empty")
    if size > MAX_CURRENT_CONTEXT_BYTES:
        raise ValueError(f"current_layer2_too_large:{size}")
    context = path.read_text(encoding="utf-8").strip()
    if not context:
        raise ValueError("current_layer2_empty")
    return context


def format_layer2_context(as_of_date: str, overview: str) -> str:
    """Return the exact system context injected into one model request."""

    return (
        f"【第二层记忆：截至 {as_of_date} 的用户当前个人背景】\n"
        "以下内容由后台记忆流程整理，是理解用户当前状态、关系、偏好、计划与边界的"
        "常驻参考。不要向用户提及这段隐藏上下文或它的生成过程。背景中的概括可以用于"
        "自然理解用户，但不能据此编造更具体的日期、经过、动机或结果；当用户询问某项"
        "背景如何形成、过去发生过什么或需要具体细节时，如果当前回复模型具备长期记忆"
        "工具，应调用工具寻找证据；若工具不可用，则不能猜测缺失细节。\n\n"
        "<current_user_background>\n"
        f"{overview.strip()}\n"
        "</current_user_background>"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_root_not_object:{path.name}")
    return value


def _resolve_inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("publication_path_missing")
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("publication_path_escapes_layer2_root")
    return candidate


def _file_from_manifest(
    publication: Path, manifest: dict[str, Any], key: str
) -> Path:
    files = manifest.get("files")
    if not isinstance(files, dict) or not isinstance(files.get(key), dict):
        raise ValueError(f"manifest_file_missing:{key}")
    record = files[key]
    relative = record.get("name")
    expected_hash = record.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise ValueError(f"manifest_file_invalid:{key}")
    path = _resolve_inside(publication, relative)
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise ValueError(f"manifest_file_hash_mismatch:{key}")
    return path


def load_layer2_publication(
    history_root: str | Path | None = None,
) -> Layer2ContextResult:
    """Load one self-consistent publication through the atomic pointer."""

    history = resolve_character_history_root(history_root).resolve()
    root = history / LAYER2_DIRECTORY_NAME
    pointer_path = root / CURRENT_POINTER_NAME
    base_diagnostics: dict[str, Any] = {
        "status": "missing",
        "layer2_root": str(root),
        "pointer_path": str(pointer_path),
    }
    if not pointer_path.is_file():
        return Layer2ContextResult("", base_diagnostics)
    try:
        pointer = _read_json(pointer_path)
        publication = _resolve_inside(root, pointer.get("publication", ""))
        manifest_path = publication / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("publication_manifest_missing")
        expected_manifest_hash = pointer.get("manifest_sha256")
        if not isinstance(expected_manifest_hash, str):
            raise ValueError("pointer_manifest_hash_missing")
        if sha256_file(manifest_path) != expected_manifest_hash:
            raise ValueError("pointer_manifest_hash_mismatch")
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "published":
            raise ValueError("publication_status_not_published")
        as_of_date = manifest.get("as_of_date")
        if not isinstance(as_of_date, str) or pointer.get("as_of_date") != as_of_date:
            raise ValueError("pointer_as_of_date_mismatch")
        background_path = _file_from_manifest(publication, manifest, "background")
        overview_path = _file_from_manifest(publication, manifest, "model_facing_overview")
        _file_from_manifest(publication, manifest, "projection_result")
        overview = overview_path.read_text(encoding="utf-8").strip()
        if not overview:
            raise ValueError("model_facing_overview_empty")
        files = manifest.get("files", {})
        if isinstance(files, dict) and "model_context_preview" in files:
            preview_path = _file_from_manifest(
                publication, manifest, "model_context_preview"
            )
            context = preview_path.read_text(encoding="utf-8").strip()
            if context != format_layer2_context(as_of_date, overview):
                raise ValueError("model_context_preview_content_mismatch")
        else:
            # Backward-compatible loader for the one bootstrap publication made
            # before the exact preview artifact became part of the contract.
            context = format_layer2_context(as_of_date, overview)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return Layer2ContextResult(
            "",
            {
                **base_diagnostics,
                "status": "invalid",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )

    return Layer2ContextResult(
        context,
        {
            **base_diagnostics,
            "status": "loaded",
            "as_of_date": as_of_date,
            "publication": str(publication),
            "character_count": len(context),
            "overview_character_count": len(overview),
            "diary_sha256": manifest.get("diary_sha256"),
        },
        background_path=background_path,
        overview_path=overview_path,
    )


def load_current_layer2_context(
    history_root: str | Path | None = None,
) -> Layer2ContextResult:
    """Load the root-level editable context, falling back to the publication.

    ``current_layer2.md`` is intentionally outside the immutable manifest. This
    makes a user's saved edit visible on the next conversation turn. The
    verified publication remains the recovery source when the editable file is
    missing, empty, too large, or not valid UTF-8.
    """

    history = resolve_character_history_root(history_root).resolve()
    root = history / LAYER2_DIRECTORY_NAME
    current_context_path = root / CURRENT_CONTEXT_NAME
    publication = load_layer2_publication(history)
    try:
        context = _read_editable_current_context(current_context_path)
    except FileNotFoundError:
        return Layer2ContextResult(
            publication.context,
            {
                **publication.diagnostics,
                "context_source": "publication_fallback",
                "current_context_status": "missing",
                "current_context_path": str(current_context_path),
            },
            background_path=publication.background_path,
            overview_path=publication.overview_path,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        return Layer2ContextResult(
            publication.context,
            {
                **publication.diagnostics,
                "context_source": "publication_fallback",
                "current_context_status": "invalid",
                "current_context_error": str(exc),
                "current_context_path": str(current_context_path),
            },
            background_path=publication.background_path,
            overview_path=publication.overview_path,
        )

    diagnostics = {
        **publication.diagnostics,
        "status": "loaded",
        "context_source": "editable_root",
        "current_context_status": "loaded",
        "current_context_path": str(current_context_path),
        "character_count": len(context),
    }
    if publication.diagnostics.get("status") != "loaded":
        diagnostics["publication_status"] = publication.diagnostics.get("status")
        if "error" in publication.diagnostics:
            diagnostics["publication_error"] = publication.diagnostics["error"]
        diagnostics.pop("error", None)
        diagnostics.pop("error_type", None)
    return Layer2ContextResult(
        context,
        diagnostics,
        background_path=publication.background_path,
        overview_path=publication.overview_path,
    )


def load_layer2_publication_as_of(
    history_root: str | Path | None,
    max_as_of_date: str | date,
) -> Layer2ContextResult:
    """Load the newest verified publication not newer than ``max_as_of_date``.

    Daily diary generation must not use a Layer-2 publication built from the
    target day's own diary (or a later day).  This read-only selector scans
    published manifests instead of trusting the live ``current.json`` pointer.
    """

    cutoff = (
        max_as_of_date
        if isinstance(max_as_of_date, date)
        else date.fromisoformat(str(max_as_of_date))
    )
    history = resolve_character_history_root(history_root).resolve()
    root = history / LAYER2_DIRECTORY_NAME
    publications = root / "publications"
    warnings: list[str] = []
    if not publications.is_dir():
        return Layer2ContextResult(
            "",
            {
                "status": "missing",
                "as_of_cutoff": cutoff.isoformat(),
                "warnings": warnings,
            },
        )

    candidates: list[tuple[date, Path, dict[str, Any]]] = []
    for publication in sorted(publications.iterdir()):
        if not publication.is_dir():
            continue
        try:
            manifest = _read_json(publication / "manifest.json")
            as_of = date.fromisoformat(str(manifest.get("as_of_date")))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            warnings.append(f"跳过无效第二层历史发布 {publication.name}：{exc}")
            continue
        if manifest.get("status") == "published" and as_of <= cutoff:
            candidates.append((as_of, publication, manifest))
    if not candidates:
        return Layer2ContextResult(
            "",
            {
                "status": "missing",
                "as_of_cutoff": cutoff.isoformat(),
                "warnings": warnings,
            },
        )

    as_of, publication, manifest = max(candidates, key=lambda item: item[0])
    try:
        background_path = _file_from_manifest(publication, manifest, "background")
        overview_path = _file_from_manifest(
            publication, manifest, "model_facing_overview"
        )
        _file_from_manifest(publication, manifest, "projection_result")
        overview = overview_path.read_text(encoding="utf-8").strip()
        if not overview:
            raise ValueError("model_facing_overview_empty")
        context = format_layer2_context(as_of.isoformat(), overview)
        files = manifest.get("files", {})
        if isinstance(files, dict) and "model_context_preview" in files:
            preview_path = _file_from_manifest(
                publication, manifest, "model_context_preview"
            )
            if preview_path.read_text(encoding="utf-8").strip() != context:
                raise ValueError("model_context_preview_content_mismatch")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return Layer2ContextResult(
            "",
            {
                "status": "invalid",
                "as_of_cutoff": cutoff.isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "warnings": warnings,
            },
        )
    return Layer2ContextResult(
        context,
        {
            "status": "loaded",
            "as_of_date": as_of.isoformat(),
            "as_of_cutoff": cutoff.isoformat(),
            "publication": str(publication),
            "character_count": len(context),
            "diary_sha256": manifest.get("diary_sha256"),
            "warnings": warnings,
        },
        background_path=background_path,
        overview_path=overview_path,
    )


__all__ = [
    "CURRENT_CONTEXT_NAME",
    "CURRENT_POINTER_NAME",
    "LAYER2_DIRECTORY_NAME",
    "Layer2ContextResult",
    "format_layer2_context",
    "load_current_layer2_context",
    "load_layer2_publication",
    "load_layer2_publication_as_of",
    "sha256_file",
]
