"""Local file library for Rinne.

Users may organise files below ``documents``, ``images`` and ``videos``.  This
module handles deterministic listing and literal filename lookup; the MCP layer
may add a local semantic filename ranking without opening file contents.
"""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import os
import re
import shutil
import subprocess
import zipfile
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree


LIBRARY_ENV = "RINNE_LIBRARY_ROOT"
DEFAULT_LIBRARY_ROOT = Path("rinne_library") / "rinne_01"
DOCUMENT_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mpeg", ".mpg"}
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_VIDEO_FILE_BYTES = 2 * 1024 * 1024 * 1024
MAX_INLINE_VIDEO_BYTES = 70 * 1024 * 1024
MAX_READ_CHARS = 50_000
_PDF_TJ_PATTERN = re.compile(r"\(((?:\\.|[^\\)])*)\)\s*Tj", re.S)
_PDF_TJ_ARRAY_PATTERN = re.compile(r"\[((?:\\.|[^\]])*)\]\s*TJ", re.S)
_PDF_LITERAL_PATTERN = re.compile(r"\(((?:\\.|[^\\)])*)\)", re.S)
_NUMBERED_COPY_PATTERN = re.compile(r" \(\d+\)$")


def get_library_root(root: str | Path | None = None) -> Path:
    """Return the fixed Rinne library root without creating it."""

    value = root or os.environ.get(LIBRARY_ENV) or DEFAULT_LIBRARY_ROOT
    return Path(value).expanduser().resolve()


def ensure_library_structure(root: str | Path | None = None) -> Path:
    """Create the top-level folders and the default inboxes on first use."""

    library_root = get_library_root(root)
    for top in ("documents", "images", "videos"):
        (library_root / top / "待整理").mkdir(parents=True, exist_ok=True)
    return library_root


def _safe_relative_path(relative_path: str) -> str:
    value = str(relative_path or "").replace("\\", "/").strip("/")
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or any(":" in part or "\x00" in part for part in path.parts)
    ):
        raise ValueError("library path must be relative and stay inside the library")
    return "/".join(part for part in path.parts if part not in {"", "."})


def _path_for_relative(root: Path, relative_path: str) -> Path:
    safe = _safe_relative_path(relative_path)
    candidate = (root / safe).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("library path escapes the library root")
    return candidate


def _kind_for_name(name: str) -> str | None:
    suffix = Path(name).suffix.casefold()
    if suffix in DOCUMENT_EXTENSIONS:
        return "document"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return None


def kind_for_filename(name: str) -> str | None:
    """Return the supported library kind for a user-supplied filename."""

    return _kind_for_name(Path(str(name or "")).name)


def _mime_type(name: str, fallback: str | None = None) -> str:
    return fallback or mimetypes.guess_type(name)[0] or "application/octet-stream"


def _file_id(path: Path) -> str:
    return _sha256_path(path)[:24]


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_duplicate_content(root: Path, data: bytes, kind: str) -> Path | None:
    """Return the canonical existing file with the same complete SHA-256."""

    content_hash = hashlib.sha256(data).hexdigest()
    candidates = sorted(
        _iter_files(root, kind),
        key=lambda path: (
            bool(_NUMBERED_COPY_PATTERN.search(path.stem)),
            path.relative_to(root).as_posix().casefold(),
        ),
    )
    for path in candidates:
        if path.stat().st_size != len(data):
            continue
        if _sha256_path(path) == content_hash:
            return path
    return None


def _find_duplicate_path(root: Path, source: Path, kind: str) -> Path | None:
    content_hash = _sha256_path(source)
    candidates = sorted(
        _iter_files(root, kind),
        key=lambda path: (
            bool(_NUMBERED_COPY_PATTERN.search(path.stem)),
            path.relative_to(root).as_posix().casefold(),
        ),
    )
    for path in candidates:
        if path.stat().st_size == source.stat().st_size and _sha256_path(path) == content_hash:
            return path
    return None


def _record(root: Path, path: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    kind = _kind_for_name(path.name)
    stat = path.stat()
    return {
        "file_id": _file_id(path),
        "name": path.name,
        "relative_path": relative,
        "kind": kind,
        "mime_type": _mime_type(path.name),
        "size": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
    }


def _iter_files(root: Path, kind: str = "all") -> Iterable[Path]:
    if not root.is_dir():
        return []
    allowed_by_kind = {
        "document": DOCUMENT_EXTENSIONS,
        "image": IMAGE_EXTENSIONS,
        "video": VIDEO_EXTENSIONS,
    }
    if kind not in {"all", *allowed_by_kind}:
        raise ValueError("kind must be all, document, image, or video")
    return (
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(
            part.startswith(".") for part in path.relative_to(root).parts[:-1]
        )
        and _kind_for_name(path.name) is not None
        and (kind == "all" or path.suffix.casefold() in allowed_by_kind[kind])
    )


def list_library_files(
    root: str | Path | None = None,
    *,
    kind: str = "all",
    folder: str = "",
    name_contains: str = "",
) -> list[dict[str, Any]]:
    """List files, optionally restricted to a folder and filename substring."""

    library_root = get_library_root(root)
    scan_root = library_root if not folder else _path_for_relative(library_root, folder)
    query = str(name_contains or "").casefold()
    records = []
    for path in _iter_files(scan_root, kind):
        if query and query not in path.name.casefold():
            continue
        records.append(_record(library_root, path))
    return sorted(records, key=lambda item: item["relative_path"].casefold())


def list_library_folders(
    root: str | Path | None = None,
    *,
    kind: str = "all",
    folder: str = "",
    recursive: bool = False,
) -> list[dict[str, Any]]:
    """List real library folders for navigation.

    ``folder`` is a library-relative path.  It may include ``images``,
    ``documents`` or ``videos``; without a top-level prefix it is checked under
    every root when ``kind`` is ``all``.  Only directories are returned, never
    file contents.
    """

    if kind not in {"all", "document", "image", "video"}:
        raise ValueError("kind must be all, document, image, or video")

    library_root = get_library_root(root)
    top_by_kind = {
        "document": "documents",
        "image": "images",
        "video": "videos",
    }
    requested_tops = (
        [top_by_kind[kind]]
        if kind in top_by_kind
        else ["documents", "images", "videos"]
    )
    normalized_folder = str(folder or "").strip().replace("\\", "/").strip("/")
    folder_parts = Path(normalized_folder).parts if normalized_folder else ()
    explicit_top = folder_parts[0] if folder_parts and folder_parts[0] in requested_tops else None
    if (
        folder_parts
        and folder_parts[0] in {"documents", "images", "videos"}
        and explicit_top is None
    ):
        return []

    scan_roots: list[tuple[str, Path]] = []
    if explicit_top:
        scan_roots.append(
            (explicit_top, _path_for_relative(library_root, normalized_folder))
        )
    elif normalized_folder:
        for top in requested_tops:
            scan_roots.append(
                (top, _path_for_relative(library_root, f"{top}/{normalized_folder}"))
            )
    else:
        scan_roots = [(top, library_root / top) for top in requested_tops]

    folders: list[dict[str, Any]] = []
    for top, scan_root in scan_roots:
        if not scan_root.is_dir():
            continue
        candidates = scan_root.rglob("*") if recursive else scan_root.iterdir()
        for path in candidates:
            if not path.is_dir():
                continue
            folders.append(
                {
                    "name": path.name,
                    "relative_path": path.relative_to(library_root).as_posix(),
                    "kind": {
                        "documents": "document",
                        "images": "image",
                        "videos": "video",
                    }[top],
                }
            )

    return sorted(
        folders,
        key=lambda item: (item["relative_path"].casefold(), item["kind"]),
    )


def find_library_files(
    name: str,
    root: str | Path | None = None,
    *,
    kind: str = "all",
) -> list[dict[str, Any]]:
    """Find literal filename substrings without reading file contents."""

    if not str(name or "").strip():
        raise ValueError("name is required")
    library_root = get_library_root(root)
    query = str(name).casefold()
    records = [
        _record(library_root, path)
        for path in _iter_files(library_root, kind)
        if query in path.name.casefold()
    ]
    return sorted(records, key=lambda item: item["relative_path"].casefold())


def _resolve_file_reference(
    reference: str,
    root: str | Path | None = None,
    *,
    kind: str = "all",
) -> tuple[Path, dict[str, Any]]:
    library_root = get_library_root(root)
    raw = str(reference or "").strip()
    if not raw:
        raise ValueError("file_id or relative_path is required")

    # Prefer an explicit relative path.  A bare filename is intentionally not
    # treated as an arbitrary filesystem path; it is resolved by filename.
    candidate: Path | None = None
    if "/" in raw or "\\" in raw:
        candidate = _path_for_relative(library_root, raw)
    if candidate is not None and candidate.is_file():
        record = _record(library_root, candidate)
        if kind != "all" and record["kind"] != kind:
            raise ValueError(f"file is not a {kind}")
        return candidate, record

    matches = [
        (path, _record(library_root, path))
        for path in _iter_files(library_root, kind)
        if raw == _file_id(path) or raw.casefold() == path.name.casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("filename is ambiguous; use the relative_path or file_id")
    raise FileNotFoundError(f"library file not found: {raw}")


def save_upload(
    filename: str,
    data: bytes,
    *,
    mime_type: str | None = None,
    kind: str = "auto",
    subdir: str = "",
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Save a user upload under documents/images/videos and return its stable ref."""

    safe_name = Path(str(filename or "")).name.strip()
    detected_kind = _kind_for_name(safe_name)
    if kind == "auto":
        kind = detected_kind or ""
    if kind not in {"document", "image", "video"} or detected_kind != kind:
        raise ValueError(
            "only txt, md, pdf, docx, png, jpg, jpeg, webp, gif, mp4, mov, "
            "avi, webm, mpeg, or mpg files are supported"
        )
    if not safe_name or safe_name in {".", ".."}:
        raise ValueError("filename is required")
    size_limit = MAX_VIDEO_FILE_BYTES if kind == "video" else MAX_FILE_BYTES
    if len(data) > size_limit:
        raise ValueError(f"file exceeds the {size_limit // (1024 * 1024)} MB limit")

    library_root = ensure_library_structure(root)
    top = {
        "document": "documents",
        "image": "images",
        "video": "videos",
    }[kind]
    duplicate = _find_duplicate_content(library_root, data, kind)
    if duplicate is not None:
        return _record(library_root, duplicate)

    destination_dir = library_root / top
    if subdir:
        destination_dir = _path_for_relative(library_root, f"{top}/{subdir}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / safe_name
    stem, suffix = destination.stem, destination.suffix
    index = 1
    while destination.exists():
        destination = destination_dir / f"{stem} ({index}){suffix}"
        index += 1
    destination.write_bytes(data)
    result = _record(library_root, destination)
    result["mime_type"] = _mime_type(safe_name, mime_type)
    return result


def save_upload_path(
    filename: str,
    source_path: str | Path,
    *,
    mime_type: str | None = None,
    kind: str = "auto",
    subdir: str = "",
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Stream-copy a prepared upload into the library without loading it in memory."""

    source = Path(source_path).resolve(strict=True)
    if not source.is_file():
        raise ValueError("upload source must be a file")
    safe_name = Path(str(filename or "")).name.strip()
    detected_kind = _kind_for_name(safe_name)
    if kind == "auto":
        kind = detected_kind or ""
    if kind not in {"document", "image", "video"} or detected_kind != kind:
        raise ValueError("upload filename extension is unsupported or mismatched")
    if not safe_name or safe_name in {".", ".."}:
        raise ValueError("filename is required")
    size_limit = MAX_VIDEO_FILE_BYTES if kind == "video" else MAX_FILE_BYTES
    if source.stat().st_size > size_limit:
        raise ValueError(f"file exceeds the {size_limit // (1024 * 1024)} MB limit")

    library_root = ensure_library_structure(root)
    top = {"document": "documents", "image": "images", "video": "videos"}[kind]
    duplicate = _find_duplicate_path(library_root, source, kind)
    if duplicate is not None:
        return _record(library_root, duplicate)

    destination_dir = library_root / top
    if subdir:
        destination_dir = _path_for_relative(library_root, f"{top}/{subdir}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / safe_name
    stem, suffix = destination.stem, destination.suffix
    index = 1
    while destination.exists():
        destination = destination_dir / f"{stem} ({index}){suffix}"
        index += 1
    shutil.copyfile(source, destination)
    result = _record(library_root, destination)
    result["mime_type"] = _mime_type(safe_name, mime_type)
    return result


def save_data_url(
    data_url: str,
    *,
    source: str,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist a camera/screen data URL into the automatic-capture folder."""

    match = re.match(r"^data:([^;,]+);base64,(.+)$", str(data_url or ""), re.S)
    if not match:
        raise ValueError("image must be a base64 data URL")
    mime_type, encoded = match.groups()
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("invalid base64 image data") from exc
    captured_at = datetime.now()
    source_key = "摄像头" if source == "camera" else "屏幕"
    suffix = mimetypes.guess_extension(mime_type) or ".jpg"
    if suffix == ".jpe":
        suffix = ".jpg"
    filename = f"{captured_at.strftime('%Y%m%d_%H%M%S_%f')}_{source}{suffix}"
    return save_upload(
        filename,
        data,
        mime_type=mime_type,
        kind="image",
        subdir=f"自动保存/{source_key}/{captured_at:%Y/%m}",
        root=root,
    )


def normalize_attachment(
    attachment: Any,
    root: str | Path | None = None,
) -> dict[str, Any] | None:
    """Validate a frontend attachment ref without trusting its path."""

    if not isinstance(attachment, dict):
        return None
    references = (
        attachment.get("relative_path"),
        attachment.get("file_id"),
        attachment.get("name"),
    )
    attempted: set[str] = set()
    for reference in references:
        value = str(reference or "").strip()
        if not value or value in attempted:
            continue
        attempted.add(value)
        try:
            _path, record = _resolve_file_reference(value, root)
        except (FileNotFoundError, ValueError):
            continue
        return record
    return None


def read_document(
    reference: str,
    *,
    start: int = 0,
    max_chars: int = 12_000,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Read a text/markdown/PDF/DOCX document by stable ref or relative path."""

    path, record = _resolve_file_reference(reference, root, kind="document")
    if start < 0 or max_chars <= 0:
        raise ValueError("start must be >= 0 and max_chars must be positive")
    max_chars = min(max_chars, MAX_READ_CHARS)
    suffix = path.suffix.casefold()
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        text = _read_pdf_with_pdftotext(path)
        if text is None:
            try:
                from pypdf import PdfReader  # type: ignore
            except ImportError:
                text = _read_pdf_without_dependency(path)
            else:
                pages = []
                for index, page in enumerate(PdfReader(str(path)).pages, start=1):
                    pages.append(f"[第{index}页]\n{page.extract_text() or ''}")
                text = "\n\n".join(pages)
    elif suffix == ".docx":
        try:
            from docx import Document  # type: ignore
        except ImportError:
            text = _read_docx_without_dependency(path)
        else:
            text = "\n".join(paragraph.text for paragraph in Document(str(path)).paragraphs)
    else:
        raise ValueError("unsupported document type")
    return {
        **record,
        "start": start,
        "max_chars": max_chars,
        "text": text[start : start + max_chars],
        "has_more": start + max_chars < len(text),
        "total_chars": len(text),
    }


def _read_docx_without_dependency(path: Path) -> str:
    """Small dependency-free DOCX paragraph reader used when python-docx is absent."""

    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    paragraphs: list[str] = []
    for paragraph in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
        pieces = [
            node.text or ""
            for node in paragraph.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
        ]
        if pieces:
            paragraphs.append("".join(pieces))
    return "\n".join(paragraphs)


def _read_pdf_with_pdftotext(path: Path) -> str | None:
    """Use an installed Poppler/TeX Live pdftotext without invoking a shell."""

    executable = shutil.which("pdftotext")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "-enc", "UTF-8", "-nopgbrk", str(path), "-"],
            capture_output=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    text = result.stdout.decode("utf-8", errors="replace").strip()
    return text or None


def _read_pdf_without_dependency(path: Path) -> str:
    """Best-effort extraction for simple PDFs when pypdf is unavailable.

    This intentionally handles only the common literal-text operators.  A
    normal installation can use pypdf for full PDF extraction; returning a
    clear message is safer than pretending an image-only PDF has text.
    """

    raw = path.read_bytes()
    pieces: list[str] = []
    for stream in re.findall(rb"stream\r?\n(.*?)\r?\nendstream", raw, re.S):
        try:
            decoded = zlib.decompress(stream)
        except zlib.error:
            decoded = stream
        if b"Tj" not in decoded and b"TJ" not in decoded:
            continue
        text = decoded.decode("latin-1", errors="ignore")
        for match in _PDF_TJ_PATTERN.finditer(text):
            pieces.append(_decode_pdf_literal(match.group(1)))
        for match in _PDF_TJ_ARRAY_PATTERN.finditer(text):
            values = _PDF_LITERAL_PATTERN.findall(match.group(1))
            if values:
                pieces.append("".join(_decode_pdf_literal(value) for value in values))
    return "\n".join(pieces) or "[此 PDF 没有可直接提取的文字，可能是扫描图片。]"


def _decode_pdf_literal(value: str) -> str:
    value = re.sub(r"\\([\\()])", r"\1", value)
    return value.replace(r"\n", "\n").replace(r"\r", "\r")


def view_image(
    reference: str,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Return image bytes as base64 for the MCP image bridge."""

    path, record = _resolve_file_reference(reference, root, kind="image")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("image exceeds the library size limit")
    return {
        **record,
        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        "mime_type": _mime_type(path.name),
    }


def resolve_video_path(
    reference: str,
    *,
    root: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Resolve one validated library video without reading it into memory."""

    path, record = _resolve_file_reference(reference, root, kind="video")
    if path.stat().st_size > MAX_VIDEO_FILE_BYTES:
        raise ValueError("video exceeds the library size limit")
    return record, path
