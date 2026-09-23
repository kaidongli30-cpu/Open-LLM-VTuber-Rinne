"""Validate QQ media staged in the user-selected private inbox.

The AstrBot process may download transport media, but the Rinne backend never
accepts an arbitrary filesystem path.  It receives a basename plus size and
SHA-256, resolves that basename below one configured inbox, validates the
actual bytes, then copies supported content into Rinne's read-only library.
"""

from __future__ import annotations

import base64
import hashlib
import io
import re
import wave
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..file_library import (
    MAX_FILE_BYTES,
    VIDEO_EXTENSIONS,
    save_upload,
    save_upload_path,
)


MAX_PRIVATE_MEDIA_ITEMS = 16
MAX_PRIVATE_VIDEO_BYTES = 100 * 1024 * 1024
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
DOCUMENT_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}
VIDEO_SUFFIXES = VIDEO_EXTENSIONS
AUDIO_SUFFIXES = {".wav"}
MAX_PRIVATE_AUDIO_BYTES = 20 * 1024 * 1024
MAX_PRIVATE_AUDIO_SECONDS = 5 * 60
PRIVATE_AUDIO_SAMPLE_RATE = 16_000
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REPARSE_POINT_ATTRIBUTE = 0x400


class PrivateMediaValidationError(ValueError):
    """Raised when staged QQ media is not safe to expose to the model."""


def _has_reparse_point(path: Path) -> bool:
    try:
        stat_result = path.stat(follow_symlinks=False)
    except (OSError, TypeError):
        return path.is_symlink()
    attributes = int(getattr(stat_result, "st_file_attributes", 0))
    return path.is_symlink() or bool(attributes & _REPARSE_POINT_ATTRIBUTE)


def _resolve_inbox_file(inbox_root: Path, relative_path: str) -> Path:
    raw = str(relative_path or "").strip()
    relative = Path(raw)
    if (
        not raw
        or relative.is_absolute()
        or relative.name != raw
        or raw in {".", ".."}
        or any(character in raw for character in ("/", "\\", ":", "\0"))
    ):
        raise PrivateMediaValidationError("media path must be one inbox basename")

    root = inbox_root.expanduser().resolve(strict=True)
    if _has_reparse_point(root):
        raise PrivateMediaValidationError("media inbox must not be a reparse point")
    candidate = root / raw
    if _has_reparse_point(candidate):
        raise PrivateMediaValidationError("media file must not be a reparse point")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise PrivateMediaValidationError("media file is missing") from exc
    if resolved.parent != root or not resolved.is_file():
        raise PrivateMediaValidationError("media file escapes the configured inbox")
    return resolved


def _read_verified(path: Path, *, expected_size: int, expected_sha256: str) -> bytes:
    if expected_size < 1 or expected_size > MAX_FILE_BYTES:
        raise PrivateMediaValidationError("media size is outside the allowed range")
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    total = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_FILE_BYTES:
                    raise PrivateMediaValidationError("media exceeds the size limit")
                digest.update(chunk)
                chunks.append(chunk)
    except OSError as exc:
        raise PrivateMediaValidationError("media could not be read") from exc
    if total != expected_size or digest.hexdigest() != expected_sha256:
        raise PrivateMediaValidationError("media integrity metadata does not match")
    return b"".join(chunks)


def _verify_video_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> bytes:
    if expected_size < 1 or expected_size > MAX_PRIVATE_VIDEO_BYTES:
        raise PrivateMediaValidationError("video size is outside the allowed range")
    digest = hashlib.sha256()
    total = 0
    header = bytearray()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_PRIVATE_VIDEO_BYTES:
                    raise PrivateMediaValidationError("video exceeds the size limit")
                if len(header) < 64:
                    header.extend(chunk[: 64 - len(header)])
                digest.update(chunk)
    except OSError as exc:
        raise PrivateMediaValidationError("video could not be read") from exc
    if total != expected_size or digest.hexdigest() != expected_sha256:
        raise PrivateMediaValidationError("video integrity metadata does not match")
    return bytes(header)


def _image_mime(data: bytes, suffix: str) -> str:
    detected = ""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        detected = ".png"
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff") and data.endswith(b"\xff\xd9"):
        detected = ".jpg"
        mime = "image/jpeg"
    elif data.startswith((b"GIF87a", b"GIF89a")):
        detected = ".gif"
        mime = "image/gif"
    elif len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        detected = ".webp"
        mime = "image/webp"
    else:
        raise PrivateMediaValidationError("image signature is unsupported")
    normalized_suffix = ".jpg" if suffix == ".jpeg" else suffix
    if normalized_suffix != detected:
        raise PrivateMediaValidationError("image extension does not match its bytes")
    return mime


def _document_mime(data: bytes, suffix: str) -> str:
    if suffix in {".txt", ".md"}:
        try:
            data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise PrivateMediaValidationError("text document must be UTF-8") from exc
        return "text/markdown" if suffix == ".md" else "text/plain"
    if suffix == ".pdf":
        if not data[:1024].lstrip().startswith(b"%PDF-"):
            raise PrivateMediaValidationError("PDF signature is invalid")
        return "application/pdf"
    if suffix == ".docx":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                names = set(archive.namelist())
                if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                    raise PrivateMediaValidationError("DOCX structure is invalid")
        except zipfile.BadZipFile as exc:
            raise PrivateMediaValidationError("DOCX container is invalid") from exc
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    raise PrivateMediaValidationError("document type is unsupported")


def _video_mime(header: bytes, suffix: str) -> str:
    if suffix in {".mp4", ".mov"}:
        if len(header) < 12 or header[4:8] != b"ftyp":
            raise PrivateMediaValidationError("MP4/MOV signature is invalid")
        return "video/quicktime" if suffix == ".mov" else "video/mp4"
    if suffix == ".avi":
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"AVI ":
            raise PrivateMediaValidationError("AVI signature is invalid")
        return "video/x-msvideo"
    if suffix == ".webm":
        if not header.startswith(b"\x1a\x45\xdf\xa3"):
            raise PrivateMediaValidationError("WebM signature is invalid")
        return "video/webm"
    if suffix in {".mpeg", ".mpg"}:
        if not header.startswith((b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3")):
            raise PrivateMediaValidationError("MPEG signature is invalid")
        return "video/mpeg"
    raise PrivateMediaValidationError("video type is unsupported")


def _decode_private_audio(data: bytes) -> np.ndarray:
    """Decode a bounded PCM WAV and normalize it for the configured ASR."""

    try:
        with wave.open(io.BytesIO(data), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            frame_count = source.getnframes()
            compression = source.getcomptype()
            if channels not in {1, 2}:
                raise PrivateMediaValidationError("audio must be mono or stereo")
            if sample_width != 2 or compression != "NONE":
                raise PrivateMediaValidationError("audio must be 16-bit PCM WAV")
            if not 8_000 <= sample_rate <= 48_000:
                raise PrivateMediaValidationError("audio sample rate is unsupported")
            if frame_count < 1 or frame_count > sample_rate * MAX_PRIVATE_AUDIO_SECONDS:
                raise PrivateMediaValidationError("audio duration is outside the limit")
            raw = source.readframes(frame_count)
    except (EOFError, wave.Error) as exc:
        raise PrivateMediaValidationError("audio WAV container is invalid") from exc

    expected_bytes = frame_count * channels * sample_width
    if len(raw) != expected_bytes:
        raise PrivateMediaValidationError("audio PCM data is incomplete")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1, dtype=np.float32)
    if sample_rate != PRIVATE_AUDIO_SAMPLE_RATE:
        target_count = max(
            1,
            round(len(samples) * PRIVATE_AUDIO_SAMPLE_RATE / sample_rate),
        )
        source_positions = np.arange(len(samples), dtype=np.float64)
        target_positions = np.linspace(
            0,
            len(samples) - 1,
            target_count,
            dtype=np.float64,
        )
        samples = np.interp(target_positions, source_positions, samples).astype(
            np.float32
        )
    return np.ascontiguousarray(samples, dtype=np.float32)


def import_private_audio(
    descriptor: Mapping[str, Any],
    *,
    inbox_root: str | Path,
) -> np.ndarray:
    """Validate one staged QQ voice recording and return 16 kHz mono samples."""

    if not isinstance(descriptor, Mapping):
        raise PrivateMediaValidationError("audio descriptor must be an object")
    if str(descriptor.get("kind") or "").strip().casefold() != "audio":
        raise PrivateMediaValidationError("audio descriptor kind is invalid")
    name = Path(str(descriptor.get("name") or "")).name.strip()
    if not name or name in {".", ".."} or "\0" in name:
        raise PrivateMediaValidationError("audio filename is invalid")
    if Path(name).suffix.casefold() not in AUDIO_SUFFIXES:
        raise PrivateMediaValidationError("audio extension is unsupported")
    relative_path = str(descriptor.get("relative_path") or "").strip()
    if Path(relative_path).suffix.casefold() != ".wav":
        raise PrivateMediaValidationError("staged audio must be WAV")
    try:
        expected_size = int(descriptor.get("size"))
    except (TypeError, ValueError) as exc:
        raise PrivateMediaValidationError("audio size is invalid") from exc
    if not 0 < expected_size <= MAX_PRIVATE_AUDIO_BYTES:
        raise PrivateMediaValidationError("audio size is outside the limit")
    expected_sha256 = str(descriptor.get("sha256") or "").strip().casefold()
    if not _SHA256_PATTERN.fullmatch(expected_sha256):
        raise PrivateMediaValidationError("audio SHA-256 is invalid")
    source = _resolve_inbox_file(Path(inbox_root), relative_path)
    data = _read_verified(
        source,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
    )
    return _decode_private_audio(data)


def import_private_media(
    descriptors: Iterable[Mapping[str, Any]],
    *,
    inbox_root: str | Path,
    library_root: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate staged descriptors and return vision inputs plus library refs."""

    raw_items = list(descriptors)
    if len(raw_items) > MAX_PRIVATE_MEDIA_ITEMS:
        raise PrivateMediaValidationError("too many media items in one turn")

    images: list[dict[str, Any]] = []
    attachments: list[dict[str, Any]] = []
    inbox = Path(inbox_root)
    library = Path(library_root)
    for descriptor in raw_items:
        if not isinstance(descriptor, Mapping):
            raise PrivateMediaValidationError("media descriptor must be an object")
        kind = str(descriptor.get("kind") or "").strip().casefold()
        if kind not in {"image", "document", "video"}:
            raise PrivateMediaValidationError("media kind is unsupported")
        name = Path(str(descriptor.get("name") or "")).name.strip()
        if not name or name in {".", ".."} or "\0" in name:
            raise PrivateMediaValidationError("media filename is invalid")
        suffix = Path(name).suffix.casefold()
        allowed = {
            "image": IMAGE_SUFFIXES,
            "document": DOCUMENT_SUFFIXES,
            "video": VIDEO_SUFFIXES,
        }[kind]
        if suffix not in allowed:
            raise PrivateMediaValidationError("media extension is unsupported")

        relative_path = str(descriptor.get("relative_path") or "").strip()
        if Path(relative_path).suffix.casefold() != suffix:
            raise PrivateMediaValidationError("staged extension does not match filename")
        try:
            expected_size = int(descriptor.get("size"))
        except (TypeError, ValueError) as exc:
            raise PrivateMediaValidationError("media size is invalid") from exc
        expected_sha256 = str(descriptor.get("sha256") or "").strip().casefold()
        if not _SHA256_PATTERN.fullmatch(expected_sha256):
            raise PrivateMediaValidationError("media SHA-256 is invalid")

        source = _resolve_inbox_file(inbox, relative_path)
        if kind == "video":
            header = _verify_video_file(
                source,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            )
            mime_type = _video_mime(header, suffix)
            record = save_upload_path(
                name,
                source,
                mime_type=mime_type,
                kind="video",
                subdir="QQ收件箱",
                root=library,
            )
            attachments.append(record)
            continue

        data = _read_verified(
            source,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        mime_type = (
            _image_mime(data, suffix)
            if kind == "image"
            else _document_mime(data, suffix)
        )
        record = save_upload(
            name,
            data,
            mime_type=mime_type,
            kind=kind,
            subdir="QQ收件箱",
            root=library,
        )
        attachments.append(record)
        if kind == "image":
            encoded = base64.b64encode(data).decode("ascii")
            images.append(
                {
                    "source": "upload",
                    "data": f"data:{mime_type};base64,{encoded}",
                    "mime_type": mime_type,
                    "persist": False,
                }
            )
    return images, attachments


__all__ = [
    "AUDIO_SUFFIXES",
    "DOCUMENT_SUFFIXES",
    "IMAGE_SUFFIXES",
    "MAX_PRIVATE_AUDIO_BYTES",
    "MAX_PRIVATE_MEDIA_ITEMS",
    "MAX_PRIVATE_VIDEO_BYTES",
    "PRIVATE_AUDIO_SAMPLE_RATE",
    "PrivateMediaValidationError",
    "VIDEO_SUFFIXES",
    "import_private_audio",
    "import_private_media",
]
