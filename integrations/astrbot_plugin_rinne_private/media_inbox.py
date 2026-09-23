"""Download QQ transport media into the user-selected private inbox."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import io
import os
import socket
import uuid
import wave
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

import aiohttp


MAX_MEDIA_BYTES = 50 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_AUDIO_BYTES = 20 * 1024 * 1024
MAX_AUDIO_SECONDS = 5 * 60
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
DOCUMENT_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".webm", ".mpeg", ".mpg"}


class MediaInboxError(ValueError):
    pass


def _safe_name(name: str, *, fallback: str) -> str:
    normalized = str(name or "").replace("\\", "/")
    basename = PurePosixPath(normalized).name.replace("\0", "").strip()
    for character in ':*?"<>|':
        basename = basename.replace(character, "_")
    return basename if basename not in {"", ".", ".."} else fallback


def _validate_bytes(kind: str, name: str, data: bytes) -> tuple[str, str]:
    suffix = Path(name).suffix.casefold()
    if kind == "image":
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            detected, mime = ".png", "image/png"
        elif data.startswith(b"\xff\xd8\xff") and data.endswith(b"\xff\xd9"):
            detected, mime = ".jpg", "image/jpeg"
        elif data.startswith((b"GIF87a", b"GIF89a")):
            detected, mime = ".gif", "image/gif"
        elif len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            detected, mime = ".webp", "image/webp"
        else:
            raise MediaInboxError("unsupported image bytes")
        if suffix:
            normalized = ".jpg" if suffix == ".jpeg" else suffix
            if normalized != detected:
                raise MediaInboxError("image extension does not match bytes")
            return suffix, mime
        return detected, mime

    if kind == "audio":
        try:
            with wave.open(io.BytesIO(data), "rb") as source:
                channels = source.getnchannels()
                sample_width = source.getsampwidth()
                sample_rate = source.getframerate()
                frame_count = source.getnframes()
                compression = source.getcomptype()
                if channels not in {1, 2}:
                    raise MediaInboxError("audio must be mono or stereo")
                if sample_width != 2 or compression != "NONE":
                    raise MediaInboxError("audio must be 16-bit PCM WAV")
                if not 8_000 <= sample_rate <= 48_000:
                    raise MediaInboxError("audio sample rate is unsupported")
                if frame_count < 1 or frame_count > sample_rate * MAX_AUDIO_SECONDS:
                    raise MediaInboxError("audio duration is outside the limit")
                if len(source.readframes(frame_count)) != frame_count * channels * 2:
                    raise MediaInboxError("audio PCM data is incomplete")
        except (EOFError, wave.Error) as exc:
            raise MediaInboxError("invalid WAV audio") from exc
        if suffix not in {"", ".wav"}:
            raise MediaInboxError("audio extension must be WAV")
        return ".wav", "audio/wav"

    if suffix not in DOCUMENT_SUFFIXES:
        raise MediaInboxError("unsupported document extension")
    if suffix in {".txt", ".md"}:
        try:
            data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise MediaInboxError("text document must be UTF-8") from exc
        return suffix, "text/markdown" if suffix == ".md" else "text/plain"
    if suffix == ".pdf":
        if not data[:1024].lstrip().startswith(b"%PDF-"):
            raise MediaInboxError("invalid PDF signature")
        return suffix, "application/pdf"
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                raise MediaInboxError("invalid DOCX structure")
    except zipfile.BadZipFile as exc:
        raise MediaInboxError("invalid DOCX container") from exc
    return suffix, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


async def _require_public_https(url: str) -> str:
    parsed = urlsplit(str(url or "").strip())
    try:
        port = parsed.port
    except ValueError as exc:
        raise MediaInboxError("QQ media URL has an invalid port") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or port not in {None, 443}
    ):
        raise MediaInboxError("QQ media URL must be public HTTPS")
    try:
        addresses = await asyncio.to_thread(
            socket.getaddrinfo,
            parsed.hostname,
            443,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise MediaInboxError("QQ media host could not be resolved") from exc
    if not addresses:
        raise MediaInboxError("QQ media host did not resolve")
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0].split("%", 1)[0])
        if not ip.is_global:
            raise MediaInboxError("QQ media URL resolved outside public internet")
    return parsed.geturl()


class MediaInbox:
    def __init__(self, data_root: str | Path) -> None:
        self.root = Path(data_root).expanduser().resolve(strict=False) / "inbox"
        if os.name == "nt" and self.root.drive.casefold() != "g:":
            raise MediaInboxError("private QQ inbox must stay on G drive")
        self.root.mkdir(parents=True, exist_ok=True)

    async def stage_url(
        self,
        *,
        kind: str,
        name: str,
        url: str,
    ) -> dict[str, object]:
        if kind not in {"image", "document"}:
            raise MediaInboxError("unsupported media kind")
        safe_url = await _require_public_https(url)
        url_name = Path(unquote(urlsplit(safe_url).path)).name
        safe_name = _safe_name(name, fallback=url_name or kind)
        timeout = aiohttp.ClientTimeout(total=120, connect=10, sock_read=60)
        chunks: list[bytes] = []
        total = 0
        async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
            async with session.get(safe_url, allow_redirects=False) as response:
                if response.status != 200:
                    raise MediaInboxError("QQ media download did not return HTTP 200")
                declared = response.content_length
                if declared is not None and not 0 < declared <= MAX_MEDIA_BYTES:
                    raise MediaInboxError("QQ media size is outside the limit")
                async for chunk in response.content.iter_chunked(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_MEDIA_BYTES:
                        raise MediaInboxError("QQ media exceeds the size limit")
                    chunks.append(chunk)
        if total < 1:
            raise MediaInboxError("QQ media is empty")
        data = b"".join(chunks)
        suffix, _mime = _validate_bytes(kind, safe_name, data)
        if not Path(safe_name).suffix:
            safe_name = f"{safe_name}{suffix}"
        stored_name = f"{uuid.uuid4().hex}{suffix}"
        destination = self.root / stored_name
        temporary = self.root / f".{stored_name}.part"
        try:
            with temporary.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise MediaInboxError("QQ media could not be staged") from exc
        return {
            "name": safe_name,
            "relative_path": stored_name,
            "size": total,
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    async def stage_local_audio(
        self,
        *,
        name: str,
        path: str | Path,
    ) -> dict[str, object]:
        """Copy one AstrBot-normalized QQ voice into the private G-drive inbox."""

        return await asyncio.to_thread(self._stage_local_audio, name, Path(path))

    async def stage_local_media(
        self,
        *,
        kind: str,
        name: str,
        path: str | Path,
    ) -> dict[str, object]:
        """Copy AstrBot-normalized image/document media from a G-drive temp file."""

        if kind not in {"image", "document"}:
            raise MediaInboxError("unsupported local media kind")
        return await asyncio.to_thread(
            self._stage_local_media,
            kind,
            name,
            Path(path),
        )

    async def stage_local_video(
        self,
        *,
        name: str,
        path: str | Path,
    ) -> dict[str, object]:
        """Stream-copy one AstrBot-resolved QQ video into the private inbox."""

        return await asyncio.to_thread(self._stage_local_video, name, Path(path))

    def _stage_local_video(self, name: str, path: Path) -> dict[str, object]:
        try:
            source = path.expanduser().resolve(strict=True)
        except OSError as exc:
            raise MediaInboxError("QQ local video is missing") from exc
        if os.name == "nt" and source.drive.casefold() != "g:":
            raise MediaInboxError("QQ local video source must stay on G drive")
        if not source.is_file() or source.is_symlink():
            raise MediaInboxError("QQ local video source is not a regular file")
        try:
            declared_size = source.stat().st_size
        except OSError as exc:
            raise MediaInboxError("QQ local video could not be inspected") from exc
        if not 0 < declared_size <= MAX_VIDEO_BYTES:
            raise MediaInboxError("QQ local video size is outside the limit")

        source_name = source.name or "video.mp4"
        safe_name = _safe_name(name, fallback=source_name)
        suffix = Path(safe_name).suffix.casefold()
        if suffix not in VIDEO_SUFFIXES:
            safe_name = _safe_name(source_name, fallback="video.mp4")
            suffix = Path(safe_name).suffix.casefold()
        if suffix not in VIDEO_SUFFIXES:
            raise MediaInboxError("unsupported video extension")

        stored_name = f"{uuid.uuid4().hex}{suffix}"
        destination = self.root / stored_name
        temporary = self.root / f".{stored_name}.part"
        digest = hashlib.sha256()
        total = 0
        header = bytearray()
        try:
            with source.open("rb") as source_handle, temporary.open("xb") as target:
                while chunk := source_handle.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_VIDEO_BYTES:
                        raise MediaInboxError("QQ local video exceeds the size limit")
                    if len(header) < 64:
                        header.extend(chunk[: 64 - len(header)])
                    digest.update(chunk)
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            if total != declared_size:
                raise MediaInboxError("QQ local video changed while being copied")
            _video_mime(bytes(header), suffix)
            os.replace(temporary, destination)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise MediaInboxError("QQ local video could not be staged") from exc
        except MediaInboxError:
            temporary.unlink(missing_ok=True)
            raise
        return {
            "name": safe_name,
            "relative_path": stored_name,
            "size": total,
            "sha256": digest.hexdigest(),
        }

    def _stage_local_media(
        self,
        kind: str,
        name: str,
        path: Path,
    ) -> dict[str, object]:
        try:
            source = path.expanduser().resolve(strict=True)
        except OSError as exc:
            raise MediaInboxError("QQ local media is missing") from exc
        if os.name == "nt" and source.drive.casefold() != "g:":
            raise MediaInboxError("QQ local media source must stay on G drive")
        if not source.is_file() or source.is_symlink():
            raise MediaInboxError("QQ local media source is not a regular file")
        try:
            declared_size = source.stat().st_size
        except OSError as exc:
            raise MediaInboxError("QQ local media could not be inspected") from exc
        if not 0 < declared_size <= MAX_MEDIA_BYTES:
            raise MediaInboxError("QQ local media size is outside the limit")
        try:
            data = source.read_bytes()
        except OSError as exc:
            raise MediaInboxError("QQ local media could not be read") from exc
        if len(data) != declared_size:
            raise MediaInboxError("QQ local media changed while being read")

        source_name = source.name or kind
        safe_name = _safe_name(name, fallback=source_name)
        try:
            suffix, _mime = _validate_bytes(kind, safe_name, data)
        except MediaInboxError:
            # AstrBot may replace the original filename while normalizing an
            # image to JPEG.  The normalized local suffix is authoritative,
            # but bytes are still validated below.
            safe_name = _safe_name(source_name, fallback=kind)
            suffix, _mime = _validate_bytes(kind, safe_name, data)
        if not Path(safe_name).suffix:
            safe_name = f"{safe_name}{suffix}"
        stored_name = f"{uuid.uuid4().hex}{suffix}"
        destination = self.root / stored_name
        temporary = self.root / f".{stored_name}.part"
        try:
            with temporary.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise MediaInboxError("QQ local media could not be staged") from exc
        return {
            "name": safe_name,
            "relative_path": stored_name,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def _stage_local_audio(self, name: str, path: Path) -> dict[str, object]:
        try:
            source = path.expanduser().resolve(strict=True)
        except OSError as exc:
            raise MediaInboxError("QQ audio conversion output is missing") from exc
        if os.name == "nt" and source.drive.casefold() != "g:":
            raise MediaInboxError("QQ audio source must stay on G drive")
        if not source.is_file() or source.is_symlink():
            raise MediaInboxError("QQ audio source is not a regular file")
        try:
            declared_size = source.stat().st_size
        except OSError as exc:
            raise MediaInboxError("QQ audio source could not be inspected") from exc
        if not 0 < declared_size <= MAX_AUDIO_BYTES:
            raise MediaInboxError("QQ audio size is outside the limit")
        chunks: list[bytes] = []
        total = 0
        try:
            with source.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_AUDIO_BYTES:
                        raise MediaInboxError("QQ audio exceeds the size limit")
                    chunks.append(chunk)
        except OSError as exc:
            raise MediaInboxError("QQ audio could not be read") from exc
        data = b"".join(chunks)
        safe_name = _safe_name(name, fallback="voice.wav")
        suffix, _mime = _validate_bytes("audio", safe_name, data)
        if not Path(safe_name).suffix:
            safe_name = f"{safe_name}{suffix}"
        stored_name = f"{uuid.uuid4().hex}.wav"
        destination = self.root / stored_name
        temporary = self.root / f".{stored_name}.part"
        try:
            with temporary.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise MediaInboxError("QQ audio could not be staged") from exc
        return {
            "kind": "audio",
            "name": safe_name,
            "relative_path": stored_name,
            "size": total,
            "sha256": hashlib.sha256(data).hexdigest(),
        }


def _video_mime(header: bytes, suffix: str) -> str:
    if suffix in {".mp4", ".mov"}:
        if len(header) < 12 or header[4:8] != b"ftyp":
            raise MediaInboxError("MP4/MOV signature is invalid")
        return "video/quicktime" if suffix == ".mov" else "video/mp4"
    if suffix == ".avi":
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"AVI ":
            raise MediaInboxError("AVI signature is invalid")
        return "video/x-msvideo"
    if suffix == ".webm":
        if not header.startswith(b"\x1a\x45\xdf\xa3"):
            raise MediaInboxError("WebM signature is invalid")
        return "video/webm"
    if suffix in {".mpeg", ".mpg"}:
        if not header.startswith((b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3")):
            raise MediaInboxError("MPEG signature is invalid")
        return "video/mpeg"
    raise MediaInboxError("unsupported video extension")


__all__ = [
    "MAX_VIDEO_BYTES",
    "MediaInbox",
    "MediaInboxError",
    "VIDEO_SUFFIXES",
]
