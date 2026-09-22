"""Local video validation, ordinary-mode compression, and lossless splitting."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .file_library import MAX_INLINE_VIDEO_BYTES


MAX_INLINE_VIDEO_DURATION_SECONDS = 45.0


class VideoMediaError(RuntimeError):
    """Raised when local video preparation cannot be completed safely."""


@dataclass(frozen=True)
class VideoProbe:
    duration_seconds: float
    width: int
    height: int
    has_audio: bool


@dataclass(frozen=True)
class VideoSegment:
    path: Path
    start_seconds: float
    duration_seconds: float


def _executable(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        raise VideoMediaError(f"{name} is required for video processing")
    return executable


def _run(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as exc:
        raise VideoMediaError("video processing timed out") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip().splitlines()
        message = detail[-1][:300] if detail else "unknown ffmpeg error"
        raise VideoMediaError(f"video processing failed: {message}") from exc


def probe_video(path: str | Path) -> VideoProbe:
    source = Path(path).resolve(strict=True)
    completed = _run(
        [
            _executable("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height",
            "-of",
            "json",
            str(source),
        ],
        timeout=60.0,
    )
    try:
        payload = json.loads(completed.stdout)
        duration = float((payload.get("format") or {}).get("duration") or 0)
        streams = payload.get("streams") or []
        video = next(item for item in streams if item.get("codec_type") == "video")
        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
    except (ValueError, TypeError, StopIteration, json.JSONDecodeError) as exc:
        raise VideoMediaError("file does not contain a readable video stream") from exc
    if not math.isfinite(duration) or duration <= 0 or width <= 0 or height <= 0:
        raise VideoMediaError("video duration or dimensions are invalid")
    return VideoProbe(
        duration_seconds=duration,
        width=width,
        height=height,
        has_audio=any(item.get("codec_type") == "audio" for item in streams),
    )


def prepare_ordinary_video(
    source_path: str | Path,
    output_path: str | Path,
) -> VideoProbe:
    """Create a space-saving 720p H.264/AAC copy for ordinary sending."""

    source = Path(source_path).resolve(strict=True)
    destination = Path(output_path).resolve()
    probe = probe_video(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _executable("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            "scale=w='min(1280,iw)':h=-2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "25",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-sn",
            "-dn",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        timeout=max(300.0, probe.duration_seconds * 4),
    )
    probe_video(destination)
    return probe


def _segment_once(
    source: Path,
    destination: Path,
    segment_seconds: float,
) -> list[Path]:
    for existing in destination.glob("segment-*.*"):
        existing.unlink()
    pattern = destination / f"segment-%03d{source.suffix.casefold()}"
    _run(
        [
            _executable("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0",
            "-c",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            f"{segment_seconds:.3f}",
            "-reset_timestamps",
            "1",
            str(pattern),
        ],
        timeout=max(300.0, probe_video(source).duration_seconds * 2),
    )
    return sorted(destination.glob("segment-*.*"))


def split_video_for_inline(
    source_path: str | Path,
    destination_dir: str | Path,
    *,
    max_bytes: int = MAX_INLINE_VIDEO_BYTES,
    max_duration_seconds: float = MAX_INLINE_VIDEO_DURATION_SECONDS,
) -> list[VideoSegment]:
    """Losslessly split a source into provider-safe media parts."""

    source = Path(source_path).resolve(strict=True)
    source_probe = probe_video(source)
    source_size = source.stat().st_size
    if max_duration_seconds <= 0:
        raise ValueError("max_duration_seconds must be positive")
    if (
        source_size <= max_bytes
        and source_probe.duration_seconds <= max_duration_seconds
    ):
        return [VideoSegment(source, 0.0, source_probe.duration_seconds)]

    destination = Path(destination_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    target_bytes = max_bytes * 0.86
    byte_limited_seconds = source_probe.duration_seconds * target_bytes / source_size
    segment_seconds = min(
        max_duration_seconds,
        max(1.0, byte_limited_seconds),
    )
    paths: list[Path] = []
    for _attempt in range(6):
        paths = _segment_once(source, destination, segment_seconds)
        if paths and all(path.stat().st_size <= max_bytes for path in paths):
            break
        segment_seconds = max(0.5, segment_seconds * 0.7)
    else:
        raise VideoMediaError("lossless video segments still exceed the 70 MiB limit")

    segments: list[VideoSegment] = []
    offset = 0.0
    for path in paths:
        probe = probe_video(path)
        segments.append(VideoSegment(path, offset, probe.duration_seconds))
        offset += probe.duration_seconds
    return segments


__all__ = [
    "MAX_INLINE_VIDEO_DURATION_SECONDS",
    "VideoMediaError",
    "VideoProbe",
    "VideoSegment",
    "prepare_ordinary_video",
    "probe_video",
    "split_video_for_inline",
]
