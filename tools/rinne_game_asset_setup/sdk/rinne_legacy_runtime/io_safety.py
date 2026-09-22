from __future__ import annotations

from pathlib import Path


def ensure_distinct_file_paths(source: Path, destination: Path) -> None:
    """Reject an output path that could overwrite its read-only source."""

    source_resolved = source.resolve(strict=True)
    destination_resolved = destination.resolve(strict=False)
    if source_resolved == destination_resolved:
        raise ValueError("output path must not be the input file")
    if destination.exists() and source.samefile(destination):
        raise ValueError("output path must not be the input file or a hard link to it")


def write_new_bytes(destination: Path, data: bytes) -> None:
    """Create a new output exclusively and refuse to replace an existing file."""

    with destination.open("xb") as handle:
        handle.write(data)


__all__ = ["ensure_distinct_file_paths", "write_new_bytes"]
