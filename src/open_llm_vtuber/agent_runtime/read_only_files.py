"""Bounded, allowlisted and strictly read-only local file access.

The service has no mutation or command-execution method.  Every target is
resolved against an explicit root before it is inspected, which also prevents
``..`` traversal and existing symlink escapes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


class ReadOnlyFileAccessDenied(PermissionError):
    """Raised when a path is outside every explicitly allowed root."""


class ReadOnlyFileLimitExceeded(ValueError):
    """Raised when a read or directory scan would exceed a safety bound."""


class UnsupportedTextFile(ValueError):
    """Raised when a file cannot safely be exposed as ordinary text."""


@dataclass(frozen=True, slots=True)
class FileEntry:
    path: str
    kind: str
    size_bytes: int
    modified_at: str
    is_symlink: bool


class ReadOnlyFileService:
    """Read files only within narrow, explicit roots.

    Empty roots mean no access.  Filesystem roots such as ``C:\\`` are rejected
    so a configuration mistake cannot expose an entire drive.
    """

    def __init__(
        self,
        allowed_roots: Iterable[str | Path] = (),
        *,
        max_read_bytes: int = 512 * 1024,
        max_directory_entries: int = 200,
        max_scan_entries: int = 20_000,
    ) -> None:
        if max_read_bytes < 1:
            raise ValueError("max_read_bytes must be positive")
        if max_directory_entries < 1:
            raise ValueError("max_directory_entries must be positive")
        if max_scan_entries < 1:
            raise ValueError("max_scan_entries must be positive")

        roots: list[Path] = []
        for root_value in allowed_roots:
            root = Path(root_value).expanduser().resolve(strict=True)
            if not root.is_dir():
                raise ValueError(f"allowed root is not a directory: {root}")
            if root == Path(root.anchor):
                raise ValueError("a filesystem root cannot be allowlisted")
            if root not in roots:
                roots.append(root)

        self._allowed_roots = tuple(roots)
        self._max_read_bytes = max_read_bytes
        self._max_directory_entries = max_directory_entries
        self._max_scan_entries = max_scan_entries

    @property
    def allowed_roots(self) -> tuple[str, ...]:
        return tuple(str(root) for root in self._allowed_roots)

    def _resolve_allowed(self, path_value: str | Path) -> Path:
        candidate = Path(path_value).expanduser().resolve(strict=True)
        for root in self._allowed_roots:
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            return candidate
        raise ReadOnlyFileAccessDenied("path is outside the configured read roots")

    @staticmethod
    def _entry(path: Path) -> FileEntry:
        is_symlink = path.is_symlink()
        stat = path.lstat()
        if is_symlink:
            kind = "symlink"
        elif path.is_dir():
            kind = "directory"
        elif path.is_file():
            kind = "file"
        else:
            kind = "other"
        return FileEntry(
            path=str(path),
            kind=kind,
            size_bytes=stat.st_size,
            modified_at=datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
            is_symlink=is_symlink,
        )

    def stat(self, path_value: str | Path) -> FileEntry:
        return self._entry(self._resolve_allowed(path_value))

    def list_directory(
        self,
        path_value: str | Path,
        *,
        limit: int | None = None,
    ) -> list[FileEntry]:
        directory = self._resolve_allowed(path_value)
        if not directory.is_dir():
            raise NotADirectoryError(str(directory))

        effective_limit = self._bounded_result_limit(limit)
        children = sorted(directory.iterdir(), key=self._directory_sort_key)
        return [self._entry(path) for path in children[:effective_limit]]

    def read_text(self, path_value: str | Path) -> str:
        path = self._resolve_allowed(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"not a regular file: {path}")
        size = path.stat().st_size
        if size > self._max_read_bytes:
            raise ReadOnlyFileLimitExceeded(
                f"file is {size} bytes; limit is {self._max_read_bytes} bytes"
            )

        data = path.read_bytes()
        if b"\x00" in data and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
            raise UnsupportedTextFile("file appears to be binary")

        encodings = ("utf-8-sig", "utf-16") if data.startswith(
            (b"\xff\xfe", b"\xfe\xff")
        ) else ("utf-8-sig",)
        for encoding in encodings:
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise UnsupportedTextFile("file is not valid UTF-8 or BOM-marked UTF-16 text")

    def find_by_name(
        self,
        query: str,
        *,
        within: str | Path,
        limit: int | None = None,
    ) -> list[FileEntry]:
        needle = query.strip().casefold()
        if not needle:
            raise ValueError("query must not be blank")
        directory = self._resolve_allowed(within)
        if not directory.is_dir():
            raise NotADirectoryError(str(directory))

        effective_limit = self._bounded_result_limit(limit)
        scanned = 0
        matches: list[FileEntry] = []
        for current_root, directory_names, file_names in os.walk(
            directory, followlinks=False
        ):
            directory_names.sort(key=str.casefold)
            file_names.sort(key=str.casefold)
            names = [*directory_names, *file_names]
            for name in names:
                scanned += 1
                if scanned > self._max_scan_entries:
                    raise ReadOnlyFileLimitExceeded(
                        "directory scan exceeded the configured entry limit"
                    )
                if needle not in name.casefold():
                    continue
                path = Path(current_root) / name
                resolved = self._resolve_allowed(path)
                matches.append(self._entry(resolved))
                if len(matches) >= effective_limit:
                    return matches
        return matches

    def _bounded_result_limit(self, requested: int | None) -> int:
        if requested is None:
            return self._max_directory_entries
        if requested < 1:
            raise ValueError("limit must be positive")
        return min(requested, self._max_directory_entries)

    @staticmethod
    def _directory_sort_key(path: Path) -> tuple[bool, str]:
        is_plain_directory = not path.is_symlink() and path.is_dir()
        return (not is_plain_directory, path.name.casefold())
