"""Lazy local semantic search over library filenames only."""

from __future__ import annotations

import hashlib
import os
import re
import threading
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .file_library import get_library_root, kind_for_filename


DEFAULT_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
DEFAULT_DEVICE = "cpu"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
_COPY_SUFFIX = re.compile(r"\s*\(\d+\)$")
_SEPARATORS = re.compile(r"[_\-]+")
_QUERY_SEPARATORS = re.compile(r"[\s,，、;；+＋|｜/&（）()]+")


def scan_library_filenames(
    root: str | Path | None = None, *, kind: str = "all"
) -> tuple[dict[str, Any], ...]:
    """Scan names below the three library folders, without opening files."""
    if kind not in {"all", "document", "image", "video"}:
        raise ValueError("kind must be all, document, image, or video")
    library_root = get_library_root(root)
    candidates = []
    for top in ("documents", "images", "videos"):
        scan_root = library_root / top
        if not scan_root.is_dir():
            continue
        for path in scan_root.rglob("*"):
            if not path.is_file() or library_root not in path.resolve().parents:
                continue
            detected_kind = kind_for_filename(path.name) or "file"
            if kind != "all" and detected_kind != kind:
                continue
            candidates.append(
                {
                    "name": path.name,
                    "relative_path": path.relative_to(library_root).as_posix(),
                    "kind": detected_kind,
                    "search_text": _searchable_filename(path.name),
                }
            )
    return tuple(sorted(candidates, key=lambda item: item["relative_path"].casefold()))


def find_filename_keywords(
    query: str, root: str | Path | None = None, *, kind: str = "all"
) -> list[dict[str, Any]]:
    """Rank complete keyword matches before partial matches using names only."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        raise ValueError("name is required")
    keywords = tuple(
        dict.fromkeys(
            term.casefold()
            for term in _QUERY_SEPARATORS.split(normalized_query)
            if term
        )
    )
    if not keywords:
        raise ValueError("name must contain a keyword")
    matches = []
    for item in scan_library_filenames(root, kind=kind):
        folded_name = item["name"].casefold()
        compact_name = _SEPARATORS.sub("", folded_name)
        matched = [
            term
            for term in keywords
            if term in folded_name
            or (_SEPARATORS.sub("", term) and _SEPARATORS.sub("", term) in compact_name)
        ]
        if not matched:
            continue
        matches.append(
            {
                "name": item["name"],
                "relative_path": item["relative_path"],
                "kind": item["kind"],
                "match_type": (
                    "literal_filename"
                    if normalized_query.casefold() in folded_name
                    else "keyword_filename"
                ),
                "keyword_match_count": len(matched),
                "keyword_count": len(keywords),
                "matched_keywords": matched,
            }
        )
    return sorted(
        matches,
        key=lambda item: (
            -item["keyword_match_count"],
            item["relative_path"].casefold(),
        ),
    )


def _searchable_filename(name: str) -> str:
    stem = _COPY_SUFFIX.sub("", Path(name).stem)
    return _SEPARATORS.sub(" ", stem).strip() or Path(name).name


def browse_filename_records(root=None, *, kind="all", folder=""):
    """Return descendant names in a requested folder without opening contents."""
    normalized = str(folder or "").strip().replace("\\", "/").strip("/")
    parts = Path(normalized).parts
    if ".." in parts or ":" in normalized:
        raise ValueError("folder must be a library-relative folder")
    tops = {"document": "documents", "image": "images", "video": "videos"}
    requested = [tops[kind]] if kind in tops else list(tops.values())
    if parts and parts[0] in tops.values():
        prefixes = [normalized] if parts[0] in requested else []
    else:
        prefixes = [f"{top}/{normalized}".rstrip("/") for top in requested]
    return [
        {key: item[key] for key in ("name", "relative_path", "kind")}
        for item in scan_library_filenames(root, kind=kind)
        if any(item["relative_path"].startswith(prefix + "/") for prefix in prefixes)
    ]


class FilenameSemanticSearch:
    """Keep one in-process BGE index containing names, never file contents."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        model_cache_dir: str | Path | None = None,
        device: str = DEFAULT_DEVICE,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.root = get_library_root(root)
        self.model_name = str(model_name or DEFAULT_MODEL_NAME)
        self.model_cache_dir = (
            Path(model_cache_dir).expanduser().resolve() if model_cache_dir else None
        )
        self.device = str(device or DEFAULT_DEVICE)
        self._model_factory = model_factory
        self._model: Any | None = None
        self._signature = ""
        self._candidates: tuple[dict[str, Any], ...] = ()
        self._embeddings: np.ndarray | None = None
        self._lock = threading.RLock()
        self._pending_lock = threading.Lock()
        self._pending_search: Future | None = None
        self._dependency_error: str | None = None

    def _scan(self, kind: str) -> tuple[dict[str, Any], ...]:
        return scan_library_filenames(self.root, kind=kind)

    def prepare_dependencies(self) -> None:
        """Import native dependencies before the Windows stdio reader starts."""
        if self._model_factory is not None:
            return
        if self._dependency_error is not None:
            raise RuntimeError(self._dependency_error)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            self._dependency_error = (
                f"Local filename dependencies unavailable: {type(exc).__name__}"
            )
            raise
        self._model_factory = SentenceTransformer

    def search_bounded(self, query: str, *, timeout_seconds: float = 20.0, **kwargs):
        """Bound semantic work without blocking subsequent keyword-only calls."""
        with self._pending_lock:
            if self._pending_search is not None and not self._pending_search.done():
                raise RuntimeError("Local filename semantic search is still busy")
            future: Future = Future()
            self._pending_search = future

            def work():
                try:
                    future.set_result(self.search(query, **kwargs))
                except BaseException as exc:
                    future.set_exception(exc)

            threading.Thread(
                target=work, name="library-filename-search", daemon=True
            ).start()
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError as exc:
            raise TimeoutError("Local filename semantic search timed out") from exc

    @staticmethod
    def _fingerprint(candidates: tuple[dict[str, Any], ...]) -> str:
        digest = hashlib.sha256()
        for item in candidates:
            digest.update(item["relative_path"].encode("utf-8"))
            digest.update(b"\0")
            digest.update(item["search_text"].encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        self.prepare_dependencies()
        factory = self._model_factory
        kwargs: dict[str, Any] = {
            "device": self.device,
            "local_files_only": True,
        }
        if self.model_cache_dir is not None:
            kwargs["cache_folder"] = str(self.model_cache_dir)
        self._model = factory(self.model_name, **kwargs)
        return self._model

    def _ensure_index(self, candidates: tuple[dict[str, Any], ...]) -> None:
        signature = self._fingerprint(candidates)
        if signature == self._signature and self._embeddings is not None:
            return
        model = self._load_model()
        if candidates:
            embeddings = model.encode(
                [item["search_text"] for item in candidates],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device,
            )
            self._embeddings = np.asarray(embeddings, dtype=np.float32)
        else:
            self._embeddings = np.empty((0, 0), dtype=np.float32)
        self._candidates = candidates
        self._signature = signature

    def search(
        self,
        query: str,
        *,
        kind: str = "all",
        limit: int = 8,
        min_score: float = 0.45,
    ) -> list[dict[str, Any]]:
        """Return semantically related filenames without opening any file."""

        normalized_query = str(query or "").strip()
        if not normalized_query:
            raise ValueError("query is required")
        if not 1 <= int(limit) <= 50:
            raise ValueError("limit must be between 1 and 50")
        if not -1.0 <= float(min_score) <= 1.0:
            raise ValueError("min_score must be between -1 and 1")

        with self._lock:
            candidates = self._scan(kind)
            if not candidates:
                return []
            self._ensure_index(candidates)
            if self._embeddings is None or self._embeddings.size == 0:
                return []
            model = self._load_model()
            query_embedding = model.encode(
                QUERY_INSTRUCTION + normalized_query,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device,
            )
            scores = self._embeddings @ np.asarray(
                query_embedding,
                dtype=np.float32,
            )
            ranked = sorted(
                (
                    (float(scores[index]), item)
                    for index, item in enumerate(self._candidates)
                    if float(scores[index]) >= float(min_score)
                ),
                key=lambda pair: (-pair[0], pair[1]["relative_path"].casefold()),
            )
            return [
                {
                    "name": item["name"],
                    "relative_path": item["relative_path"],
                    "kind": item["kind"],
                    "match_type": "semantic_filename",
                    "semantic_score": round(score, 4),
                }
                for score, item in ranked[: int(limit)]
            ]


def default_filename_search(
    root: str | Path | None = None,
    *,
    repo_root: str | Path,
) -> FilenameSemanticSearch:
    """Build the lazy production searcher from non-secret environment settings."""

    model_name = os.environ.get("RINNE_LIBRARY_EMBEDDING_MODEL", DEFAULT_MODEL_NAME)
    device = os.environ.get("RINNE_LIBRARY_EMBEDDING_DEVICE", DEFAULT_DEVICE)
    model_cache = os.environ.get("RINNE_LIBRARY_MODEL_CACHE")
    if not model_cache:
        model_cache = str(Path(repo_root).resolve() / "models" / "huggingface")
    return FilenameSemanticSearch(
        root,
        model_name=model_name,
        model_cache_dir=model_cache,
        device=device,
    )


__all__ = [
    "browse_filename_records",
    "FilenameSemanticSearch",
    "default_filename_search",
    "find_filename_keywords",
    "scan_library_filenames",
]
