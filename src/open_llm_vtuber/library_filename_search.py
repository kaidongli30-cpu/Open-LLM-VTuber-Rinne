"""Lazy local semantic search over library filenames only."""

from __future__ import annotations

import hashlib
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .file_library import get_library_root, kind_for_filename


DEFAULT_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
DEFAULT_DEVICE = "cpu"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
_COPY_SUFFIX = re.compile(r"\s*\(\d+\)$")
_SEPARATORS = re.compile(r"[_\-]+")


def _searchable_filename(name: str) -> str:
    stem = _COPY_SUFFIX.sub("", Path(name).stem)
    return _SEPARATORS.sub(" ", stem).strip() or Path(name).name


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
            Path(model_cache_dir).expanduser().resolve()
            if model_cache_dir
            else None
        )
        self.device = str(device or DEFAULT_DEVICE)
        self._model_factory = model_factory
        self._model: Any | None = None
        self._signature = ""
        self._candidates: tuple[dict[str, Any], ...] = ()
        self._embeddings: np.ndarray | None = None
        self._lock = threading.RLock()

    def _scan(self, kind: str) -> tuple[dict[str, Any], ...]:
        if kind not in {"all", "document", "image", "video"}:
            raise ValueError("kind must be all, document, image, or video")
        top_by_kind = {
            "document": "documents",
            "image": "images",
            "video": "videos",
        }
        tops = (
            [top_by_kind[kind]]
            if kind in top_by_kind
            else ["documents", "images", "videos"]
        )
        candidates: list[dict[str, Any]] = []
        for top in tops:
            scan_root = self.root / top
            if not scan_root.is_dir():
                continue
            for path in scan_root.rglob("*"):
                if not path.is_file():
                    continue
                relative = path.relative_to(self.root)
                if any(part.startswith(".") for part in relative.parts[:-1]):
                    continue
                detected_kind = kind_for_filename(path.name)
                if detected_kind is None:
                    continue
                candidates.append(
                    {
                        "name": path.name,
                        "relative_path": relative.as_posix(),
                        "kind": detected_kind,
                        "search_text": _searchable_filename(path.name),
                    }
                )
        return tuple(
            sorted(candidates, key=lambda item: item["relative_path"].casefold())
        )

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
        factory = self._model_factory
        if factory is None:
            # This MCP is a private local-file reader.  Never let a filename
            # query become an implicit model or metadata download.
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from sentence_transformers import SentenceTransformer

            factory = SentenceTransformer
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


__all__ = ["FilenameSemanticSearch", "default_filename_search"]
