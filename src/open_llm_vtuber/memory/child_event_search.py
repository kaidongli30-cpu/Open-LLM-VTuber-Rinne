"""Read-only parsing and retrieval workers for published child-event TXT files."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent_types import MemoryCandidate
from .read_only_tools import _compact, _features


_DATE_DIRECTORY = re.compile(r"^20\d{2}-\d{2}-\d{2}$")
_EVENT_NAME = re.compile(r"^事件名称：(.+?)\s*$", re.MULTILINE)
_EVENT_TIME = re.compile(r"^时间：(20\d{2}年\d{1,2}月\d{1,2}日)\s*$", re.MULTILINE)
_SOURCE_DIARY = re.compile(r"^日记：(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class ChildEventRecord:
    candidate_id: str
    date: str
    filename: str
    title: str
    summary: str
    source_diary: str
    path: Path

    @property
    def search_text(self) -> str:
        return f"{self.filename}\n{self.title}\n{self.summary}"

    def to_candidate(self, score: float) -> MemoryCandidate:
        return MemoryCandidate(
            candidate_id=self.candidate_id,
            source_kind="child_event",
            source_file=self.filename,
            period=self.date,
            snippet=f"事件名称：{self.title}\n时间：{self.date}\n{self.summary}",
            score=max(0.0, min(1.0, score)),
            chunk_index=0,
            source_refs=(self.source_diary,),
        )


def _parse_event(path: Path, date_value: str, root: Path) -> ChildEventRecord:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_relative_to(root):
        raise ValueError(f"child event escapes root: {path}")
    content = resolved.read_text(encoding="utf-8").strip()
    name_match = _EVENT_NAME.search(content)
    source_match = _SOURCE_DIARY.search(content)
    if name_match is None or source_match is None:
        raise ValueError(f"invalid child event fields: {path}")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content)]
    summary = paragraphs[1] if len(paragraphs) >= 2 else ""
    if not summary or summary.startswith("来源："):
        raise ValueError(f"missing child event summary: {path}")
    relative = resolved.relative_to(root).as_posix()
    return ChildEventRecord(
        candidate_id=f"child_event:{relative}",
        date=date_value,
        filename=path.name,
        title=name_match.group(1).strip(),
        summary=summary,
        source_diary=source_match.group(1).strip(),
        path=resolved,
    )


class ChildEventSearchTools:
    """Read-only keyword and local-embedding search over event TXT files."""

    def __init__(
        self,
        child_event_root: str | Path,
        *,
        model_cache_dir: str | Path,
        index_cache_dir: str | Path,
    ) -> None:
        self.root = Path(child_event_root).resolve(strict=True)
        self.model_cache_dir = Path(model_cache_dir).expanduser().resolve()
        self.index_cache_dir = Path(index_cache_dir).expanduser().resolve()
        self.records: dict[str, ChildEventRecord] = {}
        self.scan_warnings: list[str] = []
        self._feature_cache: dict[str, Counter[str]] = {}
        self._semantic_model: Any | None = None
        self._semantic_embeddings: Any | None = None
        self._semantic_ids: tuple[str, ...] = ()
        self._semantic_runtime: dict[str, Any] = {}
        self._scan()

    def _scan(self) -> None:
        for date_dir in sorted(self.root.iterdir()):
            if not date_dir.is_dir() or not _DATE_DIRECTORY.fullmatch(date_dir.name):
                continue
            txt_dir = date_dir / "事件TXT"
            if not txt_dir.is_dir():
                continue
            for path in sorted(txt_dir.glob("*.txt")):
                try:
                    record = _parse_event(path, date_dir.name, self.root)
                except (OSError, UnicodeError, ValueError) as exc:
                    self.scan_warnings.append(str(exc))
                    continue
                self.records[record.candidate_id] = record
                self._feature_cache[record.candidate_id] = _features(
                    record.search_text
                )

    @property
    def candidate_count(self) -> int:
        return len(self.records)

    @property
    def semantic_runtime(self) -> dict[str, Any]:
        return dict(self._semantic_runtime)

    def search_keyword(self, query: str, *, top_k: int = 10) -> list[MemoryCandidate]:
        if not query.strip():
            raise ValueError("query must not be empty")
        query_features = _features(query)
        if not query_features:
            return []
        document_frequency: Counter[str] = Counter()
        for features in self._feature_cache.values():
            document_frequency.update(features.keys())
        document_count = len(self.records)
        weighted_query: dict[str, float] = {}
        for feature, count in query_features.items():
            idf = math.log(
                (document_count + 1) / (document_frequency[feature] + 1)
            ) + 1.0
            weighted_query[feature] = idf * (len(feature) ** 1.35) * min(count, 2)
        denominator = sum(weighted_query.values()) or 1.0
        compact_query = _compact(query)
        ranked: list[tuple[float, ChildEventRecord]] = []
        for candidate_id, record in self.records.items():
            features = self._feature_cache[candidate_id]
            raw_score = 0.0
            long_matches = 0
            for feature, weight in weighted_query.items():
                frequency = features.get(feature, 0)
                if not frequency:
                    continue
                raw_score += weight * (1.0 + math.log(frequency))
                if len(feature) >= 3:
                    long_matches += 1
            score = min(1.0, raw_score / denominator)
            if long_matches >= 2:
                score = min(1.0, score + min(0.12, long_matches * 0.01))
            if len(compact_query) >= 4 and compact_query in _compact(record.search_text):
                score = min(1.0, score + 0.25)
            if score > 0.0:
                ranked.append((score, record))
        ranked.sort(key=lambda item: (-item[0], item[1].date, item[1].filename))
        return [record.to_candidate(score) for score, record in ranked[:top_k]]

    def _fingerprint(self, model_name: str) -> str:
        digest = hashlib.sha256(model_name.encode("utf-8"))
        for candidate_id in sorted(self.records):
            record = self.records[candidate_id]
            digest.update(candidate_id.encode("utf-8"))
            digest.update(record.search_text.encode("utf-8"))
        return digest.hexdigest()

    def warm_semantic_index(self, model_name: str, device: str = "cpu") -> None:
        if self._semantic_model is not None:
            return
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.index_cache_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = self._fingerprint(model_name)
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
        index_path = self.index_cache_dir / f"child_events_{safe_model}.npz"
        metadata_path = self.index_cache_dir / f"child_events_{safe_model}.json"
        ids = tuple(sorted(self.records))
        started = time.perf_counter()
        model = SentenceTransformer(
            model_name,
            cache_folder=str(self.model_cache_dir),
            device=device,
            local_files_only=True,
        )
        model_seconds = time.perf_counter() - started
        embeddings = None
        cache_hit = False
        if index_path.is_file() and metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("fingerprint") == fingerprint and metadata.get("ids") == list(ids):
                    loaded = np.load(index_path)
                    embeddings = np.asarray(loaded["embeddings"], dtype=np.float32)
                    cache_hit = embeddings.shape[0] == len(ids)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                embeddings = None
                cache_hit = False
        build_started = time.perf_counter()
        if embeddings is None:
            texts = [
                "为这个句子生成表示以用于检索相关文章：" + self.records[item].search_text
                for item in ids
            ]
            embeddings = np.asarray(
                model.encode(
                    texts,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    device=device,
                ),
                dtype=np.float32,
            )
            np.savez_compressed(index_path, embeddings=embeddings)
            metadata_path.write_text(
                json.dumps(
                    {"fingerprint": fingerprint, "ids": list(ids)},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        self._semantic_model = model
        self._semantic_embeddings = embeddings
        self._semantic_ids = ids
        self._semantic_runtime = {
            "model_name": model_name,
            "device": device,
            "model_load_seconds": round(model_seconds, 3),
            "index_build_seconds": round(time.perf_counter() - build_started, 3),
            "cache_hit": cache_hit,
            "candidate_count": len(ids),
            "index_path": str(index_path),
        }

    def search_semantic(
        self,
        query: str,
        *,
        top_k: int = 10,
        model_name: str = "BAAI/bge-base-zh-v1.5",
        device: str = "cpu",
    ) -> list[MemoryCandidate]:
        if not query.strip():
            raise ValueError("query must not be empty")
        self.warm_semantic_index(model_name, device)
        if self._semantic_model is None or self._semantic_embeddings is None:
            raise RuntimeError("semantic index is unavailable")
        import numpy as np

        query_embedding = self._semantic_model.encode(
            "为这个句子生成表示以用于检索相关文章：" + query.strip(),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
        )
        scores = self._semantic_embeddings @ np.asarray(
            query_embedding,
            dtype=np.float32,
        )
        ranked = sorted(
            (
                (float(scores[index]), self.records[candidate_id])
                for index, candidate_id in enumerate(self._semantic_ids)
            ),
            key=lambda item: (-item[0], item[1].date, item[1].filename),
        )
        return [
            record.to_candidate(max(0.0, min(1.0, score)))
            for score, record in ranked[:top_k]
        ]
