"""Read-only tools exposed to the standalone memory-agent prototype."""

from __future__ import annotations

import json
import hashlib
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Iterable

from .agent_types import MemoryCandidate


_MONTHLY_PATTERN = re.compile(r"^monthly_(\d{4}-\d{2})\.txt$")
_WEEKLY_PATTERN = re.compile(
    r"^weekly_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})\.txt$"
)
_DIARY_PATTERN = re.compile(r"^diary_(\d{4}-\d{2}-\d{2})\.txt$")
_ASCII_WORD_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.+-]{1,}")
_LOW_INFORMATION_CHARS = frozenset(
    "我你他她它们的是的了呢吗啊呀吧嘛哈哦嗯"
    "想再去来回一二三几次个些这那"
    "好很真更最也都就还又能会要"
    "把被让在从到和与及而或才可不"
)
_MAX_FILE_BYTES = 2 * 1024 * 1024
_MAX_JSON_FILE_BYTES = 32 * 1024 * 1024
_MAX_CHUNK_CHARS = 700
_RAW_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "ai": "assistant",
    "assistant": "assistant",
}


@dataclass(frozen=True)
class _MemoryChunk:
    candidate_id: str
    source_kind: str
    source_file: str
    period: str
    chunk_index: int
    content: str
    source_refs: tuple[str, ...] = ()


@dataclass
class _RawRecord:
    role: str
    timestamp: datetime
    content: str
    source_refs: list[str]


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def _compact(text: str) -> str:
    normalized = _normalize(text)
    return "".join(
        char
        for char in normalized
        if char.isalnum() or "\u3400" <= char <= "\u9fff"
    )


def _features(text: str) -> Counter[str]:
    normalized = _normalize(text)
    compact = _compact(normalized)
    result: Counter[str] = Counter(_ASCII_WORD_PATTERN.findall(normalized))
    for size in range(2, 5):
        for index in range(max(0, len(compact) - size + 1)):
            feature = compact[index : index + size]
            if feature and all(char in _LOW_INFORMATION_CHARS for char in feature):
                continue
            result[feature] += 1
    return result


def _split_content(content: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content)]
    chunks: list[str] = []
    for paragraph in paragraphs:
        if not paragraph:
            continue
        if len(paragraph) <= _MAX_CHUNK_CHARS:
            chunks.append(paragraph)
            continue
        start = 0
        while start < len(paragraph):
            end = min(len(paragraph), start + _MAX_CHUNK_CHARS)
            if end < len(paragraph):
                boundary = max(
                    paragraph.rfind(mark, start, end)
                    for mark in ("。", "！", "？", "；", "\n")
                )
                if boundary > start + (_MAX_CHUNK_CHARS // 2):
                    end = boundary + 1
            chunk = paragraph[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end
    return chunks


def _period_for_file(kind: str, filename: str) -> str | None:
    if kind == "diary":
        match = _DIARY_PATTERN.fullmatch(filename)
        if not match:
            return None
        try:
            return date.fromisoformat(match.group(1)).isoformat()
        except ValueError:
            return None
    if kind == "weekly":
        match = _WEEKLY_PATTERN.fullmatch(filename)
        if not match:
            return None
        try:
            start = date.fromisoformat(match.group(1))
            end = date.fromisoformat(match.group(2))
        except ValueError:
            return None
        if start.weekday() != 0 or end - start != timedelta(days=6):
            return None
        return f"{start.isoformat()} to {end.isoformat()}"
    match = _MONTHLY_PATTERN.fullmatch(filename)
    if not match:
        return None
    try:
        datetime.strptime(match.group(1), "%Y-%m")
    except ValueError:
        return None
    return match.group(1)


class ReadOnlyMemoryTools:
    """Bounded read-only index for summaries and archived raw chat."""

    _SOURCE_DIRECTORIES = {
        "monthly": "monthly",
        "weekly": "weekly",
        "diary": "diaries",
    }

    def __init__(
        self,
        history_root: str | Path,
        *,
        model_cache_dir: str | Path | None = None,
        index_cache_dir: str | Path | None = None,
        local_timezone: tzinfo | None = None,
        boundary_hour: int = 3,
    ) -> None:
        if not 0 <= boundary_hour <= 23:
            raise ValueError("boundary_hour must be between 0 and 23")
        self.history_root = Path(history_root).resolve()
        self.model_cache_dir = (
            Path(model_cache_dir).expanduser().resolve()
            if model_cache_dir is not None
            else self.history_root / ".memory_test_cache" / "models"
        )
        self.index_cache_dir = (
            Path(index_cache_dir).expanduser().resolve()
            if index_cache_dir is not None
            else self.history_root / ".memory_test_cache"
        )
        self.local_timezone = local_timezone or datetime.now().astimezone().tzinfo
        if self.local_timezone is None:
            raise ValueError("a local timezone is required")
        self.boundary_hour = boundary_hour
        self._chunks: dict[str, _MemoryChunk] = {}
        self._feature_cache: dict[str, Counter[str]] = {}
        self._semantic_context: dict[str, str] = {}
        self._semantic_model: Any | None = None
        self._semantic_embeddings: Any | None = None
        self._semantic_ids: tuple[str, ...] = ()
        self._semantic_runtime: dict[str, Any] = {}
        self._reranker_model: Any | None = None
        self._reranker_runtime: dict[str, Any] = {}
        self._scan_warnings: list[str] = []
        self._raw_order_by_day: dict[str, list[str]] = defaultdict(list)
        self._raw_positions: dict[str, int] = {}
        self._raw_files_scanned = 0
        self._raw_files_parsed = 0
        self._raw_duplicate_records = 0
        self._scan()

    @property
    def scan_warnings(self) -> tuple[str, ...]:
        return tuple(self._scan_warnings)

    @property
    def candidate_count(self) -> int:
        return len(self._chunks)

    @property
    def summary_candidate_count(self) -> int:
        return sum(
            chunk.source_kind != "raw_chat"
            for chunk in self._chunks.values()
        )

    @property
    def raw_candidate_count(self) -> int:
        return sum(
            chunk.source_kind == "raw_chat"
            for chunk in self._chunks.values()
        )

    @property
    def raw_files_scanned(self) -> int:
        return self._raw_files_scanned

    @property
    def raw_files_parsed(self) -> int:
        return self._raw_files_parsed

    @property
    def raw_duplicate_records(self) -> int:
        return self._raw_duplicate_records

    def warm_semantic_model(
        self,
        model_name: str,
        device: str = "cpu",
        *,
        source_kinds: Iterable[str] = ("diary",),
    ) -> bool:
        """Load the semantic model using one small indexed note, not the archive."""

        allowed_kinds = set(source_kinds)
        candidate_id = next(
            (
                item_id
                for item_id, chunk in sorted(self._chunks.items())
                if chunk.source_kind in allowed_kinds
                and item_id in self._semantic_context
            ),
            None,
        )
        if candidate_id is None:
            return False
        self._ensure_semantic_index(
            model_name,
            device,
            candidate_ids=(candidate_id,),
            cache_scope="runtime_model_warmup",
        )
        return True

    def _scan(self) -> None:
        for kind, directory_name in self._SOURCE_DIRECTORIES.items():
            directory = (self.history_root / directory_name).resolve()
            if not directory.is_relative_to(self.history_root):
                continue
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.txt")):
                self._scan_file(kind, directory, path)
        self._scan_raw_chat()

    def _scan_file(self, kind: str, directory: Path, path: Path) -> None:
        try:
            resolved = path.resolve(strict=True)
            if path.is_symlink() or not resolved.is_relative_to(directory):
                self._scan_warnings.append(f"跳过越界文件：{path.name}")
                return
            period = _period_for_file(kind, path.name)
            if period is None:
                return
            if resolved.stat().st_size > _MAX_FILE_BYTES:
                self._scan_warnings.append(f"跳过过大文件：{path.name}")
                return
            content = resolved.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            self._scan_warnings.append(f"无法读取 {path.name}: {exc}")
            return
        content_chunks = _split_content(content)
        for chunk_index, chunk in enumerate(content_chunks):
            candidate_id = f"{kind}:{path.name}:{chunk_index}"
            memory_chunk = _MemoryChunk(
                candidate_id=candidate_id,
                source_kind=kind,
                source_file=path.name,
                period=period,
                chunk_index=chunk_index,
                content=chunk,
            )
            self._chunks[candidate_id] = memory_chunk
        for chunk_index, _chunk in enumerate(content_chunks):
            candidate_id = f"{kind}:{path.name}:{chunk_index}"
            searchable_context = "\n".join(
                content_chunks[
                    max(0, chunk_index - 2) : chunk_index + 3
                ]
            )
            self._feature_cache[candidate_id] = _features(searchable_context)
            self._semantic_context[candidate_id] = searchable_context

    def _scan_raw_chat(self) -> None:
        paths = list(self.history_root.glob("*.json"))
        archive = (self.history_root / "past_history").resolve()
        if archive.is_relative_to(self.history_root) and archive.is_dir():
            paths.extend(archive.glob("*.json"))
        json_paths = sorted(path for path in paths if path.is_file())
        self._raw_files_scanned = len(json_paths)

        unique: dict[tuple[str, str, str], _RawRecord] = {}
        for path in json_paths:
            self._scan_raw_file(path, unique)

        ordered = sorted(
            unique.values(),
            key=lambda item: (
                item.timestamp.astimezone(timezone.utc),
                item.role,
                item.content,
            ),
        )
        for record in ordered:
            first_ref = record.source_refs[0]
            relative_file, _, raw_index = first_ref.rpartition("#")
            try:
                chunk_index = int(raw_index)
            except ValueError:
                chunk_index = 0
            candidate_id = f"raw_chat:{relative_file}:{chunk_index}"
            suffix = 1
            base_id = candidate_id
            while candidate_id in self._chunks:
                suffix += 1
                candidate_id = f"{base_id}:{suffix}"
            local_timestamp = record.timestamp.astimezone(self.local_timezone)
            period = self._memory_day_for(local_timestamp).isoformat()
            content = (
                f"[{local_timestamp.isoformat()}] "
                f"{record.role}: {record.content}"
            )
            chunk = _MemoryChunk(
                candidate_id=candidate_id,
                source_kind="raw_chat",
                source_file=relative_file,
                period=period,
                chunk_index=chunk_index,
                content=content,
                source_refs=tuple(record.source_refs),
            )
            self._chunks[candidate_id] = chunk
            self._feature_cache[candidate_id] = _features(content)
            self._raw_positions[candidate_id] = len(
                self._raw_order_by_day[period]
            )
            self._raw_order_by_day[period].append(candidate_id)

    def _scan_raw_file(
        self,
        path: Path,
        unique: dict[tuple[str, str, str], _RawRecord],
    ) -> None:
        try:
            resolved = path.resolve(strict=True)
            if path.is_symlink() or not resolved.is_relative_to(self.history_root):
                self._scan_warnings.append(f"跳过越界聊天文件：{path.name}")
                return
            relative = resolved.relative_to(self.history_root).as_posix()
            if resolved.stat().st_size > _MAX_JSON_FILE_BYTES:
                self._scan_warnings.append(f"跳过过大聊天文件：{relative}")
                return
            records = json.loads(resolved.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                raise TypeError("history file must contain a JSON array")
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
            self._scan_warnings.append(f"无法读取聊天文件 {path.name}: {exc}")
            return

        self._raw_files_parsed += 1
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            role = _RAW_ROLE_MAP.get(str(record.get("role", "")).casefold())
            content = record.get("content")
            if role is None or not isinstance(content, str) or not content.strip():
                continue
            try:
                timestamp = self._parse_timestamp(record.get("timestamp"))
            except (TypeError, ValueError, OverflowError):
                continue
            normalized_content = content.strip()
            key = (
                role,
                timestamp.astimezone(timezone.utc).isoformat(),
                normalized_content,
            )
            source_ref = f"{relative}#{index}"
            previous = unique.get(key)
            if previous is not None:
                self._raw_duplicate_records += 1
                previous.source_refs.append(source_ref)
                continue
            unique[key] = _RawRecord(
                role=role,
                timestamp=timestamp,
                content=normalized_content,
                source_refs=[source_ref],
            )

    def _parse_timestamp(self, value: Any) -> datetime:
        if not isinstance(value, str):
            raise TypeError("timestamp must be an ISO string")
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            parsed = parsed.replace(tzinfo=self.local_timezone)
        return parsed.astimezone(self.local_timezone)

    def _memory_day_for(self, timestamp: datetime) -> date:
        local_timestamp = timestamp.astimezone(self.local_timezone)
        if local_timestamp.hour < self.boundary_hour:
            local_timestamp -= timedelta(days=1)
        return local_timestamp.date()

    def search_memory(
        self,
        query: str,
        sources: Iterable[str] | None = None,
        top_k: int = 5,
        *,
        dedupe_sources: bool = True,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[MemoryCandidate]:
        """Search allowed memory sources without modifying any file."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        requested = set(sources or self._SOURCE_DIRECTORIES)
        if not requested or not requested.issubset(self._SOURCE_DIRECTORIES):
            raise ValueError("sources contains an unsupported memory source")
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together")
        start = date.fromisoformat(start_date) if start_date else None
        end = date.fromisoformat(end_date) if end_date else None
        if start is not None and end is not None:
            if end < start:
                raise ValueError("end_date cannot be before start_date")
            if requested != {"diary"}:
                raise ValueError("date-bounded summary search only supports diaries")

        source_order = ["monthly", "weekly", "diary"]
        rankings: dict[str, list[tuple[float, _MemoryChunk]]] = {}
        for source in source_order:
            if source not in requested:
                continue
            eligible = [
                chunk
                for chunk in self._chunks.values()
                if chunk.source_kind == source
                and (
                    start is None
                    or start <= date.fromisoformat(chunk.period) <= end
                )
            ]
            ranked = self._rank_chunks(query, eligible)
            rankings[source] = (
                self._dedupe_by_source_file(ranked)
                if dedupe_sources
                else ranked
            )

        if not any(rankings.values()):
            return []

        selected: list[tuple[float, _MemoryChunk]] = []
        selected_ids: set[str] = set()
        quota = max(1, top_k // max(1, len(rankings)))
        for source in source_order:
            for score, chunk in rankings.get(source, [])[:quota]:
                selected.append((score, chunk))
                selected_ids.add(chunk.candidate_id)

        remaining = sorted(
            (
                item
                for ranking in rankings.values()
                for item in ranking
                if item[1].candidate_id not in selected_ids
            ),
            key=self._score_sort_key,
        )
        for item in remaining:
            if len(selected) >= top_k:
                break
            selected.append(item)
            selected_ids.add(item[1].candidate_id)

        selected.sort(key=self._score_sort_key)
        return [
            self._to_candidate(chunk, score)
            for score, chunk in selected[:top_k]
        ]

    def search_raw_chat(
        self,
        query: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        top_k: int = 5,
        dedupe_sources: bool = False,
    ) -> list[MemoryCandidate]:
        """Search raw chat inside an optional inclusive memory-day range.

        Raw messages from one JSON file are distinct evidence items.  Callers
        may explicitly request the old one-result-per-file behavior, but the
        diagnostic pipeline keeps every matching message so an answer is not
        lost merely because it shares a file with the user's question.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together")

        start: date | None = None
        end: date | None = None
        if start_date is not None and end_date is not None:
            start = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)
            if end < start:
                raise ValueError("end_date cannot be before start_date")
            if end - start > timedelta(days=31):
                raise ValueError("raw chat date range cannot exceed 32 days")

        eligible = []
        for chunk in self._chunks.values():
            if chunk.source_kind != "raw_chat":
                continue
            if start is not None and end is not None:
                memory_day = date.fromisoformat(chunk.period)
                if not start <= memory_day <= end:
                    continue
            eligible.append(chunk)
        ranked = self._rank_chunks(query, eligible)
        if dedupe_sources:
            ranked = self._dedupe_by_source_file(ranked)
        return [
            self._to_candidate(chunk, score)
            for score, chunk in ranked[:top_k]
        ]

    def search_semantic_memory(
        self,
        query: str,
        sources: Iterable[str] | None = None,
        *,
        top_k: int = 10,
        dedupe_sources: bool = True,
        model_name: str = "BAAI/bge-base-zh-v1.5",
        device: str = "cpu",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[MemoryCandidate]:
        """Search monthly, weekly, and diary chunks with a cached local BGE index."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        requested = set(sources or self._SOURCE_DIRECTORIES)
        if not requested or not requested.issubset(self._SOURCE_DIRECTORIES):
            raise ValueError("sources contains an unsupported memory source")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together")
        start = date.fromisoformat(start_date) if start_date else None
        end = date.fromisoformat(end_date) if end_date else None
        if start is not None and end is not None and end < start:
            raise ValueError("end_date cannot be before start_date")

        eligible_ids = tuple(
            sorted(
                candidate_id
                for candidate_id, chunk in self._chunks.items()
                if candidate_id in self._semantic_context
                and chunk.source_kind in requested
                and (
                    start is None
                    or (
                        chunk.source_kind == "diary"
                        and start <= date.fromisoformat(chunk.period) <= end
                    )
                )
            )
        )
        if not eligible_ids:
            return []
        source_scope = "-".join(sorted(requested))
        date_scope = (
            f"{start.isoformat()}_{end.isoformat()}"
            if start is not None and end is not None
            else "all"
        )
        self._ensure_semantic_index(
            model_name,
            device,
            candidate_ids=eligible_ids,
            cache_scope=f"{source_scope}_{date_scope}",
        )
        if self._semantic_model is None or self._semantic_embeddings is None:
            raise RuntimeError("semantic index is unavailable")

        import numpy as np

        query_started = datetime.now().timestamp()
        query_embedding = self._semantic_model.encode(
            "为这个句子生成表示以用于检索相关文章：" + query.strip(),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self._semantic_runtime["device"],
        )
        scores = self._semantic_embeddings @ np.asarray(
            query_embedding,
            dtype=np.float32,
        )
        ranked: list[tuple[float, _MemoryChunk]] = []
        for index, candidate_id in enumerate(self._semantic_ids):
            chunk = self._chunks[candidate_id]
            ranked.append((float(scores[index]), chunk))
        ranked.sort(key=self._score_sort_key)
        if dedupe_sources:
            ranked = self._dedupe_by_source_file(ranked)
        self._semantic_runtime["last_query_seconds"] = (
            datetime.now().timestamp() - query_started
        )
        return [
            self._to_candidate(chunk, max(0.0, min(1.0, score)))
            for score, chunk in ranked[:top_k]
        ]

    @property
    def semantic_runtime(self) -> dict[str, Any]:
        return dict(self._semantic_runtime)

    @property
    def reranker_runtime(self) -> dict[str, Any]:
        return dict(self._reranker_runtime)

    def rerank_candidates(
        self,
        query: str,
        candidates: Iterable[MemoryCandidate],
        *,
        top_k: int = 20,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        batch_size: int = 4,
    ) -> list[MemoryCandidate]:
        """Rerank an existing candidate set without reading new memories."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(top_k, int) or not 1 <= top_k:
            raise ValueError("top_k must be a positive integer")
        if not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be between 1 and 32")
        items = list(candidates)
        if not items:
            return []
        if top_k > max(20, len(items)):
            raise ValueError(
                "top_k cannot exceed 20 unless the candidate set is larger"
            )

        model, resolved_device, model_seconds = self._load_reranker(
            model_name,
            device,
        )
        import numpy as np

        started = datetime.now().timestamp()
        raw_scores = np.asarray(
            model.predict(
                [(query.strip(), item.snippet) for item in items],
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
            dtype=np.float64,
        ).reshape(-1)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(raw_scores, -60, 60)))
        order = sorted(
            range(len(items)),
            key=lambda index: (
                -float(probabilities[index]),
                items[index].source_file,
                items[index].chunk_index,
            ),
        )
        reranked = [
            replace(
                items[index],
                score=float(probabilities[index]),
                fusion_score=(
                    items[index].fusion_score
                    if items[index].fusion_score is not None
                    else items[index].score
                ),
                reranker_rank=rank,
                reranker_score=float(probabilities[index]),
            )
            for rank, index in enumerate(order[:top_k], start=1)
        ]
        self._reranker_runtime = {
            "model_name": model_name,
            "device": resolved_device,
            "model_cache_dir": str(self.model_cache_dir),
            "model_load_seconds": model_seconds,
            "last_query_seconds": datetime.now().timestamp() - started,
            "candidate_count": len(items),
        }
        return reranked

    def _load_reranker(
        self,
        model_name: str,
        device: str,
    ) -> tuple[Any, str, float]:
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "缺少sentence-transformers或PyTorch，reranker不可用"
            ) from exc

        resolved_device = device
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("当前PyTorch没有可用的CUDA支持")

        if (
            self._reranker_model is not None
            and self._reranker_runtime.get("model_name") == model_name
            and self._reranker_runtime.get("device") == resolved_device
        ):
            return self._reranker_model, resolved_device, 0.0

        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        started = datetime.now().timestamp()
        model = CrossEncoder(
            model_name,
            num_labels=1,
            max_length=512,
            device=resolved_device,
            activation_fn=torch.nn.Identity(),
            cache_folder=str(self.model_cache_dir),
        )
        load_seconds = datetime.now().timestamp() - started
        self._reranker_model = model
        self._reranker_runtime = {
            "model_name": model_name,
            "device": resolved_device,
            "model_cache_dir": str(self.model_cache_dir),
            "model_load_seconds": load_seconds,
        }
        return model, resolved_device, load_seconds

    def _ensure_semantic_index(
        self,
        model_name: str,
        device: str,
        *,
        candidate_ids: tuple[str, ...],
        cache_scope: str,
    ) -> None:
        if self._semantic_model is not None and self._semantic_embeddings is not None:
            if (
                self._semantic_runtime.get("model_name") == model_name
                and self._semantic_runtime.get("cache_scope") == cache_scope
                and self._semantic_ids == candidate_ids
            ):
                return

        try:
            import numpy as np
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "缺少sentence-transformers、numpy或PyTorch，BGE语义检索不可用"
            ) from exc

        resolved_device = device
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("当前PyTorch没有可用的CUDA支持")

        ids = candidate_ids
        if not ids:
            raise RuntimeError("没有可用于BGE检索的摘要记忆")
        digest = hashlib.sha256()
        for candidate_id in ids:
            digest.update(candidate_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(self._semantic_context[candidate_id].encode("utf-8"))
            digest.update(b"\0")
        signature = digest.hexdigest()

        safe_scope = re.sub(r"[^A-Za-z0-9_.-]+", "_", cache_scope)
        index_dir = self.index_cache_dir / "memory_agent_semantic"
        model_dir = self.model_cache_dir
        index_path = index_dir / f"{safe_scope}.npz"
        metadata_path = index_dir / f"{safe_scope}.json"
        index_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_seconds = 0.0
        if (
            self._semantic_model is not None
            and self._semantic_runtime.get("model_name") == model_name
            and self._semantic_runtime.get("device") == resolved_device
        ):
            model = self._semantic_model
        else:
            model_started = datetime.now().timestamp()
            model = SentenceTransformer(
                model_name,
                cache_folder=str(model_dir),
                device=resolved_device,
            )
            model_seconds = datetime.now().timestamp() - model_started

        embeddings = None
        cache_hit = False
        if index_path.is_file() and metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (
                    metadata.get("format_version") == 1
                    and metadata.get("model_name") == model_name
                    and metadata.get("content_sha256") == signature
                    and tuple(metadata.get("candidate_ids", ())) == ids
                ):
                    with np.load(index_path, allow_pickle=False) as loaded:
                        embeddings = np.asarray(
                            loaded["embeddings"],
                            dtype=np.float32,
                        )
                    if embeddings.shape[0] != len(ids):
                        embeddings = None
                    else:
                        cache_hit = True
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                embeddings = None

        build_seconds = 0.0
        if embeddings is None:
            build_started = datetime.now().timestamp()
            embeddings = np.asarray(
                model.encode(
                    [self._semantic_context[item] for item in ids],
                    batch_size=64,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    device=resolved_device,
                ),
                dtype=np.float32,
            )
            build_seconds = datetime.now().timestamp() - build_started
            temporary_index = index_dir / f"{safe_scope}.tmp.npz"
            temporary_metadata = index_dir / f"{safe_scope}.tmp.json"
            np.savez_compressed(temporary_index, embeddings=embeddings)
            temporary_metadata.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "model_name": model_name,
                        "content_sha256": signature,
                        "candidate_ids": list(ids),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            temporary_index.replace(index_path)
            temporary_metadata.replace(metadata_path)

        self._semantic_model = model
        self._semantic_embeddings = embeddings
        self._semantic_ids = ids
        self._semantic_runtime = {
            "model_name": model_name,
            "device": resolved_device,
            "cache_scope": cache_scope,
            "model_load_seconds": model_seconds,
            "index_build_seconds": build_seconds,
            "cache_hit": cache_hit,
            "candidate_count": len(ids),
            "index_path": str(index_path),
        }

    def _rank_chunks(
        self,
        query: str,
        eligible: list[_MemoryChunk],
    ) -> list[tuple[float, _MemoryChunk]]:
        if not eligible:
            return []
        query_features = _features(query)
        if not query_features:
            return []

        document_frequency: Counter[str] = Counter()
        for chunk in eligible:
            document_frequency.update(
                self._feature_cache[chunk.candidate_id].keys()
            )
        document_count = len(eligible)
        weighted_query: dict[str, float] = {}
        for feature, count in query_features.items():
            idf = math.log(
                (document_count + 1) / (document_frequency[feature] + 1)
            ) + 1.0
            weighted_query[feature] = (
                idf * (len(feature) ** 1.35) * min(count, 2)
            )
        denominator = sum(weighted_query.values()) or 1.0
        compact_query = _compact(query)

        scored: list[tuple[float, _MemoryChunk]] = []
        for chunk in eligible:
            chunk_features = self._feature_cache[chunk.candidate_id]
            raw_score = 0.0
            matched_long_features = 0
            for feature, query_weight in weighted_query.items():
                frequency = chunk_features.get(feature, 0)
                if frequency:
                    raw_score += query_weight * (1.0 + math.log(frequency))
                    if len(feature) >= 3:
                        matched_long_features += 1
            score = min(1.0, raw_score / denominator)
            if matched_long_features >= 2:
                score = min(1.0, score + min(0.12, matched_long_features * 0.01))
            if len(compact_query) >= 4 and compact_query in _compact(chunk.content):
                score = min(1.0, score + 0.25)
            if score > 0.0:
                scored.append((score, chunk))
        scored.sort(key=self._score_sort_key)
        return scored

    @staticmethod
    def _dedupe_by_source_file(
        ranked: list[tuple[float, _MemoryChunk]],
    ) -> list[tuple[float, _MemoryChunk]]:
        deduplicated: list[tuple[float, _MemoryChunk]] = []
        seen_files: set[str] = set()
        for item in ranked:
            source_file = item[1].source_file
            if source_file in seen_files:
                continue
            seen_files.add(source_file)
            deduplicated.append(item)
        return deduplicated

    @staticmethod
    def _score_sort_key(
        item: tuple[float, _MemoryChunk],
    ) -> tuple[float, str, str, int]:
        return (
            -item[0],
            item[1].period,
            item[1].source_file,
            item[1].chunk_index,
        )

    def open_memory(self, candidate_id: str) -> dict[str, Any]:
        """Open one candidate plus bounded neighboring context."""

        if not isinstance(candidate_id, str) or candidate_id not in self._chunks:
            raise ValueError("candidate_id is unknown or unavailable")
        chunk = self._chunks[candidate_id]
        context: list[dict[str, Any]] = []
        if chunk.source_kind == "raw_chat":
            day_order = self._raw_order_by_day[chunk.period]
            position = self._raw_positions[candidate_id]
            # A raw hit is often the user's setup/question, while the answer is
            # one or more messages later.  Bias the bounded window forward so a
            # matching setup does not crowd its answer out with older context.
            neighbor_ids = day_order[max(0, position - 1) : position + 5]
        else:
            neighbor_ids = [
                f"{chunk.source_kind}:{chunk.source_file}:{chunk_index}"
                for chunk_index in range(
                    max(0, chunk.chunk_index - 2),
                    chunk.chunk_index + 3,
                )
            ]
        for neighbor_id in neighbor_ids:
            neighbor = self._chunks.get(neighbor_id)
            if neighbor is None:
                continue
            context.append(
                {
                    "chunk_index": neighbor.chunk_index,
                    "is_requested_chunk": neighbor_id == candidate_id,
                    "content": neighbor.content,
                }
            )
        return {
            "candidate_id": chunk.candidate_id,
            "source_kind": chunk.source_kind,
            "source_file": chunk.source_file,
            "period": chunk.period,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "source_refs": list(chunk.source_refs),
            "neighbor_context": context,
        }

    def get_candidate_with_context(
        self,
        candidate_id: str,
        *,
        score: float = 0.0,
    ) -> MemoryCandidate:
        """Return one candidate carrying its bounded opened context."""

        opened = self.open_memory(candidate_id)
        parts = [
            f"[片段 {item['chunk_index']}] {item['content']}"
            for item in opened["neighbor_context"]
        ]
        snippet = "\n".join(parts)
        chunk = self._chunks[candidate_id]
        # Raw-chat evidence commonly consists of a long user message followed by
        # the assistant's answer.  Keep the bounded five-message window intact
        # long enough for that answer to survive the final display/judge cut.
        # This remains a character bound and never expands the date scope.
        context_limit = 2400 if chunk.source_kind == "raw_chat" else 1200
        if len(snippet) > context_limit:
            snippet = f"{snippet[: context_limit - 3]}..."
        return MemoryCandidate(
            candidate_id=chunk.candidate_id,
            source_kind=chunk.source_kind,
            source_file=chunk.source_file,
            period=chunk.period,
            snippet=snippet,
            score=max(0.0, min(1.0, score)),
            chunk_index=chunk.chunk_index,
            source_refs=chunk.source_refs,
        )

    def get_candidate(
        self,
        candidate_id: str,
        *,
        score: float = 0.0,
    ) -> MemoryCandidate:
        if candidate_id not in self._chunks:
            raise ValueError("candidate_id is unknown or unavailable")
        return self._to_candidate(self._chunks[candidate_id], score)

    def get_source_file_candidates(
        self,
        source_kind: str,
        source_file: str,
    ) -> list[MemoryCandidate]:
        """Return every indexed chunk from one already-known source file."""

        if source_kind not in {*self._SOURCE_DIRECTORIES, "raw_chat"}:
            raise ValueError("source_kind is unsupported")
        chunks = sorted(
            (
                chunk
                for chunk in self._chunks.values()
                if chunk.source_kind == source_kind
                and chunk.source_file == source_file
            ),
            key=lambda item: item.chunk_index,
        )
        return [self._to_candidate(chunk, 0.0) for chunk in chunks]

    @staticmethod
    def _to_candidate(
        chunk: _MemoryChunk,
        score: float,
    ) -> MemoryCandidate:
        snippet = " ".join(chunk.content.split())
        if len(snippet) > 420:
            snippet = f"{snippet[:417]}..."
        return MemoryCandidate(
            candidate_id=chunk.candidate_id,
            source_kind=chunk.source_kind,
            source_file=chunk.source_file,
            period=chunk.period,
            snippet=snippet,
            score=max(0.0, min(1.0, score)),
            chunk_index=chunk.chunk_index,
            source_refs=chunk.source_refs,
        )
