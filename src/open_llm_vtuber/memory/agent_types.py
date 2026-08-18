"""Data contract shared by the production memory retrieval workers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MemoryCandidate:
    """One read-only memory excerpt returned by a local search tool."""

    candidate_id: str
    source_kind: str
    source_file: str
    period: str
    snippet: str
    score: float
    chunk_index: int
    source_refs: tuple[str, ...] = ()
    keyword_rank: int | None = None
    keyword_score: float | None = None
    semantic_rank: int | None = None
    semantic_score: float | None = None
    reranker_rank: int | None = None
    reranker_score: float | None = None
    fusion_score: float | None = None
    matched_queries: tuple[str, ...] = ()
    ranking_details: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.source_kind not in {
            "monthly",
            "weekly",
            "diary",
            "raw_chat",
            "child_event",
        }:
            raise ValueError(f"unsupported source kind: {self.source_kind}")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("candidate score must be between 0.0 and 1.0")
        if self.chunk_index < 0:
            raise ValueError("chunk_index cannot be negative")
        for rank in (
            self.keyword_rank,
            self.semantic_rank,
            self.reranker_rank,
        ):
            if rank is not None and rank <= 0:
                raise ValueError("retrieval ranks must be positive")
        for score in (
            self.keyword_score,
            self.semantic_score,
            self.reranker_score,
            self.fusion_score,
        ):
            if score is not None and not 0.0 <= score <= 1.0:
                raise ValueError("retrieval scores must be between 0.0 and 1.0")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_refs"] = list(self.source_refs)
        data["matched_queries"] = list(self.matched_queries)
        data["ranking_details"] = [dict(item) for item in self.ranking_details]
        return data
