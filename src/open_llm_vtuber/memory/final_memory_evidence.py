"""Final memory ranking and dated evidence payloads for a cloud reply model."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

from .agent_types import MemoryCandidate


_WEEKLY_FILE = re.compile(
    r"^weekly_(20\d{2}-\d{2}-\d{2})_to_(20\d{2}-\d{2}-\d{2})\.txt$"
)
_MONTHLY_FILE = re.compile(r"^monthly_(20\d{2})-(\d{2})\.txt$")
_EXPLICIT_QUERY_ANCHOR_PATTERNS = (
    re.compile(
        r"(?:名叫|叫做|名为|称为)[《“\"']?"
        r"([A-Za-z0-9\u4e00-\u9fff·._-]{2,24}?)(?=的|[》”\"'，。！？?、]|$)"
    ),
    re.compile(
        r"(?:聊过|谈过|提过|说过|讨论过)(?:一个|一位|一名|关于)?[《“\"']?"
        r"([A-Za-z0-9\u4e00-\u9fff·._-]{2,16}?)(?=被)"
    ),
    re.compile(r"《([^《》\r\n]{2,40})》"),
)


@dataclass(frozen=True)
class FinalRankBlendConfig:
    """Blend RRF and cross-encoder positions without mixing raw score scales."""

    fusion_weight: float = 0.6
    reranker_weight: float = 0.4
    rank_constant: int = 20

    def __post_init__(self) -> None:
        if self.fusion_weight < 0.0 or self.reranker_weight < 0.0:
            raise ValueError("rank weights cannot be negative")
        if self.fusion_weight + self.reranker_weight <= 0.0:
            raise ValueError("at least one rank weight must be positive")
        if self.rank_constant < 0:
            raise ValueError("rank_constant cannot be negative")


@dataclass(frozen=True)
class CloudEvidenceLimits:
    """Bound evidence volume by question granularity and retrieval line."""

    specific_event_limit: int = 6
    overview_event_limit: int = 20
    overview_summary_limit: int = 12
    content_character_limit: int = 500
    diary_content_character_limit: int = 5000

    def __post_init__(self) -> None:
        if min(
            self.specific_event_limit,
            self.overview_event_limit,
            self.overview_summary_limit,
            self.content_character_limit,
            self.diary_content_character_limit,
        ) <= 0:
            raise ValueError("evidence limits must be positive")


@dataclass(frozen=True)
class EvidenceRelevanceGateConfig:
    """Drop cross-encoder results too weak to support a cloud-memory claim."""

    min_reranker_score: float = 0.015
    strong_reranker_score: float = 0.025
    min_weak_fusion_score: float = 0.8

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_reranker_score <= self.strong_reranker_score <= 1.0:
            raise ValueError("reranker thresholds must be ordered within 0 and 1")
        if not 0.0 <= self.min_weak_fusion_score <= 1.0:
            raise ValueError("min_weak_fusion_score must be between 0 and 1")


def filter_low_relevance_candidates(
    candidates: Sequence[MemoryCandidate],
    config: EvidenceRelevanceGateConfig = EvidenceRelevanceGateConfig(),
    *,
    required_anchors: Sequence[str] = (),
) -> list[MemoryCandidate]:
    """Keep only candidates that passed the final cross-encoder relevance gate.

    A missing reranker score is treated as unverified rather than relevant.  The
    gate is deliberately independent of question text, event identity and rank.
    """

    normalized_anchors = tuple(
        _normalize_anchor(item) for item in required_anchors if item.strip()
    )
    retained: list[MemoryCandidate] = []
    for item in candidates:
        if (
            item.reranker_score is None
            or item.reranker_score < config.min_reranker_score
        ):
            continue
        if (
            item.reranker_score < config.strong_reranker_score
            and (
                item.fusion_score is None
                or item.fusion_score < config.min_weak_fusion_score
            )
        ):
            continue
        evidence_text = _normalize_anchor(
            "\n".join((item.candidate_id, item.source_file, item.snippet))
        )
        if normalized_anchors and not all(
            anchor in evidence_text for anchor in normalized_anchors
        ):
            continue
        retained.append(item)
    return retained


def _normalize_anchor(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", value).casefold()


def extract_explicit_query_anchors(question: str) -> tuple[str, ...]:
    """Extract only names or titles that the question explicitly identifies.

    This deliberately avoids general keyword extraction.  It is a final
    anti-hallucination check for named requests, not another retrieval worker.
    """

    anchors: list[str] = []
    for pattern in _EXPLICIT_QUERY_ANCHOR_PATTERNS:
        for match in pattern.finditer(question):
            anchor = " ".join(match.group(1).split()).strip()
            if any(
                marker in anchor
                for marker in ("名叫", "叫做", "名为", "称为")
            ):
                continue
            if len(_normalize_anchor(anchor)) >= 2 and anchor not in anchors:
                anchors.append(anchor)
    return tuple(anchors)


def blend_fusion_and_reranker(
    fused: Sequence[MemoryCandidate],
    reranked: Sequence[MemoryCandidate],
    config: FinalRankBlendConfig = FinalRankBlendConfig(),
) -> list[MemoryCandidate]:
    """Return a deterministic rank blend over the exact candidate intersection."""

    fused_by_id = {item.candidate_id: item for item in fused}
    reranked_by_id = {item.candidate_id: item for item in reranked}
    fused_ranks = {
        item.candidate_id: rank for rank, item in enumerate(fused, start=1)
    }
    reranker_ranks = {
        item.candidate_id: rank for rank, item in enumerate(reranked, start=1)
    }
    shared = fused_by_id.keys() & reranked_by_id.keys()
    raw_scores = {
        candidate_id: (
            config.fusion_weight
            / (config.rank_constant + fused_ranks[candidate_id])
            + config.reranker_weight
            / (config.rank_constant + reranker_ranks[candidate_id])
        )
        for candidate_id in shared
    }
    maximum = max(raw_scores.values(), default=0.0) or 1.0
    ordered_ids = sorted(
        shared,
        key=lambda candidate_id: (
            -raw_scores[candidate_id],
            reranker_ranks[candidate_id],
            fused_ranks[candidate_id],
            candidate_id,
        ),
    )
    return [
        replace(
            reranked_by_id[candidate_id],
            score=raw_scores[candidate_id] / maximum,
            fusion_score=(
                fused_by_id[candidate_id].fusion_score
                if fused_by_id[candidate_id].fusion_score is not None
                else fused_by_id[candidate_id].score
            ),
            reranker_rank=reranker_ranks[candidate_id],
            matched_queries=fused_by_id[candidate_id].matched_queries,
            ranking_details=fused_by_id[candidate_id].ranking_details,
        )
        for candidate_id in ordered_ids
    ]


def occurrence_range(candidate: MemoryCandidate) -> tuple[str, str]:
    """Derive an inclusive occurrence range from a typed memory source."""

    weekly = _WEEKLY_FILE.fullmatch(candidate.source_file)
    if weekly:
        return weekly.group(1), weekly.group(2)
    monthly = _MONTHLY_FILE.fullmatch(candidate.source_file)
    if monthly:
        year = int(monthly.group(1))
        month = int(monthly.group(2))
        last_day = calendar.monthrange(year, month)[1]
        prefix = f"{year:04d}-{month:02d}"
        return f"{prefix}-01", f"{prefix}-{last_day:02d}"
    return candidate.period, candidate.period


def _compact_content(content: str, limit: int) -> str:
    normalized = "\n".join(line.strip() for line in content.splitlines() if line.strip())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(1, limit - 1)].rstrip() + "…"


def _condense_candidates_by_source(
    candidates: Sequence[MemoryCandidate],
    *,
    excerpts_per_source: int = 5,
) -> list[MemoryCandidate]:
    """Represent one source once while retaining diverse parts of long files."""

    grouped: dict[tuple[str, str], list[MemoryCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(
            (candidate.source_kind, candidate.source_file), []
        ).append(candidate)
    condensed: list[MemoryCandidate] = []
    for items in grouped.values():
        plain_items = [
            item
            for item in items
            if not item.snippet.lstrip().startswith("[片段 ")
        ]
        if plain_items:
            selected = list(plain_items)
            window_items = [
                item
                for item in items
                if item.snippet.lstrip().startswith("[片段 ")
            ]
            if window_items:
                selected.append(window_items[0])
            selected.sort(key=lambda item: item.chunk_index)
        else:
            selected = [items[0]]
            remaining = list(items[1:])
            while remaining and len(selected) < excerpts_per_source:
                next_item = max(
                    remaining,
                    key=lambda item: (
                        min(
                            abs(item.chunk_index - chosen.chunk_index)
                            for chosen in selected
                        ),
                        -remaining.index(item),
                    ),
                )
                selected.append(next_item)
                remaining.remove(next_item)
        snippets = list(dict.fromkeys(item.snippet for item in selected))
        condensed.append(
            replace(
                items[0],
                snippet="\n\n[同一来源的另一处片段]\n".join(snippets),
                source_refs=tuple(
                    dict.fromkeys(
                        ref for item in selected for ref in item.source_refs
                    )
                ),
            )
        )
    return condensed


def _evidence_entry(
    candidate: MemoryCandidate,
    *,
    rank: int,
    retrieval_line: str,
    content_limit: int,
) -> dict[str, Any]:
    start, end = occurrence_range(candidate)
    return {
        "rank": rank,
        "retrieval_line": retrieval_line,
        "memory_type": candidate.source_kind,
        "occurrence_start": start,
        "occurrence_end": end,
        "source_file": candidate.source_file,
        "candidate_id": candidate.candidate_id,
        "content": _compact_content(candidate.snippet, content_limit),
        "fusion_score": candidate.fusion_score,
        "reranker_score": candidate.reranker_score,
    }


def build_cloud_memory_payload(
    *,
    question: str = "",
    question_granularity: str,
    event_candidates: Sequence[MemoryCandidate],
    summary_candidates: Sequence[MemoryCandidate] = (),
    diary_candidates: Sequence[MemoryCandidate] = (),
    limits: CloudEvidenceLimits = CloudEvidenceLimits(),
    relevance_gate: EvidenceRelevanceGateConfig = EvidenceRelevanceGateConfig(),
) -> dict[str, Any]:
    """Build bounded, dated context without claiming that retrieval is correct.

    Specific questions prioritize the final event ranking, followed by reached
    diary evidence, and are capped at six entries.  Overview questions
    retain independent event and weekly/monthly lines because one broad request
    may legitimately refer to more than six experiences.
    """

    if question_granularity not in {
        "overview",
        "specific_event",
        "exact_detail",
    }:
        raise ValueError("unsupported question granularity")

    original_candidate_count = (
        len(event_candidates) + len(summary_candidates) + len(diary_candidates)
    )
    required_anchors = extract_explicit_query_anchors(question)
    event_candidates = filter_low_relevance_candidates(
        event_candidates,
        relevance_gate,
        required_anchors=required_anchors,
    )
    summary_candidates = filter_low_relevance_candidates(
        summary_candidates,
        relevance_gate,
        required_anchors=required_anchors,
    )
    diary_candidates = filter_low_relevance_candidates(
        diary_candidates,
        relevance_gate,
        required_anchors=required_anchors,
    )
    retained_candidate_count = (
        len(event_candidates) + len(summary_candidates) + len(diary_candidates)
    )
    entries: list[dict[str, Any]] = []
    if question_granularity == "overview":
        selected_lines: Iterable[tuple[str, Sequence[MemoryCandidate], int]] = (
            ("child_events", event_candidates, limits.overview_event_limit),
            ("weekly_monthly", summary_candidates, limits.overview_summary_limit),
        )
    else:
        combined: list[tuple[str, MemoryCandidate]] = []
        seen_candidates: set[str] = set()
        seen_sources: set[tuple[str, str]] = set()
        condensed_diaries = _condense_candidates_by_source(diary_candidates)
        for line, items in (
            ("child_events", event_candidates),
            ("bounded_diaries", condensed_diaries),
        ):
            for item in items:
                source_key = (item.source_kind, item.source_file)
                if (
                    item.candidate_id in seen_candidates
                    or source_key in seen_sources
                ):
                    continue
                seen_candidates.add(item.candidate_id)
                seen_sources.add(source_key)
                combined.append((line, item))
                if len(combined) >= limits.specific_event_limit:
                    break
            if len(combined) >= limits.specific_event_limit:
                break
        for rank, (line, candidate) in enumerate(combined, start=1):
            entries.append(
                _evidence_entry(
                    candidate,
                    rank=rank,
                    retrieval_line=line,
                    content_limit=(
                        limits.diary_content_character_limit
                        if line == "bounded_diaries"
                        else limits.content_character_limit
                    ),
                )
            )
        selected_lines = ()

    for line, candidates, limit in selected_lines:
        for rank, candidate in enumerate(candidates[:limit], start=1):
            entries.append(
                _evidence_entry(
                    candidate,
                    rank=rank,
                    retrieval_line=line,
                    content_limit=limits.content_character_limit,
                )
            )

    no_match = not entries
    return {
        "retrieval_status": "no_match" if no_match else "evidence_ready",
        "retrieval_notice": (
            "未找到相关记忆。请不要根据常识或无关候选猜测用户经历；"
            "请用户下一轮提供更多细节。"
            if no_match
            else "检索结果可能不是用户所指的经历；只依据证据回答。"
            "若证据不足或互相冲突，请说明没有可靠想起，并请用户补充细节。"
        ),
        "question_granularity": question_granularity,
        "evidence_count": len(entries),
        "relevance_gate": {
            "min_reranker_score": relevance_gate.min_reranker_score,
            "strong_reranker_score": relevance_gate.strong_reranker_score,
            "min_weak_fusion_score": relevance_gate.min_weak_fusion_score,
            "required_anchors": list(required_anchors),
            "input_candidate_count": original_candidate_count,
            "retained_candidate_count": retained_candidate_count,
            "dropped_candidate_count": (
                original_candidate_count - retained_candidate_count
            ),
        },
        "evidence": entries,
    }


__all__ = [
    "CloudEvidenceLimits",
    "EvidenceRelevanceGateConfig",
    "FinalRankBlendConfig",
    "blend_fusion_and_reranker",
    "build_cloud_memory_payload",
    "extract_explicit_query_anchors",
    "filter_low_relevance_candidates",
    "occurrence_range",
]
