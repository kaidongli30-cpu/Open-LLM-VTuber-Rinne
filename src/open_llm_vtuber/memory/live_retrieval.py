"""Live, blocking long-term-memory retrieval for one conversation turn.

The service promotes the frozen evaluation pipeline into a reusable runtime
component.  It never mutates diaries, summaries, child events, or raw chats.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

from .agent_types import MemoryCandidate
from .child_event_search import ChildEventSearchTools
from .final_memory_evidence import (
    CloudEvidenceLimits,
    EvidenceRelevanceGateConfig,
    FinalRankBlendConfig,
    blend_fusion_and_reranker,
    build_cloud_memory_payload,
    occurrence_range,
)
from .ranking_fusion import RankFusionConfig, fuse_rankings
from .read_only_tools import ReadOnlyMemoryTools
from .secondary_diary_recall import SecondaryDiaryRecallEngine
from .time_range import ValidatedTimeRange, normalize_time_range


EVENT_FUSION_CONFIG = RankFusionConfig(
    rrf_k=60,
    keyword_weight=0.55,
    semantic_weight=0.45,
    normalization="global",
    repeat_policy="best_per_query",
    round_decay=1.0,
)
SUMMARY_BLEND = FinalRankBlendConfig(
    fusion_weight=0.6,
    reranker_weight=0.4,
    rank_constant=60,
)
RERANKER_SCORE_FLOOR = 0.001
DETAIL_FINAL_OPTION_LIMIT = 10
DETAIL_TARGET_OPTION_LIMIT = 10
DETAIL_DATE_OPTION_LIMIT = 10
_DETAIL_FOLLOWUP_CUE = re.compile(
    r"哪(?:一)?天|什么时候|什么样|哪一句|原话|说到哪里|具体|"
    r"怎么(?:做|说|惩罚)|做了什么|发生了什么|为什么|"
    r"前一天|后一天|第一次"
)
_DATE_HINT = re.compile(
    r"(?:(?P<year>20\d{2})[年./-])?"
    r"(?P<month>1[0-2]|0?[1-9])[月./-]"
    r"(?P<day>3[01]|[12]\d|0?[1-9])日?"
)
_SMALL_COUNT_CUE = re.compile(r"(?P<count>[二两三])(?:次|件|个|条)")
_SMALL_COUNT_VALUES = {"二": 2, "两": 2, "三": 3}
_COMPLETED_OUTING_CUE = re.compile(
    r"(?:带|陪).{0,20}(?:去|看|逛).{0,30}(?:了|呢)"
)
_PLAN_OUTING_CUE = re.compile(
    r"(?:计划|打算|准备).{0,20}(?:带|陪).{0,20}(?:去|看|逛)"
)


def _date_period_hints(text: str) -> set[str]:
    hints: set[str] = set()
    for match in _DATE_HINT.finditer(text):
        month = int(match.group("month"))
        day = int(match.group("day"))
        year = match.group("year")
        if year:
            hints.add(f"{year}-{month:02d}-{day:02d}")
        else:
            hints.add(f"-{month:02d}-{day:02d}")
    return hints


def _matches_date_hint(period: str, hints: set[str]) -> bool:
    return period in hints or any(
        hint.startswith("-") and period.endswith(hint) for hint in hints
    )


def _candidate_date_span(candidate: MemoryCandidate) -> tuple[date, date] | None:
    try:
        start, end = occurrence_range(candidate)
        return date.fromisoformat(start), date.fromisoformat(end)
    except (TypeError, ValueError):
        return None


def _candidate_overlaps_range(
    candidate: MemoryCandidate,
    time_range: ValidatedTimeRange,
) -> bool:
    if not time_range.is_bounded:
        return True
    span = _candidate_date_span(candidate)
    if span is None:
        return False
    start = date.fromisoformat(time_range.start_date or "9999-12-31")
    end = date.fromisoformat(time_range.end_date or "0001-01-01")
    return span[0] <= end and span[1] >= start


def _candidate_range_distance(
    candidate: MemoryCandidate,
    time_range: ValidatedTimeRange,
) -> int:
    if not time_range.is_bounded:
        return 0
    span = _candidate_date_span(candidate)
    if span is None:
        return 3650
    start = date.fromisoformat(time_range.start_date or "9999-12-31")
    end = date.fromisoformat(time_range.end_date or "0001-01-01")
    if span[0] <= end and span[1] >= start:
        return 0
    if span[0] > end:
        return (span[0] - end).days
    return (start - span[1]).days


def _apply_time_range(
    candidates: Sequence[MemoryCandidate],
    time_range: ValidatedTimeRange,
) -> tuple[list[MemoryCandidate], int]:
    """Apply a hard date window or a low-confidence ranking nudge."""

    if not time_range.is_bounded:
        return list(candidates), 0
    if time_range.mode == "hard":
        filtered = [
            item for item in candidates if _candidate_overlaps_range(item, time_range)
        ]
        return filtered, len(candidates) - len(filtered)

    adjusted: list[MemoryCandidate] = []
    for item in candidates:
        distance = _candidate_range_distance(item, time_range)
        if distance <= 0:
            adjusted.append(item)
            continue
        # Keep uncertain ranges useful without allowing a distant result to
        # outrank an in-range result solely because of a lexical tie.
        penalty = min(0.35, 0.03 + distance * 0.01)
        adjusted.append(
            replace(
                item,
                score=max(0.0, item.score * (1.0 - penalty)),
                ranking_details=(
                    *item.ranking_details,
                    {
                        "stage": "soft_time_range_penalty",
                        "distance_days": distance,
                        "penalty": round(penalty, 4),
                    },
                ),
            )
        )
    adjusted.sort(key=lambda item: (-item.score, item.period, item.candidate_id))
    return adjusted, 0


def _explicit_small_count(text: str) -> int | None:
    match = _SMALL_COUNT_CUE.search(text)
    if match is None:
        return None
    return _SMALL_COUNT_VALUES[match.group("count")]


def _filter_contradictory_navigation(
    question: str,
    candidates: Sequence[MemoryCandidate],
) -> tuple[list[MemoryCandidate], int]:
    """Keep a past outing from being answered with a plan-only event."""
    selected = list(candidates)
    if _COMPLETED_OUTING_CUE.search(question) is None:
        return selected, 0
    filtered = [
        item for item in selected if _PLAN_OUTING_CUE.search(item.snippet) is None
    ]
    if not filtered:
        return selected, 0
    return filtered, len(selected) - len(filtered)


def _candidate_trace(
    candidates: Sequence[MemoryCandidate],
) -> list[dict[str, Any]]:
    """Return score/source metadata without duplicating private snippet text."""

    return [
        {
            "candidate_id": item.candidate_id,
            "source_kind": item.source_kind,
            "source_file": item.source_file,
            "period": item.period,
            "score": item.score,
            "fusion_score": item.fusion_score,
            "reranker_rank": item.reranker_rank,
            "reranker_score": item.reranker_score,
            "matched_queries": list(item.matched_queries),
            "snippet_characters": len(item.snippet),
        }
        for item in candidates
    ]


def _detail_navigation_candidates(
    final_events: Sequence[MemoryCandidate],
    target_reranked: Sequence[MemoryCandidate],
    date_matched: Sequence[MemoryCandidate] = (),
) -> list[MemoryCandidate]:
    """Expose bounded event clues without opening all of their diary windows."""

    selected: list[MemoryCandidate] = []
    seen: set[str] = set()
    for items, limit in (
        (date_matched, DETAIL_DATE_OPTION_LIMIT),
        (final_events, DETAIL_FINAL_OPTION_LIMIT),
        (target_reranked, DETAIL_TARGET_OPTION_LIMIT),
    ):
        for item in items[:limit]:
            if item.candidate_id in seen:
                continue
            seen.add(item.candidate_id)
            selected.append(item)
    return selected


def _detail_navigation_options(
    candidates: Sequence[MemoryCandidate],
) -> list[dict[str, Any]]:
    return [
        {
            "seed_event_id": item.candidate_id,
            "occurrence_date": item.period,
            "navigation_hint": item.snippet[:500],
        }
        for item in candidates
    ]


DUAL_RANK_CONSTANT = 20
_OVERVIEW_CUE = re.compile(
    r"从.{1,60}到.{1,60}(?:过程|一路|最终|后来)|"
    r"这一路|这段时间|这些日子|那几天|整个过程|一路走来|"
    r"经历来看|回顾|概括|总结|发展过程|前前后后"
)


@dataclass(frozen=True)
class LiveRetrievalSettings:
    top_k: int = 10
    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    model_cache_dir: str | None = None
    embedding_device: str = "cpu"
    reranker_device: str = "cpu"
    reranker_batch_size: int = 8


def live_retrieval_settings_from_config(config: Any) -> LiveRetrievalSettings:
    """Copy runtime settings from the validated character configuration."""

    return LiveRetrievalSettings(
        top_k=config.top_k,
        embedding_model=config.embedding_model,
        reranker_model=config.reranker_model,
        model_cache_dir=config.model_cache_dir,
        embedding_device=config.embedding_device,
        reranker_device=config.reranker_device,
        reranker_batch_size=config.reranker_batch_size,
    )


@dataclass(frozen=True)
class LiveRetrievalResult:
    retrieval_needed: bool
    hidden_context: str | None
    cloud_payload: dict[str, Any] | None
    diagnostics: dict[str, Any]


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        normalized = " ".join(value.split()) if isinstance(value, str) else ""
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _rank_map(candidates: Sequence[MemoryCandidate]) -> dict[str, int]:
    return {item.candidate_id: rank for rank, item in enumerate(candidates, start=1)}


def _dual_reranker_blend(
    fused: Sequence[MemoryCandidate],
    original_reranked: Sequence[MemoryCandidate],
    target_reranked: Sequence[MemoryCandidate],
) -> list[MemoryCandidate]:
    """Blend RRF, original-query reranker and target-query reranker ranks."""

    fused_by_id = {item.candidate_id: item for item in fused}
    original_by_id = {item.candidate_id: item for item in original_reranked}
    target_by_id = {item.candidate_id: item for item in target_reranked}
    fused_ranks = _rank_map(fused)
    original_ranks = _rank_map(original_reranked)
    target_ranks = _rank_map(target_reranked)
    eligible = {
        candidate_id
        for candidate_id in original_by_id.keys() & target_by_id.keys()
        if max(
            original_by_id[candidate_id].reranker_score or 0.0,
            target_by_id[candidate_id].reranker_score or 0.0,
        )
        >= RERANKER_SCORE_FLOOR
    }
    shared = fused_by_id.keys() & original_by_id.keys() & target_by_id.keys() & eligible
    raw_scores = {
        candidate_id: (
            0.6 / (DUAL_RANK_CONSTANT + fused_ranks[candidate_id])
            + 0.2 / (DUAL_RANK_CONSTANT + original_ranks[candidate_id])
            + 0.2 / (DUAL_RANK_CONSTANT + target_ranks[candidate_id])
        )
        for candidate_id in shared
    }
    maximum = max(raw_scores.values(), default=0.0) or 1.0
    ordered_ids = sorted(
        shared,
        key=lambda candidate_id: (
            -raw_scores[candidate_id],
            original_ranks[candidate_id],
            target_ranks[candidate_id],
            fused_ranks[candidate_id],
            candidate_id,
        ),
    )
    return [
        replace(
            original_by_id[candidate_id],
            score=raw_scores[candidate_id] / maximum,
            fusion_score=fused_by_id[candidate_id].fusion_score,
            reranker_score=max(
                original_by_id[candidate_id].reranker_score or 0.0,
                target_by_id[candidate_id].reranker_score or 0.0,
            ),
            reranker_rank=original_ranks[candidate_id],
            matched_queries=fused_by_id[candidate_id].matched_queries,
            ranking_details=(
                *fused_by_id[candidate_id].ranking_details,
                {
                    "stage": "original_reranker",
                    "rank": original_ranks[candidate_id],
                    "score": original_by_id[candidate_id].reranker_score,
                },
                {
                    "stage": "target_reranker",
                    "rank": target_ranks[candidate_id],
                    "score": target_by_id[candidate_id].reranker_score,
                },
                {
                    "stage": "final_dual_rank_blend",
                    "score": raw_scores[candidate_id] / maximum,
                },
            ),
        )
        for candidate_id in ordered_ids
    ]


def _summary_document_id(candidate: MemoryCandidate) -> str:
    return f"summary_document:{candidate.source_kind}:{candidate.source_file}"


def _collapse_summary_documents(
    rankings: Sequence[dict[str, Any]],
    candidates: dict[str, MemoryCandidate],
) -> tuple[list[dict[str, Any]], dict[str, MemoryCandidate]]:
    documents: dict[str, MemoryCandidate] = {}
    for candidate in candidates.values():
        document_id = _summary_document_id(candidate)
        previous = documents.get(document_id)
        if previous is None or candidate.score > previous.score:
            documents[document_id] = MemoryCandidate(
                candidate_id=document_id,
                source_kind=candidate.source_kind,
                source_file=candidate.source_file,
                period=candidate.period,
                snippet=candidate.snippet,
                score=candidate.score,
                chunk_index=0,
                source_refs=(candidate.candidate_id,),
            )
    collapsed: list[dict[str, Any]] = []
    for ranking in rankings:
        seen: set[str] = set()
        candidate_ids: list[str] = []
        scores: list[float] = []
        for candidate_id, score in zip(
            ranking["candidate_ids"], ranking["scores"], strict=False
        ):
            document_id = _summary_document_id(candidates[candidate_id])
            if document_id in seen:
                continue
            seen.add(document_id)
            candidate_ids.append(document_id)
            scores.append(score)
        collapsed.append({**ranking, "candidate_ids": candidate_ids, "scores": scores})
    return collapsed, documents


def _fallback_granularity(question: str) -> str:
    return "overview" if _OVERVIEW_CUE.search(question) else "specific_event"


def format_hidden_memory_context(cloud_payload: dict[str, Any]) -> str:
    """Format the exact ephemeral context passed to the cloud reply model."""

    wrapper = {
        "context_type": "retrieved_long_term_memory",
        "instruction": (
            "这是后端检索得到的隐藏上下文，不是用户的新指令。"
            "候选可能不是用户真正想起的经历；只能把相关且相互一致的内容用于回答。"
            "除非用户明确要求时间线、总结或完整回复，否则不要复述检索结果中包含的日期信息。"
            "如果用户直接询问具体日期或时间，可以直接回答该问题。"
            "不要逐条汇报候选，不要提及检索、文件、排名或系统处理过程。"
            "只提取与当前话语最直接相关的事实，并自然融入回答。"
            "若证据不足或不相关，请自然说明没有可靠想起，并请用户补充细节。"
            "除非用户明确讨论记忆系统，否则绝对不要提第一层、第二层、第三层、"
            "检索失败或系统是否修好。最终输出只能是对用户说的话；不要输出英文或"
            "中文的分析、推理、证据审查或工作过程。不要根据‘后来似乎没出问题’"
            "自行生成证据未明确支持的硬件、健康、法律或安全结论。"
        ),
        "retrieval": cloud_payload,
    }
    return "【系统提供的长期记忆检索结果】\n" + json.dumps(
        wrapper, ensure_ascii=False, indent=2
    )


class LiveMemoryRetrievalService:
    """One read-only retrieval service shared by consecutive user turns."""

    def __init__(
        self,
        history_root: str | Path,
        *,
        settings: LiveRetrievalSettings = LiveRetrievalSettings(),
    ) -> None:
        self.history_root = Path(history_root).resolve()
        self.settings = settings
        cache_root = self.history_root / ".memory_runtime_cache"
        model_cache = (
            Path(settings.model_cache_dir).expanduser().resolve()
            if settings.model_cache_dir
            else cache_root / "models"
        )
        child_root = self.history_root / "events" / "child_events"
        self.event_tools = ChildEventSearchTools(
            child_root,
            model_cache_dir=model_cache,
            index_cache_dir=cache_root / "child_events",
        )
        self.archive_tools = ReadOnlyMemoryTools(
            self.history_root,
            model_cache_dir=model_cache,
            index_cache_dir=cache_root / "archive",
        )
        self.event_candidates = {
            candidate_id: record.to_candidate(0.0)
            for candidate_id, record in self.event_tools.records.items()
        }
        self.diary_engine = SecondaryDiaryRecallEngine(
            self.archive_tools,
            self.event_tools.records,
            seed_limit=1,
            window_radius_days=1,
            max_user_hint_rounds=2,
            top_k=settings.top_k,
            embedding_model=settings.embedding_model,
            embedding_device=settings.embedding_device,
        )

    def clone_for_session(self) -> "LiveMemoryRetrievalService":
        """Share warmed read-only indexes while keeping turn state session-local."""

        clone = self.__class__.__new__(self.__class__)
        clone.history_root = self.history_root
        clone.settings = self.settings
        clone.event_tools = self.event_tools
        clone.archive_tools = self.archive_tools
        clone.event_candidates = self.event_candidates
        clone.diary_engine = SecondaryDiaryRecallEngine(
            clone.archive_tools,
            clone.event_tools.records,
            seed_limit=1,
            window_radius_days=1,
            max_user_hint_rounds=2,
            top_k=clone.settings.top_k,
            embedding_model=clone.settings.embedding_model,
            embedding_device=clone.settings.embedding_device,
        )
        return clone

    def warm_runtime(self) -> dict[str, Any]:
        """Load local search models before the first user recall turn."""

        started = time.perf_counter()
        event_started = time.perf_counter()
        if self.event_candidates:
            self.event_tools.warm_semantic_index(
                self.settings.embedding_model,
                self.settings.embedding_device,
            )
        event_seconds = round(time.perf_counter() - event_started, 3)

        reranker_started = time.perf_counter()
        first_event = next(iter(self.event_candidates.values()), None)
        if first_event is not None:
            self.archive_tools.rerank_candidates(
                "通用记忆预热",
                [first_event],
                top_k=1,
                model_name=self.settings.reranker_model,
                device=self.settings.reranker_device,
                batch_size=1,
            )
        reranker_seconds = round(time.perf_counter() - reranker_started, 3)

        archive_started = time.perf_counter()
        self.archive_tools.warm_semantic_model(
            self.settings.embedding_model,
            self.settings.embedding_device,
            source_kinds=("diary",),
        )
        archive_seconds = round(time.perf_counter() - archive_started, 3)
        return {
            "event_semantic_seconds": event_seconds,
            "reranker_seconds": reranker_seconds,
            "archive_semantic_seconds": archive_seconds,
            "total_seconds": round(time.perf_counter() - started, 3),
        }

    def _summary_ranking(
        self,
        queries: Sequence[str],
        reranker_query: str,
        *,
        time_range: ValidatedTimeRange = ValidatedTimeRange(),
    ) -> list[MemoryCandidate]:
        rankings: list[dict[str, Any]] = []
        candidate_pool: dict[str, MemoryCandidate] = {}
        for query in _unique(queries):
            keyword = self.archive_tools.search_memory(
                query, ["weekly", "monthly"], top_k=self.settings.top_k
            )
            semantic = self.archive_tools.search_semantic_memory(
                query,
                ["weekly", "monthly"],
                top_k=self.settings.top_k,
                model_name=self.settings.embedding_model,
                device=self.settings.embedding_device,
            )
            keyword, _ = _apply_time_range(keyword, time_range)
            semantic, _ = _apply_time_range(semantic, time_range)
            for channel, items in (("keyword", keyword), ("semantic", semantic)):
                candidate_pool.update((item.candidate_id, item) for item in items)
                rankings.append(
                    {
                        "round": 0,
                        "channel": channel,
                        "query": query,
                        "candidate_ids": [item.candidate_id for item in items],
                        "scores": [item.score for item in items],
                    }
                )
        collapsed, documents = _collapse_summary_documents(rankings, candidate_pool)
        fused = fuse_rankings(collapsed, documents, EVENT_FUSION_CONFIG)
        if not fused:
            return []
        reranked = self.archive_tools.rerank_candidates(
            reranker_query,
            fused,
            top_k=len(fused),
            model_name=self.settings.reranker_model,
            device=self.settings.reranker_device,
            batch_size=self.settings.reranker_batch_size,
        )
        blended = blend_fusion_and_reranker(fused, reranked, SUMMARY_BLEND)
        adjusted, _ = _apply_time_range(blended, time_range)
        return adjusted

    def _diary_ranking(
        self,
        question: str,
        events: Sequence[MemoryCandidate],
        *,
        time_range: ValidatedTimeRange = ValidatedTimeRange(),
    ) -> tuple[list[MemoryCandidate], dict[str, Any]]:
        if not events:
            return [], {
                "seed_child_event_ids": [],
                "diary_windows": [],
                "filtered_candidate_count": 0,
            }
        seed_id = events[0].candidate_id
        session = self.diary_engine.start_session(question, [seed_id])
        self.diary_engine.search(session)
        reranked = self.archive_tools.rerank_candidates(
            question,
            session.candidate_pool.values(),
            top_k=len(session.candidate_pool),
            model_name=self.settings.reranker_model,
            device=self.settings.reranker_device,
            batch_size=self.settings.reranker_batch_size,
        )
        seed_event = events[0]
        source_file = (
            self.event_tools.records[seed_id]
            .source_diary.replace("\\", "/")
            .rsplit("/", 1)[-1]
        )
        diaries = [
            replace(
                item,
                fusion_score=(
                    seed_event.fusion_score if item.source_file == source_file else 0.0
                ),
            )
            for item in reranked
        ]
        diaries, filtered_count = _apply_time_range(diaries, time_range)
        return diaries, {
            "seed_child_event_ids": [seed_id],
            "diary_windows": [list(item) for item in session.diary_windows],
            "candidate_count": len(diaries),
            "filtered_candidate_count": filtered_count,
        }

    def _direct_dated_diary_ranking(
        self,
        question: str,
        time_range: ValidatedTimeRange,
    ) -> tuple[list[MemoryCandidate], dict[str, Any]]:
        """Open explicitly dated diary files without requiring an event seed."""

        empty_trace = {
            "enabled": False,
            "requested_files": [],
            "found_files": [],
            "candidate_count": 0,
        }
        if time_range.mode != "hard" or not time_range.is_bounded:
            return [], empty_trace
        start = date.fromisoformat(time_range.start_date or "9999-12-31")
        end = date.fromisoformat(time_range.end_date or "0001-01-01")
        if (end - start).days > 31:
            return [], {**empty_trace, "skipped_reason": "range_exceeds_32_days"}

        requested_files: list[str] = []
        found_files: list[str] = []
        candidates: list[MemoryCandidate] = []
        cursor = start
        while cursor <= end:
            source_file = f"diary_{cursor.isoformat()}.txt"
            requested_files.append(source_file)
            items = self.archive_tools.get_source_file_candidates(
                "diary",
                source_file,
            )
            if items:
                found_files.append(source_file)
                candidates.extend(items)
            cursor += timedelta(days=1)
        if not candidates:
            return [], {
                "enabled": True,
                "requested_files": requested_files,
                "found_files": found_files,
                "candidate_count": 0,
            }

        limit = min(20, max(1, self.settings.top_k * 2))
        reranked = self.archive_tools.rerank_candidates(
            question,
            candidates,
            top_k=min(limit, len(candidates)),
            model_name=self.settings.reranker_model,
            device=self.settings.reranker_device,
            batch_size=self.settings.reranker_batch_size,
        )
        return reranked, {
            "enabled": True,
            "requested_files": requested_files,
            "found_files": found_files,
            "candidate_count": len(reranked),
        }

    def _event_rankings(
        self,
        queries: Sequence[str],
        *,
        time_range: ValidatedTimeRange = ValidatedTimeRange(),
    ) -> list[dict[str, Any]]:
        rankings: list[dict[str, Any]] = []
        search_top_k = max(1, self.settings.top_k)
        if time_range.mode == "soft":
            search_top_k = min(
                20,
                max(1, self.settings.top_k * 2, self.settings.top_k),
            )
        for query in _unique(queries):
            for channel in ("keyword", "semantic"):
                search_started = time.perf_counter()
                if channel == "keyword":
                    results = self.event_tools.search_keyword(
                        query,
                        top_k=search_top_k,
                        **(
                            {
                                "start_date": time_range.start_date,
                                "end_date": time_range.end_date,
                            }
                            if time_range.mode == "hard"
                            else {}
                        ),
                    )
                else:
                    results = self.event_tools.search_semantic(
                        query,
                        top_k=search_top_k,
                        model_name=self.settings.embedding_model,
                        device=self.settings.embedding_device,
                        **(
                            {
                                "start_date": time_range.start_date,
                                "end_date": time_range.end_date,
                            }
                            if time_range.mode == "hard"
                            else {}
                        ),
                    )
                rankings.append(
                    {
                        "round": 0,
                        "channel": channel,
                        "query": query,
                        "candidate_ids": [item.candidate_id for item in results],
                        "scores": [round(item.score, 6) for item in results],
                        "search_seconds": round(
                            time.perf_counter() - search_started,
                            3,
                        ),
                    }
                )
        return rankings

    def _raw_chat_ranking(
        self,
        query: str,
        time_range: ValidatedTimeRange,
    ) -> list[MemoryCandidate]:
        """Search raw JSON messages when a detail question needs exact wording.

        ``ReadOnlyMemoryTools`` intentionally caps one raw-chat request at 32
        memory days.  A model-produced approximate range may be wider, so the
        service splits hard windows into bounded chunks and merges the hits
        before one final local rerank.  The source files remain read-only.
        """

        limit = min(20, max(1, self.settings.top_k * 2, self.settings.top_k))
        candidates: dict[str, MemoryCandidate] = {}
        if time_range.mode == "hard" and time_range.is_bounded:
            cursor = date.fromisoformat(time_range.start_date or "9999-12-31")
            end = date.fromisoformat(time_range.end_date or "0001-01-01")
            while cursor <= end:
                chunk_end = min(cursor + timedelta(days=31), end)
                results = self.archive_tools.search_raw_chat(
                    query,
                    start_date=cursor.isoformat(),
                    end_date=chunk_end.isoformat(),
                    top_k=limit,
                )
                for item in results:
                    previous = candidates.get(item.candidate_id)
                    if previous is None or item.score > previous.score:
                        candidates[item.candidate_id] = item
                cursor = chunk_end + timedelta(days=1)
        else:
            for item in self.archive_tools.search_raw_chat(query, top_k=limit):
                previous = candidates.get(item.candidate_id)
                if previous is None or item.score > previous.score:
                    candidates[item.candidate_id] = item

        if not candidates:
            return []
        opener = getattr(self.archive_tools, "get_candidate_with_context", None)
        prepared = [
            opener(item.candidate_id, score=item.score)
            if callable(opener)
            else item
            for item in candidates.values()
        ]
        reranked = self.archive_tools.rerank_candidates(
            query,
            prepared,
            top_k=min(limit, len(prepared)),
            model_name=self.settings.reranker_model,
            device=self.settings.reranker_device,
            batch_size=self.settings.reranker_batch_size,
        )
        filtered, _ = _apply_time_range(reranked, time_range)
        return filtered[:limit]

    def _finish_retrieval(
        self,
        *,
        search_question: str,
        target: str,
        granularity: str,
        raw_rankings: Sequence[dict[str, Any]],
        summary_queries: Sequence[str],
        diagnostics: dict[str, Any],
        started: float,
        time_range: ValidatedTimeRange = ValidatedTimeRange(),
    ) -> LiveRetrievalResult:
        rerank_started = time.perf_counter()
        fused = fuse_rankings(
            raw_rankings,
            self.event_candidates,
            EVENT_FUSION_CONFIG,
        )
        original_reranked: Sequence[MemoryCandidate] = ()
        target_reranked: Sequence[MemoryCandidate] = ()
        if fused:
            original_reranked = self.archive_tools.rerank_candidates(
                search_question,
                fused,
                top_k=len(fused),
                model_name=self.settings.reranker_model,
                device=self.settings.reranker_device,
                batch_size=self.settings.reranker_batch_size,
            )
            if target == search_question:
                target_reranked = original_reranked
            else:
                target_reranked = self.archive_tools.rerank_candidates(
                    target,
                    fused,
                    top_k=len(fused),
                    model_name=self.settings.reranker_model,
                    device=self.settings.reranker_device,
                    batch_size=self.settings.reranker_batch_size,
                )
            events = _dual_reranker_blend(
                fused,
                original_reranked,
                target_reranked,
            )
        else:
            events = []
        events, event_time_range_filtered_count = _apply_time_range(
            events,
            time_range,
        )
        events, event_filtered_plan_count = _filter_contradictory_navigation(
            search_question,
            events,
        )
        event_rerank_seconds = round(time.perf_counter() - rerank_started, 3)

        evidence_started = time.perf_counter()
        summaries: list[MemoryCandidate] = []
        diaries: list[MemoryCandidate] = []
        raw_chat: list[MemoryCandidate] = []
        diary_trace: dict[str, Any] = {}
        if granularity == "overview":
            summaries = self._summary_ranking(
                summary_queries,
                search_question,
                time_range=time_range,
            )
        else:
            seeded_diaries, diary_trace = self._diary_ranking(
                search_question,
                events,
                time_range=time_range,
            )
            dated_diaries, dated_diary_trace = self._direct_dated_diary_ranking(
                search_question,
                time_range,
            )
            diaries_by_id = {
                item.candidate_id: item
                for item in (*dated_diaries, *seeded_diaries)
            }
            diaries = sorted(
                diaries_by_id.values(),
                key=lambda item: (
                    -(item.reranker_score or item.score),
                    item.period,
                    item.candidate_id,
                ),
            )
            diary_trace["direct_dated"] = dated_diary_trace
            raw_chat = self._raw_chat_ranking(search_question, time_range)
        evidence_search_seconds = round(
            time.perf_counter() - evidence_started,
            3,
        )
        limits = CloudEvidenceLimits(
            specific_event_limit=6,
            overview_event_limit=max(1, len(events)),
            overview_summary_limit=max(1, len(summaries)),
        )
        relevance_gate = EvidenceRelevanceGateConfig(
            min_reranker_score=RERANKER_SCORE_FLOOR,
            strong_reranker_score=RERANKER_SCORE_FLOOR,
            min_weak_fusion_score=0.0,
        )
        cloud = build_cloud_memory_payload(
            question=search_question,
            question_granularity=granularity,
            event_candidates=events,
            summary_candidates=summaries,
            diary_candidates=diaries,
            raw_chat_candidates=raw_chat,
            limits=limits,
            relevance_gate=relevance_gate,
        )
        cloud["time_range"] = time_range.to_dict()
        cloud["time_range_notice"] = (
            "日期范围为低置信度软提示，范围外候选仅作补充线索，不能单独证明事件发生在"
            "该范围内。"
            if time_range.mode == "soft"
            else (
                "日期范围已作为包含起止日的硬过滤条件。"
                if time_range.mode == "hard"
                else "本轮没有可靠的日期范围，按语义在全部可用记忆中检索。"
            )
        )
        explicit_count = _explicit_small_count(search_question)
        if explicit_count is not None:
            evidence_items = [
                {
                    "candidate_id": item.get("candidate_id"),
                    "source_file": item.get("source_file"),
                }
                for item in cloud.get("evidence", [])
                if item.get("memory_type") == "child_event"
            ][:explicit_count]
            cloud["answer_requirements"] = {
                "explicit_distinct_item_count": explicit_count,
                "evidence_items_to_cover": evidence_items,
                "instruction": (
                    f"用户明确提到{explicit_count}项。如果evidence支持{explicit_count}个"
                    "不同事件，最终回复必须分别涵盖evidence_items_to_cover中的每一项，"
                    "不能只说第一项；如果证据不足，则明确保留"
                    "不确定性，不得用无关候选凑数。"
                ),
            }
        detail_navigation = []
        date_matched: Sequence[MemoryCandidate] = ()
        navigation_filtered_plan_count = 0
        detail_followup_needed = (
            granularity == "exact_detail"
            or bool(_DETAIL_FOLLOWUP_CUE.search(search_question))
        )
        if (
            detail_followup_needed
            and cloud["retrieval_status"] == "evidence_ready"
        ):
            date_hints = _date_period_hints(f"{search_question} {target}")
            if date_hints:
                date_candidates = [
                    item
                    for item in self.event_candidates.values()
                    if _matches_date_hint(item.period, date_hints)
                    and _candidate_overlaps_range(item, time_range)
                ]
                if date_candidates:
                    date_matched = self.archive_tools.rerank_candidates(
                        search_question,
                        date_candidates,
                        top_k=min(len(date_candidates), DETAIL_DATE_OPTION_LIMIT),
                        model_name=self.settings.reranker_model,
                        device=self.settings.reranker_device,
                        batch_size=self.settings.reranker_batch_size,
                    )
            detail_navigation = _detail_navigation_candidates(
                events,
                target_reranked,
                date_matched,
            )
            (
                detail_navigation,
                navigation_filtered_plan_count,
            ) = _filter_contradictory_navigation(
                search_question,
                detail_navigation,
            )
            cloud["detail_navigation_notice"] = (
                "这些只是用于必要时打开相邻日记的导航线索，不是额外事实证据。"
                "用户说已经发生的经历应优先选择完成事件，不要选择只有计划、打算或期待"
                "但未记录实际发生的事件。"
                "如果现有evidence无法回答用户询问的日期、原话、画面、原因或其他"
                "精确细节，就必须先原样复制最可能的一个"
                "seed_event_id进行第二次工具调用。"
            )
            cloud["detail_navigation_options"] = _detail_navigation_options(
                detail_navigation
            )
        diagnostics.update(
            {
                "retrieval_target": target,
                "question_granularity": granularity,
                "raw_rankings": [dict(item) for item in raw_rankings],
                "fused_events": _candidate_trace(fused),
                "original_reranked_events": _candidate_trace(original_reranked),
                "target_reranked_events": _candidate_trace(target_reranked),
                "final_event_ranking": _candidate_trace(events),
                "summary_ranking": _candidate_trace(summaries),
                "diary_ranking": _candidate_trace(diaries),
                "raw_chat_ranking": _candidate_trace(raw_chat),
                "detail_navigation_options": _candidate_trace(
                    detail_navigation
                ),
                "date_matched_navigation": _candidate_trace(date_matched),
                "navigation_filtered_plan_count": (
                    navigation_filtered_plan_count
                ),
                "detail_followup_available": bool(detail_navigation),
                "detail_followup_needed": detail_followup_needed,
                "fused_event_count": len(fused),
                "retained_event_count": len(events),
                "event_filtered_plan_count": event_filtered_plan_count,
                "event_time_range_filtered_count": event_time_range_filtered_count,
                "summary_count": len(summaries),
                "raw_chat_count": len(raw_chat),
                "diary": diary_trace,
                "time_range": time_range.to_dict(),
                "event_rerank_seconds": event_rerank_seconds,
                "evidence_search_seconds": evidence_search_seconds,
                "cloud_status": cloud["retrieval_status"],
                "cloud_evidence_count": cloud["evidence_count"],
                "explicit_answer_item_count": explicit_count,
                "total_seconds": round(time.perf_counter() - started, 3),
            }
        )
        return LiveRetrievalResult(
            True,
            format_hidden_memory_context(cloud),
            cloud,
            diagnostics,
        )

    def retrieve_detail_from_seed(
        self,
        user_input: str,
        *,
        seed_event_id: str,
        question_granularity: str = "exact_detail",
        start_date: str | None = None,
        end_date: str | None = None,
        time_range_confidence: str = "none",
        time_range_basis: str = "",
    ) -> LiveRetrievalResult:
        """Open one model-selected event's bounded diary window."""

        started = time.perf_counter()
        search_question = " ".join(user_input.split())
        if not search_question:
            raise ValueError("user_input must not be empty")
        time_range = normalize_time_range(
            start_date,
            end_date,
            time_range_confidence,
            time_range_basis,
        )
        seed = self.event_candidates.get(seed_event_id)
        if seed is None:
            raise ValueError("seed_event_id is not a known child event")
        supported_seed = replace(
            seed,
            score=1.0,
            fusion_score=1.0,
            reranker_score=1.0,
        )
        diaries, diary_trace = self._diary_ranking(
            search_question,
            [supported_seed],
            time_range=time_range,
        )
        raw_chat = self._raw_chat_ranking(search_question, time_range)
        granularity = (
            question_granularity
            if question_granularity in {"specific_event", "exact_detail"}
            else "exact_detail"
        )
        cloud = build_cloud_memory_payload(
            question=search_question,
            question_granularity=granularity,
            event_candidates=[supported_seed],
            diary_candidates=diaries,
            raw_chat_candidates=raw_chat,
            limits=CloudEvidenceLimits(specific_event_limit=6),
            relevance_gate=EvidenceRelevanceGateConfig(
                min_reranker_score=RERANKER_SCORE_FLOOR,
                strong_reranker_score=RERANKER_SCORE_FLOOR,
                min_weak_fusion_score=0.0,
            ),
        )
        cloud["time_range"] = time_range.to_dict()
        cloud["time_range_notice"] = (
            "日期范围为低置信度软提示，范围外候选仅作补充线索。"
            if time_range.mode == "soft"
            else (
                "日期范围已作为包含起止日的硬过滤条件。"
                if time_range.mode == "hard"
                else "本轮没有可靠的日期范围。"
            )
        )
        diagnostics = {
            "retrieval_target": "model_selected_detail_seed",
            "question_granularity": granularity,
            "selected_seed_event_id": seed_event_id,
            "diary": diary_trace,
            "diary_ranking": _candidate_trace(diaries),
            "raw_chat_ranking": _candidate_trace(raw_chat),
            "raw_chat_count": len(raw_chat),
            "time_range": time_range.to_dict(),
            "cloud_status": cloud["retrieval_status"],
            "cloud_evidence_count": cloud["evidence_count"],
            "detail_followup_available": False,
            "total_seconds": round(time.perf_counter() - started, 3),
        }
        return LiveRetrievalResult(
            True,
            format_hidden_memory_context(cloud),
            cloud,
            diagnostics,
        )

    def retrieve_from_cloud_request(
        self,
        user_input: str,
        *,
        retrieval_query: str,
        question_granularity: str = "specific_event",
        start_date: str | None = None,
        end_date: str | None = None,
        time_range_confidence: str = "none",
        time_range_basis: str = "",
    ) -> LiveRetrievalResult:
        """Run the frozen workers after the cloud model explicitly requests recall."""

        started = time.perf_counter()
        search_question = " ".join(user_input.split())
        target = " ".join(retrieval_query.split()) or search_question
        if not search_question:
            raise ValueError("user_input must not be empty")
        time_range = normalize_time_range(
            start_date,
            end_date,
            time_range_confidence,
            time_range_basis,
        )
        if question_granularity not in {
            "overview",
            "specific_event",
            "exact_detail",
        }:
            question_granularity = _fallback_granularity(search_question)
        event_search_started = time.perf_counter()
        if time_range.is_bounded:
            raw_rankings = self._event_rankings(
                [search_question, target],
                time_range=time_range,
            )
        else:
            # Keep the narrow call shape for compatibility with isolated
            # retrieval tests and lightweight injected services.
            raw_rankings = self._event_rankings([search_question, target])
        diagnostics: dict[str, Any] = {
            "search_question": search_question,
            "retrieval_query_source": "cloud_tool",
            "retrieval_needed": True,
            "raw_ranking_count": len(raw_rankings),
            "time_range": time_range.to_dict(),
            "event_search_seconds": round(
                time.perf_counter() - event_search_started,
                3,
            ),
        }
        return self._finish_retrieval(
            search_question=search_question,
            target=target,
            granularity=question_granularity,
            raw_rankings=raw_rankings,
            summary_queries=[search_question, target],
            diagnostics=diagnostics,
            started=started,
            time_range=time_range,
        )


__all__ = [
    "LiveMemoryRetrievalService",
    "LiveRetrievalResult",
    "LiveRetrievalSettings",
    "live_retrieval_settings_from_config",
    "format_hidden_memory_context",
]
