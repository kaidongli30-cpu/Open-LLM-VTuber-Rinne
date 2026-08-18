"""Generic rank fusion for independently produced memory candidate lists.

This module deliberately knows nothing about evaluation questions or expected
answers.  It combines already-returned keyword and semantic rankings while
preserving enough diagnostics to explain every final position.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .agent_types import MemoryCandidate


@dataclass(frozen=True)
class RankFusionConfig:
    """Controls reciprocal-rank fusion without inspecting candidate content."""

    rrf_k: int = 60
    keyword_weight: float = 0.45
    semantic_weight: float = 0.55
    normalization: str = "channel"
    repeat_policy: str = "sum"
    round_decay: float = 1.0

    def __post_init__(self) -> None:
        if self.rrf_k < 0:
            raise ValueError("rrf_k cannot be negative")
        if self.keyword_weight < 0.0 or self.semantic_weight < 0.0:
            raise ValueError("channel weights cannot be negative")
        if self.keyword_weight + self.semantic_weight <= 0.0:
            raise ValueError("at least one channel weight must be positive")
        if self.normalization not in {"none", "global", "channel", "round"}:
            raise ValueError("unsupported normalization")
        if self.repeat_policy not in {"sum", "best_per_channel", "best_per_query"}:
            raise ValueError("unsupported repeat policy")
        if not 0.0 < self.round_decay <= 1.0:
            raise ValueError("round_decay must be in (0, 1]")


def fuse_rankings(
    rankings: Sequence[Mapping[str, Any]],
    candidates_by_id: Mapping[str, MemoryCandidate],
    config: RankFusionConfig = RankFusionConfig(),
) -> list[MemoryCandidate]:
    """Fuse keyword/semantic rankings into one transparent candidate list.

    Unknown candidate IDs and malformed list entries are ignored.  The input
    candidates remain immutable; returned candidates carry normalized fusion
    scores, matched queries, raw channel ranks, and per-list contributions.
    """

    valid_lists: list[dict[str, Any]] = []
    for list_index, raw in enumerate(rankings):
        channel = raw.get("channel")
        if channel not in {"keyword", "semantic"}:
            continue
        candidate_ids = raw.get("candidate_ids")
        if not isinstance(candidate_ids, Sequence) or isinstance(
            candidate_ids, (str, bytes)
        ):
            continue
        scores = raw.get("scores")
        if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)):
            scores = ()
        try:
            round_number = max(0, int(raw.get("round", 0)))
        except (TypeError, ValueError):
            round_number = 0
        query = raw.get("query")
        query = " ".join(query.split()) if isinstance(query, str) else ""
        valid_lists.append(
            {
                "list_index": list_index,
                "channel": channel,
                "round": round_number,
                "query": query,
                "candidate_ids": tuple(candidate_ids),
                "scores": tuple(scores),
            }
        )

    if not valid_lists:
        return []

    channel_counts = Counter(item["channel"] for item in valid_lists)
    round_channel_counts = Counter(
        (item["round"], item["channel"]) for item in valid_lists
    )
    round_decay_total = sum(
        config.round_decay**round_number
        for round_number in {item["round"] for item in valid_lists}
    )
    channel_weights = {
        "keyword": config.keyword_weight,
        "semantic": config.semantic_weight,
    }

    details_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    best_rank_by_id: dict[str, dict[str, int]] = defaultdict(dict)
    best_score_by_id: dict[str, dict[str, float]] = defaultdict(dict)
    queries_by_id: dict[str, list[str]] = defaultdict(list)
    for item in valid_lists:
        channel = item["channel"]
        list_weight = channel_weights[channel]
        if config.normalization == "global":
            list_weight /= len(valid_lists)
        elif config.normalization == "channel":
            list_weight /= channel_counts[channel]
        elif config.normalization == "round":
            list_weight /= round_channel_counts[(item["round"], channel)]
            list_weight *= config.round_decay ** item["round"]
            list_weight /= round_decay_total
        elif config.round_decay < 1.0:
            list_weight *= config.round_decay ** item["round"]

        seen_in_list: set[str] = set()
        for position, candidate_id in enumerate(item["candidate_ids"], start=1):
            if not isinstance(candidate_id, str):
                continue
            if candidate_id in seen_in_list or candidate_id not in candidates_by_id:
                continue
            seen_in_list.add(candidate_id)
            raw_score: float | None = None
            if position <= len(item["scores"]):
                try:
                    raw_score = float(item["scores"][position - 1])
                except (TypeError, ValueError):
                    raw_score = None
            contribution = list_weight / (config.rrf_k + position)
            detail = {
                "list_index": item["list_index"],
                "round": item["round"],
                "channel": channel,
                "query": item["query"],
                "rank": position,
                "score": raw_score,
                "rrf_contribution": contribution,
            }
            details_by_id[candidate_id].append(detail)
            previous_rank = best_rank_by_id[candidate_id].get(channel)
            if previous_rank is None or position < previous_rank:
                best_rank_by_id[candidate_id][channel] = position
            if raw_score is not None:
                previous_score = best_score_by_id[candidate_id].get(channel)
                if previous_score is None or raw_score > previous_score:
                    best_score_by_id[candidate_id][channel] = raw_score
            query = item["query"]
            if query and query not in queries_by_id[candidate_id]:
                queries_by_id[candidate_id].append(query)

    raw_fusion_scores: dict[str, float] = {}
    for candidate_id, details in details_by_id.items():
        if config.repeat_policy == "sum":
            score = sum(item["rrf_contribution"] for item in details)
        elif config.repeat_policy == "best_per_channel":
            score = sum(
                max(
                    item["rrf_contribution"]
                    for item in details
                    if item["channel"] == channel
                )
                for channel in {item["channel"] for item in details}
            )
        else:
            by_query: dict[tuple[str, str], float] = {}
            for item in details:
                key = (item["channel"], item["query"])
                by_query[key] = max(
                    by_query.get(key, 0.0), item["rrf_contribution"]
                )
            score = sum(by_query.values())
        raw_fusion_scores[candidate_id] = score

    maximum = max(raw_fusion_scores.values(), default=0.0) or 1.0
    ordered_ids = sorted(
        raw_fusion_scores,
        key=lambda candidate_id: (
            -raw_fusion_scores[candidate_id],
            min(item["rank"] for item in details_by_id[candidate_id]),
            -len(details_by_id[candidate_id]),
            candidate_id,
        ),
    )
    result: list[MemoryCandidate] = []
    for candidate_id in ordered_ids:
        candidate = candidates_by_id[candidate_id]
        channel_ranks = best_rank_by_id[candidate_id]
        channel_scores = best_score_by_id[candidate_id]
        normalized_score = raw_fusion_scores[candidate_id] / maximum
        result.append(
            replace(
                candidate,
                score=normalized_score,
                keyword_rank=channel_ranks.get("keyword"),
                keyword_score=channel_scores.get("keyword"),
                semantic_rank=channel_ranks.get("semantic"),
                semantic_score=channel_scores.get("semantic"),
                fusion_score=normalized_score,
                matched_queries=tuple(queries_by_id[candidate_id]),
                ranking_details=tuple(details_by_id[candidate_id]),
            )
        )
    return result
