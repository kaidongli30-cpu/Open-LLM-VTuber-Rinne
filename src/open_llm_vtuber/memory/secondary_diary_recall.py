"""Stateful second-pass diary recall seeded by child-event candidates.

This module is deliberately independent from the production conversation path.
It turns a few already-ranked child events into bounded diary windows, searches
only those windows, and carries up to two user hints across later attempts.
The later RRF/reranker stage remains responsible for deciding whether any
candidate is reliable enough to answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Mapping, Sequence

from .agent_types import MemoryCandidate
from .child_event_search import ChildEventRecord
from .read_only_tools import ReadOnlyMemoryTools


_TERMINAL_STATES = {"resolved", "no_match", "cancelled"}


def _source_diary_date(record: ChildEventRecord) -> date:
    filename = record.source_diary.replace("\\", "/").rsplit("/", 1)[-1]
    prefix = "diary_"
    suffix = ".txt"
    if filename.startswith(prefix) and filename.endswith(suffix):
        try:
            return date.fromisoformat(filename[len(prefix) : -len(suffix)])
        except ValueError:
            pass
    return date.fromisoformat(record.date)


def select_seed_child_events(
    ranked_candidate_ids: Sequence[str],
    records: Mapping[str, ChildEventRecord],
    *,
    limit: int = 3,
) -> tuple[str, ...]:
    """Keep the first distinct, existing child events supplied by a ranker."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    selected: list[str] = []
    seen: set[str] = set()
    for candidate_id in ranked_candidate_ids:
        if candidate_id in seen or candidate_id not in records:
            continue
        seen.add(candidate_id)
        selected.append(candidate_id)
        if len(selected) >= limit:
            break
    return tuple(selected)


def diary_windows_from_child_events(
    candidate_ids: Sequence[str],
    records: Mapping[str, ChildEventRecord],
    *,
    radius_days: int = 1,
) -> tuple[tuple[str, str], ...]:
    """Expand source diary dates and merge overlapping/adjacent windows."""

    if radius_days < 0:
        raise ValueError("radius_days cannot be negative")
    radius = timedelta(days=radius_days)
    intervals = sorted(
        (
            _source_diary_date(records[item]) - radius,
            _source_diary_date(records[item]) + radius,
        )
        for item in candidate_ids
        if item in records
    )
    merged: list[list[date]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + timedelta(days=1):
            merged.append([start, end])
            continue
        if end > merged[-1][1]:
            merged[-1][1] = end
    return tuple((start.isoformat(), end.isoformat()) for start, end in merged)


@dataclass
class SecondaryDiaryRecallSession:
    original_question: str
    seed_child_event_ids: tuple[str, ...]
    diary_windows: tuple[tuple[str, str], ...]
    max_user_hint_rounds: int = 2
    user_hints: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    candidate_pool: dict[str, MemoryCandidate] = field(default_factory=dict)
    state: str = "ready_for_automatic_search"

    @property
    def hint_rounds_used(self) -> int:
        return len(self.user_hints)

    @property
    def hints_remaining(self) -> int:
        return max(0, self.max_user_hint_rounds - self.hint_rounds_used)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_question": self.original_question,
            "seed_child_event_ids": list(self.seed_child_event_ids),
            "diary_windows": [list(item) for item in self.diary_windows],
            "max_user_hint_rounds": self.max_user_hint_rounds,
            "user_hints": list(self.user_hints),
            "hint_rounds_used": self.hint_rounds_used,
            "hints_remaining": self.hints_remaining,
            "state": self.state,
            "candidate_pool": [
                item.to_dict() for item in self.candidate_pool.values()
            ],
            "attempts": list(self.attempts),
        }


class SecondaryDiaryRecallEngine:
    """Search candidate-linked diaries while keeping clarification state."""

    def __init__(
        self,
        diary_tools: ReadOnlyMemoryTools,
        child_event_records: Mapping[str, ChildEventRecord],
        *,
        seed_limit: int = 3,
        window_radius_days: int = 1,
        max_user_hint_rounds: int = 2,
        top_k: int = 10,
        embedding_model: str = "BAAI/bge-base-zh-v1.5",
        embedding_device: str = "cpu",
    ) -> None:
        if seed_limit <= 0:
            raise ValueError("seed_limit must be positive")
        if max_user_hint_rounds < 0:
            raise ValueError("max_user_hint_rounds cannot be negative")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        self.diary_tools = diary_tools
        self.child_event_records = child_event_records
        self.seed_limit = seed_limit
        self.window_radius_days = window_radius_days
        self.max_user_hint_rounds = max_user_hint_rounds
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device

    def start_session(
        self,
        original_question: str,
        ranked_child_event_ids: Sequence[str],
    ) -> SecondaryDiaryRecallSession:
        question = " ".join(original_question.split())
        if not question:
            raise ValueError("original_question must not be empty")
        selected = select_seed_child_events(
            ranked_child_event_ids,
            self.child_event_records,
            limit=self.seed_limit,
        )
        if not selected:
            raise ValueError("no usable ranked child-event candidates")
        windows = diary_windows_from_child_events(
            selected,
            self.child_event_records,
            radius_days=self.window_radius_days,
        )
        return SecondaryDiaryRecallSession(
            original_question=question,
            seed_child_event_ids=selected,
            diary_windows=windows,
            max_user_hint_rounds=self.max_user_hint_rounds,
        )

    @staticmethod
    def _queries(session: SecondaryDiaryRecallSession) -> tuple[str, str]:
        if not session.user_hints:
            return session.original_question, session.original_question
        hint_text = "；".join(session.user_hints)
        # The user's own hint is kept verbatim for Keyword.  Semantic receives
        # the original request and every hint, so it cannot mistake the hint for
        # a new, unrelated question.
        return hint_text, f"原问题：{session.original_question}\n补充提示：{hint_text}"

    @staticmethod
    def _add_candidates(
        session: SecondaryDiaryRecallSession,
        candidates: Sequence[MemoryCandidate],
    ) -> None:
        for candidate in candidates:
            previous = session.candidate_pool.get(candidate.candidate_id)
            if previous is None or candidate.score > previous.score:
                session.candidate_pool[candidate.candidate_id] = candidate

    def _with_bounded_context(
        self,
        candidates: Sequence[MemoryCandidate],
    ) -> list[MemoryCandidate]:
        opener = getattr(self.diary_tools, "get_candidate_with_context", None)
        if not callable(opener):
            return list(candidates)
        return [
            opener(candidate.candidate_id, score=candidate.score)
            for candidate in candidates
        ]

    def _expand_reached_diaries(
        self,
        session: SecondaryDiaryRecallSession,
        candidates: Sequence[MemoryCandidate],
    ) -> tuple[list[str], list[str]]:
        """Expose all chunks of already-reached diary files to the later ranker."""

        getter = getattr(self.diary_tools, "get_source_file_candidates", None)
        if not callable(getter):
            return [], []
        source_files = list(
            dict.fromkeys(
                candidate.source_file
                for candidate in candidates
                if candidate.source_kind == "diary"
            )
        )
        expanded: list[MemoryCandidate] = []
        for source_file in source_files:
            expanded.extend(getter("diary", source_file))
        self._add_candidates(session, expanded)
        return source_files, [item.candidate_id for item in expanded]

    def search(self, session: SecondaryDiaryRecallSession) -> dict[str, Any]:
        if session.state in _TERMINAL_STATES:
            raise RuntimeError(f"session is already {session.state}")
        if not session.diary_windows:
            raise RuntimeError("session has no authorized diary window")
        keyword_query, semantic_query = self._queries(session)
        rankings: list[dict[str, Any]] = []
        for start_date, end_date in session.diary_windows:
            keyword_results = self.diary_tools.search_memory(
                keyword_query,
                ["diary"],
                top_k=self.top_k,
                dedupe_sources=False,
                start_date=start_date,
                end_date=end_date,
            )
            keyword_results = self._with_bounded_context(keyword_results)
            semantic_results = self.diary_tools.search_semantic_memory(
                semantic_query,
                ["diary"],
                top_k=self.top_k,
                dedupe_sources=False,
                model_name=self.embedding_model,
                device=self.embedding_device,
                start_date=start_date,
                end_date=end_date,
            )
            semantic_results = self._with_bounded_context(semantic_results)
            expanded_source_files, expanded_candidate_ids = (
                self._expand_reached_diaries(
                    session,
                    [*keyword_results, *semantic_results],
                )
            )
            for channel, query, results in (
                ("keyword_bounded_diary", keyword_query, keyword_results),
                ("semantic_bounded_diary", semantic_query, semantic_results),
            ):
                self._add_candidates(session, results)
                rankings.append(
                    {
                        "channel": channel,
                        "query": query,
                        "start_date": start_date,
                        "end_date": end_date,
                        "candidate_ids": [item.candidate_id for item in results],
                        "scores": [round(item.score, 6) for item in results],
                        "expanded_source_files": expanded_source_files,
                        "expanded_candidate_ids": expanded_candidate_ids,
                    }
                )
        attempt = {
            "attempt_number": len(session.attempts) + 1,
            "kind": "automatic" if not session.user_hints else "user_hint",
            "user_hints": list(session.user_hints),
            "rankings": rankings,
            "candidate_pool_size": len(session.candidate_pool),
        }
        session.attempts.append(attempt)
        session.state = "candidates_ready_for_ranker"
        return attempt

    def apply_ranker_decision(
        self,
        session: SecondaryDiaryRecallSession,
        *,
        reliable_match: bool,
    ) -> str:
        """Advance state without letting retrieval workers declare truth."""

        if session.state != "candidates_ready_for_ranker":
            raise RuntimeError("ranker decision requires fresh candidates")
        if reliable_match:
            session.state = "resolved"
        elif session.hints_remaining:
            session.state = "awaiting_user_hint"
        else:
            session.state = "no_match"
        return session.state

    def add_user_hint(
        self,
        session: SecondaryDiaryRecallSession,
        hint: str,
    ) -> None:
        if session.state != "awaiting_user_hint":
            raise RuntimeError("session is not waiting for a user hint")
        normalized = " ".join(hint.split())
        if not normalized:
            raise ValueError("hint must not be empty")
        if not session.hints_remaining:
            session.state = "no_match"
            raise RuntimeError("user hint limit reached")
        session.user_hints.append(normalized)
        session.state = "ready_for_hint_search"

    @staticmethod
    def cancel(session: SecondaryDiaryRecallSession) -> None:
        session.state = "cancelled"


__all__ = [
    "SecondaryDiaryRecallEngine",
    "SecondaryDiaryRecallSession",
    "diary_windows_from_child_events",
    "select_seed_child_events",
]
