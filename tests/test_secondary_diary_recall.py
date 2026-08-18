from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from src.open_llm_vtuber.memory.agent_types import MemoryCandidate
from src.open_llm_vtuber.memory.child_event_search import ChildEventRecord
from src.open_llm_vtuber.memory.secondary_diary_recall import (
    SecondaryDiaryRecallEngine,
    diary_windows_from_child_events,
    select_seed_child_events,
)


def event(candidate_id: str, day: str) -> ChildEventRecord:
    return ChildEventRecord(
        candidate_id=candidate_id,
        date=day,
        filename=f"{day}_事件.txt",
        title="候选事件",
        summary="候选概述",
        source_diary=f"chat_history/rinne_01/diaries/diary_{day}.txt",
        path=Path(f"{day}.txt"),
    )


def diary_candidate(candidate_id: str, day: str, score: float) -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=candidate_id,
        source_kind="diary",
        source_file=f"diary_{day}.txt",
        period=day,
        snippet="日记证据",
        score=score,
        chunk_index=0,
    )


class FakeDiaryTools:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search_memory(self, query: str, sources: list[str], **kwargs: Any):
        self.calls.append({"channel": "keyword", "query": query, **kwargs})
        return [diary_candidate("diary:keyword:0", kwargs["start_date"], 0.7)]

    def search_semantic_memory(
        self, query: str, sources: list[str], **kwargs: Any
    ):
        self.calls.append({"channel": "semantic", "query": query, **kwargs})
        return [diary_candidate("diary:semantic:0", kwargs["start_date"], 0.8)]


class SecondaryDiaryRecallTests(unittest.TestCase):
    def test_seed_selection_keeps_first_three_existing_distinct_events(self):
        records = {item: event(item, f"2026-05-{24 + index:02d}") for index, item in enumerate("abcd")}
        self.assertEqual(
            select_seed_child_events(
                ["missing", "a", "a", "b", "c", "d"], records, limit=3
            ),
            ("a", "b", "c"),
        )

    def test_source_dates_expand_one_day_and_merge(self):
        records = {
            "a": event("a", "2026-05-25"),
            "b": event("b", "2026-05-27"),
            "c": event("c", "2026-07-02"),
        }
        self.assertEqual(
            diary_windows_from_child_events(
                ["a", "b", "c"], records, radius_days=1
            ),
            (("2026-05-24", "2026-05-28"), ("2026-07-01", "2026-07-03")),
        )

    def test_hint_search_keeps_original_question_and_verbatim_hint(self):
        records = {"seed": event("seed", "2026-05-27")}
        tools = FakeDiaryTools()
        engine = SecondaryDiaryRecallEngine(
            tools,  # type: ignore[arg-type]
            records,
            max_user_hint_rounds=2,
        )
        session = engine.start_session("我是哪天装上日记系统？", ["seed"])
        engine.search(session)
        self.assertEqual(session.state, "candidates_ready_for_ranker")
        self.assertEqual(
            engine.apply_ranker_decision(session, reliable_match=False),
            "awaiting_user_hint",
        )
        engine.add_user_hint(session, "第二天修过一次逻辑问题")
        engine.search(session)
        keyword_call, semantic_call = tools.calls[-2:]
        self.assertEqual(keyword_call["query"], "第二天修过一次逻辑问题")
        self.assertIn("我是哪天装上日记系统", semantic_call["query"])
        self.assertIn("第二天修过一次逻辑问题", semantic_call["query"])

    def test_at_most_two_user_hint_rounds_then_no_match(self):
        records = {"seed": event("seed", "2026-05-27")}
        engine = SecondaryDiaryRecallEngine(
            FakeDiaryTools(),  # type: ignore[arg-type]
            records,
            max_user_hint_rounds=2,
        )
        session = engine.start_session("回忆过去", ["seed"])
        engine.search(session)
        engine.apply_ranker_decision(session, reliable_match=False)
        engine.add_user_hint(session, "提示一")
        engine.search(session)
        engine.apply_ranker_decision(session, reliable_match=False)
        engine.add_user_hint(session, "提示二")
        engine.search(session)
        self.assertEqual(
            engine.apply_ranker_decision(session, reliable_match=False),
            "no_match",
        )
        self.assertEqual(session.hint_rounds_used, 2)


if __name__ == "__main__":
    unittest.main()
