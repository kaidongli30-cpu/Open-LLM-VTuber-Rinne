from __future__ import annotations

import unittest

from src.open_llm_vtuber.memory.agent_types import MemoryCandidate
from src.open_llm_vtuber.memory.ranking_fusion import (
    RankFusionConfig,
    fuse_rankings,
)


def candidate(candidate_id: str) -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=candidate_id,
        source_kind="child_event",
        source_file=f"{candidate_id}.txt",
        period="2026-01-01",
        snippet=f"事件名称：{candidate_id}",
        score=0.5,
        chunk_index=0,
    )


class RankingFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = {
            item: candidate(item) for item in ("event-a", "event-b", "event-c")
        }
        self.rankings = [
            {
                "round": 0,
                "channel": "keyword",
                "query": "线索甲",
                "candidate_ids": ["event-a", "event-b"],
                "scores": [0.8, 0.7],
            },
            {
                "round": 0,
                "channel": "semantic",
                "query": "经历描述",
                "candidate_ids": ["event-b", "event-c"],
                "scores": [0.9, 0.6],
            },
        ]

    def test_candidate_seen_by_both_workers_rises_to_top(self) -> None:
        fused = fuse_rankings(self.rankings, self.candidates)

        self.assertEqual(fused[0].candidate_id, "event-b")
        self.assertEqual(fused[0].keyword_rank, 2)
        self.assertEqual(fused[0].semantic_rank, 1)
        self.assertEqual(set(fused[0].matched_queries), {"线索甲", "经历描述"})
        self.assertEqual(len(fused[0].ranking_details), 2)

    def test_unknown_and_duplicate_ids_do_not_enter_result(self) -> None:
        rankings = [
            {
                "channel": "keyword",
                "candidate_ids": ["event-a", "missing", "event-a"],
                "scores": [0.9, 0.8, 0.7],
            }
        ]

        fused = fuse_rankings(rankings, self.candidates)

        self.assertEqual([item.candidate_id for item in fused], ["event-a"])

    def test_round_normalization_can_decay_feedback_rounds(self) -> None:
        rankings = self.rankings + [
            {
                "round": 1,
                "channel": "keyword",
                "query": "新增线索",
                "candidate_ids": ["event-c"],
                "scores": [1.0],
            }
        ]
        config = RankFusionConfig(
            rrf_k=0,
            normalization="round",
            round_decay=0.5,
        )

        fused = fuse_rankings(rankings, self.candidates, config)

        event_c = next(item for item in fused if item.candidate_id == "event-c")
        round_one = next(
            item for item in event_c.ranking_details if item["round"] == 1
        )
        self.assertLess(round_one["rrf_contribution"], 0.45)

    def test_best_per_query_ignores_exact_repeated_list(self) -> None:
        duplicate = dict(self.rankings[0])
        config = RankFusionConfig(repeat_policy="best_per_query")

        once = fuse_rankings(self.rankings, self.candidates, config)
        twice = fuse_rankings(self.rankings + [duplicate], self.candidates, config)

        self.assertEqual(
            [item.candidate_id for item in once],
            [item.candidate_id for item in twice],
        )

    def test_invalid_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RankFusionConfig(normalization="unsupported")


if __name__ == "__main__":
    unittest.main()
