from __future__ import annotations

import unittest

from src.open_llm_vtuber.memory.agent_types import MemoryCandidate
from src.open_llm_vtuber.memory.final_memory_evidence import (
    CloudEvidenceLimits,
    EvidenceRelevanceGateConfig,
    FinalRankBlendConfig,
    blend_fusion_and_reranker,
    build_cloud_memory_payload,
    extract_explicit_query_anchors,
    filter_low_relevance_candidates,
    occurrence_range,
)


def candidate(
    candidate_id: str,
    *,
    source_kind: str = "child_event",
    source_file: str | None = None,
    period: str = "2026-01-02",
    score: float = 0.5,
    fusion_score: float | None = None,
) -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=candidate_id,
        source_kind=source_kind,
        source_file=source_file or f"{candidate_id}.txt",
        period=period,
        snippet=f"事件名称：{candidate_id}\n时间：{period}\n中性测试内容",
        score=score,
        chunk_index=0,
        reranker_score=score,
        fusion_score=fusion_score,
    )


class FinalMemoryEvidenceTests(unittest.TestCase):
    def test_rank_blend_keeps_shared_candidates_and_diagnostics(self) -> None:
        first = candidate("first")
        second = candidate("second")
        fused = [first, second]
        reranked = [second, first]

        blended = blend_fusion_and_reranker(
            fused,
            reranked,
            FinalRankBlendConfig(
                fusion_weight=0.7,
                reranker_weight=0.3,
                rank_constant=20,
            ),
        )

        self.assertEqual([item.candidate_id for item in blended], ["first", "second"])
        self.assertEqual(blended[0].reranker_rank, 2)
        self.assertAlmostEqual(blended[0].score, 1.0)

    def test_weekly_and_monthly_sources_have_inclusive_date_ranges(self) -> None:
        weekly = candidate(
            "weekly",
            source_kind="weekly",
            source_file="weekly_2026-02-02_to_2026-02-08.txt",
        )
        monthly = candidate(
            "monthly",
            source_kind="monthly",
            source_file="monthly_2028-02.txt",
        )

        self.assertEqual(occurrence_range(weekly), ("2026-02-02", "2026-02-08"))
        self.assertEqual(occurrence_range(monthly), ("2028-02-01", "2028-02-29"))

    def test_specific_payload_puts_event_before_bounded_diary(self) -> None:
        event = candidate("event")
        diary = candidate(
            "diary",
            source_kind="diary",
            source_file="diary_2026-01-02.txt",
        )

        payload = build_cloud_memory_payload(
            question_granularity="exact_detail",
            event_candidates=[event],
            diary_candidates=[diary],
        )

        self.assertEqual(payload["evidence_count"], 2)
        self.assertEqual(payload["evidence"][0]["retrieval_line"], "child_events")
        self.assertEqual(payload["evidence"][1]["retrieval_line"], "bounded_diaries")
        self.assertEqual(payload["evidence"][1]["occurrence_start"], "2026-01-02")
        self.assertIn("可能不是用户所指", payload["retrieval_notice"])

    def test_specific_payload_allows_only_one_chunk_per_source_file(self) -> None:
        first = candidate(
            "diary-chunk-1",
            source_kind="diary",
            source_file="diary_2026-01-02.txt",
        )
        duplicate = candidate(
            "diary-chunk-2",
            source_kind="diary",
            source_file="diary_2026-01-02.txt",
        )
        next_day = candidate(
            "diary-chunk-3",
            source_kind="diary",
            source_file="diary_2026-01-03.txt",
            period="2026-01-03",
        )

        payload = build_cloud_memory_payload(
            question_granularity="exact_detail",
            event_candidates=[],
            diary_candidates=[first, duplicate, next_day],
        )

        self.assertEqual(payload["evidence_count"], 2)
        self.assertEqual(
            [item["source_file"] for item in payload["evidence"]],
            ["diary_2026-01-02.txt", "diary_2026-01-03.txt"],
        )
        self.assertIn(
            "同一来源的另一处片段", payload["evidence"][0]["content"]
        )

    def test_overview_payload_keeps_independent_bounded_lines(self) -> None:
        events = [candidate(f"event-{index}") for index in range(4)]
        summaries = [
            candidate(
                f"summary-{index}",
                source_kind="weekly",
                source_file=f"weekly_2026-01-0{index + 1}_to_2026-01-0{index + 1}.txt",
            )
            for index in range(3)
        ]

        payload = build_cloud_memory_payload(
            question_granularity="overview",
            event_candidates=events,
            summary_candidates=summaries,
            limits=CloudEvidenceLimits(
                overview_event_limit=3,
                overview_summary_limit=2,
            ),
        )

        self.assertEqual(payload["evidence_count"], 5)
        self.assertEqual(
            [item["retrieval_line"] for item in payload["evidence"]],
            ["child_events"] * 3 + ["weekly_monthly"] * 2,
        )

    def test_payload_content_is_hard_limited(self) -> None:
        long = MemoryCandidate(
            candidate_id="long",
            source_kind="child_event",
            source_file="long.txt",
            period="2026-01-01",
            snippet="字" * 100,
            score=0.5,
            chunk_index=0,
            reranker_score=0.5,
        )

        payload = build_cloud_memory_payload(
            question_granularity="specific_event",
            event_candidates=[long],
            limits=CloudEvidenceLimits(content_character_limit=20),
        )

        self.assertLessEqual(len(payload["evidence"][0]["content"]), 20)

    def test_relevance_gate_drops_unverified_and_extremely_weak_results(self) -> None:
        strong = candidate("strong", score=0.4)
        weak = candidate("weak", score=0.01)
        unverified = MemoryCandidate(
            candidate_id="unverified",
            source_kind="child_event",
            source_file="unverified.txt",
            period="2026-01-02",
            snippet="中性测试内容",
            score=0.9,
            chunk_index=0,
        )

        filtered = filter_low_relevance_candidates(
            [weak, unverified, strong],
            EvidenceRelevanceGateConfig(min_reranker_score=0.015),
        )

        self.assertEqual([item.candidate_id for item in filtered], ["strong"])

    def test_borderline_reranker_score_needs_strong_fusion_support(self) -> None:
        supported = candidate("supported", score=0.02, fusion_score=0.9)
        unsupported = candidate("unsupported", score=0.02, fusion_score=0.7)

        filtered = filter_low_relevance_candidates([unsupported, supported])

        self.assertEqual([item.candidate_id for item in filtered], ["supported"])

    def test_payload_reports_no_match_when_gate_removes_every_candidate(self) -> None:
        payload = build_cloud_memory_payload(
            question_granularity="specific_event",
            event_candidates=[candidate("weak", score=0.01)],
        )

        self.assertEqual(payload["retrieval_status"], "no_match")
        self.assertEqual(payload["evidence_count"], 0)
        self.assertEqual(payload["evidence"], [])
        self.assertIn("未找到相关记忆", payload["retrieval_notice"])
        self.assertEqual(payload["relevance_gate"]["dropped_candidate_count"], 1)

    def test_named_request_requires_the_explicit_name_in_final_evidence(self) -> None:
        named = candidate("named", score=0.4)
        named = MemoryCandidate(
            **{
                **named.__dict__,
                "snippet": "事件名称：星河计划\n这是与该项目直接相关的记录。",
            }
        )
        unrelated = candidate("unrelated", score=0.8)

        payload = build_cloud_memory_payload(
            question="我以前是否提过一个叫做星河计划的项目？",
            question_granularity="specific_event",
            event_candidates=[unrelated, named],
        )

        self.assertEqual(extract_explicit_query_anchors("名叫晨星的设备"), ("晨星",))
        self.assertEqual(payload["relevance_gate"]["required_anchors"], ["星河计划"])
        self.assertEqual(
            [item["candidate_id"] for item in payload["evidence"]], ["named"]
        )

    def test_named_request_reports_no_match_when_name_is_absent(self) -> None:
        payload = build_cloud_memory_payload(
            question="之前是否讨论过远航号被拆除的事情？",
            question_granularity="specific_event",
            event_candidates=[candidate("generic-history", score=0.7)],
        )

        self.assertEqual(payload["retrieval_status"], "no_match")
        self.assertEqual(payload["relevance_gate"]["required_anchors"], ["远航号"])

    def test_ordinary_clause_after_said_is_not_treated_as_a_name(self) -> None:
        self.assertEqual(
            extract_explicit_query_anchors(
                "我以前是否说过我想让设备运行得更稳定？"
            ),
            (),
        )


if __name__ == "__main__":
    unittest.main()
