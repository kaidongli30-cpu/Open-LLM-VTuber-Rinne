import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.input_types import BatchInput, TextData, TextSource
from src.open_llm_vtuber.config_manager.agent import BasicMemoryAgentConfig
from src.open_llm_vtuber.memory.agent_types import MemoryCandidate
from src.open_llm_vtuber.memory.live_retrieval import (
    LiveMemoryRetrievalService,
    LiveRetrievalSettings,
    _dual_reranker_blend,
    _fallback_granularity,
    format_hidden_memory_context,
    live_retrieval_settings_from_config,
)
from src.open_llm_vtuber.memory.read_only_tools import ReadOnlyMemoryTools
from src.open_llm_vtuber.memory.trial_log import write_memory_trial_record


def candidate(
    candidate_id: str,
    *,
    score: float = 0.5,
    fusion_score: float | None = None,
    reranker_score: float | None = None,
) -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=candidate_id,
        source_kind="child_event",
        source_file=f"{candidate_id}.txt",
        period="2026-01-01",
        snippet=candidate_id,
        score=score,
        chunk_index=0,
        fusion_score=fusion_score,
        reranker_score=reranker_score,
    )


class LiveMemoryRetrievalTests(unittest.TestCase):
    def test_runtime_is_disabled_by_default_for_other_characters(self):
        config = BasicMemoryAgentConfig(llm_provider="ollama_llm")

        self.assertFalse(config.long_term_memory_retrieval.enabled)

    def test_hidden_context_is_ephemeral_in_basic_agent_memory(self):
        agent = BasicMemoryAgent.__new__(BasicMemoryAgent)
        agent._memory = [{"role": "assistant", "content": "earlier"}]
        agent._llm = SimpleNamespace(base_url="https://example.invalid")
        batch = BatchInput(
            texts=[TextData(TextSource.INPUT, "current question")],
            metadata={"memory_retrieval_context": "PRIVATE RETRIEVAL"},
        )

        messages = agent._to_messages(batch)

        self.assertEqual(messages[-1]["content"][0]["text"], "PRIVATE RETRIEVAL")
        self.assertEqual(agent._memory[-1]["content"], "current question")
        self.assertNotIn("PRIVATE RETRIEVAL", json.dumps(agent._memory))

    def test_dual_reranker_uses_original_score_floor_and_three_rank_lists(self):
        fused = [
            candidate("a", score=1.0, fusion_score=1.0),
            candidate("b", score=0.8, fusion_score=0.8),
            candidate("too_low", score=0.7, fusion_score=0.7),
        ]
        original = [
            candidate("b", reranker_score=0.8),
            candidate("a", reranker_score=0.7),
            candidate("too_low", reranker_score=0.0009),
        ]
        target = [
            candidate("a", reranker_score=0.9),
            candidate("b", reranker_score=0.7),
            candidate("too_low", reranker_score=0.9),
        ]

        blended = _dual_reranker_blend(fused, original, target)

        self.assertEqual([item.candidate_id for item in blended], ["a", "b"])
        self.assertEqual(blended[0].reranker_score, 0.7)
        self.assertTrue(
            any(
                detail.get("stage") == "target_reranker"
                for detail in blended[0].ranking_details
            )
        )

    def test_overview_fallback_is_generic(self):
        self.assertEqual(
            _fallback_granularity("回顾这段时间整个过程发生了什么"),
            "overview",
        )
        self.assertEqual(
            _fallback_granularity("那一天具体发生了什么"),
            "specific_event",
        )

    def test_cloud_request_searches_original_and_cloud_target_without_planner(self):
        service = LiveMemoryRetrievalService.__new__(LiveMemoryRetrievalService)
        observed: dict[str, object] = {}

        def event_rankings(queries):
            observed["queries"] = list(queries)
            return [{"channel": "keyword", "candidate_ids": []}]

        def finish_retrieval(**kwargs):
            observed["finish"] = kwargs
            return "finished"

        service._event_rankings = event_rankings
        service._finish_retrieval = finish_retrieval

        result = service.retrieve_from_cloud_request(
            "original user wording",
            retrieval_query="neutral retrieval target",
            question_granularity="exact_detail",
        )

        self.assertEqual(result, "finished")
        self.assertEqual(
            observed["queries"],
            ["original user wording", "neutral retrieval target"],
        )
        finish = observed["finish"]
        self.assertEqual(
            finish["diagnostics"]["retrieval_query_source"],
            "cloud_tool",
        )

    def test_cloud_request_falls_back_to_neutral_overview_detection(self):
        service = LiveMemoryRetrievalService.__new__(LiveMemoryRetrievalService)
        observed = {}
        service._event_rankings = lambda _queries: []

        def finish_retrieval(**kwargs):
            observed.update(kwargs)
            return "finished"

        service._finish_retrieval = finish_retrieval

        service.retrieve_from_cloud_request(
            "回顾那段时间的整个过程",
            retrieval_query="a long-running experience",
            question_granularity="auto",
        )

        self.assertEqual(observed["granularity"], "overview")

    def test_validated_config_copies_only_active_worker_settings(self):
        config = SimpleNamespace(
            top_k=7,
            embedding_model="embedding",
            reranker_model="reranker",
            model_cache_dir="models",
            embedding_device="cpu",
            reranker_device="cpu",
            reranker_batch_size=4,
        )

        settings = live_retrieval_settings_from_config(config)

        self.assertEqual(settings.top_k, 7)
        self.assertEqual(settings.embedding_model, "embedding")
        self.assertEqual(settings.model_cache_dir, "models")

    def test_clone_shares_warmed_indexes(self):
        service = LiveMemoryRetrievalService.__new__(LiveMemoryRetrievalService)
        service.history_root = Path("history").resolve()
        service.settings = LiveRetrievalSettings()
        service.event_tools = SimpleNamespace(records={})
        service.archive_tools = object()
        service.event_candidates = {}

        clone = service.clone_for_session()

        self.assertIs(clone.event_tools, service.event_tools)
        self.assertIs(clone.archive_tools, service.archive_tools)

    def test_archive_warmup_encodes_only_one_diary_chunk(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diaries = root / "diaries"
            diaries.mkdir()
            (diaries / "diary_2026-01-01.txt").write_text(
                "first diary paragraph\n\nsecond diary paragraph",
                encoding="utf-8",
            )
            tools = ReadOnlyMemoryTools(root)
            observed = {}

            def ensure(model_name, device, *, candidate_ids, cache_scope):
                observed.update(
                    {
                        "model_name": model_name,
                        "device": device,
                        "candidate_ids": candidate_ids,
                        "cache_scope": cache_scope,
                    }
                )

            tools._ensure_semantic_index = ensure

            warmed = tools.warm_semantic_model("embedding", "cpu")

            self.assertTrue(warmed)
            self.assertEqual(len(observed["candidate_ids"]), 1)
            self.assertEqual(observed["cache_scope"], "runtime_model_warmup")

    def test_hidden_payload_warns_cloud_model_about_uncertain_candidates(self):
        context = format_hidden_memory_context(
            {"retrieval_status": "evidence_ready", "evidence": []}
        )

        self.assertIn("隐藏上下文", context)
        self.assertIn("候选可能不是", context)
        self.assertIn("evidence_ready", context)

    def test_trial_log_is_private_jsonl_under_history_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "history"
            root.mkdir()
            path = write_memory_trial_record(
                root,
                {"user_input": "question", "assistant_response": "answer"},
            )

            self.assertEqual(path.parent, root / "memory_trial_logs")
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["user_input"], "question")

    def test_dated_diary_search_builds_only_the_requested_window_index(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "history"
            diaries = root / "diaries"
            weekly = root / "weekly"
            diaries.mkdir(parents=True)
            weekly.mkdir()
            (diaries / "diary_2026-01-01.txt").write_text(
                "第一天的记录", encoding="utf-8"
            )
            (diaries / "diary_2026-01-05.txt").write_text(
                "第五天的记录", encoding="utf-8"
            )
            (weekly / "weekly_2025-12-29_to_2026-01-04.txt").write_text(
                "这一周的记录", encoding="utf-8"
            )
            tools = ReadOnlyMemoryTools(root)
            captured: dict[str, object] = {}

            class FakeModel:
                def encode(self, *_args, **_kwargs):
                    return np.asarray([1.0, 0.0], dtype=np.float32)

            def fake_ensure(
                model_name,
                device,
                *,
                candidate_ids,
                cache_scope,
            ):
                captured["candidate_ids"] = candidate_ids
                captured["cache_scope"] = cache_scope
                tools._semantic_model = FakeModel()
                tools._semantic_embeddings = np.tile(
                    np.asarray([[1.0, 0.0]], dtype=np.float32),
                    (len(candidate_ids), 1),
                )
                tools._semantic_ids = candidate_ids
                tools._semantic_runtime = {"device": device}

            tools._ensure_semantic_index = fake_ensure
            results = tools.search_semantic_memory(
                "第一天",
                ["diary"],
                start_date="2026-01-01",
                end_date="2026-01-02",
            )

            self.assertEqual(len(captured["candidate_ids"]), 1)
            self.assertIn("diary_2026-01-01.txt", results[0].source_file)
            self.assertEqual(
                captured["cache_scope"],
                "diary_2026-01-01_2026-01-02",
            )


if __name__ == "__main__":
    unittest.main()
