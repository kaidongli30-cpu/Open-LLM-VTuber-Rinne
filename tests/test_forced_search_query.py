import asyncio
import unittest

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.search_query import (
    clean_model_search_query,
    extract_search_query,
    should_force_search,
)


class SearchQueryExtractionTests(unittest.TestCase):
    def test_problem_report_example_extracts_movie_topic(self):
        user_text = (
    "哦对了，用户，嗯……你其实，你的联网功能还有一个小bug没有修复，"
            "我先让你触发一个那个bug。可以吗？就一句话的事，比如……"
            "就还是上午的那个问题吧，你可以搜索一下Obsession这部电影。"
        )

        self.assertTrue(should_force_search(user_text))
        self.assertEqual(extract_search_query(user_text), "Obsession这部电影")

    def test_later_explicit_command_wins_over_search_used_as_a_noun(self):
        user_text = "搜索功能先放一边，请帮我搜索一下 OpenAI 最新模型，然后做个总结"

        self.assertEqual(extract_search_query(user_text), "OpenAI 最新模型")

    def test_search_algorithm_is_not_mutilated(self):
        self.assertEqual(extract_search_query("搜索算法是什么？"), "搜索算法是什么")

    def test_model_output_is_cleaned(self):
        query = clean_model_search_query(
            '查询词：“Obsession 电影”\n',
            fallback_query="Obsession这部电影",
            original_text="请搜索一下Obsession这部电影",
        )

        self.assertEqual(query, "Obsession 电影")

    def test_full_original_model_output_falls_back_to_local_extraction(self):
        original = "用户，先说一句别的，然后请搜索一下 Obsession 这部电影"
        query = clean_model_search_query(
            original,
            fallback_query="Obsession 这部电影",
            original_text=original,
        )

        self.assertEqual(query, "Obsession 这部电影")


class _FakeRewriteLLM:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    async def chat_completion(self, messages, system=None):
        self.calls.append((messages, system))
        for chunk in self.chunks:
            await asyncio.sleep(0)
            yield chunk


class _FakeSearchExecutor:
    def __init__(self):
        self.calls = []

    async def run_single_tool(self, **kwargs):
        self.calls.append(kwargs)
        return False, "1. Obsession - Wikipedia", {}, []


class ForcedSearchQueryRewriteTests(unittest.IsolatedAsyncioTestCase):
    async def test_query_rewriter_uses_model_output(self):
        agent = BasicMemoryAgent.__new__(BasicMemoryAgent)
        agent._memory = [
            {"role": "user", "content": "我们上午聊过电影 Obsession。"},
            {"role": "assistant", "content": "记得。"},
        ]
        agent._llm = _FakeRewriteLLM(["Obsession ", "电影"])

        query = await agent._build_forced_search_query("你可以搜索一下它")

        self.assertEqual(query, "Obsession 电影")
        self.assertEqual(len(agent._llm.calls), 1)
        self.assertIn("我们上午聊过电影 Obsession", agent._llm.calls[0][0][0]["content"])

    async def test_query_rewriter_error_uses_deterministic_fallback(self):
        agent = BasicMemoryAgent.__new__(BasicMemoryAgent)
        agent._memory = []
        agent._llm = _FakeRewriteLLM(
            ["Error calling the chat endpoint: Connection error."]
        )

        query = await agent._build_forced_search_query(
            "请搜索一下 Obsession 这部电影"
        )

        self.assertEqual(query, "Obsession 这部电影")

    async def test_forced_search_sends_rewritten_query_to_search_tool(self):
        original = (
    "哦对了，用户，你的搜索功能还有个bug。"
            "你可以搜索一下Obsession这部电影。"
        )
        agent = BasicMemoryAgent.__new__(BasicMemoryAgent)
        agent._memory = []
        agent._llm = _FakeRewriteLLM(["Obsession 电影"])
        agent._tool_executor = _FakeSearchExecutor()

        result_message = await agent._run_forced_search(original)

        self.assertEqual(
            agent._tool_executor.calls[0]["tool_input"],
            {"query": "Obsession 电影", "max_results": 5},
        )
        self.assertIn("Obsession 电影", result_message["content"])


if __name__ == "__main__":
    unittest.main()
