import asyncio
import json
import unittest

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.mcpp.types import (
    ToolCallFunctionObject,
    ToolCallObject,
)
from src.open_llm_vtuber.memory.cloud_memory_tool import (
    CloudLongTermMemoryTool,
    CloudMemoryToolHandlerResult,
)


class CloudLongTermMemoryToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_executes_once_and_reuses_the_same_result(self):
        calls: list[dict[str, str]] = []

        async def handler(arguments):
            calls.append(arguments)
            await asyncio.sleep(0)
            return CloudMemoryToolHandlerResult(
                content="neutral evidence",
                diagnostics={"status": "completed"},
            )

        tool = CloudLongTermMemoryTool(
            handler,
            turn_started_perf=0.0,
        )
        arguments = {
            "query": "  an older experience  ",
            "question_granularity": "exact_detail",
        }

        first = await tool.execute(arguments)
        second = await tool.execute({"query": "different"})

        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["query"], "an older experience")
        self.assertTrue(tool.snapshot()["called"])
        self.assertEqual(
            tool.snapshot()["tool_result_sent_to_cloud"],
            "neutral evidence",
        )

    async def test_invalid_query_returns_a_safe_error_without_calling_handler(self):
        calls = 0

        async def handler(_arguments):
            nonlocal calls
            calls += 1
            return CloudMemoryToolHandlerResult(content="unused")

        tool = CloudLongTermMemoryTool(handler, turn_started_perf=0.0)

        result = await tool.execute({"query": ""})

        self.assertTrue(result.is_error)
        self.assertEqual(calls, 0)
        self.assertIn("不要猜测", result.content)

    async def test_missing_granularity_is_marked_for_neutral_runtime_fallback(self):
        received = None

        async def handler(arguments):
            nonlocal received
            received = arguments
            return CloudMemoryToolHandlerResult(content="unused")

        tool = CloudLongTermMemoryTool(handler, turn_started_perf=0.0)

        await tool.execute({"query": "older experience"})

        self.assertEqual(received["question_granularity"], "auto")

    def test_schema_is_neutral_and_limits_the_tool_to_long_term_recall(self):
        async def handler(_arguments):
            return CloudMemoryToolHandlerResult(content="unused")

        tool = CloudLongTermMemoryTool(handler, turn_started_perf=0.0)
        description = tool.openai_schema["function"]["description"]

        self.assertIn("最近14天记忆", description)
        self.assertIn("每轮最多调用一次", description)
        self.assertIn("不是匹配某几个固定词", tool.decision_instructions)
        self.assertNotIn("武汉", description)
        self.assertNotIn("求婚", description)

    async def test_openai_loop_hides_tool_turn_and_removes_tool_after_use(self):
        tool_lists: list[list[dict]] = []

        class FakeLLM:
            call_count = 0

            async def chat_completion(self, _messages, _system, tools=None):
                self.call_count += 1
                tool_lists.append(list(tools or []))
                if self.call_count == 1:
                    yield [
                        ToolCallObject(
                            id="memory-1",
                            function=ToolCallFunctionObject(
                                name="search_long_term_memory",
                                arguments=json.dumps(
                                    {
                                        "query": "an older experience",
                                        "question_granularity": "specific_event",
                                    }
                                ),
                            ),
                        )
                    ]
                else:
                    yield "natural final reply"

        async def handler(_arguments):
            return CloudMemoryToolHandlerResult(content="hidden evidence")

        agent = BasicMemoryAgent.__new__(BasicMemoryAgent)
        agent._llm = FakeLLM()
        agent._system = "neutral system"
        agent._memory = []
        agent._last_ai_message_texts = []
        agent.prompt_mode_flag = False
        agent._mcp_prompt_string = ""
        agent._use_mcpp = False
        agent._tool_manager = None
        agent._tool_executor = None
        memory_tool = CloudLongTermMemoryTool(handler, turn_started_perf=0.0)

        outputs = [
            item
            async for item in agent._openai_tool_interaction_loop(
                [{"role": "user", "content": "question"}],
                [memory_tool.openai_schema],
                internal_memory_tool=memory_tool,
            )
        ]

        self.assertEqual(outputs, ["natural final reply"])
        self.assertEqual(len(tool_lists[0]), 1)
        self.assertEqual(tool_lists[1], [])
        self.assertEqual(agent._memory[-1]["content"], "natural final reply")

    async def test_openai_loop_never_silently_returns_an_empty_reply(self):
        class EmptyLLM:
            async def chat_completion(self, _messages, _system, tools=None):
                if False:
                    yield "unreachable"

        agent = BasicMemoryAgent.__new__(BasicMemoryAgent)
        agent._llm = EmptyLLM()
        agent._system = "neutral system"
        agent._memory = []
        agent._last_ai_message_texts = []
        agent.prompt_mode_flag = False
        agent._mcp_prompt_string = ""
        agent._use_mcpp = False
        agent._tool_manager = None
        agent._tool_executor = None

        outputs = [
            item
            async for item in agent._openai_tool_interaction_loop(
                [{"role": "user", "content": "question"}],
                [],
            )
        ]

        self.assertEqual(outputs, [agent._build_missing_final_reply_message()])
        self.assertEqual(agent._memory[-1]["content"], outputs[0])


if __name__ == "__main__":
    unittest.main()
