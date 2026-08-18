import unittest
from types import SimpleNamespace

from src.open_llm_vtuber.agent.stateless_llm.openai_compatible_llm import AsyncLLM


class _EmptyStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def close(self):
        return None


class _FakeCompletions:
    def __init__(self):
        self.stream_flags = []

    async def create(self, **kwargs):
        self.stream_flags.append(kwargs["stream"])
        if kwargs["stream"]:
            return _EmptyStream()
        message = SimpleNamespace(content="recovered reply", tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )


class EmptyStreamRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_stream_falls_back_to_one_non_streaming_request(self):
        completions = _FakeCompletions()
        llm = AsyncLLM.__new__(AsyncLLM)
        llm.base_url = "https://example.invalid/v1"
        llm.model = "test-model"
        llm.temperature = 1.0
        llm.max_concurrent_requests = 1
        llm.min_request_interval_seconds = 0.0
        llm.support_tools = True
        llm._limiter_key = "empty-stream-recovery-test"
        llm.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

        outputs = [
            item
            async for item in llm.chat_completion(
                [{"role": "user", "content": "hello"}],
                "system",
            )
        ]

        self.assertEqual(outputs, ["recovered reply"])
        self.assertEqual(completions.stream_flags, [True, False])


if __name__ == "__main__":
    unittest.main()
