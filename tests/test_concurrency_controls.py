import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.open_llm_vtuber.agent.stateless_llm.request_limiter import (
    build_backend_key,
    limit_request_concurrency,
)
from src.open_llm_vtuber.conversations.conversation_handler import (
    handle_conversation_trigger,
)


class RequestLimiterTests(unittest.IsolatedAsyncioTestCase):
    async def test_limit_request_concurrency_serializes_requests(self):
        backend_key = build_backend_key(
            provider_name="openai_compatible_llm",
            base_url="https://example.test/v1",
            api_key="secret",
        )
        order: list[str] = []
        first_entered = asyncio.Event()
        release_first = asyncio.Event()

        async def first_request():
            async with limit_request_concurrency(backend_key, 1):
                order.append("first-enter")
                first_entered.set()
                await release_first.wait()
                order.append("first-exit")

        async def second_request():
            await first_entered.wait()
            async with limit_request_concurrency(backend_key, 1):
                order.append("second-enter")

        first_task = asyncio.create_task(first_request())
        second_task = asyncio.create_task(second_request())

        await first_entered.wait()
        await asyncio.sleep(0.05)
        self.assertEqual(order, ["first-enter"])

        release_first.set()
        await asyncio.gather(first_task, second_task)
        self.assertEqual(order, ["first-enter", "first-exit", "second-enter"])


class ConversationTriggerTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_client_does_not_start_overlapping_conversations(self):
        active_tasks: dict[str, asyncio.Task | None] = {}
        conversation_started = asyncio.Event()
        release_conversation = asyncio.Event()

        async def fake_process_single_conversation(**_kwargs):
            conversation_started.set()
            await release_conversation.wait()
            return ""

        websocket = SimpleNamespace(send_text=AsyncMock())
        context = SimpleNamespace(system_config=SimpleNamespace(tool_prompts={}))
        chat_group_manager = SimpleNamespace(get_client_group=lambda _client_uid: None)

        with patch(
            "src.open_llm_vtuber.conversations.conversation_handler.process_single_conversation",
            side_effect=fake_process_single_conversation,
        ):
            await handle_conversation_trigger(
                msg_type="text-input",
                data={"text": "first"},
                client_uid="client-1",
                context=context,
                websocket=websocket,
                client_contexts={},
                client_connections={},
                chat_group_manager=chat_group_manager,
                received_data_buffers={"client-1": []},
                current_conversation_tasks=active_tasks,
                broadcast_to_group=AsyncMock(),
            )

            first_task = active_tasks["client-1"]
            await conversation_started.wait()

            await handle_conversation_trigger(
                msg_type="text-input",
                data={"text": "second"},
                client_uid="client-1",
                context=context,
                websocket=websocket,
                client_contexts={},
                client_connections={},
                chat_group_manager=chat_group_manager,
                received_data_buffers={"client-1": []},
                current_conversation_tasks=active_tasks,
                broadcast_to_group=AsyncMock(),
            )

            self.assertIs(active_tasks["client-1"], first_task)
            self.assertFalse(first_task.done())

            release_conversation.set()
            await first_task
            await asyncio.sleep(0)

            self.assertNotIn("client-1", active_tasks)


if __name__ == "__main__":
    unittest.main()
