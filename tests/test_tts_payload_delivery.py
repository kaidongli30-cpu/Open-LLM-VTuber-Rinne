import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from open_llm_vtuber.conversations.conversation_utils import (
    finalize_conversation_turn,
)
from open_llm_vtuber.conversations.tts_manager import TTSTaskManager


class TTSPayloadDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_payload_barrier_waits_for_websocket_send(self):
        manager = TTSTaskManager()
        send_started = asyncio.Event()
        release_send = asyncio.Event()
        sent_payloads = []

        async def delayed_send(payload: str) -> None:
            send_started.set()
            await release_send.wait()
            sent_payloads.append(json.loads(payload))

        manager._sequence_counter = 1
        manager._sender_task = asyncio.create_task(
            manager._process_payload_queue(delayed_send)
        )
        await manager._payload_queue.put(({"type": "audio"}, 0))
        await send_started.wait()

        barrier = asyncio.create_task(manager.wait_until_payloads_sent())
        await asyncio.sleep(0)
        self.assertFalse(barrier.done())

        release_send.set()
        await asyncio.wait_for(barrier, timeout=1)
        self.assertEqual(sent_payloads, [{"type": "audio"}])
        manager.clear()

    async def test_finalize_announces_completion_after_payload_barrier(self):
        events = []
        manager = TTSTaskManager()
        manager.task_list = [asyncio.create_task(asyncio.sleep(0))]

        async def wait_until_payloads_sent() -> None:
            events.append("payloads-sent")

        async def websocket_send(payload: str) -> None:
            message = json.loads(payload)
            events.append(message.get("type", message.get("text")))

        manager.wait_until_payloads_sent = wait_until_payloads_sent

        with patch(
            "open_llm_vtuber.conversations.conversation_utils.message_handler.wait_for_response",
            new=AsyncMock(return_value=True),
        ), patch(
            "open_llm_vtuber.conversations.conversation_utils.send_conversation_end_signal",
            new=AsyncMock(),
        ):
            await finalize_conversation_turn(
                tts_manager=manager,
                websocket_send=websocket_send,
                client_uid="test-client",
            )

        self.assertLess(events.index("payloads-sent"), events.index("backend-synth-complete"))


if __name__ == "__main__":
    unittest.main()
