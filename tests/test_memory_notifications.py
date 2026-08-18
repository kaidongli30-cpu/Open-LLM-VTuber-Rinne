import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.open_llm_vtuber.memory.daily_child_events import ChildEventWorkerLaunch
from src.open_llm_vtuber.server import WebSocketServer
from src.open_llm_vtuber.websocket_handler import WebSocketHandler


class FakeWebSocket:
    def __init__(self):
        self.messages = []

    async def send_text(self, text):
        self.messages.append(json.loads(text))


class PendingNotificationTests(unittest.TestCase):
    def test_notification_is_retained_until_a_client_connects(self):
        handler = WebSocketHandler(SimpleNamespace())
        notification = {
            "type": "memory-notification",
            "level": "success",
            "message": "昨日事件已整理完成",
        }

        asyncio.run(handler.publish_system_notification(notification))
        self.assertEqual(handler.pending_system_notifications, [notification])

        websocket = FakeWebSocket()
        asyncio.run(handler._flush_pending_system_notifications(websocket))
        self.assertEqual(websocket.messages, [notification])
        self.assertEqual(handler.pending_system_notifications, [])


class WorkerMonitorTests(unittest.TestCase):
    def _server(self):
        server = WebSocketServer.__new__(WebSocketServer)
        server.ws_handler = SimpleNamespace(
            publish_system_notification=AsyncMock()
        )
        return server

    def _launch(self, result_path: Path, exit_code: int):
        return ChildEventWorkerLaunch(
            process=SimpleNamespace(wait=lambda: exit_code),
            memory_day="2026-08-11",
            result_path=result_path,
        )

    def test_published_worker_sends_green_success_notification(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "result.json"
            result_path.write_text(
                json.dumps({"status": "published"}), encoding="utf-8"
            )
            server = self._server()

            asyncio.run(
                server._monitor_child_event_worker(self._launch(result_path, 0))
            )

            notification = server.ws_handler.publish_system_notification.await_args.args[0]
            self.assertEqual(notification["level"], "success")
            self.assertEqual(notification["message"], "昨日事件已整理完成")

    def test_validation_failure_sends_short_red_error_notification(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "error_type": "ValidationError",
                        "error": "missing field with private details",
                    }
                ),
                encoding="utf-8",
            )
            server = self._server()

            asyncio.run(
                server._monitor_child_event_worker(self._launch(result_path, 1))
            )

            notification = server.ws_handler.publish_system_notification.await_args.args[0]
            self.assertEqual(notification["level"], "error")
            self.assertEqual(notification["message"], "昨日事件整理失败")
            self.assertEqual(notification["description"], "24B输出未通过格式校验")
            self.assertNotIn("private details", notification["description"])

    def test_already_published_worker_does_not_repeat_notification(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "result.json"
            result_path.write_text(
                json.dumps({"status": "already_published"}), encoding="utf-8"
            )
            server = self._server()

            asyncio.run(
                server._monitor_child_event_worker(self._launch(result_path, 0))
            )

            server.ws_handler.publish_system_notification.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
