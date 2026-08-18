import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

from src.open_llm_vtuber.memory import (
    ConversationTurn,
    TodayHistoryDiagnostics,
    TodayHistoryLoadResult,
    TodayMessagePartitioner,
    build_today_llm_delivery_preview,
)
from src.open_llm_vtuber.memory.today_preview_cli import main as preview_main


LOCAL_TZ = timezone(timedelta(hours=8))
WINDOW_START = datetime(2026, 5, 21, 3, 0, tzinfo=LOCAL_TZ)


def make_history(message_count: int) -> TodayHistoryLoadResult:
    messages = [
        ConversationTurn(
            turn_id=f"message-{index:02d}",
            role="user" if index % 2 == 0 else "assistant",
            content=f"content-{index:02d}",
            timestamp=WINDOW_START + timedelta(hours=6, minutes=index),
            source_ref=f"history.json#{index}",
        )
        for index in range(message_count)
    ]
    return TodayHistoryLoadResult(
        memory_day=WINDOW_START.date(),
        window_start=WINDOW_START,
        window_end=WINDOW_START + timedelta(days=1),
        turns=messages,
        diagnostics=TodayHistoryDiagnostics(returned_turns=message_count),
    )


class TodayMessagePartitionerTests(unittest.TestCase):
    def test_exactly_twenty_messages_before_cutoff_are_kept(self):
        history = make_history(30)
        original = history.to_dict()
        as_of = WINDOW_START + timedelta(hours=6, minutes=26)

        partition = TodayMessagePartitioner(20).partition(
            history,
            as_of=as_of,
        )

        self.assertEqual(len(partition.all_messages_before_as_of), 26)
        self.assertEqual(
            [message.turn_id for message in partition.older_today_messages],
            [f"message-{index:02d}" for index in range(6)],
        )
        self.assertEqual(
            [message.turn_id for message in partition.recent_messages],
            [f"message-{index:02d}" for index in range(6, 26)],
        )
        self.assertEqual(
            partition.diagnostics.messages_at_or_after_as_of,
            4,
        )
        self.assertEqual(
            partition.diagnostics.complete_user_assistant_pairs_in_recent,
            10,
        )
        self.assertEqual(partition.diagnostics.unpaired_messages_in_recent, 0)
        self.assertTrue(partition.diagnostics.recent_window_filled)
        self.assertTrue(partition.diagnostics.chronological)
        self.assertEqual(history.to_dict(), original)

    def test_message_exactly_at_as_of_is_not_in_hidden_history(self):
        history = make_history(4)
        as_of = history.turns[2].timestamp

        partition = TodayMessagePartitioner(20).partition(
            history,
            as_of=as_of,
        )

        self.assertEqual(
            [message.turn_id for message in partition.recent_messages],
            ["message-00", "message-01"],
        )
        self.assertEqual(
            partition.diagnostics.messages_at_or_after_as_of,
            2,
        )
        self.assertFalse(partition.diagnostics.recent_window_filled)

    def test_fewer_than_twenty_returns_every_available_earlier_message(self):
        history = make_history(8)
        as_of = WINDOW_START + timedelta(hours=7)

        partition = TodayMessagePartitioner().partition(history, as_of=as_of)

        self.assertEqual(len(partition.recent_messages), 8)
        self.assertEqual(partition.older_today_messages, [])
        self.assertEqual(
            partition.all_messages_before_as_of,
            partition.recent_messages,
        )

    def test_as_of_must_be_aware_and_inside_loaded_memory_day(self):
        history = make_history(2)
        partitioner = TodayMessagePartitioner()

        with self.assertRaises(ValueError):
            partitioner.partition(
                history,
                as_of=datetime(2026, 5, 21, 10, 0),
            )
        with self.assertRaises(ValueError):
            partitioner.partition(
                history,
                as_of=WINDOW_START + timedelta(days=1),
            )

    def test_recent_message_count_must_be_positive(self):
        with self.assertRaises(ValueError):
            TodayMessagePartitioner(0)

    def test_preview_separates_llm_text_from_internal_trace_data(self):
        history = make_history(24)
        as_of = WINDOW_START + timedelta(hours=7)
        partition = TodayMessagePartitioner(20).partition(
            history,
            as_of=as_of,
        )

        preview = build_today_llm_delivery_preview(
            partition,
            current_user_input="current input",
        )
        serialized = preview.to_dict()
        json.dumps(serialized, ensure_ascii=False)

        self.assertFalse(preview.live_pipeline_connected)
        self.assertTrue(preview.current_user_input_included)
        self.assertEqual(preview.dynamic_system_context, "")
        self.assertEqual(
            preview.simulated_messages_for_llm[0],
            {"role": "user", "content": "content-04"},
        )
        self.assertEqual(
            preview.simulated_messages_for_llm[-1],
            {"role": "user", "content": "current input"},
        )
        self.assertEqual(
            set(preview.simulated_messages_for_llm[0]),
            {"role", "content"},
        )
        self.assertIn(
            "timestamp",
            preview.internal_delivery_manifest_not_sent_to_llm[0],
        )
        self.assertEqual(preview.older_today_message_count_not_sent, 4)
        self.assertEqual(
            serialized["would_send_to_llm"]["messages"][-1],
            {
                "role": "user",
                "content": "current input",
            },
        )
        self.assertNotIn(
            "timestamp",
            serialized["would_send_to_llm"]["messages"][0],
        )


class TodayPreviewCliTests(unittest.TestCase):
    def test_cli_writes_preview_below_temporary_history_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            history_root = Path(temporary_directory)
            source = history_root / "history.json"
            records = [
                {
                    "role": "human" if index % 2 == 0 else "ai",
                    "content": f"content-{index:02d}",
                    "timestamp": (
                        WINDOW_START + timedelta(hours=6, minutes=index)
                    ).isoformat(),
                }
                for index in range(24)
            ]
            source.write_text(
                json.dumps(records, ensure_ascii=False),
                encoding="utf-8",
            )
            original_source = source.read_bytes()
            output = history_root / "preview" / "delivery.json"

            with redirect_stdout(StringIO()):
                exit_code = preview_main(
                    [
                        "--history-root",
                        str(history_root),
                        "--as-of",
                        "2026-05-21T10:00:00+08:00",
                        "--current-input",
                        "first current input",
                        "--output",
                        str(output),
                    ]
                )

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertFalse(payload["preview_metadata"]["live_pipeline_connected"])
            self.assertEqual(
                len(payload["would_send_to_llm"]["messages"]),
                21,
            )
            self.assertEqual(
                payload["would_send_to_llm"]["messages"][0],
                {"role": "user", "content": "content-04"},
            )
            self.assertEqual(
                payload["would_send_to_llm"]["messages"][-1],
                {"role": "user", "content": "first current input"},
            )

            with redirect_stdout(StringIO()):
                second_exit_code = preview_main(
                    [
                        "--history-root",
                        str(history_root),
                        "--as-of",
                        "2026-05-21T10:00:00+08:00",
                        "--current-input",
                        "second current input",
                        "--output",
                        str(output),
                    ]
                )

            updated_payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(second_exit_code, 0)
            self.assertEqual(
                updated_payload["would_send_to_llm"]["messages"][-1],
                {"role": "user", "content": "second current input"},
            )
            self.assertEqual(len(list(output.parent.glob("*.json"))), 1)
            self.assertEqual(list(output.parent.glob("*.tmp")), [])
            self.assertEqual(source.read_bytes(), original_source)


if __name__ == "__main__":
    unittest.main()
