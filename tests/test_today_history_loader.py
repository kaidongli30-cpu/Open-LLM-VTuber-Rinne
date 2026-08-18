import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.open_llm_vtuber.memory import TodayHistoryLoader


LOCAL_TZ = timezone(timedelta(hours=8))


class TodayHistoryLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_directory.name) / "rinne_01"
        self.root.mkdir()

    def tearDown(self):
        self.temp_directory.cleanup()

    def write_json(self, relative_path: str, records) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(records, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def test_three_am_memory_day_boundary(self):
        loader = TodayHistoryLoader(self.root, local_timezone=LOCAL_TZ)

        memory_day, start, end = loader.memory_day_window(
            datetime(2026, 7, 20, 2, 30, tzinfo=LOCAL_TZ)
        )
        self.assertEqual(memory_day.isoformat(), "2026-07-19")
        self.assertEqual(start, datetime(2026, 7, 19, 3, 0, tzinfo=LOCAL_TZ))
        self.assertEqual(end, datetime(2026, 7, 20, 3, 0, tzinfo=LOCAL_TZ))

        next_day, next_start, _ = loader.memory_day_window(
            datetime(2026, 7, 20, 3, 0, tzinfo=LOCAL_TZ)
        )
        self.assertEqual(next_day.isoformat(), "2026-07-20")
        self.assertEqual(
            next_start,
            datetime(2026, 7, 20, 3, 0, tzinfo=LOCAL_TZ),
        )

    def test_scan_merge_sort_deduplicate_limit_and_diagnostics(self):
        first = self.write_json(
            "2026-07-19_02-00-00_first.json",
            [
                {"role": "metadata", "content": {"name": "session"}},
                {
                    "role": "human",
                    "content": "previous memory day",
                    "timestamp": "2026-07-19T02:59:59",
                },
                {
                    "role": "human",
                    "content": "boundary turn",
                    "timestamp": "2026-07-19T03:00:00",
                },
                {
                    "role": "ai",
                    "content": "shared turn",
                    "timestamp": "2026-07-19T04:00:00",
                },
                {"role": "human", "content": "missing time"},
                {
                    "role": "ai",
                    "content": "invalid time",
                    "timestamp": "not-a-time",
                },
            ],
        )
        second = self.write_json(
            "2026-07-19_04-00-00_second.json",
            [
                {
                    "role": "ai",
                    "content": "shared turn",
                    "timestamp": "2026-07-19T04:00:00",
                },
                {
                    "role": "human",
                    "content": "latest turn",
                    "timestamp": "2026-07-19T10:00:00",
                },
                {
                    "role": "system",
                    "content": "unsupported",
                    "timestamp": "2026-07-19T11:00:00",
                },
                {
                    "role": "human",
                    "content": {"not": "text"},
                    "timestamp": "2026-07-19T12:00:00",
                },
            ],
        )
        corrupted = self.root / "broken.json"
        corrupted.write_text("{broken", encoding="utf-8")
        self.write_json(
            "past_history/ignored.json",
            [
                {
                    "role": "human",
                    "content": "must not be scanned",
                    "timestamp": "2026-07-19T13:00:00",
                }
            ],
        )
        self.write_json(
            "diaries/ignored.json",
            [
                {
                    "role": "human",
                    "content": "must not be scanned either",
                    "timestamp": "2026-07-19T14:00:00",
                }
            ],
        )
        before = {
            path: path.read_bytes()
            for path in (first, second, corrupted)
        }
        loader = TodayHistoryLoader(self.root, local_timezone=LOCAL_TZ)

        result = loader.load(
            reference_time=datetime(2026, 7, 19, 12, 0, tzinfo=LOCAL_TZ),
            limit=2,
        )
        json.dumps(result.to_dict(), ensure_ascii=False)

        diagnostics = result.diagnostics
        self.assertEqual(diagnostics.scanned_files, 3)
        self.assertEqual(diagnostics.parsed_files, 2)
        self.assertEqual(diagnostics.corrupted_files, 1)
        self.assertEqual(diagnostics.corrupted_file_names, ["broken.json"])
        self.assertEqual(diagnostics.metadata_records, 1)
        self.assertEqual(diagnostics.missing_timestamp_records, 1)
        self.assertEqual(diagnostics.invalid_timestamp_records, 1)
        self.assertEqual(diagnostics.invalid_content_records, 1)
        self.assertEqual(diagnostics.unsupported_role_records, 1)
        self.assertEqual(diagnostics.outside_memory_day_records, 1)
        self.assertEqual(diagnostics.eligible_records, 4)
        self.assertEqual(diagnostics.duplicate_records, 1)
        self.assertEqual(diagnostics.truncated_records, 1)
        self.assertEqual(diagnostics.returned_turns, 2)
        self.assertEqual(diagnostics.mixed_memory_day_files, 1)
        self.assertEqual(
            diagnostics.mixed_file_names,
            ["2026-07-19_02-00-00_first.json"],
        )
        self.assertEqual(
            [turn.timestamp.hour for turn in result.turns],
            [4, 10],
        )
        self.assertTrue(
            all(turn.timestamp.utcoffset() is not None for turn in result.turns)
        )
        self.assertTrue(
            all("past_history" not in turn.source_ref for turn in result.turns)
        )
        self.assertEqual(
            {path: path.read_bytes() for path in before},
            before,
        )

    def test_one_corrupted_file_does_not_hide_valid_files(self):
        self.write_json(
            "valid.json",
            [
                {
                    "role": "human",
                    "content": "valid",
                    "timestamp": "2026-07-19T05:00:00+08:00",
                }
            ],
        )
        (self.root / "invalid.json").write_text("not json", encoding="utf-8")
        loader = TodayHistoryLoader(self.root, local_timezone=LOCAL_TZ)

        result = loader.load(
            reference_time=datetime(2026, 7, 19, 12, 0, tzinfo=LOCAL_TZ)
        )

        self.assertEqual(len(result.turns), 1)
        self.assertEqual(result.diagnostics.corrupted_files, 1)
        self.assertEqual(result.diagnostics.parsed_files, 1)

    def test_missing_root_returns_empty_diagnostics_instead_of_raising(self):
        missing_root = self.root / "missing"
        loader = TodayHistoryLoader(missing_root, local_timezone=LOCAL_TZ)

        result = loader.load(
            reference_time=datetime(2026, 7, 19, 12, 0, tzinfo=LOCAL_TZ)
        )

        self.assertEqual(result.turns, [])
        self.assertTrue(result.diagnostics.history_root_missing)
        self.assertEqual(result.diagnostics.scanned_files, 0)


if __name__ == "__main__":
    unittest.main()
