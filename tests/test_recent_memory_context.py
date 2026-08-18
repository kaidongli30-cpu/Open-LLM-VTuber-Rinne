import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.open_llm_vtuber.memory.recent_memory_context import (
    load_recent_memory_context,
)


class RecentMemoryContextTests(unittest.TestCase):
    def test_loads_rolling_diaries_and_only_fully_contained_weeklies(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diaries = root / "diaries"
            weekly = root / "weekly"
            diaries.mkdir()
            weekly.mkdir()
            (diaries / "diary_2026-08-01.txt").write_text(
                "window start", encoding="utf-8"
            )
            (diaries / "diary_2026-07-31.txt").write_text(
                "too old", encoding="utf-8"
            )
            (diaries / "diary_2026-08-13.txt").write_text(
                "yesterday", encoding="utf-8"
            )
            (diaries / "diary_2026-08-04.txt").write_text(
                "covered by weekly", encoding="utf-8"
            )
            (weekly / "weekly_2026-08-03_to_2026-08-09.txt").write_text(
                "contained week", encoding="utf-8"
            )
            (weekly / "weekly_2026-07-27_to_2026-08-02.txt").write_text(
                "partial week", encoding="utf-8"
            )

            result = load_recent_memory_context(
                root,
                reference_time=datetime(2026, 8, 14, 12, tzinfo=timezone.utc),
            )

            self.assertEqual(result.window_start.isoformat(), "2026-08-01")
            self.assertEqual(result.window_end.isoformat(), "2026-08-14")
            self.assertEqual(
                [item.source_file for item in result.diary_entries],
                ["diary_2026-08-01.txt", "diary_2026-08-13.txt"],
            )
            self.assertEqual(
                [item.source_file for item in result.weekly_entries],
                ["weekly_2026-08-03_to_2026-08-09.txt"],
            )
            self.assertEqual(
                result.covered_diary_files,
                ["diary_2026-08-04.txt"],
            )
            self.assertIn("2026-08-13｜星期四", result.context)
            self.assertNotIn("too old", result.context)
            self.assertNotIn("partial week", result.context)
            self.assertNotIn("covered by weekly", result.context)

    def test_includes_natural_recall_instruction_and_date_question_exception(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = load_recent_memory_context(
                temporary_directory,
                reference_time=datetime(2026, 8, 14, 12, tzinfo=timezone.utc),
            )

            self.assertIn("不要复述检索结果中包含的日期信息", result.context)
            self.assertIn("直接询问具体日期", result.context)


if __name__ == "__main__":
    unittest.main()
