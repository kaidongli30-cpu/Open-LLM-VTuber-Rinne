import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.open_llm_vtuber.memory.long_term_archive import (
    load_today_messages,
    select_long_term_memories,
)


class LongTermArchiveSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        for directory in ("diaries", "weekly", "monthly"):
            (self.root / directory).mkdir()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write(self, relative_path: str, content: str = "记忆正文") -> None:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def test_august_third_uses_three_monthlies_and_two_diaries(self):
        for month in ("2026-05", "2026-06", "2026-07"):
            self._write(f"monthly/monthly_{month}.txt", f"{month}月记")
        self._write("weekly/weekly_2026-07-27_to_2026-08-02.txt", "跨月周记")
        self._write("diaries/diary_2026-08-01.txt", "八月一日日记")
        self._write("diaries/diary_2026-08-02.txt", "八月二日日记")

        selection = select_long_term_memories(self.root)

        self.assertEqual(len(selection.monthly_entries), 3)
        self.assertEqual(len(selection.weekly_entries), 0)
        self.assertEqual(len(selection.diary_entries), 2)
        self.assertEqual(selection.diagnostics.skipped_overlapping_weeklies, 1)

    def test_weeklies_and_diaries_fill_dates_not_covered_by_monthlies(self):
        self._write("monthly/monthly_2026-07.txt", "七月月记")
        self._write("weekly/weekly_2026-08-03_to_2026-08-09.txt", "八月第一周")
        self._write("weekly/weekly_2026-08-10_to_2026-08-16.txt", "八月第二周")
        for day in (1, 2, 3, 17, 18, 19):
            self._write(f"diaries/diary_2026-08-{day:02d}.txt", f"八月{day}日")

        selection = select_long_term_memories(self.root)

        self.assertEqual(len(selection.monthly_entries), 1)
        self.assertEqual(len(selection.weekly_entries), 2)
        self.assertEqual(
            [entry.start_date.day for entry in selection.diary_entries],
            [1, 2, 17, 18, 19],
        )
        self.assertEqual(selection.diagnostics.skipped_covered_diaries, 1)

    def test_empty_and_invalid_files_are_not_injected(self):
        self._write("monthly/monthly_2026-07.txt", "")
        self._write("weekly/weekly_bad.txt", "不能读取")
        self._write("diaries/diary_2026-08-01.txt", "有效日记")

        selection = select_long_term_memories(self.root)

        self.assertEqual(selection.total_entries, 1)
        self.assertEqual(selection.diagnostics.empty_files, 1)
        self.assertEqual(selection.diagnostics.invalid_filenames, 1)

    def test_llm_text_contains_content_but_not_internal_diagnostics(self):
        self._write("monthly/monthly_2026-07.txt", "七月的正文")
        selection = select_long_term_memories(self.root)

        llm_text = selection.to_llm_text()

        self.assertIn("七月的正文", llm_text)
        self.assertIn("以最新对话为准", llm_text)
        self.assertNotIn("skipped_overlapping_weeklies", llm_text)


class TodayMessageLoadingTests(unittest.TestCase):
    def test_only_root_json_after_three_am_is_loaded(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            before = root / "2026-08-03_02-59-00_before.json"
            after = root / "2026-08-03_03-01-00_after.json"
            child = root / "past_history" / "2026-08-03_04-00-00_child.json"
            child.parent.mkdir()
            before.write_text(
                json.dumps([{"role": "human", "content": "边界以前"}]),
                encoding="utf-8",
            )
            after.write_text(
                json.dumps(
                    [
                        {"role": "metadata", "content": "元数据"},
                        {"role": "human", "content": "用户消息"},
                        {"role": "ai", "content": "凛祢回复"},
                    ]
                ),
                encoding="utf-8",
            )
            child.write_text(
                json.dumps([{"role": "human", "content": "子目录消息"}]),
                encoding="utf-8",
            )

            messages = load_today_messages(root, datetime(2026, 8, 3, 20, 0, 0))

            self.assertEqual(
                messages,
                [
                    {"role": "user", "content": "用户消息"},
                    {"role": "assistant", "content": "凛祢回复"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
