import json
import tempfile
import unittest
from pathlib import Path

from src.open_llm_vtuber.memory.diary_review import (
    DiaryReviewError,
    approval_path,
    load_matching_approval,
    wait_for_diary_approval,
)


class DiaryReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.history = Path(self.temporary_directory.name)
        self.diary = self.history / "diaries" / "diary_2026-08-12.txt"
        self.diary.parent.mkdir(parents=True)
        self.diary.write_text("初始日记", encoding="utf-8")

    def test_approve_records_the_reviewed_file_hash(self):
        result = wait_for_diary_approval(
            self.history,
            "2026-08-12",
            self.diary,
            input_func=lambda _prompt: "approve",
        )

        self.assertEqual(result["status"], "approved")
        marker = approval_path(self.history, "2026-08-12")
        self.assertTrue(marker.is_file())
        self.assertEqual(
            json.loads(marker.read_text(encoding="utf-8"))["diary_sha256"],
            result["diary_sha256"],
        )
        self.assertIsNotNone(
            load_matching_approval(self.history, "2026-08-12", self.diary)
        )

    def test_matching_approval_skips_later_prompt(self):
        wait_for_diary_approval(
            self.history,
            "2026-08-12",
            self.diary,
            input_func=lambda _prompt: "approve",
        )

        result = wait_for_diary_approval(
            self.history,
            "2026-08-12",
            self.diary,
            input_func=lambda _prompt: self.fail("prompt must be skipped"),
        )

        self.assertEqual(result["status"], "already_approved")

    def test_edit_after_approval_requires_new_approval(self):
        wait_for_diary_approval(
            self.history,
            "2026-08-12",
            self.diary,
            input_func=lambda _prompt: "approve",
        )
        self.diary.write_text("人工修改后的日记", encoding="utf-8")
        prompts = []

        result = wait_for_diary_approval(
            self.history,
            "2026-08-12",
            self.diary,
            input_func=lambda prompt: prompts.append(prompt) or "approve",
        )

        self.assertEqual(result["status"], "approved")
        self.assertEqual(len(prompts), 1)

    def test_invalid_command_does_not_approve(self):
        commands = iter(["", "yes", "确认"])
        result = wait_for_diary_approval(
            self.history,
            "2026-08-12",
            self.diary,
            input_func=lambda _prompt: next(commands),
        )
        self.assertEqual(result["status"], "approved")

    def test_abort_fails_closed_without_marker(self):
        with self.assertRaisesRegex(DiaryReviewError, "aborted_by_user"):
            wait_for_diary_approval(
                self.history,
                "2026-08-12",
                self.diary,
                input_func=lambda _prompt: "abort",
            )
        self.assertFalse(approval_path(self.history, "2026-08-12").exists())

    def test_missing_interactive_input_fails_closed(self):
        def no_input(_prompt):
            raise EOFError

        with self.assertRaisesRegex(DiaryReviewError, "input_unavailable"):
            wait_for_diary_approval(
                self.history,
                "2026-08-12",
                self.diary,
                input_func=no_input,
            )


if __name__ == "__main__":
    unittest.main()
