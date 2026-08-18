import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import memory_generation_config
import monthly_generator
import weekly_generator


class WeeklyGeneratorTests(unittest.TestCase):
    def test_latest_completed_week_uses_three_am_boundary(self):
        self.assertEqual(
            weekly_generator.latest_completed_week(datetime(2026, 8, 3, 20, 0)),
            (date(2026, 7, 27), date(2026, 8, 2)),
        )
        self.assertEqual(
            weekly_generator.latest_completed_week(datetime(2026, 8, 3, 2, 0)),
            (date(2026, 7, 20), date(2026, 7, 26)),
        )

    def test_generates_metadata_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diary_dir = root / "diaries"
            weekly_dir = root / "weekly"
            diary_dir.mkdir()
            (diary_dir / "diary_2026-07-27.txt").write_text(
                "第一篇日记", encoding="utf-8"
            )
            with (
                patch.object(weekly_generator, "DIARY_DIR", diary_dir),
                patch.object(weekly_generator, "WEEKLY_DIR", weekly_dir),
            ):
                result = weekly_generator.generate_week(
                    date(2026, 7, 27), lambda _source, _period: "自由散文正文"
                )
                original = result.output_path.read_text(encoding="utf-8")
                second = weekly_generator.generate_week(
                    date(2026, 7, 27), lambda _source, _period: "不应覆盖"
                )

            self.assertEqual(result.status, "created")
            self.assertEqual(second.status, "existing")
            self.assertIn("实际读取日记：1篇", original)
            self.assertIn("缺失日记：2026-07-28", original)
            self.assertIn("自由散文正文", original)
            self.assertEqual(result.output_path.read_text(encoding="utf-8"), original)

    def test_failed_generation_leaves_no_output_or_temporary_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diary_dir = root / "diaries"
            weekly_dir = root / "weekly"
            diary_dir.mkdir()
            (diary_dir / "diary_2026-07-27.txt").write_text("日记", encoding="utf-8")
            with (
                patch.object(weekly_generator, "DIARY_DIR", diary_dir),
                patch.object(weekly_generator, "WEEKLY_DIR", weekly_dir),
            ):
                result = weekly_generator.generate_week(
                    date(2026, 7, 27), lambda _source, _period: ""
                )

            self.assertEqual(result.status, "failed")
            self.assertFalse(result.output_path.exists())
            self.assertEqual(
                list(weekly_dir.glob("*.tmp")) if weekly_dir.exists() else [], []
            )


class MonthlyGeneratorTests(unittest.TestCase):
    def test_latest_completed_month_uses_three_am_boundary(self):
        self.assertEqual(
            monthly_generator.latest_completed_month(datetime(2026, 8, 3, 20, 0)),
            (date(2026, 7, 1), date(2026, 7, 31)),
        )
        self.assertEqual(
            monthly_generator.latest_completed_month(datetime(2026, 8, 1, 2, 0)),
            (date(2026, 6, 1), date(2026, 6, 30)),
        )

    def test_monthly_reads_daily_diaries_and_records_missing_dates(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diary_dir = root / "diaries"
            monthly_dir = root / "monthly"
            diary_dir.mkdir()
            (root / "weekly").mkdir()
            (root / "weekly" / "weekly_2026-07-06_to_2026-07-12.txt").write_text(
                "这篇周记不应作为月记来源", encoding="utf-8"
            )
            (diary_dir / "diary_2026-07-01.txt").write_text(
                "七月一日日记", encoding="utf-8"
            )
            captured_source = ""

            def fake_llm(source: str, _period: str) -> str:
                nonlocal captured_source
                captured_source = source
                return "七月自由散文正文"

            with (
                patch.object(monthly_generator, "DIARY_DIR", diary_dir),
                patch.object(monthly_generator, "MONTHLY_DIR", monthly_dir),
            ):
                result = monthly_generator.generate_month(date(2026, 7, 1), fake_llm)

            document = result.output_path.read_text(encoding="utf-8")
            self.assertEqual(result.status, "created")
            self.assertIn("七月一日日记", captured_source)
            self.assertNotIn("这篇周记不应作为月记来源", captured_source)
            self.assertIn("实际读取日记：1篇", document)
            self.assertIn("缺失日记：2026-07-02", document)
            self.assertIn("七月自由散文正文", document)


class VisibleMemoryGenerationConfigTests(unittest.TestCase):
    def _response(self, content: str) -> Mock:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": content}}]}
        return response

    def test_weekly_uses_values_from_visible_config_file(self):
        response = self._response("周记正文")
        with (
            patch.object(memory_generation_config, "API_KEY", "test-key"),
            patch.object(memory_generation_config, "BASE_URL", "https://test/weekly"),
            patch.object(memory_generation_config, "MODEL", "test-model"),
            patch.object(memory_generation_config, "WEEKLY_MAX_TOKENS", 321),
            patch.object(memory_generation_config, "WEEKLY_TIMEOUT_SECONDS", 45),
            patch.object(
                weekly_generator.requests, "post", return_value=response
            ) as post,
        ):
            result = weekly_generator.call_weekly_llm("来源", "测试周")

        self.assertEqual(result, "周记正文")
        self.assertEqual(post.call_args.args[0], "https://test/weekly")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "test-model")
        self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 321)
        self.assertEqual(post.call_args.kwargs["timeout"], 45)

    def test_monthly_uses_values_from_visible_config_file(self):
        response = self._response("月记正文")
        with (
            patch.object(memory_generation_config, "API_KEY", "test-key"),
            patch.object(memory_generation_config, "BASE_URL", "https://test/monthly"),
            patch.object(memory_generation_config, "MODEL", "test-model"),
            patch.object(memory_generation_config, "MONTHLY_MAX_TOKENS", 654),
            patch.object(memory_generation_config, "MONTHLY_TIMEOUT_SECONDS", 90),
            patch.object(
                monthly_generator.requests, "post", return_value=response
            ) as post,
        ):
            result = monthly_generator.call_monthly_llm("来源", "测试月")

        self.assertEqual(result, "月记正文")
        self.assertEqual(post.call_args.args[0], "https://test/monthly")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "test-model")
        self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 654)
        self.assertEqual(post.call_args.kwargs["timeout"], 90)


class PeriodicMemoryPromptContractTests(unittest.TestCase):
    def test_weekly_prompt_keeps_time_causality_and_grounded_intimacy(self):
        prompt = weekly_generator.WEEKLY_SYSTEM_PROMPT
        self.assertIn("连续发生或发展了多天", prompt)
        self.assertIn("缺少前因后果的状态", prompt)
        self.assertIn("不得自行补充原因", prompt)
        self.assertIn("私密亲密经历", prompt)
        self.assertIn("1200至2200个中文字符", prompt)

    def test_monthly_prompt_rejects_invention_and_uses_soft_length(self):
        prompt = monthly_generator.MONTHLY_SYSTEM_PROMPT
        self.assertIn("事实准确性是最高原则", prompt)
        self.assertIn("找不到依据的内容必须删除", prompt)
        self.assertIn("最小因果链", prompt)
        self.assertIn("私密亲密经历", prompt)
        self.assertIn("2500至4000个中文字符", prompt)


if __name__ == "__main__":
    unittest.main()
