import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_server
from src.open_llm_vtuber.config_manager.daily_child_event import (
    DailyChildEventGenerationConfig,
)


class StartupMemoryGenerationTests(unittest.TestCase):
    def test_runs_diary_then_child_event_worker_then_weekly_then_monthly(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diary_path = root / "diaries" / "diary_2026-08-02.txt"
            diary_path.parent.mkdir()
            diary_path.write_text("昨天的日记", encoding="utf-8")
            call_order: list[str] = []

            def diary_step(_target):
                call_order.append("diary")

            def weekly_step(_now):
                call_order.append("weekly")
                return SimpleNamespace(status="existing", error=None)

            def monthly_step(_now):
                call_order.append("monthly")
                return SimpleNamespace(status="existing", error=None)

            def child_event_step(_day, _root):
                call_order.append("child_event")
                return SimpleNamespace(process=SimpleNamespace(pid=1234))

            def review_step(_root, _day, _path):
                call_order.append("review")
                return {"status": "approved"}

            with (
                patch("diary_generator.generate_for_date", side_effect=diary_step),
                patch(
                    "weekly_generator.generate_latest_completed_week",
                    side_effect=weekly_step,
                ),
                patch(
                    "monthly_generator.generate_latest_completed_month",
                    side_effect=monthly_step,
                ),
                patch(
                    "src.open_llm_vtuber.memory.daily_child_events."
                    "launch_daily_child_event_worker",
                    side_effect=child_event_step,
                ),
                patch(
                    "src.open_llm_vtuber.memory.diary_review."
                    "wait_for_diary_approval",
                    side_effect=review_step,
                ),
            ):
                run_server.prepare_rinne_memories_on_startup(
                    datetime(2026, 8, 3, 20, 0), root
                )

            self.assertEqual(
                call_order, ["diary", "review", "child_event", "weekly", "monthly"]
            )

    def test_existing_child_events_skip_worker_before_process_start(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diary_path = root / "diaries" / "diary_2026-08-02.txt"
            diary_path.parent.mkdir()
            diary_path.write_text("已经整理过的日记", encoding="utf-8")

            with (
                patch("diary_generator.generate_for_date"),
                patch(
                    "src.open_llm_vtuber.memory.diary_review."
                    "wait_for_diary_approval",
                    return_value={"status": "already_approved"},
                ),
                patch(
                    "src.open_llm_vtuber.memory.daily_child_events."
                    "daily_child_event_publication_status",
                    return_value="current",
                ),
                patch(
                    "src.open_llm_vtuber.memory.daily_child_events."
                    "launch_daily_child_event_worker"
                ) as child_event_step,
                patch(
                    "weekly_generator.generate_latest_completed_week",
                    return_value=SimpleNamespace(status="existing", error=None),
                ),
                patch(
                    "monthly_generator.generate_latest_completed_month",
                    return_value=SimpleNamespace(status="existing", error=None),
                ),
                patch("run_server.logger.info") as info_log,
            ):
                outcome = run_server.prepare_rinne_memories_on_startup(
                    datetime(2026, 8, 3, 20, 0), root
                )

            child_event_step.assert_not_called()
            self.assertIsNone(outcome.child_event_worker)
            self.assertTrue(
                any(
                    "2026-08-02 事件已存在，跳过整理；未调用事件生成模型"
                    in call.args[0]
                    for call in info_log.call_args_list
                )
            )

    def test_stale_child_events_skip_worker_and_explain_diary_change(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diary_path = root / "diaries" / "diary_2026-08-02.txt"
            diary_path.parent.mkdir()
            diary_path.write_text("后来修改的日记", encoding="utf-8")

            with (
                patch("diary_generator.generate_for_date"),
                patch(
                    "src.open_llm_vtuber.memory.diary_review."
                    "wait_for_diary_approval",
                    return_value={"status": "approved"},
                ),
                patch(
                    "src.open_llm_vtuber.memory.daily_child_events."
                    "daily_child_event_publication_status",
                    return_value="stale",
                ),
                patch(
                    "src.open_llm_vtuber.memory.daily_child_events."
                    "launch_daily_child_event_worker"
                ) as child_event_step,
                patch(
                    "weekly_generator.generate_latest_completed_week",
                    return_value=SimpleNamespace(status="existing", error=None),
                ),
                patch(
                    "monthly_generator.generate_latest_completed_month",
                    return_value=SimpleNamespace(status="existing", error=None),
                ),
                patch("run_server.logger.warning") as warning_log,
            ):
                outcome = run_server.prepare_rinne_memories_on_startup(
                    datetime(2026, 8, 3, 20, 0), root
                )

            child_event_step.assert_not_called()
            self.assertIsNone(outcome.child_event_worker)
            self.assertIn(
                "事件已存在，但日记已在事件生成后修改",
                warning_log.call_args.args[0],
            )

    def test_unapproved_diary_blocks_all_downstream_startup_work(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diary_path = root / "diaries" / "diary_2026-08-02.txt"
            diary_path.parent.mkdir()
            diary_path.write_text("等待验收的日记", encoding="utf-8")

            with (
                patch("diary_generator.generate_for_date"),
                patch(
                    "src.open_llm_vtuber.memory.diary_review."
                    "wait_for_diary_approval",
                    side_effect=RuntimeError("review stopped"),
                ),
                patch(
                    "src.open_llm_vtuber.memory.daily_child_events."
                    "launch_daily_child_event_worker"
                ) as child_event_step,
                patch("weekly_generator.generate_latest_completed_week") as weekly_step,
                patch(
                    "monthly_generator.generate_latest_completed_month"
                ) as monthly_step,
            ):
                with self.assertRaisesRegex(RuntimeError, "review stopped"):
                    run_server.prepare_rinne_memories_on_startup(
                        datetime(2026, 8, 3, 20, 0), root
                    )

            child_event_step.assert_not_called()
            weekly_step.assert_not_called()
            monthly_step.assert_not_called()

    def test_missing_diary_stops_weekly_and_monthly_but_does_not_raise(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with (
                patch("diary_generator.generate_for_date"),
                patch("weekly_generator.generate_latest_completed_week") as weekly_step,
                patch(
                    "monthly_generator.generate_latest_completed_month"
                ) as monthly_step,
                patch(
                    "src.open_llm_vtuber.memory.daily_child_events."
                    "launch_daily_child_event_worker"
                ) as child_event_step,
                patch(
                    "src.open_llm_vtuber.memory.diary_review."
                    "wait_for_diary_approval"
                ) as review_step,
            ):
                run_server.prepare_rinne_memories_on_startup(
                    datetime(2026, 8, 3, 20, 0), root
                )

            weekly_step.assert_not_called()
            monthly_step.assert_not_called()
            child_event_step.assert_not_called()
            review_step.assert_not_called()

    def test_child_event_launch_failure_does_not_stop_periodic_memories(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diary_path = root / "diaries" / "diary_2026-08-02.txt"
            diary_path.parent.mkdir()
            diary_path.write_text("昨天的日记", encoding="utf-8")

            with (
                patch("diary_generator.generate_for_date"),
                patch(
                    "src.open_llm_vtuber.memory.daily_child_events."
                    "launch_daily_child_event_worker",
                    side_effect=OSError("后台启动失败"),
                ),
                patch(
                    "src.open_llm_vtuber.memory.diary_review."
                    "wait_for_diary_approval",
                    return_value={"status": "approved"},
                ),
                patch(
                    "weekly_generator.generate_latest_completed_week",
                    return_value=SimpleNamespace(status="existing", error=None),
                ) as weekly_step,
                patch(
                    "monthly_generator.generate_latest_completed_month",
                    return_value=SimpleNamespace(status="existing", error=None),
                ) as monthly_step,
            ):
                run_server.prepare_rinne_memories_on_startup(
                    datetime(2026, 8, 3, 20, 0), root
                )

            weekly_step.assert_called_once()
            monthly_step.assert_called_once()

    def test_disabled_daily_event_provider_does_not_start_worker(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diary_path = root / "diaries" / "diary_2026-08-02.txt"
            diary_path.parent.mkdir()
            diary_path.write_text("已经通过人工验收的日记", encoding="utf-8")
            disabled = DailyChildEventGenerationConfig(enabled=False)

            with (
                patch("diary_generator.generate_for_date"),
                patch(
                    "src.open_llm_vtuber.memory.diary_review.wait_for_diary_approval",
                    return_value={"status": "approved"},
                ),
                patch(
                    "src.open_llm_vtuber.memory.daily_child_events.daily_child_event_publication_status",
                    return_value="missing",
                ),
                patch(
                    "src.open_llm_vtuber.memory.daily_child_events.launch_daily_child_event_worker"
                ) as child_event_step,
                patch(
                    "weekly_generator.generate_latest_completed_week",
                    return_value=SimpleNamespace(status="existing", error=None),
                ),
                patch(
                    "monthly_generator.generate_latest_completed_month",
                    return_value=SimpleNamespace(status="existing", error=None),
                ),
                patch("run_server.logger.info") as info_log,
            ):
                run_server.prepare_rinne_memories_on_startup(
                    datetime(2026, 8, 3, 20, 0),
                    root,
                    generation_settings=disabled,
                )

            child_event_step.assert_not_called()
            self.assertTrue(
                any(
                    "接口已禁用" in call.args[0]
                    for call in info_log.call_args_list
                )
            )


if __name__ == "__main__":
    unittest.main()
