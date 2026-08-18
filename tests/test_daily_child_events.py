import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.open_llm_vtuber.config_manager.daily_child_event import (
    DailyChildEventGenerationConfig,
)
from src.open_llm_vtuber.memory.daily_child_events import (
    EVENT_DIRECTORY_NAME,
    MANIFEST_FILENAME,
    MOJIBAKE_EVENT_DIRECTORY_NAME,
    PublicationConflict,
    ValidationError,
    audit_mojibake_directories,
    daily_child_event_publication_status,
    event_json_schema,
    is_daily_child_event_published,
    main,
    run_daily_child_event,
    validate_model_response,
)
from src.open_llm_vtuber.memory.daily_child_event_providers import (
    DailyChildEventGenerationResult,
    DailyChildEventProviderError,
)


def valid_response(memory_day: str = "2026-08-11") -> str:
    return json.dumps(
        {
            "memory_day": memory_day,
            "day_richness": "普通",
            "memories": [
                {
                    "type": "事件",
                    "title": '用户讨论"human-centered applications"',
                    "description": "用户和凛祢讨论了英文短语，并保留了原话。",
                    "why_independent": "项目节点",
                    "source_fact": "双方讨论英文短语",
                    "subject_check": "用户与凛祢共同讨论",
                    "time_check": "当天中段",
                    "location_check": "原文未明确",
                    "salience_check": "保留英文短语锚点",
                    "inference_check": "确认没有新增关系结论",
                }
            ],
            "merged_or_discarded": [],
        },
        ensure_ascii=False,
    )


class StructuredResponseTests(unittest.TestCase):
    def test_schema_requires_one_to_five_complete_memories(self):
        schema = event_json_schema("2026-08-11")

        memories = schema["properties"]["memories"]
        self.assertEqual(memories["minItems"], 1)
        self.assertEqual(memories["maxItems"], 5)
        self.assertFalse(memories["items"]["additionalProperties"])

    def test_valid_json_preserves_quoted_text(self):
        result = validate_model_response(valid_response(), "2026-08-11")

        self.assertIn("human-centered applications", result["memories"][0]["title"])

    def test_unescaped_model_quote_is_rejected_instead_of_guessed(self):
        broken = valid_response().replace(
            '\\"human-centered applications\\"',
            '"human-centered applications"',
        )

        with self.assertRaises(ValidationError):
            validate_model_response(broken, "2026-08-11")

    def test_missing_check_field_is_rejected(self):
        payload = json.loads(valid_response())
        del payload["memories"][0]["subject_check"]

        with self.assertRaises(ValidationError):
            validate_model_response(
                json.dumps(payload, ensure_ascii=False), "2026-08-11"
            )


class DailyPublicationTests(unittest.TestCase):
    def test_complete_date_is_published_only_to_correct_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            history = root / "history"
            work = root / "work"
            diary = history / "diaries" / "diary_2026-08-11.txt"
            diary.parent.mkdir(parents=True)
            diary.write_text("昨日の日記", encoding="utf-8")

            def generator(_day, _system, _task, _run_dir):
                return valid_response()

            result = run_daily_child_event(
                "2026-08-11", history, work, generator=generator
            )
            day = history / "events" / "child_events" / "2026-08-11"
            event_dir = day / EVENT_DIRECTORY_NAME

            self.assertEqual(result["status"], "published")
            self.assertTrue(event_dir.is_dir())
            self.assertFalse((day / MOJIBAKE_EVENT_DIRECTORY_NAME).exists())
            self.assertEqual(len(list(event_dir.glob("*.txt"))), 1)
            self.assertTrue((day / MANIFEST_FILENAME).is_file())
            self.assertEqual(
                {item.name for item in day.iterdir() if item.is_dir()},
                {EVENT_DIRECTORY_NAME},
            )

            second = run_daily_child_event(
                "2026-08-11", history, work, generator=generator
            )
            self.assertEqual(second["status"], "already_published")
            self.assertTrue(is_daily_child_event_published("2026-08-11", history))

            diary.write_text("验收后修改过的日记", encoding="utf-8")
            self.assertEqual(
                daily_child_event_publication_status("2026-08-11", history),
                "stale",
            )
            self.assertFalse(is_daily_child_event_published("2026-08-11", history))

    def test_manifest_records_configured_provider_and_model(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            history = root / "history"
            diary = history / "diaries" / "diary_2026-08-11.txt"
            diary.parent.mkdir(parents=True)
            diary.write_text("昨天的日记", encoding="utf-8")
            settings = DailyChildEventGenerationConfig(
                llm_provider="openai_compatible_llm",
                model="event-test-model",
            )

            result = run_daily_child_event(
                "2026-08-11",
                history,
                root / "work",
                generation_settings=settings,
                generator=lambda *_args: DailyChildEventGenerationResult(
                    text=valid_response(),
                    metadata={
                        "llm_provider": settings.llm_provider,
                        "model": settings.model,
                        "base_url": settings.base_url,
                        "temperature": settings.temperature,
                        "max_output_tokens": settings.max_output_tokens,
                        "options": {},
                    },
                ),
            )

            self.assertEqual(result["status"], "published")
            manifest = json.loads(
                (
                    history
                    / "events"
                    / "child_events"
                    / "2026-08-11"
                    / MANIFEST_FILENAME
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["llm_provider"], "openai_compatible_llm")
            self.assertEqual(manifest["model"], "event-test-model")

    def test_provider_failure_never_creates_official_date(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            history = root / "history"
            diary = history / "diaries" / "diary_2026-08-11.txt"
            diary.parent.mkdir(parents=True)
            diary.write_text("昨天的日记", encoding="utf-8")

            with patch(
                "src.open_llm_vtuber.memory.daily_child_events.generate_with_provider",
                side_effect=DailyChildEventProviderError("provider_empty_response"),
            ):
                with self.assertRaises(DailyChildEventProviderError):
                    run_daily_child_event(
                        "2026-08-11",
                        history,
                        root / "work",
                        generation_settings=DailyChildEventGenerationConfig(),
                    )

            self.assertFalse(
                (history / "events" / "child_events" / "2026-08-11").exists()
            )

    def test_cli_writes_machine_readable_success_result(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "result.json"
            with patch(
                "src.open_llm_vtuber.memory.daily_child_events."
                "run_daily_child_event",
                return_value={
                    "status": "published",
                    "memory_day": "2026-08-11",
                    "event_count": 2,
                },
            ):
                exit_code = main(
                    [
                        "--date",
                        "2026-08-11",
                        "--result-path",
                        str(result_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                json.loads(result_path.read_text(encoding="utf-8"))["status"],
                "published",
            )

    def test_cli_writes_machine_readable_failure_result(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "result.json"
            with patch(
                "src.open_llm_vtuber.memory.daily_child_events."
                "run_daily_child_event",
                side_effect=ValidationError("bad schema"),
            ):
                exit_code = main(
                    [
                        "--date",
                        "2026-08-11",
                        "--result-path",
                        str(result_path),
                    ]
                )

            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 1)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error_type"], "ValidationError")

    def test_invalid_json_never_creates_official_date(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            history = root / "history"
            diary = history / "diaries" / "diary_2026-08-11.txt"
            diary.parent.mkdir(parents=True)
            diary.write_text("昨天的日记", encoding="utf-8")

            with self.assertRaises(ValidationError):
                run_daily_child_event(
                    "2026-08-11",
                    history,
                    root / "work",
                    generator=lambda *_args: '{"broken":"quote "inside""}',
                )

            self.assertFalse(
                (history / "events" / "child_events" / "2026-08-11").exists()
            )

    def test_legacy_existing_date_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            history = root / "history"
            diary = history / "diaries" / "diary_2026-08-11.txt"
            diary.parent.mkdir(parents=True)
            diary.write_text("昨天的日记", encoding="utf-8")
            existing = history / "events" / "child_events" / "2026-08-11"
            existing.mkdir(parents=True)

            with self.assertRaises(PublicationConflict):
                run_daily_child_event(
                    "2026-08-11",
                    history,
                    root / "work",
                    generator=lambda *_args: valid_response(),
                )


class MojibakeCleanupTests(unittest.TestCase):
    def _make_pair(self, root: Path, *, same: bool) -> Path:
        day = root / "2026-08-01"
        good = day / EVENT_DIRECTORY_NAME
        bad = day / MOJIBAKE_EVENT_DIRECTORY_NAME
        good.mkdir(parents=True)
        bad.mkdir()
        (good / "event.txt").write_text("正确内容", encoding="utf-8")
        (bad / "event.txt").write_text(
            "正确内容" if same else "不同内容", encoding="utf-8"
        )
        return bad

    def test_exact_duplicate_is_removed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bad = self._make_pair(root, same=True)

            result = audit_mojibake_directories(root, remove_exact_duplicates=True)

            self.assertFalse(bad.exists())
            self.assertEqual(result["records"][0]["status"], "removed_exact_duplicate")

    def test_different_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bad = self._make_pair(root, same=False)

            result = audit_mojibake_directories(root, remove_exact_duplicates=True)

            self.assertTrue(bad.exists())
            self.assertEqual(result["records"][0]["status"], "not_exact_duplicate")


if __name__ == "__main__":
    unittest.main()
