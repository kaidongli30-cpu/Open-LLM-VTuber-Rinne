from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.open_llm_vtuber.memory.child_event_search import ChildEventSearchTools


def _event_file(root: Path, date_value: str, name: str, summary: str) -> Path:
    directory = root / date_value / "事件TXT"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{date_value}_{name}.txt"
    path.write_text(
        "\n".join(
            [
                "记忆类型：事件",
                f"事件名称：{name}",
                f"时间：{date_value.replace('-', '年', 1).replace('-', '月', 1)}日",
                "",
                summary,
                "",
                "来源：",
                f"日记：chat_history/rinne_01/diaries/diary_{date_value}.txt",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


class ChildEventSearchToolTests(unittest.TestCase):
    def test_parser_indexes_only_filename_title_and_summary_for_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "child_events"
            _event_file(root, "2026-07-11", "安装内存条", "掉下一颗小电阻。")
            tools = ChildEventSearchTools(
                root,
                model_cache_dir=base / "models",
                index_cache_dir=base / "index",
            )
            self.assertEqual(tools.candidate_count, 1)
            record = next(iter(tools.records.values()))
            self.assertIn("安装内存条", record.search_text)
            self.assertIn("小电阻", record.search_text)
            self.assertNotIn("chat_history", record.search_text)
            self.assertEqual(tools.search_keyword("小电阻")[0].candidate_id, record.candidate_id)

    def test_answer_key_only_references_existing_published_events(self):
        root = (
            Path(__file__).resolve().parents[1]
            / "chat_history"
            / "rinne_01"
            / "events"
            / "child_events"
        )
        with tempfile.TemporaryDirectory() as temporary:
            tools = ChildEventSearchTools(
                root,
                model_cache_dir=temporary,
                index_cache_dir=temporary,
            )
        key = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "docs"
                / "memory_13_child_event_exam_answer_key.json"
            ).read_text(encoding="utf-8")
        )
        missing: list[str] = []
        for item in key["questions"].values():
            ids = [
                *item.get("direct_event_ids", []),
                *item.get("proxy_event_ids", []),
            ]
            for group in item.get("coverage_groups", []):
                ids.extend(group.get("event_ids", []))
            missing.extend(item for item in ids if item not in tools.records)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
