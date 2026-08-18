import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from open_llm_vtuber.config_manager.tts_preprocessor import TranslatorConfig
from open_llm_vtuber.conversations.conversation_utils import (
    _translate_text_if_needed,
)
from open_llm_vtuber.translate.ollama_local_translate import OllamaLocalTranslate
from open_llm_vtuber.translate.translate_factory import TranslateFactory


def _response(content: str) -> Mock:
    response = Mock()
    response.json.return_value = {"message": {"content": content}}
    response.raise_for_status.return_value = None
    return response


class OllamaLocalTranslateTests(unittest.TestCase):
    def _engine(self, **overrides) -> OllamaLocalTranslate:
        config = {
            "system_prompt": "只输出日语。",
            "glossary": {"凛祢": "りんね", "美九": "みく"},
            "timeout_seconds": 3,
            "max_validation_attempts": 2,
        }
        config.update(overrides)
        return OllamaLocalTranslate(config)

    @patch("open_llm_vtuber.translate.ollama_local_translate.requests.post")
    def test_injects_only_glossary_terms_present_in_source(self, post: Mock):
        post.return_value = _response("りんねはここにいるよ。")

        result = self._engine().translate("凛祢在这里。")

        self.assertEqual(result, "りんねはここにいるよ。")
        payload = post.call_args.kwargs["json"]
        system_prompt = payload["messages"][0]["content"]
        self.assertIn("凛祢→りんね", system_prompt)
        self.assertNotIn("美九→みく", system_prompt)
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["num_ctx"], 2048)

    @patch("open_llm_vtuber.translate.ollama_local_translate.requests.post")
    def test_retries_simplified_chinese_residual_once(self, post: Mock):
        post.side_effect = [_response("这句话没有翻译。"), _response("翻訳しました。")]

        result = self._engine().translate("翻译这句话。")

        self.assertEqual(result, "翻訳しました。")
        self.assertEqual(post.call_count, 2)
        retry_prompt = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("上一次结果未通过输出检查", retry_prompt)

    @patch("open_llm_vtuber.translate.ollama_local_translate.requests.post")
    def test_retries_multi_line_explanation_once(self, post: Mock):
        post.side_effect = [
            _response("次の問題だよ。\n『次』は日本語です。"),
            _response("次の問題だよ。"),
        ]

        result = self._engine().translate("下一题来吧。")

        self.assertEqual(result, "次の問題だよ。")
        self.assertEqual(post.call_count, 2)

    @patch("open_llm_vtuber.translate.ollama_local_translate.requests.post")
    def test_raises_after_two_invalid_outputs(self, post: Mock):
        post.side_effect = [_response(""), _response("还是中文。")]

        with self.assertRaisesRegex(ValueError, "failed output validation"):
            self._engine().translate("请翻译。")

    @patch("open_llm_vtuber.translate.ollama_local_translate.requests.post")
    def test_removes_middle_dot_for_current_tts_runtime(self, post: Mock):
        post.return_value = _response("デート・ア・ライブオンリー")

        result = self._engine().translate("约战Only")

        self.assertEqual(result, "デートアライブオンリー")

    def test_factory_and_config_accept_ollama_local(self):
        config = TranslatorConfig(
            translate_audio=True,
            translate_provider="ollama_local",
            ollama_local={},
        )
        engine = TranslateFactory.get_translator(
            config.translate_provider,
            config.ollama_local.model_dump(),
        )
        self.assertIsInstance(engine, OllamaLocalTranslate)

    def test_active_project_config_selects_ollama_local(self):
        project_root = Path(__file__).resolve().parents[1]
        config_data = yaml.safe_load(
            (project_root / "conf.yaml").read_text(encoding="utf-8")
        )
        translator_data = config_data["character_config"]["tts_preprocessor_config"][
            "translator_config"
        ]

        config = TranslatorConfig(**translator_data)

        self.assertTrue(config.translate_audio)
        self.assertEqual(config.translate_provider, "ollama_local")


class TranslationFailureSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_translation_failure_suppresses_tts_instead_of_returning_chinese(
        self,
    ):
        engine = Mock()
        engine.translate.side_effect = TimeoutError("local translator timed out")

        result = await _translate_text_if_needed("不能送进日语TTS。", engine)

        self.assertEqual(result, "")

    async def test_empty_translation_suppresses_tts(self):
        engine = Mock()
        engine.translate.return_value = ""

        result = await _translate_text_if_needed("不能送进日语TTS。", engine)

        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
