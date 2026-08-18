import http.client
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.open_llm_vtuber.config_manager.daily_child_event import (
    DailyChildEventGenerationConfig,
)
from src.open_llm_vtuber.memory import daily_child_event_providers as providers


class DailyChildEventProviderTests(unittest.TestCase):
    def test_incomplete_http_body_is_retryable_connection_error(self):
        response = MagicMock()
        response.__enter__.return_value.read.side_effect = http.client.IncompleteRead(
            b'{"partial":', 20
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            response_path = Path(temp_dir) / "response.json"
            with patch.object(providers.urllib.request, "urlopen", return_value=response):
                with self.assertRaisesRegex(
                    providers.DailyChildEventProviderError,
                    "provider_connection_error:IncompleteRead",
                ):
                    providers._post_json(
                        "https://example.invalid/v1/chat/completions",
                        {"model": "test"},
                        {"Content-Type": "application/json"},
                        30,
                        response_path,
                    )
            self.assertFalse(response_path.exists())

    def test_legacy_default_is_ollama_with_explicit_model(self):
        settings = providers.load_daily_child_event_settings(
            Path("./rinne_daily_child_event_provider_test_2026-08-14/missing.yaml")
        )

        self.assertTrue(settings.enabled)
        self.assertEqual(settings.llm_provider, "ollama_llm")
        self.assertEqual(settings.model, "mistral-small3.2:24b")

    def test_ollama_endpoint_accepts_root_and_v1_urls(self):
        self.assertEqual(
            providers._ollama_generate_url("http://localhost:11434"),
            "http://localhost:11434/api/generate",
        )
        self.assertEqual(
            providers._ollama_generate_url("http://localhost:11434/"),
            "http://localhost:11434/api/generate",
        )
        self.assertEqual(
            providers._ollama_generate_url("http://localhost:11434/v1"),
            "http://localhost:11434/api/generate",
        )

    def test_openai_compatible_request_uses_configured_model_and_hides_key(self):
        settings = DailyChildEventGenerationConfig(
            llm_provider="openai_compatible_llm",
            base_url="https://example.invalid/v1?token=should-not-be-logged",
            llm_api_key="secret-key",
            model="test-model",
        )
        writes = []
        response = {
            "choices": [{"message": {"content": '{"memory_day":"2026-08-11"}'}}]
        }
        with (
            patch.object(providers, "_post_json", return_value=response),
            patch.object(
                providers,
                "_write_json",
                side_effect=lambda path, value: writes.append((path, value)),
            ),
            patch.object(providers, "_write_text"),
        ):
            result = providers._generate_with_openai_compatible(
                "2026-08-11",
                "system",
                "task",
                Path("unused"),
                settings,
                {"type": "object"},
            )

        self.assertEqual(result.text, '{"memory_day":"2026-08-11"}')
        self.assertEqual(result.metadata["model"], "test-model")
        metadata = writes[0][1]
        self.assertNotIn("secret-key", repr(metadata))
        self.assertNotIn("token=should-not-be-logged", repr(metadata))
        request_payload = metadata["request_payload"]
        self.assertEqual(request_payload["model"], "test-model")

    def test_claude_request_uses_same_prompt_contract(self):
        settings = DailyChildEventGenerationConfig(
            llm_provider="claude_llm",
            base_url="https://example.invalid",
            llm_api_key="secret-key",
            model="claude-test",
        )
        writes = []
        response = {"content": [{"type": "text", "text": "{}"}]}
        with (
            patch.object(providers, "_post_json", return_value=response) as post,
            patch.object(
                providers,
                "_write_json",
                side_effect=lambda path, value: writes.append((path, value)),
            ),
            patch.object(providers, "_write_text"),
        ):
            result = providers._generate_with_claude(
                "2026-08-11",
                "system",
                "task",
                Path("unused"),
                settings,
                {"type": "object"},
            )

        self.assertEqual(result.text, "{}")
        self.assertEqual(post.call_args.args[0], "https://example.invalid/v1/messages")
        payload = post.call_args.args[1]
        self.assertEqual(payload["system"], "system")
        self.assertEqual(payload["messages"][0]["content"], "task")
        self.assertNotIn("secret-key", repr(writes[0][1]))

    def test_provider_aliases_share_openai_compatible_adapter(self):
        aliases = {
            "lmstudio_llm",
            "openai_compatible_llm",
            "openai_llm",
            "gemini_llm",
            "zhipu_llm",
            "deepseek_llm",
            "groq_llm",
            "mistral_llm",
        }
        with patch.object(
            providers,
            "_generate_with_openai_compatible",
            return_value="sentinel",
        ) as openai_adapter:
            for alias in aliases:
                settings = DailyChildEventGenerationConfig(llm_provider=alias)
                result = providers.generate_with_provider(
                    "2026-08-11",
                    "system",
                    "task",
                    Path("unused"),
                    settings,
                    {"type": "object"},
                )
                self.assertEqual(result, "sentinel")

        self.assertEqual(openai_adapter.call_count, len(aliases))

    def test_unsupported_provider_is_rejected_by_config(self):
        with self.assertRaises(ValueError):
            DailyChildEventGenerationConfig(llm_provider="local_24b")


if __name__ == "__main__":
    unittest.main()
