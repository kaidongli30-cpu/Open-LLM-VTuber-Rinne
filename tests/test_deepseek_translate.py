import unittest
from unittest.mock import patch

from open_llm_vtuber.translate.deepseek_translate import DeepSeekTranslate


class _FakeResponse:
    def __init__(self, data, status=200):
        self._data = data
        self.status = status

    async def json(self):
        return self._data

    async def text(self):
        return "error"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _FakeSession:
    def __init__(self, responses, captured_payloads):
        self._responses = responses
        self._captured_payloads = captured_payloads

    def post(self, _url, json, headers):
        self._captured_payloads.append(json)
        return self._responses.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class DeepSeekTranslateTests(unittest.IsolatedAsyncioTestCase):
    async def test_disables_thinking_and_retries_empty_content_once(self):
        engine = DeepSeekTranslate({"api_key": ""})
        responses = [
            _FakeResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": ""},
                        }
                    ],
                    "usage": {"completion_tokens": 512},
                }
            ),
            _FakeResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "翻訳結果"},
                        }
                    ],
                    "usage": {"completion_tokens": 12},
                }
            ),
        ]
        captured_payloads = []

        with patch(
            "open_llm_vtuber.translate.deepseek_translate.aiohttp.ClientSession",
            side_effect=lambda: _FakeSession(responses, captured_payloads),
        ):
            result = await engine._async_translate("翻译这句话")

        self.assertEqual(result, "翻訳結果")
        self.assertEqual(len(captured_payloads), 2)
        self.assertTrue(
            all(
                payload["thinking"] == {"type": "disabled"}
                for payload in captured_payloads
            )
        )

    async def test_raises_after_two_empty_results(self):
        engine = DeepSeekTranslate({"api_key": ""})
        empty_data = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": ""},
                }
            ],
            "usage": {"completion_tokens": 512},
        }
        responses = [_FakeResponse(empty_data), _FakeResponse(empty_data)]

        with patch(
            "open_llm_vtuber.translate.deepseek_translate.aiohttp.ClientSession",
            side_effect=lambda: _FakeSession(responses, []),
        ):
            with self.assertRaisesRegex(Exception, "empty content twice"):
                await engine._async_translate("翻译这句话")


if __name__ == "__main__":
    unittest.main()
