import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from open_llm_vtuber.agent.output_types import Actions
from open_llm_vtuber.conversations.conversation_utils import (
    _select_tts_reference_emotion,
)
from open_llm_vtuber.conversations.tts_manager import (
    DEFAULT_TTS_REFERENCE,
    TTSTaskManager,
)
from open_llm_vtuber.live2d_model import Live2dModel
from open_llm_vtuber.tts.gpt_sovits_tts import TTSEngine


class EmotionReferenceRoutingTests(unittest.TestCase):
    def test_rinne_accepts_surprised_as_surprise_alias(self):
        live2d = Live2dModel("rinne")

        expressions = live2d.extract_emotion("[surprised] えっ、本当なの？")

        self.assertEqual(expressions, [7])
        self.assertEqual(
            _select_tts_reference_emotion(
                Actions(expressions=expressions),
                live2d,
            ),
            "surprise",
        )

    def test_live2d_expression_selects_only_three_special_references(self):
        live2d = SimpleNamespace(
            emo_map={
                "surprise": 7,
                "shy": 6,
                "angry": 1,
                "happy": 4,
                "sad": 5,
            }
        )

        self.assertEqual(
            _select_tts_reference_emotion(Actions(expressions=[7]), live2d),
            "surprise",
        )
        self.assertEqual(
            _select_tts_reference_emotion(Actions(expressions=[6]), live2d),
            "shy",
        )
        self.assertEqual(
            _select_tts_reference_emotion(Actions(expressions=[1]), live2d),
            "angry",
        )
        self.assertEqual(
            _select_tts_reference_emotion(Actions(expressions=[4]), live2d),
            DEFAULT_TTS_REFERENCE,
        )
        self.assertIsNone(_select_tts_reference_emotion(Actions(), live2d))

    def test_reference_emotion_persists_across_chunks_and_can_reset(self):
        manager = TTSTaskManager()

        first = manager.resolve_reference_emotion("shy")
        inherited_chunk = manager.resolve_reference_emotion(None)
        reset = manager.resolve_reference_emotion(DEFAULT_TTS_REFERENCE)

        self.assertEqual(first, "shy")
        self.assertEqual(inherited_chunk, "shy")
        self.assertIsNone(reset)

        manager.clear()
        self.assertIsNone(manager._active_reference_emotion)

    def test_every_shy_expression_uses_the_same_reference(self):
        manager = TTSTaskManager()
        resolved = [
            manager.resolve_reference_emotion(item)
            for item in [
                DEFAULT_TTS_REFERENCE,
                "shy",
                None,
                "shy",
                "shy",
            ]
        ]
        self.assertEqual(
            resolved,
            [
                None,
                "shy",
                "shy",
                "shy",
                "shy",
            ],
        )

        interrupted = TTSTaskManager()
        resolved_after_interruptions = [
            interrupted.resolve_reference_emotion(item)
            for item in ["surprise", "shy", "angry", "shy"]
        ]
        self.assertEqual(
            resolved_after_interruptions,
            [
                "surprise",
                "shy",
                "angry",
                "shy",
            ],
        )

    def test_new_response_does_not_inherit_previous_shy(self):
        manager = TTSTaskManager()
        self.assertEqual(manager.resolve_reference_emotion("shy"), "shy")

        manager.start_response()

        self.assertIsNone(manager.resolve_reference_emotion(None))
        self.assertEqual(manager.resolve_reference_emotion("shy"), "shy")

    def test_special_reference_failure_retries_with_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            special_audio = Path(temp_dir) / "shy.wav"
            special_audio.write_bytes(b"RIFF")
            engine = TTSEngine(
                ref_audio_path="default.wav",
                prompt_lang="ja",
                prompt_text="default prompt",
                emotion_references={
                    "shy": {
                        "ref_audio_path": str(special_audio),
                        "prompt_lang": "ja",
                        "prompt_text": "shy prompt",
                    }
                },
            )
            failed = SimpleNamespace(status_code=500, content=b"")
            succeeded = SimpleNamespace(status_code=200, content=b"audio")
            engine.generate_cache_file_name = Mock(return_value="result.wav")
            engine._request_audio = Mock(side_effect=[failed, succeeded])
            engine._save_audio = Mock(return_value="result.wav")

            self.assertEqual(engine.generate_audio("test", emotion="shy"), "result.wav")
            first_reference = engine._request_audio.call_args_list[0].args[1]
            fallback_reference = engine._request_audio.call_args_list[1].args[1]
            self.assertEqual(first_reference["ref_audio_path"], str(special_audio))
            self.assertEqual(fallback_reference["ref_audio_path"], "default.wav")


    def test_shy_reference_carries_aux_audio_and_logs_actual_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            default_audio = Path(temp_dir) / "default.wav"
            aux_audio = Path(temp_dir) / "shy.wav"
            default_audio.write_bytes(b"RIFF")
            aux_audio.write_bytes(b"RIFF")
            engine = TTSEngine(
                ref_audio_path=str(default_audio),
                prompt_lang="ja",
                prompt_text="default prompt",
                emotion_references={
                    "shy": {
                        "ref_audio_path": str(default_audio),
                        "aux_ref_audio_paths": [str(aux_audio)],
                        "prompt_lang": "ja",
                        "prompt_text": "default prompt",
                    }
                },
            )

            reference, selected = engine._resolve_reference("shy")
            self.assertEqual(selected, "shy")
            self.assertEqual(reference["ref_audio_path"], str(default_audio))
            self.assertEqual(reference["aux_ref_audio_paths"], [str(aux_audio)])

            response = SimpleNamespace(status_code=200, content=b"audio")
            with patch(
                "open_llm_vtuber.tts.gpt_sovits_tts.requests.post",
                return_value=response,
            ) as post_request:
                engine._request_audio("test", reference)

            request_json = post_request.call_args.kwargs["json"]
            self.assertEqual(
                request_json["aux_ref_audio_paths"],
                [str(aux_audio)],
            )
            self.assertFalse(request_json["streaming_mode"])

            engine.generate_cache_file_name = Mock(return_value="result.wav")
            engine._request_audio = Mock(return_value=response)
            engine._save_audio = Mock(return_value="result.wav")
            with patch(
                "open_llm_vtuber.tts.gpt_sovits_tts.logger.debug"
            ) as log_reference:
                engine.generate_audio("test", emotion="shy")

            log_message = log_reference.call_args.args[0]
            self.assertIn("emotion=shy", log_message)
            self.assertIn(f"main={default_audio}", log_message)
            self.assertIn(str(aux_audio), log_message)


if __name__ == "__main__":
    unittest.main()
