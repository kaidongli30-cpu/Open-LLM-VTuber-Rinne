import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


MANAGER_PATH = (
    Path(__file__).parents[1]
    / "Rinne_model"
    / "rinne_voice_runtime_bundle"
    / "rinne_voice_manager.py"
)
SPEC = importlib.util.spec_from_file_location("rinne_voice_manager", MANAGER_PATH)
assert SPEC is not None and SPEC.loader is not None
MANAGER = importlib.util.module_from_spec(SPEC)
with patch.dict(
    sys.modules,
    {"psutil": MagicMock(), "requests": MagicMock(), "yaml": MagicMock()},
):
    SPEC.loader.exec_module(MANAGER)


class RinneVoiceManagerValidationTests(unittest.TestCase):
    def test_v2_validation_does_not_require_v4_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            required_names = (
                "runtime_python",
                "api_script",
                "base_tts_config",
                "project_config",
                "v2_gpt",
                "v2_sovits",
                "bert",
                "hubert",
                "reference_default",
            )
            required_paths = {name: root / name for name in required_names}
            for path in required_paths.values():
                path.touch()

            missing_v4_gpt = root / "missing_v4_gpt"
            missing_v4_sovits = root / "missing_v4_sovits"
            missing_proxy = root / "missing_proxy"
            with (
                patch.object(MANAGER, "RUNTIME_PYTHON", required_paths["runtime_python"]),
                patch.object(MANAGER, "API_SCRIPT", required_paths["api_script"]),
                patch.object(
                    MANAGER, "BASE_TTS_CONFIG", required_paths["base_tts_config"]
                ),
                patch.object(MANAGER, "PROJECT_CONFIG", required_paths["project_config"]),
                patch.object(MANAGER, "V2_GPT", required_paths["v2_gpt"]),
                patch.object(MANAGER, "V2_SOVITS", required_paths["v2_sovits"]),
                patch.object(MANAGER, "BERT", required_paths["bert"]),
                patch.object(MANAGER, "HUBERT", required_paths["hubert"]),
                patch.object(
                    MANAGER,
                    "EMOTION_REFERENCES",
                    (required_paths["reference_default"],),
                ),
                patch.object(MANAGER, "V4_GPT", missing_v4_gpt),
                patch.object(MANAGER, "V4_SOVITS", missing_v4_sovits),
                patch.object(MANAGER, "PROXY_SCRIPT", missing_proxy),
            ):
                v2_result = MANAGER.validate_files()
                extended_result = MANAGER.validate_files(include_v4=True)

            self.assertTrue(v2_result["ok"])
            self.assertNotIn("v4br1_weights", v2_result)
            self.assertFalse(extended_result["ok"])
            self.assertEqual(
                set(extended_result["missing_paths"]),
                {str(missing_v4_gpt), str(missing_v4_sovits), str(missing_proxy)},
            )


if __name__ == "__main__":
    unittest.main()
