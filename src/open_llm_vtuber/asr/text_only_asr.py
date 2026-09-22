"""No-model ASR option for a first-time text-only installation."""

from __future__ import annotations

import numpy as np

from .asr_interface import ASRInterface


class TextOnlyASR(ASRInterface):
    def transcribe_np(self, audio: np.ndarray) -> str:
        raise RuntimeError("当前配置为仅文字输入；如需麦克风，请配置语音识别模型")
