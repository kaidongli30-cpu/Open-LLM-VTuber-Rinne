from __future__ import annotations

import atexit
import base64
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import uuid

from loguru import logger
import numpy as np

from .asr_interface import ASRInterface
from .terminology import ASRTerminology


_PROTOCOL = "rinne_qwen3_asr_v1"


class VoiceRecognition(ASRInterface):
    """Use Qwen3-ASR through an isolated, persistent local worker process."""

    def __init__(
        self,
        model_path: str = "./models/Qwen3-ASR-0.6B",
        worker_python: str = "./.venv-qwen3-asr/Scripts/python.exe",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        language: str | None = None,
        context: str = "",
        terminology_path: str | None = "./asr_terminology.json",
        max_new_tokens: int = 256,
        startup_timeout_seconds: float = 180.0,
        request_timeout_seconds: float = 120.0,
    ) -> None:
        self.model_path = self._resolve_path(model_path)
        self.worker_python = self._resolve_path(worker_python)
        self.device = device
        self.dtype = dtype
        self.language = self._normalize_language(language)
        self._terminology = ASRTerminology(
            terminology_path,
            engine_name="Qwen3-ASR",
        )
        self.context = self._terminology.build_context(context)
        self.max_new_tokens = max_new_tokens
        self.startup_timeout_seconds = startup_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds

        self._process: subprocess.Popen[str] | None = None
        self._messages: queue.Queue[dict] = queue.Queue()
        self._request_lock = threading.Lock()
        self._closed = False

        self._validate_runtime_paths()
        self._start_worker()
        atexit.register(self.close)

    @staticmethod
    def _resolve_path(value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    @staticmethod
    def _normalize_language(language: str | None) -> str | None:
        normalized = (language or "").strip()
        if not normalized or normalized.lower() == "auto":
            return None
        return normalized

    def _validate_runtime_paths(self) -> None:
        if not self.worker_python.is_file():
            raise FileNotFoundError(
                "Qwen3-ASR 独立环境不存在："
                f"{self.worker_python}。请先运行 scripts/setup_qwen3_asr.ps1。"
            )
        if not self.model_path.is_dir():
            raise FileNotFoundError(
                "Qwen3-ASR 模型目录不存在："
                f"{self.model_path}。请先运行 scripts/setup_qwen3_asr.ps1。"
            )

    def _start_worker(self) -> None:
        worker_script = Path(__file__).with_name("qwen3_asr_worker.py")
        command = [
            str(self.worker_python),
            "-u",
            str(worker_script),
            "--model-path",
            str(self.model_path),
            "--device",
            self.device,
            "--dtype",
            self.dtype,
            "--max-new-tokens",
            str(self.max_new_tokens),
            "--parent-pid",
            str(os.getpid()),
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

        ready = self._wait_for_message(self.startup_timeout_seconds)
        if ready.get("event") != "ready":
            self.close()
            raise RuntimeError(
                f"Qwen3-ASR worker failed during startup: {ready.get('error', ready)}"
            )
        logger.info(
            "Qwen3-ASR ready: model={}, device={}, dtype={}",
            self.model_path.name,
            self.device,
            self.dtype,
        )

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Qwen3-ASR worker stdout: {}", line)
                continue
            if message.get("protocol") == _PROTOCOL:
                self._messages.put(message)
            else:
                logger.debug("Qwen3-ASR worker stdout: {}", line)

    def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for raw_line in process.stderr:
            line = raw_line.rstrip()
            if line:
                logger.debug("Qwen3-ASR worker: {}", line)

    def _wait_for_message(self, timeout: float) -> dict:
        try:
            return self._messages.get(timeout=timeout)
        except queue.Empty as exc:
            process = self._process
            return_code = process.poll() if process is not None else None
            self.close()
            raise TimeoutError(
                "Qwen3-ASR 本地进程等待超时"
                + (f"，退出码={return_code}" if return_code is not None else "")
            ) from exc

    def transcribe_np(self, audio: np.ndarray) -> str:
        if self._closed:
            raise RuntimeError("Qwen3-ASR worker is closed")
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return ""

        request_id = uuid.uuid4().hex
        request = {
            "protocol": _PROTOCOL,
            "event": "transcribe",
            "id": request_id,
            "sample_rate": self.SAMPLE_RATE,
            "language": self.language,
            "context": self.context,
            "audio": base64.b64encode(samples.tobytes()).decode("ascii"),
        }
        with self._request_lock:
            process = self._process
            if process is None or process.stdin is None or process.poll() is not None:
                raise RuntimeError("Qwen3-ASR 本地进程未运行")
            process.stdin.write(json.dumps(request, ensure_ascii=True) + "\n")
            process.stdin.flush()

            while True:
                response = self._wait_for_message(self.request_timeout_seconds)
                if response.get("id") != request_id:
                    logger.warning("Ignoring unmatched Qwen3-ASR worker response")
                    continue
                if response.get("event") == "error":
                    raise RuntimeError(
                        "Qwen3-ASR 识别失败："
                        + str(response.get("error", "unknown error"))
                    )
                text = self._terminology.correct(str(response.get("text", "")).strip())
                detected_language = response.get("language")
                logger.info(
                    "Qwen3-ASR transcription completed: chars={}, language={}",
                    len(text),
                    detected_language or "unknown",
                )
                return text

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.write(
                    json.dumps({"protocol": _PROTOCOL, "event": "shutdown"}) + "\n"
                )
                process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    def __del__(self) -> None:
        self.close()
