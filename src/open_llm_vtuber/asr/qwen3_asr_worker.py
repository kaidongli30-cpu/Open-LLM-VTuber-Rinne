from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
import traceback

import numpy as np


_PROTOCOL = "rinne_qwen3_asr_v1"


def _send(event: str, **payload) -> None:
    message = {"protocol": _PROTOCOL, "event": event, **payload}
    # Keep the wire format ASCII-only because Windows child-process pipes can
    # otherwise encode multilingual text with the active system code page.
    print(json.dumps(message, ensure_ascii=True), flush=True)


def _watch_parent(parent_pid: int) -> None:
    import psutil

    try:
        parent = psutil.Process(parent_pid)
    except psutil.Error:
        os._exit(0)
    while True:
        time.sleep(2)
        if not parent.is_running():
            os._exit(0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--parent-pid", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    threading.Thread(target=_watch_parent, args=(args.parent_pid,), daemon=True).start()

    try:
        import torch
        from qwen_asr import Qwen3ASRModel

        dtype = getattr(torch, args.dtype)
        model = Qwen3ASRModel.from_pretrained(
            args.model_path,
            dtype=dtype,
            device_map=args.device,
            max_inference_batch_size=1,
            max_new_tokens=args.max_new_tokens,
        )
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _send("startup_error", error=f"{type(exc).__name__}: {exc}")
        return 1

    _send("ready")
    for raw_line in sys.stdin:
        try:
            request = json.loads(raw_line)
            if request.get("protocol") != _PROTOCOL:
                continue
            event = request.get("event")
            if event == "shutdown":
                return 0
            if event != "transcribe":
                continue

            request_id = str(request.get("id", ""))
            audio = np.frombuffer(
                base64.b64decode(request["audio"]), dtype=np.float32
            ).copy()
            language = request.get("language") or None
            results = model.transcribe(
                audio=(audio, int(request.get("sample_rate", 16000))),
                language=language,
                context=str(request.get("context", "")),
            )
            result = results[0]
            _send(
                "result",
                id=request_id,
                text=str(result.text),
                language=str(result.language),
            )
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _send(
                "error",
                id=str(request.get("id", "")) if "request" in locals() else "",
                error=f"{type(exc).__name__}: {exc}",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
