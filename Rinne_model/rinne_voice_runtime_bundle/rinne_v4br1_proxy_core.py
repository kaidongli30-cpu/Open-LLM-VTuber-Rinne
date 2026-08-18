from __future__ import annotations

import argparse
import io
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import requests
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response


V4_PARAMETERS: dict[str, Any] = {
    "top_k": 10,
    "top_p": 0.9,
    "temperature": 0.8,
    "text_split_method": "cut0",
    "batch_size": 1,
    "split_bucket": False,
    "speed_factor": 0.97,
    "seed": 20260727,
    "media_type": "wav",
    "streaming_mode": False,
    "parallel_infer": False,
    "repetition_penalty": 1.15,
    "sample_steps": 32,
}

SILENCE_THRESHOLD_DBFS = -50
INTERNAL_SILENCE_CAP_SECONDS = 0.6
TRAILING_SILENCE_CAP_SECONDS = 1.0

APP = FastAPI(title="Rinne V4-B-R1 hybrid TTS proxy")
BACKENDS = {"v2": "http://127.0.0.1:9882", "v4": "http://127.0.0.1:9883"}
EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rinne-tts")


def port_open(url: str) -> bool:
    host_port = url.removeprefix("http://").split("/", 1)[0]
    host, raw_port = host_port.rsplit(":", 1)
    try:
        with socket.create_connection((host, int(raw_port)), timeout=1):
            return True
    except OSError:
        return False


def call_backend(name: str, params: dict[str, Any]) -> requests.Response:
    return requests.get(
        f"{BACKENDS[name]}/tts",
        params=params,
        timeout=900,
    )


def wav_data(raw: bytes) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(
        io.BytesIO(raw),
        dtype="float32",
        always_2d=True,
    )
    if len(audio) == 0 or sample_rate <= 0:
        raise ValueError("后端返回了空 WAV")
    return audio, sample_rate


def wav_seconds(raw: bytes) -> float:
    audio, sample_rate = wav_data(raw)
    return len(audio) / sample_rate


def visible_units(text: str) -> int:
    return len(
        re.sub(r"[\s、。！？…「」『』（）・―—,.!?/:;／“”\"']", "", text)
    )


def limit_generated_silence(raw: bytes) -> tuple[bytes, list[dict[str, Any]]]:
    """Apply the reviewed R1 rule without touching leading silence."""
    audio, sample_rate = wav_data(raw)
    frame_samples = max(1, round(sample_rate * 0.01))
    frame_count = (len(audio) + frame_samples - 1) // frame_samples
    padding = frame_count * frame_samples - len(audio)
    padded = np.pad(audio, ((0, padding), (0, 0)))
    framed = padded.reshape(frame_count, frame_samples, audio.shape[1])
    rms = np.sqrt(np.mean(np.square(framed), axis=(1, 2)) + 1e-12)
    voiced = rms > 10 ** (SILENCE_THRESHOLD_DBFS / 20)

    removals: list[tuple[int, int]] = []
    changes: list[dict[str, Any]] = []
    frame_index = 0
    while frame_index < frame_count:
        if voiced[frame_index]:
            frame_index += 1
            continue

        run_end_frame = frame_index + 1
        while run_end_frame < frame_count and not voiced[run_end_frame]:
            run_end_frame += 1

        run_start = min(frame_index * frame_samples, len(audio))
        run_end = min(run_end_frame * frame_samples, len(audio))
        run_samples = max(0, run_end - run_start)
        is_leading = frame_index == 0
        is_trailing = run_end_frame == frame_count
        cap_seconds = (
            TRAILING_SILENCE_CAP_SECONDS
            if is_trailing
            else INTERNAL_SILENCE_CAP_SECONDS
        )
        cap_samples = round(sample_rate * cap_seconds)

        if not is_leading and run_samples > cap_samples:
            if is_trailing:
                remove_start = run_start + cap_samples
                remove_end = run_end
                kind = "trailing"
            else:
                keep_left = cap_samples // 2
                keep_right = cap_samples - keep_left
                remove_start = run_start + keep_left
                remove_end = run_end - keep_right
                kind = "internal"
            if remove_end > remove_start:
                removals.append((remove_start, remove_end))
                changes.append(
                    {
                        "kind": kind,
                        "before_seconds": round(run_samples / sample_rate, 3),
                        "after_seconds": cap_seconds,
                    }
                )
        frame_index = run_end_frame

    if not removals:
        return raw, changes

    pieces: list[np.ndarray] = []
    cursor = 0
    for remove_start, remove_end in removals:
        pieces.append(audio[cursor:remove_start])
        cursor = remove_end
    pieces.append(audio[cursor:])
    processed = np.concatenate(pieces, axis=0)
    output = io.BytesIO()
    sf.write(output, processed, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue(), changes


def obvious_v4_failure(text: str, v2_raw: bytes, v4_raw: bytes) -> tuple[bool, str]:
    v2_seconds = wav_seconds(v2_raw)
    v4_seconds = wav_seconds(v4_raw)
    ratio = v4_seconds / v2_seconds if v2_seconds else 0.0
    units = visible_units(text)
    units_per_second = units / v4_seconds if v4_seconds else 999.0
    failed = (
        v4_seconds < 0.55
        or (
            v2_seconds - v4_seconds > 2.0
            and ratio < 0.70
            and units_per_second > 6.6
        )
    )
    detail = (
        f"v2={v2_seconds:.3f}s,v4={v4_seconds:.3f}s,"
        f"ratio={ratio:.3f},units_per_second={units_per_second:.3f}"
    )
    return failed, detail


def audio_response(
    raw: bytes,
    source: str,
    reason: str,
    silence_changes: int = 0,
) -> Response:
    return Response(
        raw,
        media_type="audio/wav",
        headers={
            "X-Rinne-Voice-Source": source,
            "X-Rinne-Fallback-Reason": reason[:500],
            "X-Rinne-Silence-Changes": str(silence_changes),
        },
    )


@APP.get("/health")
def health() -> JSONResponse:
    state = {name: port_open(url) for name, url in BACKENDS.items()}
    return JSONResponse(
        {"status": "ok" if all(state.values()) else "degraded", "backends": state},
        status_code=200 if all(state.values()) else 503,
    )


@APP.get("/tts")
async def tts(request: Request) -> Response:
    incoming: dict[str, Any] = dict(request.query_params)
    text = str(incoming.get("text", ""))
    if not text.strip():
        return JSONResponse({"message": "text is required"}, status_code=400)

    v2_params = dict(incoming)
    v2_params["media_type"] = "wav"
    v2_params["streaming_mode"] = False
    v4_params = {**incoming, **V4_PARAMETERS}

    v2_future = EXECUTOR.submit(call_backend, "v2", v2_params)
    v4_future = EXECUTOR.submit(call_backend, "v4", v4_params)
    v2_response = v2_future.result()
    v4_response = v4_future.result()
    v2_ok = v2_response.status_code == 200 and len(v2_response.content) >= 44
    v4_ok = v4_response.status_code == 200 and len(v4_response.content) >= 44

    if not v4_ok and v2_ok:
        return audio_response(
            v2_response.content,
            "v2-fallback",
            f"V4 backend failed with HTTP {v4_response.status_code}",
        )
    if not v2_ok and v4_ok:
        processed, changes = limit_generated_silence(v4_response.content)
        return audio_response(
            processed,
            "v4-b-r1",
            f"V2 validation unavailable (HTTP {v2_response.status_code})",
            len(changes),
        )
    if not v2_ok and not v4_ok:
        return JSONResponse(
            {
                "message": "both TTS backends failed",
                "v2_status": v2_response.status_code,
                "v4_status": v4_response.status_code,
                "v2_detail": v2_response.text[:1000],
                "v4_detail": v4_response.text[:1000],
            },
            status_code=502,
        )

    processed_v4, changes = limit_generated_silence(v4_response.content)
    failed, detail = obvious_v4_failure(text, v2_response.content, processed_v4)
    if failed:
        print(f"[V2 fallback] {text!r} {detail}", flush=True)
        return audio_response(
            v2_response.content,
            "v2-fallback",
            f"obvious omission check: {detail}",
        )

    print(f"[V4-B-R1] {text!r} {detail}; silence_changes={len(changes)}", flush=True)
    return audio_response(
        processed_v4,
        "v4-b-r1",
        f"passed omission check: {detail}",
        len(changes),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9880)
    parser.add_argument("--v2-url", default=BACKENDS["v2"])
    parser.add_argument("--v4-url", default=BACKENDS["v4"])
    args = parser.parse_args()
    BACKENDS["v2"] = args.v2_url.rstrip("/")
    BACKENDS["v4"] = args.v4_url.rstrip("/")
    uvicorn.run(APP, host=args.host, port=args.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
