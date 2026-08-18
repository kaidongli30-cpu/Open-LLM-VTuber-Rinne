from __future__ import annotations

import importlib.util
from pathlib import Path

import requests


HERE = Path(__file__).resolve().parent
CORE_PATH = HERE / "rinne_v4br1_proxy_core.py"


def load_core():
    spec = importlib.util.spec_from_file_location("rinne_v4br1_proxy_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 V4-B-R1 代理主体：{CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = load_core()
original_call_backend = core.call_backend


def guarded_call_backend(name: str, params: dict):
    """Turn a disconnected backend into an HTTP-like failure so the other model can win."""
    try:
        return original_call_backend(name, params)
    except requests.RequestException as exc:
        response = requests.Response()
        response.status_code = 599
        response._content = f"{type(exc).__name__}: {exc}".encode(
            "utf-8", errors="replace"
        )
        response.headers["Content-Type"] = "text/plain; charset=utf-8"
        return response


core.call_backend = guarded_call_backend


if __name__ == "__main__":
    raise SystemExit(core.main())
