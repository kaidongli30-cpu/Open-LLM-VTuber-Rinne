from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import requests
import yaml


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_GSV_PARENT = PROJECT_ROOT / "GPT-SoVITS-v2pro-20250604"
DEFAULT_GSV_NESTED = DEFAULT_GSV_PARENT / "GPT-SoVITS-v2pro-20250604"
DEFAULT_GSV_ROOT = (
    DEFAULT_GSV_PARENT
    if (DEFAULT_GSV_PARENT / "runtime" / "python.exe").exists()
    else DEFAULT_GSV_NESTED
)
GSV_ROOT = Path(os.environ.get("RINNE_GPT_SOVITS_ROOT", DEFAULT_GSV_ROOT)).expanduser()
RUNTIME_PYTHON = GSV_ROOT / "runtime" / "python.exe"
API_SCRIPT = GSV_ROOT / "api_v2.py"
BASE_TTS_CONFIG = GSV_ROOT / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
PROJECT_CONFIG = PROJECT_ROOT / "conf.yaml"
STATE_PATH = HERE / "runtime_state.json"
CONFIG_DIR = HERE / "configs"
LOG_DIR = HERE / "logs"
BACKUP_DIR = HERE / "配置备份"
PROXY_SCRIPT = HERE / "rinne_v4br1_proxy.py"
EMOTION_REFERENCES = (
    HERE / "emotion_references" / "rinne_default.wav",
    HERE / "emotion_references" / "surprise_RINNE_JP_000093.wav",
    HERE / "emotion_references" / "shy_RINNE_JP_001192.wav",
    HERE / "emotion_references" / "angry_RINNE_JP_000533.wav",
)

V2_GPT = GSV_ROOT / "GPT_weights_v2" / "rinne_e15.ckpt"
V2_SOVITS = GSV_ROOT / "SoVITS_weights_v2" / "rinne_e8_s456.pth"
V4_GPT = GSV_ROOT / "GPT_weights_v4" / "rinne_v4-e10.ckpt"
V4_SOVITS = GSV_ROOT / "SoVITS_weights_v4" / "rinne_v4_e1_s1297_l32.pth"
BERT = GSV_ROOT / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"
HUBERT = GSV_ROOT / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"
V4_DEVICE = os.environ.get("RINNE_TTS_DEVICE", "cuda").strip() or "cuda"
V4_IS_HALF = os.environ.get("RINNE_TTS_HALF", "true").strip().lower() not in {
    "0",
    "false",
    "no",
}


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def validate_files(*, include_v4: bool = False) -> dict[str, Any]:
    required = [
        RUNTIME_PYTHON,
        API_SCRIPT,
        BASE_TTS_CONFIG,
        PROJECT_CONFIG,
        V2_GPT,
        V2_SOVITS,
        BERT,
        HUBERT,
        *EMOTION_REFERENCES,
    ]
    if include_v4:
        required.extend((PROXY_SCRIPT, V4_GPT, V4_SOVITS))
    missing = [str(path) for path in required if not path.exists()]

    result = {
        "ok": not missing,
        "gpt_sovits_root": str(GSV_ROOT),
        "required_paths": len(required),
        "missing_paths": missing,
        "v2_weights": [str(V2_GPT), str(V2_SOVITS)],
    }
    if include_v4:
        result["v4br1_weights"] = [str(V4_GPT), str(V4_SOVITS)]
    return result


def make_config(path: Path, version: str) -> None:
    with BASE_TTS_CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"配置格式异常：{BASE_TTS_CONFIG}")
    if version == "v2":
        custom = {
            "bert_base_path": str(BERT),
            "cnhuhbert_base_path": str(HUBERT),
            "device": "cpu",
            "is_half": False,
            "t2s_weights_path": str(V2_GPT),
            "version": "v2",
            "vits_weights_path": str(V2_SOVITS),
        }
    elif version == "v4":
        custom = {
            "bert_base_path": str(BERT),
            "cnhuhbert_base_path": str(HUBERT),
            "device": V4_DEVICE,
            "is_half": V4_IS_HALF,
            "t2s_weights_path": str(V4_GPT),
            "version": "v4",
            "vits_weights_path": str(V4_SOVITS),
        }
    else:
        raise ValueError(version)
    config["custom"] = custom
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def managed_command_matches(pid: int, expected: str) -> bool:
    try:
        process = psutil.Process(pid)
        command = " ".join(process.cmdline()).lower()
        return expected.lower() in command
    except (psutil.Error, OSError):
        return False


def load_state() -> dict[str, Any]:
    if not STATE_PATH.is_file():
        return {}
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return state if isinstance(state, dict) else {}


def stop_managed() -> None:
    state = load_state()
    records = state.get("processes", [])
    for record in reversed(records if isinstance(records, list) else []):
        pid = int(record.get("pid", -1))
        expected = str(record.get("expected", ""))
        if pid <= 0 or not expected or not managed_command_matches(pid, expected):
            continue
        try:
            process = psutil.Process(pid)
            children = process.children(recursive=True)
            for child in children:
                child.terminate()
            process.terminate()
            _, alive = psutil.wait_procs([*children, process], timeout=10)
            for remaining in alive:
                remaining.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if STATE_PATH.exists():
        STATE_PATH.unlink()


def spawn(
    name: str,
    command: list[str],
    expected: str,
    records: list[dict[str, Any]],
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_path = LOG_DIR / f"{timestamp}_{name}.log"
    stderr_path = LOG_DIR / f"{timestamp}_{name}.error.log"
    stdout = stdout_path.open("ab")
    stderr = stderr_path.open("ab")
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen(
        command,
        cwd=GSV_ROOT,
        stdout=stdout,
        stderr=stderr,
        creationflags=flags,
    )
    stdout.close()
    stderr.close()
    records.append(
        {
            "name": name,
            "pid": process.pid,
            "expected": expected,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
    )


def wait_port(port: int, timeout: float = 240.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_open(port):
            return
        time.sleep(0.5)
    raise TimeoutError(f"端口 {port} 在 {timeout:.0f} 秒内没有就绪")


def backup_and_set_first_response(enabled: bool) -> Path | None:
    with PROJECT_CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    agent_config = config["character_config"]["agent_config"]
    selected = agent_config["conversation_agent_choice"]
    selected_config = agent_config["agent_settings"][selected]
    if bool(selected_config.get("faster_first_response")) == enabled:
        return None
    selected_config["faster_first_response"] = enabled

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_DIR / f"conf_切换前_{timestamp}.yaml"
    backup.write_bytes(PROJECT_CONFIG.read_bytes())
    PROJECT_CONFIG.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return backup


def ensure_ports_free(ports: tuple[int, ...]) -> None:
    occupied = [port for port in ports if port_open(port)]
    if occupied:
        raise RuntimeError(
            "以下端口仍被其他程序占用："
            + ", ".join(str(port) for port in occupied)
            + "。请先关闭旧的 GPT-SoVITS API 窗口，再重新运行切换 BAT。"
        )


def save_state(mode: str, records: list[dict[str, Any]]) -> None:
    STATE_PATH.write_text(
        json.dumps(
            {
                "mode": mode,
                "started_at": dt.datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
                "processes": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def switch(mode: str) -> None:
    validation = validate_files(include_v4=mode == "v4br1")
    if not validation["ok"]:
        raise RuntimeError(
            "保留文件检查未通过：\n"
            + json.dumps(validation, ensure_ascii=False, indent=2)
        )

    stop_managed()
    backup: Path | None = None
    records: list[dict[str, Any]] = []
    try:
        if mode == "v2":
            ensure_ports_free((9880,))
            v2_config = CONFIG_DIR / "v2.yaml"
            make_config(v2_config, "v2")
            spawn(
                "v2_api",
                [
                    str(RUNTIME_PYTHON),
                    str(API_SCRIPT),
                    "-a",
                    "127.0.0.1",
                    "-p",
                    "9880",
                    "-c",
                    str(v2_config),
                ],
                "api_v2.py",
                records,
            )
            save_state(mode, records)
            wait_port(9880)
            backup = backup_and_set_first_response(True)
        elif mode == "v4br1":
            ensure_ports_free((9880, 9882, 9883))
            v2_config = CONFIG_DIR / "v2_fallback.yaml"
            v4_config = CONFIG_DIR / "v4br1.yaml"
            make_config(v2_config, "v2")
            make_config(v4_config, "v4")
            spawn(
                "v2_fallback_api",
                [
                    str(RUNTIME_PYTHON),
                    str(API_SCRIPT),
                    "-a",
                    "127.0.0.1",
                    "-p",
                    "9882",
                    "-c",
                    str(v2_config),
                ],
                "-p 9882",
                records,
            )
            spawn(
                "v4br1_api",
                [
                    str(RUNTIME_PYTHON),
                    str(API_SCRIPT),
                    "-a",
                    "127.0.0.1",
                    "-p",
                    "9883",
                    "-c",
                    str(v4_config),
                ],
                "-p 9883",
                records,
            )
            save_state(mode, records)
            wait_port(9882)
            wait_port(9883)
            spawn(
                "v4br1_proxy",
                [
                    str(RUNTIME_PYTHON),
                    str(PROXY_SCRIPT),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "9880",
                    "--v2-url",
                    "http://127.0.0.1:9882",
                    "--v4-url",
                    "http://127.0.0.1:9883",
                ],
                "rinne_v4br1_proxy.py",
                records,
            )
            save_state(mode, records)
            wait_port(9880)
            response = requests.get("http://127.0.0.1:9880/health", timeout=5)
            response.raise_for_status()
            backup = backup_and_set_first_response(False)
        else:
            raise ValueError(mode)
    except Exception:
        save_state(mode, records)
        stop_managed()
        raise

    print(f"切换完成：{mode}")
    if backup is not None:
        print(f"项目配置备份：{backup}")
    if mode == "v4br1":
        print("桌宠继续访问 127.0.0.1:9880；该端口现在由 V4-B-R1 混合代理提供。")
    else:
        print("桌宠继续访问 127.0.0.1:9880；该端口现在由旧 V2 提供。")
    print("切换后请重启桌宠，使首句分段开关重新载入。")


def status() -> dict[str, Any]:
    state = load_state()
    processes = []
    for record in state.get("processes", []) if isinstance(state, dict) else []:
        pid = int(record.get("pid", -1))
        expected = str(record.get("expected", ""))
        processes.append(
            {
                "name": record.get("name"),
                "pid": pid,
                "running": managed_command_matches(pid, expected),
            }
        )
    return {
        "managed_mode": state.get("mode", "未由本保留包启动"),
        "ports": {str(port): port_open(port) for port in (9880, 9882, 9883)},
        "processes": processes,
        "file_validation": validate_files(),
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("switch-v2", "switch-v4br1", "stop", "status", "validate"),
    )
    args = parser.parse_args()
    if args.command == "switch-v2":
        switch("v2")
    elif args.command == "switch-v4br1":
        switch("v4br1")
    elif args.command == "stop":
        stop_managed()
        print("已停止由本保留包启动的凛祢语音服务。")
    elif args.command == "status":
        print(json.dumps(status(), ensure_ascii=False, indent=2))
    elif args.command == "validate":
        result = validate_files()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
