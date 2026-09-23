"""Safely onboard an official QQ Bot and start Rinne for one local session.

The QQ credentials returned by the official SDK are passed only to the child
backend process.  This launcher never writes them to a project file, chat
history, command line, or Git-tracked configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import webbrowser
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT_ENV = "RINNE_DATA_ROOT"
QQ_ENABLED_ENV = "RINNE_QQ_ENABLED"
QQ_APP_ID_ENV = "RINNE_QQ_APP_ID"
QQ_CLIENT_SECRET_ENV = "RINNE_QQ_CLIENT_SECRET"
QQ_ALLOWED_OPENIDS_ENV = "RINNE_QQ_ALLOWED_USER_OPENIDS"
PROACTIVE_ENABLED_ENV = "RINNE_PROACTIVE_ENABLED"
PROACTIVE_RECIPIENT_ENV = "RINNE_QQ_PROACTIVE_RECIPIENT_OPENID"


class QQOnboardLauncherError(RuntimeError):
    """Raised for safe, user-actionable launcher failures."""


def _read_windows_user_data_root() -> str:
    if os.name != "nt":
        return ""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _value_type = winreg.QueryValueEx(key, DATA_ROOT_ENV)
    except (FileNotFoundError, OSError):
        return ""
    return str(value).strip()


def prepare_data_root_environment(
    base_environment: Mapping[str, str],
    *,
    user_data_root_reader: Callable[[], str] = _read_windows_user_data_root,
) -> dict[str, str]:
    """Require the verified private-data directory before QQ can start."""

    environment = dict(base_environment)
    raw_root = str(environment.get(DATA_ROOT_ENV, "")).strip()
    if not raw_root:
        raw_root = str(user_data_root_reader()).strip()
    if not raw_root:
        raise QQOnboardLauncherError(
            f"未配置 {DATA_ROOT_ENV}；为避免误用其他聊天数据目录，已拒绝启动。"
        )

    data_root = Path(raw_root).expanduser()
    if not data_root.is_absolute() or data_root.resolve() == Path(
        data_root.anchor
    ).resolve():
        raise QQOnboardLauncherError(
            f"{DATA_ROOT_ENV} 必须是安全的绝对目录，不能是盘符根目录。"
        )
    if not data_root.is_dir():
        raise QQOnboardLauncherError(
            f"{DATA_ROOT_ENV} 指向的目录不存在；已拒绝启动。"
        )
    environment[DATA_ROOT_ENV] = str(data_root.resolve())
    return environment


def _nonempty(environment: Mapping[str, str], name: str) -> bool:
    return bool(str(environment.get(name, "")).strip())


def has_complete_qq_environment(environment: Mapping[str, str]) -> bool:
    """Return whether all three existing QQ identity values are present."""

    return all(
        _nonempty(environment, name)
        for name in (
            QQ_APP_ID_ENV,
            QQ_CLIENT_SECRET_ENV,
            QQ_ALLOWED_OPENIDS_ENV,
        )
    )


def require_single_allowed_openid(environment: Mapping[str, str]) -> str:
    """Return the only allowlisted QQ user or fail closed."""

    allowed = [
        part.strip()
        for part in str(environment.get(QQ_ALLOWED_OPENIDS_ENV, "")).split(",")
        if part.strip()
    ]
    if len(allowed) != 1:
        raise QQOnboardLauncherError(
            "私人 QQ 启动入口要求白名单中恰好只有一个账号。"
        )
    return allowed[0]


def apply_onboard_result(
    environment: MutableMapping[str, str],
    result: Any,
) -> str:
    """Apply a validated SDK result without exposing its values in errors."""

    app_id = str(getattr(result, "app_id", "")).strip()
    client_secret = str(getattr(result, "client_secret", "")).strip()
    user_openid = str(getattr(result, "user_openid", "")).strip()
    if not app_id or not client_secret or not user_openid:
        raise QQOnboardLauncherError(
            "QQ 扫码结果缺少必要字段；没有启动后端，也没有保存任何凭据。"
        )

    environment[QQ_APP_ID_ENV] = app_id
    environment[QQ_CLIENT_SECRET_ENV] = client_secret
    environment[QQ_ALLOWED_OPENIDS_ENV] = user_openid
    environment[QQ_ENABLED_ENV] = "true"
    return user_openid


def enable_proactive_for_same_user(
    environment: MutableMapping[str, str],
    user_openid: str,
) -> None:
    """Enable proactive delivery only for the exact onboarded user."""

    environment[PROACTIVE_ENABLED_ENV] = "true"
    environment[PROACTIVE_RECIPIENT_ENV] = user_openid


def build_server_command(
    python_executable: str,
    *,
    verbose: bool,
    hf_mirror: bool,
) -> list[str]:
    command = [python_executable, str(PROJECT_ROOT / "run_server.py")]
    if verbose:
        command.append("--verbose")
    if hf_mirror:
        command.append("--hf_mirror")
    return command


def _show_onboard_link(url: str, *, open_browser: bool) -> None:
    print("\nQQ 官方绑定页面已生成，请在 5 分钟内按页面提示用手机 QQ 完成绑定。")
    opened = False
    if open_browser:
        try:
            opened = bool(webbrowser.open(url, new=2))
        except Exception:
            opened = False
    if not opened:
        print("未能自动打开浏览器，请复制下面的一次性链接：")
        print(url)
    print("等待扫码结果……（按 Ctrl+C 可安全取消）")


async def _run_onboard(
    *,
    open_browser: bool,
    onboard_factory: Callable[..., Any] | None = None,
) -> Any:
    if onboard_factory is None:
        try:
            from qqbot_agent_sdk import start_onboard
        except ImportError as exc:
            raise QQOnboardLauncherError(
                "未安装官方 qqbot-agent-sdk==1.2.2，无法开始扫码。"
            ) from exc
        onboard_factory = start_onboard

    return await onboard_factory(
        on_qr_ready=lambda url: _show_onboard_link(
            str(url),
            open_browser=open_browser,
        ),
        poll_timeout=300.0,
    )


async def prepare_session_environment(
    base_environment: Mapping[str, str],
    *,
    open_browser: bool,
    proactive: bool,
    onboard_factory: Callable[..., Any] | None = None,
) -> dict[str, str]:
    """Return one child-only environment for an authenticated QQ session."""

    environment = dict(base_environment)
    if has_complete_qq_environment(environment):
        user_openid = require_single_allowed_openid(environment)
        environment[QQ_ENABLED_ENV] = "true"
        print("检测到当前窗口已有完整 QQ 凭据，本次跳过扫码。")
    else:
        result = await _run_onboard(
            open_browser=open_browser,
            onboard_factory=onboard_factory,
        )
        user_openid = apply_onboard_result(environment, result)
        print("QQ 绑定成功；凭据只会交给本次凛祢后端进程。")

    if proactive:
        enable_proactive_for_same_user(environment, user_openid)
        print("本次会话已启用受限主动消息；收件人锁定为同一绑定账号。")
    else:
        environment.pop(PROACTIVE_ENABLED_ENV, None)
        environment.pop(PROACTIVE_RECIPIENT_ENV, None)
        print("主动消息保持关闭；先进行被动私聊实测。")

    return environment


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="扫码绑定官方 QQ Bot，并仅在本次进程中启动凛祢",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="不自动打开绑定页面，只在终端显示一次性链接",
    )
    parser.add_argument(
        "--proactive",
        action="store_true",
        help="本次会话额外启用受策略限制的主动消息",
    )
    parser.add_argument("--verbose", action="store_true", help="显示详细后端日志")
    parser.add_argument(
        "--hf-mirror",
        action="store_true",
        help="把 --hf_mirror 传给现有后端",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        base_environment = prepare_data_root_environment(os.environ)
        environment = asyncio.run(
            prepare_session_environment(
                base_environment,
                open_browser=not arguments.no_browser,
                proactive=arguments.proactive,
            )
        )
    except KeyboardInterrupt:
        print("\n已取消 QQ 绑定；没有启动后端，也没有保存凭据。")
        return 130
    except QQOnboardLauncherError as exc:
        print(f"QQ 启动准备失败：{exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            f"QQ 官方绑定没有完成（{type(exc).__name__}）；"
            "没有启动后端，也没有保存凭据。",
            file=sys.stderr,
        )
        return 2

    command = build_server_command(
        sys.executable,
        verbose=arguments.verbose,
        hf_mirror=arguments.hf_mirror,
    )
    print("正在启动凛祢后端；按 Ctrl+C 停止后，本次 QQ 凭据随进程一起清除。")
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
        )
    except KeyboardInterrupt:
        return 130
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
