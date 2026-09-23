from __future__ import annotations

import argparse
import hmac
import html
import json
import os
import re
import secrets
import sys
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{43}")
ROUTE_PREFIX = "/qq-recovery"
MAX_QR_BYTES = 5 * 1024 * 1024
STATUS_MAX_AGE = timedelta(seconds=90)
TRANSPORT_RECENT_AGE = timedelta(minutes=5)
REMOTE_PROBE_RECENT_AGE = timedelta(minutes=3)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def load_or_create_token(token_file: Path) -> str:
    token_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        token = token_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        token = secrets.token_urlsafe(32)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(token_file, flags, 0o600)
        except FileExistsError:
            token = token_file.read_text(encoding="utf-8").strip()
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(token)
    if not TOKEN_PATTERN.fullmatch(token):
        raise ValueError("恢复页令牌文件格式无效。")
    return token


def _parse_updated_at(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    # PowerShell/.NET round-trip timestamps use seven fractional digits, while
    # Python 3.10's fromisoformat accepts at most six. Truncate only the extra
    # fractional precision and preserve the explicit timezone offset.
    normalized = re.sub(
        r"(\.\d{6})\d+(?=[+-]\d{2}:\d{2}$)",
        r"\1",
        normalized,
    )
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _empty_transport_evidence() -> dict[str, object]:
    return {
        "last_inbound_at": "",
        "last_reply_submitted_at": "",
        "last_probe_at": "",
        "last_probe_ok_at": "",
        "last_probe_failed_at": "",
        "consecutive_probe_failures": 0,
    }


def _load_transport_evidence(transport_file: Path) -> dict[str, object]:
    try:
        if transport_file.is_symlink() or transport_file.stat().st_size > 64 * 1024:
            return _empty_transport_evidence()
        payload = json.loads(transport_file.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return _empty_transport_evidence()
    if not isinstance(payload, dict) or payload.get("version") != 1:
        return _empty_transport_evidence()
    result = _empty_transport_evidence()
    for key in (
        "last_inbound_at",
        "last_reply_submitted_at",
        "last_probe_at",
        "last_probe_ok_at",
        "last_probe_failed_at",
    ):
        value = payload.get(key)
        result[key] = value if _parse_updated_at(value) is not None else ""
    failures = payload.get("consecutive_probe_failures", 0)
    if isinstance(failures, int) and not isinstance(failures, bool) and failures >= 0:
        result["consecutive_probe_failures"] = failures
    return result


def load_public_status(
    status_file: Path,
    qr_file: Path,
    transport_file: Path | None = None,
) -> dict[str, object]:
    try:
        payload = json.loads(status_file.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return {
            "state": "unavailable",
            "updated_at": "",
            "qr_available": False,
        }

    state = payload.get("state")
    if state not in {
        "starting",
        "ready",
        "recovering",
        "scan_required",
        "account_restricted",
        "failed",
        "stopped",
    }:
        state = "unavailable"
    updated_at = _parse_updated_at(payload.get("updated_at"))
    is_fresh = bool(
        updated_at
        and datetime.now(timezone.utc) - updated_at <= STATUS_MAX_AGE
        and updated_at - datetime.now(timezone.utc) <= timedelta(seconds=30)
    )
    if not is_fresh and state not in {"stopped", "failed"}:
        state = "unavailable"

    qr_available = False
    if state == "scan_required" and payload.get("qr_available") is True:
        try:
            qr_stat = qr_file.stat()
            qr_available = bool(
                not qr_file.is_symlink()
                and 0 < qr_stat.st_size <= MAX_QR_BYTES
            )
        except OSError:
            qr_available = False

    evidence = _load_transport_evidence(
        transport_file or status_file.with_name("transport_health.json")
    )
    now = datetime.now(timezone.utc)
    inbound_at = _parse_updated_at(evidence["last_inbound_at"])
    reply_at = _parse_updated_at(evidence["last_reply_submitted_at"])
    probe_ok_at = _parse_updated_at(evidence["last_probe_ok_at"])
    probe_failed_at = _parse_updated_at(evidence["last_probe_failed_at"])
    inbound_recent = bool(inbound_at and now - inbound_at <= TRANSPORT_RECENT_AGE)
    reply_recent = bool(reply_at and now - reply_at <= TRANSPORT_RECENT_AGE)
    probe_ok_recent = bool(
        probe_ok_at
        and now - probe_ok_at <= REMOTE_PROBE_RECENT_AGE
        and (probe_failed_at is None or probe_ok_at >= probe_failed_at)
        and evidence["consecutive_probe_failures"] == 0
    )
    probe_failed = bool(
        probe_failed_at
        and (probe_ok_at is None or probe_failed_at > probe_ok_at)
        and evidence["consecutive_probe_failures"] >= 3
    )

    return {
        "state": state,
        "updated_at": payload.get("updated_at", "") if is_fresh else "",
        "qr_available": qr_available,
        "last_inbound_at": evidence["last_inbound_at"],
        "last_reply_submitted_at": evidence["last_reply_submitted_at"],
        "last_probe_at": evidence["last_probe_at"],
        "remote_probe_state": (
            "verified" if probe_ok_recent else "failed" if probe_failed else "checking"
        ),
        "transport_state": (
            "roundtrip_recent"
            if inbound_recent and reply_recent
            else "inbound_recent"
            if inbound_recent
            else "unverified"
        ),
    }


def _state_copy(
    state: str,
    qr_available: bool,
    transport_state: str,
    remote_probe_state: str,
) -> tuple[str, str, str]:
    if state == "ready":
        if remote_probe_state == "failed":
            return (
                "QQ 连接异常",
                "连续远端探测失败，监护程序正在尝试自动恢复。",
                "waiting",
            )
        if remote_probe_state != "verified":
            return (
                "正在验证 QQ",
                "本机组件已经运行，正在等待 QQ 服务器的真实响应。",
                "waiting",
            )
        if transport_state == "roundtrip_recent":
            return (
                "最近已验证",
                "最近收到了你的 QQ 消息，并成功向 QQ 提交了回复。",
                "ready",
            )
        if transport_state == "inbound_recent":
            return (
                "收信已验证",
                "最近收到了你的 QQ 消息，但回复尚未在本页确认。",
                "waiting",
            )
        return (
            "QQ 远端已验证",
            "QQ 服务器最近响应了静默探测；尚无最近聊天往返记录。",
            "ready",
        )
    if state == "scan_required" and qr_available:
        return "等待扫码", "请用 QQ 摄像头扫描另一块可信屏幕上的二维码。", "scan"
    if state == "scan_required":
        return "正在生成二维码", "NapCat 正在准备新的二维码，本页会自动刷新。", "waiting"
    if state == "account_restricted":
        return (
            "QQ 账号暂时受限",
            "QQ 拒绝生成新的登录二维码。请先在最新版手机 QQ 中按提示恢复账号使用，然后在电脑上停止并重新启动私人 QQ 凛祢。",
            "failed",
        )
    if state == "recovering":
        return "自动恢复中", "正在尝试快速登录，暂时不需要操作。", "waiting"
    if state == "starting":
        return "启动中", "QQ 凛祢的本地服务正在启动。", "waiting"
    if state == "failed":
        return "本地服务异常", "需要回到电脑查看 QQ 通道运行日志。", "failed"
    if state == "stopped":
        return "已停止", "QQ 凛祢当前没有运行。", "stopped"
    return "暂时无法确认", "电脑可能休眠、断网，或本地服务未能更新状态。", "failed"


def render_status_page(status: dict[str, object], qr_url: str) -> bytes:
    state = str(status["state"])
    qr_available = bool(status["qr_available"])
    title, description, css_state = _state_copy(
        state,
        qr_available,
        str(status.get("transport_state") or "unverified"),
        str(status.get("remote_probe_state") or "checking"),
    )
    updated_at = html.escape(str(status.get("updated_at") or "暂无"))
    inbound_at = html.escape(str(status.get("last_inbound_at") or "暂无"))
    reply_at = html.escape(str(status.get("last_reply_submitted_at") or "暂无"))
    probe_at = html.escape(str(status.get("last_probe_at") or "暂无"))
    qr_markup = ""
    if qr_available:
        qr_markup = (
            '<div class="qr"><img src="'
            + html.escape(qr_url, quote=True)
            + '" alt="QQ 登录二维码" width="320" height="320"></div>'
            '<ol><li>在电脑、iPad 或另一部可信手机上打开本恢复地址。</li>'
            '<li>切换到凛祢小号，用 QQ 摄像头扫描另一块屏幕。</li>'
            '<li>确认登录后，等待本页显示“组件已就绪”，再发送 /status 验证实际往返。</li></ol>'
            '<p class="notice">QQ 可能拒绝同一手机的相册或长按识别；只有一块屏幕时无法完成此次授权。</p>'
        )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="refresh" content="10">
  <title>QQ 凛祢登录恢复</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui,-apple-system,"Segoe UI",sans-serif; }}
    body {{ margin:0; min-height:100vh; display:grid; place-items:center; background:#f5f2f6; color:#28232a; }}
    main {{ width:min(92vw,560px); box-sizing:border-box; padding:28px; border-radius:24px; background:#fff; box-shadow:0 12px 40px #4b284015; }}
    h1 {{ margin:0 0 22px; font-size:1.35rem; }}
    .state {{ display:inline-block; padding:7px 12px; border-radius:999px; font-weight:700; }}
    .ready {{ background:#def7e7; color:#176235; }} .scan {{ background:#fff0cf; color:#815300; }}
    .waiting {{ background:#e7f0ff; color:#214d8b; }} .failed,.stopped {{ background:#f2e9ed; color:#684955; }}
    p {{ line-height:1.65; }} .updated {{ color:#7d747b; font-size:.88rem; }}
    .qr {{ margin:24px auto 18px; width:min(100%,320px); }}
    .qr img {{ display:block; width:100%; height:auto; border-radius:16px; image-rendering:auto; }}
    ol {{ padding-left:1.4rem; line-height:1.8; }}
    .notice {{ padding:12px 14px; border-radius:12px; background:#fff3e1; color:#744b12; font-size:.9rem; }}
    footer {{ margin-top:24px; color:#8a8087; font-size:.78rem; }}
  </style>
</head>
<body><main>
  <h1>QQ 凛祢登录恢复</h1>
  <div class="state {css_state}">{html.escape(title)}</div>
  <p>{html.escape(description)}</p>
  {qr_markup}
  <p class="updated">状态更新时间：{updated_at}</p>
  <p class="updated">最后收到 QQ 消息：{inbound_at}</p>
  <p class="updated">最后向 QQ 提交回复：{reply_at}</p>
  <p class="updated">最后一次 QQ 远端探测：{probe_at}</p>
  <footer>页面以 QQ 服务器的静默响应和最近消息往返作为凭据，不再把本地端口等同于在线。本页每 10 秒自动刷新。</footer>
</main></body></html>"""
    return document.encode("utf-8")


class RecoveryServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        status_file: Path,
        qr_file: Path,
        transport_file: Path | None = None,
        token: str,
    ) -> None:
        self.status_file = status_file
        self.qr_file = qr_file
        self.transport_file = transport_file or status_file.with_name(
            "transport_health.json"
        )
        self.token = token
        self.route = f"{ROUTE_PREFIX}/{token}/"
        super().__init__(server_address, RecoveryRequestHandler)


class RecoveryRequestHandler(BaseHTTPRequestHandler):
    server: RecoveryServer
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        # Request URLs contain the secret capability token and must never be logged.
        return

    def _security_headers(self, content_type: str, length: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
        )

    def _send_bytes(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
        *,
        include_body: bool,
    ) -> None:
        self.send_response(status)
        self._security_headers(content_type, len(body))
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def _authorized_path(self, request_path: str) -> str | None:
        route = self.server.route
        if request_path == route:
            return "page"
        if request_path == route + "qr.png":
            return "qr"
        if request_path.rstrip("/") == route.rstrip("/"):
            return "redirect"
        return None

    def _serve(self, *, include_body: bool) -> None:
        request_path = urlsplit(self.path).path
        expected_token = self.server.token.encode("ascii")
        parts = request_path.split("/")
        supplied = parts[2].encode("ascii", errors="ignore") if len(parts) > 2 else b""
        if not hmac.compare_digest(supplied, expected_token):
            self._send_bytes(
                HTTPStatus.NOT_FOUND,
                b"Not Found\n",
                "text/plain; charset=utf-8",
                include_body=include_body,
            )
            return

        target = self._authorized_path(request_path)
        if target == "redirect":
            self.send_response(HTTPStatus.PERMANENT_REDIRECT)
            self.send_header("Location", self.server.route)
            self._security_headers("text/plain; charset=utf-8", 0)
            self.end_headers()
            return
        if target == "page":
            status = load_public_status(
                self.server.status_file,
                self.server.qr_file,
                self.server.transport_file,
            )
            body = render_status_page(status, self.server.route + "qr.png")
            self._send_bytes(
                HTTPStatus.OK,
                body,
                "text/html; charset=utf-8",
                include_body=include_body,
            )
            return
        if target == "qr":
            status = load_public_status(
                self.server.status_file,
                self.server.qr_file,
                self.server.transport_file,
            )
            if not status["qr_available"]:
                self._send_bytes(
                    HTTPStatus.NOT_FOUND,
                    b"Not Found\n",
                    "text/plain; charset=utf-8",
                    include_body=include_body,
                )
                return
            try:
                body = self.server.qr_file.read_bytes()
            except OSError:
                body = b""
            if not body.startswith(b"\x89PNG\r\n\x1a\n") or len(body) > MAX_QR_BYTES:
                body = b""
            if not body:
                self._send_bytes(
                    HTTPStatus.NOT_FOUND,
                    b"Not Found\n",
                    "text/plain; charset=utf-8",
                    include_body=include_body,
                )
                return
            self._send_bytes(
                HTTPStatus.OK,
                body,
                "image/png",
                include_body=include_body,
            )
            return
        self._send_bytes(
            HTTPStatus.NOT_FOUND,
            b"Not Found\n",
            "text/plain; charset=utf-8",
            include_body=include_body,
        )

    def do_GET(self) -> None:  # noqa: N802
        self._serve(include_body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._serve(include_body=False)

    def do_POST(self) -> None:  # noqa: N802
        self._send_bytes(
            HTTPStatus.METHOD_NOT_ALLOWED,
            b"Method Not Allowed\n",
            "text/plain; charset=utf-8",
            include_body=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the private QQ QR recovery page.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12395)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--qr-file", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--url-file", type=Path, required=True)
    parser.add_argument("--public-origin", default="http://127.0.0.1:12395")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.host != "127.0.0.1":
        raise ValueError("恢复页只能监听 127.0.0.1。")
    token = load_or_create_token(args.token_file)
    origin = args.public_origin.rstrip("/")
    public_url = f"{origin}{ROUTE_PREFIX}/{token}/"
    _atomic_write_text(args.url_file, public_url)
    server = RecoveryServer(
        (args.host, args.port),
        status_file=args.status_file,
        qr_file=args.qr_file,
        token=token,
    )
    print(f"QQ recovery page listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
