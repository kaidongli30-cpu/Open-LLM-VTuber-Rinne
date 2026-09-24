"""Safely fast-forward a Rinne checkout while preserving its local conf.yaml."""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ruamel.yaml import YAML


ROOT = Path.cwd().resolve()
MISSING = object()
APP_RELEASES_API = (
    "https://api.github.com/repos/kaidongli30-cpu/"
    "Open-LLM-VTuber-Rinne/releases?per_page=30"
)
APP_RELEASE_TAG = re.compile(r"^rinne-app-v(\d+)\.(\d+)\.(\d+)$")
OFFICIAL_REMOTES = {
    "https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne",
    "git@github.com:kaidongli30-cpu/Open-LLM-VTuber-Rinne",
    "ssh://git@github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne",
}


class UpdateError(Exception):
    """An update cannot proceed without risking local data."""


def _git(*args: str, strip: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode:
        raise UpdateError(f"Git 操作失败：{args[0]}")
    return result.stdout.strip() if strip else result.stdout


def _yaml() -> YAML:
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    return yaml


def _parse_config(text: str) -> Mapping[str, Any]:
    config = _yaml().load(text)
    if not isinstance(config, Mapping):
        raise UpdateError("conf.yaml 格式不是 YAML 配置对象，更新已停止")
    return config


def merge_config(old_text: str, user_text: str, new_text: str) -> tuple[str, list[str]]:
    """Apply upstream-only field edits to the user's round-trip YAML tree."""
    old = _parse_config(old_text)
    user = copy.deepcopy(_parse_config(user_text))
    new = _parse_config(new_text)
    conflicts: list[str] = []

    def merge(
        old_node: Any, user_node: Any, new_node: Any, path: tuple[str, ...]
    ) -> Any:
        if user_node == new_node:
            return user_node
        if user_node == old_node:
            return MISSING if new_node is MISSING else copy.deepcopy(new_node)
        if new_node == old_node:
            return user_node
        if isinstance(user_node, Mapping) and isinstance(new_node, Mapping):
            if old_node is MISSING:
                old_node = {}
            if isinstance(old_node, Mapping):
                for key in dict.fromkeys((*old_node, *user_node, *new_node)):
                    child = merge(
                        old_node.get(key, MISSING),
                        user_node.get(key, MISSING),
                        new_node.get(key, MISSING),
                        (*path, str(key)),
                    )
                    if child is MISSING:
                        user_node.pop(key, None)
                    else:
                        user_node[key] = child
                return user_node
        conflicts.append(".".join(path))
        return user_node

    merged = merge(old, user, new, ())
    if conflicts:
        return "", sorted(conflicts)
    if merged is MISSING or not isinstance(merged, Mapping):
        raise UpdateError("配置合并结果无效，更新已停止")
    output = io.StringIO()
    _yaml().dump(merged, output)
    result = output.getvalue()
    _parse_config(result)
    return result, []


def apply_local_overlay(user_text: str, overlay_text: str) -> str:
    """Materialize an older conf.local.yaml into the directly editable config."""
    user = copy.deepcopy(_parse_config(user_text))
    overlay = _parse_config(overlay_text)

    def apply(target: Any, source: Mapping[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
                apply(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    apply(user, overlay)
    output = io.StringIO()
    _yaml().dump(user, output)
    return output.getvalue()


def _ensure_backend_stopped(config_text: str) -> None:
    config = _parse_config(config_text)
    system = config.get("system_config")
    if not isinstance(system, Mapping):
        return
    port = system.get("port")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        return
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            raise UpdateError("请先关闭正在运行的凛祢后端，再点击更新；未修改文件")
    except OSError:
        return


def _atomic_write(path: Path, data: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="conf.yaml.updating-",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _check_checkout() -> None:
    if Path(_git("rev-parse", "--show-toplevel")).resolve() != ROOT:
        raise UpdateError("请从凛祢项目目录运行更新")
    if _git("symbolic-ref", "--short", "HEAD") != "main":
        raise UpdateError("仅自动更新 main 分支；当前分支未修改")
    origin = _git("remote", "get-url", "origin").removesuffix(".git").rstrip("/")
    if origin not in OFFICIAL_REMOTES:
        raise UpdateError("origin 不是凛祢的官方仓库；未修改文件")
    changed = _git("status", "--porcelain", "--untracked-files=no", strip=False)
    if any(line[3:] != "conf.yaml" for line in changed.splitlines()):
        raise UpdateError("发现 conf.yaml 之外的本地改动，请先自行处理；未修改文件")
    if _git("diff", "--cached", "--name-only"):
        raise UpdateError("暂存区有本地改动；未修改文件")


def _latest_release_tag(requested_tag: str | None = None) -> str | None:
    if requested_tag is not None and not APP_RELEASE_TAG.fullmatch(requested_tag):
        raise UpdateError("指定的正式版本格式无效；未修改文件")
    request = Request(
        APP_RELEASES_API,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "Rinne-Safe-Update",
        },
    )
    try:
        with urlopen(request, timeout=10) as response:
            payload = response.read(2_000_001)
    except (HTTPError, URLError, TimeoutError) as error:
        raise UpdateError("无法检查 GitHub 正式版本；未修改文件") from error
    if len(payload) > 2_000_000:
        raise UpdateError("GitHub 版本信息过大；未修改文件")
    try:
        releases = json.loads(payload)
    except (UnicodeError, ValueError) as error:
        raise UpdateError("GitHub 版本信息无法解析；未修改文件") from error
    if not isinstance(releases, list):
        raise UpdateError("GitHub 版本信息格式异常；未修改文件")
    candidates: list[tuple[tuple[int, int, int], str]] = []
    for release in releases:
        if not isinstance(release, dict):
            continue
        tag = release.get("tag_name")
        if (
            not isinstance(tag, str)
            or release.get("draft")
            or release.get("prerelease")
        ):
            continue
        if not release.get("published_at"):
            continue
        match = APP_RELEASE_TAG.fullmatch(tag)
        if match:
            candidates.append((tuple(map(int, match.groups())), tag))
    if requested_tag is not None:
        if any(tag == requested_tag for _, tag in candidates):
            return requested_tag
        raise UpdateError("指定的后端正式版本尚未发布；未修改文件")
    return max(candidates)[1] if candidates else None


def run_update(
    *, apply: bool, release_tag: str | None = None
) -> tuple[str, Path | None]:
    _check_checkout()
    config_path = ROOT / "conf.yaml"
    user_bytes = config_path.read_bytes()
    user_text = user_bytes.decode("utf-8-sig")
    overlay_path = ROOT / "conf.local.yaml"
    overlay_present = overlay_path.is_file()
    if overlay_present:
        user_text = apply_local_overlay(
            user_text, overlay_path.read_bytes().decode("utf-8-sig")
        )
    if apply:
        _ensure_backend_stopped(user_text)
    head = _git("rev-parse", "HEAD")
    release_tag = (
        _latest_release_tag(release_tag)
        if release_tag is not None
        else _latest_release_tag()
    )
    if release_tag is None:
        return "暂无正式应用版本", None
    _git("fetch", "origin", "tag", release_tag)
    target = _git("rev-parse", f"refs/tags/{release_tag}^{{commit}}")
    needs_git_update = head != target
    if not needs_git_update and not overlay_present:
        if apply:
            _git("submodule", "update", "--init", "--recursive")
        return "已经是最新版", None
    if needs_git_update and _git("merge-base", head, target) != head:
        raise UpdateError("本地与 GitHub 的提交已分叉，不能自动更新；未修改文件")

    old_text = _git("show", f"{head}:conf.yaml", strip=False)
    new_text = _git("show", f"{target}:conf.yaml", strip=False)
    merged_text, conflicts = merge_config(old_text, user_text, new_text)
    if conflicts:
        fields = "、".join(conflicts)
        raise UpdateError(f"这些配置项在本机和新版中都被修改：{fields}；未修改文件")
    if not apply:
        return (
            "发现可更新版本；配置可安全合并"
            if needs_git_update
            else "代码已是最新版；旧版配置可迁入 conf.yaml",
            None,
        )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup = config_path.with_name(f"conf.yaml.backup-{stamp}")
    if backup.exists():
        raise UpdateError("配置备份文件名冲突；未修改文件")
    overlay_backup = overlay_path.with_name(f"conf.local.yaml.backup-{stamp}")
    if overlay_present and overlay_backup.exists():
        raise UpdateError("旧配置备份文件名冲突；未修改文件")
    shutil.copy2(config_path, backup)
    if needs_git_update:
        try:
            # Return only this tracked file to its exact HEAD bytes so a
            # fast-forward can proceed. The user's copy is backed up above.
            _atomic_write(config_path, old_text.encode("utf-8"))
            # Refresh Git's index stat cache after replacing the file on Windows.
            _git("add", "--", "conf.yaml")
            _git("diff", "--cached", "--quiet")
            _git("merge", "--ff-only", target)
        except Exception:
            if _git("rev-parse", "HEAD") == head:
                _atomic_write(config_path, user_bytes)
            else:
                _atomic_write(config_path, merged_text.encode("utf-8"))
            raise
    try:
        _atomic_write(config_path, merged_text.encode("utf-8"))
    except OSError:
        _atomic_write(config_path, user_bytes)
        raise
    if needs_git_update:
        _git("submodule", "update", "--init", "--recursive")
    if overlay_present:
        os.replace(overlay_path, overlay_backup)
        return (
            f"更新完成；旧版配置已转入 conf.yaml，旧覆盖文件备份：{overlay_backup}。"
            "重启后端和桌面客户端后生效",
            backup,
        )
    return "更新完成；重启后端和桌面客户端后生效", backup


def main() -> int:
    parser = argparse.ArgumentParser(description="安全更新凛祢并保留本机 conf.yaml")
    parser.add_argument("--apply", action="store_true", help="备份配置后执行更新")
    parser.add_argument(
        "--release", help="只更新到此正式后端版本，例如 rinne-app-v1.2.2"
    )
    args = parser.parse_args()
    try:
        message, backup = run_update(apply=args.apply, release_tag=args.release)
    except (UpdateError, OSError, UnicodeError, ValueError) as error:
        print(
            error
            if isinstance(error, UpdateError)
            else "更新前检查失败；未继续修改文件"
        )
        return 1
    print(message)
    if backup is not None:
        print(f"原配置备份：{backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
