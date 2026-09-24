"""Safely fast-forward a Rinne checkout while preserving its local conf.yaml."""

from __future__ import annotations

import argparse
import copy
import io
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


ROOT = Path(__file__).resolve().parents[1]
MISSING = object()


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
    changed = _git("status", "--porcelain", "--untracked-files=no", strip=False)
    if any(line[3:] != "conf.yaml" for line in changed.splitlines()):
        raise UpdateError("发现 conf.yaml 之外的本地改动，请先自行处理；未修改文件")
    if _git("diff", "--cached", "--name-only"):
        raise UpdateError("暂存区有本地改动；未修改文件")


def run_update(*, apply: bool) -> tuple[str, Path | None]:
    _check_checkout()
    config_path = ROOT / "conf.yaml"
    user_bytes = config_path.read_bytes()
    user_text = user_bytes.decode("utf-8-sig")
    _git("fetch", "origin", "main")
    head = _git("rev-parse", "HEAD")
    target = _git("rev-parse", "origin/main")
    if head == target:
        return "已经是最新版", None
    if _git("merge-base", head, target) != head:
        raise UpdateError("本地与 GitHub 的提交已分叉，不能自动更新；未修改文件")

    old_text = _git("show", f"{head}:conf.yaml", strip=False)
    new_text = _git("show", f"{target}:conf.yaml", strip=False)
    merged_text, conflicts = merge_config(old_text, user_text, new_text)
    if conflicts:
        fields = "、".join(conflicts)
        raise UpdateError(f"这些配置项在本机和新版中都被修改：{fields}；未修改文件")
    if not apply:
        return "发现可更新版本；配置可安全合并", None

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup = config_path.with_name(f"conf.yaml.backup-{stamp}")
    if backup.exists():
        raise UpdateError("配置备份文件名冲突；未修改文件")
    shutil.copy2(config_path, backup)
    try:
        # Return only this tracked file to its exact HEAD bytes so a fast-forward
        # can proceed. The user's copy is already safely backed up above.
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
    _git("submodule", "update", "--init", "--recursive")
    return "更新完成；重启后端和桌面客户端后生效", backup


def main() -> int:
    parser = argparse.ArgumentParser(description="安全更新凛祢并保留本机 conf.yaml")
    parser.add_argument("--apply", action="store_true", help="备份配置后执行更新")
    args = parser.parse_args()
    try:
        message, backup = run_update(apply=args.apply)
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
