"""Update official Rinne releases while retaining local configuration and keys."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError


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
# Only this released, pre-sanitization checkout has an audited rewritten base.
# Never use this exception for arbitrary user commits or divergent branches.
LEGACY_RELEASE_BASES = {
    "0c66f0f350a4955e2094c0d84e663bcef0cc4875": "3f63381720bf97375160cbd37fa11b998ede81d5",
}
PYTHON_KEY_FILES = {
    "diary_generator.py": "LLM_API_KEY",
    "memory_generation_config.py": "API_KEY",
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
    try:
        config = _yaml().load(text)
    except YAMLError as error:
        raise UpdateError(
            "conf.yaml 格式错误，请检查缩进、冒号和引号；未修改文件"
        ) from error
    if not isinstance(config, Mapping):
        raise UpdateError("conf.yaml 格式不是 YAML 配置对象，更新已停止")
    return config


def merge_config(
    old_text: str,
    user_text: str,
    new_text: str,
    resolve: Callable[[str, Any, Any], Any] | None = None,
) -> tuple[str, list[str]]:
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
        if path and _personal_field(path[-1]) and user_node is not MISSING:
            return user_node
        if user_node == old_node:
            return MISSING if new_node is MISSING else copy.deepcopy(new_node)
        if new_node == old_node:
            return user_node
        if (
            path
            and path[-1].lower().endswith("api_key")
            and new_node == ""
            and isinstance(user_node, str)
        ):
            # New public defaults intentionally leave credentials blank. A
            # filled-in key from an older checkout belongs to the user.
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
        if resolve is not None:
            return resolve(".".join(path), user_node, new_node)
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
            prefix=f"{path.name}.updating-",
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


def _repair_game_asset_manifests(skip: set[str] | None = None) -> int:
    """Restore Git's exact JSON bytes after Windows checkout converted LF to CRLF.

    Only a pure line-ending conversion of a tracked file may be repaired. Any
    other difference is treated as a local edit and left untouched.
    """
    paths_result = subprocess.run(
        ["git", "ls-files", "-z", "--", "local_game_assets"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if paths_result.returncode:
        raise UpdateError("无法检查游戏资源清单；更新已停止")
    names = [
        item.decode("utf-8")
        for item in paths_result.stdout.split(b"\0")
        if item.endswith(b".json")
    ]
    if not names:
        return 0
    requests = b"".join(f"HEAD:{name}\n".encode("utf-8") for name in names)
    blobs_result = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=ROOT,
        input=requests,
        capture_output=True,
        check=False,
    )
    if blobs_result.returncode:
        raise UpdateError("无法读取游戏资源清单的原始内容；更新已停止")
    output = blobs_result.stdout
    cursor = 0
    repairs: list[tuple[Path, bytes]] = []
    for name in names:
        end = output.find(b"\n", cursor)
        if end < 0:
            raise UpdateError("游戏资源清单格式异常；更新已停止")
        header = output[cursor:end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise UpdateError("游戏资源清单格式异常；更新已停止")
        try:
            size = int(header[2])
        except ValueError as error:
            raise UpdateError("游戏资源清单长度异常；更新已停止") from error
        start = end + 1
        finish = start + size
        if size < 0 or output[finish : finish + 1] != b"\n":
            raise UpdateError("游戏资源清单长度异常；更新已停止")
        blob = output[start:finish]
        cursor = finish + 1
        if skip and name in skip:
            continue
        path = ROOT / name
        actual = path.read_bytes()
        if actual != blob:
            if actual.replace(b"\r\n", b"\n") != blob:
                raise UpdateError(f"游戏资源清单有本地改动：{name}；未覆盖该文件")
            repairs.append((path, blob))
    if cursor != len(output):
        raise UpdateError("游戏资源清单数据异常；更新已停止")
    for path, blob in repairs:
        _atomic_write(path, blob)
    # On Git for Windows with core.autocrlf=true, replacing a file can leave
    # stale index metadata even when its bytes match HEAD. Stage only the
    # repaired paths; their blob IDs must remain identical to HEAD.
    for offset in range(0, len(repairs), 32):
        names_to_refresh = [
            path.relative_to(ROOT).as_posix()
            for path, _ in repairs[offset : offset + 32]
        ]
        _git("add", "--", *names_to_refresh)
    if _git("diff", "--cached", "--name-only"):
        raise UpdateError("游戏资源清单与正式版本不一致；更新已停止")
    return len(repairs)


def _check_checkout() -> None:
    if Path(_git("rev-parse", "--show-toplevel")).resolve() != ROOT:
        raise UpdateError("请从凛祢项目目录运行更新")
    if _git("symbolic-ref", "--short", "HEAD") != "main":
        raise UpdateError("仅自动更新 main 分支；当前分支未修改")
    origin = _git("remote", "get-url", "origin").removesuffix(".git").rstrip("/")
    if origin not in OFFICIAL_REMOTES:
        raise UpdateError("origin 不是凛祢的官方仓库；未修改文件")
    if _git("diff", "--cached", "--name-only"):
        raise UpdateError(
            "暂存区有尚未提交的改动，请先取消暂存（保留文件内容）后重试；未修改文件"
        )


def _key_assignment(text: str, variable: str) -> tuple[ast.expr, int, int]:
    """Locate one top-level value without executing user Python code."""
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        raise UpdateError("密钥配置文件存在语法错误；未修改文件") from error
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == variable
    ]
    if len(assignments) != 1:
        raise UpdateError(f"无法安全定位 {variable} 配置；未修改文件")
    value = assignments[0].value
    lines = text.encode("utf-8").splitlines(keepends=True)
    start = sum(map(len, lines[: value.lineno - 1])) + value.col_offset
    end = sum(map(len, lines[: value.end_lineno - 1])) + value.end_col_offset
    return value, start, end


def _replace_key_value(text: str, variable: str, expression: str) -> str:
    _, start, end = _key_assignment(text, variable)
    data = text.encode("utf-8")
    return (data[:start] + expression.encode("utf-8") + data[end:]).decode("utf-8")


def _key_file_updates(head: str, target: str) -> dict[str, tuple[bytes, str, str]]:
    updates = {}
    for name, variable in PYTHON_KEY_FILES.items():
        old = _git("show", f"{head}:{name}", strip=False)
        new = _git("show", f"{target}:{name}", strip=False)
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            raise UpdateError(f"密钥配置文件路径无效：{name}；未修改文件")
        user_bytes = path.read_bytes()
        user = user_bytes.decode("utf-8-sig").replace("\r\n", "\n")
        old = old.replace("\r\n", "\n")
        value, _, _ = _key_assignment(user, variable)
        if user != old:
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                raise UpdateError(f"{name} 仅支持直接填写字符串密钥；未修改文件")
            if _replace_key_value(user, variable, "''") != _replace_key_value(
                old, variable, "''"
            ):
                raise UpdateError(f"{name} 除密钥外还有源码改动；未修改文件")
        # A literal key also belongs to the user when already in their old HEAD.
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            new = _replace_key_value(new, variable, repr(value.value))
        updates[name] = (user_bytes, old, new)
    return updates


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


def _personal_field(name: str) -> bool:
    return bool(re.search(r"(?i)(api_?key|token|password|secret|authorization)$", name))


def _digest(data: bytes | None) -> str:
    return "missing" if data is None else hashlib.sha256(data).hexdigest()


def _protected(name: str) -> bool:
    parts = Path(name).parts
    return bool(
        parts and parts[0].lower() in {"chat_history", "rinne_library"}
    ) and name not in {"rinne_library/README.md", "rinne_library/.gitignore"}


def _safe_path(name: str) -> Path:
    relative = Path(name)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or ".git" in {part.lower() for part in relative.parts}
    ):
        raise UpdateError("更新文件路径无效；未修改文件")
    path = ROOT / relative
    for item in (path, *path.parents):
        if item == ROOT:
            break
        if item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
            raise UpdateError(f"文件路径包含链接，请先人工确认：{name}")
    if path.exists() and not path.is_file():
        raise UpdateError(f"更新路径不是普通文件：{name}")
    return path


def _tree(ref: str) -> dict[str, tuple[str, str]]:
    entries = {}
    for record in _git("ls-tree", "-r", "-z", ref, strip=False).split("\0"):
        if record:
            metadata, name = record.split("\t", 1)
            mode, _kind, oid = metadata.split()
            entries[name] = (mode, oid)
    return entries


def _blob(ref: str, name: str, tree: Mapping) -> bytes | None:
    if name not in tree:
        return None
    result = subprocess.run(
        ["git", "cat-file", "blob", f"{ref}:{name}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise UpdateError(f"无法读取正式版本文件：{name}")
    return result.stdout


def _read_local(name: str) -> bytes | None:
    path = _safe_path(name)
    return path.read_bytes() if path.is_file() else None


def _text(data: bytes | None) -> str | None:
    if data is None or b"\0" in data:
        return None
    try:
        return data.decode("utf-8-sig").replace("\r\n", "\n")
    except UnicodeError:
        return None


class Resolutions:
    def __init__(self, choices: Mapping[str, str] | None = None):
        if choices is not None and not isinstance(choices, Mapping):
            raise UpdateError("更新选择格式无效；未修改文件")
        self.choices = dict(choices or {})
        if any(value not in {"mine", "new"} for value in self.choices.values()):
            raise UpdateError("更新选择无效；未修改文件")
        self.items: list[dict[str, str]] = []
        self.secrets: set[str] = set()

    def preview(self, value: Any, label: str = "") -> str:
        if _personal_field(label.rsplit(".", 1)[-1]):
            return "[敏感设置已隐藏]"
        if value is MISSING or value is None:
            return "[此版本没有这一项]"
        if isinstance(value, bytes):
            decoded = _text(value)
            if decoded is None:
                return f"[二进制文件，{len(value)} 字节，SHA256 {_digest(value)[:12]}]"
            value = decoded
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, default=str)
        for secret in sorted(self.secrets, key=len, reverse=True):
            if secret:
                value = value.replace(secret, "[已隐藏]")
        lines = []
        sensitive_block = False
        for line in value.splitlines():
            # Hide whole credential-bearing lines, including multiline literals.
            if re.search(r"(?i)(api.?key|token|password|secret|authorization)", line):
                sensitive_block = '"""' in line or "'''" in line
                lines.append("[敏感设置已隐藏]")
            elif sensitive_block:
                if '"""' in line or "'''" in line:
                    sensitive_block = False
            else:
                lines.append(line)
        preview = "\n".join(lines)
        if len(preview) > 6000:
            preview = (
                preview[:6000]
                + "\n[内容较长，以上为节选；不确定时请取消后查看本地文件]"
            )
        return preview

    def choose(self, name: str, section: str, mine: Any, new: Any) -> Any:
        identity = hashlib.sha256((name + "\0" + section).encode("utf-8")).hexdigest()
        item = {
            "id": identity,
            "file": name,
            "section": section,
            "mine": self.preview(mine, section),
            "new": self.preview(new, section),
        }
        self.items.append(item)
        return new if self.choices.get(identity) == "new" else mine

    def collect_secrets(self, node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if _personal_field(str(key)) and isinstance(value, str) and value:
                    self.secrets.add(value)
                else:
                    self.collect_secrets(value)


def _merge_text(name: str, old: str, mine: str, new: str, book: Resolutions) -> str:
    if mine == old:
        return new
    if new == old or mine == new:
        return mine
    # Keep Git's non-overlapping edits; only expose complete conflict blocks.
    marker = "RINNE_UPDATE_"
    for text in (old, mine, new):
        if any(
            line.startswith(("<<<<<<<", "|||||||", "=======", ">>>>>>>"))
            for line in text.splitlines()
        ):
            return book.choose(name, "整份文件（含合并标记）", mine, new)
    with tempfile.TemporaryDirectory(prefix="rinne-merge-") as directory:
        paths = [Path(directory) / key for key in ("mine", "base", "new")]
        for path, text in zip(paths, (mine, old, new), strict=True):
            path.write_text(text, encoding="utf-8", newline="")
        result = subprocess.run(
            [
                "git",
                "merge-file",
                "--diff3",
                "-p",
                "-L",
                marker + "MINE",
                "-L",
                marker + "BASE",
                "-L",
                marker + "NEW",
                *map(str, paths),
            ],
            capture_output=True,
            check=False,
        )
    if result.returncode > 127 or result.returncode < 0:
        return book.choose(name, "整份文件（无法自动合并）", mine, new)
    merged = result.stdout.decode("utf-8")
    pattern = re.compile(
        r"^<<<<<<< RINNE_UPDATE_MINE\n(.*?)"
        r"^\|\|\|\|\|\|\| RINNE_UPDATE_BASE\n(.*?)"
        r"^=======\n(.*?)^>>>>>>> RINNE_UPDATE_NEW(?:\n|$)",
        re.M | re.S,
    )
    count = 0

    def select(match: re.Match) -> str:
        nonlocal count
        count += 1
        return book.choose(name, f"冲突代码段 {count}", match[1], match[3])

    output = pattern.sub(select, merged)
    if result.returncode and not count:
        return book.choose(name, "整份文件（无法分解冲突）", mine, new)
    return output


def _merge_file(
    name: str,
    old: bytes | None,
    mine: bytes | None,
    new: bytes | None,
    book: Resolutions,
) -> bytes | None:
    if mine == old:
        return new
    if mine == new or new == old:
        return mine
    texts = [_text(data) for data in (old, mine, new)]
    if any(text is None for text in texts):
        return book.choose(name, "整份文件（新增、删除或二进制内容）", mine, new)
    base_text, user_text, next_text = texts
    return _merge_text(name, base_text, user_text, next_text, book).encode("utf-8")


def _personal_file(
    name: str,
    old: bytes | None,
    mine: bytes | None,
    new: bytes | None,
    book: Resolutions,
) -> bytes | None:
    variable = PYTHON_KEY_FILES.get(name)
    texts = [_text(data) for data in (old, mine, new)]
    if any(text is None for text in texts):
        return _merge_file(name, old, mine, new, book)
    base, user, target = texts
    if variable:
        value, _, _ = _key_assignment(user, variable)
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            book.secrets.add(value.value)
            placeholder = "'__RINNE_PERSONAL_KEY__'"
            merged = _merge_text(
                name,
                _replace_key_value(base, variable, placeholder),
                _replace_key_value(user, variable, placeholder),
                _replace_key_value(target, variable, placeholder),
                book,
            )
            return _replace_key_value(merged, variable, repr(value.value)).encode(
                "utf-8"
            )
    if name == "启动凛祢.bat":
        pattern = re.compile(r'(?im)^set "TTS_DIR=([^"\r\n]*)"\s*$')
        matches = [list(pattern.finditer(text)) for text in texts]
        if all(len(items) == 1 for items in matches):
            value = matches[1][0].group(1)
            neutral = [pattern.sub('set "TTS_DIR="', text) for text in texts]
            merged = _merge_text(name, *neutral, book)
            merged = pattern.sub(lambda _: f'set "TTS_DIR={value}"', merged)
            return merged.replace("\n", "\r\n").encode("utf-8")
        raise UpdateError("启动脚本的 TTS_DIR 行无法唯一定位，请保留原来的 set 写法")
    return _merge_file(name, old, mine, new, book)


def _prepare_update(release_tag: str | None, choices: Mapping | None = None) -> dict:
    _check_checkout()
    head = _git("rev-parse", "HEAD")
    release = _latest_release_tag(release_tag) if release_tag else _latest_release_tag()
    if release is None:
        return {
            "message": "暂无正式应用版本",
            "conflicts": [],
            "snapshot": "",
            "target": head,
        }
    _git("fetch", "origin", "tag", release)
    target = _git("rev-parse", f"refs/tags/{release}^{{commit}}")
    legacy = False
    if head != target and _git("merge-base", head, target) != head:
        rewritten = LEGACY_RELEASE_BASES.get(head)
        if not rewritten or _git("merge-base", rewritten, target) != rewritten:
            raise UpdateError(
                "本地提交与正式版本分叉，不能自动改写提交历史；未修改文件"
            )
        legacy = True
    old_tree, new_tree = _tree(head), _tree(target)
    incoming = {
        name
        for name in old_tree.keys() | new_tree.keys()
        if old_tree.get(name) != new_tree.get(name)
    }
    if any(_protected(name) for name in incoming):
        raise UpdateError("新版涉及已跟踪的个人记忆，请人工确认；未修改文件")
    local = set(
        filter(None, _git("diff", "--name-only", "-z", "HEAD", strip=False).split("\0"))
    )
    if "frontend" in local:
        raise UpdateError("前端子模块有本地改动，请先保留并处理；未修改文件")
    candidates = (incoming | local | {"conf.yaml"} | set(PYTHON_KEY_FILES)) - {
        "frontend"
    }
    if "启动凛祢.bat" in old_tree or "启动凛祢.bat" in new_tree:
        candidates.add("启动凛祢.bat")
    candidates = {name for name in candidates if not _protected(name)}
    originals, outputs = {}, {}
    book = Resolutions(choices)
    config_bytes = _read_local("conf.yaml")
    if config_bytes is None:
        raise UpdateError("缺少 conf.yaml；未修改文件")
    user_config = config_bytes.decode("utf-8-sig")
    overlay = _read_local("conf.local.yaml")
    if overlay is not None:
        user_config = apply_local_overlay(user_config, overlay.decode("utf-8-sig"))
    book.collect_secrets(_parse_config(user_config))
    # Collect private Python keys before previewing any other file.
    for name, variable in PYTHON_KEY_FILES.items():
        source = _text(_read_local(name))
        if source:
            value, _, _ = _key_assignment(source, variable)
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                book.secrets.add(value.value)
    for name in sorted(candidates):
        for tree in (old_tree, new_tree):
            if name in tree and tree[name][0] not in {"100644", "100755"}:
                raise UpdateError(f"不支持自动更新链接或子模块文件：{name}")
        mine = _read_local(name)
        originals[name] = mine
        old = _blob(head, name, old_tree)
        new = _blob(target, name, new_tree)
        if name == "conf.yaml":
            if old is None or new is None:
                raise UpdateError("正式版本缺少 conf.yaml；未修改文件")
            merged, _ = merge_config(
                old.decode("utf-8-sig"),
                user_config,
                new.decode("utf-8-sig"),
                lambda section, user, nxt: book.choose(name, section, user, nxt),
            )
            output = merged.encode("utf-8")
        elif name in PYTHON_KEY_FILES or name == "启动凛祢.bat":
            output = _personal_file(name, old, mine, new, book)
        else:
            output = _merge_file(name, old, mine, new, book)
        outputs[name] = output
    known = {item["id"] for item in book.items}
    if set(book.choices) - known:
        raise UpdateError("文件内容或冲突已变化，请重新检查并选择；未修改文件")
    signature = {
        "head": head,
        "target": target,
        "overlay": _digest(overlay),
        "files": {name: _digest(data) for name, data in originals.items()},
        "status": _git(
            "status", "--porcelain=v1", "-z", "--untracked-files=all", strip=False
        ),
    }
    snapshot = _digest(json.dumps(signature, sort_keys=True).encode("utf-8"))
    return {
        "message": "请确认冲突后更新"
        if book.items
        else ("已经是最新版" if head == target and overlay is None else "可以安全更新"),
        "head": head,
        "target": target,
        "release": release,
        "legacy": legacy,
        "conflicts": book.items,
        "snapshot": snapshot,
        "originals": originals,
        "outputs": outputs,
        "overlay": overlay,
        "config": user_config,
        "local": local,
        "incoming": incoming,
        "unresolved": known - set(book.choices),
        "status": signature["status"],
    }


def _public_plan(plan: dict) -> dict:
    return {key: plan[key] for key in ("message", "conflicts", "snapshot", "target")}


def _store_file(name: str, data: bytes | None) -> None:
    path = _safe_path(name)
    if data is None:
        path.unlink(missing_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, data)


def _execute_plan(plan: dict) -> tuple[str, Path | None]:
    if "head" not in plan:
        return plan["message"], None
    _ensure_backend_stopped(plan["config"])
    for name, content in plan["outputs"].items():
        if name.endswith(".py") and content is not None:
            try:
                ast.parse(content.decode("utf-8-sig"), filename=name)
            except (SyntaxError, UnicodeError) as error:
                raise UpdateError(
                    f"选择后的 Python 文件无法通过语法检查：{name}；未修改文件"
                ) from error
    if plan["head"] == plan["target"] and plan["overlay"] is None:
        return "已经是最新版", None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup = Path(_git("rev-parse", "--git-path", f"rinne-update-backups/{stamp}"))
    if not backup.is_absolute():
        backup = ROOT / backup
    backup.mkdir(parents=True, exist_ok=False)
    index_path = Path(_git("rev-parse", "--git-path", "index"))
    if not index_path.is_absolute():
        index_path = ROOT / index_path
    original_index = index_path.read_bytes()
    (backup / "index").write_bytes(original_index)
    manifest = {}
    for i, (name, content) in enumerate(plan["originals"].items()):
        filename = f"{i}.bin" if content is not None else None
        if filename:
            (backup / filename).write_bytes(content)
        manifest[name] = filename
    if plan["overlay"] is not None:
        (backup / "conf.local.yaml").write_bytes(plan["overlay"])
    (backup / "manifest.json").write_text(
        json.dumps(
            {"head": plan["head"], "target": plan["target"], "files": manifest},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    old_tree = _tree(plan["head"])
    try:
        # Only return reviewed, backed-up local edits to the base for fast-forward.
        for name in plan["originals"]:
            if name in plan["local"]:
                _store_file(name, _blob(plan["head"], name, old_tree))
            elif name not in old_tree and plan["originals"][name] is not None:
                # The reviewed untracked collision is already backed up.
                _store_file(name, None)
        if plan["local"]:
            names = [name for name in plan["originals"] if name in plan["local"]]
            for offset in range(0, len(names), 32):
                _git("add", "--", *names[offset : offset + 32])
        _git("diff", "--cached", "--quiet")
        if plan["head"] != plan["target"]:
            if plan["legacy"]:
                _git("branch", f"codex/pre-update-{stamp}", plan["head"])
                _git("checkout", "--no-overwrite-ignore", "-B", "main", plan["target"])
            else:
                _git("merge", "--no-overwrite-ignore", "--ff-only", plan["target"])
        for name, content in plan["outputs"].items():
            _store_file(name, content)
        if plan["head"] != plan["target"]:
            _git("submodule", "update", "--init", "--recursive")
        _repair_game_asset_manifests(set(plan["local"]))
        if plan["overlay"] is not None:
            _safe_path("conf.local.yaml").unlink()
    except Exception as error:
        # Restore precisely the reviewed files, never reset/clean the workspace.
        try:
            current = _git("rev-parse", "HEAD")
            if current not in {plan["head"], plan["target"]}:
                raise UpdateError("更新期间 HEAD 被外部改变")
            if current != plan["head"]:
                _git("update-ref", "HEAD", plan["head"], current)
            for name, content in plan["originals"].items():
                _store_file(name, content)
            _atomic_write(index_path, original_index)
            if plan["overlay"] is not None:
                _store_file("conf.local.yaml", plan["overlay"])
            _git("submodule", "update", "--init", "--recursive")
        except Exception as rollback_error:
            raise UpdateError(
                f"更新未完成，自动恢复也未完成；请勿继续更新。备份：{backup}"
            ) from rollback_error
        raise UpdateError(f"更新未完成，原文件已恢复。备份：{backup}") from error
    return f"更新完成；文件备份：{backup}。请重新启动后端和客户端", backup


def _apply_plan(plan: dict) -> tuple[str, Path | None]:
    if "head" not in plan:
        return plan["message"], None
    lock = Path(_git("rev-parse", "--git-path", "rinne-update.lock"))
    if not lock.is_absolute():
        lock = ROOT / lock
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise UpdateError(
            "另一个更新正在进行，或上次更新被中断；请先确认再重试"
        ) from error
    try:
        with handle:
            handle.write(str(os.getpid()))
        if (
            _git("rev-parse", "HEAD") != plan["head"]
            or _read_local("conf.local.yaml") != plan["overlay"]
            or any(
                _read_local(name) != data for name, data in plan["originals"].items()
            )
            or _git(
                "status", "--porcelain=v1", "-z", "--untracked-files=all", strip=False
            )
            != plan["status"]
        ):
            raise UpdateError("检查之后文件发生变化，请重新检查；未修改文件")
        return _execute_plan(plan)
    finally:
        lock.unlink(missing_ok=True)


def run_update(
    *,
    apply: bool,
    release_tag: str | None = None,
    choices: Mapping[str, str] | None = None,
    expected_snapshot: str | None = None,
    interactive: bool = False,
) -> tuple[str, Path | None]:
    plan = _prepare_update(release_tag, choices)
    if expected_snapshot is not None and expected_snapshot != plan["snapshot"]:
        raise UpdateError("检查之后文件发生了变化，请重新检查并选择；未修改文件")
    if plan.get("unresolved"):
        if not interactive or not sys.stdin.isatty():
            raise UpdateError(
                "发现需要选择的改动；未修改文件。请先手动安装 2.1.0 或之后的客户端，"
                "再用新客户端更新后端；也可在终端运行本脚本 --apply --interactive。"
            )
        decisions = dict(choices or {})
        for item in plan["conflicts"]:
            if item["id"] in decisions:
                continue
            print(f"\n文件：{item['file']}\n位置：{item['section']}")
            print(f"你的内容：\n{item['mine']}\n新版内容：\n{item['new']}")
            while True:
                answer = input("1 保留我的 / 2 采用新版 / 0 取消：").strip()
                if answer in {"0", "1", "2"}:
                    break
            if answer == "0":
                return "已取消，未修改文件", None
            decisions[item["id"]] = "mine" if answer == "1" else "new"
        return run_update(
            apply=apply,
            release_tag=release_tag,
            choices=decisions,
            expected_snapshot=plan["snapshot"],
        )
    if not apply:
        return plan["message"], None
    return _apply_plan(plan)


def main() -> int:
    parser = argparse.ArgumentParser(description="安全更新凛祢并保留本机 conf.yaml")
    parser.add_argument("--apply", action="store_true", help="备份配置后执行更新")
    parser.add_argument(
        "--plan-json", action="store_true", help="只检查并输出脱敏的选择清单"
    )
    parser.add_argument("--choices-json", help="JSON 格式的选择及检查指纹")
    parser.add_argument("--choices-file", help="客户端生成的选择文件，不包含原始配置")
    parser.add_argument("--interactive", action="store_true", help="在终端逐项选择冲突")
    parser.add_argument(
        "--release", help="只更新到此正式后端版本，例如 rinne-app-v1.2.2"
    )
    args = parser.parse_args()
    try:
        if args.plan_json:
            print(
                json.dumps(
                    _public_plan(_prepare_update(args.release)), ensure_ascii=False
                )
            )
            return 0
        raw_choices = args.choices_json
        if args.choices_file:
            raw_choices = Path(args.choices_file).read_text(encoding="utf-8")
        decisions = json.loads(raw_choices) if raw_choices else {}
        if not isinstance(decisions, dict):
            raise UpdateError("更新选择格式无效")
        message, backup = run_update(
            apply=args.apply,
            release_tag=args.release,
            choices=decisions.get("choices"),
            expected_snapshot=decisions.get("snapshot"),
            interactive=args.interactive,
        )
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
