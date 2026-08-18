"""Durable human-approval gate for generated diary files."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable


APPROVAL_DIRECTORY_NAME = "diary_reviews"
APPROVAL_SCHEMA_VERSION = 1
APPROVE_COMMANDS = {"approve", "确认"}
ABORT_COMMANDS = {"abort", "取消"}


class DiaryReviewError(RuntimeError):
    """The diary could not be safely approved for downstream processing."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def approval_path(history_root: str | Path, memory_day: str) -> Path:
    date.fromisoformat(memory_day)
    return (
        Path(history_root)
        / APPROVAL_DIRECTORY_NAME
        / f"diary_{memory_day}.approved.json"
    )


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{uuid.uuid4().hex[:12]}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def load_matching_approval(
    history_root: str | Path,
    memory_day: str,
    diary_path: str | Path,
) -> dict[str, Any] | None:
    """Return the approval only when it matches the diary's current bytes."""

    diary = Path(diary_path)
    marker = approval_path(history_root, memory_day)
    if not diary.is_file() or diary.stat().st_size == 0 or not marker.is_file():
        return None
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    expected = {
        "status": "approved",
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "memory_day": memory_day,
        "diary_file": diary.name,
        "diary_sha256": sha256_file(diary),
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        return None
    return value


def wait_for_diary_approval(
    history_root: str | Path,
    memory_day: str,
    diary_path: str | Path,
    *,
    input_func: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Block until the user approves the diary version currently on disk.

    A matching durable marker skips the prompt on later backend starts. If the
    diary changes after approval, its hash no longer matches and review is
    required again.
    """

    date.fromisoformat(memory_day)
    diary = Path(diary_path).resolve()
    history = Path(history_root).resolve()
    existing = load_matching_approval(history, memory_day, diary)
    if existing is not None:
        return {
            "status": "already_approved",
            "memory_day": memory_day,
            "diary_sha256": existing["diary_sha256"],
            "approval_path": str(approval_path(history, memory_day)),
        }
    if not diary.is_file() or diary.stat().st_size == 0:
        raise DiaryReviewError(f"diary_missing_or_empty:{diary}")

    reader = input_func or input
    prompt = (
        "\n日记已生成，等待人工验收……\n"
        f"文件：{diary}\n"
        "请检查或修改完成后，在当前终端输入 approve 并回车。\n"
        "如需停止本次启动，请输入 abort："
    )
    while True:
        try:
            command = reader(prompt).strip().lower()
        except EOFError as exc:
            raise DiaryReviewError(
                "diary_review_input_unavailable:请在可交互终端启动后端"
            ) from exc
        if command in ABORT_COMMANDS:
            raise DiaryReviewError("diary_review_aborted_by_user")
        if command not in APPROVE_COMMANDS:
            prompt = "请输入 approve 确认，或输入 abort 停止本次启动："
            continue
        if not diary.is_file() or diary.stat().st_size == 0:
            raise DiaryReviewError(f"diary_missing_or_empty_after_review:{diary}")
        diary_hash = sha256_file(diary)
        marker = approval_path(history, memory_day)
        approval = {
            "status": "approved",
            "schema_version": APPROVAL_SCHEMA_VERSION,
            "memory_day": memory_day,
            "approved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "diary_file": diary.name,
            "diary_sha256": diary_hash,
        }
        _write_json_atomic(marker, approval)
        return {
            "status": "approved",
            "memory_day": memory_day,
            "diary_sha256": diary_hash,
            "approval_path": str(marker),
        }
