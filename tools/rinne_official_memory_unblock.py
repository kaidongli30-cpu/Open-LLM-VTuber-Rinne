"""Explicit, local-only release of uncertain memory-check receipts."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys


def local_processes():
    """Windows' built-in process inventory; no extra runtime dependency."""
    if os.name != "nt":
        raise ValueError("windows_process_inventory_required")
    command = (
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
        "Get-CimInstance Win32_Process -ErrorAction Stop | "
        "Select-Object ProcessId,Name,ExecutablePath,CommandLine | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode:
        raise ValueError("process_inventory_failed")
    rows = json.loads(result.stdout.lstrip("\ufeff"))
    rows = [rows] if isinstance(rows, dict) else rows
    if (
        not isinstance(rows, list)
        or not rows
        or any(
            not isinstance(row, dict)
            or type(row.get("ProcessId")) is not int
            or not isinstance(row.get("Name"), str)
            for row in rows
        )
    ):
        raise ValueError("invalid_process_inventory")
    return rows


def running_backends(project_root):
    matches = []
    for row in local_processes():
        command = (row.get("CommandLine") or "").replace("\\", "/").casefold()
        if any(
            name in command
            for name in ("run_server.py", "start_rinne_qq.py", "run_qq_backend.py")
        ):
            matches.append(row["ProcessId"])
        elif row["Name"].casefold().startswith("python") and not row.get("CommandLine"):
            raise ValueError("python_command_line_unreadable")
        elif row["Name"].casefold() in {
            "open-llm-vtuber-electron.exe",
            "open-llm-vtuber-rinne.exe",
        }:
            matches.append(row["ProcessId"])
    return matches


def pending_memory(history_root, now=None):
    from src.open_llm_vtuber.memory.layer2_context import load_current_layer2_context
    from src.open_llm_vtuber.memory.recent_memory_context import (
        RecentMemoryContextResult,
        RecentMemoryEntry,
        _format_context,
    )
    from src.open_llm_vtuber.official_route.approved_memory import ApprovedMemoryView
    from src.open_llm_vtuber.official_route.clean_view import background_parts

    now = now or datetime.now().astimezone()
    day = (now - timedelta(hours=3)).date()
    recent = RecentMemoryContextResult(
        day, day - timedelta(days=3), day - timedelta(days=1)
    )
    for age in (1, 2, 3):
        date = day - timedelta(days=age)
        path = history_root / "recent_diary_context" / f"diary_{date.isoformat()}.md"
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            title, separator, body = text.partition("\n")
            recent.diary_entries.append(
                RecentMemoryEntry(
                    "recent_diary_context",
                    path.name,
                    date,
                    date,
                    body if separator else title,
                )
            )
    approved = ApprovedMemoryView(
        history_root / "official_route/approved_memory_view.json"
    )
    values = []
    for scope, text, options in (
        ("layer2", load_current_layer2_context(history_root).context, {"layer2": True}),
        ("recent", _format_context(recent), {"recent": True}),
    ):
        if not text:
            continue
        projected = approved.project(
            text,
            scope,
            background_parts(
                text, **{k: v for k, v in options.items() if k != "recent"}
            ),
        )
        values += [
            p["text"]
            for p in background_parts(projected, **options)
            if not p["protected"]
            and p["text"].strip()
            and not approved.approved(p["text"])
        ]
    return list(dict.fromkeys(values))


def _failed_retry_plan(directory, now=None):
    """Read a selective plan without locks, writes, model calls or raw output."""
    from src.open_llm_vtuber.official_route.content_check import fingerprint
    from src.open_llm_vtuber.official_route.memory_projection import (
        MEMORY_PROJECTION_PROMPT,
    )
    from src.open_llm_vtuber.official_route.memory_recovery import (
        _valid_attempt,
        read_attempts,
    )

    receipts = directory / "memory_check_attempts.json"
    data = read_attempts(receipts)
    if data is None or any(
        not _valid_attempt(key, row) for key, row in data["attempts"].items()
    ):
        raise ValueError("invalid_memory_attempts")
    failed = {key for key, row in data["attempts"].items() if row["status"] == "failed"}
    empty_context = fingerprint(json.dumps((), ensure_ascii=False))

    def matching_keys(source_sha):
        # Same identity as ContentChecker._identity and MemoryChecks._attempt.
        # Saved background views have no context; never guess another context.
        return {
            fingerprint(json.dumps([data["version"], kind, source_sha, empty_context]))
            for kind in (
                "memory_shared",
                "memory_layer2",
                "memory_recent",
                "memory_tool",
            )
        }

    views = []
    removed_items = 0
    for kind in ("layer2", "recent", "tool"):
        path = directory / f"official_memory_{kind}.json"
        if not path.exists():
            continue
        view = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(view, dict) or not isinstance(view.get("items"), list):
            raise ValueError("invalid_official_memory_view")
        kept = []
        for item in view["items"]:
            if not isinstance(item, dict):
                raise ValueError("invalid_official_memory_view")
            source_sha = item.get("source_sha256")
            if item.get("status") != "pending":
                kept.append(item)
                continue
            if (
                not isinstance(source_sha, str)
                or len(source_sha) != 64
                or any(c not in "0123456789abcdef" for c in source_sha)
            ):
                raise ValueError("invalid_pending_memory_identity")
            matching = matching_keys(source_sha) & data["attempts"].keys()
            # An existing done/started alias still owns this item. Removing a
            # failed alias must not discard that completed or in-flight view.
            if matching & failed and matching <= failed:
                removed_items += 1
            else:
                kept.append(item)
        if len(kept) != len(view["items"]):
            views.append((path, {**view, "items": kept}))

    values = pending_memory(directory.parent, now) if failed else []
    chars = sum(map(len, values))
    # This estimates the current Layer2/recent startup work, conservatively
    # including all currently unapproved units. It never replays a saved view.
    batches = min(2, len(values))
    estimated_cny = (
        (chars + batches * (len(MEMORY_PROJECTION_PROMPT) + 1500)) * 3 * 2.5 * 1.25
        + batches * 4096 * 12.5
    ) / 1_000_000
    result = dict(
        retry_failed=True,
        failed_receipts=len(failed),
        pending_view_items=removed_items,
        pending_segments=len(values),
        pending_characters=chars,
        estimated_cny_upper_bound=round(estimated_cny, 6),
        estimate_scope="current_layer2_and_recent_memory",
        recovery_blocked=bool(data.get("recovery_blocked")),
        records_unverified=bool(data.get("records_unverified")),
        changed=False,
    )
    return data, failed, views, result, estimated_cny


def retry_failed_receipts(
    project_root, directory, *, confirm=False, process_reader=running_backends, now=None
):
    """Explicitly forget only failed receipts and their pending view entries.

    View replacements precede receipt removal: an interrupted cleanup retains
    the failed claim and cannot silently start another paid check. Flags,
    successful/in-flight claims, ledgers and original memories are never reset.
    """
    from src.open_llm_vtuber.official_route.record_storage import (
        process_lock,
        replace_text,
    )

    directory = Path(directory)
    if process_reader(project_root):
        raise ValueError("backends_still_running")
    _, failed, _, result, estimate = _failed_retry_plan(directory, now)
    if not confirm or not failed:
        return result
    if estimate > 0.5:
        raise ValueError("recheck_estimate_exceeds_half_yuan")
    receipts = directory / "memory_check_attempts.json"
    with process_lock(receipts.with_suffix(".lock")):
        with process_lock(directory / ".records.lock"):
            if process_reader(project_root):
                raise ValueError("backends_still_running")
            data, failed, views, result, estimate = _failed_retry_plan(directory, now)
            if estimate > 0.5:
                raise ValueError("recheck_estimate_exceeds_half_yuan")
            if not failed:
                return result
            for path, view in views:
                replace_text(
                    path, json.dumps(view, ensure_ascii=False, indent=2) + "\n"
                )
            data["attempts"] = {
                key: row for key, row in data["attempts"].items() if key not in failed
            }
            replace_text(
                receipts, json.dumps(data, ensure_ascii=False, indent=2) + "\n"
            )
    result["changed"] = True
    return result


def release(
    project_root,
    directory,
    *,
    confirm=False,
    process_reader=running_backends,
    now=None,
    retry_failed=False,
):
    if retry_failed:
        return retry_failed_receipts(
            project_root,
            directory,
            confirm=confirm,
            process_reader=process_reader,
            now=now,
        )
    from src.open_llm_vtuber.official_route.record_storage import (
        process_lock,
        replace_text,
    )

    directory = Path(directory)
    receipts = directory / "memory_check_attempts.json"
    data = json.loads(receipts.read_text(encoding="utf-8-sig"))
    ledger = directory / "memory_checks_opus.jsonl"
    row_count = (
        sum(bool(line.strip()) for line in ledger.read_bytes().splitlines())
        if ledger.exists()
        else 0
    )
    values = pending_memory(directory.parent, now)
    # Relay estimate: ¥2.5/M input, ¥12.5/M
    # output, 1.25x for unknown cache writes. This is not official API pricing.
    # Reserve three tokens per character and the full 16K output per batch.
    chars = sum(map(len, values))
    batches = min(3, len(values))
    estimated_cny = (
        (chars + batches * 1500) * 3 * 2.5 * 1.25 + batches * 16000 * 12.5
    ) / 1_000_000
    result = dict(
        recovery_blocked=bool(data.get("recovery_blocked")),
        records_unverified=bool(data.get("records_unverified")),
        retained_rows=row_count,
        pending_segments=len(values),
        pending_characters=chars,
        estimated_cny_upper_bound=round(estimated_cny, 6),
        changed=False,
    )
    if not confirm:
        return result
    if estimated_cny > 1:
        raise ValueError("recheck_estimate_exceeds_one_yuan")
    if process_reader(project_root):
        raise ValueError("backends_still_running")
    with process_lock(receipts.with_suffix(".lock")):
        with process_lock(directory / ".records.lock"):
            if process_reader(project_root):
                raise ValueError("backends_still_running")
            data = json.loads(receipts.read_text(encoding="utf-8-sig"))
            # Keep the hold durable until untrusted results have been removed.
            replace_text(ledger, "")
            recent = directory / "official_memory_recent.json"
            if recent.exists():
                replace_text(
                    recent, json.dumps({"items": [], "recovery_pending": True}) + "\n"
                )
            data.update(attempts={}, recovery_blocked=False, records_unverified=False)
            replace_text(receipts, json.dumps(data) + "\n")
    result.update(
        recovery_blocked=False, records_unverified=False, retained_rows=0, changed=True
    )
    return result


def route_directory(project_root: Path) -> Path:
    import yaml

    from src.open_llm_vtuber.config_manager.agent import OfficialRouteConfig

    config = yaml.safe_load(
        (project_root / "conf.yaml").read_text(encoding="utf-8-sig")
    )
    character = config["character_config"]
    values = character["agent_config"]["agent_settings"]["basic_memory_agent"].get(
        "official_route", {}
    )
    data_dir = OfficialRouteConfig(**values).data_dir
    directory = (
        Path(data_dir)
        if data_dir
        else (project_root / "chat_history" / character["conf_uid"] / "official_route")
    )
    if not directory.is_absolute():
        directory = project_root / directory
    return directory.resolve()


def main():
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root))
    directory = route_directory(root)
    try:
        result = release(
            root, directory, confirm=args.confirm, retry_failed=args.retry_failed
        )
    except Exception as error:
        print(json.dumps({"ok": False, "error": type(error).__name__}))
        return 1
    print(json.dumps({"ok": True, **result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
