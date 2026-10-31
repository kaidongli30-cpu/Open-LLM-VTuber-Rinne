"""Recover complete memory-check records without retrying uncertain paid work."""

from dataclasses import asdict
import json
from pathlib import Path
import re

from loguru import logger

from .content_check import CheckRecord, fingerprint
from .record_storage import process_lock, replace_text


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def valid_record(value):
    row = CheckRecord(**value)
    if (
        not all(isinstance(v, str) for k, v in asdict(row).items() if k != "confidence")
        or not all(_sha(v) for v in (row.key, row.sha256, row.context_sha256))
        or row.label not in {"A", "B", "U"}
        or type(row.confidence) is not int
        or not 0 <= row.confidence <= 100
        or row.key
        != fingerprint(
            json.dumps([row.version, row.kind, row.sha256, row.context_sha256])
        )
    ):
        raise ValueError("invalid_memory_check_record")
    return asdict(row)


def read_attempts(path):
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("version"), str)
        or not isinstance(value.get("attempts"), dict)
        or type(value.get("recovery_blocked", False)) is not bool
        or type(value.get("records_unverified", False)) is not bool
    ):
        raise ValueError("invalid_memory_attempts")
    return value


def _valid_attempt(key, row):
    return (
        _sha(key)
        and isinstance(row, dict)
        and isinstance(row.get("status"), str)
        and row["status"] in {"started", "done", "failed"}
        and isinstance(row.get("kind"), str)
        and row["kind"] in {"memory_layer2", "memory_recent", "memory_tool"}
        and (
            row["status"] != "done"
            or (
                isinstance(row.get("official_text"), str)
                and isinstance(row.get("action"), str)
                and row["action"] in {"keep", "rewrite", "omit"}
            )
        )
    )


def recover_memory_storage(directory, version, *, records_uncertain=False):
    """Keep independently valid rows; never back up or guess missing results."""
    directory = Path(directory)
    path = directory / "memory_check_attempts.json"
    ledger = directory / "memory_checks_opus.jsonl"
    report = dict(
        records_kept=0, records_rejected=0, repaired=[], recovery_blocked=False
    )
    with process_lock(path.with_suffix(".lock")):
        with process_lock(directory / ".records.lock"):
            rows = []
            if ledger.exists():
                for line in ledger.read_bytes().splitlines():
                    if not line.strip():
                        continue
                    try:
                        rows.append(valid_record(json.loads(line)))
                    except (ValueError, TypeError, KeyError, UnicodeError):
                        report["records_rejected"] += 1
            report["records_kept"] = len(rows)
            views = [
                directory / f"official_memory_{kind}.json"
                for kind in ("layer2", "recent", "tool")
            ]
            # A vanished receipt beside prior outputs is not a fresh install.
            bad_attempts = not path.exists() and (
                ledger.exists() or any(view.exists() for view in views)
            )
            try:
                data = read_attempts(path)
            except (ValueError, TypeError, KeyError, UnicodeError):
                data, bad_attempts = None, True
            if data is None:
                data = dict(version=version, attempts={})
            attempts = {
                key: row
                for key, row in data["attempts"].items()
                if _valid_attempt(key, row)
            }
            bad_attempts |= len(attempts) != len(data["attempts"])
            records_unverified = (
                data.get("records_unverified", False)
                or records_uncertain
                or bool(report["records_rejected"])
            )
            blocked = (
                data.get("recovery_blocked", False)
                or bad_attempts
                or bool(report["records_rejected"])
                or records_unverified
            )
            report["recovery_blocked"] = blocked
            report["records_unverified"] = records_unverified
            if bad_attempts or report["records_rejected"] or records_uncertain:
                # Persist the stop marker before discarding damaged bytes. A
                # second crash/reload must never turn lost receipts into new calls.
                data.update(
                    attempts=attempts,
                    recovery_blocked=True,
                    records_unverified=records_unverified,
                )
                replace_text(path, json.dumps(data, ensure_ascii=False) + "\n")
                report["repaired"].append(path.name)
            if report["records_rejected"]:
                replace_text(
                    ledger,
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                )
                report["repaired"].append(ledger.name)
            for kind in ("layer2", "recent", "tool"):
                view = directory / f"official_memory_{kind}.json"
                if not view.exists():
                    continue
                try:
                    value = json.loads(view.read_text(encoding="utf-8-sig"))
                    if not isinstance(value, dict) or not isinstance(
                        value.get("items"), list
                    ):
                        raise ValueError("invalid_memory_view")
                except (ValueError, TypeError, UnicodeError):
                    replace_text(
                        view, json.dumps(dict(items=[], recovery_pending=True)) + "\n"
                    )
                    report["repaired"].append(view.name)
    if report["repaired"]:
        logger.warning(
            "[官方记忆检查恢复] 保留完整记录={}，丢弃损坏记录={}；不自动补发检查={}，原始记忆未改",
            report["records_kept"],
            report["records_rejected"],
            blocked,
        )
    return report
