"""Isolated delta-based replay harness for Rinne layer-two background tests.

The model emits only daily changes. This runner validates the changes, applies
them deterministically to the previous complete background, and preserves the
patch, merge audit, request, raw response, metrics, and full daily snapshot.
It never edits live desktop-pet configuration, chats, diaries, or memory files.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
import threading
import time
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import layer2_background_experiment as legacy  # noqa: E402


ExperimentError = legacy.ExperimentError
ManifestDay = legacy.ManifestDay
CATEGORIES = legacy.CATEGORIES
MODEL_SPECS = legacy.MODEL_SPECS

DEFAULT_END_DATE = date.today() - timedelta(days=1)
DEFAULT_START_DATE = DEFAULT_END_DATE - timedelta(days=6)
DEFAULT_EXPERIMENT_ROOT = Path("tmp/layer2-delta-experiment")
DEFAULT_SOURCE_DIARIES = legacy.DEFAULT_SOURCE_DIARIES
DEFAULT_OLLAMA_URL = legacy.DEFAULT_OLLAMA_URL

BACKGROUND_SCHEMA_SOURCE = TOOLS_DIR / "layer2_background_schema.json"
DELTA_SCHEMA_SOURCE = TOOLS_DIR / "layer2_background_delta_schema.json"
DELTA_PROMPT_SOURCE = TOOLS_DIR / "layer2_background_delta_system_prompt.txt"
HANDOFF_SOURCE = (
    TOOLS_DIR.parent
    / "docs"
    / "layer2_delta_successor_model_luna_handoff_2026-08-16.md"
)

BACKGROUND_SCHEMA_VERSION = "1.2-atomic-background"
DELTA_SCHEMA_VERSION = "2.4-atomic-rolling-delta"
MAX_TOTAL_OPERATIONS = 12
FORBIDDEN_SUBJECT_PATTERNS = (
    re.compile(
        r"^用户.{0,6}(?:是|属于)(?:一个|一名)?(?:由(?:代码|模型|程序)构成的)?"
        r"(?:AI|人工智能|模型|程序|桌宠|虚拟角色)(?:[，,。；;]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^用户.{0,6}由(?:代码|模型|程序)(?:构成|生成|驱动)(?:[，,。；;]|$)",
        re.IGNORECASE,
    ),
)
RINNE_FIRST_PERSON_STATE_PATTERN = re.compile(
    r"(?:^|[。！？]\s*)(?:说实话[，,]?)?(?:也许[，,]?)?\s*"
    r"我(?:知道|觉得|认为|希望|愿意|会|想|喜欢|爱|珍惜|担心|害怕|开心|难过|感到|感受|心里)"
)
USER_INTERNAL_STATE_PATTERN = re.compile(
    r"^用户.{0,120}(?:希望|愿意|想要|想|珍惜|喜欢|爱|重视|期待|担心|认为|觉得|感到)"
)
QUOTE_NORMALIZATION = str.maketrans(
    {
        "'": '"',
        "‘": '"',
        "’": '"',
        "“": '"',
        "”": '"',
    }
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _load_manifest(root: Path) -> list[ManifestDay]:
    return legacy._load_manifest(root)


def _empty_delta(current_day: date, summary: str) -> dict[str, Any]:
    return {
        "schema_version": DELTA_SCHEMA_VERSION,
        "source_date": current_day.isoformat(),
        "added_facts": [],
        "updated_facts": [],
        "removed_facts": [],
        "summary": summary,
    }


def _change_payload(item: dict[str, Any], category: str) -> dict[str, Any]:
    return {
        "fact_id": item["fact_id"],
        "category": category,
        "statement": item["statement"],
        "status": item["status"],
        "subject_scope": item["subject_scope"],
        "temporal_class": item["temporal_class"],
        "context_bucket": item["context_bucket"],
        "expected_end_on": item["expected_end_on"],
        "review_after": item["review_after"],
        "evidence_type": item["evidence_type"],
        "evidence_excerpts": copy.deepcopy(item["evidence_excerpts"]),
        "supersedes_fact_ids": copy.deepcopy(item["supersedes_fact_ids"]),
        "notes": item["notes"],
    }


def _empty_background(previous_date: date) -> dict[str, Any]:
    """Create an isolated atomic-schema snapshot without changing legacy code."""
    result = legacy._empty_background(previous_date)
    result["schema_version"] = BACKGROUND_SCHEMA_VERSION
    result["historical_overview"] = "尚无可用的历史背景。"
    return result


def _lifecycle_consistency_errors(
    *,
    fact_id: str,
    status: Any,
    temporal_class: Any,
    context_bucket: Any,
) -> list[str]:
    """Reject field combinations that would hide a fact in the wrong view."""
    errors: list[str] = []
    if context_bucket == "current":
        if status in {"historical", "superseded"}:
            errors.append(
                f"{fact_id}: current context cannot use {status} status"
            )
        if temporal_class == "historical":
            errors.append(
                f"{fact_id}: current context cannot use historical temporal_class"
            )
    elif context_bucket == "history":
        if status == "active":
            errors.append(f"{fact_id}: history context cannot use active status")
        if temporal_class != "historical":
            errors.append(
                f"{fact_id}: history context must use historical temporal_class"
            )
    return errors


def _legacy_validation_view(background: dict[str, Any]) -> dict[str, Any]:
    """Project the atomic snapshot onto the old validator's compatible shape."""
    projected = copy.deepcopy(background)
    projected["schema_version"] = legacy.SCHEMA_VERSION
    projected.pop("historical_overview", None)
    for section in projected.get("sections", []):
        for item in section.get("items", []):
            for key in (
                "subject_scope",
                "temporal_class",
                "context_bucket",
                "expected_end_on",
                "review_after",
                "evidence_history",
            ):
                item.pop(key, None)
    return projected


def _validate_background(
    background: dict[str, Any],
    *,
    expected_day: date,
    available_days: set[str],
    previous: dict[str, Any] | None,
    missing_diary: bool = False,
) -> list[str]:
    """Validate the new snapshot and reuse the mature old bookkeeping checks."""
    errors: list[str] = []
    required_root = {
        "schema_version",
        "as_of_date",
        "current_overview",
        "historical_overview",
        "sections",
        "daily_update",
        "uncertainty_warnings",
    }
    if set(background) != required_root:
        errors.append(
            "background root keys differ: "
            f"missing={sorted(required_root - set(background))}, "
            f"extra={sorted(set(background) - required_root)}"
        )
    if background.get("schema_version") != BACKGROUND_SCHEMA_VERSION:
        errors.append(f"schema_version is not {BACKGROUND_SCHEMA_VERSION}")
    if not isinstance(background.get("historical_overview"), str):
        errors.append("historical_overview is not a string")

    projected = _legacy_validation_view(background)
    projected_previous = (
        _legacy_validation_view(previous) if previous is not None else None
    )
    errors.extend(
        legacy._validate_background(
            projected,
            expected_day=expected_day,
            available_days=available_days,
            previous=projected_previous,
            missing_diary=missing_diary,
        )
    )

    required_item = {
        "fact_id",
        "statement",
        "status",
        "subject_scope",
        "temporal_class",
        "context_bucket",
        "expected_end_on",
        "review_after",
        "first_observed_on",
        "last_updated_on",
        "source_dates",
        "evidence_type",
        "evidence_excerpts",
        "evidence_history",
        "supersedes_fact_ids",
        "notes",
    }
    valid_dates = available_days | {expected_day.isoformat()}
    for section in background.get("sections", []):
        for item in section.get("items", []):
            fact_id = item.get("fact_id", "<unknown>")
            if set(item) != required_item:
                errors.append(f"{fact_id}: atomic item keys differ")
                continue
            if item["subject_scope"] not in {"user", "user_rinne_relationship"}:
                errors.append(f"{fact_id}: invalid subject_scope")
            if item["temporal_class"] not in {
                "persistent", "ongoing", "scheduled", "transient", "historical"
            }:
                errors.append(f"{fact_id}: invalid temporal_class")
            if item["context_bucket"] not in {"current", "history"}:
                errors.append(f"{fact_id}: invalid context_bucket")
            errors.extend(
                _lifecycle_consistency_errors(
                    fact_id=fact_id,
                    status=item["status"],
                    temporal_class=item["temporal_class"],
                    context_bucket=item["context_bucket"],
                )
            )
            for field in ("expected_end_on", "review_after"):
                value = item[field]
                if value is not None and (
                    not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
                ):
                    errors.append(f"{fact_id}: invalid {field}")
            history = item["evidence_history"]
            if not isinstance(history, list) or not history:
                errors.append(f"{fact_id}: evidence_history is empty")
                continue
            valid_history_entries: list[dict[str, Any]] = []
            for evidence in history:
                if not isinstance(evidence, dict) or set(evidence) != {
                    "source_date", "evidence_type", "evidence_excerpts"
                }:
                    errors.append(f"{fact_id}: invalid evidence_history entry")
                    continue
                valid_history_entries.append(evidence)
                if evidence["source_date"] not in valid_dates:
                    errors.append(f"{fact_id}: evidence_history has unavailable date")
                if evidence["source_date"] not in item["source_dates"]:
                    errors.append(f"{fact_id}: evidence_history date is absent from source_dates")
                if evidence["evidence_type"] not in {
                    "diary_explicit_user_statement",
                    "diary_reported_event",
                    "diary_interpretation",
                }:
                    errors.append(f"{fact_id}: invalid evidence_history evidence_type")
                excerpts = evidence["evidence_excerpts"]
                if not isinstance(excerpts, list) or not excerpts or any(
                    not isinstance(text, str) or not text for text in excerpts
                ):
                    errors.append(f"{fact_id}: invalid evidence_history excerpts")
            if len(valid_history_entries) == len(history):
                history_dates = [entry["source_date"] for entry in history]
                if len(history_dates) != len(set(history_dates)):
                    errors.append(f"{fact_id}: duplicate evidence_history source_date")
                if item["source_dates"] != history_dates:
                    errors.append(
                        f"{fact_id}: source_dates must exactly match evidence_history"
                    )
                latest = history[-1]
                if (
                    item["evidence_type"] != latest["evidence_type"]
                    or item["evidence_excerpts"] != latest["evidence_excerpts"]
                ):
                    errors.append(
                        f"{fact_id}: current evidence must match latest evidence_history"
                    )
    if isinstance(background.get("current_overview"), str) and (
        background["current_overview"] != _deterministic_overview(background)
    ):
        errors.append("current_overview differs from deterministic current facts")
    if isinstance(background.get("historical_overview"), str) and (
        background["historical_overview"]
        != _deterministic_overview(background, context_bucket="history")
    ):
        errors.append("historical_overview differs from deterministic history facts")
    return errors


def _render_markdown(background: dict[str, Any], generation_mode: str) -> str:
    lines = [
        f"# 用户背景快照：{background['as_of_date']}",
        "",
        f"- 生成方式：`{generation_mode}`",
        f"- 当日来源：`{background['daily_update']['source_date']}`",
        f"- 是否无变化：`{background['daily_update']['no_change']}`",
        "",
        "## 当前概览",
        "",
        background["current_overview"],
        "",
        "## 历史概览",
        "",
        background["historical_overview"],
        "",
    ]
    for section in background["sections"]:
        lines.extend([f"## {section['category']}", ""])
        if not section["items"]:
            lines.extend(["（空）", ""])
            continue
        for item in section["items"]:
            evidence = " / ".join(item["evidence_excerpts"])
            lines.extend(
                [
                    f"- **{item['fact_id']}** [{item['status']}] {item['statement']}",
                    f"  - 状态：{item['status']}；范围：{item['subject_scope']}；"
                    f"时间：{item['temporal_class']}；上下文：{item['context_bucket']}",
                    f"  - 首次/更新：{item['first_observed_on']} / {item['last_updated_on']}",
                    f"  - 预计结束/复核：{item['expected_end_on'] or '无'} / "
                    f"{item['review_after'] or '无'}",
                    f"  - 来源：{', '.join(item['source_dates'])}；"
                    f"最新类型：{item['evidence_type']}",
                    f"  - 最新证据摘录：{evidence}",
                    f"  - 替代：{', '.join(item['supersedes_fact_ids']) or '无'}",
                    f"  - 备注：{item['notes'] or '无'}",
                    "  - 累积证据：",
                ]
            )
            for evidence_entry in item["evidence_history"]:
                lines.append(
                    "    - "
                    f"{evidence_entry['source_date']} "
                    f"[{evidence_entry['evidence_type']}] "
                    + " / ".join(evidence_entry["evidence_excerpts"])
                )
        lines.append("")
    update = background["daily_update"]
    lines.extend(
        [
            "## 当日变更",
            "",
            f"- 新增：{', '.join(update['added_fact_ids']) or '无'}",
            f"- 更新：{', '.join(update['updated_fact_ids']) or '无'}",
            f"- 失效/替代：{', '.join(update['superseded_fact_ids']) or '无'}",
            f"- 删除：{', '.join(update['removed_fact_ids']) or '无'}",
            f"- 摘要：{update['summary']}",
            "",
            "## 不确定与禁止推断",
            "",
        ]
    )
    warnings = background["uncertainty_warnings"]
    lines.extend([f"- {item}" for item in warnings] or ["（空）"])
    return "\n".join(lines) + "\n"


def _carry_forward(previous: dict[str, Any], current_day: date) -> dict[str, Any]:
    carried = copy.deepcopy(previous)
    carried["as_of_date"] = current_day.isoformat()
    carried["daily_update"] = {
        "source_date": None,
        "added_fact_ids": [],
        "updated_fact_ids": [],
        "superseded_fact_ids": [],
        "removed_fact_ids": [],
        "no_change": True,
        "summary": "当日没有冻结日记，程序确定性沿用上一日背景。",
    }
    return carried


def _compact_previous_view(
    previous: dict[str, Any], *, review_date: date | None = None
) -> dict[str, Any]:
    """Keep model context bounded while the stored snapshot stays fully auditable."""
    compact_sections: list[dict[str, Any]] = []
    fields = (
        "fact_id", "statement", "status", "subject_scope", "temporal_class",
        "context_bucket", "expected_end_on", "review_after", "first_observed_on",
        "last_updated_on", "source_dates", "supersedes_fact_ids", "notes",
    )
    for section in previous["sections"]:
        compact_sections.append(
            {
                "category": section["category"],
                "items": [
                    {field: copy.deepcopy(item[field]) for field in fields}
                    for item in section["items"]
                    if item["context_bucket"] == "current"
                ],
            }
        )
    historical_fact_count = sum(
        1
        for section in previous["sections"]
        for item in section["items"]
        if item["context_bucket"] == "history"
    )
    result = {
        "schema_version": previous["schema_version"],
        "as_of_date": previous["as_of_date"],
        "current_overview": previous["current_overview"],
        "history_archive": {
            "included_in_daily_context": False,
            "fact_count": historical_fact_count,
        },
        "sections": compact_sections,
    }
    if review_date is not None:
        due_ids: list[str] = []
        for section in previous["sections"]:
            for item in section["items"]:
                due_dates = (item["expected_end_on"], item["review_after"])
                if item["context_bucket"] == "current" and any(
                    value is not None and value <= review_date.isoformat()
                    for value in due_dates
                ):
                    due_ids.append(item["fact_id"])
        result["review_due_fact_ids"] = sorted(due_ids)
    return result


def _previous_fact_map(
    background: dict[str, Any],
) -> dict[str, tuple[str, dict[str, Any]]]:
    return legacy._categorized_items_by_id(background)


def _validate_subject_statement(statement: str, fact_id: str) -> list[str]:
    errors: list[str] = []
    for pattern in FORBIDDEN_SUBJECT_PATTERNS:
        if pattern.search(statement):
            errors.append(
                f"{fact_id}: subject reversal guard matched statement: {statement}"
            )
    if not statement.strip().startswith("用户"):
        errors.append(f"{fact_id}: statement must start with 用户")
    return errors


def _validate_delta(
    delta: dict[str, Any],
    *,
    current_day: date,
    previous: dict[str, Any],
    diary_text: str,
    evidence_is_supported: Callable[[str], bool] | None = None,
) -> list[str]:
    errors: list[str] = []
    required_root = {
        "schema_version",
        "source_date",
        "added_facts",
        "updated_facts",
        "removed_facts",
        "summary",
    }
    if set(delta) != required_root:
        errors.append(
            "delta root keys differ: "
            f"missing={sorted(required_root - set(delta))}, "
            f"extra={sorted(set(delta) - required_root)}"
        )
    if delta.get("schema_version") != DELTA_SCHEMA_VERSION:
        errors.append(f"delta schema_version is not {DELTA_SCHEMA_VERSION}")
    if delta.get("source_date") != current_day.isoformat():
        errors.append("delta source_date does not match the processed day")
    if not isinstance(delta.get("summary"), str):
        errors.append("delta summary is not a string")
    previous_map = _previous_fact_map(previous)
    add_ids: set[str] = set()
    update_ids: set[str] = set()
    remove_ids: set[str] = set()
    statement_fact_ids: dict[str, str] = {}
    evidence_fact_ids: dict[tuple[str, tuple[str, ...]], str] = {}
    total_operations = 0

    required_change = {
        "fact_id",
        "category",
        "statement",
        "status",
        "subject_scope",
        "temporal_class",
        "context_bucket",
        "expected_end_on",
        "review_after",
        "evidence_type",
        "evidence_excerpts",
        "supersedes_fact_ids",
        "notes",
    }
    for list_name, expected_existing, seen in (
        ("added_facts", False, add_ids),
        ("updated_facts", True, update_ids),
    ):
        changes = delta.get(list_name)
        if not isinstance(changes, list):
            errors.append(f"delta {list_name} is not a list")
            continue
        total_operations += len(changes)
        for change in changes:
            if not isinstance(change, dict):
                errors.append(f"{list_name} contains a non-object")
                continue
            if set(change) != required_change:
                errors.append(f"{list_name} item has missing or extra keys")
            fact_id = change.get("fact_id")
            category = change.get("category")
            if not isinstance(fact_id, str) or not fact_id:
                errors.append(f"{list_name} contains an invalid fact_id")
                continue
            if fact_id in seen:
                errors.append(f"{list_name} contains duplicate fact_id {fact_id}")
            seen.add(fact_id)
            exists = fact_id in previous_map
            if exists is not expected_existing:
                requirement = "existing" if expected_existing else "new"
                errors.append(f"{fact_id}: {list_name} fact_id must be {requirement}")
            if category not in CATEGORIES:
                errors.append(f"{fact_id}: invalid category {category}")
            statement = change.get("statement")
            if not isinstance(statement, str) or not statement.strip():
                errors.append(f"{fact_id}: statement is empty or invalid")
            else:
                errors.extend(_validate_subject_statement(statement, fact_id))
                normalized_statement = statement.strip()
                previous_statement_id = statement_fact_ids.get(normalized_statement)
                if previous_statement_id is not None:
                    errors.append(
                        f"{fact_id}: duplicate statement already proposed by "
                        f"{previous_statement_id}"
                    )
                else:
                    statement_fact_ids[normalized_statement] = fact_id
            if change.get("status") not in {
                "active",
                "historical",
                "superseded",
                "uncertain",
            }:
                errors.append(f"{fact_id}: invalid status")
            if change.get("subject_scope") not in {
                "user",
                "user_rinne_relationship",
            }:
                errors.append(f"{fact_id}: invalid subject_scope")
            if change.get("temporal_class") not in {
                "persistent", "ongoing", "scheduled", "transient", "historical"
            }:
                errors.append(f"{fact_id}: invalid temporal_class")
            if change.get("context_bucket") not in {"current", "history"}:
                errors.append(f"{fact_id}: invalid context_bucket")
            errors.extend(
                _lifecycle_consistency_errors(
                    fact_id=fact_id,
                    status=change.get("status"),
                    temporal_class=change.get("temporal_class"),
                    context_bucket=change.get("context_bucket"),
                )
            )
            for field in ("expected_end_on", "review_after"):
                value = change.get(field)
                if value is not None and (
                    not isinstance(value, str)
                    or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
                ):
                    errors.append(f"{fact_id}: invalid {field}")
            evidence_type = change.get("evidence_type")
            if evidence_type not in {
                "diary_explicit_user_statement",
                "diary_reported_event",
                "diary_interpretation",
            }:
                errors.append(f"{fact_id}: invalid evidence_type")
            if (
                evidence_type == "diary_interpretation"
                and change.get("status") != "uncertain"
            ):
                errors.append(
                    f"{fact_id}: diary_interpretation must use uncertain status"
                )
            excerpts = change.get("evidence_excerpts")
            if not isinstance(excerpts, list) or not excerpts:
                errors.append(f"{fact_id}: evidence_excerpts is empty")
            else:
                for excerpt in excerpts:
                    if not isinstance(excerpt, str) or not excerpt:
                        errors.append(f"{fact_id}: evidence excerpt is invalid")
                    elif not (
                        evidence_is_supported(excerpt)
                        if evidence_is_supported is not None
                        else excerpt in diary_text
                    ):
                        errors.append(
                            f"{fact_id}: evidence excerpt is not verbatim in today's diary: "
                            f"{excerpt}"
                        )
                    elif (
                        evidence_type == "diary_interpretation"
                        and RINNE_FIRST_PERSON_STATE_PATTERN.search(excerpt)
                    ):
                        errors.append(
                            f"{fact_id}: diary-author first-person state cannot support "
                            f"a user fact: {excerpt}"
                        )
                    elif (
                        isinstance(statement, str)
                        and USER_INTERNAL_STATE_PATTERN.search(statement)
                        and RINNE_FIRST_PERSON_STATE_PATTERN.search(excerpt)
                        and evidence_type != "diary_explicit_user_statement"
                    ):
                        errors.append(
                            f"{fact_id}: diary-author first-person state cannot support "
                            f"the asserted user internal state: {excerpt}"
                        )
                if isinstance(category, str) and all(
                    isinstance(excerpt, str) for excerpt in excerpts
                ):
                    evidence_signature = (category, tuple(excerpts))
                    previous_evidence_id = evidence_fact_ids.get(evidence_signature)
                    if previous_evidence_id is not None:
                        errors.append(
                            f"{fact_id}: identical evidence already supports "
                            f"{previous_evidence_id} in the same category"
                        )
                    else:
                        evidence_fact_ids[evidence_signature] = fact_id
            supersedes = change.get("supersedes_fact_ids")
            if not isinstance(supersedes, list) or any(
                not isinstance(item, str) for item in supersedes
            ):
                errors.append(f"{fact_id}: supersedes_fact_ids is invalid")
            elif any(item not in previous_map for item in supersedes):
                errors.append(f"{fact_id}: supersedes an unknown prior fact")
            elif fact_id in supersedes:
                errors.append(f"{fact_id}: cannot supersede itself")
            if not isinstance(change.get("notes"), str):
                errors.append(f"{fact_id}: notes is not a string")
            if exists and _change_payload(previous_map[fact_id][1], category) == change:
                errors.append(f"{fact_id}: unchanged fact was emitted as an update")

    removals = delta.get("removed_facts")
    required_removal = {
        "fact_id",
        "category",
        "removal_basis",
        "evidence_type",
        "evidence_excerpts",
        "reason",
    }
    if not isinstance(removals, list):
        errors.append("delta removed_facts is not a list")
    else:
        total_operations += len(removals)
        for removal in removals:
            if not isinstance(removal, dict):
                errors.append("removed_facts contains a non-object")
                continue
            if set(removal) != required_removal:
                errors.append("removed_facts item has missing or extra keys")
            fact_id = removal.get("fact_id")
            category = removal.get("category")
            if not isinstance(fact_id, str) or not fact_id:
                errors.append("removed_facts contains an invalid fact_id")
                continue
            if fact_id in remove_ids:
                errors.append(f"removed_facts contains duplicate fact_id {fact_id}")
            remove_ids.add(fact_id)
            if fact_id not in previous_map:
                errors.append(f"{fact_id}: removed fact does not exist")
            elif previous_map[fact_id][0] != category:
                errors.append(f"{fact_id}: removal category differs from prior fact")
            removal_basis = removal.get("removal_basis")
            if removal_basis not in {
                "explicitly_changed",
                "superseded_by_current_fact",
                "duplicate_or_invalid",
                "stale_for_current_context",
            }:
                errors.append(f"{fact_id}: invalid removal_basis")
            timeline_removal = removal_basis in {
                "duplicate_or_invalid",
                "stale_for_current_context",
            }
            excerpts = removal.get("evidence_excerpts")
            if not isinstance(excerpts, list):
                errors.append(f"{fact_id}: removal evidence_excerpts is invalid")
            elif timeline_removal:
                if excerpts:
                    errors.append(
                        f"{fact_id}: timeline removal evidence_excerpts must be empty"
                    )
            elif not excerpts:
                errors.append(f"{fact_id}: removal evidence_excerpts is empty")
            else:
                for excerpt in excerpts:
                    if not isinstance(excerpt, str) or excerpt not in diary_text:
                        errors.append(
                            f"{fact_id}: removal evidence is not verbatim in today's diary"
                        )
            evidence_type = removal.get("evidence_type")
            if timeline_removal:
                if evidence_type != "background_timeline":
                    errors.append(
                        f"{fact_id}: timeline removal must use background_timeline"
                    )
            elif evidence_type not in {
                "diary_explicit_user_statement",
                "diary_reported_event",
                "diary_interpretation",
            }:
                errors.append(f"{fact_id}: invalid removal evidence_type")
            if not isinstance(removal.get("reason"), str) or not removal["reason"]:
                errors.append(f"{fact_id}: removal reason is empty")

    if total_operations > MAX_TOTAL_OPERATIONS:
        errors.append(
            f"delta contains {total_operations} operations; maximum is "
            f"{MAX_TOTAL_OPERATIONS}"
        )
    if (add_ids & update_ids) or (add_ids & remove_ids) or (update_ids & remove_ids):
        errors.append("delta operation fact_id sets are not disjoint")
    superseded_targets = {
        target
        for change in [*delta.get("added_facts", []), *delta.get("updated_facts", [])]
        if isinstance(change, dict)
        for target in change.get("supersedes_fact_ids", [])
        if isinstance(target, str)
    }
    for change in [*delta.get("added_facts", []), *delta.get("updated_facts", [])]:
        if not isinstance(change, dict):
            continue
        replacement_id = change.get("fact_id")
        if not isinstance(replacement_id, str):
            continue
        for target in change.get("supersedes_fact_ids", []):
            if not isinstance(target, str):
                continue
            if target in update_ids:
                errors.append(
                    f"{target}: cannot be updated and superseded by "
                    f"{replacement_id} in the same delta"
                )
            if target in remove_ids:
                errors.append(
                    f"{target}: cannot be removed and superseded by "
                    f"{replacement_id} in the same delta"
                )
            prior = previous_map.get(target)
            if prior is not None and (
                prior[1].get("context_bucket") == "history"
                or prior[1].get("status") == "superseded"
            ):
                errors.append(
                    f"{replacement_id}: cannot supersede already historical "
                    f"fact {target}"
                )
    superseded_target_count = sum(
        1
        for change in [*delta.get("added_facts", []), *delta.get("updated_facts", [])]
        if isinstance(change, dict)
        for target in change.get("supersedes_fact_ids", [])
        if isinstance(target, str)
    )
    if len(superseded_targets) != superseded_target_count:
        errors.append("a prior fact cannot be superseded more than once in one delta")
    return errors


def _quarantine_invalid_operations(
    delta: dict[str, Any], validation_errors: list[str]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    operation_locations: dict[str, tuple[str, dict[str, Any]]] = {}
    for list_name in ("added_facts", "updated_facts", "removed_facts"):
        operations = delta.get(list_name)
        if not isinstance(operations, list):
            continue
        for operation in operations:
            if not isinstance(operation, dict):
                continue
            fact_id = operation.get("fact_id")
            if isinstance(fact_id, str) and fact_id:
                operation_locations[fact_id] = (list_name, operation)

    errors_by_fact_id: dict[str, list[str]] = {}
    global_errors: list[str] = []
    for error in validation_errors:
        prefix, separator, _rest = error.partition(":")
        if separator and prefix in operation_locations:
            errors_by_fact_id.setdefault(prefix, []).append(error)
        else:
            global_errors.append(error)

    applied = copy.deepcopy(delta)
    quarantined: list[dict[str, Any]] = []
    for fact_id, errors in errors_by_fact_id.items():
        list_name, operation = operation_locations[fact_id]
        applied[list_name] = [
            item for item in applied[list_name] if item.get("fact_id") != fact_id
        ]
        quarantined.append(
            {
                "fact_id": fact_id,
                "operation_list": list_name,
                "proposed_operation": copy.deepcopy(operation),
                "validation_errors": errors,
            }
        )
    quarantined.sort(key=lambda item: (item["operation_list"], item["fact_id"]))
    return applied, quarantined, global_errors


def _repair_quote_normalized_evidence(
    delta: dict[str, Any], diary_text: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    repaired = copy.deepcopy(delta)
    normalized_diary = diary_text.translate(QUOTE_NORMALIZATION)
    repairs: list[dict[str, Any]] = []
    for list_name in ("added_facts", "updated_facts", "removed_facts"):
        operations = repaired.get(list_name)
        if not isinstance(operations, list):
            continue
        for operation in operations:
            if not isinstance(operation, dict):
                continue
            excerpts = operation.get("evidence_excerpts")
            if not isinstance(excerpts, list):
                continue
            for index, excerpt in enumerate(excerpts):
                if not isinstance(excerpt, str) or excerpt in diary_text:
                    continue
                normalized_excerpt = excerpt.translate(QUOTE_NORMALIZATION)
                start = normalized_diary.find(normalized_excerpt)
                if start < 0 or normalized_diary.find(normalized_excerpt, start + 1) >= 0:
                    continue
                exact_excerpt = diary_text[start : start + len(excerpt)]
                if exact_excerpt.translate(QUOTE_NORMALIZATION) != normalized_excerpt:
                    continue
                excerpts[index] = exact_excerpt
                repairs.append(
                    {
                        "fact_id": operation.get("fact_id"),
                        "operation_list": list_name,
                        "excerpt_index": index,
                        "model_excerpt": excerpt,
                        "exact_diary_excerpt": exact_excerpt,
                        "repair_kind": "quote_style_only",
                    }
                )
    return repaired, repairs


def _deterministic_delta_summary(delta: dict[str, Any]) -> str:
    parts: list[str] = []
    added = [
        item.get("statement", "")
        for item in delta.get("added_facts", [])
        if isinstance(item, dict) and isinstance(item.get("statement"), str)
    ]
    updated = [
        item.get("statement", "")
        for item in delta.get("updated_facts", [])
        if isinstance(item, dict) and isinstance(item.get("statement"), str)
    ]
    removed = [
        item.get("fact_id", "")
        for item in delta.get("removed_facts", [])
        if isinstance(item, dict) and isinstance(item.get("fact_id"), str)
    ]
    if added:
        parts.append("新增：" + "；".join(added))
    if updated:
        parts.append("更新：" + "；".join(updated))
    if removed:
        parts.append("删除：" + "、".join(removed))
    if not parts:
        return "当日没有通过校验的第二层背景变化。"
    return " ".join(parts)[:1200]


def _deterministic_overview(
    background: dict[str, Any], *, context_bucket: str = "current"
) -> str:
    sections: list[str] = []
    for section in background["sections"]:
        statements: list[str] = []
        for item in section["items"]:
            include = item["context_bucket"] == context_bucket and (
                item["status"] in {"active", "uncertain"}
                if context_bucket == "current"
                else item["status"] in {"historical", "superseded", "uncertain", "active"}
            )
            if include:
                statements.append("- " + item["statement"].strip())
        if statements:
            sections.append(
                f"### {section['category']}\n" + "\n".join(statements)
            )
    if not sections:
        return (
            "尚无可用的用户背景。"
            if context_bucket == "current"
            else "尚无可用的历史背景。"
        )
    return "\n\n".join(sections)


def _merge_delta(
    previous: dict[str, Any],
    delta: dict[str, Any],
    current_day: date,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = copy.deepcopy(previous)
    result["as_of_date"] = current_day.isoformat()
    sections = {section["category"]: section["items"] for section in result["sections"]}
    previous_map = _previous_fact_map(previous)
    added_ids: list[str] = []
    updated_ids: list[str] = []
    category_moves: list[dict[str, str]] = []
    superseded_ids: list[str] = []
    removed_ids: list[str] = []

    for change in delta["added_facts"]:
        item = {
            "fact_id": change["fact_id"],
            "statement": change["statement"],
            "status": change["status"],
            "subject_scope": change["subject_scope"],
            "temporal_class": change["temporal_class"],
            "context_bucket": change["context_bucket"],
            "expected_end_on": change["expected_end_on"],
            "review_after": change["review_after"],
            "first_observed_on": current_day.isoformat(),
            "last_updated_on": current_day.isoformat(),
            "source_dates": [current_day.isoformat()],
            "evidence_type": change["evidence_type"],
            "evidence_excerpts": copy.deepcopy(change["evidence_excerpts"]),
            "evidence_history": [
                {
                    "source_date": current_day.isoformat(),
                    "evidence_type": change["evidence_type"],
                    "evidence_excerpts": copy.deepcopy(change["evidence_excerpts"]),
                }
            ],
            "supersedes_fact_ids": copy.deepcopy(change["supersedes_fact_ids"]),
            "notes": change["notes"],
        }
        sections[change["category"]].append(item)
        added_ids.append(change["fact_id"])

    for change in delta["updated_facts"]:
        previous_category, old_item = previous_map[change["fact_id"]]
        target = next(
            item
            for item in sections[previous_category]
            if item["fact_id"] == change["fact_id"]
        )
        new_category = change["category"]
        if new_category != previous_category:
            sections[previous_category].remove(target)
            sections[new_category].append(target)
            category_moves.append(
                {
                    "fact_id": change["fact_id"],
                    "from_category": previous_category,
                    "to_category": new_category,
                }
            )
        target.update(
            {
                "statement": change["statement"],
                "status": change["status"],
                "subject_scope": change["subject_scope"],
                "temporal_class": change["temporal_class"],
                "context_bucket": change["context_bucket"],
                "expected_end_on": change["expected_end_on"],
                "review_after": change["review_after"],
                "last_updated_on": current_day.isoformat(),
                "source_dates": list(
                    dict.fromkeys([*old_item["source_dates"], current_day.isoformat()])
                ),
                "evidence_type": change["evidence_type"],
                "evidence_excerpts": copy.deepcopy(change["evidence_excerpts"]),
                "evidence_history": [
                    *copy.deepcopy(old_item["evidence_history"]),
                    {
                        "source_date": current_day.isoformat(),
                        "evidence_type": change["evidence_type"],
                        "evidence_excerpts": copy.deepcopy(change["evidence_excerpts"]),
                    },
                ],
                "supersedes_fact_ids": copy.deepcopy(change["supersedes_fact_ids"]),
                "notes": change["notes"],
            }
        )
        if change["status"] == "superseded":
            superseded_ids.append(change["fact_id"])
        else:
            updated_ids.append(change["fact_id"])

    # A declared replacement is not merely descriptive: deterministically retire
    # the old fact so it cannot remain in the current hidden context.
    deterministic_supersedes: list[dict[str, str]] = []
    for change in [*delta["added_facts"], *delta["updated_facts"]]:
        for old_fact_id in change["supersedes_fact_ids"]:
            old_category, old_item = previous_map[old_fact_id]
            target = next(
                item for item in sections[old_category] if item["fact_id"] == old_fact_id
            )
            if old_category != "historical_transitions":
                sections[old_category].remove(target)
                sections["historical_transitions"].append(target)
            target["status"] = "superseded"
            target["temporal_class"] = "historical"
            target["context_bucket"] = "history"
            target["last_updated_on"] = current_day.isoformat()
            deterministic_supersedes.append(
                {
                    "fact_id": old_fact_id,
                    "superseded_by_fact_id": change["fact_id"],
                    "from_category": old_category,
                    "to_category": "historical_transitions",
                }
            )
            superseded_ids.append(old_fact_id)

    for removal in delta["removed_facts"]:
        category = removal["category"]
        sections[category][:] = [
            item for item in sections[category] if item["fact_id"] != removal["fact_id"]
        ]
        removed_ids.append(removal["fact_id"])

    changed = bool(added_ids or updated_ids or superseded_ids or removed_ids)
    if changed:
        result["current_overview"] = _deterministic_overview(result)
        result["historical_overview"] = _deterministic_overview(
            result, context_bucket="history"
        )
    result["daily_update"] = {
        "source_date": current_day.isoformat(),
        "added_fact_ids": sorted(added_ids),
        "updated_fact_ids": sorted(updated_ids),
        "superseded_fact_ids": sorted(superseded_ids),
        "removed_fact_ids": sorted(removed_ids),
        "no_change": not changed,
        "summary": delta["summary"],
    }
    audit = {
        "merge_version": "1.1-atomic-background",
        "merged_at": legacy._iso_now(),
        "previous_background_sha256": _sha256_json(previous),
        "delta_sha256": _sha256_json(delta),
        "result_background_sha256": _sha256_json(result),
        "added_fact_ids": sorted(added_ids),
        "updated_fact_ids": sorted(updated_ids),
        "category_moves": sorted(
            category_moves, key=lambda item: item["fact_id"]
        ),
        "superseded_fact_ids": sorted(superseded_ids),
        "deterministic_supersedes": sorted(
            deterministic_supersedes, key=lambda item: item["fact_id"]
        ),
        "removed_fact_ids": sorted(removed_ids),
        "removal_details": copy.deepcopy(delta["removed_facts"]),
        "unchanged_fact_count": len(previous_map)
        - len(updated_ids)
        - len(superseded_ids)
        - len(removed_ids),
        "result_fact_count": len(legacy._items_by_id(result)),
        "runner_git_head": legacy._git_head(),
        "runner_sha256": legacy._sha256_file(Path(__file__).resolve()),
    }
    return result, audit


def _build_user_prompt(
    *, current_day: date, previous: dict[str, Any], diary_text: str
) -> str:
    previous_json = json.dumps(
        _compact_previous_view(previous, review_date=current_day),
        ensure_ascii=False,
        indent=2,
    )
    return (
        f"现在处理的唯一日期是 {current_day.isoformat()}。\n"
        "角色映射再次确认：日记作者和‘我’是凛祢；日记中对交流对象的称呼在上下文明确时指用户。\n"
        "请比较上一日紧凑用户背景与今日唯一日记，只输出今日必要差量。\n"
        "不得输出未变化旧事实，也不得输出完整背景。\n\n"
        "<previous_user_background>\n"
        f"{previous_json}\n"
        "</previous_user_background>\n\n"
        "<today_rinne_diary>\n"
        f"{diary_text}\n"
        "</today_rinne_diary>\n\n"
        f"source_date必须为{current_day.isoformat()}。"
        "新增、更新以及有当日证据的删除，其证据摘录必须逐字来自今日凛祢日记；"
        "基于背景时间线的自然移除按Schema输出空证据数组。"
    )


def _snapshot_path(model_dir: Path, current_day: date) -> Path:
    return model_dir / "snapshots" / current_day.isoformat()


def _load_previous_snapshot(
    model_dir: Path, current_day: date, first_day: date
) -> dict[str, Any] | None:
    if current_day == first_day:
        return None
    path = _snapshot_path(model_dir, current_day - timedelta(days=1)) / "background.json"
    if not path.is_file():
        raise ExperimentError(f"Previous calendar-day snapshot is missing: {path}")
    return legacy._read_json(path)


def _save_delta_snapshot(
    *,
    model_dir: Path,
    current_day: date,
    background: dict[str, Any],
    delta: dict[str, Any],
    applied_delta: dict[str, Any],
    merge_audit: dict[str, Any],
    generation_mode: str,
    request_payload: dict[str, Any] | None,
    raw_response: dict[str, Any] | None,
    metrics: dict[str, Any],
) -> None:
    final_dir = _snapshot_path(model_dir, current_day)
    if final_dir.exists():
        raise ExperimentError(f"Refusing to overwrite snapshot: {final_dir}")
    snapshots_dir = final_dir.parent
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = snapshots_dir / f".tmp-{current_day.isoformat()}-{uuid.uuid4().hex}"
    temp_dir.mkdir()
    try:
        legacy._write_json_new(temp_dir / "background.json", background)
        legacy._write_text_new(
            temp_dir / "background.md",
            _render_markdown(background, generation_mode),
        )
        legacy._write_json_new(temp_dir / "patch.json", delta)
        legacy._write_json_new(temp_dir / "applied_patch.json", applied_delta)
        legacy._write_json_new(temp_dir / "merge_audit.json", merge_audit)
        legacy._write_json_new(temp_dir / "metrics.json", metrics)
        if request_payload is not None:
            legacy._write_json_new(temp_dir / "request.json", request_payload)
        if raw_response is not None:
            legacy._write_json_new(temp_dir / "raw_response.json", raw_response)
        temp_dir.rename(final_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


class _Heartbeat:
    def __init__(self, label: str, interval_seconds: int = 30) -> None:
        self.label = label
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        started = time.monotonic()
        while not self._stop.wait(self.interval_seconds):
            elapsed = int(time.monotonic() - started)
            print(f"[heartbeat] {self.label}: waiting {elapsed}s", flush=True)

    def __enter__(self) -> _Heartbeat:
        self._thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def _review_template(model_key: str, label: str, experiment_days: list[date]) -> str:
    rows = "\n".join(
        f"| {day.isoformat()} | 未运行 | 未审查 |  |" for day in experiment_days
    )
    return (
        f"# {label}（{model_key}）差量背景审查\n\n"
        "| 日期 | 运行状态 | 内容结论 | 备注 |\n"
        "| --- | --- | --- | --- |\n"
        f"{rows}\n\n"
        "## 硬门槛\n\n"
        "- [ ] 没有把用户写成AI、代码、模型、桌宠或虚拟角色\n"
        "- [ ] 没有把用户的不同称呼写成两个人\n"
        "- [ ] relationship_with_rinne正确收录用户明确表达\n"
        "- [ ] 每条证据逐字存在于当日日记\n"
        "- [ ] 未变化旧事实没有被模型重新输出\n\n"
        "## 质量\n\n"
        "- 重要信息遗漏：\n"
        "- 日记作者解释被升级为用户事实：\n"
        "- 状态更新问题：\n"
        "- 是否允许继续下一日：\n"
    )


def prepare_experiment(args: argparse.Namespace) -> None:
    root = args.experiment_root.resolve()
    source = args.source_diaries.resolve()
    if root.exists():
        raise ExperimentError(f"Experiment root already exists; refusing overwrite: {root}")
    if not source.is_dir():
        raise ExperimentError(f"Diary source directory not found: {source}")
    for asset in (
        BACKGROUND_SCHEMA_SOURCE,
        DELTA_SCHEMA_SOURCE,
        DELTA_PROMPT_SOURCE,
    ):
        if not asset.is_file():
            raise ExperimentError(f"Experiment asset is missing: {asset}")

    version, inventory = legacy._model_inventory(args.ollama_url)
    missing_models = [
        spec["ollama_model"]
        for spec in MODEL_SPECS.values()
        if spec["ollama_model"].casefold() not in inventory
    ]
    if missing_models:
        raise ExperimentError(f"Required Ollama models are missing: {missing_models}")

    experiment_days = legacy._days(args.start_date, args.end_date)
    root.mkdir(parents=True, exist_ok=False)
    frozen_dir = root / "frozen_inputs" / "diaries"
    assets_dir = root / "frozen_assets"
    frozen_dir.mkdir(parents=True)
    assets_dir.mkdir(parents=True)
    for asset in (
        BACKGROUND_SCHEMA_SOURCE,
        DELTA_SCHEMA_SOURCE,
        DELTA_PROMPT_SOURCE,
    ):
        shutil.copy2(asset, assets_dir / asset.name)

    manifest_days: list[dict[str, Any]] = []
    for current_day in experiment_days:
        filename = f"diary_{current_day.isoformat()}.txt"
        source_path = source / filename
        if not source_path.is_file():
            manifest_days.append(
                {
                    "date": current_day.isoformat(),
                    "status": "missing",
                    "source_path": str(source_path),
                    "frozen_path": None,
                    "sha256": None,
                    "byte_count": 0,
                    "character_count": 0,
                }
            )
            continue
        data = source_path.read_bytes()
        destination = frozen_dir / filename
        destination.write_bytes(data)
        manifest_days.append(
            {
                "date": current_day.isoformat(),
                "status": "available",
                "source_path": str(source_path),
                "frozen_path": str(destination.relative_to(root)),
                "sha256": legacy._sha256_bytes(data),
                "byte_count": len(data),
                "character_count": len(data.decode("utf-8")),
            }
        )

    model_records: dict[str, Any] = {}
    for key, spec in MODEL_SPECS.items():
        inventory_item = inventory[spec["ollama_model"].casefold()]
        model_records[key] = {
            **spec,
            "digest": inventory_item.get("digest"),
            "size": inventory_item.get("size"),
        }
        (root / "outputs" / key).mkdir(parents=True)
        review_dir = root / "reviews" / key
        review_dir.mkdir(parents=True)
        legacy._write_text_new(
            review_dir / "REVIEW_TEMPLATE.md",
            _review_template(key, spec["label"], experiment_days),
        )

    manifest = {
        "experiment": "Rinne layer-two delta chronological diary replay",
        "created_at": legacy._iso_now(),
        "start_date": args.start_date.isoformat(),
        "end_date": args.end_date.isoformat(),
        "calendar_day_count": len(experiment_days),
        "available_diary_count": sum(
            item["status"] == "available" for item in manifest_days
        ),
        "missing_diary_count": sum(
            item["status"] == "missing" for item in manifest_days
        ),
        "days": manifest_days,
    }
    legacy._write_json_new(root / "input_manifest.json", manifest)
    config = {
        "experiment": "layer2_delta_prompt_pilot",
        "run_kind": "delta_prompt_pilot",
        "created_at": legacy._iso_now(),
        "private_data": True,
        "ollama_url": args.ollama_url,
        "ollama_version": version,
        "script_git_head": legacy._git_head(),
        "runner_sha256": legacy._sha256_file(Path(__file__).resolve()),
        "background_schema_sha256": legacy._sha256_file(
            assets_dir / BACKGROUND_SCHEMA_SOURCE.name
        ),
        "delta_schema_sha256": legacy._sha256_file(
            assets_dir / DELTA_SCHEMA_SOURCE.name
        ),
        "delta_prompt_sha256": legacy._sha256_file(
            assets_dir / DELTA_PROMPT_SOURCE.name
        ),
        "generation_options": {
            "temperature": 0.0,
            "seed": 20260816,
            "num_ctx": 32768,
            "num_predict": 2048,
        },
        "request_timeout_seconds": 1800,
        "keep_alive_during_run": "10m",
        "models": model_records,
        "rule": (
            "Run Mistral only through 2026-05-15, stop, and review the hard gate "
            "before any later day or model."
        ),
    }
    legacy._write_json_new(root / "experiment_config.json", config)
    legacy._write_text_new(
        root / "README.md",
        (
            "# 凛祢第二层差量提示词试跑\n\n"
            "此目录是私人、隔离、不可覆盖的实验目录。模型只输出每日差量，"
            "程序确定性合并完整快照。\n\n"
            f"- 冻结范围：{args.start_date.isoformat()} 至 {args.end_date.isoformat()}\n"
            "- 首次执行只允许运行到2026-05-15\n"
            "- 每日快照同时保存patch、merge_audit、request、raw_response和metrics\n"
            "- 不修改桌宠代码、配置、原始聊天、原始日记或正式记忆\n"
        ),
    )
    print(f"Prepared isolated delta experiment at {root}")


def preflight(args: argparse.Namespace, *, print_result: bool = True) -> dict[str, Any]:
    root = args.experiment_root.resolve()
    config = legacy._read_json(root / "experiment_config.json")
    manifest = _load_manifest(root)
    checks = (
        (BACKGROUND_SCHEMA_SOURCE.name, "background_schema_sha256"),
        (DELTA_SCHEMA_SOURCE.name, "delta_schema_sha256"),
        (DELTA_PROMPT_SOURCE.name, "delta_prompt_sha256"),
    )
    errors: list[str] = []
    for filename, config_key in checks:
        path = root / "frozen_assets" / filename
        if not path.is_file() or legacy._sha256_file(path) != config.get(config_key):
            errors.append(f"frozen asset changed or missing: {filename}")
    for item in manifest:
        if item.status != "available":
            continue
        path = root / str(item.frozen_path)
        if not path.is_file() or legacy._sha256_file(path) != item.sha256:
            errors.append(f"frozen diary changed or missing: {item.day}")
    version, inventory = legacy._model_inventory(config["ollama_url"])
    model_status: dict[str, Any] = {}
    for key, spec in config["models"].items():
        current = inventory.get(spec["ollama_model"].casefold())
        ok = current is not None and current.get("digest") == spec.get("digest")
        model_status[key] = {
            "model": spec["ollama_model"],
            "available_with_frozen_digest": ok,
        }
        if not ok:
            errors.append(f"model missing or digest changed: {spec['ollama_model']}")
    result = {
        "ok": not errors,
        "checked_at": legacy._iso_now(),
        "ollama_version_current": version,
        "ollama_version_prepared": config.get("ollama_version"),
        "errors": errors,
        "models": model_status,
    }
    if print_result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise ExperimentError("Preflight failed: " + "; ".join(errors))
    return result


def _save_failure(
    model_dir: Path,
    current_day: date,
    payload: dict[str, Any],
) -> Path:
    return legacy._save_failure(model_dir, current_day, payload)


def _validate_partial_output(
    root: Path,
    model_key: str,
    through_date: date,
) -> dict[str, Any]:
    manifest = _load_manifest(root)
    available_days = {
        item.day.isoformat() for item in manifest if item.status == "available"
    }
    model_dir = root / "outputs" / model_key
    previous: dict[str, Any] = _empty_background(manifest[0].day - timedelta(days=1))
    checked = 0
    model_calls = 0
    errors: list[str] = []
    for manifest_day in manifest:
        if manifest_day.day > through_date:
            break
        snapshot = _snapshot_path(model_dir, manifest_day.day)
        background_path = snapshot / "background.json"
        patch_path = snapshot / "patch.json"
        applied_patch_path = snapshot / "applied_patch.json"
        audit_path = snapshot / "merge_audit.json"
        metrics_path = snapshot / "metrics.json"
        for path in (
            background_path,
            patch_path,
            applied_patch_path,
            audit_path,
            metrics_path,
        ):
            if not path.is_file():
                errors.append(f"missing snapshot file: {path}")
        if errors:
            break
        background = legacy._read_json(background_path)
        metrics = legacy._read_json(metrics_path)
        day_errors = _validate_background(
            background,
            expected_day=manifest_day.day,
            available_days=available_days,
            previous=previous,
            missing_diary=manifest_day.status == "missing",
        )
        errors.extend(f"{manifest_day.day}: {error}" for error in day_errors)
        previous = background
        checked += 1
        model_calls += int(bool(metrics.get("model_called")))
    result = {
        "ok": not errors and checked > 0,
        "through_date": through_date.isoformat(),
        "snapshot_count": checked,
        "model_call_count": model_calls,
        "final_fact_count": len(legacy._items_by_id(previous)) if checked else 0,
        "errors": errors,
    }
    if errors:
        raise ExperimentError("Partial validation failed: " + "; ".join(errors))
    return result


def run_model(args: argparse.Namespace) -> None:
    root = args.experiment_root.resolve()
    preflight(args, print_result=False)
    config = legacy._read_json(root / "experiment_config.json")
    manifest = _load_manifest(root)
    spec = config["models"][args.model_key]
    model_dir = root / "outputs" / args.model_key
    stop_after = args.stop_after
    manifest_dates = {item.day for item in manifest}
    if stop_after is not None and stop_after not in manifest_dates:
        raise ExperimentError(f"stop-after date is outside the frozen manifest: {stop_after}")
    complete_path = model_dir / "RUN_COMPLETE.json"
    if complete_path.exists():
        raise ExperimentError(f"Model run is already complete: {complete_path}")
    existing = [
        item.day for item in manifest if _snapshot_path(model_dir, item.day).exists()
    ]
    if existing and not args.resume:
        raise ExperimentError(
            "Snapshots already exist. Use --resume to continue without overwriting."
        )
    event_log = model_dir / "run_events.jsonl"
    legacy._append_jsonl(
        event_log,
        {
            "event": "run_started" if not existing else "run_resumed",
            "at": legacy._iso_now(),
            "model_key": args.model_key,
            "model": spec["ollama_model"],
            "existing_snapshot_count": len(existing),
            "stop_after": stop_after.isoformat() if stop_after else None,
        },
    )

    delta_schema = legacy._read_json(
        root / "frozen_assets" / DELTA_SCHEMA_SOURCE.name
    )
    system_prompt = (
        root / "frozen_assets" / DELTA_PROMPT_SOURCE.name
    ).read_text(encoding="utf-8")
    available_days = {
        item.day.isoformat() for item in manifest if item.status == "available"
    }
    model = spec["ollama_model"]
    base_url = config["ollama_url"]
    first_day = manifest[0].day
    started = time.monotonic()
    completed_this_process = 0
    try:
        for manifest_day in manifest:
            current_day = manifest_day.day
            if stop_after is not None and current_day > stop_after:
                break
            snapshot_dir = _snapshot_path(model_dir, current_day)
            if snapshot_dir.exists():
                print(f"[skip] {current_day}: snapshot already exists", flush=True)
                continue
            previous = _load_previous_snapshot(model_dir, current_day, first_day)
            previous_for_merge = previous or _empty_background(
                current_day - timedelta(days=1)
            )

            if manifest_day.status == "missing":
                if previous is None:
                    raise ExperimentError("Cannot carry forward a missing first day")
                background = _carry_forward(previous, current_day)
                delta = _empty_delta(
                    current_day,
                    "当日没有日记输入；程序确定性沿用上一日背景，未调用模型。",
                )
                audit = {
                    "merge_version": "1.0",
                    "merged_at": legacy._iso_now(),
                    "previous_background_sha256": _sha256_json(previous),
                    "delta_sha256": _sha256_json(delta),
                    "result_background_sha256": _sha256_json(background),
                    "added_fact_ids": [],
                    "updated_fact_ids": [],
                    "superseded_fact_ids": [],
                    "removed_fact_ids": [],
                    "unchanged_fact_count": len(legacy._items_by_id(previous)),
                    "result_fact_count": len(legacy._items_by_id(background)),
                    "runner_git_head": legacy._git_head(),
                    "runner_sha256": legacy._sha256_file(Path(__file__).resolve()),
                }
                errors = _validate_background(
                    background,
                    expected_day=current_day,
                    available_days=available_days,
                    previous=previous,
                    missing_diary=True,
                )
                if errors:
                    raise ExperimentError(
                        f"Carry-forward validation failed on {current_day}: {errors}"
                    )
                _save_delta_snapshot(
                    model_dir=model_dir,
                    current_day=current_day,
                    background=background,
                    delta=delta,
                    applied_delta=delta,
                    merge_audit=audit,
                    generation_mode="carry_forward_missing_diary",
                    request_payload=None,
                    raw_response=None,
                    metrics={
                        "date": current_day.isoformat(),
                        "generation_mode": "carry_forward_missing_diary",
                        "model_called": False,
                        "created_at": legacy._iso_now(),
                    },
                )
                completed_this_process += 1
                print(f"[carry] {current_day}: no diary; model not called", flush=True)
                continue

            diary_path = root / str(manifest_day.frozen_path)
            if legacy._sha256_file(diary_path) != manifest_day.sha256:
                raise ExperimentError(f"Frozen diary hash changed: {diary_path}")
            diary_text = diary_path.read_text(encoding="utf-8")
            request_payload: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": _build_user_prompt(
                            current_day=current_day,
                            previous=previous_for_merge,
                            diary_text=diary_text,
                        ),
                    },
                ],
                "stream": False,
                "format": delta_schema,
                "options": config["generation_options"],
                "keep_alive": config["keep_alive_during_run"],
            }
            if spec.get("think") is not None:
                request_payload["think"] = spec["think"]
            print(f"[start] {current_day}: {model} delta", flush=True)
            call_started = time.monotonic()
            raw_response: dict[str, Any] | None = None
            parsed_delta: dict[str, Any] | None = None
            repaired_delta: dict[str, Any] | None = None
            applied_delta: dict[str, Any] | None = None
            validation_errors: list[str] = []
            applied_validation_errors: list[str] = []
            quarantined_operations: list[dict[str, Any]] = []
            evidence_repairs: list[dict[str, Any]] = []
            merge_audit: dict[str, Any] | None = None
            try:
                with _Heartbeat(f"{model} {current_day} delta"):
                    raw_response = legacy._post_json(
                        f"{base_url}/api/chat",
                        request_payload,
                        timeout=int(config["request_timeout_seconds"]),
                    )
                content = raw_response.get("message", {}).get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ExperimentError("Ollama response has empty message.content")
                parsed_delta = json.loads(content)
                if not isinstance(parsed_delta, dict):
                    raise ExperimentError("Structured delta is not a JSON object")
                repaired_delta, evidence_repairs = _repair_quote_normalized_evidence(
                    parsed_delta, diary_text
                )
                validation_errors = _validate_delta(
                    repaired_delta,
                    current_day=current_day,
                    previous=previous_for_merge,
                    diary_text=diary_text,
                )
                (
                    applied_delta,
                    quarantined_operations,
                    global_validation_errors,
                ) = _quarantine_invalid_operations(repaired_delta, validation_errors)
                if global_validation_errors:
                    raise ExperimentError("; ".join(global_validation_errors))
                applied_delta["summary"] = _deterministic_delta_summary(applied_delta)
                applied_validation_errors = _validate_delta(
                    applied_delta,
                    current_day=current_day,
                    previous=previous_for_merge,
                    diary_text=diary_text,
                )
                if applied_validation_errors:
                    raise ExperimentError("; ".join(applied_validation_errors))
                background, merge_audit = _merge_delta(
                    previous_for_merge,
                    applied_delta,
                    current_day,
                )
                merge_audit["model_delta_sha256"] = _sha256_json(parsed_delta)
                merge_audit["quote_repaired_delta_sha256"] = _sha256_json(
                    repaired_delta
                )
                merge_audit["applied_delta_sha256"] = _sha256_json(applied_delta)
                merge_audit["evidence_repairs"] = evidence_repairs
                merge_audit["quarantined_operations"] = quarantined_operations
                background_errors = _validate_background(
                    background,
                    expected_day=current_day,
                    available_days=available_days,
                    previous=previous_for_merge,
                )
                if background_errors:
                    raise ExperimentError("; ".join(background_errors))
            except (ExperimentError, json.JSONDecodeError) as exc:
                failure_path = _save_failure(
                    model_dir,
                    current_day,
                    {
                        "date": current_day.isoformat(),
                        "failed_at": legacy._iso_now(),
                        "model": model,
                        "error": str(exc),
                        "elapsed_seconds": round(time.monotonic() - call_started, 3),
                        "request": request_payload,
                        "raw_response": raw_response,
                        "parsed_delta": parsed_delta,
                        "quote_repaired_delta": repaired_delta,
                        "evidence_repairs": evidence_repairs,
                        "delta_validation_errors": validation_errors,
                        "applied_delta": applied_delta,
                        "applied_delta_validation_errors": applied_validation_errors,
                        "quarantined_operations": quarantined_operations,
                        "merge_audit": merge_audit,
                    },
                )
                legacy._append_jsonl(
                    event_log,
                    {
                        "event": "day_failed",
                        "at": legacy._iso_now(),
                        "date": current_day.isoformat(),
                        "error": str(exc),
                        "failure_path": str(failure_path),
                    },
                )
                raise ExperimentError(
                    f"Stopped on {current_day}; failure preserved at {failure_path}: {exc}"
                ) from exc

            assert raw_response is not None
            assert parsed_delta is not None
            assert applied_delta is not None
            assert merge_audit is not None
            elapsed = time.monotonic() - call_started
            metrics = {
                "date": current_day.isoformat(),
                "generation_mode": "ollama_structured_delta_deterministic_merge",
                "model_called": True,
                "model": model,
                "model_digest": spec["digest"],
                "think_setting": spec.get("think"),
                "elapsed_wall_seconds": round(elapsed, 3),
                "ollama_total_seconds": round(
                    float(raw_response.get("total_duration", 0)) / 1_000_000_000, 3
                ),
                "ollama_load_seconds": round(
                    float(raw_response.get("load_duration", 0)) / 1_000_000_000, 3
                ),
                "prompt_eval_count": raw_response.get("prompt_eval_count"),
                "prompt_eval_seconds": round(
                    float(raw_response.get("prompt_eval_duration", 0)) / 1_000_000_000,
                    3,
                ),
                "eval_count": raw_response.get("eval_count"),
                "eval_seconds": round(
                    float(raw_response.get("eval_duration", 0)) / 1_000_000_000, 3
                ),
                "thinking_character_count": len(
                    str(raw_response.get("message", {}).get("thinking", ""))
                ),
                "content_character_count": len(
                    str(raw_response.get("message", {}).get("content", ""))
                ),
                "delta_operation_counts": {
                    "added": len(applied_delta["added_facts"]),
                    "updated": len(applied_delta["updated_facts"]),
                    "removed": len(applied_delta["removed_facts"]),
                },
                "model_delta_operation_counts": {
                    "added": len(parsed_delta["added_facts"]),
                    "updated": len(parsed_delta["updated_facts"]),
                    "removed": len(parsed_delta["removed_facts"]),
                },
                "quarantined_operation_count": len(quarantined_operations),
                "evidence_repair_count": len(evidence_repairs),
                "result_fact_count": len(legacy._items_by_id(background)),
                "completed_at": legacy._iso_now(),
            }
            _save_delta_snapshot(
                model_dir=model_dir,
                current_day=current_day,
                background=background,
                delta=parsed_delta,
                applied_delta=applied_delta,
                merge_audit=merge_audit,
                generation_mode="ollama_structured_delta_deterministic_merge",
                request_payload=request_payload,
                raw_response=raw_response,
                metrics=metrics,
            )
            completed_this_process += 1
            legacy._append_jsonl(
                event_log,
                {
                    "event": "day_completed",
                    "at": legacy._iso_now(),
                    "date": current_day.isoformat(),
                    "elapsed_wall_seconds": round(elapsed, 3),
                    "fact_count": len(legacy._items_by_id(background)),
                    "delta_operation_counts": metrics["delta_operation_counts"],
                },
            )
            print(
                f"[done] {current_day}: {elapsed:.1f}s, "
                f"delta_ops={sum(metrics['delta_operation_counts'].values())}, "
                f"facts={metrics['result_fact_count']}",
                flush=True,
            )

            if stop_after is not None and current_day == stop_after:
                validation = _validate_partial_output(
                    root, args.model_key, through_date=current_day
                )
                checkpoint_dir = model_dir / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint = {
                    "status": "paused_for_human_review",
                    "paused_at": legacy._iso_now(),
                    "model_key": args.model_key,
                    "model": model,
                    "through_date": current_day.isoformat(),
                    "validation": validation,
                    "total_wall_seconds_this_process": round(
                        time.monotonic() - started, 3
                    ),
                    "completed_this_process": completed_this_process,
                    "next_action": "STOP. Human review is required before resume.",
                }
                legacy._write_json_new(
                    checkpoint_dir / f"PAUSED_AFTER_{current_day.isoformat()}.json",
                    checkpoint,
                )
                legacy._append_jsonl(event_log, {"event": "run_paused", **checkpoint})
                print(f"PAUSED_AFTER: {current_day}", flush=True)
                print("STOP NOW. Human review is required before resume.", flush=True)
                return

        final_day = manifest[-1].day
        validation = _validate_partial_output(root, args.model_key, final_day)
        complete = {
            "status": "complete",
            "completed_at": legacy._iso_now(),
            "model_key": args.model_key,
            "model": model,
            "validation": validation,
            "total_wall_seconds_this_process": round(time.monotonic() - started, 3),
            "completed_this_process": completed_this_process,
            "next_action": "STOP. Review this model before running another model.",
        }
        legacy._write_json_new(complete_path, complete)
        legacy._append_jsonl(event_log, {"event": "run_completed", **complete})
        print(f"MODEL_COMPLETE: {args.model_key}", flush=True)
    finally:
        legacy._unload_model(base_url, model)


def validate_command(args: argparse.Namespace) -> None:
    root = args.experiment_root.resolve()
    preflight(args, print_result=False)
    result = _validate_partial_output(root, args.model_key, args.through_date)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def seal_experiment(args: argparse.Namespace) -> None:
    root = args.experiment_root.resolve()
    preflight_result = preflight(args, print_result=False)
    if not HANDOFF_SOURCE.is_file():
        raise ExperimentError(f"Missing Luna handoff source: {HANDOFF_SOURCE}")
    handoff_text = HANDOFF_SOURCE.read_text(encoding="utf-8")
    destination = root / "LUNA_HANDOFF.md"
    legacy._write_text_new(destination, handoff_text)
    seal = {
        "sealed_at": legacy._iso_now(),
        "authoritative_repository_head": legacy._git_head(),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": legacy._sha256_file(Path(__file__).resolve()),
        "luna_handoff_path": str(destination),
        "luna_handoff_sha256": legacy._sha256_file(destination),
        "preflight": preflight_result,
    }
    legacy._write_json_new(root / "PREPARATION_SEAL.json", seal)
    print(json.dumps(seal, ensure_ascii=False, indent=2))


def list_models_command(_args: argparse.Namespace) -> None:
    legacy.list_models_command(_args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
    )
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--source-diaries",
        type=Path,
        default=DEFAULT_SOURCE_DIARIES,
    )
    prepare_parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=DEFAULT_START_DATE,
    )
    prepare_parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=DEFAULT_END_DATE,
    )
    prepare_parser.set_defaults(handler=prepare_experiment)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.set_defaults(handler=preflight)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model-key", required=True, choices=MODEL_SPECS)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--stop-after", type=date.fromisoformat)
    run_parser.set_defaults(handler=run_model)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--model-key", required=True, choices=MODEL_SPECS)
    validate_parser.add_argument("--through-date", required=True, type=date.fromisoformat)
    validate_parser.set_defaults(handler=validate_command)

    seal_parser = subparsers.add_parser("seal")
    seal_parser.set_defaults(handler=seal_experiment)

    list_parser = subparsers.add_parser("list-models")
    list_parser.set_defaults(handler=list_models_command)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        args.handler(args)
    except ExperimentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
