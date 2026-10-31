"""Two independent relay checks; only complete decisions reach the wire gate."""

import asyncio
from contextlib import asynccontextmanager
from copy import copy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from time import monotonic
import threading

from .content_check import (
    CHECK_SYSTEM,
    LABELS,
    CheckBudgetExceeded,
    CheckRecord,
    ContentChecker,
    fingerprint,
    normalized_lines,
)
from .checks_deadline import (
    check_deadline,
    is_pending,
    remaining_timeout,
    is_foreground,
    check_day,
    audit_call,
)
from .check_model import CheckOutputTruncated
from .program_text import (
    PROGRAM_TEXT_VERSION,
    fixed_literals,
    schema_texts,
    strip_program_text,
)
from .rulings import CacheAliases, HumanRulings, RANK, canonical_kind, valid_sha
from .record_storage import JsonlTail, append_row
from .approved_memory import ApprovedMemoryView
from .memory_checks import MemoryChecks, MEMORY_KINDS
from .memory_projection import MEMORY_PROJECTION_PROMPT
from .second_pass import SECOND_PASS_PROMPT, SECOND_PASS_PROMPT_SHA256, sentence_units

_MODEL_SLOTS = threading.BoundedSemaphore(2)


@asynccontextmanager
async def _model_slot(timeout):
    """Only background work shares the two slots; foreground never queues here."""
    if is_foreground():
        yield remaining_timeout(timeout)
        return
    until = monotonic() + timeout
    while not _MODEL_SLOTS.acquire(blocking=False):
        left = min(until - monotonic(), remaining_timeout(timeout))
        if left <= 0:
            raise asyncio.TimeoutError
        await asyncio.sleep(min(0.01, left))
    try:
        left = min(until - monotonic(), remaining_timeout(timeout))
        if left <= 0:
            raise asyncio.TimeoutError
        yield left
    finally:
        _MODEL_SLOTS.release()


class _SinglePass(ContentChecker):
    """Preserve identity; split truncated batches without retrying a failed call."""

    def __init__(
        self, *args, second=False, system_prompt=None, partial_items=False, **kwargs
    ):
        self.second = second
        self.system_prompt = system_prompt
        self.partial_items = partial_items
        self.durable_records = bool(system_prompt and partial_items)
        self.format_failures = {}
        super().__init__(*args, retries=0, **kwargs)
        if second:
            self.version = "official-second-pass-v4:" + SECOND_PASS_PROMPT_SHA256
            if system_prompt:
                self.version += ":memory:" + fingerprint(system_prompt)
            self._record_reader = JsonlTail(self.record_path)
            self._refresh_records()

    async def _classify_split(self, items, kind):
        muted = [
            item
            for item in items
            if not self.retry_allowed(item["text"], kind, item.get("context", ()))
        ]
        if muted:
            remaining = [item for item in items if item not in muted]
            found, errors = (
                await self._classify_split(remaining, kind) if remaining else ({}, {})
            )
            return found, {**errors, **{item["id"]: "格式错误" for item in muted}}
        found, error = await self._classify(items, kind)
        if found is None and error == "输出截断" and len(items) > 1:
            self.stats["split_batches"] += 1
            middle = len(items) // 2
            left, left_errors = await self._classify_split(items[:middle], kind)
            right, right_errors = await self._classify_split(items[middle:], kind)
            return {**left, **right}, {**left_errors, **right_errors}
        return (
            (found, {})
            if found is not None
            else ({}, {item["id"]: error for item in items})
        )

    async def _classify(self, items, kind):
        self.last_reply_characters = 0
        if self.model is None:
            return None, "未配置检查模型"
        try:
            waiting = monotonic()
            async with _model_slot(remaining_timeout(self.timeout_seconds)) as timeout:
                waited = monotonic() - waiting
                started = monotonic()
                try:
                    result = await self._classify_in_slot(items, kind, timeout)
                    for item in items:
                        key = self._identity(
                            item["text"], kind, tuple(item.get("context", ()))
                        )[0]
                        prior_day, count = self.format_failures.get(
                            key, (check_day(), 0)
                        )
                        count = count if prior_day == check_day() else 0
                        self.format_failures[key] = (
                            check_day(),
                            count + 1 if result[1] == "格式错误" else 0,
                        )
                    return result
                finally:
                    audit_call(monotonic() - started, waited)
        except asyncio.TimeoutError:
            return None, "超时"

    def retry_allowed(self, text, kind, context=()):
        key = self._identity(text, kind, tuple(context))[0]
        day, count = self.format_failures.get(key, (check_day(), 0))
        return day != check_day() or count < 3

    async def check_segments(self, entries):
        """One segment request; retain each item's original cache identity."""
        items = [
            dict(id=i, text=text, context=list(context))
            for i, (text, _, context) in enumerate(entries)
        ]
        if not items:
            return
        labels, errors = await self._classify_split(items, "segment")
        for i, (text, kind, context) in enumerate(entries):
            if kind != "segment":
                source_key = self._identity(text, "segment", context)[0]
                if source_key in self.format_failures:
                    self.format_failures[self._identity(text, kind, context)[0]] = (
                        self.format_failures[source_key]
                    )
            if i in labels:
                item = labels[i]
                label, confidence, reason, source = (
                    item["label"],
                    item["confidence"],
                    "模型判定",
                    str(getattr(self.model, "model", "relay_checker")),
                )
            else:
                label, confidence, reason, source = "U", 0, errors[i], "check_failure"
                self.stats["check_failures"] += 1
            self._remember(text, kind, context, label, confidence, source, reason)

    async def _classify_in_slot(self, items, kind, timeout):
        model = self.model
        if hasattr(model, "_limiter_key"):
            model = copy(model)
            key = model._limiter_key
            lane = "foreground" if is_foreground() else "background"
            model._limiter_key = (
                key[0] + ":official-check:" + lane + ("2" if self.second else "1"),
                *key[1:],
            )
        cleaned = [
            dict(
                id=item["id"],
                text=strip_program_text(item["text"]),
                context=[
                    strip_program_text(text)
                    for text in getattr(self, "context_filter", tuple)(
                        item.get("context", [])
                    )
                ],
            )
            for item in items
        ]
        if self.second:
            for item in cleaned:
                item["sentences"] = sentence_units(item["text"])
        started = monotonic()
        self.stats["model_calls"] += 1
        stream = None
        parts = []
        try:

            async def collect():
                nonlocal stream
                stream = model.chat_completion(
                    [
                        dict(
                            role="user",
                            content=json.dumps(
                                dict(kind=kind, items=cleaned), ensure_ascii=False
                            ),
                        )
                    ],
                    self.system_prompt
                    or (SECOND_PASS_PROMPT if self.second else CHECK_SYSTEM),
                )
                async for event in stream:
                    if isinstance(event, str):
                        if event.lstrip().startswith("Error calling the chat endpoint"):
                            raise ConnectionError("relay_connection_failed")
                        parts.append(event)
                    elif isinstance(event, dict) and event.get("type") == "text_delta":
                        parts.append(event.get("text", ""))
                    elif isinstance(event, dict) and event.get("type") in (
                        "error",
                        "provider_error",
                    ):
                        raise ConnectionError("relay_connection_failed")

            await asyncio.wait_for(collect(), timeout=timeout)
            raw = "".join(parts).strip()
            if raw.startswith("Error calling the chat endpoint"):
                raise ConnectionError("relay_connection_failed")
            if raw.startswith("```json") and raw.endswith("```"):
                raw = raw[7:-3].strip()
            if self.system_prompt and self.partial_items:
                from .memory_reply import parse_memory_reply

                values = parse_memory_reply(raw)["items"]
            else:
                values = json.loads(raw)["items"]
            if not isinstance(values, list) or (
                not self.partial_items and len(values) != len(items)
            ):
                raise ValueError("incomplete_result")
            found = {}
            seen = set()
            expected = {item["id"] for item in items}
            for item in values:
                if self.partial_items:
                    if not isinstance(item, dict) or type(item.get("id")) is not int:
                        continue
                    item_id = item["id"]
                    if item_id not in expected:
                        continue
                    if item_id in seen:
                        found.pop(item_id, None)
                        continue
                    seen.add(item_id)
                if (
                    not isinstance(item, dict)
                    or type(item.get("id")) is not int
                    or item["id"] in found
                    or not isinstance(item.get("label"), str)
                    or item.get("label") not in LABELS
                    or type(item.get("confidence")) is not int
                    or not 0 <= item["confidence"] <= 100
                ):
                    if self.partial_items:
                        continue
                    raise ValueError("invalid_result")
                found[item["id"]] = item
            if not self.partial_items and set(found) != expected:
                raise ValueError("invalid_ids")
            return found, ""
        except (asyncio.CancelledError, CheckBudgetExceeded):
            raise
        except asyncio.TimeoutError:
            return None, "超时"
        except ConnectionError:
            return None, "连接失败"
        except CheckOutputTruncated:
            return None, "输出截断"
        except (ValueError, KeyError, TypeError):
            return None, "格式错误"
        except Exception:
            return None, "连接失败"
        finally:
            self.last_reply_characters = sum(
                len(part) for part in parts if isinstance(part, str)
            )
            self.latencies.append(monotonic() - started)
            close = getattr(stream, "aclose", None)
            if callable(close):
                try:
                    await asyncio.wait_for(
                        close(), timeout=min(0.1, remaining_timeout(0.1))
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass


class RuntimeContentChecker:
    runtime_two_pass = True
    schema_texts = staticmethod(schema_texts)
    deadline = staticmethod(check_deadline)
    is_pending = staticmethod(is_pending)

    def __init__(
        self,
        keyword_path,
        record_path,
        first_model,
        second_model,
        rulings_path,
        second_record_paths=(),
        aliases_path=None,
        timeout_seconds=20,
        *,
        batch_size=16,
        runtime_record_path=None,
        second_write_path=None,
        chat_check_mode="two_pass",
    ):
        if chat_check_mode not in {"two_pass", "explicit_only"}:
            raise ValueError("invalid_chat_check_mode")
        self.single_safety_pass = chat_check_mode == "explicit_only"
        self.keyword_path = Path(keyword_path)
        self.record_path = Path(record_path)
        self.approved_memory = ApprovedMemoryView(
            self.record_path.with_name("approved_memory_view.json")
        )
        self.rulings_path = Path(rulings_path)
        self.aliases_path = Path(aliases_path) if aliases_path else None
        self.rulings = HumanRulings(self.rulings_path)
        self.aliases = CacheAliases(aliases_path)
        self.route_marks_path = self.record_path.with_name("route_marks.jsonl")
        self._route_marks_reader = JsonlTail(self.route_marks_path)
        self._route_marks = {}
        self._refresh_route_marks()
        if second_model is not None and getattr(
            second_model, "model", "claude-opus-4-6"
        ) not in ("claude-opus-4-6", "claude-sonnet-4-6"):
            raise ValueError("invalid_check_model")
        if first_model is not None and getattr(
            first_model, "model", "claude-sonnet-4-6"
        ) not in ("claude-opus-4-6", "claude-sonnet-4-6"):
            raise ValueError("invalid_check_model")
        first_type = _SinglePass
        if self.single_safety_pass:
            from .explicit_safety import ExplicitSafetyPass

            first_type = ExplicitSafetyPass
            if (
                first_model is not None
                and getattr(first_model, "model", "") != "claude-sonnet-4-6"
            ):
                raise ValueError("explicit_safety_requires_sonnet")
        self.first = first_type(
            keyword_path,
            self.record_path.with_name("chat_safety_sonnet.jsonl")
            if self.single_safety_pass
            else record_path,
            first_model,
            timeout_seconds=timeout_seconds,
            batch_size=128 if self.single_safety_pass else batch_size,
        )
        # A policy switch must not make the completed stricter two-pass cache
        # disappear. This reader has no model and can never start a request.
        self.legacy_first = (
            _SinglePass(keyword_path, record_path, None)
            if self.single_safety_pass
            else None
        )
        self.second_write_path = (
            Path(second_write_path)
            if second_write_path
            else self.record_path.with_name("content_checks_second_pass_runtime.jsonl")
        )
        self.second = _SinglePass(
            keyword_path,
            self.second_write_path,
            second_model,
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
            second=True,
        )
        from .memory_recovery import recover_memory_storage

        recover_memory_storage(
            self.record_path.parent,
            "official-second-pass-v4:"
            + SECOND_PASS_PROMPT_SHA256
            + ":memory:"
            + fingerprint(MEMORY_PROJECTION_PROMPT),
        )
        self.memory_checks = MemoryChecks(
            self,
            _SinglePass(
                keyword_path,
                self.record_path.with_name("memory_checks_opus.jsonl"),
                second_model,
                timeout_seconds=timeout_seconds,
                second=True,
                system_prompt=MEMORY_PROJECTION_PROMPT,
                partial_items=True,
            ),
        )
        self.first.context_filter = self._without_manual_context
        self.second.context_filter = self._without_manual_context
        self.final_record_path = (
            Path(runtime_record_path)
            if runtime_record_path
            else self.record_path.with_name("content_checks_runtime.jsonl")
        )
        identity = json.dumps(
            [
                self.first.version,
                SECOND_PASS_PROMPT_SHA256,
                PROGRAM_TEXT_VERSION,
                sorted(fixed_literals()),
                sorted(
                    (kind, sha, row["label"])
                    for (kind, sha), row in self.rulings.records.items()
                ),
                sorted(
                    (kind, sha, sorted(values))
                    for (kind, sha), values in self.aliases.by_canonical.items()
                ),
            ],
            ensure_ascii=False,
        )
        self.version = "official-runtime-two-pass-v1:" + fingerprint(identity)
        self.legacy_version = None
        if self.single_safety_pass:
            old_identity = json.loads(identity)
            old_identity[0] = self.legacy_first.version
            self.legacy_version = "official-runtime-two-pass-v1:" + fingerprint(
                json.dumps(old_identity, ensure_ascii=False)
            )
            self.version = "official-runtime-explicit-v1:" + fingerprint(identity)
        self.records = {}
        self.legacy_records = {}
        self.pending = {}
        self._final_reader = JsonlTail(self.final_record_path)
        self._refresh_final_records()
        self.external_second = {}
        self._external_readers = {
            Path(path): JsonlTail(path) for path in second_record_paths
        }
        self._external_rows = {}
        self._refresh_external_second()
        self._stats = dict(cache_hits=0, keyword_hits=0, ruling_hits=0, fixed_hits=0)

    def _refresh_route_marks(self):
        rows, reset = self._route_marks_reader.read()
        marks = {} if reset else dict(self._route_marks)
        for row in rows:
            if (
                set(row) != {"kind", "sha256", "route", "marked_at"}
                or row["kind"] != "turn"
                or row["route"] not in {"manual_relay", "official"}
                or not valid_sha(row["sha256"])
            ):
                raise ValueError("invalid_route_mark")
            datetime.fromisoformat(row["marked_at"])
            if marks.get(row["sha256"]) != "manual_relay":
                marks[row["sha256"]] = row["route"]
        self._route_marks = marks

    def route_mark(self, text):
        self._refresh_route_marks()
        return self._route_marks.get(fingerprint(text))

    def _without_manual_context(self, texts):
        """Keep marked turns out of model references as well as target items."""
        self._refresh_route_marks()
        return [
            text
            for text in texts
            if self._route_marks.get(fingerprint(text)) != "manual_relay"
        ]

    def mark_route(self, text, route, marked_at):
        """Append only a turn hash, its route and its completion time."""
        if route not in {"manual_relay", "official"}:
            raise ValueError("invalid_route_mark")
        sha = fingerprint(text)
        append_row(
            self.route_marks_path,
            dict(kind="turn", sha256=sha, route=route, marked_at=marked_at.isoformat()),
        )
        if self._route_marks.get(sha) != "manual_relay":
            self._route_marks[sha] = route

    def _refresh_external_second(self):
        changed = False
        for path, reader in self._external_readers.items():
            rows, reset = reader.read()
            if rows or reset:
                self._external_rows[path] = (
                    rows if reset else self._external_rows.get(path, []) + rows
                )
                changed = True
        if not changed:
            return
        self.external_second = {}
        for rows in self._external_rows.values():
            for row in rows:
                # Historical files include rejected v1/v2/v3/v5 and Sonnet trials.
                if row.get("version", "").split(":")[:2] != [
                    "second-pass-20261026",
                    "v4",
                ]:
                    continue
                if (
                    row.get("model") != "claude-opus-4-6"
                    or row.get("label") not in LABELS
                    or row.get("failure")
                    or row.get("source") == "check_failure"
                ):
                    continue
                key = (canonical_kind(row["kind"]), row["sha256"])
                old = self.external_second.get(key)
                if old is None or RANK[row["label"]] >= RANK[old["label"]]:
                    self.external_second[key] = row

    def _refresh_final_records(self):
        rows, reset = self._final_reader.read(lambda row: CheckRecord(**row))
        if not rows and not reset:
            return
        records = {} if reset else dict(self.records)
        legacy = {} if reset else dict(self.legacy_records)
        for record in rows:
            if (
                record.version == self.version
                and record.label in LABELS
                and not is_pending(record)
            ):
                records[record.key] = record
            elif (
                self.legacy_version is not None
                and record.version == self.legacy_version
                and record.label in LABELS
                and not is_pending(record)
                and record.key
                == fingerprint(
                    json.dumps(
                        [
                            record.version,
                            record.kind,
                            record.sha256,
                            record.context_sha256,
                        ]
                    )
                )
            ):
                legacy[record.key] = record
        self.records = records
        self.legacy_records = legacy

    def _prior_chat_record(self, text, kind, context):
        """Reuse exact complete decisions, not an unpaired first-pass A."""
        if self.legacy_first is None:
            return None
        _, sha, context_sha = self._identity(text, kind, context)
        old_key = fingerprint(json.dumps([self.legacy_version, kind, sha, context_sha]))
        record = self.legacy_records.get(old_key)
        root = record
        seen = set()
        while root and root.source.startswith("final_derived:"):
            if root.key in seen:
                return None
            seen.add(root.key)
            root = self.legacy_records.get(root.source.removeprefix("final_derived:"))
        if record is not None and root is not None and root.source == "two_pass":
            # Derived lines are valid only beneath the completed A parent.
            if record.key == root.key or record.label == root.label == "A":
                return record
        first = self.legacy_first.cached(text, kind, context=context)
        second = self._second_cached(text, kind, context)
        # A prior B/U already establishes that this text must not be sent.
        # It needs no second paid judgment just to exclude it again.
        if first is not None and first.label in {"B", "U"}:
            return (
                first
                if second is None
                else max((first, second), key=lambda row: RANK[row.label])
            )
        if first is None or second is None:
            return None
        return max((first, second), key=lambda row: RANK[row.label])

    @property
    def stats(self):
        result = dict(self._stats)
        for name in {**self.first.stats, **self.second.stats}:
            result[name] = (
                result.get(name, 0)
                + self.first.stats.get(name, 0)
                + self.second.stats.get(name, 0)
            )
        result["first_model_calls"] = self.first.stats["model_calls"]
        result["second_model_calls"] = self.second.stats["model_calls"]
        result["memory_model_calls"] = self.memory_checks.single.stats["model_calls"]
        result["model_calls"] += result["memory_model_calls"]
        return result

    @property
    def latencies(self):
        return [*self.first.latencies, *self.second.latencies]

    def keyword_hit(self, text):
        return (
            not self.approved_memory.approved(text)
            and not self.memory_checks.known_official(text)
            and self.first.keyword_hit(text)
        )

    def schema_ready(self, tools):
        """Memoize an exact schema only after all its textual parts passed."""
        self._refresh_final_records()
        text = json.dumps(tools, ensure_ascii=False, sort_keys=True)
        if self.keyword_hit(text):
            return False
        key, _, _ = self._identity(text, "tool_schema", ())
        old = self.records.get(key)
        if (
            old is not None
            and old.label == "A"
            and old.source == "final_derived:tool_schema"
        ):
            return True
        parts = list(dict.fromkeys(schema_texts(tools)))
        records = [self.cached(t, "line") for t in parts]
        if any(is_pending(r) or r.label != "A" for r in records):
            return False
        self._save(
            text, "tool_schema", (), "A", 100, "final_derived:tool_schema", "两遍检查"
        )
        return True

    def retry_allowed(self, text, kind="segment", context=()):
        if kind in MEMORY_KINDS:
            return self.memory_checks.cached(text, kind, context=context) is None
        return all(p.retry_allowed(text, kind, context) for p in self._chat_passes())

    def _chat_passes(self):
        return (self.first,) if self.single_safety_pass else (self.first, self.second)

    @property
    def format_suppressed_count(self):
        return len(
            {
                key
                for p in (self.first, self.second)
                for key, (day, count) in p.format_failures.items()
                if day == check_day() and count >= 3
            }
        )

    def _identity(self, text, kind, context):
        sha = fingerprint(text)
        context_sha = fingerprint(json.dumps(tuple(context)[-3:], ensure_ascii=False))
        return (
            fingerprint(json.dumps([self.version, kind, sha, context_sha])),
            sha,
            context_sha,
        )

    def _save(self, text, kind, context, label, confidence, source, reason):
        key, sha, context_sha = self._identity(text, kind, context)
        record = CheckRecord(
            key,
            sha,
            kind,
            self.version,
            context_sha,
            label,
            confidence,
            source,
            datetime.now(timezone.utc).isoformat(),
            reason,
            model="+".join(
                getattr(p.model, "model", "") or "" for p in self._chat_passes()
            ),
        )
        if source == "check_failure":
            self.pending[key] = record
            return record
        self.pending.pop(key, None)
        old = self.records.get(key)
        # Human A and verified program templates intentionally override stale model U.
        authoritative = source in {
            "manual_relay",
            "manual_ruling",
            "program_template",
            "keyword",
        }
        if old is not None and not authoritative and RANK[old.label] > RANK[label]:
            return old
        if old is not None and (old.label, old.source) == (label, source):
            return old
        append_row(self.final_record_path, asdict(record))
        self.records[key] = record
        return record

    def _known_override(self, text, kind, context):
        if kind in {"segment", "line"} and self.approved_memory.approved(text):
            key, sha, context_sha = self._identity(text, kind, context)
            self._stats["reviewed_memory_hits"] = (
                self._stats.get("reviewed_memory_hits", 0) + 1
            )
            return CheckRecord(
                key,
                sha,
                kind,
                self.version,
                context_sha,
                "A",
                100,
                "reviewed_memory",
                self.approved_memory.approved_at,
                "已确认的官方记忆版本",
                model="human_review",
            )
        # A memory projection receipt must never exempt current speech or a
        # raw conversation turn with identical text.
        raw_chat = canonical_kind(kind) in {"input", "turn"}
        if self.first.keyword_hit(text) if raw_chat else self.keyword_hit(text):
            self._stats["keyword_hits"] += 1
            return self._save(text, kind, context, "B", 100, "keyword", "关键词命中")
        if canonical_kind(kind) == "turn" and self.route_mark(text) == "manual_relay":
            return self._save(
                text, kind, context, "U", 100, "manual_relay", "手动中转站，没查"
            )
        ruling = self.rulings.lookup(fingerprint(text), kind)
        if ruling is not None and (
            not self.single_safety_pass or ruling["label"] == "A"
        ):
            self._stats["ruling_hits"] += 1
            return self._save(
                text, kind, context, ruling["label"], 100, "manual_ruling", "人工裁定"
            )
        if not strip_program_text(text).strip():
            self._stats["fixed_hits"] += 1
            return self._save(
                text, kind, context, "A", 100, "program_template", "固定格式"
            )
        return None

    def _second_cached(self, text, kind, context):
        self._refresh_external_second()
        records = []
        found = self.second.cached(text, kind, context=context)
        if found is not None:
            records.append(found)
        for sha in self.aliases.candidates(fingerprint(text), kind):
            value = self.external_second.get((canonical_kind(kind), sha))
            if value is not None:
                key, _, context_sha = self._identity(text, kind, context)
                records.append(
                    CheckRecord(
                        key,
                        fingerprint(text),
                        kind,
                        self.version,
                        context_sha,
                        value["label"],
                        value.get("confidence", 0),
                        "archived_second_v4",
                        "",
                        "模型判定",
                    )
                )
        return max(records, key=lambda record: RANK[record.label]) if records else None

    def cached(self, text, kind="segment", *, context=()):
        if kind in MEMORY_KINDS:
            return self.memory_checks.cached(text, kind, context=context)
        self._refresh_final_records()
        context = tuple(context)[-3:]
        override = self._known_override(text, kind, context)
        if override is not None:
            return self._expose_safe_lines(text, override)
        if kind == "line" and not context:
            memory_line = self.memory_checks.line(text)
            if memory_line is not None:
                return memory_line
        if self.single_safety_pass and kind == "segment" and not context:
            # A memory output's own Opus receipt is sufficient. Never extend
            # this exemption to fresh input or raw conversation turns.
            if self.memory_checks.known_official(text):
                return self.memory_checks.single.cached(
                    text, "memory_official"
                ) or self.memory_checks.line(text)
        first = self.first.cached(text, kind, context=context)
        if first is None and self.single_safety_pass:
            prior = self._prior_chat_record(text, kind, context)
            if prior is not None:
                record = self._save(
                    text,
                    kind,
                    context,
                    prior.label,
                    prior.confidence,
                    "reused_two_pass",
                    "复用已完成的检查",
                )
                return self._expose_safe_lines(text, record)
            if kind == "segment" and not context:
                lines = normalized_lines(text)
                known = [self.cached(line, "line") for line in lines]
                if lines and all(
                    not is_pending(row) and row.label == "A" for row in known
                ):
                    record = self._save(
                        text,
                        kind,
                        context,
                        "A",
                        min(row.confidence for row in known),
                        "reused_checked_lines",
                        "各行已有完整检查结论",
                    )
                    return self._expose_safe_lines(text, record)
        second = (
            first
            if self.single_safety_pass
            else self._second_cached(text, kind, context)
        )
        if first is not None and second is not None:
            winner = max((first, second), key=lambda record: RANK[record.label])
            record = self._save(
                text,
                kind,
                context,
                winner.label,
                min(first.confidence, second.confidence),
                "single_sonnet" if self.single_safety_pass else "two_pass",
                "单次明确风险检查" if self.single_safety_pass else "两遍检查",
            )
            return self._expose_safe_lines(text, record)
        key, _, _ = self._identity(text, kind, context)
        final = self.records.get(key)
        if final is not None and final.source == "official_parts":
            return self.record_official_turn(text, context)
        # Lines derived from a previously complete final record remain valid.
        if final is not None and final.source.startswith("final_derived:"):
            if any(
                record is not None and record.label != "A" for record in (first, second)
            ):
                return None
            return final
        return None

    @staticmethod
    def official_reply_context(user_text, context):
        # Include this turn's user input when checking the assistant separately.
        return (*tuple(context), json.dumps(dict(user=user_text), ensure_ascii=False))[
            -3:
        ]

    def official_input_record(self, text, context):
        turn = json.loads(text)
        return self.cached(turn["user"], "input", context=context)

    def record_official_turn(self, text, context):
        """A route receipt alone can never create an A decision."""
        if self.route_mark(text) != "official":
            return None
        turn = json.loads(text)
        parts = (
            self.official_input_record(text, context),
            self.cached(
                turn["assistant"],
                "segment",
                context=self.official_reply_context(turn["user"], context),
            ),
        )
        if any(is_pending(part) for part in parts):
            return None
        winner = max(parts, key=lambda part: RANK[part.label])
        result = self._save(
            text,
            "turn",
            tuple(context)[-3:],
            winner.label,
            min(part.confidence for part in parts),
            "official_parts",
            "两段从严合并",
        )
        return self._expose_safe_lines(text, result)

    async def check_segments(self, entries):
        """Check up to 16 non-turn items together without changing the prompts."""
        entries = [(text, kind, tuple(context)[-3:]) for text, kind, context in entries]
        allowed_kinds = (
            {"input", "segment", "turn", "line"}
            if self.single_safety_pass
            else {"input", "segment"}
        )
        if len(entries) > (128 if self.single_safety_pass else 16) or any(
            kind not in allowed_kinds for _, kind, _ in entries
        ):
            raise ValueError("invalid_segment_batch")
        results = [
            self.cached(text, kind, context=context) for text, kind, context in entries
        ]
        pending = [entry for entry, result in zip(entries, results) if result is None]
        if pending:
            first = [
                entry
                for entry in pending
                if self.first.cached(entry[0], entry[1], context=entry[2]) is None
            ]
            second = [
                entry
                for entry in pending
                if self._second_cached(entry[0], entry[1], entry[2]) is None
            ]
            work = [self.first.check_segments(first)]
            if not self.single_safety_pass:
                work.append(self.second.check_segments(second))
            outcomes = await asyncio.gather(
                *work,
                return_exceptions=True,
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
            for i, (text, kind, context) in enumerate(entries):
                if results[i] is None:
                    results[i] = self.cached(text, kind, context=context)
                    if results[i] is None:
                        failures = [
                            checker.pending[key]
                            for checker in self._chat_passes()
                            if (key := checker._identity(text, kind, context)[0])
                            in checker.pending
                        ]
                        results[i] = self._save(
                            text,
                            kind,
                            context,
                            "U",
                            0,
                            "check_failure",
                            failures[0].reason if failures else "无法确认",
                        )
        self._stats["cache_hits"] += len(entries) - len(pending)
        return results

    def _expose_safe_lines(self, text, record):
        """Reuse a complete parent decision, never a lone first-pass A.

        This is also used by synchronous cached(), so already-known background
        and schema data are gate-ready even when scheduling skips a model job.
        Each line still checks keywords, human rulings and prior stricter facts.
        """
        if record.source == "reviewed_memory":
            return record
        if record.label != "A" or is_pending(record) or record.kind == "line":
            return record
        if canonical_kind(record.kind) == "turn":
            try:
                value = json.loads(text)
            except (ValueError, TypeError):
                value = None
            if isinstance(value, dict):
                for key in ("user", "assistant"):
                    if isinstance(value.get(key), str):
                        self.record_lines(value[key], record)
                return record
        self.record_lines(text, record)
        return record

    def ready(self, text, kind="segment", *, context=()):
        return self.cached(text, kind, context=context) is not None

    def _remember(self, text, kind, context, label, confidence, source, reason):
        """Compatibility hook; an arbitrary caller cannot manufacture final A."""
        override = self._known_override(text, kind, context)
        if override is not None:
            return override
        if label != "A":
            return self._save(
                text, kind, context, label, confidence, source, "无法确认"
            )
        known = self.cached(text, kind, context=context)
        if known is not None:
            return known
        rest = strip_program_text(text)
        safe_rest = self.cached(rest, "line") if rest != text and rest else None
        if safe_rest is not None and safe_rest.label == "A":
            return self._save(
                text,
                kind,
                context,
                "A",
                confidence,
                "final_derived:" + safe_rest.key,
                "固定标签与已知内容",
            )
        return self._save(text, kind, context, "U", 0, "unverified_A", "无法确认")

    def record_fixed_line(self, line):
        return self._remember(line, "line", (), "A", 100, "fixed_format", "固定格式")

    def record_lines(self, text, record):
        if record.source == "reviewed_memory":
            return
        if (
            is_pending(record)
            or record.label != "A"
            or record.version != self.version
            or self.records.get(record.key) != record
        ):
            return
        for line in normalized_lines(text):
            override = self._known_override(line, "line", ())
            if override is not None:
                continue
            self._save(
                line,
                "line",
                (),
                "A",
                record.confidence,
                "final_derived:" + record.key,
                "已完成两遍的原文行",
            )

    async def check_text(
        self, text, kind="segment", *, context=(), timeout_seconds=None, deadline=None
    ):
        return (
            await self.check_many(
                [text],
                kind=kind,
                contexts=[tuple(context)],
                timeout_seconds=timeout_seconds,
                deadline=deadline,
            )
        )[0]

    async def check_turn(self, user_text, assistant_text, previous_turns=()):
        text = json.dumps(
            dict(user=user_text, assistant=assistant_text), ensure_ascii=False
        )
        record = await self.check_text(text, "turn", context=tuple(previous_turns)[-3:])
        if record.label == "A":
            self.record_lines(user_text, record)
            self.record_lines(assistant_text, record)
        return record

    async def check_many(
        self,
        texts,
        kind="segment",
        *,
        contexts=None,
        timeout_seconds=None,
        deadline=None,
    ):
        if kind in MEMORY_KINDS:
            return await self.memory_checks.check_many(texts, kind, contexts=contexts)
        if timeout_seconds is not None or deadline is not None:
            with check_deadline(
                20 if timeout_seconds is None else timeout_seconds, deadline=deadline
            ):
                return await self.check_many(texts, kind=kind, contexts=contexts)
        contexts = (
            [tuple(value)[-3:] for value in contexts]
            if contexts is not None
            else [()] * len(texts)
        )
        if len(contexts) != len(texts):
            raise ValueError("context_count_mismatch")
        results = [
            self.cached(text, kind, context=context)
            for text, context in zip(texts, contexts)
        ]
        pending = [i for i, record in enumerate(results) if record is None]
        if pending:
            values = [texts[i] for i in pending]
            references = [contexts[i] for i in pending]
            # Independent passes share one wall-clock window, never serial 20+20.
            work = [self.first.check_many(values, kind=kind, contexts=references)]
            if not self.single_safety_pass:
                work.append(self._check_second_missing(values, kind, references))
            outcomes = await asyncio.gather(
                *work,
                return_exceptions=True,
            )
            # Drain the other already-paid pass before propagating a test stop.
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
            for index in pending:
                results[index] = self.cached(
                    texts[index], kind, context=contexts[index]
                )
                if results[index] is None:
                    failures = []
                    for checker in self._chat_passes():
                        key, _, _ = checker._identity(
                            texts[index], kind, contexts[index]
                        )
                        if key in checker.pending:
                            failures.append(checker.pending[key])
                    reason = failures[0].reason if failures else "无法确认"
                    results[index] = self._save(
                        texts[index],
                        kind,
                        contexts[index],
                        "U",
                        0,
                        "check_failure",
                        reason,
                    )
        self._stats["cache_hits"] += len(texts) - len(pending)
        return results

    async def _check_second_missing(self, texts, kind, contexts):
        pending = [
            i
            for i, text in enumerate(texts)
            if self._second_cached(text, kind, contexts[i]) is None
        ]
        if pending:
            await self.second.check_many(
                [texts[i] for i in pending],
                kind=kind,
                contexts=[contexts[i] for i in pending],
            )
