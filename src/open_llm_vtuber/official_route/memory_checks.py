"""One shared Opus request per changed memory batch, never a chat exemption."""

from datetime import datetime, timezone
from copy import deepcopy
import json
import re
from loguru import logger

from .checks_deadline import is_pending, check_scope
from .clean_view import BACKGROUND_PENDING, BACKGROUND_PLACEHOLDER
from .content_check import CheckRecord, fingerprint, normalized_lines
from .record_storage import process_lock, replace_text
from .memory_recovery import read_attempts
from .memory_fact_policy import projection_keyword_hit


MEMORY_KINDS = {"memory_layer2", "memory_recent", "memory_tool"}
_SHARED_KIND = "memory_shared"


class MemoryChecks:
    def __init__(self, owner, single):
        self.owner, self.single = owner, single
        if (
            single.model is not None
            and getattr(single.model, "model", "") != "claude-opus-4-6"
        ):
            raise ValueError("memory_check_requires_opus")
        self.path = owner.record_path.with_name("memory_check_attempts.json")
        self.lock = self.path.with_suffix(".lock")
        self._recovery_blocked = False
        self._records_unverified = False
        self._initialized = False
        self._flags_stamp = None
        with process_lock(self.lock):
            self._read()
            if not self.path.exists():
                self._write({})
        self._initialized = True

    @property
    def recovery_blocked(self):
        self._refresh_recovery_flags()
        return self._recovery_blocked

    @property
    def records_unverified(self):
        self._refresh_recovery_flags()
        return self._records_unverified

    def _stamp(self):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)

    def _refresh_recovery_flags(self):
        # Most wire-gate lines only need the unchanged stop flags. Avoid a
        # process-lock/file-open transaction for every one of those lines.
        if self._stamp() != self._flags_stamp:
            with process_lock(self.lock):
                self._read()

    def _read(self):
        try:
            data = read_attempts(self.path)
        except (ValueError, TypeError, KeyError, UnicodeError):
            self._recovery_blocked = True
            self._flags_stamp = self._stamp()
            return {}
        if data is None:
            if self._initialized:
                self._recovery_blocked = True
            self._flags_stamp = None
            return {}
        self._recovery_blocked |= data.get("recovery_blocked", False)
        self._records_unverified |= data.get("records_unverified", False)
        self._flags_stamp = self._stamp()
        if data["version"] != self.single.version:
            return {}
        return data["attempts"]

    def _write(self, attempts):
        replace_text(
            self.path,
            json.dumps(
                dict(
                    version=self.single.version,
                    attempts=attempts,
                    recovery_blocked=self._recovery_blocked,
                    records_unverified=self._records_unverified,
                )
            ),
        )
        self._flags_stamp = None

    def _key(self, text, kind, context):
        return self.single._identity(text, _SHARED_KIND, tuple(context))[0]

    def _attempt(self, text, kind, context, attempts):
        keys = [self._key(text, kind, context)]
        keys += [
            self.single._identity(text, old, tuple(context))[0]
            for old in sorted(MEMORY_KINDS)
        ]
        entries = [attempts[key] for key in keys if key in attempts]
        for entry in entries:
            target = entry.get("official_text")
            if (
                entry.get("status") != "done"
                or not isinstance(target, str)
                or not target
            ):
                continue
            row = self.single.cached(target, "memory_official")
            if (
                row is not None
                and row.label == "A"
                and not self.output_keyword_hit(target)
            ):
                return entry
        return entries[0] if entries else None

    def _source_row(self, text, kind, context):
        kinds = [_SHARED_KIND, kind, *sorted(MEMORY_KINDS - {kind})]
        return next(
            (
                row
                for old in kinds
                if (row := self.single.cached(text, old, context=context)) is not None
            ),
            None,
        )

    def _pending(self, text, kind, context):
        key, sha, context_sha = self.single._identity(text, kind, tuple(context))
        return CheckRecord(
            key,
            sha,
            kind,
            self.single.version,
            context_sha,
            "U",
            0,
            "check_failure",
            "",
            "记忆检查未完成",
            "claude-opus-4-6",
        )

    def _lines(self, text, row):
        if is_pending(row) or row.label != "A":
            return row
        for line in normalized_lines(text):
            old = self.single.cached(line, "line")
            if self.output_keyword_hit(line) or (
                old is not None and (old.label != "A" or self._projection_receipt(old))
            ):
                continue
            self.single._remember(
                line,
                "line",
                (),
                "A",
                row.confidence,
                "memory_parent:" + row.key,
                "单次记忆检查的原文行",
            )
        return row

    def cached(self, text, kind, *, context=()):
        override = self.owner._known_override(text, "segment", context)
        if override is not None and override.label == "A":
            return override
        if self.records_unverified:
            # A lost stricter row may supersede any surviving A. Keep the
            # readable rows on disk, but never derive permission from them.
            return self._pending(text, kind, context)
        # Reuse an already-paid Opus decision, including checks made before
        # this change. Sonnet's result is not required for memory-only input.
        old = self.owner._second_cached(text, "segment", tuple(context))
        if old is not None and (old.label != "A" or self.owner.keyword_hit(text)):
            old = None
        target = self.single.cached(text, "memory_official", context=()) or self.line(
            text
        )
        row = target or old or self._source_row(text, kind, context)
        if row is not None:
            return self._lines(text, row)
        with process_lock(self.lock):
            attempted = self._attempt(text, kind, context, self._read()) is not None
        return (
            self._pending(text, kind, context)
            if attempted or self._recovery_blocked
            else None
        )

    def line(self, text):
        if self.records_unverified:
            return None
        row = self.single.cached(text, "line")
        return row if row is not None and row.label == "A" else None

    def known_official(self, text):
        if self.records_unverified:
            return False
        row = self.single.cached(text, "memory_official") or self.line(text)
        return (
            row is not None
            and row.label == "A"
            and self._projection_receipt(row)
            and not self.output_keyword_hit(text)
        )

    def _projection_receipt(self, row):
        if row.kind == "memory_official" and row.source == "memory_projection":
            return True
        if not row.source.startswith("memory_parent:"):
            return False
        parent = self.single.records.get(row.source.removeprefix("memory_parent:"))
        return (
            parent is not None
            and parent.kind == "memory_official"
            and parent.source == "memory_projection"
            and parent.label == "A"
            and parent.version == self.single.version
        )

    def output_keyword_hit(self, text):
        return projection_keyword_hit(text, self.owner.first.keywords)

    def _claim(self, texts, kind, contexts):
        # The short reservation is durable *before* model I/O. Other QQ or
        # desktop processes neither duplicate an in-flight call nor retry an
        # uncertain/failed paid attempt after a restart.
        with process_lock(self.lock):
            attempts = self._read()
            if self._recovery_blocked:
                return []
            claimed = []
            for i, (text, context) in enumerate(zip(texts, contexts)):
                key = self._key(text, kind, context)
                if self._attempt(
                    text, kind, context, attempts
                ) is not None or self._source_row(text, kind, context):
                    continue
                attempts[key] = dict(status="started", kind=kind)
                claimed.append(i)
            if claimed:
                self._write(attempts)
            return claimed

    def _finish(self, texts, kind, contexts, indices, found, received=None):
        with process_lock(self.lock):
            attempts = self._read()
            for i in indices:
                attempts[self._key(texts[i], kind, contexts[i])] = dict(
                    status="done" if found is not None and i in found else "failed",
                    kind=kind,
                    official_text=found[i]["official_text"]
                    if found is not None and i in found
                    else "",
                    action=found[i]["action"]
                    if found is not None and i in found
                    else "omit",
                )
                if (found is None or i not in found) and i in (received or {}):
                    # Keep the paid structured result locally so a later parser
                    # fix can recover it without another HTTP request.
                    attempts[self._key(texts[i], kind, contexts[i])][
                        "received_item"
                    ] = received[i]
            self._write(attempts)

    def _remember_projection(
        self,
        source,
        context,
        item,
        *,
        decision_source="claude-opus-4-6",
        decision_reason="模型判定",
    ):
        self.single._remember(
            source,
            _SHARED_KIND,
            tuple(context),
            item["label"],
            item["confidence"],
            decision_source,
            decision_reason,
        )
        target = item["official_text"]
        if target:
            row = self.single._remember(
                target,
                "memory_official",
                (),
                "A",
                item["confidence"],
                "memory_projection",
                "单次Opus确认的非露骨记忆版本",
            )
            self._lines(target, row)

    def _recover_received(self, texts, kind, contexts):
        recovered = 0
        with process_lock(self.lock):
            attempts = self._read()
            if self._recovery_blocked or self._records_unverified:
                return
            for source, context in zip(texts, contexts):
                key = self._key(source, kind, context)
                entry = attempts.get(key, {})
                item = deepcopy(entry.get("received_item"))
                if entry.get("status") != "failed" or not isinstance(item, dict):
                    continue
                self._prepare_projection(item, source)
                if not self._valid_projection(item, source):
                    continue
                self._remember_projection(source, context, item)
                attempts[key] = dict(
                    status="done",
                    kind=kind,
                    official_text=item["official_text"],
                    action=item["action"],
                )
                recovered += 1
            if recovered:
                self._write(attempts)
        if recovered:
            logger.info(
                "[官方记忆本地恢复] 有效条目={}；复用已付费结果，不调用模型", recovered
            )

    def project_unit(self, text, kind):
        row = self.cached(text, kind)
        if is_pending(row):
            return BACKGROUND_PENDING
        if row.source == "reviewed_memory":
            return text
        with process_lock(self.lock):
            entry = self._attempt(text, kind, (), self._read())
        if entry is not None:
            target = entry.get("official_text", "")
            if (
                entry.get("status") == "done"
                and entry.get("action") == "omit"
                and not target
            ):
                return BACKGROUND_PLACEHOLDER
            known = self.single.cached(target, "memory_official") if target else None
            if (
                known is not None
                and known.label == "A"
                and not self.output_keyword_hit(target)
            ):
                return target
            return BACKGROUND_PENDING
        if row.label == "A":
            return (
                text
                if row.source == "reviewed_memory" or not self.output_keyword_hit(text)
                else BACKGROUND_PENDING
            )
        return BACKGROUND_PLACEHOLDER

    def project(self, text, kind, options):
        from .clean_view import background_parts

        return "".join(
            part["text"]
            if part["protected"] or not part["text"].strip()
            else self.project_unit(part["text"], kind)
            for part in background_parts(text, **options)
        )

    def clean_view(self, text, kind, options):
        """Use original unit boundaries; never merge a good item with an omission."""
        from .clean_view import (
            CleanBackground,
            ProtectedContentBlocked,
            _check_protected,
            background_parts,
        )

        output, manifest = [], []
        for part in background_parts(text, **options):
            source = part["text"]
            if not source.strip():
                output.append(source)
                continue
            if part["protected"]:
                row = _check_protected(part, self.owner)
                if row.label != "A":
                    raise ProtectedContentBlocked(
                        [dict(kind=part["name"], sha256=row.sha256, label=row.label)]
                    )
                target = source
            else:
                row = self.cached(source, kind) or self._pending(source, kind, ())
                target = self.project_unit(source, kind)
            output.append(target)
            manifest.append(
                dict(
                    sha256=fingerprint(source),
                    label=row.label,
                    confidence=row.confidence,
                    source=row.source,
                    protected=part["protected"],
                    kind=part["name"],
                    replaced=target != source,
                    pending=is_pending(row) or target == BACKGROUND_PENDING,
                )
            )
        return CleanBackground("".join(output), manifest)

    def _valid_projection(self, item, source):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("label"), str)
            or item.get("label") not in {"A", "B", "U"}
            or type(item.get("confidence")) is not int
            or not 0 <= item["confidence"] <= 100
        ):
            return False
        target = item.get("official_text")
        action = item.get("action")
        if (
            not isinstance(target, str)
            or not isinstance(action, str)
            or action not in {"keep", "rewrite", "omit"}
        ):
            return False
        if action == "omit":
            return item.get("official_label") == "U" and target == ""
        if not target.strip() or item.get("official_label") != "A":
            return False
        if self.output_keyword_hit(target):
            return False
        if action == "keep":
            return item["label"] == "A" and target == source
        # The source label and the action are separate statements. An A source
        # may still have a useful safe rewrite; validate the output exactly as
        # for B instead of discarding it solely for that label mismatch.
        return item["label"] in {"A", "B"} and target != source

    def _prepare_projection(self, item, source):
        if not isinstance(item, dict):
            return
        if (
            item.get("label") == "A"
            and item.get("official_label") == "A"
            and item.get("action") == "rewrite"
            and item.get("official_text") == source
        ):
            item["action"] = "keep"
        self._restore_keep_format(item, source)
        target = item.get("official_text")
        if (
            item.get("label") == "A"
            and item.get("official_label") == "A"
            and item.get("action") == "keep"
            and isinstance(target, str)
            and target.strip()
            and source.strip()
            and not self.output_keyword_hit(source)
            and not self.output_keyword_hit(target)
        ):
            # Keep is a decision about the input, not a request to trust the
            # model's transcription. Use the exact source, never invented facts.
            item["official_text"] = source
        prefix = re.match(r"^[ \t]*(?:[-*]|\d+[.)、])\s+", source)
        target = item.get("official_text")
        if (
            prefix
            and item.get("action") == "rewrite"
            and isinstance(target, str)
            and target.strip()
            and not target.startswith(prefix[0])
        ):
            item["official_text"] = prefix[0] + target

    @staticmethod
    def _restore_keep_format(item, source):
        """Honor keep with the original bytes, only for quote/list typography."""
        if (
            not isinstance(item, dict)
            or item.get("label") != "A"
            or item.get("official_label") != "A"
            or item.get("action") != "keep"
            or not isinstance(item.get("official_text"), str)
        ):
            return
        target = item["official_text"]
        marker = r"^[ \t]*(?:[-*]|\d+[.)、])\s+"
        prefix = re.match(marker, source)
        other = re.match(marker, target)
        original_body = source[prefix.end() :] if prefix else source
        target_body = target[other.end() :] if prefix and other else target
        quotes = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
        if original_body.translate(quotes) == target_body.translate(quotes):
            item["official_text"] = source

    def _projection_failure_code(self, item, source):
        """Return only a fixed code, never a fragment of model or memory text."""
        if not isinstance(item, dict):
            return "item_missing_or_invalid"
        if not isinstance(item.get("label"), str) or item["label"] not in {
            "A",
            "B",
            "U",
        }:
            return "label_invalid"
        if (
            type(item.get("confidence")) is not int
            or not 0 <= item["confidence"] <= 100
        ):
            return "confidence_invalid"
        action, target = item.get("action"), item.get("official_text")
        if (
            not isinstance(target, str)
            or not isinstance(action, str)
            or action not in {"keep", "rewrite", "omit"}
        ):
            return "projection_schema_invalid"
        if action == "omit":
            return (
                "omit_invalid"
                if item.get("official_label") != "U" or target != ""
                else ""
            )
        if not target.strip() or item.get("official_label") != "A":
            return "official_not_a_or_empty"
        if self.output_keyword_hit(target):
            return "output_keyword"
        if action == "keep":
            if item["label"] != "A":
                return "keep_label_mismatch"
            if target == source:
                return ""
            marker = r"^[ \t]*(?:[-*]|\d+[.)、])\s+"
            original = re.match(marker, source)
            candidate = re.match(marker, target)
            body = source[original.end() :] if original else source
            target_body = target[candidate.end() :] if candidate else target
            return (
                "keep_list_prefix"
                if original and body == target_body
                else "keep_text_changed"
            )
        if item["label"] not in {"A", "B"}:
            return "rewrite_source_label"
        return "rewrite_unchanged" if target == source else ""

    def _save_view(self, texts, kind, results):
        path = self.path.with_name("official_" + kind + ".json")
        items = [
            dict(
                source_sha256=fingerprint(text),
                status="pending" if is_pending(row) else row.label,
                text=self.project_unit(text, kind),
            )
            for text, row in zip(texts, results)
        ]
        with process_lock(self.lock):
            replace_text(
                path,
                json.dumps(
                    dict(
                        source_sha256=fingerprint(
                            json.dumps(texts, ensure_ascii=False)
                        ),
                        checked_at=datetime.now(timezone.utc).isoformat(),
                        model="claude-opus-4-6",
                        items=items,
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
            )

    async def check_many(self, texts, kind, *, contexts=None):
        contexts = contexts if contexts is not None else [()] * len(texts)
        if len(texts) != len(contexts):
            raise ValueError("context_count_mismatch")
        self._recover_received(texts, kind, contexts)
        results = [self.cached(t, kind, context=c) for t, c in zip(texts, contexts)]
        missing = [i for i, row in enumerate(results) if row is None]
        if missing:
            values, refs = [texts[i] for i in missing], [contexts[i] for i in missing]
            claimed = self._claim(values, kind, refs)
            if claimed:
                logger.info(
                    "[官方记忆处理] 范围={}，新条目={}；准备单次Opus处理，不自动重试",
                    kind,
                    len(claimed),
                )
                items = [
                    dict(id=i, text=values[i], context=list(refs[i])) for i in claimed
                ]
                # Deliberately one call: no batching into extra requests, no
                # format-repair request, no truncation split, no automatic retry.
                with check_scope(
                    category=kind,
                    phase="tool_memory"
                    if kind == "memory_tool"
                    else "memory_background",
                ):
                    found, classification_error = await self.single._classify(
                        items, "segment"
                    )
                received = deepcopy(found)
                if found is not None:
                    for i, item in found.items():
                        self._prepare_projection(item, values[i])
                if found is not None:
                    failure_codes = {
                        i: self._projection_failure_code(found.get(i), values[i])
                        for i in claimed
                        if i not in found
                        or not self._valid_projection(found[i], values[i])
                    }
                    found = {
                        i: found[i]
                        for i in claimed
                        if i in found and self._valid_projection(found[i], values[i])
                    }
                else:
                    failure_codes = {i: "classification_failed" for i in claimed}
                if failure_codes:
                    logger.warning(
                        "[官方记忆处理失败] 类别={}，回复字数={}，条目原因={}",
                        classification_error or "格式错误",
                        getattr(self.single, "last_reply_characters", 0),
                        failure_codes,
                    )
                if found is not None:
                    for i, item in found.items():
                        self._remember_projection(values[i], refs[i], item)
                self._finish(values, kind, refs, claimed, found, received)
                logger.info(
                    "[官方记忆处理] 范围={}，本批有效={}，未完成={}；其余正常条目保留",
                    kind,
                    len(found or {}),
                    len(claimed) - len(found or {}),
                )
            results = [
                self.cached(t, kind, context=c) or self._pending(t, kind, c)
                for t, c in zip(texts, contexts)
            ]
        self._save_view(texts, kind, results)
        return results

    async def prepare_tool_results(self, texts, structure_texts=()):
        """Project all new result blocks and validate untouched metadata once."""
        from .clean_view import background_parts

        structures = list(
            dict.fromkeys(text for text in structure_texts if text.strip())
        )
        content_units = list(
            dict.fromkeys(
                [
                    part["text"]
                    for text in texts
                    for part in background_parts(text, recent=True)
                    if not part["protected"] and part["text"].strip()
                ]
            )
        )
        units = list(dict.fromkeys(content_units + structures))
        await self.check_many(units, "memory_tool")
        projected = [
            self.project(text, "memory_tool", dict(recent=True)) for text in texts
        ]
        status = "ready"
        for text in structures:
            row = self.cached(text, "memory_tool")
            if is_pending(row):
                status = "pending"
            elif self.project_unit(text, "memory_tool") != text:
                return projected, "blocked"
        if content_units and all(
            self.project_unit(text, "memory_tool")
            in {BACKGROUND_PENDING, BACKGROUND_PLACEHOLDER}
            for text in content_units
        ):
            status = (
                "pending"
                if any(
                    self.project_unit(text, "memory_tool") == BACKGROUND_PENDING
                    for text in content_units
                )
                else "blocked"
            )
        return projected, status
