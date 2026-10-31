"""Text-free human rulings, independent of classifier and keyword versions."""

from .record_storage import read_rows

RANK = {"A": 0, "U": 1, "B": 2}


def canonical_kind(kind):
    kind = str(kind).rsplit(":", 1)[-1]
    if kind in {"history", "turn"}:
        return "turn"
    if kind in {"background", "paragraph", "item", "protected", "segment"}:
        return "segment"
    if kind in {"current", "input", "user_input"}:
        return "input"
    return kind


def valid_sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def alias_shas(value):
    if isinstance(value, str):
        return [value] if valid_sha(value) else []
    if isinstance(value, list):
        return [sha for item in value for sha in alias_shas(item)]
    if isinstance(value, dict):
        return [sha for item in value.values() for sha in alias_shas(item)]
    return []


class HumanRulings:
    def __init__(self, path):
        self.records = {}
        for row in read_rows(path):
            if row.get("label") not in RANK or not valid_sha(row.get("sha256")):
                raise ValueError("invalid_human_ruling")
            kind = canonical_kind(row["kind"])
            for sha in [row["sha256"], *alias_shas(row.get("aliases", []))]:
                key = (kind, sha)
                previous = self.records.get(key)
                # Conflicting duplicated metadata is resolved conservatively.
                if previous is None or RANK[row["label"]] > RANK[previous["label"]]:
                    self.records[key] = row

    def lookup(self, sha, kind):
        kind = canonical_kind(kind)
        if kind == "line":
            values = [
                row
                for (item_kind, item_sha), row in self.records.items()
                if item_sha == sha
            ]
            return max(values, key=lambda row: RANK[row["label"]]) if values else None
        return self.records.get((kind, sha))


class CacheAliases:
    def __init__(self, path=None):
        self.by_canonical = {}
        if path is None:
            return
        for row in read_rows(path):
            if not valid_sha(row.get("sha256")):
                raise ValueError("invalid_cache_alias")
            kind = canonical_kind(row["kind"])
            values = {
                row["sha256"],
                *alias_shas(row.get("second_sha256")),
                *alias_shas(row.get("masked_sha256")),
            }
            # Every exact alias is usable in either direction; never match prefixes.
            for sha in values:
                self.by_canonical.setdefault((kind, sha), set()).update(values)

    def candidates(self, sha, kind):
        return self.by_canonical.get((canonical_kind(kind), sha), {sha})
