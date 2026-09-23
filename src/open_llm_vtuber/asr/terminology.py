from __future__ import annotations

import json
from pathlib import Path
import re

from loguru import logger


class ASRTerminology:
    """Load one shared ASR terminology file and apply exact corrections."""

    def __init__(
        self,
        terminology_path: str | None,
        *,
        engine_name: str = "ASR",
    ) -> None:
        self.engine_name = engine_name
        self.path = self._resolve_path(terminology_path) if terminology_path else None
        self.terms, self.replacements = self._load()
        self._pattern = self._build_pattern()
        if self.path is not None and self.path.is_file():
            logger.info(
                "{} terminology loaded: terms={}, replacements={}, path={}",
                self.engine_name,
                len(self.terms),
                len(self.replacements),
                self.path,
            )

    @staticmethod
    def _resolve_path(value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    def _load(self) -> tuple[list[str], dict[str, str]]:
        path = self.path
        if path is None:
            return [], {}
        if not path.is_file():
            logger.warning("{} terminology file not found: {}", self.engine_name, path)
            return [], {}

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取 ASR 专名表 {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"ASR 专名表必须是 JSON 对象: {path}")

        terms = data.get("terms", [])
        replacements = data.get("replacements", {})
        if not isinstance(terms, list) or not all(
            isinstance(term, str) and term.strip() for term in terms
        ):
            raise ValueError(f"ASR 专名表 terms 必须是非空字符串列表: {path}")
        if not isinstance(replacements, dict) or not all(
            isinstance(source, str) and source and isinstance(target, str) and target
            for source, target in replacements.items()
        ):
            raise ValueError(f"ASR 专名表 replacements 必须是非空字符串映射: {path}")

        return [term.strip() for term in terms], dict(replacements)

    def _build_pattern(self) -> re.Pattern[str] | None:
        if not self.replacements:
            return None
        sources = sorted(self.replacements, key=lambda source: (-len(source), source))
        return re.compile("|".join(re.escape(source) for source in sources))

    def build_context(self, configured_context: str = "") -> str:
        """Build model vocabulary context for ASR engines that support it."""
        candidates = [configured_context.strip(), *self.terms]
        candidates.extend(self.replacements.values())
        unique_terms: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate and candidate not in seen:
                unique_terms.append(candidate)
                seen.add(candidate)
        return "、".join(unique_terms)

    def correct(self, text: str) -> str:
        """Apply only explicit replacements; never fuzzy-rewrite ordinary speech."""
        if self._pattern is None:
            return text
        corrected, count = self._pattern.subn(
            lambda match: self.replacements[match.group(0)], text
        )
        if count:
            logger.info(
                "{} terminology corrections applied: count={}",
                self.engine_name,
                count,
            )
            logger.debug(
                "{} corrected transcription: {} -> {}",
                self.engine_name,
                text,
                corrected,
            )
        return corrected
