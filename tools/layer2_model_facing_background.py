"""Render the exact personal background visible to a role-play model.

The complete layer-two snapshot remains an auditable fact store.  This module
projects that store to the one field that is intended for the model-facing
background: ``current_overview``.  It deliberately does not ask another model
to summarize the snapshot and keeps the preview bytes deterministic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_FACING_FILENAME = "llm_user_background.md"
MODEL_FACING_METRICS_FILENAME = "llm_user_background_metrics.json"


class ModelFacingBackgroundError(ValueError):
    """Raised when a structured background cannot produce a safe projection."""


def _items(background: dict[str, Any]) -> list[dict[str, Any]]:
    sections = background.get("sections")
    if not isinstance(sections, list):
        raise ModelFacingBackgroundError("background.sections must be a list")
    result: list[dict[str, Any]] = []
    for section in sections:
        if not isinstance(section, dict) or not isinstance(section.get("items"), list):
            raise ModelFacingBackgroundError("background section items must be lists")
        for item in section["items"]:
            if not isinstance(item, dict):
                raise ModelFacingBackgroundError("background fact must be an object")
            result.append(item)
    return result


def render_model_facing_background(background: dict[str, Any]) -> str:
    """Return the exact text that should be injected as the user background.

    The returned text is the stored deterministic ``current_overview`` plus a
    single final LF.  No title, metadata, evidence, history, fact id, or audit
    field is added to the injection body.
    """

    overview = background.get("current_overview")
    if not isinstance(overview, str):
        raise ModelFacingBackgroundError("background.current_overview must be a string")
    normalized = overview.rstrip("\r\n")
    if not normalized:
        raise ModelFacingBackgroundError("background.current_overview must not be empty")
    return normalized + "\n"


def build_model_facing_metrics(
    background: dict[str, Any], rendered_text: str | None = None
) -> dict[str, Any]:
    """Build stable, reviewable metrics for a rendered model-facing preview."""

    text = rendered_text if rendered_text is not None else render_model_facing_background(background)
    if not isinstance(text, str) or not text.endswith("\n"):
        raise ModelFacingBackgroundError("rendered model-facing text must end with LF")
    facts = _items(background)
    current_count = sum(1 for item in facts if item.get("context_bucket") == "current")
    history_count = sum(1 for item in facts if item.get("context_bucket") == "history")
    encoded = text.encode("utf-8")
    return {
        "renderer": "layer2_model_facing_background.render_model_facing_background",
        "source_field": "current_overview",
        "injection_body_exact": True,
        "source_character_count": len(text.rstrip("\r\n")),
        "character_count": len(text),
        "line_count": len(text.splitlines()),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "current_fact_count": current_count,
        "history_fact_count": history_count,
        "as_of_date": background.get("as_of_date"),
        "schema_version": background.get("schema_version"),
    }


def write_model_facing_artifacts(
    output_dir: Path,
    background: dict[str, Any],
    *,
    text_filename: str = MODEL_FACING_FILENAME,
    metrics_filename: str = MODEL_FACING_METRICS_FILENAME,
    refuse_overwrite: bool = True,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write the preview and metrics as a pair and return their paths/metrics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    text = render_model_facing_background(background)
    metrics = build_model_facing_metrics(background, text)
    text_path = output_dir / text_filename
    metrics_path = output_dir / metrics_filename
    mode = "x" if refuse_overwrite else "w"
    with text_path.open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    with metrics_path.open(mode, encoding="utf-8", newline="\n") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return text_path, metrics_path, metrics
