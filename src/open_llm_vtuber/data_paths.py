"""Central, side-effect-free paths for Rinne's private runtime data."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


RINNE_DATA_ROOT_ENV = "RINNE_DATA_ROOT"
DEFAULT_CHARACTER_CONF_UID = "rinne_01"


class DataRootConfigurationError(ValueError):
    """Raised when the configured private-data root is unsafe or ambiguous."""


def configured_data_root(
    environment: Mapping[str, str] | None = None,
) -> Path | None:
    """Return the configured absolute data root, or ``None`` for legacy paths."""

    values = os.environ if environment is None else environment
    raw_root = str(values.get(RINNE_DATA_ROOT_ENV, "")).strip()
    if not raw_root:
        return None

    candidate = Path(raw_root).expanduser()
    if not candidate.is_absolute():
        raise DataRootConfigurationError(
            f"{RINNE_DATA_ROOT_ENV} must be an absolute path"
        )
    resolved = candidate.resolve(strict=False)
    anchor = Path(resolved.anchor).resolve(strict=False)
    if resolved == anchor:
        raise DataRootConfigurationError(
            f"{RINNE_DATA_ROOT_ENV} must not be a drive or filesystem root"
        )
    if resolved.exists() and not resolved.is_dir():
        raise DataRootConfigurationError(
            f"{RINNE_DATA_ROOT_ENV} must reference a directory"
        )
    return resolved


def chat_history_root(
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Return the shared chat-and-memory root without creating directories."""

    data_root = configured_data_root(environment)
    if data_root is None:
        return Path("chat_history")
    return data_root / "chat_history"


def character_history_root(
    conf_uid: str = DEFAULT_CHARACTER_CONF_UID,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Return one character's private chat-and-memory directory."""

    normalized_uid = str(conf_uid).strip()
    if (
        not normalized_uid
        or normalized_uid in {".", ".."}
        or Path(normalized_uid).name != normalized_uid
        or "/" in normalized_uid
        or "\\" in normalized_uid
        or any(character in '<>:"|?*\0' for character in normalized_uid)
    ):
        raise DataRootConfigurationError("conf_uid must be one safe path component")
    return chat_history_root(environment) / normalized_uid


def resolve_character_history_root(
    history_root: str | Path | None,
    *,
    conf_uid: str = DEFAULT_CHARACTER_CONF_UID,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Use an explicit history root, otherwise resolve the configured default."""

    if history_root is not None:
        return Path(history_root)
    return character_history_root(conf_uid, environment)


__all__ = [
    "DEFAULT_CHARACTER_CONF_UID",
    "DataRootConfigurationError",
    "RINNE_DATA_ROOT_ENV",
    "character_history_root",
    "chat_history_root",
    "configured_data_root",
    "resolve_character_history_root",
]
