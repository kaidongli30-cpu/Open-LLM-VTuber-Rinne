"""Environment-only opt-in for Rinne's local read-only computer tools."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .read_only_files import ReadOnlyFileService


READ_ONLY_ENABLED_ENV = "RINNE_READONLY_ENABLED"
READ_ONLY_ROOTS_ENV = "RINNE_READONLY_ROOTS"
READ_ONLY_MCP_SERVER_NAME = "rinne-computer-readonly"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})


class ReadOnlyComputerConfigurationError(ValueError):
    """Raised when local read-only access is enabled without safe roots."""


@dataclass(frozen=True, slots=True)
class ReadOnlyComputerConfig:
    enabled: bool = False
    allowed_roots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized = tuple(
            dict.fromkeys(
                str(root).strip() for root in self.allowed_roots if str(root).strip()
            )
        )
        object.__setattr__(self, "allowed_roots", normalized)

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "ReadOnlyComputerConfig":
        values = os.environ if environ is None else environ
        enabled = _parse_enabled(values.get(READ_ONLY_ENABLED_ENV, ""))
        roots = tuple(
            part.strip()
            for part in values.get(READ_ONLY_ROOTS_ENV, "").split(os.pathsep)
            if part.strip()
        )
        return cls(enabled=enabled, allowed_roots=roots)

    def create_service(self) -> ReadOnlyFileService:
        if self.enabled and not self.allowed_roots:
            raise ReadOnlyComputerConfigurationError(
                f"enabled read-only computer access requires {READ_ONLY_ROOTS_ENV}"
            )
        try:
            return ReadOnlyFileService(Path(root) for root in self.allowed_roots)
        except (OSError, ValueError) as exc:
            raise ReadOnlyComputerConfigurationError(
                "read-only computer roots are invalid or too broad"
            ) from exc


def effective_read_only_mcp_settings(
    *,
    use_mcpp: bool,
    enabled_servers: Sequence[str] | None,
    enable_local_computer_tools: bool,
    environ: Mapping[str, str] | None = None,
) -> tuple[bool, list[str]]:
    """Add the local server only for an explicitly enabled desktop context."""

    servers = list(dict.fromkeys(enabled_servers or ()))
    if not enable_local_computer_tools:
        return bool(use_mcpp), servers

    config = ReadOnlyComputerConfig.from_environment(environ)
    if not config.enabled:
        return bool(use_mcpp), servers

    # Resolve and validate every root in the parent process before advertising
    # the MCP server to the model.
    config.create_service()
    if READ_ONLY_MCP_SERVER_NAME not in servers:
        servers.append(READ_ONLY_MCP_SERVER_NAME)
    return True, servers


def _parse_enabled(raw_value: str) -> bool:
    normalized = raw_value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ReadOnlyComputerConfigurationError(
        f"{READ_ONLY_ENABLED_ENV} must be one of: 1, 0, true, false, yes, no, on, off"
    )


__all__ = [
    "READ_ONLY_ENABLED_ENV",
    "READ_ONLY_MCP_SERVER_NAME",
    "READ_ONLY_ROOTS_ENV",
    "ReadOnlyComputerConfig",
    "ReadOnlyComputerConfigurationError",
    "effective_read_only_mcp_settings",
]
