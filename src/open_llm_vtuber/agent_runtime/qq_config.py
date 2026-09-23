"""Default-disabled configuration for Rinne's official QQ Bot channel."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Mapping


QQ_ENABLED_ENV = "RINNE_QQ_ENABLED"
QQ_ALLOWED_OPENIDS_ENV = "RINNE_QQ_ALLOWED_USER_OPENIDS"
QQ_APP_ID_ENV = "RINNE_QQ_APP_ID"
QQ_CLIENT_SECRET_ENV = "RINNE_QQ_CLIENT_SECRET"

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})


class QQConfigurationError(ValueError):
    """Raised when an enabled QQ channel is not safely configured."""


@dataclass(frozen=True, slots=True)
class QQResolvedCredentials:
    """Short-lived credential value whose repr never reveals the secret."""

    app_id: str = field(repr=False)
    client_secret: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class QQChannelConfig:
    """QQ channel settings with credentials referenced by environment name.

    The configuration object itself never stores an AppSecret.  The default is
    fully disabled and has an empty sender allowlist, so importing the project
    cannot accidentally connect a bot or authorize a QQ account.
    """

    enabled: bool = False
    allowed_user_openids: frozenset[str] = field(default_factory=frozenset)
    app_id_env: str = QQ_APP_ID_ENV
    client_secret_env: str = QQ_CLIENT_SECRET_ENV

    def __post_init__(self) -> None:
        normalized_openids = frozenset(
            value.strip() for value in self.allowed_user_openids if value.strip()
        )
        object.__setattr__(self, "allowed_user_openids", normalized_openids)
        for field_name in ("app_id_env", "client_secret_env"):
            value = getattr(self, field_name)
            if not _ENV_NAME_RE.fullmatch(value):
                raise QQConfigurationError(
                    f"{field_name} must be a valid environment variable name"
                )
        if self.app_id_env == self.client_secret_env:
            raise QQConfigurationError(
                "app_id_env and client_secret_env must be different"
            )

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "QQChannelConfig":
        values = os.environ if environ is None else environ
        enabled = _parse_enabled(values.get(QQ_ENABLED_ENV, ""))
        allowed_openids = frozenset(
            part.strip()
            for part in values.get(QQ_ALLOWED_OPENIDS_ENV, "").split(",")
            if part.strip()
        )
        return cls(enabled=enabled, allowed_user_openids=allowed_openids)

    def resolve_credentials(
        self,
        environ: Mapping[str, str] | None = None,
    ) -> QQResolvedCredentials:
        if not self.enabled:
            raise QQConfigurationError("QQ channel is disabled")
        if not self.allowed_user_openids:
            raise QQConfigurationError(
                f"enabled QQ channel requires {QQ_ALLOWED_OPENIDS_ENV}"
            )

        values = os.environ if environ is None else environ
        app_id = values.get(self.app_id_env, "").strip()
        client_secret = values.get(self.client_secret_env, "").strip()
        missing = [
            env_name
            for env_name, value in (
                (self.app_id_env, app_id),
                (self.client_secret_env, client_secret),
            )
            if not value
        ]
        if missing:
            raise QQConfigurationError(
                "enabled QQ channel is missing environment variables: "
                + ", ".join(missing)
            )
        return QQResolvedCredentials(
            app_id=app_id,
            client_secret=client_secret,
        )


def _parse_enabled(raw_value: str) -> bool:
    normalized = raw_value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise QQConfigurationError(
        f"{QQ_ENABLED_ENV} must be one of: 1, 0, true, false, yes, no, on, off"
    )


__all__ = [
    "QQ_ALLOWED_OPENIDS_ENV",
    "QQ_APP_ID_ENV",
    "QQ_CLIENT_SECRET_ENV",
    "QQ_ENABLED_ENV",
    "QQChannelConfig",
    "QQConfigurationError",
    "QQResolvedCredentials",
]
