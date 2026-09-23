"""Safe building blocks for Rinne's cross-channel agent runtime."""

from typing import TYPE_CHECKING, Any

from .access_policy import ChannelAccessDenied, ChannelIdentityPolicy
from .channel_runtime import (
    ChannelMessageRuntime,
    ChannelTurnResult,
    ChannelTurnStatus,
    ConversationTurnCoordinator,
)
from .message import AgentChannel, InboundChannelMessage
from .qq_adapter import (
    QQChannelAdapter,
    QQSDKBindings,
    QQSDKUnavailable,
    load_official_qq_sdk,
)
from .qq_config import (
    QQ_ALLOWED_OPENIDS_ENV,
    QQ_APP_ID_ENV,
    QQ_CLIENT_SECRET_ENV,
    QQ_ENABLED_ENV,
    QQChannelConfig,
    QQConfigurationError,
)
from .proactive import (
    InMemoryReminderScheduler,
    ProactiveAuditEntry,
    ProactiveAuditStage,
    ProactiveBlockReason,
    ProactiveCategory,
    ProactiveDeliveryReceipt,
    ProactiveDispatchResult,
    ProactiveDispatchStatus,
    ProactiveMessageDispatcher,
    ProactiveMessageGate,
    ProactiveMessagePolicy,
    ProactiveMessageRequest,
    ScheduledReminder,
    task_completion_request,
)
from .proactive_config import (
    PROACTIVE_COMPANION_ENABLED_ENV,
    PROACTIVE_COOLDOWN_MINUTES_ENV,
    PROACTIVE_DAILY_LIMIT_ENV,
    PROACTIVE_ENABLED_ENV,
    PROACTIVE_QQ_RECIPIENT_ENV,
    PROACTIVE_QUIET_END_ENV,
    PROACTIVE_QUIET_START_ENV,
    PROACTIVE_REMINDER_POLL_SECONDS_ENV,
    ProactiveRuntimeConfig,
    ProactiveRuntimeConfigurationError,
)
from .read_only_files import (
    FileEntry,
    ReadOnlyFileAccessDenied,
    ReadOnlyFileLimitExceeded,
    ReadOnlyFileService,
    UnsupportedTextFile,
)
from .read_only_config import (
    READ_ONLY_ENABLED_ENV,
    READ_ONLY_MCP_SERVER_NAME,
    READ_ONLY_ROOTS_ENV,
    ReadOnlyComputerConfig,
    ReadOnlyComputerConfigurationError,
    effective_read_only_mcp_settings,
)
from .remote_approval import (
    REMOTE_APPROVAL_DEFAULT_TTL,
    REMOTE_APPROVAL_PREFIX,
    ReadOnlyApprovalAction,
    ReadOnlyRemoteApprovalBroker,
    RemoteApprovalAuditEntry,
    RemoteApprovalAuditStage,
    RemoteApprovalCapacityError,
    RemoteApprovalDecision,
    RemoteApprovalRequest,
    RemoteApprovalResolution,
    RemoteApprovalState,
    fingerprint_read_only_action,
    parse_remote_approval_button_data,
)
if TYPE_CHECKING:
    from .text_turn import RinneTextTurnProcessor


def __getattr__(name: str) -> Any:
    """Load the conversation-backed processor without creating an import cycle."""

    if name == "RinneTextTurnProcessor":
        from .text_turn import RinneTextTurnProcessor

        return RinneTextTurnProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentChannel",
    "ChannelAccessDenied",
    "ChannelIdentityPolicy",
    "ChannelMessageRuntime",
    "ChannelTurnResult",
    "ChannelTurnStatus",
    "ConversationTurnCoordinator",
    "FileEntry",
    "InboundChannelMessage",
    "InMemoryReminderScheduler",
    "ProactiveAuditEntry",
    "ProactiveAuditStage",
    "ProactiveBlockReason",
    "ProactiveCategory",
    "ProactiveDeliveryReceipt",
    "ProactiveDispatchResult",
    "ProactiveDispatchStatus",
    "ProactiveMessageDispatcher",
    "ProactiveMessageGate",
    "ProactiveMessagePolicy",
    "ProactiveMessageRequest",
    "PROACTIVE_COMPANION_ENABLED_ENV",
    "PROACTIVE_COOLDOWN_MINUTES_ENV",
    "PROACTIVE_DAILY_LIMIT_ENV",
    "PROACTIVE_ENABLED_ENV",
    "PROACTIVE_QQ_RECIPIENT_ENV",
    "PROACTIVE_QUIET_END_ENV",
    "PROACTIVE_QUIET_START_ENV",
    "PROACTIVE_REMINDER_POLL_SECONDS_ENV",
    "ProactiveRuntimeConfig",
    "ProactiveRuntimeConfigurationError",
    "QQ_ALLOWED_OPENIDS_ENV",
    "QQ_APP_ID_ENV",
    "QQ_CLIENT_SECRET_ENV",
    "QQ_ENABLED_ENV",
    "QQChannelAdapter",
    "QQChannelConfig",
    "QQConfigurationError",
    "QQSDKBindings",
    "QQSDKUnavailable",
    "READ_ONLY_ENABLED_ENV",
    "READ_ONLY_MCP_SERVER_NAME",
    "READ_ONLY_ROOTS_ENV",
    "REMOTE_APPROVAL_DEFAULT_TTL",
    "REMOTE_APPROVAL_PREFIX",
    "ReadOnlyApprovalAction",
    "ReadOnlyFileAccessDenied",
    "ReadOnlyComputerConfig",
    "ReadOnlyComputerConfigurationError",
    "ReadOnlyFileLimitExceeded",
    "ReadOnlyFileService",
    "ReadOnlyRemoteApprovalBroker",
    "RemoteApprovalAuditEntry",
    "RemoteApprovalAuditStage",
    "RemoteApprovalCapacityError",
    "RemoteApprovalDecision",
    "RemoteApprovalRequest",
    "RemoteApprovalResolution",
    "RemoteApprovalState",
    "RinneTextTurnProcessor",
    "ScheduledReminder",
    "UnsupportedTextFile",
    "effective_read_only_mcp_settings",
    "fingerprint_read_only_action",
    "load_official_qq_sdk",
    "parse_remote_approval_button_data",
    "task_completion_request",
]
