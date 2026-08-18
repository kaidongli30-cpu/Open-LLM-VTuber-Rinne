"""Foundational interfaces for the three-layer memory architecture.

Stage 2 adds independent current-conversation and today-history prototypes.
Nothing in this package is connected to the live conversation pipeline yet.
"""

from .budget import (
    DEFAULT_MEMORY_BUDGETS,
    MemoryBudgetConfig,
    MemoryBudgetProfile,
    ModelScope,
)
from .current_conversation import (
    ConversationBudgetPolicy,
    CurrentConversationState,
    TurnCountBudget,
)
from .orchestrator import MemoryOrchestrator
from .precedence import (
    MemoryPrecedence,
    compare_precedence,
    precedence_rank,
)
from .today_history_loader import (
    TodayHistoryDiagnostics,
    TodayHistoryLoader,
    TodayHistoryLoadResult,
)
from .today_partition import (
    TodayLlmDeliveryPreview,
    TodayMessagePartition,
    TodayMessagePartitionDiagnostics,
    TodayMessagePartitioner,
    build_today_llm_delivery_preview,
)
from .types import (
    ConversationTurn,
    LongTermMemoryItem,
    LongTermMemoryResult,
    LongTermMemoryResultStatus,
    MemoryContextPackage,
    MemoryDecision,
    MemoryDecisionAction,
    MemoryFact,
    MemoryItemStatus,
    MemorySource,
    NearbyReference,
    RuntimeStateOverride,
    RuntimeStateOverrides,
    UserBackgroundState,
)

__all__ = [
    "ConversationTurn",
    "ConversationBudgetPolicy",
    "CurrentConversationState",
    "DEFAULT_MEMORY_BUDGETS",
    "LongTermMemoryItem",
    "LongTermMemoryResult",
    "LongTermMemoryResultStatus",
    "MemoryBudgetConfig",
    "MemoryBudgetProfile",
    "MemoryContextPackage",
    "MemoryDecision",
    "MemoryDecisionAction",
    "MemoryFact",
    "MemoryItemStatus",
    "MemoryOrchestrator",
    "MemoryPrecedence",
    "MemorySource",
    "ModelScope",
    "NearbyReference",
    "RuntimeStateOverride",
    "RuntimeStateOverrides",
    "TodayHistoryDiagnostics",
    "TodayHistoryLoader",
    "TodayHistoryLoadResult",
    "TodayLlmDeliveryPreview",
    "TodayMessagePartition",
    "TodayMessagePartitionDiagnostics",
    "TodayMessagePartitioner",
    "TurnCountBudget",
    "UserBackgroundState",
    "compare_precedence",
    "build_today_llm_delivery_preview",
    "precedence_rank",
]
