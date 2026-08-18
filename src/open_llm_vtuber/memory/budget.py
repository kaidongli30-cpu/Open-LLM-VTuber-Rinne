"""Provider-aware placeholder budgets for future memory injection.

Token limits remain deliberately unset in stage 1. No existing configuration
file is read or modified by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .types import SerializableMemoryModel


class ModelScope(str, Enum):
    """Broad provider profile used to select a future memory budget."""

    LOCAL = "local"
    CLOUD = "cloud"


@dataclass(frozen=True)
class MemoryBudgetProfile(SerializableMemoryModel):
    """Optional token limits for each part of a future context package."""

    profile_name: str
    model_scope: ModelScope
    total_context_tokens: int | None = None
    current_conversation_tokens: int | None = None
    background_state_tokens: int | None = None
    runtime_override_tokens: int | None = None
    long_term_memory_tokens: int | None = None
    reserved_output_tokens: int | None = None
    reserved_tool_tokens: int | None = None
    max_ambiguous_candidates: int | None = None

    def __post_init__(self) -> None:
        numeric_values = (
            self.total_context_tokens,
            self.current_conversation_tokens,
            self.background_state_tokens,
            self.runtime_override_tokens,
            self.long_term_memory_tokens,
            self.reserved_output_tokens,
            self.reserved_tool_tokens,
            self.max_ambiguous_candidates,
        )
        if any(value is not None and value < 0 for value in numeric_values):
            raise ValueError("memory budget values cannot be negative")


@dataclass(frozen=True)
class MemoryBudgetConfig(SerializableMemoryModel):
    """Local and cloud budget profiles without choosing final token values."""

    local: MemoryBudgetProfile = field(
        default_factory=lambda: MemoryBudgetProfile(
            profile_name="local-unconfigured",
            model_scope=ModelScope.LOCAL,
        )
    )
    cloud: MemoryBudgetProfile = field(
        default_factory=lambda: MemoryBudgetProfile(
            profile_name="cloud-unconfigured",
            model_scope=ModelScope.CLOUD,
        )
    )

    def for_scope(self, scope: ModelScope) -> MemoryBudgetProfile:
        if scope is ModelScope.LOCAL:
            return self.local
        return self.cloud


DEFAULT_MEMORY_BUDGETS = MemoryBudgetConfig()
