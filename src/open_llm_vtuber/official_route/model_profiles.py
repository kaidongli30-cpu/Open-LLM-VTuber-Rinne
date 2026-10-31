"""Price/parameter table (USD per million tokens).

Prices follow the October 2026 official tables; Anthropic cache writes are
five-minute writes, OpenAI defaults to thirty minutes. Keep model validation
here, outside Pydantic startup
validation, so an unsupported route can safely fall back to the relay.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
LEGACY_EFFORTS = EFFORTS - {"xhigh"}


@dataclass(frozen=True)
class ModelProfile:
    input_usd: float
    output_usd: float
    cache_write_usd: float
    cache_read_usd: float
    disabled_thinking: str | None = None
    always_thinking: bool = False
    auto_tools_only: bool = False
    efforts: frozenset[str] = LEGACY_EFFORTS

    def effective_thinking(self, requested: bool) -> bool:
        return bool(requested) or self.always_thinking

    def request_parameters(self, requested: bool, effort: str | None = None) -> dict:
        self.validate_effort(effort)
        active = self.effective_thinking(requested)
        result = {"max_tokens": 16000 if active else 4096}
        if active:
            result["extra_body"] = {"thinking": {"type": "adaptive"}}
        elif self.disabled_thinking:
            # SDK 0.76.0 predates the newer thinking types. extra_body sends
            # them verbatim without upgrading or changing the SDK schema.
            result["extra_body"] = {"thinking": {"type": self.disabled_thinking}}
        if effort is not None:
            result.setdefault("extra_body", {})["output_config"] = {"effort": effort}
        return result

    def validate_effort(self, effort: str | None) -> None:
        if effort is not None and (
            not isinstance(effort, str) or effort not in self.efforts
        ):
            raise ValueError("unsupported_effort")

    def usage_cost(self, usage: Mapping[str, Any]) -> float:
        return (
            (usage.get("input_tokens", 0) or 0) * self.input_usd
            + (usage.get("output_tokens", 0) or 0) * self.output_usd
            + (usage.get("cache_creation_input_tokens", 0) or 0) * self.cache_write_usd
            + (usage.get("cache_read_input_tokens", 0) or 0) * self.cache_read_usd
        ) / 1_000_000


MODEL_PROFILES = MappingProxyType(
    {
        "claude-opus-4-6": ModelProfile(5, 25, 6.25, 0.50),
        "claude-sonnet-4-6": ModelProfile(3, 15, 3.75, 0.30),
        "claude-opus-4-7": ModelProfile(5, 25, 6.25, 0.50, efforts=EFFORTS),
        "claude-opus-4-8": ModelProfile(5, 25, 6.25, 0.50, efforts=EFFORTS),
        "claude-opus-5": ModelProfile(5, 25, 6.25, 0.50, "disabled", efforts=EFFORTS),
        "claude-sonnet-5": ModelProfile(2, 10, 2.50, 0.20, "disabled", efforts=EFFORTS),
        "claude-opus-5-5": ModelProfile(
            4,
            20,
            5.00,
            0.20,
            always_thinking=True,
            auto_tools_only=True,
            efforts=EFFORTS,
        ),
        "claude-sonnet-5-5": ModelProfile(
            2, 10, 2.50, 0.10, "between_tools", auto_tools_only=True, efforts=EFFORTS
        ),
    }
)
CHECK_MODELS = frozenset({"claude-sonnet-4-6", "claude-opus-4-6"})


@dataclass(frozen=True)
class OpenAIModelProfile(ModelProfile):
    """Responses usage includes reads/writes in input and reasoning in output.

    Standard rates: developers.openai.com/api/docs/models/gpt-6.1-sol.
    Cache buckets: developers.openai.com/api/docs/guides/prompt-caching.
    More than 272K input tokens reprices the entire request, not its excess.
    """

    efforts: frozenset[str] = EFFORTS
    long_input_threshold: int = 272000

    def usage_cost(self, usage: Mapping[str, Any]) -> float:
        details = usage.get("input_tokens_details") or {}
        inp = usage.get("input_tokens", 0) or 0
        output = usage.get("output_tokens", 0) or 0
        read = details.get("cached_tokens", usage.get("cached_tokens", 0)) or 0
        write = (
            details.get("cache_write_tokens", usage.get("cache_write_tokens", 0)) or 0
        )
        write_details = details.get("cache_write_tokens_details") or {}
        write_12h = write_details.get("ttl_12h_tokens", 0) or 0
        write_30m = write_details.get("ttl_30m_tokens", write - write_12h) or 0
        buckets = (inp, output, read, write, write_30m, write_12h)
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in buckets):
            raise ValueError("invalid_usage_tokens")
        if read + write > inp or write_30m + write_12h != write:
            raise ValueError("invalid_usage_tokens")
        input_multiplier, output_multiplier = self.price_multipliers(inp)
        return (
            (
                (inp - read - write) * self.input_usd
                + read * self.cache_read_usd
                + write_30m * self.cache_write_usd
                + write_12h * self.input_usd * 2.5
            )
            * input_multiplier
            + output * self.output_usd * output_multiplier
        ) / 1_000_000

    def price_multipliers(self, input_tokens: float) -> tuple[float, float]:
        return (2.0, 1.5) if input_tokens > self.long_input_threshold else (1.0, 1.0)


OPENAI_MODEL_PROFILES = MappingProxyType(
    {"gpt-6.1-sol": OpenAIModelProfile(2, 10, 2.50, 0.10)}
)


def get_openai_profile(model: str) -> OpenAIModelProfile:
    try:
        return OPENAI_MODEL_PROFILES[model]
    except (KeyError, TypeError):
        raise ValueError("official_model_unpriced") from None


def get_profile(model: str) -> ModelProfile:
    try:
        return MODEL_PROFILES[model]
    except (KeyError, TypeError):
        raise ValueError("official_model_unpriced") from None


def model_is_thinking(model: str, thinking: bool) -> bool:
    """Whether this model actually thinks, including mandatory Opus 5.5."""
    return get_profile(model).effective_thinking(thinking)


def request_options(
    model: str, thinking: bool, forced_tool=None, *, effort: str | None = None
) -> dict:
    """SDK 0.76.0 request options; no sampling fields or state consumption.

    5.5 models always select tools automatically; other thinking models also
    cannot force a particular tool. Callers omit tool_choice when no tools
    are supplied at all.
    """
    profile = get_profile(model)
    result = {"model": model, **profile.request_parameters(thinking, effort)}
    if profile.auto_tools_only or profile.effective_thinking(thinking):
        result["tool_choice"] = {"type": "auto"}
    elif forced_tool:
        result["tool_choice"] = {"type": "tool", "name": forced_tool}
    return result


def configuration_issue(config, provider: str = "anthropic") -> str | None:
    """Return a safe category, never configuration text or a startup exception."""

    def value(name, default):
        return (
            config.get(name, default)
            if isinstance(config, Mapping)
            else getattr(config, name, default)
        )

    model = value("model", "claude-opus-4-6")
    profiles = OPENAI_MODEL_PROFILES if provider == "openai" else MODEL_PROFILES
    if not isinstance(model, str) or model not in profiles:
        return "official_model_unpriced"
    try:
        profiles[model].validate_effort(value("effort", None))
    except ValueError:
        return "unsupported_effort"
    for field, default in (
        ("first_pass_model", "claude-sonnet-4-6"),
        ("second_pass_model", "claude-opus-4-6"),
    ):
        model = value(field, default)
        if not isinstance(model, str) or model not in CHECK_MODELS:
            return "invalid_check_model"
    return None
