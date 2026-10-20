"""Independent DeepSeek life-summary configuration; disabled for old configs."""

from typing import ClassVar, Literal

from pydantic import Field

from .i18n import Description, I18nMixin


class TodaySummaryConfig(I18nMixin):
    enabled: bool = Field(False)
    llm_provider: Literal["deepseek_llm"] = Field("deepseek_llm")
    model: str | None = Field(None)
    reasoning_effort: Literal["none", "low", "medium", "high"] = Field("low")
    interval_turns: int = Field(10, ge=1)
    max_output_tokens: int = Field(16384, ge=1)
    timeout_seconds: float = Field(180, gt=0)
    reuse_for_diary: bool = Field(True)

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "enabled": Description(
            en="Save today's life summary in background.",
            zh="后台生成并保存今日生活摘要；不改变对话大脑的历史。",
        ),
        "llm_provider": Description(
            en="Independent provider credentials from llm_configs.",
            zh="独立使用llm_configs中的deepseek_llm入口，不跟随对话或日记模型。",
        ),
        "model": Description(
            en="Optional model override for summaries only.",
            zh="摘要专用模型；为空时使用deepseek_llm入口配置的模型。",
        ),
        "interval_turns": Description(
            en="Rebuild after this many completed turns.",
            zh="每隔多少轮完整往返重建摘要。",
        ),
        "reasoning_effort": Description(
            en="Reasoning effort for summaries only.",
            zh="摘要专用思考强度，不改变对话或日记模型。",
        ),
        "max_output_tokens": Description(
            en="Completion budget including provider reasoning.",
            zh="单次调用输出预算；须容纳服务商返回的思考与结构化正文。",
        ),
        "timeout_seconds": Description(
            en="Provider read timeout in seconds.", zh="单次摘要请求的读取超时秒数。"
        ),
        "reuse_for_diary": Description(
            en="Reuse valid same-day summary plus uncovered tail for diaries.",
            zh="用有效同日摘要和未覆盖尾部写日记；不可用时沿用旧流程。",
        ),
    }
