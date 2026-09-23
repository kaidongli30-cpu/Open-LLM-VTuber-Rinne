from typing import Union, List, Dict, Any, Optional, Callable
import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import numpy as np

from .conversation_utils import (
    create_batch_input,
    process_agent_output,
    send_conversation_start_signals,
    process_user_input,
    finalize_conversation_turn,
    cleanup_conversation,
    EMOJI_LIST,
)
from .types import ConversationOutputMode, WebSocketSend
from .scene_continuity import decide_scene_continuity_instruction
from .tts_manager import TTSTaskManager
from .tool_call_feedback import ToolCallFeedbackManager
from ..chat_history_manager import store_message
from ..data_paths import character_history_root
from ..service_context import ServiceContext
from ..memory.live_retrieval import (
    LiveMemoryRetrievalService,
    live_retrieval_settings_from_config,
)
from ..memory.cloud_memory_tool import (
    CloudLongTermMemoryTool,
    CloudMemoryToolHandlerResult,
    LongTermRecallPolicy,
    classify_long_term_recall_policy,
)
from ..memory.recent_diary_recall import (
    RECENT_DIARY_DAYS,
    RecentDiaryRecallTool,
    apply_long_term_cutoff,
    load_recent_diary_originals,
)
from ..memory.today_history_loader import TodayHistoryLoader
from ..memory.long_term_archive import select_long_term_memories
from ..memory.layer2_context import load_current_layer2_context
from ..memory.mandatory_response_rules import load_mandatory_response_rules
from ..memory.scene_state import SceneStateStore, correct_scene_conflicts
from ..memory.trial_log import write_memory_trial_record
from ..privacy_logging import mapping_log_fields, text_log_fields
from ..video_analysis import analyze_video_attachments, settings_from_character_config

# Import necessary types from agent outputs
from ..agent.output_types import SentenceOutput, AudioOutput


def _memory_candidate_console_label(item: dict[str, Any]) -> str:
    """Return trace metadata without exposing the remembered text itself."""

    candidate_id = str(item.get("candidate_id") or "unknown")
    period = str(item.get("period") or "unknown-date")
    source = Path(str(item.get("source_file") or "unknown").replace("\\", "/")).name
    return f"{candidate_id} ({period}, {source})"


def _memory_retrieval_console_lines(
    diagnostics: dict[str, Any],
) -> list[str]:
    """Build a compact, privacy-conscious view of one completed L3 retrieval."""

    lines: list[str] = []
    events = diagnostics.get("final_event_ranking")
    if isinstance(events, list):
        top_events = (
            ", ".join(
                _memory_candidate_console_label(item)
                for item in events[:3]
                if isinstance(item, dict)
            )
            or "无"
        )
        lines.append(
            "[第三层记忆] 事件检索完成："
            f"检索通道={diagnostics.get('raw_ranking_count', 0)}，"
            f"融合候选={diagnostics.get('fused_event_count', 0)}，"
            f"保留候选={diagnostics.get('retained_event_count', len(events))}；"
            f"Top候选={top_events}"
        )
    elif diagnostics.get("selected_seed_event_id"):
        lines.append(
            "[第三层记忆] 二次日记检索完成："
            f"seed_event_id={diagnostics['selected_seed_event_id']}"
        )

    diary = diagnostics.get("diary") or {}
    if isinstance(diary, dict):
        diary_windows = diary.get("diary_windows") or []
        lines.append(
            "[第三层记忆] 日记导航完成："
            f"窗口={diary_windows or '无'}，"
            f"日记候选={diary.get('candidate_count', 0)}"
        )

    time_range = diagnostics.get("time_range")
    if isinstance(time_range, dict):
        lines.append(
            "[第三层记忆] 时间范围："
            f"{time_range.get('start_date') or '未指定'}~"
            f"{time_range.get('end_date') or '未指定'}，"
            f"置信度={time_range.get('confidence', 'none')}，"
            f"模式={time_range.get('mode', 'none')}"
        )
    if diagnostics.get("raw_chat_count"):
        lines.append(
            "[第三层记忆] 原始聊天精确回源完成："
            f"候选={diagnostics.get('raw_chat_count', 0)}"
        )

    lines.append(
        "[第三层记忆] 证据整理完成："
        f"状态={diagnostics.get('cloud_status', 'unknown')}，"
        f"返回证据={diagnostics.get('cloud_evidence_count', 0)}，"
        f"可继续打开具体日记={bool(diagnostics.get('detail_followup_available'))}"
    )
    lines.append(
        "[第三层记忆] 本次检索耗时："
        f"事件搜索={diagnostics.get('event_search_seconds', 0)}s，"
        f"事件重排={diagnostics.get('event_rerank_seconds', 0)}s，"
        f"证据搜索={diagnostics.get('evidence_search_seconds', 0)}s，"
        f"总计={diagnostics.get('total_seconds', 0)}s"
    )
    return lines


def _compact_llm_request_trace(
    payload: dict[str, Any],
    llm: Any,
) -> dict[str, Any]:
    """Keep timing/protocol evidence without duplicating private prompts."""

    compact = {
        key: payload[key]
        for key in (
            "event",
            "trace_request_id",
            "request_index",
            "model",
            "base_url",
            "elapsed_seconds",
            "error_class",
            "error_type",
            "upstream_attempts",
            "received_meaningful_output",
            "finish_reasons",
            "usage",
        )
        if key in payload
    }
    messages = payload.get("messages")
    shape_builder = getattr(llm, "_message_shape", None)
    if isinstance(messages, list) and callable(shape_builder):
        compact["message_shape"] = shape_builder(messages)
    tools = payload.get("tools")
    compact["tool_count"] = len(tools) if isinstance(tools, list) else 0
    text = payload.get("text")
    if isinstance(text, str):
        compact["response_text_length"] = len(text)
    tool_calls = payload.get("tool_calls")
    if isinstance(tool_calls, list):
        compact["returned_tool_call_count"] = len(tool_calls)
    recovery = payload.get("protocol_recovery")
    if isinstance(recovery, dict):
        compact["protocol_recovery"] = {
            key: recovery[key]
            for key in (
                "attempted",
                "forced_tool_name",
                "reason",
                "outcome",
                "selection_option_count",
            )
            if key in recovery
        }
    return compact


async def process_single_conversation(
    context: ServiceContext,
    websocket_send: WebSocketSend,
    client_uid: str,
    user_input: Union[str, np.ndarray],
    images: Optional[List[Dict[str, Any]]] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
    session_emoji: str = np.random.choice(EMOJI_LIST),
    metadata: Optional[Dict[str, Any]] = None,
    output_mode: ConversationOutputMode = ConversationOutputMode.DESKTOP,
    response_text_filter: Optional[Callable[[str], str]] = None,
) -> str:
    """Process a single-user conversation turn

    Args:
        context: Service context containing all configurations and engines
        websocket_send: WebSocket send function
        client_uid: Client unique identifier
        user_input: Text or audio input from user
        images: Optional list of image data
        attachments: Optional validated local-library references
        session_emoji: Emoji identifier for the conversation
        metadata: Optional metadata for special processing flags

    Returns:
        str: Complete response text
    """
    metadata = dict(metadata or {})
    if attachments:
        metadata.setdefault("file_attachments", attachments)
    request_started_perf = float(
        metadata.get("_request_received_perf", time.perf_counter())
    )
    request_received_at = str(
        metadata.get("_request_received_at", datetime.now().astimezone().isoformat())
    )
    first_visible_seconds: float | None = None
    first_spoken_seconds: float | None = None
    first_tool_feedback_visible_seconds: float | None = None
    first_tool_feedback_spoken_seconds: float | None = None
    sending_tool_feedback = False

    async def monitored_websocket_send(payload: str) -> None:
        nonlocal first_visible_seconds, first_spoken_seconds
        nonlocal first_tool_feedback_visible_seconds
        nonlocal first_tool_feedback_spoken_seconds
        try:
            parsed = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict):
            message_type = parsed.get("type")
            if message_type == "full-text":
                text = parsed.get("text")
                if (
                    isinstance(text, str)
                    and text.strip()
                    and text not in {"Thinking...", "Connection established"}
                ):
                    elapsed = round(time.perf_counter() - request_started_perf, 3)
                    if sending_tool_feedback:
                        if first_tool_feedback_visible_seconds is None:
                            first_tool_feedback_visible_seconds = elapsed
                    elif first_visible_seconds is None:
                        first_visible_seconds = elapsed
            elif message_type == "audio" and parsed.get("audio"):
                elapsed = round(time.perf_counter() - request_started_perf, 3)
                if sending_tool_feedback:
                    if first_tool_feedback_spoken_seconds is None:
                        first_tool_feedback_spoken_seconds = elapsed
                elif first_spoken_seconds is None:
                    first_spoken_seconds = elapsed
        await websocket_send(payload)

    tts_manager = TTSTaskManager()
    tool_feedback_manager = ToolCallFeedbackManager(
        tts_manager=tts_manager,
        live2d_model=context.live2d_model,
        tts_engine=context.tts_engine,
        websocket_send=monitored_websocket_send,
        character_name=context.character_config.character_name,
        character_avatar=context.character_config.avatar,
        translate_engine=context.translate_engine,
        enabled=output_mode is ConversationOutputMode.DESKTOP,
    )
    full_response = ""
    input_text = ""
    hidden_memory_context: str | None = None
    retrieval_diagnostics: dict[str, Any] = {"status": "not_requested"}
    retrieval_seconds = 0.0
    memory_tool: CloudLongTermMemoryTool | None = None
    recent_diary_tool: RecentDiaryRecallTool | None = None
    recent_diary_diagnostics: dict[str, Any] = {"status": "not_requested"}
    layer2_context: str = ""
    layer2_diagnostics: dict[str, Any] = {"status": "disabled"}
    mandatory_response_rules: str = ""
    mandatory_response_rules_diagnostics: dict[str, Any] = {"status": "missing"}
    video_analysis_context: str = ""
    video_analysis_diagnostics: dict[str, Any] = {"status": "not_requested"}
    cloud_request_started_seconds: float | None = None
    trial_log_enabled = False
    llm_request_traces: list[dict[str, Any]] = []
    traced_llm = None
    previous_llm_trace_sink = None
    llm_trace_token = None
    history_root = character_history_root(context.character_config.conf_uid)
    basic_memory_config = (
        context.character_config.agent_config.agent_settings.basic_memory_agent
    )
    scene_memory_enabled = bool(
        getattr(basic_memory_config, "scene_memory_enabled", False)
    )
    scene_store = SceneStateStore(history_root) if scene_memory_enabled else None
    scene_memory_day: str | None = None
    scene_snapshot: dict[str, Any] = {}
    scene_deltas: list[dict[str, Any]] = []

    try:
        await send_conversation_start_signals(monitored_websocket_send)
        logger.info(f"New Conversation Chain {session_emoji} started!")

        input_text = await process_user_input(
            user_input,
            context.asr_engine,
            monitored_websocket_send,
            attachments,
        )
        media_settings = settings_from_character_config(context.character_config)
        if any(item.get("kind") == "video" for item in attachments or []):
            (
                video_analysis_context,
                video_analysis_diagnostics,
            ) = await analyze_video_attachments(
                attachments or [], media_settings, user_input=input_text
            )
            metadata["video_analysis_context"] = video_analysis_context
            logger.info(
                "[视频观察] 本轮处理完成：status={}，video_count={}，"
                "complete_count={}，seconds={}",
                video_analysis_diagnostics.get("status"),
                video_analysis_diagnostics.get("video_count", 0),
                video_analysis_diagnostics.get("complete_count", 0),
                video_analysis_diagnostics.get("seconds", 0),
            )

        scene_continuity = await asyncio.to_thread(
            decide_scene_continuity_instruction,
            history_root,
            source_channel=str(metadata.get("source_channel", "desktop")),
            reference_time=datetime.fromisoformat(request_received_at),
        )
        if scene_continuity.enabled:
            metadata["scene_continuity_system_instruction"] = (
                scene_continuity.instruction
            )
            logger.info(
                "[场景连续性] 本轮在系统提示末尾启用场景提醒：{}",
                scene_continuity.reason,
            )

        if scene_store is not None:
            try:
                scene_snapshot, scene_context = await asyncio.to_thread(
                    scene_store.prepare_turn,
                    input_text,
                    timestamp=request_received_at,
                    source_channel=str(metadata.get("source_channel", "desktop")),
                )
                scene_memory_day = str(scene_snapshot.get("memory_day") or "") or None
                metadata["current_scene_context"] = scene_context
                metadata["scene_memory_day"] = scene_memory_day
            except Exception as exc:
                # Scene continuity must not take down an otherwise usable reply.
                # The warning is intentionally content-free; the next turn retries
                # the shared store.
                logger.warning(f"[当前场景] 读取或写入失败，本轮跳过场景更新：{exc}")
        layer2_settings = context.character_config.layer2_memory_generation
        if layer2_settings.enabled and layer2_settings.inject_into_conversation:
            layer2_result = await asyncio.to_thread(
                load_current_layer2_context, history_root
            )
            layer2_context = layer2_result.context
            layer2_diagnostics = layer2_result.diagnostics
            if layer2_context:
                metadata["layer2_user_background"] = layer2_context
            elif layer2_diagnostics.get("status") == "invalid":
                logger.warning(
                    "[第二层记忆] 当前发布未通过完整性校验，本轮继续使用"
                    "其余记忆且不注入损坏背景："
                    f"{layer2_diagnostics.get('error', 'unknown')}"
                )
        mandatory_rules_result = await asyncio.to_thread(
            load_mandatory_response_rules, history_root
        )
        mandatory_response_rules = mandatory_rules_result.context
        mandatory_response_rules_diagnostics = mandatory_rules_result.diagnostics
        if mandatory_response_rules:
            metadata["mandatory_response_rules"] = mandatory_response_rules
            logger.info(
                "[最高回复方式] 已加载：chars={}，sha256={}",
                mandatory_response_rules_diagnostics.get("character_count", 0),
                mandatory_response_rules_diagnostics.get("sha256", "unknown"),
            )
        elif mandatory_response_rules_diagnostics.get("status") == "invalid":
            logger.error(
                "[最高回复方式] 文件无效，本轮未注入：{}",
                mandatory_response_rules_diagnostics.get("error", "unknown"),
            )
        retrieval_config = context.character_config.agent_config.agent_settings.basic_memory_agent.long_term_memory_retrieval
        trial_log_enabled = bool(retrieval_config.trial_logging)
        llm = getattr(context.agent_engine, "_llm", None)
        set_trace_sink = getattr(llm, "set_trace_sink", None)
        if trial_log_enabled and callable(set_trace_sink):
            traced_llm = llm
            previous_llm_trace_sink = getattr(llm, "_trace_sink", None)

            def capture_llm_trace(payload: dict[str, Any]) -> None:
                llm_request_traces.append(_compact_llm_request_trace(payload, llm))
                if callable(previous_llm_trace_sink):
                    previous_llm_trace_sink(payload)

            push_trace_sink = getattr(llm, "push_trace_sink", None)
            if callable(push_trace_sink):
                llm_trace_token = push_trace_sink(capture_llm_trace)
            else:
                set_trace_sink(capture_llm_trace)
        llm_base_url = str(getattr(llm, "base_url", ""))
        is_local_ollama = (
            "localhost:11434" in llm_base_url or "127.0.0.1:11434" in llm_base_url
        )
        skip_retrieval = bool(metadata.get("proactive_speak")) or is_local_ollama
        recall_policy = classify_long_term_recall_policy(input_text)

        if (
            retrieval_config.enabled
            and input_text.strip()
            and not skip_retrieval
            and recall_policy is not LongTermRecallPolicy.OFF
        ):
            request_datetime = datetime.fromisoformat(request_received_at)
            current_memory_day, _day_start, _day_end = TodayHistoryLoader(
                history_root
            ).memory_day_window(request_datetime)
            long_term_cutoff = current_memory_day - timedelta(
                days=RECENT_DIARY_DAYS + 1
            )

            async def run_recent_diary_tool(
                arguments: dict[str, Any],
            ) -> CloudMemoryToolHandlerResult:
                nonlocal recent_diary_diagnostics
                query = " ".join(
                    part
                    for part in (
                        input_text.strip(),
                        str(arguments.get("query") or "").strip(),
                    )
                    if part
                )
                result = await asyncio.to_thread(
                    load_recent_diary_originals,
                    history_root,
                    query=query,
                    reference_time=request_datetime,
                )
                recent_diary_diagnostics = result.diagnostics
                logger.info(
                    "[近三日日记原文] 静默核对完成：days={}，files={}，chars={}",
                    recent_diary_diagnostics.get("source_memory_days", []),
                    recent_diary_diagnostics.get("source_files", []),
                    recent_diary_diagnostics.get("character_count", 0),
                )
                return CloudMemoryToolHandlerResult(
                    content=result.context,
                    diagnostics=recent_diary_diagnostics,
                    is_error=False,
                )

            recent_diary_tool = RecentDiaryRecallTool(
                run_recent_diary_tool,
                turn_started_perf=request_started_perf,
            )
            metadata["recent_diary_recall_tool"] = recent_diary_tool

            async def run_long_term_memory_tool(
                arguments: dict[str, Any],
            ) -> CloudMemoryToolHandlerResult:
                nonlocal hidden_memory_context, retrieval_diagnostics
                try:
                    bounded_arguments, recent_reason = apply_long_term_cutoff(
                        arguments,
                        memory_day=current_memory_day,
                    )
                    if bounded_arguments is None:
                        logger.info(
                            "[记忆路由] 第三层请求落在近三日保护窗口，"
                            "改为静默读取近期日记原文：reason={}",
                            recent_reason,
                        )
                        return await run_recent_diary_tool(arguments)
                    arguments = bounded_arguments
                    await tool_feedback_manager.announce_memory_recall()
                    seed_event_id = arguments.get("seed_event_id")
                    if seed_event_id:
                        logger.info(
                            "[第三层记忆] 云端模型请求继续打开具体日记："
                            f"seed_event_id={seed_event_id}，"
                            f"粒度={arguments['question_granularity']}"
                        )
                    else:
                        logger.info(
                            "[第三层记忆] 云端模型已判断需要长期检索："
                            "query={}，粒度={}，时间范围={}~{}（{}）",
                            text_log_fields(arguments.get("query")),
                            arguments.get("question_granularity"),
                            arguments.get("start_date") or "未指定",
                            arguments.get("end_date") or "未指定",
                            arguments.get("time_range_confidence") or "none",
                        )
                    service = getattr(
                        context,
                        "live_memory_retrieval_service",
                        None,
                    )
                    if service is None:
                        logger.info(
                            "[第三层记忆] 正在初始化本地向量检索与重排服务；"
                            "首次调用可能需要加载模型。"
                        )
                        settings = live_retrieval_settings_from_config(retrieval_config)
                        service = await asyncio.to_thread(
                            LiveMemoryRetrievalService,
                            history_root,
                            settings=settings,
                        )
                        context.live_memory_retrieval_service = service
                        logger.info("[第三层记忆] 本地检索服务初始化完成。")
                    logger.info(
                        "[第三层记忆] 开始执行"
                        + ("二次日记检索。" if seed_event_id else "事件与日记检索。")
                    )
                    if seed_event_id:
                        retrieval_result = await asyncio.to_thread(
                            service.retrieve_detail_from_seed,
                            input_text,
                            seed_event_id=seed_event_id,
                            question_granularity=arguments["question_granularity"],
                            start_date=arguments.get("start_date"),
                            end_date=arguments.get("end_date"),
                            time_range_confidence=arguments.get(
                                "time_range_confidence", "none"
                            ),
                            time_range_basis=arguments.get("time_range_basis", ""),
                        )
                    else:
                        retrieval_result = await asyncio.to_thread(
                            service.retrieve_from_cloud_request,
                            input_text,
                            retrieval_query=arguments["query"],
                            question_granularity=arguments["question_granularity"],
                            start_date=arguments.get("start_date"),
                            end_date=arguments.get("end_date"),
                            time_range_confidence=arguments.get(
                                "time_range_confidence", "none"
                            ),
                            time_range_basis=arguments.get("time_range_basis", ""),
                        )
                    hidden_memory_context = retrieval_result.hidden_context
                    retrieval_diagnostics = {
                        "status": "completed",
                        **retrieval_result.diagnostics,
                    }
                    for trace_line in _memory_retrieval_console_lines(
                        retrieval_diagnostics
                    ):
                        logger.info(trace_line)
                    return CloudMemoryToolHandlerResult(
                        content=(
                            hidden_memory_context
                            or "未找到可供回答的长期记忆。请不要猜测用户经历。"
                        ),
                        diagnostics=retrieval_diagnostics,
                    )
                except Exception as exc:
                    logger.exception(
                        "Cloud-requested long-term-memory retrieval failed"
                    )
                    retrieval_diagnostics = {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    if not retrieval_config.legacy_archive_fallback_on_error:
                        raise
                    selection = await asyncio.to_thread(
                        select_long_term_memories,
                        history_root,
                    )
                    selection.monthly_entries = [
                        entry
                        for entry in selection.monthly_entries
                        if entry.end_date <= long_term_cutoff
                    ]
                    selection.weekly_entries = [
                        entry
                        for entry in selection.weekly_entries
                        if entry.end_date <= long_term_cutoff
                    ]
                    selection.diary_entries = [
                        entry
                        for entry in selection.diary_entries
                        if entry.end_date <= long_term_cutoff
                    ]
                    legacy_text = selection.to_llm_text()
                    if not legacy_text:
                        raise
                    hidden_memory_context = (
                        "【系统临时启用的旧归档兜底】\n"
                        "长期记忆工具本轮不可用。以下日记、周记、月记仅作为"
                        "这一次请求的临时参考，不会恢复为常驻记忆。"
                        "除非用户明确要求时间线、总结或完整回复，否则不要复述"
                        "其中包含的日期信息；不要提及检索过程。\n\n" + legacy_text
                    )
                    retrieval_diagnostics["legacy_fallback"] = {
                        "used": True,
                        "monthly_count": len(selection.monthly_entries),
                        "weekly_count": len(selection.weekly_entries),
                        "diary_count": len(selection.diary_entries),
                    }
                    return CloudMemoryToolHandlerResult(
                        content=hidden_memory_context,
                        diagnostics=retrieval_diagnostics,
                    )

            memory_tool = CloudLongTermMemoryTool(
                run_long_term_memory_tool,
                turn_started_perf=request_started_perf,
                recall_policy=recall_policy,
                max_retrieval_seconds=retrieval_config.max_retrieval_seconds,
            )
            metadata["long_term_memory_tool"] = memory_tool
            retrieval_diagnostics = {
                "status": "awaiting_cloud_decision",
                "recall_policy": recall_policy.value,
            }
        elif retrieval_config.enabled:
            if recall_policy is LongTermRecallPolicy.OFF:
                skip_reason = "explicit_memory_opt_out"
            elif metadata.get("proactive_speak"):
                skip_reason = "proactive_speak"
            elif is_local_ollama:
                skip_reason = "local_ollama_reply_model"
            else:
                skip_reason = "empty_input"
            retrieval_diagnostics = {
                "status": "skipped",
                "reason": skip_reason,
                "recall_policy": recall_policy.value,
            }

        batch_input = create_batch_input(
            input_text=input_text,
            images=images,
            from_name=context.character_config.human_name,
            metadata=metadata,
        )

        skip_history = metadata.get("skip_history", False)
        if context.history_uid and not skip_history:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="human",
                content=input_text,
                name=context.character_config.human_name,
                attachments=attachments,
            )
        if skip_history:
            logger.debug("Skipping storing user input to history (proactive speak)")

        logger.info(f"User input: {input_text}")
        if images:
            logger.info(f"With {len(images)} images")

        try:
            cloud_request_started_seconds = round(
                time.perf_counter() - request_started_perf, 3
            )
            agent_output_stream = context.agent_engine.chat(batch_input)
            async for output_item in agent_output_stream:
                if (
                    isinstance(output_item, dict)
                    and output_item.get("type") == "tool_call_status"
                ):
                    output_item["name"] = context.character_config.character_name
                    logger.debug(
                        "Sending tool status update: {}",
                        mapping_log_fields(output_item),
                    )
                    await monitored_websocket_send(json.dumps(output_item))
                    sending_tool_feedback = True
                    try:
                        await tool_feedback_manager.handle_tool_status(output_item)
                    finally:
                        sending_tool_feedback = False
                elif isinstance(output_item, (SentenceOutput, AudioOutput)):
                    if isinstance(output_item, SentenceOutput) and scene_snapshot:
                        corrected_display, display_corrections = (
                            correct_scene_conflicts(
                                output_item.display_text.text,
                                scene_snapshot,
                            )
                        )
                        corrected_tts, tts_corrections = correct_scene_conflicts(
                            output_item.tts_text,
                            scene_snapshot,
                        )
                        if display_corrections or tts_corrections:
                            logger.warning(
                                "[当前场景] 已阻止与共同场景冲突的分离表述："
                                "display_corrections={}，tts_corrections={}",
                                display_corrections,
                                tts_corrections,
                            )
                            output_item.display_text.text = corrected_display
                            output_item.tts_text = corrected_tts
                    response_part = await process_agent_output(
                        output=output_item,
                        character_config=context.character_config,
                        live2d_model=context.live2d_model,
                        tts_engine=context.tts_engine,
                        websocket_send=monitored_websocket_send,
                        tts_manager=tts_manager,
                        translate_engine=context.translate_engine,
                        output_mode=output_mode,
                    )
                    full_response += (
                        str(response_part) if response_part is not None else ""
                    )
                else:
                    logger.warning(
                        "Received unexpected item type from agent chat stream: "
                        f"{type(output_item)}"
                    )
                    logger.debug(
                        "Unexpected item metadata: {}",
                        mapping_log_fields(output_item),
                    )
        except Exception as e:
            logger.exception(f"Error processing agent response stream: {e}")
            await monitored_websocket_send(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"Error processing agent response: {str(e)}",
                    }
                )
            )

        if response_text_filter is not None:
            full_response = response_text_filter(full_response)

        await finalize_conversation_turn(
            tts_manager=tts_manager,
            websocket_send=monitored_websocket_send,
            client_uid=client_uid,
        )

        scene_deltas = metadata.get("_scene_deltas")
        if (
            scene_store is not None
            and scene_memory_day
            and isinstance(scene_deltas, list)
        ):
            for scene_delta in scene_deltas:
                if not isinstance(scene_delta, dict):
                    continue
                try:
                    scene_snapshot = await asyncio.to_thread(
                        scene_store.apply_assistant_delta,
                        scene_delta,
                        memory_day=scene_memory_day,
                        source_channel=str(metadata.get("source_channel", "desktop")),
                    )
                except Exception as exc:
                    logger.warning(f"[当前场景] 保存凛祢场景更新失败：{exc}")

        if context.history_uid and full_response:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="ai",
                content=full_response,
                name=context.character_config.character_name,
                avatar=context.character_config.avatar,
            )
            logger.info("AI response stored: {}", text_log_fields(full_response))
        return full_response

    except asyncio.CancelledError:
        logger.info(f"🤡👍 Conversation {session_emoji} cancelled because interrupted.")
        raise
    except Exception as e:
        logger.error(f"Error in conversation chain: {e}")
        await monitored_websocket_send(
            json.dumps({"type": "error", "message": f"Conversation error: {str(e)}"})
        )
        raise
    finally:
        if traced_llm is not None:
            try:
                reset_trace_sink = getattr(traced_llm, "reset_trace_sink", None)
                if llm_trace_token is not None and callable(reset_trace_sink):
                    reset_trace_sink(llm_trace_token)
                else:
                    traced_llm.set_trace_sink(previous_llm_trace_sink)
            except Exception as exc:
                logger.warning(f"[记忆试用日志] 恢复模型跟踪器失败：{exc}")
        memory_tool_snapshot = memory_tool.snapshot() if memory_tool else None
        recent_diary_tool_snapshot = (
            recent_diary_tool.snapshot() if recent_diary_tool else None
        )
        if memory_tool_snapshot is not None:
            retrieval_seconds = float(memory_tool_snapshot["elapsed_seconds"])
            if memory_tool_snapshot["called"]:
                retrieval_diagnostics = {
                    **memory_tool_snapshot["diagnostics"],
                    "cloud_tool": {
                        key: value
                        for key, value in memory_tool_snapshot.items()
                        if key
                        not in {
                            "diagnostics",
                            "tool_result_sent_to_cloud",
                        }
                    },
                }
            else:
                retrieval_diagnostics = {
                    "status": "not_requested",
                    "cloud_tool": memory_tool_snapshot,
                }
        if trial_log_enabled and not metadata.get("proactive_speak"):
            trial_record = {
                "trial_record_version": 2,
                "turn_id": str(uuid.uuid4()),
                "request_received_at": request_received_at,
                "completed_at": datetime.now().astimezone().isoformat(),
                "user_input": input_text,
                "timing": {
                    "retrieval_seconds": retrieval_seconds,
                    "cloud_request_started_seconds": cloud_request_started_seconds,
                    "tool_feedback_visible_seconds": (
                        first_tool_feedback_visible_seconds
                    ),
                    "tool_feedback_spoken_seconds": (
                        first_tool_feedback_spoken_seconds
                    ),
                    "first_visible_sentence_seconds": first_visible_seconds,
                    "first_spoken_sentence_seconds": first_spoken_seconds,
                    "turn_total_seconds": round(
                        time.perf_counter() - request_started_perf, 3
                    ),
                },
                "assistant_response": full_response,
                "hidden_context_sent_to_cloud": hidden_memory_context,
                "recent_memory_context_sent_to_cloud": getattr(
                    context,
                    "recent_memory_context",
                    "",
                ),
                "recent_memory_diagnostics": getattr(
                    context,
                    "recent_memory_diagnostics",
                    {},
                ),
                "layer2_user_background_sent_to_model": layer2_context,
                "layer2_memory_diagnostics": layer2_diagnostics,
                "mandatory_response_rules_sent_to_model": mandatory_response_rules,
                "mandatory_response_rules_diagnostics": (
                    mandatory_response_rules_diagnostics
                ),
                "video_analysis_diagnostics": video_analysis_diagnostics,
                "scene_memory": scene_snapshot,
                "scene_deltas": list(scene_deltas)
                if isinstance(scene_deltas, list)
                else [],
                "cloud_memory_tool": memory_tool_snapshot,
                "recent_diary_recall_tool": recent_diary_tool_snapshot,
                "recent_diary_recall": recent_diary_diagnostics,
                "retrieval": retrieval_diagnostics,
                "llm_requests": llm_request_traces,
            }
            try:
                path = await asyncio.to_thread(
                    write_memory_trial_record, history_root, trial_record
                )
                logger.info(f"[记忆试用日志] 已记录：{path}")
            except Exception as exc:
                logger.warning(f"[记忆试用日志] 写入失败：{exc}")
        await tool_feedback_manager.stop()
        cleanup_conversation(tts_manager, session_emoji)
