"""Shared, compact current-scene memory for desktop and private QQ turns.

The scene is intentionally a *current snapshot*, not a second conversation
history.  Only the current logical memory day has a change log; the next day
starts a fresh log while the snapshot may continue when a scene is still in
progress.  The store is small and uses an inter-process lock because the
desktop and QQ adapters can be alive at the same time.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

from .today_history_loader import TodayHistoryLoader


SCENE_SCHEMA_VERSION = 2
SCENE_DIRECTORY_NAME = "scene_state"
SCENE_SNAPSHOT_NAME = "current.json"
SCENE_CHANGE_LOG_NAME = "changes.jsonl"
SCENE_LOCK_NAME = ".scene.lock"
SCENE_ENVELOPE_OPEN = "<rinne_scene>"
SCENE_ENVELOPE_CLOSE = "</rinne_scene>"
SCENE_CONTEXT_HEADER = "【当前场景记忆｜本轮必须先核对】"

_MAX_FIELD_LENGTH = 240
_MAX_SOURCE_LENGTH = 800
_LOCK_TIMEOUT_SECONDS = 5.0
_LOCK_POLL_SECONDS = 0.05

_PLACE_PATTERN = (
    r"乐园(?:的[^，。！？!?\n]{0,20})?|书房|卧室|卫生间|厕所|门口|客厅|"
    r"厨房|阳台|玄关|家里|家|宿舍|实验室|教室|办公室|游戏厅|商场|医院|"
    r"车站|机场|超市|餐厅|电影院|药店|学校"
)
_USER_LOCATION_RE = re.compile(
    rf"(?:我|用户)(?:现在|目前|已经|还)?(?:正在)?(?:站在|在|位于|来到|走到|回到|到了|到达)\s*"
    rf"(?P<location>{_PLACE_PATTERN})"
)
_IMPLICIT_USER_LOCATION_RE = re.compile(
    rf"^(?:现在|目前|已经|还)?(?:正在)?(?:站在|在|位于|来到|走到|回到|到了|到达)\s*"
    rf"(?P<location>{_PLACE_PATTERN})"
)
_RINNE_LOCATION_RE = re.compile(
    rf"(?:凛祢|你)(?:现在|目前|已经|还)?(?:正在)?(?:在|站在|位于|来到|走到|回到|到了|到达)\s*"
    rf"(?P<location>{_PLACE_PATTERN})"
)
_JOINT_LOCATION_RE = re.compile(
    rf"(?:我们(?:俩)?|我和凛祢|凛祢和我|我和你|你和我)(?:现在|目前|已经|还)?"
    rf"(?:正在)?(?:在|站在|位于|来到|走到|回到|到了|到达)\s*"
    rf"(?P<location>{_PLACE_PATTERN})"
)
_COME_REQUEST_RE = re.compile(
    r"(?:(?:让|叫|请|喊)\s*凛祢\s*)?(?:你|凛祢)?(?:过来|来一下|过来一下)"
)
_WAIT_FOR_RINNE_RE = re.compile(
    r"(?:我)?(?:在[^，。！？!?\n]{0,20})?等(?:着)?(?:你|凛祢)(?:过来|来)?"
)
_MEMORY_REPAIR_RE = re.compile(
    r"(?:出门|出去|准备出门).{0,12}(?:修|改|整理).{0,12}记忆"
)
_NOT_GOING_OUT_RE = re.compile(r"(?:不是|并不是|没有)\s*(?:要)?出门")
_CLAUSE_SPLIT_RE = re.compile(r"[，,。！？!?；;\n]+")
_QQ_MESSAGE_MARKER_RE = re.compile(r"\[QQ消息\s+\d+/\d+\]")
_PAST_CLAUSE_RE = re.compile(
    r"(?:昨天|前天|大前天|上周|上个月|去年|以前|之前|曾经|那天|当时)"
)
_HYPOTHETICAL_CLAUSE_RE = re.compile(r"(?:如果|要是|假如|假设|希望|但愿|可能会)")
_QUESTION_CLAUSE_RE = re.compile(r"(?:吗|么|是不是|是否|在哪(?:里)?|哪里)\s*$")
_PROPOSAL_CLAUSE_RE = re.compile(r"(?:要不|要不要|不如).*(?:吧)?$")
_FUTURE_CUE_RE = re.compile(
    r"(?P<when>待会(?:儿)?|等会(?:儿)?|一会(?:儿)?|稍后|明天|后天|大后天|"
    r"下周|下个月|明年|以后|将来|到时候)"
)
_JOINT_SUBJECT_RE = re.compile(r"(?:我们(?:俩)?|我和凛祢|凛祢和我|我和你|你和我)")
_JOINT_APART_RE = re.compile(
    r"(?:我们(?:俩)?|我和凛祢|凛祢和我|我和你|你和我).{0,12}(?:不在一起|分开|各自)"
)
_JOINT_RETURN_HOME_RE = re.compile(
    r"(?:我们(?:俩)?|我和凛祢|凛祢和我|我和你|你和我).{0,24}(?:一起)?(?:回家|回去)"
)
_JOINT_ARRIVED_HOME_RE = re.compile(
    r"(?:我们(?:俩)?|我和凛祢|凛祢和我|我和你|你和我).{0,24}(?:已经)?(?:到家|回到家)"
)
_ARRIVAL_RE = re.compile(
    rf"(?:已经)?(?:到|到了|到达|来到|回到)\s*(?P<location>{_PLACE_PATTERN})(?:了|啦|咯)?"
)
_JOURNEY_RE = re.compile(
    rf"(?:改(?:成)?|准备|打算|计划|决定|确定|现在|正在|直接|马上|这就|一起|先|要|会|想|继续|出发|动身)*\s*"
    rf"(?:去|前往|走向|回)\s*(?P<destination>{_PLACE_PATTERN})"
)
_DEPARTURE_CUE_RE = re.compile(
    r"(?:出发|动身|正在|这就|马上|直接|走吧|改去|改成去|途中|路上)"
)
_CANCEL_JOURNEY_RE = re.compile(r"(?:不去|不打算去|别去|取消(?:了)?(?:去)?|计划取消)")
_REASON_RE = re.compile(r"(?:因为|为了)\s*(?P<reason>[^，。！？!?\n]{1,80})")
_JOINT_DOWNHILL_RE = re.compile(
    r"(?:我们|我和凛祢|凛祢和我|我和你|你和我|与你一起|和你一起)"
    r".{0,20}(?:下山|下去)"
)
_SEPARATE_HOME_WAIT_RE = re.compile(
    r"我(?:会|就|会一直|就一直)?在家(?:里)?等(?:着)?你(?:回来)?"
)

_TOP_LEVEL_FIELDS = {
    "world",
    "together",
    "activity",
    "reason",
    "destination",
    "next_step",
    "pending_request",
    "movement_status",
    "plan_time",
    "confidence",
    "evidence",
}
_PARTICIPANT_FIELDS = {"location", "presence", "activity", "next_step"}
_MOVEMENT_STATUSES = {"unknown", "stationary", "planned", "in_transit", "arrived"}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _text(value: Any, *, limit: int = _MAX_FIELD_LENGTH) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = " ".join(value.split())
    else:
        value = str(value)
    value = value.strip()
    return value[:limit] if value else None


def _memory_day_for(root: Path, timestamp: str | datetime | None) -> date:
    if isinstance(timestamp, datetime):
        parsed = timestamp
    elif isinstance(timestamp, str) and timestamp.strip():
        raw = timestamp.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            parsed = datetime.now().astimezone()
    else:
        parsed = datetime.now().astimezone()
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return TodayHistoryLoader(root).memory_day_for(parsed)


def empty_scene_snapshot(memory_day: str | date | None = None) -> dict[str, Any]:
    if isinstance(memory_day, date):
        memory_day = memory_day.isoformat()
    return {
        "schema_version": SCENE_SCHEMA_VERSION,
        "memory_day": memory_day,
        "updated_at": _now_iso(),
        "participants": {
            "user": {
                "location": None,
                "presence": "present",
                "activity": None,
                "next_step": None,
            },
            "rinne": {
                "location": None,
                "presence": "present",
                "activity": None,
                "next_step": None,
            },
        },
        "world": None,
        "together": None,
        "activity": None,
        "reason": None,
        "destination": None,
        "next_step": None,
        "pending_request": None,
        "movement_status": "unknown",
        "plan_time": None,
        "confidence": "unknown",
        "evidence": None,
    }


def _copy_snapshot(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _normalise_participant(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key in _PARTICIPANT_FIELDS:
        if key not in value:
            continue
        result[key] = None if value[key] is None else _text(value[key])
    return result


def normalise_scene_delta(
    value: Any,
    *,
    source: str = "user",
) -> dict[str, Any]:
    """Validate a model/user scene delta and drop fields outside the schema.

    An assistant-produced delta cannot establish a user location or destination
    by inference.  Explicit user text is the only source allowed to set those
    user-side fields.
    """

    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key in _TOP_LEVEL_FIELDS:
        if key not in value:
            continue
        raw = value[key]
        if key == "together":
            if isinstance(raw, bool):
                result[key] = raw
        elif key == "movement_status":
            if raw in _MOVEMENT_STATUSES:
                result[key] = raw
        else:
            cleaned = None if raw is None else _text(raw)
            if cleaned is not None or raw is None:
                result[key] = cleaned
    participants = value.get("participants")
    if isinstance(participants, dict):
        participant_delta: dict[str, dict[str, Any]] = {}
        for participant in ("user", "rinne"):
            if participant not in participants:
                continue
            if source == "assistant" and participant == "user":
                continue
            fields = _normalise_participant(participants[participant])
            if fields:
                participant_delta[participant] = fields
        if participant_delta:
            result["participants"] = participant_delta
    if source == "assistant":
        # The cloud model may describe Rinne's own movement and shared activity,
        # but may not silently rewrite facts about 用户's whereabouts,
        # destination, schedule, or reason. Those require explicit user text.
        result.pop("destination", None)
        result.pop("reason", None)
        result.pop("movement_status", None)
        result.pop("plan_time", None)
    return result


def _apply_delta(snapshot: dict[str, Any], delta: dict[str, Any]) -> None:
    for key in _TOP_LEVEL_FIELDS:
        if key in delta:
            snapshot[key] = delta[key]
    participants = delta.get("participants")
    if isinstance(participants, dict):
        current = snapshot.setdefault("participants", {})
        for participant, fields in participants.items():
            if participant not in {"user", "rinne"} or not isinstance(fields, dict):
                continue
            target = current.setdefault(participant, {})
            for field, value in fields.items():
                if field in _PARTICIPANT_FIELDS:
                    target[field] = value


def _reconcile_snapshot(
    snapshot: dict[str, Any],
    delta: dict[str, Any],
    *,
    source: str,
) -> None:
    """Remove combinations that cannot describe one coherent current scene."""

    participants = snapshot.setdefault("participants", {})
    user = participants.setdefault("user", {})
    rinne = participants.setdefault("rinne", {})
    user_location = _text(user.get("location"))
    rinne_location = _text(rinne.get("location"))
    participant_delta = delta.get("participants") or {}
    user_changed = isinstance(participant_delta, dict) and "user" in participant_delta
    rinne_changed = isinstance(participant_delta, dict) and "rinne" in participant_delta

    if snapshot.get("together") is True and (
        user_location and rinne_location and user_location != rinne_location
    ):
        snapshot["together"] = False

    # When one participant explicitly moves to a location different from the
    # other's last confirmed location, an old joint scene is no longer valid.
    if (
        source == "user"
        and (user_changed or rinne_changed)
        and user_location
        and rinne_location
        and user_location != rinne_location
    ):
        snapshot["together"] = False

    # A reply may move only Rinne. If she reaches 用户's confirmed location,
    # that is enough to restore a shared scene without letting the model move him.
    if (
        source == "assistant"
        and rinne_changed
        and user_location
        and rinne_location == user_location
    ):
        snapshot["together"] = True

    if snapshot.get("together") is True and user_location == rinne_location:
        snapshot["world"] = user_location
    elif snapshot.get("together") is False:
        snapshot["world"] = None
        snapshot["activity"] = None
        for fields in (user, rinne):
            participant_activity = _text(fields.get("activity"))
            if participant_activity and (
                "一起" in participant_activity
                or "用户" in participant_activity
                or "凛祢" in participant_activity
            ):
                fields["activity"] = None

    movement_status = snapshot.get("movement_status")
    destination = _text(snapshot.get("destination"))
    if destination and movement_status not in {"planned", "in_transit"}:
        snapshot["destination"] = None
        snapshot["reason"] = None
        snapshot["next_step"] = None
        snapshot["plan_time"] = None
        destination = None
    if movement_status in {"planned", "in_transit"} and not destination:
        snapshot["movement_status"] = "unknown"
        snapshot["plan_time"] = None
    if destination is None and snapshot.get("movement_status") == "stationary":
        snapshot["plan_time"] = None


def _scene_delta_from_user_text(
    text: str,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract only explicit, high-confidence scene cues from user text."""

    raw_value = str(text or "")
    evidence = " ".join(raw_value.split())
    if not evidence:
        return {}

    delta: dict[str, Any] = {}

    def participant(name: str) -> dict[str, Any]:
        return delta.setdefault("participants", {}).setdefault(name, {})

    def set_joint_location(location: str, *, arrived: bool = False) -> None:
        participant("user").update(
            {"location": location, "activity": None, "next_step": None}
        )
        participant("rinne").update(
            {"location": location, "activity": None, "next_step": None}
        )
        delta.update(
            {
                "world": location,
                "together": True,
                "activity": None,
                "pending_request": None,
            }
        )
        if arrived or not delta.get("destination"):
            delta.update(
                {
                    "destination": None,
                    "next_step": None,
                    "movement_status": "arrived" if arrived else "stationary",
                    "plan_time": None,
                }
            )
        if not arrived and not delta.get("destination"):
            delta["reason"] = None

    def set_user_location(location: str, *, arrived: bool = False) -> None:
        participant("user").update(
            {"location": location, "activity": None, "next_step": None}
        )
        if arrived or not delta.get("destination"):
            delta.update(
                {
                    "destination": None,
                    "next_step": None,
                    "movement_status": "arrived" if arrived else "stationary",
                    "plan_time": None,
                }
            )
        if not arrived and not delta.get("destination"):
            delta["reason"] = None

    def set_journey(
        destination: str,
        *,
        joint: bool,
        status: str,
        plan_time: str | None,
    ) -> None:
        going_home = destination in {"家", "家里"}
        route = "回家途中" if going_home else f"去{destination}途中"
        user_activity = (
            "和凛祢一起回家"
            if joint and going_home
            else f"和凛祢一起去{destination}"
            if joint
            else "回家"
            if going_home
            else f"去{destination}"
        )
        delta.update(
            {
                "destination": destination,
                "movement_status": status,
                "plan_time": plan_time,
                "pending_request": None,
            }
        )
        if status == "planned":
            delta["next_step"] = (
                f"一起前往{destination}" if joint else f"用户前往{destination}"
            )
            participant("user")["next_step"] = f"前往{destination}"
            if joint:
                participant("rinne")["next_step"] = f"和用户一起前往{destination}"
            return

        participant("user").update(
            {
                "location": route,
                "activity": user_activity,
                "next_step": f"到达{destination}",
            }
        )
        if joint:
            participant("rinne").update(
                {
                    "location": route,
                    "activity": "和用户一起回家"
                    if going_home
                    else f"和用户一起去{destination}",
                    "next_step": f"到达{destination}",
                }
            )
            delta.update(
                {
                    "world": route,
                    "together": True,
                    "activity": "一起回家" if going_home else f"一起去{destination}",
                    "next_step": "一起回到家"
                    if going_home
                    else f"一起到达{destination}",
                }
            )
        else:
            delta.update(
                {
                    "world": None,
                    "together": False,
                    "activity": None,
                    "next_step": f"用户到达{destination}",
                }
            )

    # QQ batches contain independently authored messages. A joint subject in
    # one message must never leak into a later message.
    message_segments = [
        segment for segment in _QQ_MESSAGE_MARKER_RE.split(raw_value) if segment.strip()
    ] or [raw_value]
    batch_joint_active = False
    batch_user_active = False
    for message_segment in message_segments:
        joint_turn_active = batch_joint_active
        user_turn_active = batch_user_active
        explicit_reason: str | None = None
        for raw_clause in _CLAUSE_SPLIT_RE.split(message_segment):
            clause = " ".join(raw_clause.split()).strip()
            if not clause:
                continue
            if (
                _PAST_CLAUSE_RE.search(clause)
                or _HYPOTHETICAL_CLAUSE_RE.search(clause)
                or _QUESTION_CLAUSE_RE.search(clause)
                or _PROPOSAL_CLAUSE_RE.search(clause)
            ):
                continue

            reason_match = _REASON_RE.search(clause)
            if reason_match:
                explicit_reason = _text(reason_match.group("reason"), limit=80)

            joint_subject = bool(_JOINT_SUBJECT_RE.search(clause))
            if joint_subject:
                joint_turn_active = True
                batch_joint_active = True
            if re.match(r"^(?:我|用户)", clause):
                user_turn_active = True
                batch_user_active = True
            rinne_subject = bool(re.match(r"^(?:你|凛祢)", clause))
            explicit_subject = bool(
                joint_subject or rinne_subject or re.match(r"^(?:我|用户)", clause)
            )

            cancellation = bool(_CANCEL_JOURNEY_RE.search(clause))
            if cancellation:
                delta.update(
                    {
                        "destination": None,
                        "next_step": None,
                        "movement_status": "stationary",
                        "plan_time": None,
                        "reason": None,
                    }
                )
                participant("user").update({"activity": None, "next_step": None})

            if _JOINT_ARRIVED_HOME_RE.search(clause):
                set_joint_location("家", arrived=True)
                continue

            arrival_match = _ARRIVAL_RE.search(clause)
            if arrival_match:
                location = arrival_match.group("location")
                previous_together = bool((snapshot or {}).get("together"))
                if joint_subject or (not explicit_subject and previous_together):
                    set_joint_location(location, arrived=True)
                elif rinne_subject:
                    participant("rinne").update(
                        {"location": location, "activity": None, "next_step": None}
                    )
                else:
                    set_user_location(location, arrived=True)
                continue

            joint_location_match = _JOINT_LOCATION_RE.search(clause)
            rinne_location_match = _RINNE_LOCATION_RE.search(clause)
            user_location_match = _USER_LOCATION_RE.search(clause)
            implicit_user_location_match = (
                _IMPLICIT_USER_LOCATION_RE.search(clause) if user_turn_active else None
            )
            if joint_location_match:
                set_joint_location(joint_location_match.group("location"))
            elif rinne_location_match:
                participant("rinne").update(
                    {
                        "location": rinne_location_match.group("location"),
                        "activity": None,
                        "next_step": None,
                    }
                )
            elif user_location_match:
                set_user_location(user_location_match.group("location"))
            elif implicit_user_location_match:
                set_user_location(implicit_user_location_match.group("location"))

            if _JOINT_APART_RE.search(clause):
                delta.update({"world": None, "together": False, "activity": None})

            if _JOINT_DOWNHILL_RE.search(clause):
                participant("user").update(
                    {
                        "location": "下山途中",
                        "activity": "和凛祢一起下山",
                        "next_step": "到达山下",
                    }
                )
                participant("rinne").update(
                    {
                        "location": "下山途中",
                        "activity": "和用户一起下山",
                        "next_step": "到达山下",
                    }
                )
                delta.update(
                    {
                        "world": "下山途中",
                        "together": True,
                        "activity": "一起下山",
                        "destination": "山下",
                        "next_step": "一起到达山下",
                        "movement_status": "in_transit",
                        "plan_time": None,
                        "pending_request": None,
                    }
                )
                continue

            if _JOINT_RETURN_HOME_RE.search(clause):
                future_match = _FUTURE_CUE_RE.search(clause)
                set_journey(
                    "家",
                    joint=True,
                    status="planned" if future_match else "in_transit",
                    plan_time=future_match.group("when") if future_match else None,
                )
                continue

            journey_matches = []
            for match in _JOURNEY_RE.finditer(clause):
                prefix = clause[max(0, match.start() - 2) : match.start()]
                if re.search(r"(?:不|别)$", prefix):
                    continue
                journey_matches.append(match)
            journey_match = journey_matches[-1] if journey_matches else None
            if journey_match:
                destination = journey_match.group("destination")
                future_match = _FUTURE_CUE_RE.search(clause)
                status = (
                    "in_transit"
                    if _DEPARTURE_CUE_RE.search(clause)
                    else "planned"
                    if future_match
                    else "in_transit"
                )
                route_is_joint = joint_subject or (
                    joint_turn_active
                    and not re.match(
                        r"^(?:我|你|凛祢)(?:现在|正在|要|会|准备|打算)?",
                        clause,
                    )
                )
                if not rinne_subject:
                    set_journey(
                        destination,
                        joint=route_is_joint,
                        status=status,
                        plan_time=future_match.group("when") if future_match else None,
                    )

            if _MEMORY_REPAIR_RE.search(clause) or "修记忆系统" in clause:
                delta.update(
                    {
                        "activity": "用户准备处理记忆系统",
                        "destination": "记忆系统",
                        "next_step": "用户准备出门处理记忆系统",
                        "movement_status": "planned",
                    }
                )
            if _NOT_GOING_OUT_RE.search(clause):
                delta.update(
                    {
                        "destination": None,
                        "next_step": None,
                        "activity": "当前不是出门",
                        "movement_status": "stationary",
                        "plan_time": None,
                        "reason": None,
                    }
                )
            if _WAIT_FOR_RINNE_RE.search(clause):
                delta.update(
                    {
                        "world": None,
                        "together": False,
                        "activity": None,
                        "pending_request": "用户等凛祢过来",
                    }
                )
                participant("rinne").update(
                    {"location": None, "activity": None, "next_step": None}
                )
            elif _COME_REQUEST_RE.search(clause):
                delta["pending_request"] = "用户让凛祢过来"

        if explicit_reason and delta.get("destination"):
            delta["reason"] = explicit_reason
    if delta:
        delta["confidence"] = "explicit_user"
        delta["evidence"] = evidence[:_MAX_SOURCE_LENGTH]
    return delta


def scene_delta_from_user_text(
    text: str,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Public pure parser used by the store and regression tests."""

    return normalise_scene_delta(
        _scene_delta_from_user_text(text, snapshot), source="user"
    )


def parse_scene_envelope(text: str) -> dict[str, Any] | None:
    """Parse one hidden assistant scene envelope without accepting prose."""

    if not isinstance(text, str):
        return None
    start = text.find(SCENE_ENVELOPE_OPEN)
    if start < 0:
        return None
    body_start = start + len(SCENE_ENVELOPE_OPEN)
    end = text.find(SCENE_ENVELOPE_CLOSE, body_start)
    if end < 0:
        return None
    try:
        value = json.loads(text[body_start:end].strip())
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    delta = normalise_scene_delta(value, source="assistant")
    return delta or None


def strip_scene_envelopes(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Remove complete hidden envelopes before any user-facing transport."""

    if not isinstance(text, str) or not text:
        return text, []
    visible_parts: list[str] = []
    deltas: list[dict[str, Any]] = []
    cursor = 0
    while True:
        start = text.find(SCENE_ENVELOPE_OPEN, cursor)
        if start < 0:
            visible_parts.append(text[cursor:])
            break
        visible_parts.append(text[cursor:start])
        body_start = start + len(SCENE_ENVELOPE_OPEN)
        end = text.find(SCENE_ENVELOPE_CLOSE, body_start)
        if end < 0:
            # An incomplete hidden block is never allowed into history/UI.
            break
        delta = parse_scene_envelope(text[start : end + len(SCENE_ENVELOPE_CLOSE)])
        if delta is not None:
            deltas.append(delta)
        cursor = end + len(SCENE_ENVELOPE_CLOSE)
    return "".join(visible_parts).strip(), deltas


def correct_scene_conflicts(
    text: str,
    snapshot: dict[str, Any] | None,
) -> tuple[str, int]:
    """Correct only explicit separation claims that contradict a joint scene."""

    if not isinstance(text, str) or not text or not isinstance(snapshot, dict):
        return text, 0
    if snapshot.get("together") is not True:
        return text, 0
    participants = snapshot.get("participants") or {}
    user = participants.get("user") or {}
    rinne = participants.get("rinne") or {}
    user_location = _text(user.get("location"))
    rinne_location = _text(rinne.get("location"))
    if not user_location or user_location != rinne_location:
        return text, 0
    return _SEPARATE_HOME_WAIT_RE.subn("我会一直陪在你身边", text)


def _changed_fields(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    changed: dict[str, Any] = {}
    for key in _TOP_LEVEL_FIELDS | {"participants"}:
        if before.get(key) != after.get(key):
            changed[key] = after.get(key)
    return changed


@contextmanager
def _scene_lock(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / SCENE_LOCK_NAME
    started = time.monotonic()
    handle: int | None = None
    while handle is None:
        try:
            handle = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() - started >= _LOCK_TIMEOUT_SECONDS:
                raise TimeoutError(f"scene_state_lock_timeout:{lock_path}")
            time.sleep(_LOCK_POLL_SECONDS)
    try:
        os.write(handle, str(os.getpid()).encode("ascii", "replace"))
        yield
    finally:
        os.close(handle)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:10]}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


class SceneStateStore:
    """Read/update the one shared current scene for a character history root."""

    def __init__(self, history_root: str | Path) -> None:
        self.history_root = Path(history_root).resolve()
        self.directory = self.history_root / SCENE_DIRECTORY_NAME
        self.snapshot_path = self.directory / SCENE_SNAPSHOT_NAME
        self.change_log_path = self.directory / SCENE_CHANGE_LOG_NAME

    def load(self) -> dict[str, Any]:
        try:
            value = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            return empty_scene_snapshot()
        if not isinstance(value, dict):
            return empty_scene_snapshot()
        snapshot = empty_scene_snapshot(value.get("memory_day"))
        loaded_delta = normalise_scene_delta(value, source="user")
        _apply_delta(snapshot, loaded_delta)
        _reconcile_snapshot(snapshot, loaded_delta, source="load")
        snapshot["schema_version"] = SCENE_SCHEMA_VERSION
        snapshot["memory_day"] = value.get("memory_day")
        snapshot["updated_at"] = value.get("updated_at") or _now_iso()
        return snapshot

    def context_text(self, snapshot: dict[str, Any] | None = None) -> str:
        snapshot = snapshot or self.load()
        participants = snapshot.get("participants") or {}
        user = participants.get("user") or {}
        rinne = participants.get("rinne") or {}

        def label(value: Any, fallback: str = "未确认") -> str:
            return _text(value) or fallback

        lines = [SCENE_CONTEXT_HEADER]
        lines.append(f"记忆日：{label(snapshot.get('memory_day'))}")
        lines.append(
            f"用户：位置={label(user.get('location'))}；"
            f"活动={label(user.get('activity'))}；下一步={label(user.get('next_step'))}"
        )
        lines.append(
            f"凛祢：位置={label(rinne.get('location'))}；"
            f"活动={label(rinne.get('activity'))}；下一步={label(rinne.get('next_step'))}"
        )
        lines.append(
            f"共同空间={label(snapshot.get('world'))}；是否在一起={label(snapshot.get('together'))}；"
            f"当前共同活动={label(snapshot.get('activity'))}"
        )
        lines.append(
            f"所在原因/目的={label(snapshot.get('reason'))}；"
            f"用户目的地={label(snapshot.get('destination'))}；"
            f"待处理请求={label(snapshot.get('pending_request'))}"
        )
        lines.append(
            f"移动阶段={label(snapshot.get('movement_status'))}；"
            f"计划时间={label(snapshot.get('plan_time'))}；"
            f"共同下一步={label(snapshot.get('next_step'))}"
        )
        lines.append(
            "只把这里已经确认的场景当作当前事实；本轮用户的新说法优先于此快照，"
            "位置、目的地、原因、时间和动作必须分开理解；计划去某处不等于已经在途中。"
            "没有证据的地点、目的和动作不得猜测。"
        )
        return "\n".join(lines)

    def _append_change(
        self,
        *,
        memory_day: str,
        source: str,
        delta: dict[str, Any],
        snapshot: dict[str, Any],
        source_text: str | None = None,
        reset_log: bool = False,
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        if reset_log:
            self.change_log_path.write_text("", encoding="utf-8", newline="\n")
        record = {
            "schema_version": SCENE_SCHEMA_VERSION,
            "memory_day": memory_day,
            "timestamp": _now_iso(),
            "source": source,
            "changed": delta,
            "snapshot": snapshot,
        }
        if source_text:
            record["source_text"] = source_text[:_MAX_SOURCE_LENGTH]
        with self.change_log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _day_change_log_path(self, memory_day: str) -> Path:
        return self.directory / f"changes_{memory_day}.jsonl"

    def load_changes(self, memory_day: str) -> list[dict[str, Any]]:
        """Load only the temporary scene changes belonging to one memory day."""

        date.fromisoformat(memory_day)
        candidates = [
            self._day_change_log_path(memory_day),
            self.change_log_path,
        ]
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for path in candidates:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                continue
            for line in lines:
                try:
                    value = json.loads(line)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if not isinstance(value, dict) or value.get("memory_day") != memory_day:
                    continue
                marker = json.dumps(value, ensure_ascii=False, sort_keys=True)
                if marker not in seen:
                    seen.add(marker)
                    records.append(value)
        return records

    def discard_change_log(self, memory_day: str) -> None:
        """Remove one day's temporary detail after its diary is accepted."""

        date.fromisoformat(memory_day)
        with _scene_lock(self.directory):
            try:
                self._day_change_log_path(memory_day).unlink()
            except FileNotFoundError:
                pass
            if not self.change_log_path.is_file():
                return
            remaining: list[dict[str, Any]] = []
            try:
                lines = self.change_log_path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                lines = []
            for line in lines:
                try:
                    value = json.loads(line)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict) and value.get("memory_day") != memory_day:
                    remaining.append(value)
            if remaining:
                self.change_log_path.write_text(
                    "\n".join(
                        json.dumps(item, ensure_ascii=False) for item in remaining
                    )
                    + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
            else:
                try:
                    self.change_log_path.unlink()
                except FileNotFoundError:
                    pass

    def prepare_turn(
        self,
        user_text: str,
        *,
        timestamp: str | datetime | None = None,
        source_channel: str = "unknown",
    ) -> tuple[dict[str, Any], str]:
        """Merge explicit user scene facts and return the prompt snapshot."""

        memory_day = _memory_day_for(self.history_root, timestamp).isoformat()
        with _scene_lock(self.directory):
            previous = self.load()
            user_delta = scene_delta_from_user_text(user_text, previous)
            previous_day = previous.get("memory_day")
            snapshot = _copy_snapshot(previous)
            if previous_day != memory_day:
                # Keep an ongoing current snapshot, but start a new temporary
                # change log for this logical memory day.  Preserve the old
                # log under its date until the diary generator has consumed it.
                if isinstance(previous_day, str) and self.change_log_path.is_file():
                    old_log = self._day_change_log_path(previous_day)
                    if not old_log.exists():
                        os.replace(self.change_log_path, old_log)
                    else:
                        self.change_log_path.write_text(
                            "", encoding="utf-8", newline="\n"
                        )
                snapshot["memory_day"] = memory_day
            if user_delta:
                _apply_delta(snapshot, user_delta)
                _reconcile_snapshot(snapshot, user_delta, source="user")
            snapshot["memory_day"] = memory_day
            snapshot["updated_at"] = _now_iso()
            if user_delta or previous_day != memory_day:
                _write_json_atomic(self.snapshot_path, snapshot)
            if previous_day != memory_day and not user_delta:
                self.change_log_path.write_text("", encoding="utf-8", newline="\n")
            if user_delta:
                self._append_change(
                    memory_day=memory_day,
                    source=f"user_explicit:{source_channel}",
                    delta=_changed_fields(previous, snapshot),
                    snapshot=snapshot,
                    source_text=user_text,
                    reset_log=False,
                )
        return snapshot, self.context_text(snapshot)

    def apply_assistant_delta(
        self,
        delta: dict[str, Any],
        *,
        memory_day: str,
        source_channel: str = "unknown",
    ) -> dict[str, Any]:
        """Apply validated hidden assistant scene state after the reply."""

        normalized = normalise_scene_delta(delta, source="assistant")
        if not normalized:
            return self.load()
        with _scene_lock(self.directory):
            previous = self.load()
            if previous.get("memory_day") not in {None, memory_day}:
                # A late reply from yesterday must not overwrite today's scene.
                return previous
            snapshot = _copy_snapshot(previous)
            snapshot["memory_day"] = memory_day
            _apply_delta(snapshot, normalized)
            _reconcile_snapshot(snapshot, normalized, source="assistant")
            snapshot["confidence"] = "assistant_structured"
            snapshot["updated_at"] = _now_iso()
            _write_json_atomic(self.snapshot_path, snapshot)
            self._append_change(
                memory_day=memory_day,
                source=f"assistant_structured:{source_channel}",
                delta=_changed_fields(previous, snapshot),
                snapshot=snapshot,
            )
            return snapshot


__all__ = [
    "SCENE_CONTEXT_HEADER",
    "SCENE_ENVELOPE_CLOSE",
    "SCENE_ENVELOPE_OPEN",
    "SCENE_SCHEMA_VERSION",
    "SceneStateStore",
    "correct_scene_conflicts",
    "empty_scene_snapshot",
    "normalise_scene_delta",
    "parse_scene_envelope",
    "scene_delta_from_user_text",
    "strip_scene_envelopes",
]
