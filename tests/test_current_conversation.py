import json
import unittest
from datetime import datetime, timedelta, timezone

from src.open_llm_vtuber.memory import (
    ConversationTurn,
    CurrentConversationState,
    LongTermMemoryItem,
    LongTermMemoryResult,
    LongTermMemoryResultStatus,
    MemoryDecisionAction,
    MemoryFact,
    MemoryItemStatus,
    MemoryOrchestrator,
    MemorySource,
    TodayHistoryDiagnostics,
    TodayHistoryLoadResult,
)


LOCAL_TZ = timezone(timedelta(hours=8))
BASE_TIME = datetime(2026, 7, 19, 9, 0, tzinfo=LOCAL_TZ)


def make_turn(
    turn_id: str,
    minute: int,
    *,
    content: str | None = None,
    origin: str = "current",
) -> ConversationTurn:
    return ConversationTurn(
        turn_id=turn_id,
        role="user" if minute % 2 == 0 else "assistant",
        content=content or f"turn-{minute}",
        timestamp=BASE_TIME + timedelta(minutes=minute),
        source_ref=f"{origin}:{turn_id}",
        metadata={"origin": origin},
    )


class CurrentConversationStateTests(unittest.TestCase):
    def test_add_sort_deduplicate_and_replace(self):
        state = CurrentConversationState(session_id="session")
        later = make_turn("later", 2)
        earlier = make_turn("earlier", 0)

        self.assertTrue(state.add_turn(later))
        self.assertTrue(state.add_turn(earlier))
        self.assertEqual(
            [turn.turn_id for turn in state.recent_turns],
            ["earlier", "later"],
        )

        duplicate = make_turn(
            "duplicate-id",
            2,
            content=later.content,
            origin="today_history",
        )
        self.assertFalse(state.add_turn(duplicate))
        self.assertEqual(len(state.recent_turns), 2)

        preferred = make_turn(
            "current-version",
            2,
            content=later.content,
            origin="current",
        )
        self.assertFalse(state.add_turn(preferred, replace_existing=True))
        self.assertEqual(state.recent_turns[-1].turn_id, "current-version")
        self.assertEqual(state.recent_turns[-1].metadata["origin"], "current")

    def test_turn_count_trim_keeps_newest_turns(self):
        state = CurrentConversationState(session_id="session", max_turns=2)
        state.add_turns(
            [make_turn("three", 3), make_turn("one", 1), make_turn("two", 2)]
        )

        self.assertEqual(
            [turn.turn_id for turn in state.recent_turns],
            ["two", "three"],
        )

    def test_custom_budget_policy_can_replace_turn_count_policy(self):
        class KeepFirstAndLast:
            def apply(self, turns):
                if len(turns) <= 1:
                    return list(turns)
                return [turns[0], turns[-1]]

        state = CurrentConversationState(session_id="session")
        state.add_turns(
            [make_turn("one", 1), make_turn("two", 2), make_turn("three", 3)]
        )

        removed = state.trim(KeepFirstAndLast())

        self.assertEqual(removed, 1)
        self.assertEqual(
            [turn.turn_id for turn in state.recent_turns],
            ["one", "three"],
        )

    def test_clear_removes_all_session_scoped_state(self):
        state = CurrentConversationState(
            session_id="session",
            recent_turns=[make_turn("one", 1)],
            current_topics=["topic"],
            source_refs=["source"],
        )

        state.clear()

        self.assertEqual(state.recent_turns, [])
        self.assertEqual(state.current_topics, [])
        self.assertEqual(state.nearby_references, [])
        self.assertEqual(state.new_facts, [])
        self.assertEqual(state.source_refs, [])

    def test_state_round_trips_through_dict_and_json(self):
        state = CurrentConversationState(
            session_id="session",
            recent_turns=[make_turn("one", 1)],
            current_topics=["topic"],
            new_facts=[
                MemoryFact(
                    fact_id="fact",
                    key="project.stage",
                    value="prototype",
                    source=MemorySource.CURRENT_CONVERSATION,
                    observed_at=BASE_TIME,
                )
            ],
            max_turns=10,
            created_at=BASE_TIME,
            updated_at=BASE_TIME,
        )

        serialized = state.to_dict()
        json.dumps(serialized, ensure_ascii=False)
        restored = CurrentConversationState.from_dict(serialized)

        self.assertEqual(restored.to_dict(), serialized)
        self.assertIsNot(restored, state)
        self.assertIsNot(restored.recent_turns[0], state.recent_turns[0])


class TimestampAndResultValidationTests(unittest.TestCase):
    def test_naive_timestamps_are_rejected(self):
        naive = datetime(2026, 7, 19, 12, 0)

        with self.assertRaises(ValueError):
            ConversationTurn(
                turn_id="turn",
                role="user",
                content="content",
                timestamp=naive,
            )
        with self.assertRaises(ValueError):
            MemoryFact(
                fact_id="fact",
                key="key",
                value="value",
                source=MemorySource.CURRENT_CONVERSATION,
                observed_at=naive,
            )
        with self.assertRaises(ValueError):
            LongTermMemoryItem(
                memory_id="memory",
                content="content",
                source_file="diary.txt",
                memory_time=naive,
            )

    def test_confirmed_result_rejects_uncertain_item(self):
        uncertain = LongTermMemoryItem(
            memory_id="uncertain",
            content="content",
            source_file="diary.txt",
            memory_time=BASE_TIME,
            confidence=0.7,
            status=MemoryItemStatus.UNCERTAIN,
        )

        with self.assertRaises(ValueError):
            LongTermMemoryResult(
                status=LongTermMemoryResultStatus.CONFIRMED,
                confirmed_memories=[uncertain],
            )

    def test_ambiguous_and_no_match_reject_cross_state_items(self):
        confirmed = LongTermMemoryItem(
            memory_id="confirmed",
            content="content",
            source_file="diary.txt",
            memory_time=BASE_TIME,
            confidence=0.9,
            status=MemoryItemStatus.ACTIVE,
        )

        with self.assertRaises(ValueError):
            LongTermMemoryResult(
                status=LongTermMemoryResultStatus.AMBIGUOUS,
                confirmed_memories=[confirmed],
                unconfirmed_candidates=[confirmed],
            )
        with self.assertRaises(ValueError):
            LongTermMemoryResult(
                status=LongTermMemoryResultStatus.NO_MATCH,
                unconfirmed_candidates=[confirmed],
            )


class Stage2OrchestratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_current_turn_wins_when_merging_with_today_history(self):
        duplicate_time = BASE_TIME + timedelta(minutes=2)
        historical_duplicate = ConversationTurn(
            turn_id="history-copy",
            role="user",
            content="same turn",
            timestamp=duplicate_time,
            metadata={"origin": "today_history"},
        )
        current_duplicate = ConversationTurn(
            turn_id="current-copy",
            role="user",
            content="same turn",
            timestamp=duplicate_time,
            metadata={"origin": "current"},
        )
        history_result = TodayHistoryLoadResult(
            memory_day=BASE_TIME.date(),
            window_start=BASE_TIME.replace(hour=3),
            window_end=BASE_TIME.replace(hour=3) + timedelta(days=1),
            turns=[make_turn("history-only", 1), historical_duplicate],
            diagnostics=TodayHistoryDiagnostics(returned_turns=2),
        )
        current = CurrentConversationState(
            session_id="session",
            recent_turns=[current_duplicate, make_turn("current-only", 3)],
        )
        original_current = current.to_dict()

        package = await MemoryOrchestrator().prepare_turn(
            "normal message",
            current_conversation=current,
            today_history=history_result,
        )

        self.assertEqual(
            package.decision.action,
            MemoryDecisionAction.CONTEXT_ONLY,
        )
        self.assertEqual(len(package.current_conversation.recent_turns), 3)
        merged_duplicate = next(
            turn
            for turn in package.current_conversation.recent_turns
            if turn.content == "same turn"
        )
        self.assertEqual(merged_duplicate.turn_id, "current-copy")
        self.assertEqual(merged_duplicate.metadata["origin"], "current")
        self.assertEqual(package.dynamic_system_context, "")
        self.assertEqual(package.llm_messages, [])
        self.assertFalse(
            package.internal_diagnostics["long_term_retrieval_called"]
        )
        self.assertEqual(current.to_dict(), original_current)


if __name__ == "__main__":
    unittest.main()
