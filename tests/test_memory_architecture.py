import json
import unittest
from datetime import datetime, timedelta, timezone

from src.open_llm_vtuber.memory import (
    DEFAULT_MEMORY_BUDGETS,
    ConversationTurn,
    CurrentConversationState,
    LongTermMemoryItem,
    LongTermMemoryResult,
    LongTermMemoryResultStatus,
    MemoryContextPackage,
    MemoryDecision,
    MemoryDecisionAction,
    MemoryFact,
    MemoryItemStatus,
    MemoryOrchestrator,
    MemoryPrecedence,
    MemorySource,
    ModelScope,
    NearbyReference,
    RuntimeStateOverride,
    RuntimeStateOverrides,
    UserBackgroundState,
    compare_precedence,
    precedence_rank,
)


FIXED_TIME = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)


def make_long_term_item(
    memory_id: str,
    *,
    status: MemoryItemStatus = MemoryItemStatus.ACTIVE,
) -> LongTermMemoryItem:
    return LongTermMemoryItem(
        memory_id=memory_id,
        content="A concrete historical memory.",
        source_file="diary_2026-01-01.txt",
        memory_time=FIXED_TIME,
        confidence=0.8,
        status=status,
    )


class MemoryDecisionTests(unittest.TestCase):
    def test_decision_enum_contains_every_architecture_action(self):
        self.assertEqual(
            {action.value for action in MemoryDecisionAction},
            {
                "context_only",
                "retrieve_long_term",
                "clarify",
                "bypass",
            },
        )


class LongTermMemoryResultTests(unittest.TestCase):
    def test_confirmed_and_ambiguous_items_use_separate_fields(self):
        confirmed = make_long_term_item("confirmed")
        candidate = make_long_term_item(
            "candidate", status=MemoryItemStatus.UNCERTAIN
        )

        confirmed_result = LongTermMemoryResult(
            status=LongTermMemoryResultStatus.CONFIRMED,
            confirmed_memories=[confirmed],
        )
        ambiguous_result = LongTermMemoryResult(
            status=LongTermMemoryResultStatus.AMBIGUOUS,
            unconfirmed_candidates=[candidate],
        )

        self.assertEqual(confirmed_result.confirmed_memories, [confirmed])
        self.assertEqual(confirmed_result.unconfirmed_candidates, [])
        self.assertEqual(ambiguous_result.confirmed_memories, [])
        self.assertEqual(ambiguous_result.unconfirmed_candidates, [candidate])

    def test_mixed_result_is_rejected(self):
        with self.assertRaises(ValueError):
            LongTermMemoryResult(
                status=LongTermMemoryResultStatus.CONFIRMED,
                confirmed_memories=[make_long_term_item("confirmed")],
                unconfirmed_candidates=[make_long_term_item("candidate")],
            )

    def test_no_match_is_explicit_and_contains_no_items(self):
        result = LongTermMemoryResult(no_match_reason="No evidence")

        self.assertTrue(result.no_match)
        self.assertEqual(result.confirmed_memories, [])
        self.assertEqual(result.unconfirmed_candidates, [])
        self.assertTrue(result.to_dict()["no_match"])


class PrecedenceTests(unittest.TestCase):
    def test_architecture_priority_order_is_encoded(self):
        self.assertGreater(
            precedence_rank(MemorySource.CURRENT_CONVERSATION),
            precedence_rank(MemorySource.RUNTIME_OVERRIDE),
        )
        self.assertGreater(
            precedence_rank(MemorySource.RUNTIME_OVERRIDE),
            precedence_rank(MemorySource.USER_BACKGROUND),
        )
        self.assertGreater(
            precedence_rank(MemorySource.USER_BACKGROUND),
            precedence_rank(MemorySource.LONG_TERM_MEMORY),
        )
        self.assertGreater(
            precedence_rank(MemorySource.LONG_TERM_MEMORY),
            precedence_rank(
                MemorySource.LONG_TERM_MEMORY,
                MemoryItemStatus.HISTORICAL,
            ),
        )
        self.assertEqual(
            precedence_rank(
                MemorySource.CURRENT_CONVERSATION,
                MemoryItemStatus.HISTORICAL,
            ),
            MemoryPrecedence.STALE_OR_HISTORICAL,
        )

    def test_newer_observation_breaks_same_level_tie(self):
        result = compare_precedence(
            MemorySource.RUNTIME_OVERRIDE,
            MemorySource.RUNTIME_OVERRIDE,
            left_observed_at=FIXED_TIME,
            right_observed_at=FIXED_TIME - timedelta(minutes=5),
        )

        self.assertEqual(result, 1)


class NoOpOrchestratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_turn_defaults_to_context_only_without_injection(self):
        conversation = CurrentConversationState(
            session_id="session-1",
            recent_turns=[
                ConversationTurn(
                    turn_id="turn-1",
                    role="user",
                    content="Current conversation text",
                    timestamp=FIXED_TIME,
                )
            ],
        )
        original = conversation.to_dict()
        package = await MemoryOrchestrator().prepare_turn(
            "A normal user message",
            current_conversation=conversation,
        )

        self.assertEqual(
            package.decision.action,
            MemoryDecisionAction.CONTEXT_ONLY,
        )
        self.assertFalse(package.decision.should_retrieve_long_term)
        self.assertEqual(package.long_term_result.retrieval_health, "not_run")
        self.assertEqual(package.dynamic_system_context, "")
        self.assertEqual(package.llm_messages, [])
        self.assertFalse(
            package.internal_diagnostics["long_term_retrieval_called"]
        )
        self.assertEqual(conversation.to_dict(), original)

    async def test_commit_turn_is_a_no_op(self):
        package = MemoryContextPackage()
        original = package.to_dict()

        result = await MemoryOrchestrator().commit_turn(
            package,
            assistant_response="No state should be changed.",
        )

        self.assertIsNone(result)
        self.assertEqual(package.to_dict(), original)


class SerializationTests(unittest.TestCase):
    def test_all_stage1_structures_convert_to_json_compatible_dicts(self):
        current_fact = MemoryFact(
            fact_id="fact-current",
            key="project.phase",
            value="architecture",
            source=MemorySource.CURRENT_CONVERSATION,
            observed_at=FIXED_TIME,
        )
        background_fact = MemoryFact(
            fact_id="fact-background",
            key="project.phase",
            value="planning",
            source=MemorySource.USER_BACKGROUND,
            observed_at=FIXED_TIME - timedelta(days=1),
        )
        conversation = CurrentConversationState(
            session_id="session-1",
            recent_turns=[
                ConversationTurn(
                    turn_id="turn-1",
                    role="user",
                    content="Start stage one.",
                    timestamp=FIXED_TIME,
                    source_ref="history.json#turn-1",
                )
            ],
            current_topics=["three-layer memory"],
            nearby_references=[
                NearbyReference(
                    expression="it",
                    resolved_to="three-layer memory",
                    source_turn_id="turn-1",
                    observed_at=FIXED_TIME,
                    confidence=0.9,
                )
            ],
            new_facts=[current_fact],
        )
        background = UserBackgroundState(
            user_id="rinne_01",
            version="v1",
            generated_at=FIXED_TIME,
            sections={"project": {"phase": "planning"}},
            facts=[background_fact],
            source_refs=["background_snapshot.json"],
        )
        runtime = RuntimeStateOverrides(
            run_id="run-1",
            user_id="rinne_01",
            items=[
                RuntimeStateOverride(
                    override_id="override-1",
                    fact=MemoryFact(
                        fact_id="fact-runtime",
                        key="project.phase",
                        value="architecture",
                        source=MemorySource.RUNTIME_OVERRIDE,
                        observed_at=FIXED_TIME,
                    ),
                    overrides_background_fact_ids=["fact-background"],
                    source_message_id="turn-1",
                    updated_at=FIXED_TIME,
                )
            ],
            started_at=FIXED_TIME,
            updated_at=FIXED_TIME,
        )
        long_term = LongTermMemoryResult(
            status=LongTermMemoryResultStatus.AMBIGUOUS,
            unconfirmed_candidates=[
                make_long_term_item(
                    "candidate", status=MemoryItemStatus.UNCERTAIN
                )
            ],
        )
        package = MemoryContextPackage(
            decision=MemoryDecision(
                action=MemoryDecisionAction.CLARIFY,
                missing_clues=["time"],
            ),
            current_conversation=conversation,
            user_background=background,
            runtime_overrides=runtime,
            long_term_result=long_term,
        )

        models = [
            current_fact,
            conversation,
            background,
            runtime,
            long_term,
            package,
            DEFAULT_MEMORY_BUDGETS,
        ]
        for model in models:
            serialized = model.to_dict()
            self.assertIsInstance(serialized, dict)
            json.dumps(serialized, ensure_ascii=False)


class BudgetTests(unittest.TestCase):
    def test_local_and_cloud_profiles_exist_without_final_token_values(self):
        local = DEFAULT_MEMORY_BUDGETS.for_scope(ModelScope.LOCAL)
        cloud = DEFAULT_MEMORY_BUDGETS.for_scope(ModelScope.CLOUD)

        self.assertEqual(local.model_scope, ModelScope.LOCAL)
        self.assertEqual(cloud.model_scope, ModelScope.CLOUD)
        self.assertIsNone(local.current_conversation_tokens)
        self.assertIsNone(local.background_state_tokens)
        self.assertIsNone(local.long_term_memory_tokens)
        self.assertIsNone(cloud.current_conversation_tokens)
        self.assertIsNone(cloud.background_state_tokens)
        self.assertIsNone(cloud.long_term_memory_tokens)


if __name__ == "__main__":
    unittest.main()
