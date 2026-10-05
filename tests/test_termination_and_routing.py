import unittest

from app.services.ai.response_builder import build_turn_response
from app.services.session_state_contract import build_persisted_session_record, mark_record_status


class TestTerminationAndRouting(unittest.TestCase):
    def test_build_turn_response_coding_ux_from_object(self):
        class DummyNextStep:
            def __init__(self, is_coding_question: bool):
                self.is_coding_question = is_coding_question

        class DummyEval:
            def __init__(self, is_coding_question: bool):
                self.next_step = DummyNextStep(is_coding_question)

        state = {
            "current_question": "Implement a queue using two stacks.",
            "ended": False,
            "end_reason": "",
            "turn_number": 4,
            "input_tokens": 10,
            "output_tokens": 5,
            "current_evaluation": DummyEval(True),
        }

        response = build_turn_response("sess-coding", state)

        self.assertTrue(response["is_coding_question"])
        self.assertTrue(response["coding_ux"]["open_editor"])
        self.assertEqual(response["coding_ux"]["question_mode"], "coding")
        self.assertTrue(response["coding_ux"]["allow_code_submission"])

    def test_build_turn_response_termination_flags(self):
        state = {
            "current_question": "Thanks, we can close here.",
            "ended": True,
            "end_reason": "BUDGET_EXHAUSTED",
            "turn_number": 9,
            "input_tokens": 300,
            "output_tokens": 120,
            "current_evaluation": {"next_step": {"is_coding_question": False}},
        }

        response = build_turn_response("sess-1", state)

        self.assertTrue(response["ended"])
        self.assertEqual(response["end_reason"], "BUDGET_EXHAUSTED")
        self.assertEqual(response["turn_number"], 9)
        self.assertEqual(response["tokens"]["total_tokens"], 420)

    def test_build_turn_response_routing_coding_flag(self):
        state = {
            "current_question": "Implement LRU cache in Python.",
            "ended": False,
            "end_reason": "",
            "turn_number": 3,
            "input_tokens": 90,
            "output_tokens": 40,
            "current_evaluation": {"next_step": {"is_coding_question": True}},
        }
        token_summary = {
            "last_turn_input_tokens": 12,
            "last_turn_output_tokens": 7,
            "last_turn_total_tokens": 19,
            "accounting_source": "graph_turn_run_turn",
        }

        telemetry = {"stage": "turn", "total_ms": 123, "graph_ms": 97}
        response = build_turn_response("sess-2", state, token_summary=token_summary, telemetry=telemetry)

        self.assertFalse(response["ended"])
        self.assertTrue(response["is_coding_question"])
        self.assertEqual(response["tokens"]["last_turn_total_tokens"], 19)
        self.assertEqual(response["tokens"]["accounting_source"], "graph_turn_run_turn")
        self.assertEqual(response["telemetry"]["total_ms"], 123)
        self.assertEqual(response["telemetry"]["stage"], "turn")

    def test_persisted_record_status_transition(self):
        record = build_persisted_session_record(
            session_id="sess-3",
            user_id="user-1",
            client_id="client-1",
            interview_state={"input_tokens": 100, "output_tokens": 30},
            agent_meta={"current_node": "RUN_TURN", "next_action": "continue", "tool_calls_count": 0},
            ttl_seconds=7200,
            status="active",
            previous_token_summary={"input_tokens": 80, "output_tokens": 20},
            accounting_source="graph_state_cumulative",
        )

        self.assertEqual(record["token_summary"]["last_turn_input_tokens"], 20)
        self.assertEqual(record["token_summary"]["last_turn_output_tokens"], 10)
        self.assertEqual(record["status"], "active")

        ended = mark_record_status(record, "ended")
        self.assertEqual(ended["status"], "ended")


if __name__ == "__main__":
    unittest.main()
