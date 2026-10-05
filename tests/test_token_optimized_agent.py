import asyncio
import json
import unittest
from types import SimpleNamespace

from app.services.ai.token_optimized_agent import InterviewGraph


class FakeResponse:
    def __init__(self, payload: dict, prompt_tokens: int, output_tokens: int):
        self.text = json.dumps(payload)
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=output_tokens,
        )


class FakeModels:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self._responses.pop(0)


class FakeGeminiClient:
    def __init__(self, responses):
        self.api_key = "fake"
        self.model_name = "fake-model"
        self.client = SimpleNamespace(models=FakeModels(responses))


class TokenOptimizedAgentTest(unittest.TestCase):
    def test_run_turn_updates_state_and_tokens(self):
        first_payload = {
            "performance_summary": "Candidate started with a solid architecture answer.",
            "answer_type": "not_applicable",
            "answer_quality": "not_applicable",
            "should_follow_up": False,
            "follow_up_hint": "",
            "newly_covered_skills": ["Redis"],
            "last_answer_evaluation": None,
            "decision": {
                "action": "continue",
                "reason": "Opening question generated.",
                "termination_flag": False,
            },
            "next_step": {
                "type": "new_topic",
                "difficulty": "medium",
                "question": "How would you design a Redis-backed session store for reconnect-safe interviews?",
                "target_skill": "Redis",
                "is_coding_question": False,
            },
            "confidence_in_candidate": "medium",
        }

        second_payload = {
            "performance_summary": "The candidate explained state storage well and can now implement serialization.",
            "answer_type": "genuine_answer",
            "answer_quality": "adequate",
            "should_follow_up": False,
            "follow_up_hint": "",
            "newly_covered_skills": ["JSON serialization"],
            "last_answer_evaluation": {
                "score": 7,
                "signal": "strong",
                "clarity": "medium",
                "depth": "medium",
                "correctness": "partially_correct",
                "key_gaps": [],
                "positive_signals": ["understood state persistence"],
            },
            "decision": {
                "action": "continue",
                "reason": "Candidate is ready for a coding task.",
                "termination_flag": False,
            },
            "next_step": {
                "type": "follow_up",
                "difficulty": "medium",
                "question": "Implement a function to safely serialize interview session state to JSON.",
                "target_skill": "JSON serialization",
                "is_coding_question": True,
            },
            "confidence_in_candidate": "high",
        }

        fake_client = FakeGeminiClient([
            FakeResponse(first_payload, prompt_tokens=120, output_tokens=40),
            FakeResponse(second_payload, prompt_tokens=88, output_tokens=42),
        ])
        agent = InterviewGraph(client=fake_client)

        state = {
            "history": [],
            "performance_summary": "Interview initialized.",
            "context_summary": "Role: Backend Engineer\nCompany: Acme\nJD: Redis + Python",
            "has_jd": True,
            "skills_remaining": ["Redis", "JSON serialization"],
            "skills_covered": [],
            "questions_asked": [],
            "last_user_input": None,
            "current_question": "",
            "current_evaluation": None,
            "follow_up_hint": "",
            "interview_type": "technical",
            "time_context": "Target duration: 30 minutes",
            "role": "Backend Engineer",
            "company": "Acme",
            "is_developer_role": True,
            "coding_questions_asked": 0,
            "last_question_was_coding": False,
            "turn_number": 0,
            "last_answer_type": "not_applicable",
            "consecutive_non_answers": 0,
            "ended": False,
            "end_reason": "",
            "max_questions": 13,
            "max_questions_per_topic": 2,
            "topic_question_counts": {},
            "max_confusion_retries": 2,
            "consecutive_disengaged": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        async def _run():
            first = await agent.run_turn(state)
            second_state = dict(first)
            second_state["last_user_input"] = "I would store the session in Redis with TTL."
            second = await agent.run_turn(second_state)
            return first, second, fake_client.client.models.calls

        first, second, calls = asyncio.run(_run())

        self.assertEqual(len(calls), 2)
        self.assertEqual(first["current_question"], "How would you design a Redis-backed session store for reconnect-safe interviews?")
        self.assertFalse(first["last_question_was_coding"])
        self.assertEqual(first["input_tokens"], 120)
        self.assertEqual(first["output_tokens"], 40)
        self.assertEqual(first["questions_asked"][-1], "How would you design a Redis-backed session store for reconnect-safe interviews?")

        self.assertTrue(second["last_question_was_coding"])
        self.assertEqual(second["current_question"], "Implement a function to safely serialize interview session state to JSON.")
        self.assertEqual(second["coding_questions_asked"], 1)
        self.assertEqual(second["skills_covered"], ["Redis", "JSON serialization"])
        self.assertEqual(second["skills_remaining"], [])
        self.assertEqual(second["input_tokens"], 208)
        self.assertEqual(second["output_tokens"], 82)
        self.assertEqual(second["current_evaluation"]["next_step"]["is_coding_question"], True)
        self.assertEqual(second["current_evaluation"]["decision"]["reason"], "Candidate is ready for a coding task.")


if __name__ == "__main__":
    unittest.main()
