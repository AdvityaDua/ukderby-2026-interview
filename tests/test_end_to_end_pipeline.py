import asyncio
import json
import unittest

import app.api.v1.interview_engine as ie
from app.services.session_state_contract import build_persisted_session_record


class FakeGraph:
    def __init__(self):
        self.turn = 0

    async def run_turn(self, state):
        self.turn += 1
        state = dict(state)
        state["turn_number"] = state.get("turn_number", 0) + 1
        state["input_tokens"] = state.get("input_tokens", 0) + (140 if self.turn == 1 else 96)
        state["output_tokens"] = state.get("output_tokens", 0) + (45 if self.turn == 1 else 52)
        if self.turn == 1:
            state["current_question"] = "How would you design a Redis-backed session store for reconnect-safe interviews?"
            state["current_evaluation"] = {"next_step": {"is_coding_question": False}}
            state["ended"] = False
            state["end_reason"] = ""
        else:
            state["current_question"] = "Implement a function to safely serialize interview session state to JSON."
            state["current_evaluation"] = {"next_step": {"is_coding_question": True}}
            state["ended"] = False
            state["end_reason"] = ""
        return state


class InMemorySessionStore:
    def __init__(self):
        self.records = {}
        self.locks = set()

    async def save_agent_state(self, *, user_id, client_id, state, ttl_seconds=None, accounting_source="graph_state_cumulative", telemetry=None):
        previous = self.records.get(state["session_id"], {})
        prev_tokens = previous.get("token_summary", {})
        record = build_persisted_session_record(
            session_id=state["session_id"],
            user_id=user_id,
            client_id=client_id,
            interview_state=state["interview_state"],
            agent_meta={
                "current_node": state["current_node"],
                "next_action": state["next_action"],
                "tool_calls_count": state["tool_calls_count"],
                "created_at": state["created_at"].isoformat(),
                "last_updated_at": state["last_updated_at"].isoformat(),
            },
            ttl_seconds=ttl_seconds or 7200,
            previous_token_summary=prev_tokens,
            accounting_source=accounting_source,
            telemetry=telemetry,
        )
        self.records[state["session_id"]] = record
        return record

    async def load_record(self, session_id):
        return self.records.get(session_id)

    async def acquire_turn_lock(self, *, session_id, owner, lock_ttl_seconds=30):
        if session_id in self.locks:
            return False
        self.locks.add(session_id)
        return True

    async def release_turn_lock(self, *, session_id, owner):
        self.locks.discard(session_id)

    async def mark_ended(self, session_id):
        record = self.records.get(session_id)
        if not record:
            return None
        record["status"] = "ended"
        self.records[session_id] = record
        return record

    async def clear_session(self, *, session_id, user_id=None):
        self.records.pop(session_id, None)
        self.locks.discard(session_id)


class EndToEndPipelineTest(unittest.TestCase):
    def test_pipeline_init_turn_end(self):
        async def _run():
            fake_store = InMemorySessionStore()
            fake_graph = FakeGraph()
            ie.session_state_store = fake_store
            ie._ensure_model_ready = lambda: fake_graph

            user = {"sub": "user-123"}
            init_payload = ie.InitInterviewRequest(
                resume_text="Built Python FastAPI services with Redis and PostgreSQL. Led backend APIs and session persistence.",
                jd_text="Need Python, FastAPI, Redis, PostgreSQL, and system design.",
                company="Acme Systems",
                role="Backend Engineer",
                interview_type="technical",
                duration_minutes=30,
                client_id="client-abc",
                session_id="sess-demo-001",
            )
            init_resp = await ie.init_interview_engine(init_payload, user=user)

            turn_payload_1 = ie.RunTurnRequest(
                session_id="sess-demo-001",
                user_input="I would keep session state in Redis with TTLs and restore by session_id.",
                client_id="client-abc",
            )
            turn_resp_1 = await ie.run_interview_turn(turn_payload_1, user=user)

            turn_payload_2 = ie.RunTurnRequest(
                session_id="sess-demo-001",
                user_input="I can implement that serialization helper now.",
                client_id="client-abc",
            )
            turn_resp_2 = await ie.run_interview_turn(turn_payload_2, user=user)

            persisted = await fake_store.load_record("sess-demo-001")
            return {
                "init": init_resp,
                "turn1": turn_resp_1,
                "turn2": turn_resp_2,
                "persisted": persisted,
            }

        result = asyncio.run(_run())
        print("=== INIT RESPONSE ===")
        print(json.dumps(result["init"], indent=2, sort_keys=True))
        print("=== TURN 1 RESPONSE ===")
        print(json.dumps(result["turn1"], indent=2, sort_keys=True))
        print("=== TURN 2 RESPONSE ===")
        print(json.dumps(result["turn2"], indent=2, sort_keys=True))
        print("=== PERSISTED SESSION SUMMARY ===")
        print(json.dumps({
            "session_id": result["persisted"]["session_id"],
            "status": result["persisted"]["status"],
            "turn_number": result["persisted"]["interview_state"]["turn_number"],
            "current_question": result["persisted"]["interview_state"]["current_question"],
            "token_summary": result["persisted"]["token_summary"],
            "telemetry": result["persisted"]["telemetry"],
            "coding_flag": result["persisted"]["interview_state"]["last_question_was_coding"],
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
