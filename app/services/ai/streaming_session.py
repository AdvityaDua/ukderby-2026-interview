import time

import asyncio
import base64
import time
from gradio_client import Client

# Initialize Gradio Client pointing to local server
try:
    tts_client = Client("http://127.0.0.1:7860/")
except:
    tts_client = None

def generate_tts_sync(text_chunk: str):
    if not tts_client or not text_chunk.strip(): return None
    try:
        # gradio API returns a tuple or filepath
        res = tts_client.predict(
            text=text_chunk,
            seed=0,
            temperature=0.8,
            top_p=0.95,
            top_k=1000,
            repetition_penalty=1.2,
            min_p=0,
            api_name="/predict"
        )
        audio_filepath = res[0] if isinstance(res, tuple) else res
        with open(audio_filepath, "rb") as af:
            return base64.b64encode(af.read()).decode("utf-8")
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

async def generate_tts_async(text_chunk: str):
    return await asyncio.to_thread(generate_tts_sync, text_chunk)
import httpx
import random
from typing import List, Dict, Any, Optional
from .gemini_client import GeminiClient
from app.core.config import settings
from .langgraph_agent import InterviewState
from .token_optimized_agent import InterviewGraph, _max_confusion_retries_for_duration, _max_questions_for_duration, _max_questions_per_topic
from .schemas import Action

class StreamingInterviewSession:
    def __init__(self, client: GeminiClient):
        self.client = client
        self.graph_engine = InterviewGraph(client=client)
        self.state: InterviewState = {
            "history": [],
            "performance_summary": "The interview is just starting.",
            "context_summary": "",
            "has_jd": False,
            "last_user_input": None,
            "current_evaluation": None,
            "interview_type": "technical",
            "time_context": "",
            "role": "",
            "company": "",
            "ended": False,
            "end_reason": "",
            # New accuracy fields
            "questions_asked": [],
            "current_question": "",
            "follow_up_hint": "",
            "is_developer_role": False,
            "coding_questions_asked": 0,
            "last_question_was_coding": False,
            "turn_number": 0,
            "last_answer_type": "not_applicable",
            "consecutive_non_answers": 0,
            "consecutive_disengaged": 0,
            "max_questions": 0,
            "max_questions_per_topic": 0,
            "max_confusion_retries": 2,
            "topic_question_counts": {},
            "input_tokens": 0,
            "output_tokens": 0,
            "rag_tokens": 0,
            "is_vertex": False,
            "opening_line": "",
            "closing_line": "",
            "source_question_candidates": [],
            "question_source_mode": "",
        }
        self.start_time: float = 0.0
        self.duration_limit: int = 0

    @property
    def history(self):
        return self.state["history"]

    @history.setter
    def history(self, value):
        self.state["history"] = value

    @property
    def ended(self):
        return self.state["ended"]

    @ended.setter
    def ended(self, value):
        self.state["ended"] = value

    @property
    def context_summary(self):
        return self.state["context_summary"]

    @context_summary.setter
    def context_summary(self, value):
        self.state["context_summary"] = value

    @property
    def interview_type(self):
        return self.state["interview_type"]

    @interview_type.setter
    def interview_type(self, value):
        self.state["interview_type"] = value

    @property
    def role(self):
        return self.state["role"]

    @role.setter
    def role(self, value):
        self.state["role"] = value

    @property
    def company(self):
        return self.state["company"]

    @company.setter
    def company(self, value):
        self.state["company"] = value

    @property
    def skills_remaining(self):
        return self.state.get("skills_remaining", [])

    @skills_remaining.setter
    def skills_remaining(self, value):
        self.state["skills_remaining"] = value

    @property
    def skills_covered(self):
        return self.state.get("skills_covered", [])

    @skills_covered.setter
    def skills_covered(self, value):
        self.state["skills_covered"] = value

    @property
    def performance_summary(self):
        return self.state.get("performance_summary", "")

    @performance_summary.setter
    def performance_summary(self, value):
        self.state["performance_summary"] = value

    @property
    def user_id(self):
        return self.state.get("user_id")

    @user_id.setter
    def user_id(self, value):
        self.state["user_id"] = value

    @property
    def session_id(self):
        return self.state.get("session_id")

    @session_id.setter
    def session_id(self, value):
        self.state["session_id"] = value

    @property
    def input_tokens(self):
        return self.state.get("input_tokens", 0)

    @input_tokens.setter
    def input_tokens(self, value):
        self.state["input_tokens"] = value

    @property
    def output_tokens(self):
        return self.state.get("output_tokens", 0)

    @output_tokens.setter
    def output_tokens(self, value):
        self.state["output_tokens"] = value

    @property
    def rag_tokens(self):
        return self.state.get("rag_tokens", 0)

    @rag_tokens.setter
    def rag_tokens(self, value):
        self.state["rag_tokens"] = value

    @property
    def is_vertex(self):
        return self.state.get("is_vertex", False)

    @is_vertex.setter
    def is_vertex(self, value):
        self.state["is_vertex"] = value


    async def initialize_session(self, user_id: str, session_id: str, resume_text: str, jd_text: str, interview_type: str = "technical", role: str = "", company: str = "", duration: int = 0, candidate_name: str = ""):
        self.start_time = time.time()
        self.duration_limit = duration
        max_q = _max_questions_for_duration(duration)
        print(f"[Session] duration={duration}min → max_questions={max_q} max_confusion_retries={_max_confusion_retries_for_duration(duration)}")
        
        # ── Token accounting ────────────────────────────────────────────────────
        self.input_tokens = 0
        self.output_tokens = 0
        self.rag_tokens = 0
        self.is_vertex = bool(getattr(settings, 'GOOGLE_APPLICATION_CREDENTIALS', None))

        # ── 1. Try to restore from resume context cache ─────────────────────────
        #       This skips the 2 most expensive init LLM calls entirely.
        from app.services.resume_cache import resume_cache
        
        cached = await resume_cache.get(resume_text, jd_text, interview_type, role, company)

        if cached:
            # ── CACHE HIT: restore summary + skills, zero LLM cost ──────────────
            context         = cached["context_summary"]
            initial_skills  = cached["skills"]
            print(f"[Session] ✅ Resume context cache HIT — skipped summarize_context + skills extraction (~2,000–5,000 tokens saved)")
        else:
            # ── CACHE MISS: run LLM calls and cache the result ───────────────────
            # 1a. Static context summarization
            context, context_usage = await self.client.summarize_context(resume_text, jd_text, interview_type, role, company, candidate_name)
            self.input_tokens  += context_usage.get("input_tokens", 0)
            self.output_tokens += context_usage.get("output_tokens", 0)
            
            # 1b. Keyword-based skill extraction (no LLM call needed with LlamaClient)
            initial_skills = self.client.extract_skills(context, interview_type, role, jd_text)
            print(f"[Session] Skills extracted (keyword-based): {initial_skills}")

            # ── Store in cache so future rounds are free ─────────────────────────
            await resume_cache.set(resume_text, jd_text, interview_type, role, company, context, initial_skills)
            print(f"[Session] 💾 Cached resume context for future rounds")


        print(f"[Session] Total init tokens: in={self.input_tokens}, out={self.output_tokens}")

        # ── Detect developer role for coding question enforcement ────────────────
        _developer_keywords = (
            "developer", "engineer", "programmer", "software", "backend", "frontend",
            "full stack", "fullstack", "sde", "swe", "coder", "architect", "devops",
            "data scientist", "ml engineer", "machine learning", "android", "ios",
        )
        role_lower = (role or "").lower()
        is_developer = (
            interview_type in ("technical", "problem")
            and any(kw in role_lower for kw in _developer_keywords)
        )
        print(f"[Session] is_developer_role={is_developer} for role={role!r} type={interview_type!r}")

        plan_opening_line = ""
        plan_closing_line = ""
        plan_candidates: list[dict[str, Any]] = []
        question_source_mode = ""
        must_ask_coding_question = interview_type in ("technical", "problem")

        try:
            from app.services.ai.source_interview_planner import source_interview_planner

            def _build_plan_blocking() -> dict[str, Any]:
                # build_plan performs synchronous network work internally (requests); run it off-loop.
                return asyncio.run(
                    source_interview_planner.build_plan(
                        resume_text=resume_text,
                        jd_text=jd_text,
                        company_name=company,
                        role=role,
                        interview_type=interview_type,
                        duration_minutes=duration,
                    )
                )

            plan = await asyncio.wait_for(asyncio.to_thread(_build_plan_blocking), timeout=8.0)
            plan_opening_line = str(plan.get("opening_line", "") or "")
            plan_closing_line = str(plan.get("closing_line", "") or "")
            plan_candidates = list(plan.get("question_candidates", []) or [])
            if len(plan_candidates) > 1:
                random.SystemRandom().shuffle(plan_candidates)
            question_source_mode = str(plan.get("question_source_mode", "") or "")
            must_ask_coding_question = bool(plan.get("needs_coding_question", must_ask_coding_question))

            planner_topics = [str(topic) for topic in (plan.get("topics", []) or []) if str(topic).strip()]
            if planner_topics:
                initial_skills = list(dict.fromkeys([*planner_topics, *initial_skills]))[:12]

            print(f"[Session] source planner ready: candidates={len(plan_candidates)} mode={question_source_mode or 'n/a'}")
        except asyncio.TimeoutError:
            print("[Session] source planner timed out after 8s, continuing with default flow")
        except Exception as exc:
            print(f"[Session] source planner unavailable, continuing with default flow: {exc}")

        # ── Generate greeting and closing lines ──────────────────────────────────
        opening_line = plan_opening_line or f"Hi {candidate_name or 'there'}, thanks for joining us today. Let's get started with an interview focused on {role or interview_type} and the role fit for {company or 'your target company'}."
        closing_line = plan_closing_line or "Thank you so much for interviewing with us today. We really appreciate your time, and we'll share the next steps soon."

        self.state.update({
            "user_id": user_id,
            "session_id": session_id,
            "context_summary": context,
            "has_jd": bool(jd_text and jd_text.strip()),
            "interview_type": interview_type,
            "role": role,
            "company": company,
            "history": [],
            "performance_summary": (
                f"Interview starting. Candidate: {candidate_name or 'unknown'}. "
                f"Role: {role or 'not specified'}. Round: {interview_type}. "
                f"No answers evaluated yet."
            ),
            "skills_remaining": initial_skills,
            "skills_covered": [],
            "ended": False,
            "end_reason": "",
            # Accuracy fields
            "questions_asked": [],
            "current_question": "",
            "follow_up_hint": "",
            "is_developer_role": is_developer,
            "coding_questions_asked": 0,
            "last_question_was_coding": False,
            "turn_number": 0,
            "last_answer_type": "not_applicable",
            "consecutive_non_answers": 0,
            "consecutive_disengaged": 0,
            "max_questions": max_q,
            "max_questions_per_topic": _max_questions_per_topic(duration),
            "max_confusion_retries": _max_confusion_retries_for_duration(duration),
            "topic_question_counts": {},
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "opening_line": opening_line,
            "closing_line": closing_line,
            "source_question_candidates": plan_candidates,
            "question_source_mode": question_source_mode,
            "must_ask_coding_question": must_ask_coding_question,
            "needs_coding_question": must_ask_coding_question,
        })

    def get_time_context(self) -> str:
        if self.duration_limit <= 0 or self.start_time <= 0:
            return ""
        elapsed = time.time() - self.start_time
        elapsed_min = elapsed / 60.0
        remaining = self.duration_limit - elapsed_min
        elapsed_str = f"{int(elapsed_min)}m {int(elapsed % 60)}s"
        pct = elapsed_min / self.duration_limit

        if remaining <= 0:
            return f"⏰ TIME UP ({elapsed_str} elapsed). CLOSE THE INTERVIEW NOW — set action=END immediately."
        if pct >= 0.90:
            return (f"🚨 FINAL MINUTES: {elapsed_str} elapsed, only ~{int(remaining * 60)}s left of {self.duration_limit}min. "
                    f"Ask at most ONE more question then wrap up and set action=END.")
        if pct >= 0.75:
            return (f"⚠ Nearing end: {elapsed_str} elapsed, ~{int(remaining)}m remaining of {self.duration_limit}min. "
                    f"Start steering toward a close — cover only the single most important topic left.")
        return f"TIME: {elapsed_str} elapsed. ~{int(remaining)}m remaining of {self.duration_limit}min."

    async def stream_response(self, user_input: str = None):
        # Soft time enforcement: Allow a 5-minute grace period beyond duration_limit
        # before forcing a hard close. The AI is already prompted to wrap up 
        # via get_time_context() in the graph.
        if self.duration_limit > 0 and self.start_time > 0:
            elapsed_min = (time.time() - self.start_time) / 60.0
            if elapsed_min >= (self.duration_limit + 5):
                closing = (
                    "We've reached the absolute time limit for this session. "
                    "Thank you so much for your time today — it was a productive conversation. "
                    "I'll now begin generating your feedback report. Best of luck!"
                )
                self.state["ended"] = True
                self.state["end_reason"] = "HARD_TIME_LIMIT"
                yield {"type": "metadata", "is_coding": False}
                chunk_size = 12
                for i in range(0, len(closing), chunk_size):
                    yield {"type": "text", "content": closing[i:i + chunk_size]}
                    await asyncio.sleep(0.01)
                self.state["history"].append({"role": "model", "content": closing})
                return

        # Update local state
        self.state["last_user_input"] = user_input
        self.state["time_context"] = self.get_time_context()
        
        # Run the Optimized Graph
        # This uses the running summary instead of full history for the LLM call
        updated_state = await self.graph_engine.run_turn(self.state)
        self.state = updated_state
        
        evaluation = self.state["current_evaluation"]
        next_step = evaluation.get("next_step", {}) if isinstance(evaluation, dict) else getattr(evaluation, "next_step", {})
        # Prefer canonical state question because run_turn may prepend greeting/closing there.
        question_text = str(self.state.get("current_question", "") or "").strip()
        if not question_text:
            question_text = next_step.get("question", "") if isinstance(next_step, dict) else getattr(next_step, "question", "")
        is_coding = next_step.get("is_coding_question", False) if isinstance(next_step, dict) else getattr(next_step, "is_coding_question", False)

        if self.state["ended"]:
            if len(question_text) < 5:
                question_text = "Thank you for the conversation. We have covered the key areas. I'll pass my feedback to the team."
            is_coding = False

        # Yield metadata
        yield {"type": "metadata", "is_coding": is_coding}

        # Splitting the response into sentences for parallel TTS
        sentences = [s.strip() + "." for s in question_text.split(".") if s.strip()]
        
        # Fire off TTS generation tasks in parallel
        tts_tasks = []
        for sentence in sentences:
            tts_tasks.append(asyncio.create_task(generate_tts_async(sentence)))
        
        # Stream text and audio chunks back to the client as they become ready
        for i, sentence in enumerate(sentences):
            # Stream text for this sentence
            for j in range(0, len(sentence), 12):
                yield {"type": "text", "content": sentence[j:j+12]}
                await asyncio.sleep(0.01)
            
            # Wait for the corresponding audio chunk
            audio_b64 = await tts_tasks[i]
            if audio_b64:
                yield {"type": "audio", "base64": audio_b64}

        # run_turn already records the model question in history; avoid duplicate entries.
        history = self.state.get("history", [])
        if not history or history[-1].get("role") != "model" or history[-1].get("content") != question_text:
            self.state["history"].append({"role": "model", "content": question_text})

        # Real-time Reporting: Send current usage to backend after each turn
        # This allows the admin dashboard to see cost/token growth in real-time
        asyncio.create_task(self.report_usage(self.state.get("user_id") or "anonymous", self.state.get("session_id") or "demo"))

    async def report_usage(self, user_id: str, session_id: str):
        """Send token usage to the NestJS backend for analytics."""
        active_model = "llama-finetuned"
        in_cost = 0.0
        out_cost = 0.0
        total_cost = 0.0

        usage_data = {
            "userId": user_id,
            "sessionId": session_id,
            "model": active_model,
            "inputTokens": self.input_tokens,
            "outputTokens": self.output_tokens,
            "totalTokens": self.input_tokens + self.output_tokens,
            "costUsd": total_cost,
            "inputCostUsd": in_cost,
            "outputCostUsd": out_cost,
            "subscriptionStatus": "free",  # Backend will refine from user DB
            "source": "interview",
            "interviewType": self.state.get("interview_type", ""),
            "role": self.state.get("role", ""),
            "company": self.state.get("company", ""),
            "endReason": self.state.get("end_reason", ""),
            "isVertex": self.is_vertex,
            "ragTokens": self.rag_tokens,
        }

        target_url = f"{settings.BACKEND_URL}/analytics/ai-usage"
        print(f"[AI] Reporting usage to {target_url}  in={self.input_tokens}, out={self.output_tokens}")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    target_url,
                    json=usage_data,
                    timeout=5.0
                )
                response.raise_for_status()
            print(f"[AI] Usage reported OK: in={self.input_tokens}, out={self.output_tokens}, total={self.input_tokens + self.output_tokens} tokens for user {user_id} ({self.state.get('interview_type', '')})")
        except Exception as e:
            print(f"[AI] Failed to report usage to {target_url}: {type(e).__name__}: {e}")
