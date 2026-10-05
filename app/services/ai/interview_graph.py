from typing import Annotated, List, TypedDict, Dict, Any, Optional
import operator
import json
import asyncio
import logging
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from app.core.config import settings
from .schemas import QuestionEvaluation, EvaluateTurnOutput, Action
from .fine_tuned_gateway import gateway_client

logger = logging.getLogger(__name__)


class InterviewState(TypedDict):
    # Full history (accumulates all turns, used for final feedback)
    history: List[dict]

    # Compressed state summaries (primary info source for turn generation)
    performance_summary: str       # Rolling evaluation of candidate quality
    context_summary: str           # Resume + JD summary from initialization
    has_jd: bool                   # Whether a Job Description was provided

    # Skill coverage tracking
    skills_remaining: List[str]
    skills_covered: List[str]

    # Anti-repetition: every question the AI has asked, in order
    questions_asked: List[str]

    # Current turn context
    last_user_input: Optional[str]
    current_question: str          # The last question the AI posed (for accurate eval)
    current_evaluation: Optional[QuestionEvaluation]
    follow_up_hint: str            # Set by evaluate_turn when answer needs probing

    # Session configuration
    interview_type: str
    time_context: str
    role: str
    company: str

    # Developer-role coding enforcement
    is_developer_role: bool        # Detected from role/context at init
    coding_questions_asked: int    # How many coding questions asked so far
    last_question_was_coding: bool # Whether the most recent question was a coding task
    turn_number: int               # Turn index for pacing decisions

    # Response classification from last evaluate_turn (drives question strategy)
    last_answer_type: str          # genuine_answer | confused | refused | off_topic | incomplete | not_applicable
    consecutive_non_answers: int   # How many confused/refused turns in a row

    # Control
    ended: bool
    end_reason: str                # USER_EXIT | AI_ENDED | TIME_EXCEEDED | DISENGAGED | BUDGET_EXHAUSTED

    # Question budget (0 = no limit)
    max_questions: int

    # Per-topic question cap (0 = no limit)
    max_questions_per_topic: int
    topic_question_counts: Dict[str, int]   # skill → #times asked

    # How many times to rephrase a confused question before moving on
    max_confusion_retries: int

    # Track consecutive refused/off_topic to auto-terminate disengaged candidates
    consecutive_disengaged: int

    # Token accounting
    input_tokens: int
    output_tokens: int


# ─────────────────────────────────────────────────────────────────────────────
# Question budget by duration
# ─────────────────────────────────────────────────────────────────────────────
def _max_questions_for_duration(duration_minutes: int) -> int:
    """Return the maximum number of questions for a given interview duration."""
    if duration_minutes <= 0:
        return 0        # unlimited
    if duration_minutes <= 15:
        return 8        # 15 min  → 8 questions
    if duration_minutes <= 30:
        return 13       # 30 min  → 13 questions
    if duration_minutes <= 45:
        return 16       # 45 min  → 16 questions
    return 18           # 60 min  → 18 questions


def _max_questions_per_topic(duration_minutes: int) -> int:
    """Return the maximum consecutive questions allowed on a single topic/skill."""
    if duration_minutes <= 0:
        return 0        # unlimited
    if duration_minutes <= 15:
        return 1        # 15 min → max 1 question per topic (move on fast)
    if duration_minutes <= 30:
        return 2        # 30 min → max 2 per topic
    if duration_minutes <= 45:
        return 2        # 45 min → max 2 per topic
    return 3            # 60 min → max 3 per topic


def _max_confusion_retries_for_duration(duration_minutes: int) -> int:
    """How many times the interviewer will rephrase a confused question before moving on."""
    if duration_minutes <= 0:
        return 2        # no limit → be patient
    if duration_minutes <= 15:
        return 1        # 15 min: rephrase once then move on
    return 2            # 30 / 45 / 60 min: rephrase twice then move on


# ─────────────────────────────────────────────────────────────────────────────
# Per-interview-type rules (used in _produce_question prompt)
# ─────────────────────────────────────────────────────────────────────────────
_TYPE_RULES: Dict[str, str] = {
    "technical": """
INTERVIEW TYPE: TECHNICAL
• ALWAYS base your first 2-3 questions on the candidate’s OWN tech stack listed in their resume (React, Node, Python, etc.). Do NOT ask generic language-agnostic questions when you know what tools they use.
• Cover: data structures, algorithms, system design, language-specific deep-dives, debugging/performance reasoning.
• Ask the candidate to WRITE CODE (is_coding_question=true) when exploring algorithms or implementation tasks.
• Also set is_coding_question=true when asking them to show step-by-step calculations, derivations, or worked numeric examples (e.g. convolution, matrix ops, complexity math) — the editor serves as a scratchpad for written work too.
• When they submit code: evaluate correctness, edge-cases, and time/space complexity.
• Progression arc: warm-up on their primary stack → core CS problem → system design / architecture → harder algorithm or optimisation.
• For SENIOR candidates (5+ years): skip entry-level questions entirely. Focus on architecture decisions, trade-offs, performance bottlenecks, and scalability.
• For JUNIOR/MID: focus on fundamentals of their stated stack, basic algorithms, debugging, and clean code.
• MINIMUM DIFFICULTY FLOOR: Never ask a question that a first-year CS student with no experience could answer trivially (e.g. “write a function that adds two numbers” is only valid if the candidate has zero experience). If the candidate has 2+ years, start at intermediate level.
""",
    "behavioral": """
INTERVIEW TYPE: BEHAVIORAL
• Reference the candidate’s ACTUAL projects and companies from their resume when asking. E.g. “In your time at [Company], tell me about a time when…”
• Use the STAR method (Situation, Task, Action, Result) to evaluate every story.
• Ask for SPECIFIC past experiences, not hypotheticals.
• Probe: teamwork, conflict resolution, leadership under pressure, handling failure, cross-functional collaboration.
• Follow up if a story lacks a concrete Result or Action — push for specifics.
• MINIMUM DEPTH: Do NOT end the interview before asking at least 5-6 substantial behavioral questions. 
• Behavioral interviews SHOULD take time to explore human experiences. Avoid rushing to END.
• Avoid asking two questions on the same competency back to back.
""",
    "problem": """
INTERVIEW TYPE: PROBLEM SOLVING
• Ground problems in the candidate’s domain (e.g. for a frontend dev, frame problems around UI performance, state management trade-offs, rendering optimisation).
• Present real-world scenarios or case studies — not abstract puzzles unless they fit the role.
• Ask them to break down ambiguous problems step-by-step: define → assumptions → approach → solution.
• You CAN present a practical implementation challenge (is_coding_question=true), especially when the candidate’s stack is known.
• Score the thought process as much as the final answer.
""",
    "hr": """
INTERVIEW TYPE: HR / CULTURE FIT
• Tone: warm, conversational, but precise.
• Cover: motivation, career goals, work-style, values alignment, salary expectations.
• Assess: communication, professionalism, long-term fit.
• Ask open-ended questions and listen for self-awareness and clarity of purpose.
• Do NOT ask coding questions. is_coding_question must always be false.
""",
}


class InterviewGraph:
    def __init__(self, api_key: str, model_name: str = None):
        from app.core.key_manager import key_manager
        resolved_model = model_name or key_manager.get_gemini_model()
        # Lower temperature for evaluation → consistent, reliable scores
        self.eval_llm = ChatGoogleGenerativeAI(
            model=resolved_model,
            google_api_key=api_key,
            temperature=0.25,
        )
        # Standard temperature for question generation → natural, focused questions
        self.gen_llm = ChatGoogleGenerativeAI(
            model=resolved_model,
            google_api_key=api_key,
            temperature=0.4,
        )

        # Structured outputs — include_raw=True lets us read usage_metadata
        self.eval_structured_llm = self.eval_llm.with_structured_output(
            EvaluateTurnOutput, include_raw=True
        )
        self.structured_llm = self.gen_llm.with_structured_output(
            QuestionEvaluation, include_raw=True
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(InterviewState)
        builder.add_node("evaluate_turn", self._evaluate_turn)
        builder.add_node("produce_question", self._produce_question)
        builder.set_entry_point("evaluate_turn")
        builder.add_edge("evaluate_turn", "produce_question")
        builder.add_edge("produce_question", END)
        return builder.compile()

    # ─────────────────────────────────────────────────────────────────────────
    # Node 1: Evaluate the candidate's last answer
    # ─────────────────────────────────────────────────────────────────────────
    async def _evaluate_turn(self, state: InterviewState) -> Dict[str, Any]:
        """
        Classifies the candidate's response type (genuine / confused / refused / etc.)
        and updates the performance summary and skill coverage accordingly.
        """
        if not state.get("last_user_input"):
            return {
                "performance_summary": state.get("performance_summary", "Interview is starting."),
                "skills_covered": state.get("skills_covered", []),
                "skills_remaining": state.get("skills_remaining", []),
                "follow_up_hint": "",
                "last_answer_type": "not_applicable",
                "consecutive_non_answers": 0,
                "max_confusion_retries": state.get("max_confusion_retries", 2),
                "input_tokens": state.get("input_tokens", 0),
                "output_tokens": state.get("output_tokens", 0),
            }

        skills_list = ", ".join(state["skills_remaining"][:12]) or "all listed skills covered"
        covered_list = ", ".join(state["skills_covered"][-10:]) or "none yet"
        last_question = state.get("current_question") or "Opening / initial question"

        prompt = f"""You are a rigorous interview evaluator for a {state['interview_type']} round.
Role being interviewed for: {state.get('role', 'Not specified')}

CURRENT PERFORMANCE SUMMARY:
{state['performance_summary']}

SKILLS STILL TO COVER: {skills_list}
SKILLS ALREADY DEMONSTRATED: {covered_list}

─── LAST EXCHANGE ───
INTERVIEWER ASKED: {last_question}
CANDIDATE RESPONDED: {state['last_user_input']}
─────────────────────

STEP 1 — Classify answer_type FIRST (this drives the entire strategy):
  end_requested      — candidate explicitly asked to stop, end, quit, leave, or said they are no longer interested in the interview — CHECK THIS FIRST
  coding_requested   — candidate explicitly asked to be given a coding question, DSA problem, or implementation task.
                       EXAMPLES: "ask me a coding question", "give me a coding problem", "I want a DSA question",
                       "can you give me a coding challenge", "please ask me something to code".
                       Classify as this type BEFORE checking confused or off_topic.
  wait_requested     — candidate explicitly asked for a moment to think, or said "give me a second", "one moment", "let me think"
  confused           — candidate said they don't understand, asked for clarification, asked you to explain / rephrase / simplify,
                     or said the question is unclear. SPECIFIC EXAMPLES that MUST be classified as confused:
                       "I didn't get your question", "can you explain in simple language", "please repeat",
                       "repeat your question", "repeat the question", "can you repeat", "say that again",
                       "what do you mean", "I don't understand", "can you rephrase", "can you simplify",
                       "explain this to me", "I'm not sure what you're asking", "please clarify".
                     ANY request for the question to be explained differently = confused.
  genuine_answer   — candidate actually attempted the question (even partially)
  refused          — candidate explicitly said they don’t want to answer, or tried to avoid the topic
  off_topic        — candidate talked about something unrelated to what was asked
  incomplete       — candidate started answering but stopped or gave only a fragment
  not_applicable   — this is a greeting or a turn with no question yet

STEP 2 — Set answer_quality based on type:
  If answer_type is end_requested: set answer_quality to "not_applicable". The interview must end.
  If answer_type is coding_requested: set answer_quality to "not_applicable". The candidate wants a coding challenge — do NOT update skill coverage.
  If answer_type is wait_requested: set answer_quality to "not_applicable". The candidate has NOT answered — do NOT score or update skill coverage.
  If answer_type is confused/refused/off_topic: set answer_quality to "not_applicable". Do NOT penalise for confusion.
  If genuine_answer: strong | adequate | weak | incorrect based on actual content.
  If incomplete: usually "weak".

STEP 3 — Update new_summary:
  Record what happened. Be specific. If confused, note the topic they were confused about.
  If wait_requested: note that candidate is still thinking; do NOT update performance.
  Do NOT claim skill coverage for confused/refused/off_topic/wait_requested answers.

STEP 4 — should_follow_up:
  true ONLY if answer_type is genuine_answer AND quality is weak/incorrect AND the topic matters.
  false for confused/refused/wait_requested (they need a different strategy, not a follow-up drill).

STEP 5 — follow_up_hint: specific gap to probe IF should_follow_up is true. Empty otherwise.
"""

        try:
            # ── Try fine-tuned evaluator gateway if enabled ───────────────────
            data: Optional[EvaluateTurnOutput] = None
            in_tok = 0
            out_tok = 0

            if settings.FINE_TUNED_EVALUATOR_ENABLED:
                eval_start = time.monotonic()
                eval_messages = [
                    {"role": "system", "content": "You are a precise interview evaluator. Classify the response type first, then evaluate."},
                    {"role": "user", "content": prompt},
                ]
                gateway_result = await gateway_client.generate_evaluator_response(
                    messages=eval_messages, temperature=0.25, max_tokens=1024
                )
                if gateway_result is not None:
                    try:
                        data = EvaluateTurnOutput.model_validate(gateway_result)
                        eval_duration = int((time.monotonic() - eval_start) * 1000)
                        logger.info(
                            "[Graph] evaluate_turn via gateway (%dms): type=%s quality=%s",
                            eval_duration, data.answer_type, data.answer_quality,
                        )
                    except Exception as parse_exc:
                        logger.warning(
                            "[Graph] Gateway evaluator response failed Pydantic validation: %s — falling back to Gemini",
                            parse_exc,
                        )
                        data = None
                else:
                    logger.warning("[Graph] Gateway evaluator returned None — falling back to Gemini")

            # ── Fallback to Gemini ────────────────────────────────────────────
            if data is None:
                result = await self.eval_structured_llm.ainvoke([
                    SystemMessage(content="You are a precise interview evaluator. Classify the response type first, then evaluate."),
                    HumanMessage(content=prompt),
                ])
                data = result["parsed"]
                raw_msg = result.get("raw")

                usage_meta = getattr(raw_msg, "usage_metadata", {}) or {}
                in_tok = usage_meta.get("input_tokens", 0)
                out_tok = usage_meta.get("output_tokens", 0)

            print(
                f"[Graph] evaluate_turn: type={data.answer_type} quality={data.answer_quality} "
                f"follow_up={data.should_follow_up} tokens=in:{in_tok} out:{out_tok}"
            )

            # Only credit skill coverage for genuine answers
            if data.answer_type in ("genuine_answer", "incomplete"):
                remaining = [s for s in state["skills_remaining"] if s not in data.newly_covered_skills]
                covered = list(set(state["skills_covered"] + data.newly_covered_skills))
            else:
                remaining = state["skills_remaining"]
                covered = state["skills_covered"]

            # Update per-topic question counts
            topic_counts = dict(state.get("topic_question_counts", {}) or {})
            if data.answer_type in ("genuine_answer", "incomplete"):
                for skill in data.newly_covered_skills:
                    topic_counts[skill] = topic_counts.get(skill, 0) + 1

            # Auto-remove skills from remaining that have hit the per-topic cap
            max_per_topic = state.get("max_questions_per_topic", 0) or 0
            if max_per_topic > 0:
                remaining = [s for s in remaining if topic_counts.get(s, 0) < max_per_topic]

            # Track consecutive non-answers
            prev_non = state.get("consecutive_non_answers", 0)
            if data.answer_type in ("confused", "refused", "off_topic", "wait_requested"):
                consecutive_non = prev_non + 1
            else:
                consecutive_non = 0  # end_requested / genuine / incomplete / coding_requested all reset the streak

            # Track disengaged separately (refused + off_topic only — not confusion)
            prev_disengaged = state.get("consecutive_disengaged", 0)
            if data.answer_type in ("refused", "off_topic"):
                consecutive_disengaged = prev_disengaged + 1
            else:
                consecutive_disengaged = 0

            return {
                "performance_summary": data.new_summary,
                "skills_covered": covered,
                "skills_remaining": remaining,
                "follow_up_hint": data.follow_up_hint if data.should_follow_up else "",
                "last_answer_type": data.answer_type,
                "consecutive_non_answers": consecutive_non,
                "consecutive_disengaged": consecutive_disengaged,
                "topic_question_counts": topic_counts,
                "max_confusion_retries": state.get("max_confusion_retries", 2),
                "input_tokens": state.get("input_tokens", 0) + in_tok,
                "output_tokens": state.get("output_tokens", 0) + out_tok,
                "end_reason": "USER_EXIT" if data.answer_type == "end_requested" else state.get("end_reason", ""),
                "ended": data.answer_type == "end_requested" or state.get("ended", False)
            }

        except Exception as e:
            print(f"[Graph] evaluate_turn error (non-fatal): {e}")
            return {
                "performance_summary": state.get("performance_summary", "Evaluation error — continuing."),
                "skills_covered": state.get("skills_covered", []),
                "skills_remaining": state.get("skills_remaining", []),
                "follow_up_hint": "",
                "last_answer_type": "genuine_answer",
                "consecutive_non_answers": 0,
                "consecutive_disengaged": 0,
                "max_confusion_retries": state.get("max_confusion_retries", 2),
                "input_tokens": state.get("input_tokens", 0),
                "output_tokens": state.get("output_tokens", 0),
            }

    # ─────────────────────────────────────────────────────────────────────────
    # Node 2: Generate the next strategic question
    # ─────────────────────────────────────────────────────────────────────────
    async def _produce_question(self, state: InterviewState) -> Dict[str, Any]:
        history = state.get("history", [])
        turn_number = state.get("turn_number", 0)
        answer_type = state.get("last_answer_type", "genuine_answer")
        consecutive_non = state.get("consecutive_non_answers", 0)

        # ── Recent conversation buffer (last 8 entries ~4 Q+A pairs) ──
        recent = history[-8:] if len(history) > 8 else history
        recent_text = (
            "\n".join(
                f"{'INTERVIEWER' if m['role'] == 'model' else 'CANDIDATE'}: {m['content']}"
                for m in recent
            )
            if recent
            else "(No exchanges yet — this is the opening turn.)"
        )

        # ── Anti-repetition ──
        questions_asked = state.get("questions_asked", [])
        anti_repeat_block = ""
        if questions_asked:
            asked_summary = "\n".join(
                f"  {i+1}. {q[:130]}" for i, q in enumerate(questions_asked[-14:])
            )
            anti_repeat_block = (
                f"\n⛔ ALREADY ASKED — DO NOT REPEAT OR REPHRASE ANY OF THESE:\n{asked_summary}"
            )

        # ── Question budget enforcement ──
        max_q = state.get("max_questions", 0)
        question_limit_block = ""
        if max_q > 0:
            questions_used = len(questions_asked)
            questions_left = max_q - questions_used
            if questions_left <= 0:
                # Hard limit reached — force END immediately
                question_limit_block = f"""
⏰ QUESTION LIMIT REACHED: This interview was allocated a maximum of {max_q} questions and all have been used.
You MUST:
  1. Wrap up warmly in one sentence.
  2. Give a brief summary of how the interview went.
  3. Set action=END — this overrides all other instructions.
  4. Do NOT ask any new question.
"""
            elif questions_left == 1:
                question_limit_block = f"""
⚠ FINAL QUESTION: This is question {questions_used + 1} of {max_q} (the last one).
  - Ask the single most important unanswered question remaining.
  - After the candidate answers, you will wrap up — do NOT plan follow-ups.
"""
            elif questions_left == 2:
                question_limit_block = f"""
⚠ APPROACHING LIMIT: {questions_left} questions remaining out of {max_q} total.
  - Prioritise the highest-value uncovered topics.
  - Start steering toward a natural close.
"""

        # ── Per-topic cap block ──
        topic_counts = state.get("topic_question_counts", {}) or {}
        max_per_topic = state.get("max_questions_per_topic", 0) or 0
        exhausted_topics = [s for s, c in topic_counts.items() if max_per_topic > 0 and c >= max_per_topic]
        exhausted_topics_block = ""
        if exhausted_topics:
            exhausted_topics_block = (
                f"\n⛔ TOPIC LIMIT REACHED — these skills have already been explored enough — DO NOT ask more about them: "
                f"{', '.join(exhausted_topics)}\n"
                f"  Pick a DIFFERENT skill from SKILLS REMAINING that has NOT been exhausted.\n"
            )

        # ── Early termination: candidate is not engaging ──
        consecutive_disengaged = state.get("consecutive_disengaged", 0)
        _DISENGAGEMENT_THRESHOLD = 3

        # ── Response-type routing (core new logic) ──
        routing_block = ""
        if consecutive_disengaged >= _DISENGAGEMENT_THRESHOLD and answer_type in ("refused", "off_topic"):
            routing_block = f"""
🛑 INTERVIEW MUST END — CANDIDATE IS NOT ENGAGING.
The candidate has given {consecutive_disengaged} consecutive responses without any genuine attempt to answer.
This indicates the candidate is not participating in good faith.
You MUST:
  1. Close the session professionally in 1–2 sentences — keep it respectful.
  2. Briefly note that a productive conversation wasn’t possible this session.
  3. Wish them well if they’d like to attempt this again with a fresh session.
  4. Set action=END — this overrides ALL other instructions. Do NOT ask any new question.
"""
        elif answer_type == "end_requested":
            routing_block = """
🛑 CANDIDATE HAS EXPLICITLY ASKED TO END THE INTERVIEW.
You MUST do ALL of the following — no exceptions:
  1. Thank them warmly for their time in one sentence.
  2. Give a single short sentence summarising how the interview went based on RUNNING PERFORMANCE ASSESSMENT.
  3. Wish them well.
  4. Set action=END in your response. This is mandatory — do NOT set action=CONTINUE.
  5. The question field should contain only your closing statement (no new question).
"""
        elif answer_type == "wait_requested":
            last_q = state.get("current_question", "")
            routing_block = f"""
⏳ CANDIDATE ASKED FOR A MOMENT / IS STILL THINKING.
Strategy — you MUST follow this exactly:
  1. Respond with exactly ONE short, warm sentence acknowledging they are thinking (e.g. "Of course, take your time!").
  2. Immediately re-state the EXACT SAME question below — word for word or a very close paraphrase. Do NOT ask anything new.
  3. Do NOT move to any other topic. Do NOT ask a follow-up or a new question.
  4. action must remain CONTINUE.
Original question to re-state: {last_q[:400]}
"""
        elif answer_type == "confused":
            last_q = state.get("current_question", "")
            max_retries = state.get("max_confusion_retries", 2)
            last_was_coding = state.get("last_question_was_coding", False)
            coding_note = (
                "\n  ⚠ THE PREVIOUS QUESTION WAS A CODING TASK. Do NOT re-open the code editor (is_coding_question=false). "
                "Convert the coding task into a verbal question about the same concept first."
            ) if last_was_coding else ""
            if consecutive_non >= max_retries + 1:
                # Exhausted all rephrases — move on.
                routing_block = f"""
⏭ CANDIDATE IS STILL CONFUSED after {max_retries} rephrase attempt(s). Move to a new topic now.

YOU MUST:
  1. One short empathetic sentence (e.g. "No worries, let’s move on!").
  2. Ask a BRAND NEW question on a DIFFERENT skill/topic from SKILLS REMAINING.
  3. The new question must be clear, concrete, one sentence. is_coding_question=false.
  4. action remains CONTINUE. Do NOT reference the previous question.

Previous question you must NOT rephrase again: "{last_q[:200]}"
"""
            elif consecutive_non >= 2:
                # Second rephrase (30+ min interviews with max_retries == 2)
                routing_block = f"""
🔄 SECOND ATTEMPT: Candidate is still confused. Rephrase once more, even simpler.
This is your LAST rephrase — if they are confused again after this, you will move on.
{coding_note}

YOU MUST:
  1. One warm sentence (e.g. "Let me try once more!").
  2. Strip the question to its absolute core — a SINGLE plain-English sentence, max 12 words.
     — No technical terms. Use an everyday analogy if needed.
     — Example: "In your project, which Python package did you use the most?"
  3. SAME topic as the original — do NOT switch subjects.
  4. action remains CONTINUE.

Original question to simplify further: "{last_q[:300]}"
"""
            else:
                # First time confused — rephrase once, same topic.
                routing_block = f"""
🔄 CANDIDATE DID NOT UNDERSTAND THE QUESTION. They asked you to explain or simplify it.
This is your ONE chance to rephrase — after this, if they are still confused you will move on.

YOU MUST FOLLOW THESE RULES — NO EXCEPTIONS:

  RULE 1 — SAME TOPIC, SIMPLER WORDS:
    Your rephrased question MUST be about the EXACT SAME TOPIC as the original.
    The original question was about: "{last_q[:200]}"
    You are NOT allowed to switch to a different topic, a different skill, or a different project.
    If the original asked about Python, your rephrased question must also be about Python.
    If the original asked about a specific project, stay on that project.

  RULE 2 — ONE SENTENCE, PLAIN LANGUAGE:
    Rewrite it as ONE short, concrete question a non-technical person could understand.
    Remove jargon. Remove sub-parts. Remove "and also". Max 20 words.
    Ground it in a real, specific example (e.g. "In your [project], why did you choose X?").

  RULE 3 — DROP CODING, BRIDGE TO VERBAL:
    If the original was a coding task (is_coding_question was true), do NOT re-open the code editor.
    Instead, ask the same concept as a verbal question first:
      e.g. If original was "Implement a binary search function" →
           Rephrase as "Can you walk me through how binary search works in plain words?"
    Only return to the coding problem after the candidate has shown verbal understanding.
    is_coding_question must be false for this rephrased question.

  RULE 4 — DO NOT ADD ANYTHING ELSE:
    Start with one warm acknowledgement sentence ("Sure!", "Of course!", "No problem!").
    Then ask ONLY the rephrased question. Nothing after it.
    action must remain CONTINUE.

  SELF-CHECK before finalising your response:
    ✓ Is my rephrased question about the same topic as the original? (If NO → rewrite it)
    ✓ Does it have only one question mark? (If NO → remove extras)
    ✓ Is it shorter and simpler than the original? (If NO → simplify further)
    ✓ If original was a coding question, is is_coding_question set to false? (If NO → fix it)

  Original question to simplify (DO NOT change the subject):
  "{last_q[:400]}"
"""
        elif answer_type == "coding_requested":
            routing_block = """
💻 CANDIDATE EXPLICITLY REQUESTED A CODING QUESTION.
You MUST follow this exactly:
  1. One short enthusiastic sentence (e.g. "Sure, let's get into some code!").
  2. Ask a concrete, hands-on coding problem that fits their tech stack and seniority level.
     — Real problem (no pseudocode prompts). Something they must actually implement.
     — Examples: "Implement a function that...", "Write a solution to...", "Given this input, code a..."
  3. You MUST set is_coding_question=true — this opens the code editor. This is NON-NEGOTIABLE.
  4. action remains CONTINUE.
  DO NOT ask a conceptual question. DO NOT ask about their experience. Write a coding problem.
"""
        elif answer_type == "refused":
            routing_block = """
⏭ CANDIDATE DECLINED TO ANSWER. Follow this strategy exactly:
  1. Acknowledge briefly and move on without pressure (one sentence).
  2. Pivot to a DIFFERENT topic from their confirmed tech stack in CANDIDATE CONTEXT.
  3. New question must still match their seniority level — do NOT simplify to beginner basics.
  4. Do NOT keep pressing on the refused topic this turn.
"""
        elif answer_type == "off_topic":
            routing_block = """
🚧 CANDIDATE WENT OFF-TOPIC.
Strategy: redirect briefly ("Interesting point, but let's stay focused on…") then ask a sharper,
more focused question on the same area they drifted from.
"""
        elif answer_type == "incomplete":
            routing_block = """
⏳ CANDIDATE GAVE AN INCOMPLETE ANSWER.
Strategy: acknowledge what they covered, then ask them to continue or address specifically
the part they left unfinished.
"""

        # ── Follow-up routing (only for genuine weak/incorrect answers) ──
        follow_up = state.get("follow_up_hint", "")
        follow_up_block = ""
        if follow_up and answer_type == "genuine_answer":
            follow_up_block = (
                f"\n🔁 FOLLOW-UP REQUIRED: Candidate's answer was weak or incomplete on a key point.\n"
                f"  Gap to probe: «{follow_up}»\n"
                f"  You MUST ask a follow-up question drilling into THIS SPECIFIC GAP before moving to any new topic.\n"
                f"  Requirements: (1) Reference something specific they said in their last answer.\n"
                f"  (2) Ask 'how', 'why', or 'what happened when' — not just 'can you explain more'.\n"
                f"  (3) If the gap is technical, ask for a concrete example or a walk-through of their implementation.\n"
                f"  Do NOT move to a new skill until this gap is properly addressed."
            )

        # ── Coding mandate for developer roles ──
        is_developer = state.get("is_developer_role", False)
        coding_asked = state.get("coding_questions_asked", 0)
        coding_block = ""
        if is_developer and answer_type not in ("end_requested",):
            if coding_asked == 0 and turn_number >= 1 and answer_type not in ("confused", "wait_requested"):
                coding_block = (
                    "\n💻 CODING MANDATE: This is a developer candidate and NO coding question has been asked yet. "
                    "You MUST ask a hands-on coding problem THIS TURN — set is_coding_question=true. "
                    "Requirements: (1) Use exactly ONE specific problem directly relevant to their stack (or a generic DSA problem from RAG if in a Company Round). "
                    "(2) Match their seniority level — for senior candidates, no trivial problems. "
                    "(3) State the problem clearly in plain English — one paragraph max. "
                    "(4) Do NOT combine a verbal question with the coding task in the same turn. "
                    "This is not optional — asking only verbal questions to a developer for the entire round is a failure."
                )
            elif coding_asked == 1 and turn_number >= 5 and answer_type not in ("confused", "refused", "wait_requested"):
                coding_block = (
                    "\n💻 CODING RECOMMENDATION: Only one coding problem asked so far for a developer candidate. "
                    "If the last 2 answers have been strong, this is a good time to present a second, harder coding problem "
                    "(is_coding_question=true). If the candidate is still struggling, skip this and continue probing concepts verbally."
                )

        # ── Difficulty calibration ──
        perf = state["performance_summary"].lower()
        strong_ct = perf.count("strong")
        weak_ct = perf.count("weak") + perf.count("incorrect")
        if strong_ct > weak_ct + 1:
            difficulty_hint = "Candidate is performing WELL → INCREASE difficulty this turn."
        elif weak_ct > strong_ct + 1:
            difficulty_hint = (
                "Candidate is struggling → break into a simpler sub-question. "
                "Do NOT drop below intermediate level for experienced candidates."
            )
        else:
            difficulty_hint = "Candidate is performing adequately → MAINTAIN current difficulty."

        # ── Seniority floor from context summary ──
        context = state["context_summary"]
        seniority_note = next(
            (ln.strip() for ln in context.splitlines()
             if "seniority" in ln.lower() or "year" in ln.lower()),
            ""
        )
        floor_block = ""
        if seniority_note:
            floor_block = (
                f"\n⚠ SENIORITY FLOOR: {seniority_note}. "
                f"Questions must match this level. NEVER ask trivial entry-level questions "
                f"(e.g. 'what is a variable', 'write a sum function') for a candidate with real experience."
            )

        # ── Type-specific rules ──
        type_rules = _TYPE_RULES.get(state["interview_type"], _TYPE_RULES["technical"])

        # ── Precedence block (Standard vs Specialized) ──
        precedence_block = ""
        company = state.get("company", "")
        has_jd = state.get("has_jd", False)

        if not company or len(company.strip()) < 2:
            # STANDARD ROUND: Resume-First
            precedence_block = "\n⭐ STANDARD ROUND RULE (RESUME-FIRST):\n"
            if has_jd:
                precedence_block += (
                    "  - PRIORITIZE the candidate's ACTUAL projects and tech stack from their resume.\n"
                    "  - Use the JD (Job Description) to filter which resume skills are most relevant to probe deep into.\n"
                    "  - If JD skills are missing in the resume, you MAY bridge to them, but the main goal is to validate what the candidate SAYS they know first.\n"
                )
            else:
                precedence_block += "  - Focus 100% on the candidate's resume, projects, and stated skills.\n"
        else:
            # SPECIALIZED ROUND: JD/company-first (handled in company_round_block)
            pass

          # ── Specialized Company Round Overrides ──
        company = state.get("company", "")
        company_round_block = ""
        if company and len(company.strip()) > 1:
            company_round_block = f"""
🚨 SPECIALIZED {company.upper()} ROUND RULES (KNOWLEDGE ASSESSMENT):
1. JD & COMPANY CONTEXT FOCUS: Your PRIMARY source for technical evaluation is the Job Description and company-specific expectations implied by the round type.
2. RESUME IS SECONDARY: Do NOT make the candidate's resume, projects, or prior companies the main source of questions. Use them only for light seniority calibration when JD topics are exhausted.
3. OPENING TURN (MANDATORY): Start by referencing one JD requirement and ask an introductory knowledge-based question related to that topic.
4. CODING CHALLENGES: Following the intro, transition into technical/coding tasks aligned to the declared round type.
5. INTERVIEW FLOW:
   - Turn 0: Professional greeting + Opening bridge (JD requirement -> 'What is your understanding of...').
    - Turn 1+: Deep technical drills and coding assessments derived from JD + company context.
6. These rules override the default resume-first guidance above for specialized company rounds.
"""

        prompt = f"""You are an expert interviewer conducting a {state['interview_type'].upper()} interview.

{'═' * 60}
CANDIDATE CONTEXT (resume + JD summary):
{context}
Role: {state.get('role', 'Not specified')} | Company: {state.get('company', 'Not specified')}
{'═' * 60}
{company_round_block}
{precedence_block}
INTERVIEW PROGRESS — Turn {turn_number + 1}{f' of {max_q}' if max_q > 0 else ''}:
• Skills covered  : {', '.join(state['skills_covered']) if state['skills_covered'] else 'None yet'}
• Skills remaining: {', '.join(state['skills_remaining'][:10]) if state['skills_remaining'] else 'All covered — deepen existing topics'}
• Difficulty      : {difficulty_hint}

RUNNING PERFORMANCE ASSESSMENT:
{state['performance_summary']}

TIME: {state['time_context'] or 'No time limit.'}
{'═' * 60}

RECENT CONVERSATION:
{recent_text}
{'═' * 60}
{question_limit_block}
{exhausted_topics_block}
{routing_block}
{follow_up_block}
{anti_repeat_block}
{coding_block}
{floor_block}

INTERVIEW TYPE RULES:
{type_rules}
{'═' * 60}

INSTRUCTIONS:
1. Opening turn (turn 0): greet candidate BY NAME from CANDIDATE CONTEXT (if available); ask a single warm-up question about their PRIMARY TECHNOLOGY (or motivation if this is a Company Round).
2. All other turns: apply the routing strategy above FIRST, then pick the best next question.
3. ALWAYS anchor questions to the candidate's stated experience and stack, UNLESS this is a Specialized Company Round, in which case use JD + company context as the primary source and keep resume usage secondary.
4. If JD is present (see 🚨 SPECIALIZED block above): prioritise those technical questions and gaps over generic resume exploration.
5. question = the exact words you say to the candidate. Natural, conversational, interviewer tone.
6. QUESTION DEPTH — STRICTLY ENFORCED:
   - Never ask a question a first-year student could answer trivially if the candidate has real experience.
   - Prefer "how/why/what trade-off" over "what is" — probe understanding, not definitions.
   - Ask for SPECIFICS: "In your [project/company], how did you handle X?" not "how would you handle X?"
   - After a strong answer: go DEEPER — ask about edge cases, performance, alternatives considered.
   - After a weak answer: go SIMPLER on the same topic — break it into a smaller concrete sub-question.
7. ⚠ ONE QUESTION ONLY — STRICTLY ENFORCED:
   - Your "question" field MUST contain exactly ONE question mark.
   - NEVER combine a conceptual question AND a coding task in the same turn.
   - If you have two things to ask, pick the more important one and save the other for the NEXT turn.
   - Violation: "Explain X, and also write a component that does Y" → FORBIDDEN.
8. is_coding_question=true when asking them to WRITE/TYPE actual code OR show written work.
   — MANDATORY: Set to true if you are asking for an implementation, an algorithm, or a specific piece of syntax.
   — FALSE (verbal): High-level conceptual questions (What is X? How does Y work?) where no actual code is needed to explain.
   — RULE: NEVER set is_coding_question=true on a confused / rephrase turn.
9. action=END only when: candidate explicitly requested to end, time is truly up, or all key JD+resume topics thoroughly explored.
   — MANDATORY MINIMUM: For Behavioral, HR, and Problem Solving rounds, do NOT set action=END before at least 5-6 questions have been asked (turn_number >= 5). 
   — If you have run out of specific JD skills but have not reached the minimum turn count, dive deeper into their past projects or ask about their career growth.
10. If the 🛑 END routing block is present above, action MUST be END — this overrides everything else.
"""

        # ── Try fine-tuned interviewer gateway if enabled ────────────────────
        evaluation: Optional[QuestionEvaluation] = None
        in_tok = 0
        out_tok = 0

        if settings.FINE_TUNED_INTERVIEWER_ENABLED:
            gen_start = time.monotonic()
            interviewer_messages = [
                {"role": "system", "content": f"You are an expert interviewer conducting a {state['interview_type'].upper()} interview."},
                {"role": "user", "content": prompt},
            ]
            gateway_result = await gateway_client.generate_interviewer_response(
                messages=interviewer_messages, temperature=0.4, max_tokens=1024
            )
            if gateway_result is not None:
                try:
                    evaluation = QuestionEvaluation.model_validate(gateway_result)
                    gen_duration = int((time.monotonic() - gen_start) * 1000)
                    logger.info(
                        "[Graph] produce_question via gateway (%dms): coding=%s diff=%s",
                        gen_duration, evaluation.next_step.is_coding_question,
                        evaluation.next_step.difficulty,
                    )
                except Exception as parse_exc:
                    logger.warning(
                        "[Graph] Gateway interviewer response failed Pydantic validation: %s — falling back to Gemini",
                        parse_exc,
                    )
                    evaluation = None
            else:
                logger.warning("[Graph] Gateway interviewer returned None — falling back to Gemini")

        # ── Fallback to Gemini ────────────────────────────────────────────────
        if evaluation is None:
            result = await self.structured_llm.ainvoke(prompt)
            evaluation = result["parsed"]
            raw_msg = result.get("raw")

            usage_meta = getattr(raw_msg, "usage_metadata", {}) or {}
            in_tok = usage_meta.get("input_tokens", 0)
            out_tok = usage_meta.get("output_tokens", 0)

        total = state.get("input_tokens", 0) + in_tok + state.get("output_tokens", 0) + out_tok
        print(
            f"[Graph] produce_question: coding={evaluation.next_step.is_coding_question} "
            f"diff={evaluation.next_step.difficulty} ans_type={answer_type} consecutive_non={consecutive_non} "
            f"tokens=in:{in_tok} out:{out_tok} cumulative:{total}"
        )

        new_history = list(history)
        if state.get("last_user_input"):
            new_history.append({"role": "user", "content": state["last_user_input"]})

        new_questions_asked = list(questions_asked) + [evaluation.next_step.question]
        new_coding_count = (
            state.get("coding_questions_asked", 0)
            + (1 if evaluation.next_step.is_coding_question else 0)
        )

        # Hard-force END if question budget is exhausted (belt + suspenders)
        max_q = state.get("max_questions", 0)
        questions_used = len(new_questions_asked)
        budget_exhausted = max_q > 0 and questions_used > max_q

        # Reset confusion streak when we've moved on to a new question
        max_retries = state.get("max_confusion_retries", 2)
        moved_on_from_confusion = (answer_type == "confused" and consecutive_non >= max_retries + 1)

        # Determine reason
        reason = state.get("end_reason", "")
        is_ended = evaluation.decision.action == Action.END or budget_exhausted
        
        if is_ended and not reason:
            if budget_exhausted:
                reason = "BUDGET_EXHAUSTED"
            elif consecutive_disengaged >= _DISENGAGEMENT_THRESHOLD:
                reason = "DISENGAGED"
            elif "TIME UP" in state.get("time_context", ""):
                reason = "TIME_EXCEEDED"
            else:
                reason = "AI_ENDED"

        return {
            "current_evaluation": evaluation,
            "history": new_history,
            "ended": is_ended,
            "end_reason": reason,
            "current_question": evaluation.next_step.question,
            "last_question_was_coding": evaluation.next_step.is_coding_question,
            "questions_asked": new_questions_asked,
            "coding_questions_asked": new_coding_count,
            "turn_number": turn_number + 1,
            "follow_up_hint": "",
            "consecutive_non_answers": 0 if moved_on_from_confusion else consecutive_non,
            "input_tokens": state.get("input_tokens", 0) + in_tok,
            "output_tokens": state.get("output_tokens", 0) + out_tok,
        }

    async def run_turn(self, state: InterviewState) -> InterviewState:
        """Run a single interview turn through the graph."""
        return await self.graph.ainvoke(state)
