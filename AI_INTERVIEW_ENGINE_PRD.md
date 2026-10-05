# AI Interview Engine PRD (No-Embedding Architecture)

# 1. Overview
This document defines the architecture and requirements for an AI-powered interview engine that conducts adaptive interviews without embeddings. The runtime model follows a one-time initialization phase and a strict two-node LangGraph execution loop aligned with the current implementation in app/services/ai/interview_graph.py.

Design intent:
- Preserve quality while controlling token growth
- Keep each turn latency low
- Enforce deterministic interview state transitions
- Maintain accurate cumulative token accounting per session

## 2. Objectives
- Simulate real company-specific interviews
- Avoid token explosion and embedding usage
- Maintain fast response time (<4s per turn)
- Adapt questions based on candidate performance
- Track cumulative input and output tokens for every interview session
- Support deterministic end conditions (user exit, disengagement, budget exhausted, etc.)

## 3. System Architecture

### Phase 1: Initialization (Run Once)

**Inputs:**
- User Resume
- Job Description (JD)
- Company Name

**Steps:**
- Parse resume and extract skills
- Analyze JD requirements
- Perform web search for company interview patterns
- Generate structured Interview Plan
- Build initial InterviewState object used by LangGraph
- Set question budget and per-topic cap based on interview duration
- Initialize token counters (input_tokens=0, output_tokens=0)

**Output:**
- interview_plan (JSON)
- initialized InterviewState (runtime state payload)

### Phase 2: Execution Loop

The execution loop is implemented as:
- Node 1: evaluate_turn
- Node 2: produce_question

Per turn flow:
1. Evaluate candidate response type and quality
2. Update performance summary, skill coverage, and routing counters
3. Generate next question (or closing statement)
4. Return updated InterviewState

State transition path:
- entry point -> evaluate_turn -> produce_question -> END (single-turn invocation)

The application invokes run_turn repeatedly across interview turns.

## 4. Interview Plan Structure
```json
{
  "company": "",
  "role": "",
  "topics": [],
  "question_strategy": {
    "interview_type": "technical|behavioral|problem|hr",
    "duration_minutes": 30,
    "max_questions": 13,
    "max_questions_per_topic": 2,
    "max_confusion_retries": 2,
    "difficulty_progression": "adaptive"
  },
  "evaluation_criteria": {
    "answer_type": [
      "genuine_answer",
      "confused",
      "refused",
      "off_topic",
      "incomplete",
      "wait_requested",
      "coding_requested",
      "end_requested",
      "not_applicable"
    ],
    "answer_quality": ["strong", "adequate", "weak", "incorrect", "not_applicable"],
    "follow_up_policy": "only for genuine weak/incorrect answers"
  }
}
```

## 5. Runtime State Contract (Aligned With interview_graph.py)

InterviewState must include the following keys:

```json
{
  "history": [],
  "performance_summary": "",
  "context_summary": "",
  "has_jd": true,
  "skills_remaining": [],
  "skills_covered": [],
  "questions_asked": [],
  "last_user_input": null,
  "current_question": "",
  "current_evaluation": null,
  "follow_up_hint": "",
  "interview_type": "technical",
  "time_context": "",
  "role": "",
  "company": "",
  "is_developer_role": false,
  "coding_questions_asked": 0,
  "last_question_was_coding": false,
  "turn_number": 0,
  "last_answer_type": "not_applicable",
  "consecutive_non_answers": 0,
  "ended": false,
  "end_reason": "",
  "max_questions": 0,
  "max_questions_per_topic": 0,
  "topic_question_counts": {},
  "max_confusion_retries": 2,
  "consecutive_disengaged": 0,
  "input_tokens": 0,
  "output_tokens": 0
}
```

Important runtime notes:
- No embedding vectors are stored or queried in this architecture.
- Token counters are cumulative across all turns.
- question budget and per-topic limits are enforced in generation logic.

## 6. Per-Turn Output Contract

Each run_turn call returns an updated InterviewState, including:
- current_question
- current_evaluation (QuestionEvaluation structured object)
- ended (boolean)
- end_reason (USER_EXIT | AI_ENDED | TIME_EXCEEDED | DISENGAGED | BUDGET_EXHAUSTED)
- turn_number
- questions_asked
- coding_questions_asked
- follow_up_hint
- input_tokens and output_tokens (updated cumulatively)

QuestionEvaluation shape (derived from schemas.py):
- meta
- last_answer_evaluation
- coverage_update
- decision
: includes action (continue or end)
- next_step
: includes question, target_skill, difficulty, and is_coding_question

## 7. Token Accounting and Cost Observability

Token accounting rules:
- evaluate_turn reads usage_metadata from model response and increments:
  - input_tokens += evaluate_turn.input_tokens
  - output_tokens += evaluate_turn.output_tokens
- produce_question repeats the same accumulation.
- Session total tokens = input_tokens + output_tokens.

Recommended telemetry per turn:
- session_id
- turn_number
- evaluate_turn in/out tokens
- produce_question in/out tokens
- cumulative input/output tokens
- latency_ms per node and total turn latency

Hard requirements:
- Never reset token counters mid-session
- Persist counters in Redis/session store for crash-safe recovery
- Expose counters in API response for frontend/admin monitoring

## 8. Routing and Termination Rules

Routing inputs:
- last_answer_type
- consecutive_non_answers
- consecutive_disengaged
- question budget usage
- follow_up_hint

Termination triggers:
- Candidate requested end -> USER_EXIT
- Question budget exhausted -> BUDGET_EXHAUSTED
- Disengagement threshold reached -> DISENGAGED
- Explicit interviewer decision -> AI_ENDED
- Time hard-stop policy -> TIME_EXCEEDED

Behavioral constraints:
- Anti-repetition through questions_asked list
- Per-topic cap via topic_question_counts
- Confusion retry cap via max_confusion_retries
- Developer roles require coding-question enforcement in strategy

## 9. Data Sources

### GitHub Repositories
- https://github.com/liquidslr/interview-company-wise-problems
- https://github.com/snehasishroy/leetcode-companywise-interview-questions
- https://github.com/krishnadey30/LeetCode-Questions-CompanyWise
- https://github.com/realabbas/big-companies-interview-questions
- https://github.com/jwasham/coding-interview-university

### Web Sources
- LeetCode (company tags)
- GeeksforGeeks (interview experiences)
- Glassdoor (interview insights)
- Reddit (r/leetcode)

## 10. Functional Requirements
- One-time plan generation
- No embeddings usage
- Adaptive questioning
- Topic tracking
- Performance evaluation
- Structured evaluation output parsing (no regex parsing)
- Token accounting at node level and session level
- Deterministic per-turn state mutation
- Question anti-repetition safeguards

## 11. Non-Functional Requirements
- Low latency (<4s median per turn; stretch goal)
- Scalable for concurrent users
- Cache company data
- Observability: per-node latency + token metrics
- Fault tolerance: non-fatal evaluation fallback on LLM exceptions

## 12. Tech Stack
- LangGraph (state orchestration)
- LLM (Gemini)
- Redis (caching)
- FastAPI (API and websocket runtime)
- Pydantic schemas for structured outputs

## 13. API/Runtime Integration Requirements

Engine interface:
- run_turn(state: InterviewState) -> InterviewState

State storage:
- Persist InterviewState in Redis keyed by session_id
- Restore state on reconnects and continue from last turn

Concurrency:
- One active turn execution per session_id
- Reject or queue overlapping turn requests

Frontend payload expectations:
- question text
- is_coding_question flag
- action (continue/end)
- end_reason when ended=true
- token counters (input_tokens, output_tokens, total_tokens)

## 14. Future Improvements
- Add memory compression
- Add scoring dashboard
- Add mock interview feedback reports
- Add token budget alerts and auto-throttling
- Add per-company cache freshness and background refresh jobs
