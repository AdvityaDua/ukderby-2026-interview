import asyncio
from typing import List

from .gemini_client import GeminiClient
from .schemas import QuestionEvaluation
from .streaming_session import StreamingInterviewSession
from .token_optimized_agent import InterviewGraph


class CompanyGeminiClient(GeminiClient):
    """Specialized client for company interviews without any RAG dependency."""

    async def generate_question(
        self,
        history: List[dict],
        context_summary: str,
        interview_type: str = "technical",
        time_context: str = "",
        company_description: str = "",
        round_type: str = "",
    ) -> QuestionEvaluation:
        prompt_history = "INTERVIEW HISTORY:\n"
        for turn in history:
            role = turn["role"]
            content = turn["content"]
            prompt_history += f"{role.upper()}: {content}\n"

        effective_round = round_type or interview_type or "technical"
        time_section = f"TIME MANAGEMENT:\n{time_context}\n" if time_context else ""

        company_context = ""
        if company_description or effective_round:
            company_context = f"""
COMPANY CONTEXT:
Company Description: {company_description if company_description else "(Generic company context)"}
Interview Round Type: {effective_round}

Round guidance:
- DSA rounds: focus on data structures, algorithms, complexity, and trade-offs.
- System design rounds: focus on architecture, scale, bottlenecks, and trade-offs.
- Behavioral rounds: focus on STAR-style depth and role alignment.
"""

        prompt = f"""
You are an elite senior principal interviewer conducting a rigorous {effective_round} interview.

CONTEXT SUMMARY:
{context_summary}

{company_context}

RULES:
- Ask ONE question at a time.
- Be rigorous and specific to the role, company context, and round type.
- If JD is present, prioritize JD requirements and skill gaps first.
- Do not ask generic filler questions.
- If the candidate struggles, simplify the same topic before switching.
- If the candidate performs well, increase depth and difficulty.
- Set is_coding_question=true only when code writing is required.

{time_section}

{prompt_history}

Evaluate the latest response and generate the next JSON QuestionEvaluation.
"""

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": QuestionEvaluation,
            },
        )
        return QuestionEvaluation.model_validate_json(response.text)


class CompanyInterviewSession(StreamingInterviewSession):
    """Company-specific interview session without RAG."""

    def __init__(self, client: CompanyGeminiClient):
        super().__init__(client)
        self.client = client
        self.graph_engine = InterviewGraph(api_key=client.api_key)
        self.company_description = ""
        self.round_type = ""

    async def initialize_session(
        self,
        user_id: str,
        session_id: str,
        resume_text: str,
        jd_text: str,
        interview_type: str = "technical",
        role: str = "",
        company: str = "",
        duration: int = 0,
        candidate_name: str = "",
    ):
        self.company_description = company or ""
        self.round_type = interview_type or "technical"

        await super().initialize_session(
            user_id,
            session_id,
            resume_text,
            jd_text,
            interview_type,
            role,
            company,
            duration,
            candidate_name,
        )

        print(
            f"[CompanySession] Initialized JD-driven company interview for '{self.company_description}' "
            f"({self.round_type} round)"
        )
