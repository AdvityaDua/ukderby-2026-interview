import asyncio
import unittest
from unittest.mock import patch

from app.services.ai.source_interview_planner import (
    SourceInterviewPlanner,
    SourceQuestionCandidate,
    question_fingerprint,
)


def _candidate(question: str, topic: str, source_label: str) -> SourceQuestionCandidate:
    return SourceQuestionCandidate(
        question=question,
        topic=topic,
        source_label=source_label,
        source_url=f"https://example.com/{source_label}",
        fingerprint=question_fingerprint(question),
        evidence=f"evidence for {source_label}",
        is_coding_question=topic in {"System Design", "APIs", "Databases", "Algorithms"},
    )


class SourceInterviewPlannerTest(unittest.TestCase):
    def test_technical_plan_interleaves_company_resume_and_jd_sources(self) -> None:
        planner = SourceInterviewPlanner()
        planner.load_documents = lambda: []  # type: ignore[method-assign]

        base_plan = {
            "topics": ["Python"],
            "question_strategy": {"priority_order": ["company", "resume", "jd"]},
            "evaluation_criteria": {},
            "company_signals": {},
            "token_budget": {},
            "metadata": {},
        }

        company_candidates = [
            _candidate("Company question 1?", "System Design", "company"),
            _candidate("Company question 2?", "APIs", "company"),
        ]
        resume_candidates = [
            _candidate("Resume question 1?", "Databases", "resume"),
            _candidate("Resume question 2?", "Python", "resume"),
        ]
        jd_candidates = [
            _candidate("JD question 1?", "Algorithms", "jd"),
            _candidate("JD question 2?", "Backend", "jd"),
        ]

        async def _run():
            with patch("app.services.ai.langgraph_agent.generate_interview_plan", return_value=base_plan), patch.object(
                planner,
                "build_question_candidates",
                side_effect=[company_candidates, resume_candidates, jd_candidates],
            ):
                return await planner.build_plan(
                    resume_text="Built backend APIs with Python and Redis.",
                    jd_text="Need Python, system design, and APIs.",
                    company_name="Acme Systems",
                    role="Backend Engineer",
                    interview_type="technical",
                    duration_minutes=30,
                )

        plan = asyncio.run(_run())

        self.assertEqual(plan["question_source_mode"], "company_resume_jd_mix")
        self.assertEqual(
            [candidate["question"] for candidate in plan["question_candidates"]],
            [
                "Company question 1?",
                "Resume question 1?",
                "JD question 1?",
                "Company question 2?",
                "Resume question 2?",
                "JD question 2?",
            ],
        )
        self.assertEqual(plan["question_strategy"]["source_mix"], {"company": 3, "resume": 2, "jd": 2})
        self.assertTrue(plan["question_strategy"]["company_first"])
        self.assertFalse(plan["question_strategy"]["jd_fallback_only"])
        self.assertTrue(plan["company_match_found"])


if __name__ == "__main__":
    unittest.main()