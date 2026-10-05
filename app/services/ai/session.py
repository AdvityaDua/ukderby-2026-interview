import asyncio
from typing import List, Optional, Dict
from .gemini_client import GeminiClient
from .schemas import QuestionEvaluation, FinalEvaluation, Action, NextStepType

class InterviewSession:
    def __init__(self, client: GeminiClient):
        self.client = client
        self.history: List[dict] = []
        self.context_summary: str = ""
        self.poor_answer_streak: int = 0
        self.questions_asked: int = 0
        self.is_active: bool = True
        self.current_turn_evaluation: Optional[QuestionEvaluation] = None

    async def initialize_session(self, resume_text: str, jd_text: str):
        """Generates context summary to start the session."""
        self.context_summary, _ = await self.client.summarize_context(resume_text, jd_text)
    
    async def process_user_input(self, user_response: Optional[str] = None) -> QuestionEvaluation:
        """
        Processes the user's response (or empty if it's the start), 
        updates history, and gets the next question/decision from Gemini.
        """
        if user_response:
            self.history.append({"role": "user", "content": user_response})
        
        # Generate next step from AI
        evaluation = await self.client.generate_question(self.history, self.context_summary)
        self.current_turn_evaluation = evaluation
        
        # Update internal state based on evaluation
        if evaluation.last_answer_evaluation and evaluation.last_answer_evaluation.signal == "weak":
            self.poor_answer_streak += 1
        elif evaluation.last_answer_evaluation:
            self.poor_answer_streak = 0 # partial reset or full reset on strong
            
        decision = evaluation.decision
        
        if decision.action == Action.END:
            self.is_active = False
        else:
            if evaluation.next_step.type != NextStepType.NONE:
                question = evaluation.next_step.question
                self.history.append({"role": "model", "content": question})
                self.questions_asked += 1
            else:
                 # Potentially logic to handle "continue" but no "next_step" (unlikely)
                 pass
                 
        return evaluation

    async def generate_final_report(self) -> FinalEvaluation:
        """Generates the final feedback report."""
        feedback, usage = await self.client.generate_feedback(self.history, self.context_summary)
        return feedback
