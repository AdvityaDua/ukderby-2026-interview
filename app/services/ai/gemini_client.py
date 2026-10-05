import asyncio
import os
from google import genai
from google.genai import types
from typing import List, Optional
from .schemas import QuestionEvaluation, FinalEvaluation, MidInterviewSnapshot
from app.core.key_manager import key_manager

class GeminiClient:
    def __init__(self, api_key: str = None, model_name: str = None):
        if not api_key:
            api_key = key_manager.get_gemini_key() or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found — set it in env or configure it via the admin panel")
        
        self.client = genai.Client(api_key=api_key)
        self.api_key = api_key
        self.model_name = model_name or key_manager.get_gemini_model()

    async def _with_retry(self, func, *args, **kwargs):
        """Helper to retry AI calls on 503 (high demand) errors with model fallback."""
        # Fallback model chain: if primary model is overloaded, try these in order
        FALLBACK_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash"]
        
        # First attempt: try with primary model (up to 2 retries)
        primary_model = self.model_name
        for attempt in range(2):
            try:
                return await asyncio.to_thread(func, *args, **kwargs)
            except Exception as e:
                error_str = str(e).upper()
                if "503" in error_str or "UNAVAILABLE" in error_str:
                    if attempt < 1:
                        print(f"[GeminiClient] 503 on {primary_model}. Retrying in 2s (attempt {attempt+1}/2)...")
                        await asyncio.sleep(2.0)
                        continue
                else:
                    raise e
        
        # Primary model exhausted — try fallback models
        for fallback in FALLBACK_MODELS:
            if fallback == primary_model:
                continue  # skip if same as primary
            print(f"[GeminiClient] ⚠️ Primary model '{primary_model}' overloaded. Falling back to '{fallback}'...")
            # Temporarily swap model for this call
            original_model = self.model_name
            self.model_name = fallback
            try:
                # Re-call the function — but we need to inject model name for direct calls
                result = await asyncio.to_thread(func, *args, **kwargs)
                print(f"[GeminiClient] ✅ Fallback to '{fallback}' succeeded.")
                return result
            except Exception as e:
                error_str = str(e).upper()
                if "503" in error_str or "UNAVAILABLE" in error_str:
                    print(f"[GeminiClient] '{fallback}' also overloaded, trying next...")
                    continue
                raise e
            finally:
                self.model_name = original_model
        
        raise Exception(f"All models overloaded (503). Last tried: {FALLBACK_MODELS}. Please try again shortly.")

    def extract_skills(self, context_summary: str, interview_type: str, role: str, jd_text: str = "") -> list:
        """Keyword-based skill extraction."""
        _TOPIC_KEYWORDS = {
            "Python": ["python", "pandas", "numpy", "fastapi", "flask", "django"],
            "JavaScript": ["javascript", "typescript", "node.js", "react", "next.js", "vue"],
            "System Design": ["system design", "microservices", "scalability", "distributed", "architecture"],
            "Databases": ["sql", "postgres", "mysql", "mongodb", "redis", "database", "nosql"],
            "APIs": ["api", "rest", "graphql", "websocket", "grpc", "http"],
            "Testing": ["test", "pytest", "jest", "unit testing", "integration testing", "tdd"],
            "DevOps": ["docker", "kubernetes", "ci/cd", "aws", "gcp", "azure", "terraform"],
            "DSA": ["algorithm", "data structure", "leetcode", "dynamic programming", "graph", "tree", "linked list"],
            "Machine Learning": ["machine learning", "ml", "deep learning", "pytorch", "tensorflow", "transformer", "llm", "rag", "nlp"],
            "Behavioral": ["leadership", "teamwork", "conflict", "communication", "ownership", "collaboration"],
            "Problem Solving": ["problem solving", "analytical", "case study", "logical reasoning"],
            "Product Thinking": ["product", "trade-off", "user impact", "metrics", "stakeholder"],
        }
        
        combined = f"{context_summary}\n{jd_text}\n{role}".lower()
        skills = [topic for topic, kws in _TOPIC_KEYWORDS.items() if any(kw in combined for kw in kws)]
        return skills[:12] if skills else ["Problem Solving", "Communication", "Technical Knowledge"]

    async def summarize_context(self, resume_text: str, jd_text: str, interview_type: str = "technical", role: str = "", company: str = "", candidate_name: str = "") -> str:
        type_focus = {
            "technical": "Focus on: technical skills, programming languages, frameworks, system design experience, algorithms knowledge.",
            "behavioral": "Focus on: teamwork experiences, leadership examples, conflict resolution, communication skills, STAR-method scenarios.",
            "problem": "Focus on: problem-solving abilities, analytical thinking, case study scenarios, logical reasoning, creative solutions.",
            "hr": "Focus on: cultural fit, career goals, salary expectations, work-life balance views, motivation and passion.",
        }.get(interview_type, "Focus on: overall candidate assessment.")

        has_jd = bool(jd_text and jd_text.strip())
        jd_block = f"""
        JOB DESCRIPTION (HIGH PRIORITY — questions must map to these requirements first):
        {jd_text}
        """ if has_jd else "JOB DESCRIPTION: Not provided."

        jd_instruction = (
            "🚨 MANDATORY JD BIAS: A Job Description is provided. You MUST prioritize validating the candidate "
            "against the EXPLICIT requirements of this JD. "
            "1. Focus 70% of the interview on core tech/skills listed in the JD. "
            "2. Identify 'Skill Gaps': specific requirements in the JD that are MISSING or weak in the resume. "
            "3. The interviewer MUST probe these gaps first to see if the candidate actually possesses the skills. "
            "4. If the JD requires a specific seniority (e.g., Lead), evaluate them against that bar specifically."
        ) if has_jd else (
            "No Job Description provided — conduct a standard holistic interview based on the resume and general role expectations."
        )

        is_study_participant = "Corporate Study Participant" in resume_text
        
        prompt = f"""
        You are an expert recruiter preparing for a {interview_type.upper()} interview round.
        
        TASK:
        1. Extract the candidate's FULL NAME directly from the RESUME. If not found, use "{candidate_name or "the candidate"}".
        {"2. IGNORE RESUME ANALYSIS: This is a standardized study session. Focus purely on the JD and Company profile." if is_study_participant else "2. Analyze the Resume and Job Description below."}
        3. Create a dense, information-rich summary for the interviewer.

        {jd_instruction}

        RESUME:
        {resume_text}

        {jd_block}
        
        {f'The role is: {role}' if role else ''}
        {f'The company is: {company}' if company else ''}
        
        {type_focus}
        
        Output in the following sections ONLY:
        IDENTIFIED NAME: [Strictly extract from resume]
        Candidate Profile
        Seniority Estimate
        JD Required Skills: [Only if JD provided — list every explicit skill/tool/language the JD demands]
        Matched Skills: [Skills present in BOTH resume and JD — mark these as HIGH PRIORITY to test]
        Missing Skills: [Skills the JD requires but resume does NOT show — flag as gaps]
        Key Projects and Experiences
        Interview Focus Areas (tailored for {interview_type} round — JD requirements come first)
        """
        
        response = await self._with_retry(
            self.client.models.generate_content,
            model=self.model_name,
            contents=prompt
        )
        
        # Return both text and usage info
        usage = {"input_tokens": 0, "output_tokens": 0}
        if hasattr(response, 'usage_metadata'):
            usage["input_tokens"] = getattr(response.usage_metadata, 'prompt_token_count', 0)
            usage["output_tokens"] = getattr(response.usage_metadata, 'candidates_token_count', 0)
            print(f"[GeminiClient] summarize_context usage: in={usage['input_tokens']}, out={usage['output_tokens']}")
        
        return response.text, usage

    async def generate_question(self, history: List[dict], context_summary: str, interview_type: str = "technical", time_context: str = "") -> QuestionEvaluation:
        prompt_history = "INTERVIEW HISTORY:\n"
        for turn in history:
            role = turn['role']
            content = turn['content']
            prompt_history += f"{role.upper()}: {content}\n"
        
        type_rules = {
            "technical": """
    INTERVIEW TYPE: TECHNICAL ROUND
    - Ask about data structures, algorithms, system design, coding patterns.
    - Present coding problems and ask the candidate to explain their approach.
    - You CAN ask the candidate to write actual code — they have a code editor available.
      When you want them to code, say something like "Please write the code for this in the editor and submit it."
    - Dig deep into technical concepts. Don't accept surface-level answers.
    - If the candidate shares code, review it for correctness, efficiency, and style.
    - Cover topics from the JD: specific languages, frameworks, and tools.
    """,
            "behavioral": """
    INTERVIEW TYPE: BEHAVIORAL ROUND
    - Use the STAR method (Situation, Task, Action, Result) framework.
    - Ask about teamwork, leadership, conflict resolution, and handling pressure.
    - Probe for specific real-world examples, not hypotheticals.
    - Evaluate communication clarity and self-awareness.
    - Ask follow-up questions to verify depth and authenticity of examples.
    """,
            "problem": """
    INTERVIEW TYPE: PROBLEM SOLVING ROUND
    - Present real-world case studies and logical reasoning challenges.
    - Assess analytical thinking and structured problem decomposition.
    - Evaluate how the candidate breaks down ambiguous problems.
    - Test estimation skills and ability to handle incomplete information.
    - Focus on the thought process, not just the final answer.
    - You CAN ask them to implement their solution in code — they have a code editor available.
    """,
            "hr": """
    INTERVIEW TYPE: HR ROUND
    - Discuss career goals, motivation, and cultural fit.
    - Ask about work-life balance expectations and team collaboration style.
    - Evaluate communication skills, professionalism, and enthusiasm.
    - Keep the tone warm and conversational but insightful.
    - Assess alignment between the candidate's values and the company's culture.
    """
        }.get(interview_type, "")

        # Build time management section if time context exists
        time_section = ""
        if time_context:
            time_section = f"""
        TIME MANAGEMENT:
        {time_context}
        - Pace your questions to fit the interview within the target duration.
        - If remaining time < 3 minutes: Begin wrapping up. Ask a final question or make closing remarks.
        - If remaining time < 1 minute or OVER TIME: Set action to END. Provide a brief closing summary.
        - You may exceed the target by ±2 minutes if you are mid-topic — do NOT cut off abruptly.
        - Prioritize the most important skills from the JD when time is limited.
        """

        prompt = f"""
        You are an expert interviewer conducting a realtime {interview_type} interview.
        
        You must internally follow these steps:
        1. Evaluate the candidate's last answer
        2. Decide whether to continue or end
        3. Decide next question strategy
        4. Generate ONE question

        Do NOT output intermediate reasoning.
        
        CONTEXT SUMMARY:
        {context_summary}
        {type_rules}
        {time_section}
        
        YOUR GOAL:
        1.  Evaluate the candidate's last answer (if any).
        2.  Decide whether to continue or end it.
        3.  Generate the next question if continuing.
        
        RULES:
        -   Start with a greeting and an initial question if the history is empty. IMPORTANT: In your very first message, you MUST greet the candidate by their actual name if it is available in the CONTEXT SUMMARY (e.g. "Hi [Name], thank you for joining...").
        -   Be professional, encouraging, but rigorous.
        -   🚨 **JD PRIORITY**: If the CONTEXT SUMMARY includes a Job Description, you MUST prioritize questions about the technologies and responsibilities mentioned there. Probe for identified 'Skill Gaps' early in the session.
        -   If the candidate struggles, offer a small hint or move to a simpler related topic.
        -   If the candidate answers well, increase difficulty.
        -   Set 'is_coding_question' in the 'next_step' to true if and only if the next question requires the candidate to write or modify code in the provided code editor. Otherwise, set it to false.

        
        OUTPUT FORMAT:
        You must return a JSON object strictly adhering to the schema.

        {prompt_history}
        
        Evaluate the last user response (if any) and generate the next step.
        """

        response = await self._with_retry(
            self.client.models.generate_content,
            model=self.model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": QuestionEvaluation,
            },
        )
        return QuestionEvaluation.model_validate_json(response.text)

    async def generate_feedback(self, history: List[dict], context_summary: str) -> FinalEvaluation:
        prompt_history = "INTERVIEW HISTORY:\n"
        user_msg_count = 0
        for turn in history:
            role = turn['role']
            if role == 'user' or role == 'candidate':
                user_msg_count += 1
            content = turn['content']
            prompt_history += f"{role.upper()}: {content}\n"
            
        if user_msg_count == 0:
            raise ValueError("Insufficient Data: The candidate did not answer any questions during the interview. No meaningful evaluation can be generated.")
            
        prompt = f"""
        You are an expert technical interviewer. The interview has ended.
        Provide a comprehensive final evaluation of the candidate.
        
        CONTEXT SUMMARY:
        {context_summary}
        
        {prompt_history}
        
        IMPORTANT INSTRUCTIONS:
        - DO NOT MAKE UP OR HALLUCINATE ANY DATA. ONLY EVALUATE BASED ON THE ACTUAL CONVERSATION HISTORY. If the candidate answered very little, strictly mention that the candidate did not provide enough details, and score them accordingly.
        - You MUST include EVERY SINGLE question you asked during the interview as a separate entry in question_wise_analysis.
        - Do NOT consolidate or merge questions. Each question gets its own entry.
        - Number the question_ids sequentially starting from 1.
        - The overall_score in summary should be out of 100. IMPORTANT: Do NOT simply take the mathematical average of individual question scores.
        - The overall_score should be a holistic assessment of the candidate's readiness for the role.
        - Prioritize core technical skills and problem-solving over greetings or simple introductory questions.
        - If the candidate performed exceptionally on difficult topics but missed minor points, the overall_score should reflect that high level of skill.
        - Conversely, if the candidate failed core role requirements from the JD, the overall_score should reflect that lack of readiness, even if they answered other questions correctly.
        - Each individual question score should be out of 10.

        - For the IMPROVEMENT PLAN, you MUST provide:
            1. immediate_actions: Things to do in the next 24-48 hours.
            2. 1_week_plan: Focused study and practice for the next 7 days.
            3. 1_month_plan: Longer-term skill development for the next 30 days.
        - You MUST populate EVERY SINGLE FIELD in the JSON schema. Do not leave any field as null or empty unless the schema explicitly allows it (and even then, prefer providing data).
        
        Generate the final detailed evaluation report.
        """
        
        response = await self._with_retry(
            self.client.models.generate_content,
            model=self.model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": FinalEvaluation,
            },
        )
        
        # Track token usage for final feedback generation
        usage = {"input_tokens": 0, "output_tokens": 0}
        if hasattr(response, 'usage_metadata'):
            usage["input_tokens"] = getattr(response.usage_metadata, 'prompt_token_count', 0)
            usage["output_tokens"] = getattr(response.usage_metadata, 'candidates_token_count', 0)
            print(f"[GeminiClient] generate_feedback usage: in={usage['input_tokens']}, out={usage['output_tokens']}")
        
        evaluation = FinalEvaluation.model_validate_json(response.text)
        return evaluation, usage

    def connect_live(self, system_instruction: str):
        # Configuration aligning with user snippet where possible, while maintaining our features
        config = {
            "response_modalities": ["AUDIO", "TEXT"],
            "system_instruction": system_instruction,
            # Keeping voice config for now as it's standard for Bidi
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {
                        "voice_name": "Puck"
                    }
                }
            }
        }
        return self.client.aio.live.connect(model=self.model_name, config=config)

    async def generate_chat_stream(self, history: List[dict], context_summary: str):
        """
        Streams regular text response based on history and context.
        Bypasses JSON structure for conversation flow to enable standard streaming.
        """
        # Construct prompt similar to live session but designed for text-only stream
        prompt_history = "INTERVIEW HISTORY:\n"
        for turn in history:
            role = turn['role']
            content = turn['content']
            prompt_history += f"{role.upper()}: {content}\n"
            
        system_instruction = f"""
        You are an expert technical interviewer.
        
        CONTEXT SUMMARY:
        {context_summary}
        
        GOAL:
        Ask questions to evaluate the candidate based on the Job Description.
        One question at a time.
        Be professional but conversational.
        If the candidate answers well, go deeper.
        If they struggle, hint or move on.
        
        Start by introducing yourself and asking the first question if history is empty.
        """
        
        # We append the history to the contents or manage it via chat session.
        # Ideally, we used ephemeral stateless calls before.
        # Let's use generaate_content_stream with the history in the prompt 
        # (or list of content objects if we wanted to be fancy, but text prompt is fine).
        
        full_prompt = f"{system_instruction}\n\n{prompt_history}\n\nCandidate just replied. Respond to the candidate."

        async for chunk in await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=full_prompt,
        ):
            if chunk.text:
                yield chunk.text
