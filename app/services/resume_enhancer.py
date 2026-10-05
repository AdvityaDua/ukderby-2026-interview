import json
import uuid
from typing import Optional, List, Any
from google import genai
from google.genai import types
import httpx
from app.core.config import settings
from app.core.key_manager import key_manager
from app.models.resume_schemas import ResumeAnalysisRequest, ResumeBuilderResponse

class ResumeEnhancerService:
    def __init__(self):
        pass

    def _get_client(self):
        api_key = key_manager.get_gemini_key() or settings.GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not configured.")
        return genai.Client(api_key=api_key)
        
    async def _report_usage(self, user_id: str, session_id: str, input_tokens: int, output_tokens: int):
        """Report token usage to the NestJS analytics backend."""
        in_cost  = (input_tokens  / 1_000_000) * 0.10
        out_cost = (output_tokens / 1_000_000) * 0.40
        payload  = {
            "userId": user_id,
            "sessionId": session_id,
            "model": key_manager.get_gemini_model(),
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
            "costUsd": in_cost + out_cost,
            "inputCostUsd": in_cost,
            "outputCostUsd": out_cost,
            "subscriptionStatus": "free",
            "source": "resume",
            "interviewType": "",
        }
        url = f"{settings.BACKEND_URL}/analytics/ai-usage"
        print(f"[Resume] POSTing usage to {url}  payload={payload}")
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(url, json=payload, timeout=5.0)
            print(f"[Resume] analytics POST status={r.status_code} body={r.text[:200]}")
        except Exception as e:
            print(f"[Resume] Failed to report usage: {type(e).__name__}: {e}")

    async def enhance_resume(self, request: ResumeAnalysisRequest, user: dict = None, raw_text: str = "") -> ResumeBuilderResponse:
        client = self._get_client()
            
        # Extract data directly or from nested resume object
        rd = request.resume if request.resume else request
        an = getattr(rd, 'analytics', None)
        en = getattr(rd, 'enhancement', None)
        
        # Build optimized prompt context
        context = {
            "cv_quality": {
                "score": an.cv_quality.overall_score if an and an.cv_quality else 0,
                "details": [s.model_dump() for s in an.cv_quality.subscores] if an and an.cv_quality and an.cv_quality.subscores else []
            },
            "jd_match": {
                "score": an.jd_match.overall_score if an and an.jd_match else 0,
                "details": [s.model_dump() for s in an.jd_match.subscores] if an and an.jd_match and an.jd_match.subscores else []
            },
            "tailored_data": en.model_dump() if en else {},
            "user_info": user if user else {}
        }
        
        prompt = f"""You are an expert resume writer. Using the analysis data and the raw resume text below, generate a professional, high-impact resume.
        
        RAW RESUME TEXT:
        {raw_text}
        
        ANALYSIS DATA:
        {json.dumps(context, indent=2)}
        
        TASK:
        Return a JSON object matching the `ResumeBuilderResponse` schema.
        - Structure `personal_info` accurately. Use the `user_info` email if provided. DO NOT use generic or dummy names (like "Akash Bargoti" or "John Doe") if personal info is not provided; leave the name as empty string or "[Your Name]" instead. Extract correct phone, GitHub, LinkedIn, and education directly from RAW RESUME TEXT.
        - Categorize `skills` into `frontend`, `backend`, and `tools_cloud`.
        - Provide `experience` descriptions as impactful bullet point lists. Use EXACT history from RAW RESUME TEXT.
        - Ensure `projects` have `technologies` (list) and `highlights` (bullets).
        - DO NOT hallucinate fake education, certifications, or experience. If the data is missing from ANALYSIS DATA or RAW RESUME TEXT, return empty lists for those fields.
        """
        
        import time
        retries = 3
        response = None
        for attempt in range(retries):
            try:
                response = client.models.generate_content(
                    model=key_manager.get_gemini_model(),
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ResumeBuilderResponse,
                        temperature=0.1,
                    )
                )
                break
            except Exception as e:
                if "503" in str(e) and attempt < retries - 1:
                    print(f"[Resume] Gemini API 503 error, retrying in 5 seconds... ({attempt+1}/{retries})")
                    time.sleep(5)
                else:
                    raise
        
        # Report token usage — await directly since we're already in async context
        um = getattr(response, 'usage_metadata', None)
        print(f"[Resume] usage_metadata raw: {um}  attrs={dir(um) if um else 'N/A'}")
        input_tokens  = getattr(um, 'prompt_token_count',     0) if um else 0
        output_tokens = getattr(um, 'candidates_token_count', 0) if um else 0
        # Fallback: some SDK versions use total_token_count when candidates_token_count is 0
        if output_tokens == 0 and um:
            output_tokens = getattr(um, 'total_token_count', 0) - input_tokens
        print(f"[Resume] Gemini usage: in={input_tokens}, out={output_tokens}  user={user}")

        user_id    = str(user.get('sub') or user.get('_id') or user.get('id', 'anonymous')) if user else 'anonymous'
        session_id = f"resume_{user_id}_{uuid.uuid4().hex[:10]}"
        await self._report_usage(user_id, session_id, input_tokens, output_tokens)

        # Parse and validate the response
        return ResumeBuilderResponse.model_validate_json(response.text)

resume_enhancer_service = ResumeEnhancerService()
