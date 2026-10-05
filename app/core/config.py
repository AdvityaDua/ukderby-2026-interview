from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    DATABASE_URL: str = ''
    PROJECT_NAME: str = "Prod-Grade FastAPI Backend"
    JWT_ACCESS_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    PORT: int = 8001
    HOST: str = "0.0.0.0"
    # Optional — keys are fetched from NestJS admin DB at runtime; env vars are fallback
    GOOGLE_API_KEY: Optional[str] = None
    DEEPGRAM_KEY: str
    GROQ_API_KEY: Optional[str] = None
    REDIS_URL: str = "redis://localhost:6379"
    PISTON_URL: str = "http://localhost:2000"
    BACKEND_URL: str = "http://localhost:3000"  # Used for reporting token usage to NestJS analytics
    NEST_API_BASE_URL: Optional[str] = None  # If unset, BACKEND_URL is used
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:3001,http://localhost:5173,http://localhost:5174,https://aiforjob.ai,https://www.aiforjob.ai"

    # Fine-tuned LLM gateway configuration
    FINE_TUNED_GATEWAY_URL: str = "https://aiforjob-fine-tune-gateway-66976dwa2.brevlab.com"
    FINE_TUNED_INTERVIEWER_ENABLED: bool = False
    FINE_TUNED_EVALUATOR_ENABLED: bool = False

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

API_DESCRIPTION = """
🚀 **Ai for job API**

This API powers the core AI models, WebSockets for the Interview module, and Resume Evaluation engine. 
It replaces the legacy backends with a robust, production-grade FastAPI architecture.

### Key Features:
* **Interview WebSockets**: Handles full duplex transcription and conversation with Gemini.
* **Resume Enhancer**: Powered by Groq for ultra-fast CV/JD analysis and scoring.
* **Authentication**: Native JWT validation syncing perfectly with the frontend & NestJS.
"""

TAGS_METADATA = [
    {"name": "Resume", "description": "Core Resume Enhancer using Gemini 2.5 Pro for full generation."},
    {"name": "CV Evaluation", "description": "Legacy routes for scoring and fit-index using Groq."},
    {"name": "Uploads", "description": "Extract text from DOCX, PDF, txt, and images using Tesseract/pdfplumber."},
    {"name": "Evaluation", "description": "Direct text evaluation bridging CV and JD."},
]
