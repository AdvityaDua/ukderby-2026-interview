import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings, API_DESCRIPTION, TAGS_METADATA
from app.core.key_manager import key_manager
from app.api.v1.router import api_router
from app.api.v1.ws_interview import ws_router as interview_ws_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"[CONFIG] BACKEND_URL  = {settings.BACKEND_URL}")
    print(f"[CONFIG] HOST:PORT    = {settings.HOST}:{settings.PORT}")
    print(f"[CONFIG] REDIS_URL    = {settings.REDIS_URL}")
    google_ok   = "✓ set" if key_manager.get_gemini_key() else "✗ MISSING"
    groq_ok     = "✓ set" if key_manager.get_groq_key()   else "✗ MISSING"
    deepgram_ok = "✓ set" if settings.DEEPGRAM_KEY else "✗ MISSING"
    print(f"[CONFIG] GOOGLE_API_KEY = {google_ok}")
    print(f"[CONFIG] GROQ_API_KEY   = {groq_ok}")
    print(f"[CONFIG] DEEPGRAM_KEY   = {deepgram_ok}")
    if "localhost" in settings.BACKEND_URL:
        print("[WARNING] BACKEND_URL points to localhost — token reporting will FAIL in production!")
        print("[WARNING] Set BACKEND_URL env var to your production NestJS URL.")
    print("=" * 60)
    yield
    # ── Shutdown ─────────────────────────────────────────────────────────────
    pass

def create_app() -> FastAPI:
    app = FastAPI(
        lifespan=lifespan,
        title="Ai for job Backend",
        description=API_DESCRIPTION,
        version="2.0.0",
        openapi_tags=TAGS_METADATA
    )

    # Add custom JWT Auth Middleware

    # Configure CORS - added after Auth so it wraps it as the outermost layer
    allowed_origins = [origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",") if origin.strip()]
    
    # Standard CORS params
    cors_params = {
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "expose_headers": ["*"],
    }
    
    # Handle wildcard with credentials restriction
    if "*" in allowed_origins:
        cors_params["allow_origins"] = ["*"]
        cors_params["allow_credentials"] = False
    else:
        cors_params["allow_origins"] = allowed_origins
        cors_params["allow_credentials"] = True

    app.add_middleware(CORSMiddleware, **cors_params)

    # Include API Routers
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(interview_ws_router, prefix="/api/v1/interview")

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
