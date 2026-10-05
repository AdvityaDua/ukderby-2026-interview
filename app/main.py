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
    # Fetch active API keys from NestJS admin DB (falls back to env vars)
    # Then start a background loop that re-fetches every 60s so admin key
    # changes are picked up without restarting the service.
    await key_manager.refresh(settings.BACKEND_URL)
    key_manager.start_auto_refresh(settings.BACKEND_URL)
    google_ok   = "✓ set" if key_manager.get_gemini_key() else "✗ MISSING"
    groq_ok     = "✓ set" if key_manager.get_groq_key()   else "✗ MISSING"
    deepgram_ok = "✓ set" if settings.DEEPGRAM_KEY else "✗ MISSING"
    print(f"[CONFIG] GOOGLE_API_KEY = {google_ok}")
    print(f"[CONFIG] GROQ_API_KEY   = {groq_ok}")
    print(f"[CONFIG] DEEPGRAM_KEY   = {deepgram_ok}")
    if "localhost" in settings.BACKEND_URL:
        print("[WARNING] BACKEND_URL points to localhost — token reporting will FAIL in production!")
        print("[WARNING] Set BACKEND_URL env var to your production NestJS URL.")
    # ── Fine-tuned gateway health check (non-blocking) ────────────────────
    if settings.FINE_TUNED_INTERVIEWER_ENABLED or settings.FINE_TUNED_EVALUATOR_ENABLED:
        try:
            from app.services.ai.fine_tuned_gateway import gateway_client
            health_ok = await gateway_client.health_check()
            gw_status = "✓ reachable" if health_ok else "✗ UNREACHABLE (will fallback to Gemini)"
        except Exception as gw_exc:
            gw_status = f"✗ ERROR: {gw_exc}"
        print(f"[CONFIG] FINE_TUNED_GATEWAY   = {gw_status}")
        print(f"[CONFIG]   INTERVIEWER_ENABLED = {settings.FINE_TUNED_INTERVIEWER_ENABLED}")
        print(f"[CONFIG]   EVALUATOR_ENABLED   = {settings.FINE_TUNED_EVALUATOR_ENABLED}")
        print(f"[CONFIG]   GATEWAY_URL         = {settings.FINE_TUNED_GATEWAY_URL}")
    print("=" * 60)
    yield
    # ── Shutdown ─────────────────────────────────────────────────────────────
    key_manager.stop_auto_refresh()

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
