from fastapi import APIRouter
from app.api.v1.meta_data import router as meta_data_router
from app.api.v1.interview_engine import router as interview_engine_router
from app.api.v1.auth import auth_router
from app.api.v1.interviews import interviews_router

api_router = APIRouter()
api_router.include_router(meta_data_router, prefix="/meta", tags=["Metadata"])
api_router.include_router(interview_engine_router, tags=["Interview Engine"])
api_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
api_router.include_router(interviews_router, prefix="/interviews", tags=["Interviews"])
