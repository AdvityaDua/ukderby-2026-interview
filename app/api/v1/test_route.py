from fastapi import APIRouter, Depends
from app.api.dependencies import get_current_user

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

@router.get("/me")
async def get_my_profile(user: dict = Depends(get_current_user)):
    return {
        "message": "You are authenticated!",
        "user_data": user
    }
