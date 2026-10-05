from fastapi import APIRouter, HTTPException, Depends
from app.models.db import SessionLocal, Interview, User
import jwt
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.api.v1.auth import SECRET_KEY

interviews_router = APIRouter()
security = HTTPBearer()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db=Depends(get_db)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = db.query(User).filter(User.id == int(user_id)).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

@interviews_router.get("/")
def get_user_interviews(current_user: User = Depends(get_current_user), db=Depends(get_db)):
    interviews = db.query(Interview).filter(Interview.user_id == current_user.id).order_by(Interview.created_at.desc()).all()
    return interviews

@interviews_router.get("/{interview_id}")
def get_interview(interview_id: int, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    interview = db.query(Interview).filter(Interview.id == interview_id, Interview.user_id == current_user.id).first()
    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")
    return interview