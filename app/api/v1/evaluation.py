from fastapi import APIRouter, HTTPException, status
from app.services.cv_eval.schemas import CVEvaluationRequest, CVEvaluationResult
from app.services.cv_eval.evaluation_engine import evaluation_engine

router = APIRouter(prefix="/evaluation")

@router.post("/cv")
async def evaluate_cv_jd(request: CVEvaluationRequest):
    """
    Evaluate CV against Job Description (plain JSON input).
    """
    try:
        if not request.cv_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CV text cannot be empty"
            )
        if not request.jd_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job description text cannot be empty"
            )

        result = evaluation_engine.evaluate(request.cv_text, request.jd_text)
        return result   # returns raw dict/JSON from engine

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )
