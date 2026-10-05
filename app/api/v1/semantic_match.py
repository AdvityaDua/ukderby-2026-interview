from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    pass

router = APIRouter()

class JobData(BaseModel):
    id: str
    text: str

class MatchRequest(BaseModel):
    resume_text: str
    jobs: List[JobData]

class MatchResponseItem(BaseModel):
    id: str
    score: float

class MatchResponse(BaseModel):
    matches: List[MatchResponseItem]

@router.post("/match", response_model=MatchResponse)
async def semantic_match(request: MatchRequest):
    if not request.jobs:
        return MatchResponse(matches=[])

    try:
        tfidf = TfidfVectorizer(stop_words='english')
        documents = [request.resume_text] + [job.text for job in request.jobs]
        tfidf_matrix = tfidf.fit_transform(documents)

        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        matches = []
        for i, job in enumerate(request.jobs):
            score = float(cosine_sim[i])
            matches.append(MatchResponseItem(id=job.id, score=score))

        matches.sort(key=lambda x: x.score, reverse=True)
        return MatchResponse(matches=matches)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
