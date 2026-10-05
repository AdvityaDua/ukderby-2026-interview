from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import os
import shutil
import uuid

from app.services.ai.question_bank_service import (
    index_md_file,
    load_question_bank,
    delete_question_bank,
    get_random_question,
    get_random_questions,
    get_bank_summary,
)

router = APIRouter()

_TEMP_DIR = "temp_rag_uploads"
os.makedirs(_TEMP_DIR, exist_ok=True)


@router.post("/upload")
async def upload_question_bank(
    topic_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Admin uploads a Markdown (.md) file containing interview questions.
    The file is parsed and stored as a question bank for the given topic.
    """
    filename = (file.filename or "").lower()
    if not filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md (Markdown) files are supported.")

    tmp_path = os.path.join(_TEMP_DIR, f"{uuid.uuid4()}_{file.filename}")
    try:
        with open(tmp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        bank = index_md_file(tmp_path, topic_id)
        return {
            "status": "success",
            "topic_id": topic_id,
            "title": bank.get("title", ""),
            "total_questions": len(bank.get("all_questions", [])),
            "entry_questions": len(bank.get("entry_questions", [])),
            "sections": list(bank.get("sections", {}).keys()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.get("/topic/{topic_id}")
async def get_question_bank(topic_id: str):
    """Return the full parsed question bank for a topic."""
    bank = load_question_bank(topic_id)
    if not bank:
        raise HTTPException(status_code=404, detail="No question bank found for this topic.")
    return bank


@router.get("/topic/{topic_id}/random")
async def random_question(topic_id: str, prefer_entry: bool = True):
    """Return a single random question from the topic's question bank."""
    question = get_random_question(topic_id, prefer_entry=prefer_entry)
    if not question:
        raise HTTPException(status_code=404, detail="No questions found for this topic.")
    return {"question": question}


@router.get("/topic/{topic_id}/random-set")
async def random_question_set(topic_id: str, n: int = 5):
    """Return up to n unique random questions (entry questions prioritised)."""
    questions = get_random_questions(topic_id, n=n)
    if not questions:
        raise HTTPException(status_code=404, detail="No questions found for this topic.")
    return {"questions": questions}


@router.get("/topic/{topic_id}/summary")
async def bank_summary(topic_id: str):
    """Return a compact text summary of the question bank (used for LLM injection)."""
    summary = get_bank_summary(topic_id)
    if not summary:
        raise HTTPException(status_code=404, detail="No question bank found for this topic.")
    return {"summary": summary}


@router.delete("/topic/{topic_id}")
async def delete_bank(topic_id: str):
    """Delete the question bank for a topic."""
    deleted = delete_question_bank(topic_id)
    return {"status": "deleted" if deleted else "not_found"}
