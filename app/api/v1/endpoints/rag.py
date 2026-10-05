from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import List, Optional
import os
import shutil
import uuid
from app.services.ai.rag_service import rag_service
router = APIRouter()

TEMP_UPLOAD_DIR = "temp_rag_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

@router.post("/index-file")
async def index_file(
    topic_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload and index a PDF or plain-text transcript file."""
    allowed_extensions = (".pdf", ".txt")
    filename_lower = (file.filename or "").lower()
    if not any(filename_lower.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
    
    file_id = str(uuid.uuid4())
    temp_path = os.path.join(TEMP_UPLOAD_DIR, f"{file_id}_{file.filename}")
    
    try:
        # Save temp file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Index
        chunk_count = await rag_service.index_pdf(temp_path, topic_id)
        
        return {
            "status": "success",
            "message": f"Indexed {chunk_count} chunks",
            "topic_id": topic_id,
            "filename": file.filename,
            "chunks": chunk_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.get("/query")
async def query_knowledge(
    topic_id: str,
    query: str,
    k: int = 5
):
    """Retrieve relevant information from a topic."""
    results = await rag_service.query_topic(topic_id, query, k)
    return results

@router.delete("/topic/{topic_id}")
async def delete_topic(topic_id: str):
    """Delete all indexed knowledge for a topic."""
    success = await rag_service.delete_topic_collection(topic_id)
    if success:
        return {"status": "deleted"}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete topic index")
