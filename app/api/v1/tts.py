import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from app.core.config import settings

router = APIRouter()

DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak?model=aura-helios-en"

@router.post("/tts")
async def text_to_speech(payload: dict):
    """
    Convert text to speech using Deepgram Aura TTS.
    Expects JSON: {"text": "string"}
    Returns: audio/mp3 stream
    """
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    headers = {
        "Authorization": f"Token {settings.DEEPGRAM_KEY}",
        "Content-Type": "application/json"
    }
    
    body = {
        "text": text
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                DEEPGRAM_TTS_URL,
                headers=headers,
                json=body,
                timeout=30.0
            )
            
            if response.status_code != 200:
                # Fallback or error
                raise HTTPException(status_code=response.status_code, detail="Deepgram TTS failed")

            return Response(content=response.content, media_type="audio/mp3")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
