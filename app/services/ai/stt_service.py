import os
import json
import time
import wave
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
import websockets

load_dotenv()

# Directory for storing answer audio recordings
AUDIO_DIR = Path(__file__).parent.parent / "audio_recordings"
AUDIO_DIR.mkdir(exist_ok=True)

DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"
logger = logging.getLogger(__name__)


class DeepgramSTTService:
    """
    Real-time STT service using Deepgram's Live WebSocket API (raw websockets).
    
    For each listening session:
    1. Opens a WebSocket to Deepgram's live API  
    2. Streams audio chunks → receives real-time transcripts
    3. Saves complete answer audio to disk
    """

    def __init__(self):
        self.api_key = os.getenv("DEEPGRAM_KEY")
        if not self.api_key:
            raise ValueError("DEEPGRAM_KEY not found in environment")

    async def create_stream(self, on_partial, on_final):
        """
        Create a new Deepgram live transcription stream.
        
        Args:
            on_partial: async callback(text) for interim results
            on_final: async callback(text) for finalized sentence  
            
        Returns:
            DeepgramStream object with send() and finish() methods
        """
        # Build WebSocket URL with query params
        params = (
            "model=nova-2"
            "&language=en-US"
            "&smart_format=true"
            "&interim_results=true"
            "&endpointing=1500"
            "&no_delay=true"
            "&filler_words=true"
            "&encoding=linear16"
            "&sample_rate=16000"
            "&channels=1"
        )
        url = f"{DEEPGRAM_WS_URL}?{params}"
        
        headers = {
            "Authorization": f"Token {self.api_key}"
        }

        # Open WebSocket to Deepgram
        ws = await websockets.connect(url, additional_headers=headers)
        logger.info("[STT] Deepgram stream connected")

        stream = DeepgramStream(ws, on_partial, on_final)
        
        # Start listening for responses in background
        stream._listen_task = asyncio.create_task(stream._listen_loop())
        
        return stream


class DeepgramStream:
    """Wraps a Deepgram WebSocket connection for sending audio and receiving transcripts."""

    def __init__(self, ws, on_partial, on_final):
        self.ws = ws
        self.on_partial = on_partial
        self.on_final = on_final
        self.audio_chunks = []
        self.accumulated_final = ""
        self._listen_task = None
        self._sample_rate = 16000
        self._closed = False

    async def _listen_loop(self):
        """Background task to receive and process Deepgram responses."""
        try:
            async for message in self.ws:
                data = json.loads(message)
                msg_type = data.get("type", "")

                if msg_type == "Results":
                    channel = data.get("channel", {})
                    alternatives = channel.get("alternatives", [])
                    if not alternatives:
                        continue
                    
                    transcript = alternatives[0].get("transcript", "")
                    if not transcript:
                        continue

                    is_final = data.get("is_final", False)

                    if is_final:
                        self.accumulated_final += (" " + transcript).strip() + " "
                        await self.on_final(self.accumulated_final.strip())
                    else:
                        # Interim: show accumulated + current partial
                        partial = (self.accumulated_final + transcript).strip()
                        await self.on_partial(partial)



        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"[STT] Deepgram websocket closed: code={e.code}, reason={e.reason}")
        except Exception as e:
            logger.exception(f"[STT] listen_loop failed: {e}")

    async def send(self, audio_bytes: bytes):
        """Send raw audio bytes to Deepgram."""
        if self._closed:
            return
        self.audio_chunks.append(audio_bytes)
        try:
            await self.ws.send(audio_bytes)
        except Exception as e:
            logger.warning(f"[STT] Failed to send audio chunk: {e}")

    async def finish(self):
        """Close the Deepgram connection gracefully."""
        if self._closed:
            return
        self._closed = True
        
        try:
            # Send close message to Deepgram
            await self.ws.send(json.dumps({"type": "CloseStream"}))
            # Wait briefly for final responses
            await asyncio.sleep(0.5)
            await self.ws.close()
        except Exception as e:
            logger.warning(f"[STT] Error during stream finish: {e}")
        
        # Cancel listen task
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass

    def save_audio(self, session_id: str, answer_index: int) -> str:
        """
        Save accumulated audio to a WAV file.
        Returns the file path, or None if no audio.
        """
        if not self.audio_chunks:
            return None

        timestamp = int(time.time())
        filename = f"{session_id}_answer_{answer_index}_{timestamp}.wav"
        filepath = AUDIO_DIR / filename

        # Combine all audio chunks (raw 16-bit PCM)
        all_audio = b"".join(self.audio_chunks)

        # Write as WAV
        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self._sample_rate)
            wf.writeframes(all_audio)

        size_kb = filepath.stat().st_size / 1024
        duration_s = len(all_audio) / (self._sample_rate * 2)
        return str(filepath)


# Global service instance
stt_service = DeepgramSTTService()
