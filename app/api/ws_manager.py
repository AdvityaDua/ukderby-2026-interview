from fastapi import WebSocket
from app.services.ai.streaming_session import StreamingInterviewSession
from app.services.ai.gemini_client import GeminiClient
from app.services.redis_cache import redis_cache


class ConnectionManager:
    def __init__(self):
        # Maps user ID (sub) to their active WebSocket connection
        self.active_connections: dict[str, WebSocket] = {}
        # Maps user ID to their active StreamingInterviewSession
        self.sessions: dict[str, StreamingInterviewSession] = {}
        # Maps user ID to list of per-answer audio metrics dicts
        self.audio_metrics: dict[str, list[dict]] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_json(self, data: dict, user_id: str):
        ws = self.active_connections.get(user_id)
        if ws:
            await ws.send_json(data)

    async def get_or_create_session(self, user_id: str) -> tuple[StreamingInterviewSession, bool]:
        """
        Returns (session, restored_from_cache).
        If a session exists in memory, returns it.
        If not, tries to restore from Redis.
        If neither, creates a brand new session.
        """
        # 1. In-memory hit
        if user_id in self.sessions:
            session = self.sessions[user_id]
            restored = bool(session.history)
            return session, restored

        # 2. Try Redis
        cached = await redis_cache.load_session(user_id)
        if cached:
            client = GeminiClient()
            session = StreamingInterviewSession(client)
            session.context_summary = cached.get("context_summary", "")
            session.history = cached.get("history", [])
            session.ended = cached.get("ended", False)
            session.interview_type = cached.get("interview_type", "technical")
            session.role = cached.get("role", "")
            session.company = cached.get("company", "")
            session.start_time = cached.get("start_time", 0.0)
            session.duration_limit = cached.get("duration_limit", 0)
            session.input_tokens = cached.get("input_tokens", 0)
            session.output_tokens = cached.get("output_tokens", 0)
            # Restore missing fields for the AI Graph
            session.skills_remaining = cached.get("skills_remaining", [])
            session.skills_covered = cached.get("skills_covered", [])
            session.performance_summary = cached.get("performance_summary", "")
            session.user_id = cached.get("user_id", user_id)
            session.session_id = cached.get("session_id", "demo")
            
            self.sessions[user_id] = session
            # Restore audio metrics from Redis too
            self.audio_metrics[user_id] = cached.get("audio_metrics", [])
            return session, True

        # 3. Brand new
        client = GeminiClient()
        session = StreamingInterviewSession(client)
        self.sessions[user_id] = session
        self.audio_metrics[user_id] = []
        return session, False

    def add_audio_metrics(self, user_id: str, metrics: dict) -> None:
        """Add per-answer audio metrics to the session."""
        if user_id not in self.audio_metrics:
            self.audio_metrics[user_id] = []
        self.audio_metrics[user_id].append(metrics)

    def get_audio_metrics(self, user_id: str) -> list[dict]:
        """Get all audio metrics for a user's session."""
        return self.audio_metrics.get(user_id, [])

    async def save_session_to_cache(self, user_id: str) -> None:
        """Persist current in-memory session state to Redis."""
        session = self.sessions.get(user_id)
        if not session:
            return
        data = {
            "context_summary": session.context_summary,
            "history": session.history,
            "ended": session.ended,
            "interview_type": session.interview_type,
            "role": session.role,
            "company": session.company,
            "start_time": session.start_time,
            "duration_limit": session.duration_limit,
            "input_tokens": session.input_tokens,
            "output_tokens": session.output_tokens,
            "skills_remaining": session.skills_remaining,
            "skills_covered": session.skills_covered,
            "performance_summary": session.performance_summary,
            "user_id": session.user_id,
            "session_id": session.session_id,
            "audio_metrics": self.audio_metrics.get(user_id, []),
        }
        await redis_cache.save_session(user_id, data)

    async def clear_session(self, user_id: str):
        """Clear session from both memory and Redis."""
        if user_id in self.sessions:
            del self.sessions[user_id]
        if user_id in self.audio_metrics:
            del self.audio_metrics[user_id]
        await redis_cache.delete_session(user_id)


manager = ConnectionManager()
