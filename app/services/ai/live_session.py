import asyncio
import json
from .gemini_client import GeminiClient

class LiveInterviewSession:
    def __init__(self, client: GeminiClient):
        self.client = client
        self.context_summary = ""
        self.resume_text = ""
        self.jd_text = ""
        self.history = []

    async def initialize_session(self, resume_text: str, jd_text: str):
        self.resume_text = resume_text
        self.jd_text = jd_text
        # We might generate the context summary first using the standard client
        self.context_summary, _ = await self.client.summarize_context(resume_text, jd_text)

    async def start_session(self, websocket, manager):
        """
        Manages the Bidi stream.
        websocket: FastAPI WebSocket
        manager: ConnectionManager
        """
        system_instruction = f"""
        You are an expert technical interviewer conducting a realtime voice interview.
        
        CONTEXT:
        {self.context_summary}
        
        GOAL:
        Ask questions to evaluate the candidate based on the Job Description.
        One question at a time.
        Be professional but conversational.
        If the candidate answers well, go deeper.
        If they struggle, hint or move on.
        
        Start by introducing yourself and asking the first question.
        """

        async with self.client.connect_live(system_instruction=system_instruction) as session:
            # Task to receive from WebSocket and send to Gemini
            async def receive_from_client():
                try:
                    while True:
                        # Expecting JSON with "data" (base64) or "text"
                        # Or raw bytes? Let's assume JSON wrapper for now as per plan
                        message_str = await websocket.receive_text()
                        message = json.loads(message_str)
                        
                        if message.get("type") == "audio":
                            # Send audio chunk
                            # Assuming message["data"] is base64 encoded PCM/encapsulated
                            # Gemini Live expects raw bytes or specific blob
                            # client.send(input=..., end_of_turn=True/False)
                            # For audio streaming, we send bytes.
                            # b64 decode
                            import base64
                            audio_data = base64.b64decode(message.get("data"))
                            await session.send(input=audio_data, end_of_turn=False)
                            
                        elif message.get("type") == "text":
                            await session.send(input=message.get("data"), end_of_turn=True)
                            
                except Exception as e:
                    pass

            # Task to receive from Gemini and send to WebSocket
            async def receive_from_gemini():
                try:
                    async for response in session.receive():
                        # response is model_turn
                        # We might get audio or text
                        
                        server_content = response.server_content
                        if server_content is None:
                            continue
                            
                        model_turn = server_content.model_turn
                        if model_turn is None:
                            continue

                        for part in model_turn.parts:
                            if part.text:
                                await manager.send_json(websocket, {"type": "text", "content": part.text})
                                self.history.append({"role": "model", "content": part.text})
                            
                            if part.inline_data:
                                # Audio data
                                import base64
                                b64_audio = base64.b64encode(part.inline_data.data).decode('utf-8')
                                await manager.send_json(websocket, {"type": "audio", "data": b64_audio})

                except Exception as e:
                    pass

            # Run both
            # send_task = asyncio.create_task(receive_from_client())
            # recv_task = asyncio.create_task(receive_from_gemini())
            # await asyncio.gather(send_task, recv_task)
            # Better handling:
            
            await asyncio.gather(receive_from_client(), receive_from_gemini())
