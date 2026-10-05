from fastapi import WebSocket, Query

async def get_ws_current_user(websocket: WebSocket, token: str = Query(None)):
    # Mock user payload for sandbox
    return {"sub": "test-user-123"}
