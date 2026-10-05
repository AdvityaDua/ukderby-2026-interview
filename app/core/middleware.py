from fastapi import Request, HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from app.core.security import verify_jwt_token

class AuthMiddleware:
    """
    Pure ASGI middleware for JWT authentication.
    """
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        # 1. Bypass non-HTTP (WebSockets)
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # 2. Extract Method/Path from scope (avoiding Request object if possible)
        method = scope.get("method", "").upper()
        path = scope.get("path", "")

        # 3. Bypass unprotected paths (including OPTIONS preflight)
        unprotected_paths = ["/docs", "/openapi.json", "/redoc", "/api/v1/health", "/api/v1/interview/", "/api/v1/code/", "/api/v1/match", "/api/v1/docs/seed", "/api/v1/rag/", "/api/v1/question-bank/"]
        if method == "OPTIONS" or any(path.startswith(p) for p in unprotected_paths):
            await self.app(scope, receive, send)
            return

        # 4. Extract Authorization header from scope
        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode("utf-8")

        if not auth_header or not auth_header.startswith("Bearer "):
            response = JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid Authorization header"}
            )
            await response(scope, receive, send)
            return

        token = auth_header.split(" ")[1]
        try:
            payload = verify_jwt_token(token)
            # Starlette's scope["state"] starts as a plain dict in ASGI.
            # Store the JWT payload there so get_current_user can read it via
            # request.scope["state"]["user"].
            existing = scope.get("state", {})
            if isinstance(existing, dict):
                scope["state"] = {**existing, "user": payload}
            else:
                # If Starlette has already created a State object, set as attribute
                existing.user = payload
        except Exception as e:
            response = JSONResponse(status_code=401, content={"detail": f"Auth failed: {str(e)}"})
            await response(scope, receive, send)
            return

        # 6. Proceed to route
        await self.app(scope, receive, send)
