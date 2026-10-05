from fastapi import Request

async def get_current_user(request: Request):
    """
    Dependency to get the authenticated user from the request state.
    AuthMiddleware stores the JWT payload in scope["state"]["user"].
    scope["state"] may be a plain dict (set by middleware) or a Starlette State object,
    so we handle both.
    """
    state = request.scope.get("state", {})
    if isinstance(state, dict):
        return state.get("user")
    return getattr(state, "user", None)
