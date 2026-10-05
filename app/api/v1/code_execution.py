"""
Code Execution API — Proxy to Piston engine.
Provides sandboxed code execution for the interview code editor.
"""

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.core.config import settings

router = APIRouter(prefix="/code", tags=["Code Execution"])


class CodeFile(BaseModel):
    name: str = "main"
    content: str


class ExecuteRequest(BaseModel):
    language: str
    version: str
    files: list[CodeFile]
    stdin: str = ""
    args: list[str] = Field(default_factory=list)
    run_timeout: int = 3000  # 3s max (Piston container limit)


class RunResult(BaseModel):
    stdout: str = ""
    stderr: str = ""
    code: Optional[int] = None
    output: str = ""


class ExecuteResponse(BaseModel):
    language: str
    version: str
    run: RunResult
    compile: Optional[RunResult] = None


class RuntimeInfo(BaseModel):
    language: str
    version: str
    aliases: list[str] = Field(default_factory=list)
    runtime: Optional[str] = None


@router.post("/execute", response_model=ExecuteResponse)
async def execute_code(req: ExecuteRequest):
    """Execute code in a sandboxed Piston container."""
    piston_url = f"{settings.PISTON_URL}/api/v2/execute"

    payload = {
        "language": req.language,
        "version": req.version,
        "files": [f.model_dump() for f in req.files],
        "stdin": req.stdin,
        "args": req.args,
        "run_timeout": req.run_timeout,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(piston_url, json=payload)

        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Piston error: {resp.text}"
            )

        return resp.json()

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Code execution service unavailable. Piston is not running."
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Code execution timed out."
        )


@router.get("/runtimes", response_model=list[RuntimeInfo])
async def list_runtimes():
    """List all available language runtimes from Piston."""
    piston_url = f"{settings.PISTON_URL}/api/v2/runtimes"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(piston_url)

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Failed to fetch runtimes")

        return resp.json()

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Code execution service unavailable. Piston is not running."
        )
