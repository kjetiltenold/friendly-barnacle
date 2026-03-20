import json
import logging
from pathlib import Path
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import iterate_in_threadpool

from app.models import SolveRequest, SolveResponse
from app.agent.solver import solve_task

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

app = FastAPI(title="Tripletex Agent")
logger = logging.getLogger(__name__)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate_text(value: str, limit: int = 4_000) -> dict[str, Any]:
    if len(value) <= limit:
        return {"text": value, "truncated": False}
    return {"text": value[:limit], "truncated": True, "original_length": len(value)}


def _sanitize_solve_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload

    prompt = payload.get("prompt", "")
    files = payload.get("files", [])
    credentials = payload.get("tripletex_credentials") or {}

    prompt_details = _truncate_text(prompt) if isinstance(prompt, str) else {"text": None, "truncated": False}

    sanitized_files = []
    if isinstance(files, list):
        for file in files:
            if not isinstance(file, dict):
                continue
            content_base64 = file.get("content_base64")
            sanitized_files.append(
                {
                    "filename": file.get("filename"),
                    "mime_type": file.get("mime_type"),
                    "content_base64_length": len(content_base64) if isinstance(content_base64, str) else None,
                }
            )

    sanitized_credentials = None
    if isinstance(credentials, dict):
        sanitized_credentials = {
            "base_url": credentials.get("base_url"),
            "session_token": "[REDACTED]",
        }

    return {
        "prompt": prompt_details["text"],
        "prompt_truncated": prompt_details["truncated"],
        "prompt_length": len(prompt) if isinstance(prompt, str) else None,
        "file_count": len(sanitized_files),
        "files": sanitized_files,
        "tripletex_credentials": sanitized_credentials,
    }


def _serialize_response_body(body: bytes) -> Any:
    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _truncate_text(body.decode("utf-8", errors="replace"))


def _log_event(event: str, **data: Any) -> None:
    logger.info(json.dumps({"event": event, "timestamp": _utc_timestamp(), **data}, default=str))


@app.middleware("http")
async def log_solve_requests(request: Request, call_next):
    if request.url.path != "/solve":
        return await call_next(request)

    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = time.monotonic()
    body = await request.body()

    try:
        parsed_body = json.loads(body.decode("utf-8")) if body else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        parsed_body = {"raw_body": _truncate_text(body.decode("utf-8", errors="replace"))}

    _log_event(
        "solve_request_received",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        query=str(request.url.query),
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        request=_sanitize_solve_payload(parsed_body),
    )

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(request.scope, receive)
    request.state.request_id = request_id

    response = await call_next(request)

    response_chunks = [chunk async for chunk in response.body_iterator]
    response.body_iterator = iterate_in_threadpool(iter(response_chunks))
    response_body = b"".join(response_chunks)

    _log_event(
        "solve_request_finished",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        duration_ms=round((time.monotonic() - start) * 1000, 1),
        response_status_code=response.status_code,
        response=_serialize_response_body(response_body),
    )

    return response


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/solve", response_model=SolveResponse)
async def solve(request: SolveRequest, http_request: Request):
    try:
        await solve_task(request)
    except Exception as exc:
        _log_event(
            "solve_agent_error",
            request_id=getattr(http_request.state, "request_id", None),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        logger.exception("Agent error (partial work may still score)")
    return SolveResponse()


@app.exception_handler(Exception)
async def fallback_handler(request: Request, exc: Exception):
    _log_event(
        "unhandled_exception",
        request_id=getattr(request.state, "request_id", None),
        path=request.url.path,
        error_type=type(exc).__name__,
        error_message=str(exc),
    )
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=200, content={"status": "completed"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
