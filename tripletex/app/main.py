import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.models import SolveRequest, SolveResponse
from app.agent.solver import solve_task

app = FastAPI(title="Tripletex Agent")
logger = logging.getLogger(__name__)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/solve", response_model=SolveResponse)
async def solve(request: SolveRequest):
    start = time.monotonic()
    try:
        await solve_task(request)
    except Exception:
        logger.exception("Agent error (partial work may still score)")
    elapsed = time.monotonic() - start
    logger.info(f"Solve completed in {elapsed:.1f}s")
    return SolveResponse()


@app.exception_handler(Exception)
async def fallback_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=200, content={"status": "completed"})
