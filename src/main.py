"""
FastAPI application entry point.

Single endpoint: POST /query
  - Accepts QueryRequest (query, session_id, user_id, user_profile)
  - Runs full pipeline: safety → classify → route → stream
  - Streams response via Server-Sent Events (SSE)
  - All errors return structured SSE error events, never stack traces

SSE is the ONLY response mode — no JSON fallback path.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from src.models.api import QueryRequest
from src.pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)



# Lifespan — initialise DB on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.session.store import get_history
    await get_history("__init__")  # triggers table creation
    logger.info("Valura AI microservice started")
    yield
    logger.info("Valura AI microservice shutting down")


# App
app = FastAPI(
    title="Valura AI",
    description="AI co-investor microservice — safety · classify · route · stream",
    version="1.0.0",
    lifespan=lifespan,
)


# Health check
@app.get("/health")
async def health():
    return {"status": "ok", "service": "valura-ai"}


# Main query endpoint
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    Run the full Valura pipeline and stream the response via SSE.

    SSE event types:
      metadata — classification result (agent, intent, entities)
      token    — response chunk from the agent
      error    — structured error (safety block, timeout, agent error)
      done     — terminal event, always sent last
    """
    async def event_generator():
        async for chunk in run_pipeline(request):
            # chunk is already SSE-formatted: "event: ...\ndata: ...\n\n"
            # EventSourceResponse expects raw strings
            yield chunk

    return EventSourceResponse(event_generator())


# Global exception handler — last resort, should never be reached
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "code": "internal_error",
            "message": "An unexpected error occurred. Please try again.",
        },
    )