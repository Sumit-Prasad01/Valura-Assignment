"""
Pipeline orchestrator — the single place that owns request lifecycle.

Flow:
  1. Safety Guard  (sync, <10ms, no LLM)
         ↓ blocked → yield error SSE, done
  2. Parse user profile
  3. Load session history from SQLite
  4. Intent Classifier (single LLM call, run in executor to avoid blocking)
  5. Emit metadata SSE event
  6. Save user turn to session
  7. Route to agent → stream chunks via SSE
  8. Save assistant response to session

Timeout wraps steps 3-8.
Errors at any step produce a structured SSE error event — never a stack trace.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from src.classifier.intent import classify
from src.config import get_settings
from src.models.api import QueryRequest
from src.models.classifier import ClassifierOutput
from src.models.user import UserProfile
from src.router import get_agent
from src.safety.guard import check
from src.session.store import get_history, save_turn

logger = logging.getLogger(__name__)


# SSE formatter
def _sse(event: str, data: dict | str) -> str:
    """Format a single SSE event string ready to stream."""
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"



# Inner pipeline (runs under timeout)
async def _run_with_timeout(
    request: QueryRequest,
    query: str,
    user: UserProfile,
    settings,
) -> AsyncIterator[str]:
    """
    Steps 3-8 of the pipeline.
    Called exclusively from run_pipeline which wraps this in asyncio.timeout.
    """

    # Step 3 — Load session history
    history = await get_history(request.session_id)

    # Step 4 — Classify intent (blocking SDK call → run in executor)
    try:
        loop = asyncio.get_event_loop()
        classification = await loop.run_in_executor(
            None,
            lambda: classify(query, history=history),
        )
    except Exception as exc:
        logger.error("Classifier error: %s", exc)
        classification = ClassifierOutput.fallback(query)

    # Step 5 — Emit metadata event
    yield _sse("metadata", {
        "agent": classification.agent,
        "intent": classification.intent,
        "entities": classification.entities.to_dict(),
        "safety_verdict": classification.safety_verdict,
        "confidence": classification.confidence,
    })

    # Step 6 — Save user turn to session
    await save_turn(request.session_id, "user", query)

    # Step 7 — Route to agent and stream response
    agent = get_agent(classification.agent)
    full_response_parts: list[str] = []

    try:
        async for chunk in agent.run(
            query=query,
            classification=classification,
            user=user,
            history=history,
        ):
            full_response_parts.append(chunk)
            yield _sse("token", {"text": chunk})

    except Exception as exc:
        logger.exception("Agent streaming error: %s", exc)
        yield _sse("error", {
            "code": "agent_error",
            "message": "An error occurred while generating the response.",
        })

    # Step 8 — Save assistant response to session
    full_response = "".join(full_response_parts)
    if full_response:
        await save_turn(request.session_id, "assistant", full_response)

    yield _sse("done", {"status": "ok", "agent": classification.agent})


# Public entry point
async def run_pipeline(request: QueryRequest) -> AsyncIterator[str]:
    """
    Main pipeline generator. Yields SSE-formatted strings.
    Called by the FastAPI endpoint — streamed directly to client.

    Steps:
      1. Safety guard (sync, always first)
      2. Parse user profile
      3-8. Classified pipeline under timeout
    """
    settings = get_settings()
    query = request.query.strip()

    # Step 1 — Safety Guard (sync, no timeout needed, <10ms)
    guard_result = check(query)
    if guard_result.blocked:
        yield _sse("error", {
            "code": "safety_block",
            "category": guard_result.category,
            "message": guard_result.message,
        })
        yield _sse("done", {"status": "blocked"})
        return

    # Step 2 — Parse user profile
    try:
        user = UserProfile(**request.user_profile)
    except Exception as exc:
        logger.error("Invalid user profile: %s", exc)
        yield _sse("error", {
            "code": "invalid_user_profile",
            "message": "User profile is malformed. Please check your request.",
        })
        yield _sse("done", {"status": "error"})
        return

    # Steps 3-8 — under timeout
    try:
        async with asyncio.timeout(settings.pipeline_timeout_s):
            async for chunk in _run_with_timeout(request, query, user, settings):
                yield chunk

    except asyncio.TimeoutError:
        yield _sse("error", {
            "code": "timeout",
            "message": (
                f"Request exceeded {settings.pipeline_timeout_s}s timeout. "
                "Please try again."
            ),
        })
        yield _sse("done", {"status": "timeout"})