"""
Intent Classifier — single LLM call per request.

Returns a structured ClassifierOutput covering:
  - agent to dispatch
  - intent label
  - extracted entities
  - informational safety verdict (does NOT block — guard does that)

Follow-up resolution: prior session turns are injected into the prompt
so the LLM can resolve pronouns and carry entity context across turns.

Fallback: any LLM failure returns ClassifierOutput.fallback() — never crashes.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.models.classifier import ClassifierOutput, ExtractedEntities
from src.utils.logger import logger


# System prompt — defines the full classification contract
_SYSTEM_PROMPT = """
You are the intent classifier for Valura, an AI wealth management platform.

Your job is to classify the user's query into exactly one agent and extract entities.

## Agent Taxonomy
- portfolio_health      — health check, diversification, concentration, benchmark comparison
- market_research       — price, news, sector info, index performance, FX rates
- investment_strategy   — should I buy/sell/rebalance/hedge, allocation guidance
- financial_planning    — retirement, education, house, FIRE, long-term goals
- financial_calculator  — DCA, mortgage, future value, FX conversion, tax calc
- risk_assessment       — beta, drawdown, stress test, exposure analysis
- product_recommendation— recommend ETFs, funds, instruments matching profile
- predictive_analysis   — forecasts, trend extrapolation, future projections
- customer_support      — login issues, account questions, platform how-to
- general_query         — definitions, education, greetings, conversational, gibberish
- portfolio_query       — questions about the user's OWN holdings of a specific ticker

## Entity Vocabulary
Extract only what is explicitly present or clearly implied:
- tickers: array of strings uppercase (AAPL, NVDA, ASML.AS, 7203.T)
- amount: number
- currency: ISO 4217 (USD, EUR, GBP, JPY)
- rate: decimal (0.08 for 8%)
- period_years: integer
- frequency: one of [daily, weekly, monthly, yearly]
- horizon: one of [6_months, 1_year, 5_years]
- time_period: one of [today, this_week, this_month, this_year]
- topics: array of strings
- sectors: array of strings
- index: one of [S&P 500, FTSE 100, NIKKEI 225, MSCI World]
- action: one of [buy, sell, hold, hedge, rebalance]
- goal: one of [retirement, education, house, FIRE, emergency_fund]
- intent: use "comparison" if user is comparing two or more things

## Follow-up Resolution Rules
- If the current query uses a pronoun ("it", "they", "them") or is very short ("what about AMD?"),
  resolve entity context from the conversation history provided.
- If the user switches topic entirely, do NOT carry over entities from prior turns.
- If the query references "that thing" or is ambiguous with no resolvable context, use general_query.

## Safety Verdict (informational only — does NOT block)
- flag: if the query touches a sensitive topic even if educational
- pass: otherwise

## Output Format
Respond with ONLY valid JSON, no markdown, no explanation:
{
  "agent": "<agent_name>",
  "intent": "<short intent label>",
  "entities": { <only non-empty fields> },
  "safety_verdict": "pass" | "flag",
  "safety_reason": "<string or null>",
  "confidence": <0.0-1.0>
}
"""


# Classifier
def _build_messages(
    query: str,
    history: list[dict],
) -> list[dict]:
    """
    Build the OpenAI messages array.
    History is injected as prior user/assistant turns so the LLM
    can resolve follow-up references.
    """
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # Inject session history (last N turns already trimmed by store)
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Current query
    messages.append({"role": "user", "content": query})
    return messages


def _parse_response(raw: str) -> ClassifierOutput:
    """Parse LLM JSON output into ClassifierOutput. Raises on malformed output."""
    # Strip markdown fences if model adds them despite instructions
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip()

    data = json.loads(clean)

    entities_raw = data.get("entities", {})
    entities = ExtractedEntities(**{
        k: v for k, v in entities_raw.items()
        if v is not None and v != [] and v != ""
    })

    return ClassifierOutput(
        agent=data.get("agent", "general_query"),
        intent=data.get("intent", "unknown"),
        entities=entities,
        safety_verdict=data.get("safety_verdict", "pass"),
        safety_reason=data.get("safety_reason"),
        confidence=float(data.get("confidence", 1.0)),
    )


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
    reraise=False,
)
def _call_llm(messages: list[dict], client: OpenAI, model: str) -> str:
    """Call the OpenAI API. Retried once on transient failure."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,          # deterministic classification
        max_tokens=400,         # structured JSON — never needs more
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def classify(
    query: str,
    history: Optional[list[dict]] = None,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> ClassifierOutput:
    """
    Classify a user query into an agent + entities.

    Args:
        query:   Raw user query string
        history: Prior session turns [{role, content}, ...]
        client:  OpenAI client (injected for testing)
        model:   Model override (injected for testing)

    Returns:
        ClassifierOutput — never raises, falls back to safe default on error
    """
    settings = get_settings()
    _client = client or OpenAI(api_key=settings.openai_api_key)
    _model = model or settings.model
    _history = history or []

    messages = _build_messages(query, _history)

    try:
        raw = _call_llm(messages, _client, _model)
        if raw is None:
            raise ValueError("LLM returned None")
        return _parse_response(raw)
    except Exception as exc:
        logger.warning(
            "Classifier LLM call failed — using fallback. Error: %s", exc
        )
        return ClassifierOutput.fallback(query)