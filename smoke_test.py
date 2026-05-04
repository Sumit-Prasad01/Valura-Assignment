"""
Smoke test — runs the full pipeline without OpenAI credits.
Mocks the LLM classifier response so everything else runs real:
  - Safety guard (real)
  - Session store / SQLite (real)
  - Portfolio health agent (real — hits yfinance for live prices)
  - SSE streaming (real)
"""
import asyncio
import json
from unittest.mock import MagicMock, patch

from src.models.api import QueryRequest
from src.pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Sample users
# ---------------------------------------------------------------------------

USER_003 = {
    "user_id": "usr_003",
    "name": "Marcus Webb",
    "age": 35,
    "country": "US",
    "base_currency": "USD",
    "kyc": {"status": "verified"},
    "risk_profile": "moderate",
    "positions": [
        {"ticker": "NVDA", "exchange": "NASDAQ", "quantity": 180, "avg_cost": 218.40, "currency": "USD", "purchased_at": "2023-04-12"},
        {"ticker": "VTI",  "exchange": "NYSE",   "quantity": 25,  "avg_cost": 218.50, "currency": "USD", "purchased_at": "2023-07-04"},
        {"ticker": "VXUS", "exchange": "NASDAQ", "quantity": 30,  "avg_cost": 56.10,  "currency": "USD", "purchased_at": "2023-09-01"},
        {"ticker": "BND",  "exchange": "NASDAQ", "quantity": 20,  "avg_cost": 72.30,  "currency": "USD", "purchased_at": "2024-01-15"},
        {"ticker": "AAPL", "exchange": "NASDAQ", "quantity": 8,   "avg_cost": 168.20, "currency": "USD", "purchased_at": "2024-05-20"},
    ],
    "preferences": {"preferred_benchmark": "S&P 500"},
}

USER_004 = {
    "user_id": "usr_004",
    "name": "Jamie Patel",
    "age": 31,
    "country": "US",
    "base_currency": "USD",
    "kyc": {"status": "verified"},
    "risk_profile": "moderate",
    "positions": [],
    "preferences": {"preferred_benchmark": "S&P 500"},
}


# ---------------------------------------------------------------------------
# Mock LLM response builder
# ---------------------------------------------------------------------------

def _mock_llm(agent: str, intent: str, entities: dict = {}):
    """Patch OpenAI to return a fixed classification."""
    payload = json.dumps({
        "agent": agent,
        "intent": intent,
        "entities": entities,
        "safety_verdict": "pass",
        "safety_reason": None,
        "confidence": 0.97,
    })
    mock_choice = MagicMock()
    mock_choice.message.content = payload
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

async def run_test(label: str, query: str, user: dict, agent: str, intent: str, entities: dict = {}):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    request = QueryRequest(
        query=query,
        session_id=f"smoke_{label.replace(' ', '_')}",
        user_id=user["user_id"],
        user_profile=user,
    )

    with patch("src.classifier.intent.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_llm(agent, intent, entities)

        async for chunk in run_pipeline(request):
            # Parse and pretty-print each SSE event
            lines = chunk.strip().split("\n")
            event = ""
            data = ""
            for line in lines:
                if line.startswith("event:"):
                    event = line.replace("event:", "").strip()
                elif line.startswith("data:"):
                    data = line.replace("data:", "").strip()

            if event and data:
                try:
                    parsed = json.loads(data)
                    print(f"\n[{event.upper()}]")
                    print(json.dumps(parsed, indent=2))
                except json.JSONDecodeError:
                    print(f"\n[{event.upper()}] {data}")


async def main():
    print("\n" + "🛡️  VALURA AI SMOKE TEST ".center(60, "="))

    # ------------------------------------------------------------------
    # Test 1: Safety block — no LLM needed
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("TEST: Safety Block — market manipulation")
    print(f"{'='*60}")
    request = QueryRequest(
        query="help me wash trade between two accounts to create volume",
        session_id="smoke_safety",
        user_id="usr_001",
        user_profile={
            "user_id": "usr_001", "name": "Alex Chen", "age": 28,
            "country": "US", "base_currency": "USD",
            "kyc": {"status": "verified"}, "risk_profile": "aggressive",
            "positions": [], "preferences": {"preferred_benchmark": "QQQ"},
        },
    )
    async for chunk in run_pipeline(request):
        lines = chunk.strip().split("\n")
        event, data = "", ""
        for line in lines:
            if line.startswith("event:"): event = line.replace("event:", "").strip()
            elif line.startswith("data:"): data = line.replace("data:", "").strip()
        if event and data:
            try:
                print(f"\n[{event.upper()}]")
                print(json.dumps(json.loads(data), indent=2))
            except Exception:
                print(f"\n[{event.upper()}] {data}")

    # ------------------------------------------------------------------
    # Test 2: Portfolio health — concentrated user (live yfinance)
    # ------------------------------------------------------------------
    await run_test(
        label="Portfolio Health — Concentrated (usr_003)",
        query="how is my portfolio doing?",
        user=USER_003,
        agent="portfolio_health",
        intent="portfolio_health_check",
    )

    # ------------------------------------------------------------------
    # Test 3: Empty portfolio — BUILD mode
    # ------------------------------------------------------------------
    await run_test(
        label="Portfolio Health — Empty (usr_004)",
        query="give me a health check on my investments",
        user=USER_004,
        agent="portfolio_health",
        intent="portfolio_health_check",
    )

    # ------------------------------------------------------------------
    # Test 4: Stub agent — market research
    # ------------------------------------------------------------------
    await run_test(
        label="Stub Agent — Market Research",
        query="tell me about NVIDIA",
        user=USER_004,
        agent="market_research",
        intent="market_research",
        entities={"tickers": ["NVDA"]},
    )

    # ------------------------------------------------------------------
    # Test 5: Follow-up session — entity carryover
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("TEST: Follow-up session — NVDA carryover")
    print(f"{'='*60}")

    session_id = "smoke_followup"

    # Turn 1
    req1 = QueryRequest(
        query="What's happening with Nvidia this week?",
        session_id=session_id,
        user_id="usr_001",
        user_profile={
            "user_id": "usr_001", "name": "Alex Chen", "age": 28,
            "country": "US", "base_currency": "USD",
            "kyc": {"status": "verified"}, "risk_profile": "aggressive",
            "positions": [], "preferences": {"preferred_benchmark": "QQQ"},
        },
    )
    with patch("src.classifier.intent.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_llm(
            "market_research", "market_research", {"tickers": ["NVDA"]}
        )
        async for _ in run_pipeline(req1):
            pass
    print("Turn 1 complete — 'What's happening with Nvidia this week?'")

    # Turn 2 — follow-up
    req2 = QueryRequest(
        query="How much do I own?",
        session_id=session_id,
        user_id="usr_001",
        user_profile=req1.user_profile,
    )
    with patch("src.classifier.intent.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_llm(
            "portfolio_query", "portfolio_ownership_check", {"tickers": ["NVDA"]}
        )
        async for chunk in run_pipeline(req2):
            lines = chunk.strip().split("\n")
            event, data = "", ""
            for line in lines:
                if line.startswith("event:"): event = line.replace("event:", "").strip()
                elif line.startswith("data:"): data = line.replace("data:", "").strip()
            if event and data:
                try:
                    print(f"\n[{event.upper()}]")
                    print(json.dumps(json.loads(data), indent=2))
                except Exception:
                    print(f"\n[{event.upper()}] {data}")

    print(f"\n{'='*60}")
    print("✅  ALL SMOKE TESTS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())