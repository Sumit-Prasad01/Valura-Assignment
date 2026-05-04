"""
Portfolio Health agent contract tests.
LLM is not used by this agent — it uses yfinance for market data.
yfinance calls are mocked so tests run offline.
"""
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.portfolio_health import PortfolioHealthAgent
from src.models.classifier import ClassifierOutput, ExtractedEntities
from src.models.user import UserProfile


def _classification():
    return ClassifierOutput(
        agent="portfolio_health",
        intent="portfolio_health_check",
        entities=ExtractedEntities(),
    )


async def _run_agent(user_dict: dict) -> dict:
    """Helper: run agent and collect full JSON output."""
    import json
    agent = PortfolioHealthAgent()
    user = UserProfile(**user_dict)
    chunks = []
    async for chunk in agent.run(
        query="how is my portfolio doing?",
        classification=_classification(),
        user=user,
        history=[],
    ):
        chunks.append(chunk)

    # Last non-status chunk is the JSON output
    json_chunks = [c for c in chunks if not c.startswith('{"status"')]
    assert json_chunks, "Agent produced no output"
    return json.loads(json_chunks[-1])


@pytest.mark.asyncio
async def test_portfolio_health_does_not_crash_on_empty_portfolio(load_user):
    """user_004 has no positions — must not crash, must include disclaimer."""
    with patch("src.agents.portfolio_health.get_current_price", new_callable=AsyncMock) as mock_price, \
         patch("src.agents.portfolio_health.get_fx_rate_to_usd", new_callable=AsyncMock) as mock_fx, \
         patch("src.agents.portfolio_health.get_benchmark_return", new_callable=AsyncMock) as mock_bm:

        mock_price.return_value = 100.0
        mock_fx.return_value = 1.0
        mock_bm.return_value = 10.0

        user = load_user("usr_004")
        result = await _run_agent(user)

    assert result is not None
    assert "disclaimer" in result
    assert "not investment advice" in result["disclaimer"].lower()
    # Empty portfolio should have build_guidance
    assert "build_guidance" in result or "observations" in result


@pytest.mark.asyncio
async def test_portfolio_health_flags_concentration(load_user):
    """user_003 has ~60% in NVDA — must surface high concentration."""
    with patch("src.agents.portfolio_health.get_current_price", new_callable=AsyncMock) as mock_price, \
         patch("src.agents.portfolio_health.get_fx_rate_to_usd", new_callable=AsyncMock) as mock_fx, \
         patch("src.agents.portfolio_health.get_benchmark_return", new_callable=AsyncMock) as mock_bm:

        mock_price.return_value = 500.0   # high NVDA price → high concentration
        mock_fx.return_value = 1.0
        mock_bm.return_value = 14.0

        user = load_user("usr_003")
        result = await _run_agent(user)

    assert "concentration_risk" in result
    assert result["concentration_risk"]["flag"] in {"high", "warning", "medium"}


@pytest.mark.asyncio
async def test_portfolio_health_includes_disclaimer(load_user):
    """Every response must include a regulatory disclaimer."""
    with patch("src.agents.portfolio_health.get_current_price", new_callable=AsyncMock) as mock_price, \
         patch("src.agents.portfolio_health.get_fx_rate_to_usd", new_callable=AsyncMock) as mock_fx, \
         patch("src.agents.portfolio_health.get_benchmark_return", new_callable=AsyncMock) as mock_bm:

        mock_price.return_value = 180.0
        mock_fx.return_value = 1.0
        mock_bm.return_value = 12.0

        user = load_user("usr_001")
        result = await _run_agent(user)

    assert result["disclaimer"]
    assert "not investment advice" in result["disclaimer"].lower()


@pytest.mark.asyncio
async def test_portfolio_health_multi_currency(load_user):
    """user_006 has USD/EUR/GBP/JPY positions — must normalize without crashing."""
    fx_map = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067}

    async def mock_fx(currency):
        return fx_map.get(currency, 1.0)

    with patch("src.agents.portfolio_health.get_current_price", new_callable=AsyncMock) as mock_price, \
         patch("src.agents.portfolio_health.get_fx_rate_to_usd", side_effect=mock_fx), \
         patch("src.agents.portfolio_health.get_benchmark_return", new_callable=AsyncMock) as mock_bm:

        mock_price.return_value = 200.0
        mock_bm.return_value = 10.0

        user = load_user("usr_006")
        result = await _run_agent(user)

    assert result is not None
    assert "disclaimer" in result


@pytest.mark.asyncio
async def test_portfolio_health_retiree_income_focus(load_user):
    """user_008 has income_focus=True — observations should mention income/yield."""
    with patch("src.agents.portfolio_health.get_current_price", new_callable=AsyncMock) as mock_price, \
         patch("src.agents.portfolio_health.get_fx_rate_to_usd", new_callable=AsyncMock) as mock_fx, \
         patch("src.agents.portfolio_health.get_benchmark_return", new_callable=AsyncMock) as mock_bm:

        mock_price.return_value = 100.0
        mock_fx.return_value = 1.0
        mock_bm.return_value = 8.0

        user = load_user("usr_008")
        result = await _run_agent(user)

    obs_texts = [o["text"].lower() for o in result.get("observations", [])]
    assert any(
        "income" in t or "yield" in t or "dividend" in t or "withdrawal" in t
        for t in obs_texts
    ), "No income-focused observation for retiree user"