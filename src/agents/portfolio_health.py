"""
Portfolio Health Agent — fully implemented.

Handles: "how is my portfolio doing", "health check", "am I diversified?",
         "what's my concentration risk?", "am I beating the market?"

For empty portfolios (usr_004): pivots to BUILD guidance instead of erroring.

Output shape matches the reference in ASSIGNMENT.md, extended with:
  - multi-currency normalization to base_currency
  - income-focus commentary for retirees
  - plain-language observations (novice-friendly)

Market data comes exclusively from src/market/data.py — nothing hardcoded.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncIterator, Optional

from src.agents.base import BaseAgent
from src.market.data import (
    get_benchmark_return,
    get_current_price,
    get_fx_rate_to_usd,
    BENCHMARK_TICKERS,
)
from src.models.agent import (
    DISCLAIMER,
    BenchmarkComparison,
    ConcentrationRisk,
    Observation,
    Performance,
    PortfolioHealthOutput,
)
from src.models.classifier import ClassifierOutput
from src.models.user import UserProfile, Position
from src.utils.logger import logger


# Concentration thresholds
_HIGH_CONCENTRATION = 40.0   # single position > 40% → high
_MEDIUM_CONCENTRATION = 20.0  # single position > 20% → medium


# Portfolio math helpers
async def _resolve_position_value_usd(pos: Position) -> tuple[str, float]:
    """
    Returns (ticker, current_value_in_usd).
    Fetches live price and FX rate concurrently.
    Falls back to cost basis if price fetch fails.
    """
    price_task = asyncio.create_task(get_current_price(pos.ticker))
    fx_task = asyncio.create_task(get_fx_rate_to_usd(pos.currency))

    price, fx_rate = await asyncio.gather(price_task, fx_task)

    if price is None:
        # Fallback to cost basis
        price = pos.avg_cost
        logger.warning("Price fetch failed for %s — using cost basis", pos.ticker)

    value_local = price * pos.quantity
    value_usd = value_local * fx_rate
    return pos.ticker, value_usd


async def _compute_portfolio_values(
    positions: list[Position],
) -> dict[str, float]:
    """Fetch all position values concurrently. Returns {ticker: usd_value}."""
    tasks = [_resolve_position_value_usd(pos) for pos in positions]
    results = await asyncio.gather(*tasks)
    return dict(results)


def _compute_concentration(values: dict[str, float]) -> ConcentrationRisk:
    total = sum(values.values())
    if total == 0:
        return ConcentrationRisk(
            top_position_pct=0, top_3_positions_pct=0, flag="low"
        )

    sorted_vals = sorted(values.values(), reverse=True)
    top1_pct = round(sorted_vals[0] / total * 100, 1)
    top3_pct = round(sum(sorted_vals[:3]) / total * 100, 1)

    if top1_pct >= _HIGH_CONCENTRATION:
        flag = "high"
    elif top1_pct >= _MEDIUM_CONCENTRATION:
        flag = "medium"
    else:
        flag = "low"

    return ConcentrationRisk(
        top_position_pct=top1_pct,
        top_3_positions_pct=top3_pct,
        flag=flag,
    )


async def _compute_performance(
    positions: list[Position],
    values: dict[str, float],
) -> Performance:
    """
    Compute total return % vs weighted average cost basis.
    Also computes annualized return using earliest purchase date.
    """
    total_cost_usd = 0.0
    earliest_date = datetime.today()

    fx_cache: dict[str, float] = {}

    for pos in positions:
        if pos.currency not in fx_cache:
            fx_cache[pos.currency] = await get_fx_rate_to_usd(pos.currency)
        fx = fx_cache[pos.currency]
        total_cost_usd += pos.avg_cost * pos.quantity * fx

        purchase_dt = datetime.combine(pos.purchased_at, datetime.min.time())
        if purchase_dt < earliest_date:
            earliest_date = purchase_dt

    total_value_usd = sum(values.values())
    total_return_pct = (
        round((total_value_usd - total_cost_usd) / total_cost_usd * 100, 2)
        if total_cost_usd > 0 else 0.0
    )

    days_held = max((datetime.today() - earliest_date).days, 1)
    years_held = days_held / 365.25

    if years_held >= 0.1:
        annualized = round(
            ((1 + total_return_pct / 100) ** (1 / years_held) - 1) * 100, 2
        )
    else:
        annualized = None

    return Performance(
        total_return_pct=total_return_pct,
        annualized_return_pct=annualized,
        period_days=days_held,
    )


async def _compute_benchmark(
    portfolio_return_pct: float,
    benchmark_name: str,
    earliest_date: datetime,
) -> Optional[BenchmarkComparison]:
    """Fetch benchmark return over same period as portfolio."""
    bm_return = await get_benchmark_return(benchmark_name, earliest_date)
    if bm_return is None:
        return None

    return BenchmarkComparison(
        benchmark=benchmark_name,
        portfolio_return_pct=portfolio_return_pct,
        benchmark_return_pct=bm_return,
        alpha_pct=round(portfolio_return_pct - bm_return, 2),
    )


# Observation builder
def _build_observations(
    user: UserProfile,
    values: dict[str, float],
    concentration: ConcentrationRisk,
    performance: Optional[Performance],
    benchmark: Optional[BenchmarkComparison],
) -> list[Observation]:
    obs: list[Observation] = []
    total = sum(values.values())
    sorted_positions = sorted(values.items(), key=lambda x: x[1], reverse=True)

    # --- Concentration warnings ---
    if concentration.flag == "high":
        top_ticker = sorted_positions[0][0]
        obs.append(Observation(
            severity="warning",
            text=(
                f"{concentration.top_position_pct}% of your portfolio is in {top_ticker}. "
                f"This is highly concentrated — a sharp move in one stock could "
                f"significantly impact your total wealth."
            ),
        ))
    elif concentration.flag == "medium":
        top_ticker = sorted_positions[0][0]
        obs.append(Observation(
            severity="info",
            text=(
                f"Your largest position ({top_ticker}) makes up "
                f"{concentration.top_position_pct}% of your portfolio. "
                f"Consider whether this aligns with your risk profile."
            ),
        ))

    # --- Performance vs benchmark ---
    if benchmark:
        if benchmark.alpha_pct > 0:
            obs.append(Observation(
                severity="info",
                text=(
                    f"You are outperforming {benchmark.benchmark} by "
                    f"{benchmark.alpha_pct:.1f}% over your holding period."
                ),
            ))
        else:
            obs.append(Observation(
                severity="info",
                text=(
                    f"You are underperforming {benchmark.benchmark} by "
                    f"{abs(benchmark.alpha_pct):.1f}% over your holding period. "
                    f"This may be worth reviewing."
                ),
            ))

    # --- Risk profile mismatches ---
    if user.risk_profile == "conservative" and concentration.flag == "high":
        obs.append(Observation(
            severity="warning",
            text=(
                "Your risk profile is conservative, but your portfolio is highly "
                "concentrated. This mismatch increases your downside exposure."
            ),
        ))

    # --- Income focus (retiree) ---
    if getattr(user.preferences, "income_focus", False):
        obs.append(Observation(
            severity="info",
            text=(
                "Your portfolio includes dividend-focused holdings. "
                "Make sure your income yield covers your withdrawal rate — "
                "a 4% withdrawal rule is a common starting point for retirees."
            ),
        ))

    # --- Negative return warning ---
    if performance and performance.total_return_pct < -10:
        obs.append(Observation(
            severity="warning",
            text=(
                f"Your portfolio is down {abs(performance.total_return_pct):.1f}% "
                f"overall. Review whether your current allocation still fits "
                f"your investment horizon."
            ),
        ))

    return obs


# Empty portfolio handler (BUILD mode)
_BUILD_GUIDANCE = {
    "conservative": (
        "Since you have no investments yet, here's a starting point for a "
        "conservative investor: consider a core allocation of 60–70% bonds "
        "(e.g. BND or TLT for US exposure), and 30–40% broad equity ETFs "
        "(e.g. VOO or VTI). This balances capital preservation with moderate growth."
    ),
    "moderate": (
        "Since you have no investments yet, a balanced starting point for a "
        "moderate risk investor might be: 60% broad equity ETFs (e.g. VOO, VTI, VXUS "
        "for international exposure) and 40% bonds (e.g. BND). "
        "Consider adding to this regularly via dollar-cost averaging."
    ),
    "aggressive": (
        "Since you have no investments yet, an aggressive investor might start with "
        "80–100% in equity ETFs — a mix of US (e.g. QQQ, VTI) and international "
        "(e.g. VXUS). Be prepared for significant short-term volatility in exchange "
        "for higher long-term growth potential."
    ),
}


# Agent
class PortfolioHealthAgent(BaseAgent):

    async def run(
        self,
        query: str,
        classification: ClassifierOutput,
        user: UserProfile,
        history: list[dict],
    ) -> AsyncIterator[str]:
        try:
            async for chunk in self._run(user):
                yield chunk
        except Exception as exc:
            logger.exception("PortfolioHealthAgent error: %s", exc)
            yield json.dumps({
                "error": "Portfolio health check failed unexpectedly.",
                "disclaimer": DISCLAIMER,
            })

    async def _run(self, user: UserProfile) -> AsyncIterator[str]:
        # --- Empty portfolio: BUILD mode ---
        if not user.has_positions:
            guidance = _BUILD_GUIDANCE.get(user.risk_profile, _BUILD_GUIDANCE["moderate"])
            output = PortfolioHealthOutput(
                observations=[
                    Observation(
                        severity="info",
                        text=(
                            f"Welcome, {user.name}! Your account is verified and "
                            f"ready to go, but you don't have any positions yet."
                        ),
                    )
                ],
                build_guidance=guidance,
                disclaimer=DISCLAIMER,
            )
            yield json.dumps(output.model_dump(exclude_none=True), indent=2)
            return

        # --- Fetch all position values concurrently ---
        yield '{"status": "fetching market data..."}\n'
        values = await _compute_portfolio_values(user.positions)

        # --- Concentration ---
        concentration = _compute_concentration(values)

        # --- Performance ---
        performance = await _compute_performance(user.positions, values)

        # --- Benchmark ---
        benchmark_name = user.preferences.preferred_benchmark
        # Map user preference to canonical benchmark name if needed
        if benchmark_name not in BENCHMARK_TICKERS:
            benchmark_name = "S&P 500"

        earliest_date = min(
            datetime.combine(p.purchased_at, datetime.min.time())
            for p in user.positions
        )
        benchmark = await _compute_benchmark(
            performance.total_return_pct, benchmark_name, earliest_date
        )

        # --- Observations ---
        observations = _build_observations(
            user, values, concentration, performance, benchmark
        )

        # --- Final output ---
        output = PortfolioHealthOutput(
            concentration_risk=concentration,
            performance=performance,
            benchmark_comparison=benchmark,
            observations=observations,
            disclaimer=DISCLAIMER,
        )

        yield json.dumps(output.model_dump(exclude_none=True), indent=2)