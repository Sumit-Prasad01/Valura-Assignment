"""
Market data wrapper around yfinance.
This is the ONLY file that touches yfinance — swap for MCP later with no other changes.

Never hardcodes prices, sectors, or benchmark returns.
All values are fetched live.
"""
from __future__ import annotations
import asyncio
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

import yfinance as yf


# Benchmark tickers for each canonical index name
BENCHMARK_TICKERS: dict[str, str] = {
    "S&P 500":    "^GSPC",
    "QQQ":        "QQQ",
    "FTSE 100":   "^FTSE",
    "NIKKEI 225": "^N225",
    "MSCI World": "URTH",   # iShares MSCI World ETF as proxy
}

# FX rates: base→USD via Yahoo Finance pairs
FX_TO_USD: dict[str, str] = {
    "USD": None,       # no conversion needed
    "EUR": "EURUSD=X",
    "GBP": "GBPUSD=X",
    "JPY": "JPYUSD=X",
    "SGD": "SGDUSD=X",
}


def _run_sync(coro):
    """Run an async coroutine from sync context safely."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


async def get_current_price(ticker: str) -> Optional[float]:
    """Fetch the latest close price for a ticker."""
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None, lambda: yf.Ticker(ticker).fast_info
        )
        price = getattr(data, "last_price", None)
        if price is None:
            hist = await loop.run_in_executor(
                None, lambda: yf.Ticker(ticker).history(period="2d")
            )
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        return float(price) if price else None
    except Exception:
        return None


async def get_fx_rate_to_usd(currency: str) -> float:
    """Return the FX rate currency→USD. Returns 1.0 for USD."""
    if currency == "USD":
        return 1.0
    pair = FX_TO_USD.get(currency)
    if not pair:
        return 1.0  # unknown currency — treat as 1:1, log in prod
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None, lambda: yf.Ticker(pair).fast_info
        )
        rate = getattr(data, "last_price", None)
        return float(rate) if rate else 1.0
    except Exception:
        return 1.0


async def get_benchmark_return(benchmark_name: str, since: datetime) -> Optional[float]:
    """
    Return the total return % of a benchmark since a given date.
    benchmark_name must be one of BENCHMARK_TICKERS keys.
    """
    ticker_sym = BENCHMARK_TICKERS.get(benchmark_name)
    if not ticker_sym:
        return None
    try:
        loop = asyncio.get_event_loop()
        hist = await loop.run_in_executor(
            None,
            lambda: yf.Ticker(ticker_sym).history(
                start=since.strftime("%Y-%m-%d"),
                end=datetime.today().strftime("%Y-%m-%d"),
            ),
        )
        if hist.empty or len(hist) < 2:
            return None
        start_price = float(hist["Close"].iloc[0])
        end_price = float(hist["Close"].iloc[-1])
        return round((end_price - start_price) / start_price * 100, 2)
    except Exception:
        return None


async def get_ticker_return(ticker: str, since: datetime) -> Optional[float]:
    """Return total return % for a ticker since a given date."""
    try:
        loop = asyncio.get_event_loop()
        hist = await loop.run_in_executor(
            None,
            lambda: yf.Ticker(ticker).history(
                start=since.strftime("%Y-%m-%d"),
                end=datetime.today().strftime("%Y-%m-%d"),
            ),
        )
        if hist.empty or len(hist) < 2:
            return None
        start_price = float(hist["Close"].iloc[0])
        end_price = float(hist["Close"].iloc[-1])
        return round((end_price - start_price) / start_price * 100, 2)
    except Exception:
        return None