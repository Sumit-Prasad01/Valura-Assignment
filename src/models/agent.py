from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

DISCLAIMER = (
    "This is not investment advice. The information provided is for informational "
    "purposes only and does not constitute financial, investment, legal, or tax advice. "
    "Past performance is not indicative of future results. Always consult a qualified "
    "financial adviser before making investment decisions."
)


class Observation(BaseModel):
    severity: Literal["info", "warning", "critical"]
    text: str


class ConcentrationRisk(BaseModel):
    top_position_pct: float
    top_3_positions_pct: float
    flag: Literal["low", "medium", "high"]


class Performance(BaseModel):
    total_return_pct: float
    annualized_return_pct: Optional[float] = None
    period_days: Optional[int] = None


class BenchmarkComparison(BaseModel):
    benchmark: str
    portfolio_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float


class PortfolioHealthOutput(BaseModel):
    concentration_risk: Optional[ConcentrationRisk] = None
    performance: Optional[Performance] = None
    benchmark_comparison: Optional[BenchmarkComparison] = None
    observations: list[Observation] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER
    build_guidance: Optional[str] = None


class StubAgentOutput(BaseModel):
    classified_intent: str
    extracted_entities: dict
    target_agent: str
    message: str = "This agent is not implemented in the current build."
    disclaimer: str = DISCLAIMER