from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class ExtractedEntities(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    amount: Optional[float] = None
    currency: Optional[str] = None
    rate: Optional[float] = None
    period_years: Optional[int] = None
    frequency: Optional[str] = None
    horizon: Optional[str] = None
    time_period: Optional[str] = None
    topics: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    index: Optional[str] = None
    action: Optional[str] = None
    goal: Optional[str] = None
    intent: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None and v != []}


class ClassifierOutput(BaseModel):
    agent: str
    intent: str
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    safety_verdict: str = "pass"
    safety_reason: Optional[str] = None
    confidence: float = 1.0

    @classmethod
    def fallback(cls, query: str) -> "ClassifierOutput":
        return cls(
            agent="customer_support",
            intent="unknown",
            entities=ExtractedEntities(),
            safety_verdict="pass",
            confidence=0.0,
        )