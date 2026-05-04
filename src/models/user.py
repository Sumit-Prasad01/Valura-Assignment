from __future__ import annotations
from datetime import date
from typing import Optional
from pydantic import BaseModel, Field


class KYC(BaseModel):
    status: str


class Position(BaseModel):
    ticker: str
    exchange: str
    quantity: float
    avg_cost: float
    currency: str
    purchased_at: date


class Preferences(BaseModel):
    preferred_benchmark: str = "S&P 500"
    reporting_currency: Optional[str] = None
    income_focus: Optional[bool] = False


class UserProfile(BaseModel):
    user_id: str
    name: str
    age: int
    country: str
    base_currency: str
    kyc: KYC
    risk_profile: str
    positions: list[Position] = Field(default_factory=list)
    preferences: Preferences = Field(default_factory=Preferences)

    @property
    def has_positions(self) -> bool:
        return len(self.positions) > 0