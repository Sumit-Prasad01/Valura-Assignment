from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    session_id: str
    user_id: str
    user_profile: dict


class SSEEvent(BaseModel):
    event: str
    data: Any


class ErrorEvent(BaseModel):
    event: str = "error"
    code: str
    message: str
    detail: Optional[str] = None