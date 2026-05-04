"""
Abstract base class for all Valura agents.

Every agent must implement run() which returns an async generator
yielding string chunks for SSE streaming.

Adding a new agent = create one file, subclass BaseAgent, register in router.py.
No other files change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from src.models.classifier import ClassifierOutput
from src.models.user import UserProfile


class BaseAgent(ABC):
    """
    All agents implement this interface.
    run() is an async generator — yields text chunks as they are produced.
    The pipeline streams each chunk to the client via SSE.
    """

    @abstractmethod
    async def run(
        self,
        query: str,
        classification: ClassifierOutput,
        user: UserProfile,
        history: list[dict],
    ) -> AsyncIterator[str]:
        """
        Yield text chunks for SSE streaming.
        Must never raise — catch internally and yield an error message.
        """
        ...  # pragma: no cover