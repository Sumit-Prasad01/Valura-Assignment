"""
StubAgent — handles all agents not yet implemented.

Returns a clean structured response that includes:
  - classified intent
  - extracted entities
  - which agent would have handled this
  - a short message

Never crashes. Never returns a raw error.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

from src.agents.base import BaseAgent
from src.models.agent import DISCLAIMER, StubAgentOutput
from src.models.classifier import ClassifierOutput
from src.models.user import UserProfile


class StubAgent(BaseAgent):
    async def run(
        self,
        query: str,
        classification: ClassifierOutput,
        user: UserProfile,
        history: list[dict],
    ) -> AsyncIterator[str]:
        output = StubAgentOutput(
            classified_intent=classification.intent,
            extracted_entities=classification.entities.to_dict(),
            target_agent=classification.agent,
            message=(
                f"The '{classification.agent}' agent is not implemented in this build. "
                f"Your query has been classified correctly and would be routed here "
                f"in the full system."
            ),
            disclaimer=DISCLAIMER,
        )
        yield json.dumps(output.model_dump(), indent=2)