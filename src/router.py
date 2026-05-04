"""
Router — maps agent name → agent instance.

Adding a new agent:
  1. Create src/agents/my_agent.py subclassing BaseAgent
  2. Add one line to _REGISTRY below
  That's it. Nothing else changes.

All unregistered agents fall through to StubAgent automatically.
"""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.portfolio_health import PortfolioHealthAgent
from src.agents.stub import StubAgent


# Registry — agent name (matches classifier taxonomy) → agent instance
# Agents are singletons — instantiated once at import time

_REGISTRY: dict[str, BaseAgent] = {
    "portfolio_health": PortfolioHealthAgent(),
    "portfolio_query":  PortfolioHealthAgent(),  # follow-up ownership queries
}

_STUB = StubAgent()


def get_agent(agent_name: str) -> BaseAgent:
    """
    Return the agent instance for the given name.
    Falls back to StubAgent for any unimplemented agent.
    Never raises.
    """
    return _REGISTRY.get(agent_name, _STUB)