"""
Entity matcher — implements the normalization rules from fixtures/README.md.

Used by test_classifier_routing.py and any future routing tests.

Rules:
  tickers      → case-fold, exchange suffix optional (AAPL matches aapl, AAPL.US)
  topics/sectors → case-fold, substring match per element
  amount/rate  → within ±5%
  period_years → exact integer
  currency     → ISO 4217 exact (case-insensitive)
  frequency    → vocabulary token exact (case-insensitive)
  horizon      → vocabulary token exact (case-insensitive)
  time_period  → vocabulary token exact (case-insensitive)
  index        → exact match against canonical names (case-insensitive)
  action       → vocabulary token exact (case-insensitive)
  goal         → vocabulary token exact (case-insensitive)
  intent       → substring match (case-insensitive)
"""
from __future__ import annotations
from typing import Any


def _normalize_ticker(t: str) -> str:
    """AAPL.US → AAPL, asml.as → ASML"""
    return t.upper().split(".")[0]


def matches_entities(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    """
    Subset match with normalization.
    actual must contain every field+value in expected.
    Extra fields/values in actual are allowed.
    Returns True if all expected fields are satisfied.
    """
    for field, exp_value in expected.items():
        act_value = actual.get(field)

        # Missing field in actual
        if act_value is None and exp_value not in (None, [], {}):
            return False

        if field == "tickers":
            exp_set = {_normalize_ticker(t) for t in exp_value}
            act_set = {_normalize_ticker(t) for t in (act_value or [])}
            if not exp_set.issubset(act_set):
                return False

        elif field in ("topics", "sectors"):
            # Substring match — every expected element must appear as
            # a substring of at least one actual element (case-insensitive)
            act_lower = [s.lower() for s in (act_value or [])]
            for exp_item in exp_value:
                exp_lower = exp_item.lower()
                if not any(exp_lower in a for a in act_lower):
                    return False

        elif field in ("amount", "rate"):
            if act_value is None:
                return False
            tolerance = abs(exp_value) * 0.05
            if abs(float(act_value) - float(exp_value)) > tolerance:
                return False

        elif field == "period_years":
            if int(act_value) != int(exp_value):
                return False

        elif field == "currency":
            if str(act_value).upper() != str(exp_value).upper():
                return False

        elif field == "index":
            # Canonical names: S&P 500, FTSE 100, NIKKEI 225, MSCI World
            # Tolerant of spacing differences
            def _norm_index(s: str) -> str:
                return s.upper().replace(" ", "").replace("&", "AND")
            if _norm_index(str(act_value)) != _norm_index(str(exp_value)):
                return False

        elif field == "intent":
            # Substring match for intent field
            if str(exp_value).lower() not in str(act_value).lower():
                return False

        else:
            # Vocabulary tokens: action, goal, frequency, horizon, time_period
            if str(act_value).lower() != str(exp_value).lower():
                return False

    return True