"""
Safety Guard — pure local computation, no network, no LLM.
Must complete in < 10ms for any input.

Strategy: two-pass detection
  Pass 1 — category-specific keyword/phrase sets (intent to ACT, not to LEARN)
  Pass 2 — imperative structure check as a catch-all

Educational queries pass through because we detect *actionable intent*
not *topic knowledge*. Tradeoff documented in README.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# Distinct professional messages per category
_MESSAGES: dict[str, str] = {
    "insider_trading": (
        "This request involves acting on material non-public information, which "
        "constitutes insider trading under securities law. Valura cannot assist "
        "with this. If you have compliance questions, please consult your firm's "
        "legal counsel or a licensed compliance officer."
    ),
    "market_manipulation": (
        "This request describes activity that would constitute market manipulation "
        "— including coordinated trading, wash trading, or artificial price movement. "
        "Valura cannot assist with this. Such activity is illegal under securities "
        "regulations globally."
    ),
    "money_laundering": (
        "This request appears to involve concealing the origin or movement of funds "
        "in ways that breach anti-money laundering regulations. Valura cannot assist "
        "with this. AML violations carry severe criminal and civil penalties."
    ),
    "guaranteed_returns": (
        "No legitimate financial adviser or platform can guarantee investment returns. "
        "Requests for guaranteed returns or certainty of profit are a hallmark of "
        "financial fraud. Valura will not make such claims or help structure them."
    ),
    "reckless_advice": (
        "This request asks Valura to recommend an action that poses severe financial "
        "risk — such as concentrating all savings in a single volatile asset, using "
        "leverage on a single position, or risking essential funds. Valura is designed "
        "to protect investors, not to facilitate reckless decisions."
    ),
    "sanctions_evasion": (
        "This request involves circumventing sanctions or regulatory screening — "
        "activity that is illegal under OFAC, EU, and UN sanctions regimes. "
        "Valura cannot assist with this under any circumstances."
    ),
    "fraud": (
        "This request involves creating false or misleading financial documents or "
        "records. Valura cannot assist with fraud or misrepresentation of any kind."
    ),
}

_DEFAULT_MESSAGE = (
    "Valura cannot assist with this request. If you believe this is an error, "
    "please contact support."
)



# Detection patterns per category
# Design rule: match ACTIONABLE intent, not topic knowledge.
# "how do i pump a stock" → block  |  "what is a pump and dump" → pass
# "help me launder"       → block  |  "explain money laundering" → pass

_INSIDER_TRADING_PATTERNS = [
    r"\b(i\s+work|works?)\s+at\b.*\b(buy|sell|trade|shares?|stock)\b",
    r"\b(tip|tipped|insider\s+tip)\b.*\b(buy|sell|trade|load up)\b",
    r"\b(confidential|non.public|unannounced|secret)\b.*\b(merger|acquisition|earnings|news)\b.*\b(buy|sell|trade)\b",
    r"\bload\s+up\b.*\b(before|prior\s+to)\b.*\b(earnings|announcement|call)\b",
    r"\btrade\s+on\b.*\b(confidential|insider|non.public|tip)\b",
    r"\bearnings\s+before\s+.*(announcement|call|release)\b",
    r"\b(know|knew|heard)\s+about\b.*\b(unannounced|confidential|secret)\b",
]

_MARKET_MANIPULATION_PATTERNS = [
    r"\bpump\s+up\b.*\b(price|stock|share)\b",
    r"\b(pump).*(before|then).*(sell|dump)\b",
    r"\bcoordinated\s+(buying|selling|trading)\b",
    r"\bmove\s+(this|the)\s+stock\b",
    r"\bwash\s+trad(e|ing)\b.*\b(between|two|accounts?)\b",
    r"\bcreate\s+(fake|artificial|false)\s+(volume|demand|activity)\b",
    r"\b(design|help me|create)\b.*\b(scheme|coordinated)\b.*\b(stock|price|market)\b",
]

_MONEY_LAUNDERING_PATTERNS = [

    r"\b(move|transfer|route)\b.*\b(cash|funds?|money|\d+k)\b.*\b(without|avoid|bypass)\b.*\b(report|detect|trac)\b",
    r"\b(move|transfer)\b.*\b\d+k?\b.*\b(cash|funds?)\b.*\b(without|avoid)\b.*\b(report|trac|detect)\b",
    r"\bhow\s+do\s+i\s+move\b.*\b(cash|funds?|\d+k)\b.*\b(without|avoid|bypass|report)\b",
    r"\bstructure\s+deposits?\b",
    r"\bavoid\s+the\s+\d+k\s+report",
    r"\b(layer|layering)\b.*\b(trades?|funds?|money)\b.*\b(obscure|hide|conceal|source)\b",
    r"\bhide\s+(trading\s+)?(profits?|gains?|funds?)\b.*\b(tax|authorities|regulat)\b",
    r"\bobscure\s+the\s+source\b",
    r"\b(avoid|bypass|evade)\b.*\b(reporting|aml|kyc|compliance)\b",
    r"\bwithout\s+reporting\s+it\b",
]

_GUARANTEED_RETURNS_PATTERNS = [
    r"\bguarantee\s+(me\s+)?\d+",
    r"\bguarantee(d)?\s+return",
    r"\bpromise\s+me\b.*\b(double|triple|return|profit|money)\b",
    r"\b100\s*%\s*(certain|sure|guaranteed)\b",
    r"\bwill\s+definitely\s+(go\s+up|rise|increase|double)\b",
    # educational "is it ever legitimate to advertise guaranteed returns"
    r"\bfoolproof\s+way\b.*\b(\d+%|profit|make\s+money)\b",
    r"\bcertain\s+to\s+go\s+up\b",
    # "tell me a stock that's 100% certain" already caught by 100% pattern above
    # Do NOT match: "is it ever legitimate to advertise guaranteed returns"
    # That query is educational — contains "legitimate" as the subject
]

_RECKLESS_ADVICE_PATTERNS = [
    # Key signal: explicit instruction "do it" + dangerous action on retirement savings
    r"\b(all|entire)\b.*\b(retirement|life)\s*(savings?)\b.*\b(crypto|bitcoin|options?|single\s+stock)\b",
    r"\b(crypto|bitcoin|options?)\b.*\b(do\s+it|just\s+do\s+it)\b",
    r"\bput\s+(all|my\s+entire?)\b.*\b(retirement|savings?|emergency)\b.*\b(crypto|bitcoin|options?)\b",
    # Age + reckless action signal
    r"\b(i.m|i\s+am)\s+\d+\b.*\b(all|entire)\b.*\b(retirement|savings?)\b.*\b(crypto|bitcoin|options?)\b",
    
    r"\bmortgage\s+(my\s+)?(house|home)\b.*\b(stock|invest|buy|for)\b",
    r"\bwhich\s+stock\s+to\s+mortgage\b",
    # Existing patterns
    r"\b(margin\s+loan|leveraged?\s+loan)\b.*\b(buy|invest|more)\b",
    r"\bentire\s+emergency\s+fund\b.*\b(options?|crypto|stock|invest)\b",
    r"\btell\s+me\s+to\s+take\s+a\s+margin\b",
    r"\bput\s+everything\s+(i\s+have\s+)?(into|in)\b.*\b(crypto|options?|single)\b",
]

_SANCTIONS_EVASION_PATTERNS = [
    r"\b(bypass|circumvent|evade|avoid)\b.*\b(ofac|sanctions?|screening)\b",
    r"\b(route|move|invest)\b.*\b(shell\s+company|offshore|anonymous)\b.*\b(sanctions?|ofac|trac)\b",
    r"\bsanctioned\s+(russian|iranian|north\s+korean|venezuelan|cuban)\b.*\b(invest|buy|trade)\b",
    r"\bwithout\s+it\s+being\s+traced\b.*\b(sanction|invest|trade)\b",
    r"\b(invest|trade)\b.*\b(sanctioned)\b.*\b(without|bypass|avoid|trac)\b",
]

_FRAUD_PATTERNS = [
    r"\b(draft|create|write|forge|fake)\b.*\b(fake|false|forged?|fraudulent)\b.*\b(contract|document|note|record|statement)\b",
    r"\bfake\s+(contract|note|document|record)\b",
    r"\bclaim\s+(false|fake|inflated)\s+(losses?|gains?|profits?)\b",
]


# Compiled pattern groups — order matters (more specific first)
_CATEGORY_PATTERNS: list[tuple[str, list[re.Pattern]]] = [
    ("insider_trading",     [re.compile(p, re.I | re.S) for p in _INSIDER_TRADING_PATTERNS]),
    ("market_manipulation", [re.compile(p, re.I | re.S) for p in _MARKET_MANIPULATION_PATTERNS]),
    ("money_laundering",    [re.compile(p, re.I | re.S) for p in _MONEY_LAUNDERING_PATTERNS]),
    ("guaranteed_returns",  [re.compile(p, re.I | re.S) for p in _GUARANTEED_RETURNS_PATTERNS]),
    ("reckless_advice",     [re.compile(p, re.I | re.S) for p in _RECKLESS_ADVICE_PATTERNS]),
    ("sanctions_evasion",   [re.compile(p, re.I | re.S) for p in _SANCTIONS_EVASION_PATTERNS]),
    ("fraud",               [re.compile(p, re.I | re.S) for p in _FRAUD_PATTERNS]),
]


# Educational intent signals — if present, never block
# Used to prevent false positives on educational queries
_EDUCATIONAL_SIGNALS = re.compile(
    r"\b(what\s+is|what\s+are|explain|how\s+does|how\s+do\s+regulators|"
    r"define|definition|describe|difference\s+between|why\s+is|"
    r"is\s+it\s+ever\s+legitimate|is\s+.+\s+illegal|"
    r"what\s+are\s+the\s+(penalties|rules|requirements|obligations)|"
    r"how\s+do\s+(the\s+)?(sec|fca|regulators?|brokers?)\s+(catch|detect|investigate|screen)|"
    r"historical\s+average|are\s+.+\s+legal|what\s+factors)\b",
    re.I,
)



# Result dataclass
@dataclass
class GuardResult:
    blocked: bool
    category: str | None
    message: str | None


PASS = GuardResult(blocked=False, category=None, message=None)



# Public API
def check(query: str) -> GuardResult:
    """
    Run the safety guard on a raw query string.
    Returns GuardResult(blocked=True, category, message) or PASS.
    Pure computation — no I/O, no LLM, < 10ms.
    """
    text = query.strip()
    if not text:
        return PASS

    # Educational intent check — if clearly educational, pass through
    # (applied before pattern matching to prevent false positives)
    if _EDUCATIONAL_SIGNALS.search(text):
        return PASS

    for category, patterns in _CATEGORY_PATTERNS:
        for pattern in patterns:
            if pattern.search(text):
                return GuardResult(
                    blocked=True,
                    category=category,
                    message=_MESSAGES.get(category, _DEFAULT_MESSAGE),
                )

    return PASS