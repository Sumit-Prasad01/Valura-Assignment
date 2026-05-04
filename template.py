import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

project_name = "valura-ai"

list_of_files = [

    # --- FIXTURES (provided, created empty so structure is clear) ---
    "fixtures/README.md",
    "fixtures/users/user_001_active_trader_us.json",
    "fixtures/users/user_003_concentrated.json",
    "fixtures/users/user_004_empty.json",
    "fixtures/users/user_006_multi_currency.json",
    "fixtures/users/user_008_retiree.json",
    "fixtures/conversations/follow_up_session.json",
    "fixtures/conversations/multi_intent_session.json",
    "fixtures/conversations/ambiguous_session.json",
    "fixtures/test_queries/intent_classification.json",
    "fixtures/test_queries/safety_pairs.json",

    # --- SOURCE ---
    "src/__init__.py",

    # Entry point
    "src/main.py",

    # Config
    "src/config.py",

    # Pipeline orchestrator (guard → classify → route → stream)
    "src/pipeline.py",

    # Router (agent name → agent instance)
    "src/router.py",

    # Helper files
    "src/utils/__init__.py",
    "src/utils/logger.py",

    # Pydantic models — no logic, only schemas
    "src/models/__init__.py",
    "src/models/user.py",           # UserProfile, Portfolio, Holding
    "src/models/classifier.py",     # ClassifierOutput, Intent, ExtractedEntities
    "src/models/agent.py",          # AgentResponse, Observation, PortfolioHealthOutput
    "src/models/api.py",            # QueryRequest, SSEEvent, ErrorEvent

    # Safety guard — pure local, zero network, <10ms
    "src/safety/__init__.py",
    "src/safety/guard.py",          # GuardResult(blocked, category, message)

    # Intent classifier — single LLM call, structured output
    "src/classifier/__init__.py",
    "src/classifier/intent.py",     # ClassifierOutput, follow-up resolution, fallback

    # Agents
    "src/agents/__init__.py",
    "src/agents/base.py",           # Abstract BaseAgent
    "src/agents/portfolio_health.py",  # Fully implemented agent
    "src/agents/stub.py",           # StubAgent for unimplemented agents

    # Session store — SQLite via aiosqlite
    "src/session/__init__.py",
    "src/session/store.py",         # CRUD: save_turn, get_history, clear_session

    # Market data — yfinance wrapper, never hardcodes prices
    "src/market/__init__.py",
    "src/market/data.py",           # get_price, get_returns, get_benchmark_return

    # --- TESTS ---
    "tests/__init__.py",
    "tests/conftest.py",            # Fixtures, mock LLM, sample users
    "tests/test_safety_pairs.py",   # Tests safety_pairs.json gold set
    "tests/test_classifier_routing.py",       # Tests intent_classification.json gold set
    "tests/test_portfolio_health_skeleton.py", # Portfolio health contract tests
    "tests/helpers/__init__.py",
    "tests/helpers/entity_matcher.py",  # Subset match + normalization per fixtures/README

    # --- CONFIG ---
    "requirements.txt",
    ".env.example",
    ".gitignore",
    "pytest.ini",
    "README.md",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    if not os.path.exists(filepath):
        with open(filepath, "w"):
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"Already exists, skipping: {filename}")

logging.info("Valura AI project structure created successfully.")