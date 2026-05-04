"""
SQLite-backed session store using aiosqlite.

Each turn is stored as: session_id, role, content, timestamp.
History is returned as a list of {role, content} dicts — ready to
inject directly into the OpenAI messages array.

Choice of SQLite over in-memory:
  - Persists across process restarts (useful for demo)
  - Zero infra — single file, no Postgres needed
  - aiosqlite is fully async, no event loop blocking
  - Tradeoff: not horizontally scalable; fine for this build
"""
from __future__ import annotations
import json
import time
import aiosqlite
from pathlib import Path
from src.config import get_settings


async def _get_db() -> aiosqlite.Connection:
    settings = get_settings()
    db_path = Path(settings.session_db_path)
    db = await aiosqlite.connect(str(db_path))
    await db.execute("""
        CREATE TABLE IF NOT EXISTS session_turns (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            ts         REAL NOT NULL
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_session ON session_turns(session_id, ts)"
    )
    await db.commit()
    return db


async def save_turn(session_id: str, role: str, content: str) -> None:
    """Append one turn to the session history."""
    db = await _get_db()
    async with db:
        await db.execute(
            "INSERT INTO session_turns (session_id, role, content, ts) VALUES (?,?,?,?)",
            (session_id, role, content, time.time()),
        )
        await db.commit()


async def get_history(session_id: str, max_turns: int = 10) -> list[dict]:
    """
    Return the last `max_turns` turns for the session.
    Format: [{"role": "user"|"assistant", "content": "..."}]
    """
    db = await _get_db()
    async with db:
        cursor = await db.execute(
            """
            SELECT role, content FROM session_turns
            WHERE session_id = ?
            ORDER BY ts DESC
            LIMIT ?
            """,
            (session_id, max_turns),
        )
        rows = await cursor.fetchall()
    # Reverse so oldest is first (chronological for LLM context)
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


async def clear_session(session_id: str) -> None:
    """Delete all turns for a session."""
    db = await _get_db()
    async with db:
        await db.execute(
            "DELETE FROM session_turns WHERE session_id = ?", (session_id,)
        )
        await db.commit()