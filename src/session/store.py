"""
SQLite-backed session store using aiosqlite.

Each turn is stored as: session_id, role, content, timestamp.
History is returned as a list of {role, content} dicts — ready to
inject directly into the OpenAI messages array.

Fix: open a fresh connection per operation — aiosqlite connections
cannot be reused across async context manager calls.
"""
from __future__ import annotations

import time
from pathlib import Path

import aiosqlite

from src.config import get_settings


async def _init_db(db: aiosqlite.Connection) -> None:
    """Create tables and indexes if they don't exist."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS session_turns (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
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


def _db_path() -> str:
    return str(Path(get_settings().session_db_path))


async def save_turn(session_id: str, role: str, content: str) -> None:
    """Append one turn to the session history."""
    async with aiosqlite.connect(_db_path()) as db:
        await _init_db(db)
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
    async with aiosqlite.connect(_db_path()) as db:
        await _init_db(db)
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
    async with aiosqlite.connect(_db_path()) as db:
        await _init_db(db)
        await db.execute(
            "DELETE FROM session_turns WHERE session_id = ?",
            (session_id,)
        )
        await db.commit()