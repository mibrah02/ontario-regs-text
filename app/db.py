from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


DB_PATH = Path(os.getenv("SQLITE_DB_PATH", "data/app.db"))


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    try:
        yield connection
    finally:
        connection.close()


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS paid_users (
                phone TEXT PRIMARY KEY,
                date_paid TEXT NOT NULL
            )
            """
        )
        connection.commit()


def is_paid_user(phone: str) -> bool:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT phone FROM paid_users WHERE phone = ?",
            (phone,),
        ).fetchone()
    return row is not None


def mark_paid(phone: str, paid_at: str | None = None) -> None:
    timestamp = paid_at or datetime.now(timezone.utc).isoformat()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO paid_users (phone, date_paid)
            VALUES (?, ?)
            ON CONFLICT(phone) DO UPDATE SET date_paid = excluded.date_paid
            """,
            (phone, timestamp),
        )
        connection.commit()

