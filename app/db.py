from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, create_engine, delete, insert, select, update
from sqlalchemy.engine import Connection, Engine


DEFAULT_SQLITE_PATH = Path(os.getenv("SQLITE_DB_PATH", "data/app.db"))
DEFAULT_DATABASE_URL = f"sqlite+pysqlite:///{DEFAULT_SQLITE_PATH}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
CLARIFICATION_TTL_MINUTES = int(os.getenv("CLARIFICATION_TTL_MINUTES", "30"))

metadata = MetaData()

paid_users = Table(
    "paid_users",
    metadata,
    Column("phone", String, primary_key=True),
    Column("date_paid", DateTime(timezone=True), nullable=False),
)

usage_counters = Table(
    "usage_counters",
    metadata,
    Column("phone", String, primary_key=True),
    Column("free_questions_used", Integer, nullable=False, default=0),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

conversation_state = Table(
    "conversation_state",
    metadata,
    Column("phone", String, primary_key=True),
    Column("pending_question", String, nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

_engine: Engine | None = None


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
        if DATABASE_URL.startswith("sqlite"):
            DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(DATABASE_URL, future=True, connect_args=connect_args)
    return _engine


@contextmanager
def get_connection() -> Iterator[Connection]:
    with _get_engine().begin() as connection:
        yield connection


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def init_db() -> None:
    metadata.create_all(_get_engine())


def _row_exists(connection: Connection, table: Table, phone: str) -> bool:
    return connection.execute(select(table.c.phone).where(table.c.phone == phone)).first() is not None


def is_paid_user(phone: str) -> bool:
    with get_connection() as connection:
        return _row_exists(connection, paid_users, phone)


def mark_paid(phone: str, paid_at: str | None = None) -> None:
    timestamp = datetime.fromisoformat(paid_at) if paid_at else _utcnow()
    with get_connection() as connection:
        if _row_exists(connection, paid_users, phone):
            connection.execute(
                update(paid_users).where(paid_users.c.phone == phone).values(date_paid=timestamp)
            )
        else:
            connection.execute(insert(paid_users).values(phone=phone, date_paid=timestamp))


def get_free_question_count(phone: str) -> int:
    with get_connection() as connection:
        row = connection.execute(
            select(usage_counters.c.free_questions_used).where(usage_counters.c.phone == phone)
        ).first()
    return int(row[0]) if row else 0


def increment_free_question_count(phone: str) -> int:
    now = _utcnow()
    with get_connection() as connection:
        row = connection.execute(
            select(usage_counters.c.free_questions_used).where(usage_counters.c.phone == phone)
        ).first()
        new_value = (int(row[0]) if row else 0) + 1
        if row:
            connection.execute(
                update(usage_counters)
                .where(usage_counters.c.phone == phone)
                .values(free_questions_used=new_value, updated_at=now)
            )
        else:
            connection.execute(
                insert(usage_counters).values(
                    phone=phone,
                    free_questions_used=new_value,
                    updated_at=now,
                )
            )
    return new_value


def get_pending_clarification(phone: str) -> str | None:
    now = _utcnow()
    with get_connection() as connection:
        row = connection.execute(
            select(conversation_state.c.pending_question, conversation_state.c.expires_at).where(
                conversation_state.c.phone == phone
            )
        ).first()
        if not row:
            return None
        pending_question, expires_at = row
        if _coerce_utc(expires_at) <= now:
            connection.execute(delete(conversation_state).where(conversation_state.c.phone == phone))
            return None
    return str(pending_question)


def set_pending_clarification(phone: str, pending_question: str) -> None:
    now = _utcnow()
    expires_at = now + timedelta(minutes=CLARIFICATION_TTL_MINUTES)
    with get_connection() as connection:
        if _row_exists(connection, conversation_state, phone):
            connection.execute(
                update(conversation_state)
                .where(conversation_state.c.phone == phone)
                .values(
                    pending_question=pending_question,
                    expires_at=expires_at,
                    updated_at=now,
                )
            )
        else:
            connection.execute(
                insert(conversation_state).values(
                    phone=phone,
                    pending_question=pending_question,
                    expires_at=expires_at,
                    updated_at=now,
                )
            )


def clear_pending_clarification(phone: str) -> None:
    with get_connection() as connection:
        connection.execute(delete(conversation_state).where(conversation_state.c.phone == phone))
