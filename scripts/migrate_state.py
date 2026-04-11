from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import and_, create_engine, insert, inspect, select, update
from sqlalchemy.engine import Engine, make_url

from app.db import (
    DEFAULT_DATABASE_URL,
    DEFAULT_SQLITE_PATH,
    conversation_state,
    metadata,
    paid_users,
    processed_events,
    sms_messages,
    usage_counters,
)

TABLES = [
    paid_users,
    usage_counters,
    conversation_state,
    sms_messages,
    processed_events,
]


def _normalize_database_url(raw_url: str) -> str:
    if raw_url.startswith("postgres://"):
        return f"postgresql+psycopg://{raw_url[len('postgres://') :]}"
    if raw_url.startswith("postgresql://"):
        return f"postgresql+psycopg://{raw_url[len('postgresql://') :]}"
    return raw_url


def _ensure_sqlite_directory(database_url: str) -> None:
    if not database_url.startswith("sqlite"):
        return
    database = make_url(database_url).database
    if not database or database == ":memory:":
        return
    Path(database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _create_engine(database_url: str) -> Engine:
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    _ensure_sqlite_directory(database_url)
    return create_engine(database_url, future=True, connect_args=connect_args, pool_pre_ping=True)


def _upsert_rows(source_engine: Engine, target_engine: Engine) -> None:
    metadata.create_all(target_engine)
    source_inspector = inspect(source_engine)

    with source_engine.connect() as source_connection, target_engine.begin() as target_connection:
        for table in TABLES:
            if not source_inspector.has_table(table.name):
                print(f"{table.name}: skipped (missing in source)")
                continue

            rows = [dict(row._mapping) for row in source_connection.execute(select(table)).fetchall()]
            copied = 0
            for row in rows:
                filters = and_(*[column == row[column.name] for column in table.primary_key.columns])
                existing = target_connection.execute(select(table).where(filters)).first()
                if existing:
                    target_connection.execute(update(table).where(filters).values(**row))
                else:
                    target_connection.execute(insert(table).values(**row))
                copied += 1
            print(f"{table.name}: copied {copied} rows")


def main() -> None:
    source_url = _normalize_database_url(os.getenv("SOURCE_DATABASE_URL", DEFAULT_DATABASE_URL))
    target_url = _normalize_database_url(os.getenv("TARGET_DATABASE_URL", os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)))

    default_sqlite_url = f"sqlite+pysqlite:///{DEFAULT_SQLITE_PATH}"
    if source_url == target_url:
        raise SystemExit(
            "Source and target database URLs are the same. Set TARGET_DATABASE_URL to your Postgres database before running this migration."
        )
    if target_url == default_sqlite_url:
        raise SystemExit(
            "TARGET_DATABASE_URL still points at the default SQLite file. Set it to your Postgres database before running this migration."
        )

    source_engine = _create_engine(source_url)
    target_engine = _create_engine(target_url)

    print(f"Source: {source_url}")
    print(f"Target: {target_url}")
    _upsert_rows(source_engine, target_engine)
    print("Migration complete.")


if __name__ == "__main__":
    main()
