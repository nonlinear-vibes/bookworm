"""SQLite connection and schema for the lending system."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from bookworm.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    name    TEXT NOT NULL
);

-- book_id matches the book_id payload of the same book in Qdrant.
CREATE TABLE IF NOT EXISTS books (
    book_id      TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    authors      TEXT NOT NULL DEFAULT '',
    total_copies INTEGER NOT NULL DEFAULT 1 CHECK (total_copies >= 0)
);

CREATE TABLE IF NOT EXISTS loans (
    loan_id     TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users (user_id),
    book_id     TEXT NOT NULL REFERENCES books (book_id),
    loaned_at   TEXT NOT NULL,
    due_date    TEXT NOT NULL,
    returned_at TEXT,
    renewals    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS reservations (
    reservation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        TEXT NOT NULL REFERENCES users (user_id),
    book_id        TEXT NOT NULL REFERENCES books (book_id),
    created_at     TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'active'
                   CHECK (status IN ('active', 'cancelled', 'fulfilled'))
);
"""


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open a connection, commit on success, and always close it."""
    path = path or settings.sqlite_path
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
