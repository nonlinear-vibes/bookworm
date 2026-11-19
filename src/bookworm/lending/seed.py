"""Seed the SQLite lending database with books from the CSV.

Books come from the same CSV and limit as the Qdrant ingestion, so the book
IDs match the Qdrant collection. Users are created by the CLI and the web UI,
one demo user per session.
"""

import random

from bookworm.config import settings
from bookworm.ingestion.loader import load_books
from bookworm.lending.db import connect, init_schema

# Seeded so every reset produces the same copy counts.
RANDOM_SEED = 42

TABLES = ("users", "books", "loans", "reservations")


def table_counts() -> dict[str, int]:
    """Row counts per table; all zero if the database does not exist yet."""
    if not settings.sqlite_path.exists():
        return dict.fromkeys(TABLES, 0)

    with connect() as conn:
        init_schema(conn)
        return {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in TABLES
        }


def seed_database(limit: int, reset: bool = False) -> dict[str, int]:
    """Create the schema and insert the first `limit` books. Returns the table counts."""
    if reset and settings.sqlite_path.exists():
        print(f"Deleting {settings.sqlite_path}...")
        settings.sqlite_path.unlink()

    books = load_books(settings.books_csv_path, limit)
    rng   = random.Random(RANDOM_SEED)

    with connect() as conn:
        init_schema(conn)

        conn.executemany(
            "INSERT OR IGNORE INTO books (book_id, title, authors, total_copies) VALUES (?, ?, ?, ?)",
            [
                (book["book_id"], book["title"], ", ".join(book["authors"]), rng.randint(0, 3))
                for book in books
            ],
        )

    return table_counts()
