"""Create the SQLite lending database and seed it with books.

Books come from the same CSV and limit as scripts/ingest_books.py, so the
book IDs match the Qdrant collection. Users are created by the CLI and the
web UI, one demo user per session.

Run with:  uv run python scripts/init_db.py --limit 2000 --reset
"""

import random
import argparse

from bookworm.config import settings
from bookworm.ingestion.loader import load_books
from bookworm.lending.db import connect, init_schema

# Seeded so every reset produces the same copy counts.
RANDOM_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=2000, help="Number of books to load (default: 2000).")
    parser.add_argument("--reset", action="store_true", help="Delete the existing database first.")
    args = parser.parse_args()

    if args.reset and settings.sqlite_path.exists():
        print(f"Deleting {settings.sqlite_path}...")
        settings.sqlite_path.unlink()

    books = load_books(settings.books_csv_path, args.limit)
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

        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("users", "books", "loans", "reservations")
        }

    print(f"Database ready at {settings.sqlite_path}")
    print(", ".join(f"{table}: {count}" for table, count in counts.items()))


if __name__ == "__main__":
    main()
