"""Create the SQLite lending database and seed it with books.

Books come from the same CSV and limit as scripts/ingest_books.py, so the
book IDs match the Qdrant collection. Users are created by the CLI and the
web UI, one demo user per session.

Run with:  uv run python scripts/init_db.py --limit 2000 --reset
"""

import argparse

from bookworm.config import settings
from bookworm.lending.seed import seed_database


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=2000, help="Number of books to load (default: 2000).")
    parser.add_argument("--reset", action="store_true", help="Delete the existing database first.")
    args = parser.parse_args()

    counts = seed_database(args.limit, reset=args.reset)

    print(f"Database ready at {settings.sqlite_path}")
    print(", ".join(f"{table}: {count}" for table, count in counts.items()))


if __name__ == "__main__":
    main()
