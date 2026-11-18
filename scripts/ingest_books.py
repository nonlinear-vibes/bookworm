"""Embed books from the CSV and upload them to Qdrant.

Run with:  uv run python scripts/ingest_books.py --limit 2000 --recreate
"""

import argparse

from bookworm.ingestion.pipeline import ingest_books


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=2000, help="Number of books to ingest (default: 2000).")
    parser.add_argument("--batch-size", type=int, default=16, help="Books per embedding request (default: 16).")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the collection first.")
    args = parser.parse_args()

    ingest_books(limit=args.limit, batch_size=args.batch_size, recreate=args.recreate)


if __name__ == "__main__":
    main()
