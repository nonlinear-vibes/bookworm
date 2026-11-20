"""Prepare all data the app needs; safe to run on every start.

1. Download the books CSV from Hugging Face (pinned commit, checksum verified).
2. Ingest the first N books into Qdrant.
3. Seed the SQLite lending database with the same N books.

Each step is skipped when its result is already in place, so restarts are
fast and keep the lending data (loans, reservations) of earlier sessions.
A step that was interrupted half-way is detected and redone from scratch.

Run with:  uv run python scripts/init_data.py --limit 2000
In Docker: the `init` service in compose.yaml runs it before the UI starts.
"""

import argparse
import hashlib
import os
import time

import requests

from bookworm.config import settings
from bookworm.ingestion.loader import load_books
from bookworm.ingestion.pipeline import ingest_books
from bookworm.lending.seed import seed_database, table_counts
from bookworm.retrieval.qdrant_store import get_qdrant_client

# Pinned to a commit so every build gets exactly the same file.
DATASET_URL = (
    "https://huggingface.co/datasets/mmhshayer/books_data/resolve/"
    "2b520269135399a6c6d5db374dd4bed3bd495aa8/data.csv"
)
DATASET_SHA256 = "ba999658185d266942a04b9ee26d0d7045ae64c45e3b1ce0bb5386ba06944efd"

CHUNK_SIZE = 1024 * 1024


def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset() -> None:
    path = settings.books_csv_path

    if path.exists():
        print(f"[1/3] CSV already present at {path}, skipping download.")
        return

    print(f"[1/3] Downloading dataset to {path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    start   = time.time()

    with requests.get(DATASET_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        with partial.open("wb") as file:
            for chunk in response.iter_content(CHUNK_SIZE):
                file.write(chunk)

    actual = sha256_of(partial)
    if actual != DATASET_SHA256:
        partial.unlink()
        raise RuntimeError(f"Checksum mismatch for the dataset: expected {DATASET_SHA256}, got {actual}.")

    # Only a complete, verified file ever appears under the final name.
    partial.rename(path)
    print(f"      Downloaded {path.stat().st_size / 1e6:.0f} MB in {time.time() - start:.1f} s.")


def qdrant_point_count() -> int:
    client = get_qdrant_client()
    if not client.collection_exists(settings.qdrant_collection):
        return 0
    return client.count(settings.qdrant_collection, exact=True).count


def ingest_vectors(expected: int, batch_size: int) -> None:
    current = qdrant_point_count()

    if current == expected:
        print(f"[2/3] Qdrant collection '{settings.qdrant_collection}' already has {current} books, skipping.")
        return

    print(f"[2/3] Qdrant has {current} books, expected {expected}: (re)ingesting...")
    ingest_books(limit=expected, batch_size=batch_size, recreate=True)


def seed_lending_db(expected: int) -> None:
    current = table_counts()["books"]

    if current == expected:
        print(f"[3/3] SQLite already has {current} books, skipping (existing loans are kept).")
        return

    print(f"[3/3] SQLite has {current} books, expected {expected}: resetting and seeding...")
    counts = seed_database(expected, reset=True)
    print("      " + ", ".join(f"{table}: {count}" for table, count in counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--limit", type=int, default=int(os.getenv("BOOK_LIMIT", "2000")),
        help="Number of books to ingest (default: $BOOK_LIMIT or 2000).",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Books per embedding request (default: 16).")
    args = parser.parse_args()

    start = time.time()
    download_dataset()

    # Count with the same loader the ingestion uses (it skips duplicate and
    # untitled rows), so the expected number matches what gets stored.
    expected = len(load_books(settings.books_csv_path, args.limit))

    ingest_vectors(expected, args.batch_size)
    seed_lending_db(expected)

    print(f"Data ready in {time.time() - start:.1f} s.")


if __name__ == "__main__":
    main()
