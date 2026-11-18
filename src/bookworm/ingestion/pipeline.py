"""Ingestion pipeline: CSV -> embeddings -> Qdrant."""

import time
from pathlib import Path

from bookworm.config import settings
from bookworm.ingestion.loader import build_search_text, load_books
from bookworm.retrieval.embeddings import embed_documents
from bookworm.retrieval.qdrant_store import ensure_collection, upsert_books


def ingest_books(
    csv_path:   Path | None = None,
    limit:      int | None = None,
    batch_size: int = 16,
    recreate:   bool = False,
) -> int:
    """Embed books from the CSV and upload them to Qdrant. Returns the number ingested."""

    csv_path = csv_path or settings.books_csv_path
    start_time = time.time()

    print(f"Loading up to {limit or 'all'} books from {csv_path}...")
    books = load_books(csv_path, limit)

    if not books:
        raise RuntimeError("No books were loaded. Check the CSV path and column names.")

    print(f"Loaded {len(books)} books.")
    ensure_collection(recreate=recreate)

    for start in range(0, len(books), batch_size):
        batch = books[start:start + batch_size]
        texts = [build_search_text(book) for book in batch]

        upsert_books(batch, texts, embed_documents(texts))

        print(f"Ingested {start + len(batch)}/{len(books)} books.")

    elapsed = time.time() - start_time
    print(f"\nIngestion completed in {elapsed:.1f} s ({elapsed / len(books):.2f} s per book).")

    return len(books)
