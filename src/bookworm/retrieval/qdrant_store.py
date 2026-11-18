"""Qdrant database access layer.

Knows how to connect to Qdrant, configure the collection and its indexes,
upload points, and run low-level searches. retrieval/search.py decides how
to use them.
"""

import uuid
from functools import lru_cache
from qdrant_client import QdrantClient, models
from bookworm.config import settings


DENSE_VECTOR  = "dense"
SPARSE_VECTOR = "bm25"
BM25_MODEL    = "qdrant/bm25"
# Qdrant computes BM25 sparse vectors on the server from raw text.


@lru_cache
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(recreate: bool = False) -> None:
    """Create the collection with dense + sparse vectors and payload indexes."""
    client = get_qdrant_client()
    name   = settings.qdrant_collection

    if recreate and client.collection_exists(name):
        print(f"Deleting collection '{name}'...")
        client.delete_collection(name)

    if client.collection_exists(name):
        return

    print(f"Creating collection '{name}'...")
    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(
                size=settings.embedding_dimensions,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            # IDF is applied by Qdrant at query time, completing the BM25 score.
            SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
    )

    # Indexes for the metadata filter. Text indexes give case-insensitive
    # word matching, so "tolkien" matches "J.R.R. Tolkien".
    text_index = models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType.WORD,
        lowercase=True,
    )
    client.create_payload_index(name, "authors", field_schema=text_index)
    client.create_payload_index(name, "genres",  field_schema=text_index)
    client.create_payload_index(name, "year",    field_schema=models.PayloadSchemaType.INTEGER)


def upsert_books(books: list[dict], texts: list[str], dense_vectors: list[list[float]]) -> None:
    """Upload books with their dense embedding and BM25 text."""
    points = [
        models.PointStruct(
            # Qdrant point IDs must be UUIDs or integers, so derive a stable
            # UUID from the short book ID. Lookups use the book_id payload.
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, book["book_id"])),
            vector={
                DENSE_VECTOR:  dense_vector,
                SPARSE_VECTOR: models.Document(text=text, model=BM25_MODEL),
            },
            payload=book,
        )
        for book, text, dense_vector in zip(books, texts, dense_vectors)
    ]

    get_qdrant_client().upsert(settings.qdrant_collection, points=points, wait=True)


def search_vectors(
    query_vector: list[float],
    query_filter: models.Filter | None = None,
    limit: int = 10,
) -> list[models.ScoredPoint]:
    """Dense (semantic) nearest-neighbour search."""
    return get_qdrant_client().query_points(
        settings.qdrant_collection,
        query=query_vector,
        using=DENSE_VECTOR,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    ).points


def search_text(
    query: str,
    query_filter: models.Filter | None = None,
    limit: int = 10,
) -> list[models.ScoredPoint]:
    """Sparse (BM25) keyword search."""
    return get_qdrant_client().query_points(
        settings.qdrant_collection,
        query=models.Document(text=query, model=BM25_MODEL),
        using=SPARSE_VECTOR,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    ).points
