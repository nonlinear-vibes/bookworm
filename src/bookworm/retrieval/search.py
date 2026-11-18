"""Catalogue search used by the RAG subgraph."""

from qdrant_client import models

from bookworm.retrieval.embeddings import embed_query
from bookworm.retrieval.qdrant_store import search_text, search_vectors


# Results returned by each search before Result Fusion.
SEARCH_LIMIT = 10

# Descriptions are shortened to keep the LLM prompts small.
DESCRIPTION_PREVIEW_CHARS = 300


def build_filter(metadata_filter: dict | None) -> models.Filter | None:
    """Translate the Query Processor's metadata filter into a Qdrant filter."""
    if not metadata_filter:
        return None

    conditions = []

    if metadata_filter.get("author"):
        conditions.append(models.FieldCondition(key="authors", match=models.MatchText(text=metadata_filter["author"])))

    if metadata_filter.get("genre"):
        conditions.append(models.FieldCondition(key="genres", match=models.MatchText(text=metadata_filter["genre"])))

    year_from = metadata_filter.get("year_from")
    year_to = metadata_filter.get("year_to")
    if year_from is not None or year_to is not None:
        conditions.append(models.FieldCondition(key="year", range=models.Range(gte=year_from, lte=year_to)))

    return models.Filter(must=conditions) if conditions else None


def _to_result(point: models.ScoredPoint) -> dict:
    """Convert a Qdrant point into the BookResult shape used by the graph."""
    payload = point.payload or {}
    description = payload.get("description", "")

    if len(description) > DESCRIPTION_PREVIEW_CHARS:
        description = description[:DESCRIPTION_PREVIEW_CHARS].rsplit(" ", 1)[0] + "..."

    return {
        "book_id": payload["book_id"],
        "title": payload["title"],
        "author": ", ".join(payload.get("authors", [])),
        "year": payload.get("year"),
        "genres": ", ".join(payload.get("genres", [])),
        "description": description,
    }


def semantic_search(query: str, metadata_filter: dict | None = None, limit: int = SEARCH_LIMIT) -> list[dict]:
    """Search by meaning with the dense embedding."""
    # Genre labels in the catalogue are unreliable. Semantic search 
    # stays unfiltered to keep mislabelled books (recall), while RRF
    # fusion ranks books found by both searches highest (precision).
    if metadata_filter:
        metadata_filter = {**metadata_filter, "genre": None}
    points = search_vectors(embed_query(query), build_filter(metadata_filter), limit)
    return [_to_result(point) for point in points]


def lexical_search(query: str, metadata_filter: dict | None = None, limit: int = SEARCH_LIMIT) -> list[dict]:
    """Search by keywords with BM25."""
    points = search_text(query, build_filter(metadata_filter), limit)
    return [_to_result(point) for point in points]
