"""Embeddings from the Ollama embedding model."""

import requests

from bookworm.config import settings


RETRIEVAL_INSTRUCTION = (
    "Given a user query for a library catalogue, "
    "retrieve books that are relevant to the query."
)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for book metadata and descriptions."""
    response = requests.post(
        f"{settings.ollama_base_url}/api/embed",
        json={"model": settings.embedding_model,
              "input": texts,
              "keep_alive": "5m"},
        timeout=300,
    )
    response.raise_for_status()

    embeddings = response.json()["embeddings"]

    if len(embeddings) != len(texts):
        raise RuntimeError(f"Expected {len(texts)} embeddings, received {len(embeddings)}.")

    for embedding in embeddings:
        if len(embedding) != settings.embedding_dimensions:
            raise RuntimeError(
                f"Expected {settings.embedding_dimensions} dimensions, received {len(embedding)}."
            )

    return embeddings


def embed_query(query: str) -> list[float]:
    """Generate an instruction-formatted embedding for a search query.

    Qwen3 embeddings expect an instruction on the query side only;
    documents are embedded as plain text.
    """
    formatted_query = (
        f"Instruct: {RETRIEVAL_INSTRUCTION}\n"
        f"Query: {query}"
    )

    return embed_documents([formatted_query])[0]
