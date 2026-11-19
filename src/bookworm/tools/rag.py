from langchain_core.tools import tool

from bookworm.retrieval import search


@tool
def semantic_search(query: str, metadata_filter: dict | None = None) -> list[dict]:
    """Search the catalogue by meaning using dense vector similarity.

    Args:
        query: A natural-language description of the wanted books.
        metadata_filter: Optional constraints such as author or year range.
    """
    return search.semantic_search(query, metadata_filter)


@tool
def lexical_search(query: str, metadata_filter: dict | None = None) -> list[dict]:
    """Search the catalogue by keywords using BM25.

    Args:
        query: Keywords such as titles, author names, or distinctive terms.
        metadata_filter: Optional constraints such as author, genre, or year range.
    """
    return search.lexical_search(query, metadata_filter)
