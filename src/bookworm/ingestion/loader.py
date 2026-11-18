"""Load and normalize books from the raw CSV dataset."""

import re
import ast
import csv
import html
import hashlib

from pathlib import Path


def normalize_column_name(name: str) -> str:
    """Normalize CSV column names to make matching more flexible."""
    return "".join(c.lower() for c in name if c.isalnum())


def get_field(row: dict, *possible_names: str) -> str:
    """Get a CSV value using any of the supplied possible column names."""
    normalized_row = {
        normalize_column_name(key): value
        for key, value in row.items()
        if key is not None
    }

    for name in possible_names:
        value = normalized_row.get(normalize_column_name(name))
        if value and value.strip():
            return html.unescape(value.strip())

    return ""


def parse_list(value: str) -> list[str]:
    """Parse list columns stored as Python literals, e.g. "['Philip Nel']"."""
    if not value:
        return []

    if value.startswith("["):
        try:
            items = ast.literal_eval(value)
            return [str(item).strip() for item in items if str(item).strip()]
        except (ValueError, SyntaxError):
            pass

    return [value]


def parse_year(published_date: str) -> int | None:
    """Extract the publication year from dates like '2005', '2005-02', or '2005-01-01'."""
    match = re.match(r"(\d{4})", published_date)
    return int(match.group(1)) if match else None


BOOK_ID_HEX_CHARS = 12   # small chance of hash collison


def book_identity(title: str, authors: list[str], published_date: str, publisher: str) -> str:
    """The fields that identify a book; equal identities are duplicate rows."""
    return "|".join([title, ", ".join(authors), published_date, publisher])


def make_book_id(identity: str) -> str:
    """Build a short, stable ID like 'BK-7F3A9C21D4E6' from the book's identity.

    Short IDs are easier for small LLMs to copy exactly than full UUIDs.
    """
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"BK-{digest[:BOOK_ID_HEX_CHARS].upper()}"


def load_books(path: Path, limit: int | None = None) -> list[dict]:
    """Load and normalize the first `limit` books from the CSV."""
    books           = []
    seen_identities = set()
    identity_by_id  = {}

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            title = get_field(row, "title")

            # Skip rows without a title.
            if not title:
                continue

            authors        = parse_list(get_field(row, "authors", "author"))
            publisher      = get_field(row, "publisher")
            published_date = get_field(row, "publishedDate", "published_date", "publication_date")
            ratings_count  = get_field(row, "ratingsCount", "ratings_count")

            identity = book_identity(title, authors, published_date, publisher)

            # The dataset contains duplicate rows.
            if identity in seen_identities:
                continue
            seen_identities.add(identity)

            # Two different books hashing to the same short ID would make one
            # of them unreachable, so fail loudly instead.
            book_id = make_book_id(identity)
            if book_id in identity_by_id:
                raise ValueError(
                    f"Book ID collision on {book_id}: {identity_by_id[book_id]!r} and {identity!r}. "
                    "Increase BOOK_ID_HEX_CHARS."
                )
            identity_by_id[book_id] = identity

            books.append({
                "book_id":        book_id,
                "title":          title,
                "authors":        authors,
                "publisher":      publisher,
                "published_date": published_date,
                "year":           parse_year(published_date),
                "genres":         parse_list(get_field(row, "genres", "genre", "categories")),
                "description":    get_field(row, "description", "summary"),
                "ratings_count":  int(float(ratings_count)) if ratings_count else None,
            })

            if limit is not None and len(books) >= limit:
                break

    return books


def build_search_text(book: dict) -> str:
    """Combine relevant metadata into the text used for embedding and BM25."""
    fields = [
        f"Title: {book['title']}",
        f"Authors: {', '.join(book['authors'])}",
        f"Publisher: {book['publisher']}",
        f"Publication date: {book['published_date']}",
        f"Genres: {', '.join(book['genres'])}",
        f"Description: {book['description']}",
    ]

    return "\n".join(
        field for field in fields
        if field.split(": ", 1)[-1].strip()
    )
