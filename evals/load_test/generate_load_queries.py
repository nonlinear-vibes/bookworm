"""Generate the query set for the load test (load_queries.json).

The queries are made from templates filled with real books and authors from
the ingested part of the catalogue (the first BOOK_LIMIT books of the CSV), so
every request follows a realistic path through the graph. A fixed seed makes
the set reproducible; the generated file is committed with the repo.

Traffic mix (out of 100 queries by default):
  topic          35%  open recommendation request          -> search
  author         20%  books by a given author              -> search
  known_item     20%  a given title                        -> search
  availability   15%  is a given title available?          -> search + lending
  reservation     5%  reserve a given title                -> search + lending + approval
  general         5%  small talk / gibberish               -> no tools

Usage:
  uv run python -m evals.load_test.generate_load_queries [--n 100] [--seed 42]
  docker compose run --rm eval python -m evals.load_test.generate_load_queries
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

from bookworm.config import settings
from bookworm.ingestion.loader import load_books

OUTPUT_PATH = Path(__file__).resolve().parent / "load_queries.json"

MIX = {
    "topic":        0.35,
    "author":       0.20,
    "known_item":   0.20,
    "availability": 0.15,
    "reservation":  0.05,
    "general":      0.05,
}

# Topics phrased as a reader would ask for them. Each one matches one of the
# larger genres of the ingested catalogue (10+ books), so the search has
# something to find.
TOPICS = [
    "World War II history", "ancient history", "Christian spirituality", "Catholic faith",
    "starting a small business", "personal finance", "leadership and management",
    "the life of a famous musician", "memoirs by politicians", "picture books for young children",
    "adventure stories for kids", "learning to play the guitar", "hiking and the outdoors",
    "meditation and mindfulness", "teaching and classroom learning", "travel in Europe",
    "philosophy for beginners", "parenting", "learning to program", "modern poetry",
    "healthy eating and fitness", "learning a foreign language", "architecture",
    "Italian cooking", "painting and art history", "political theory", "popular science",
    "psychology and the mind", "crafts and knitting", "self-improvement", "gardening",
    "detective fiction", "romance novels", "fantasy and magic", "law and justice",
]

TOPIC_TEMPLATES = [
    "Can you recommend a book about {topic}?",
    "I'm looking for something on {topic}.",
    "Do you have any good books about {topic}?",
    "What would you suggest for someone interested in {topic}?",
    "I'd like to read about {topic}. Any ideas?",
    "Recommend me a few books on {topic}, please.",
]

AUTHOR_TEMPLATES = [
    "Do you have any books by {author}?",
    "What books by {author} are in the catalogue?",
    "I'm looking for something written by {author}.",
    "Show me what you have from {author}.",
]

KNOWN_ITEM_TEMPLATES = [
    "Do you have {title}?",
    "Is {title} in your catalogue?",
    "I'm looking for the book {title} by {author}.",
    "Can you find {title} for me?",
]

AVAILABILITY_TEMPLATES = [
    "Is {title} available right now?",
    "Is {title} by {author} available?",
    "Can I borrow {title} at the moment?",
    "Are there any free copies of {title}?",
]

RESERVATION_TEMPLATES = [
    "Please reserve {title} for me.",
    "I'd like to reserve {title} by {author}.",
    "Can you reserve {title} for me, please?",
]

GENERAL_MESSAGES = [
    "Hi! What can you help me with?",
    "Thanks, that's all for today!",
    "What are your opening hours?",
    "asdf qwerty zxcv",
    "Good morning!",
    "How does borrowing work here?",
    "Who are you?",
]

# Corporate or institutional "authors" that nobody would ask for by name.
NON_PERSON_AUTHOR = re.compile(
    r"\b(inc|ltd|llc|firm|office|publications?|press|books|editors?|staff|association|"
    r"society|council|department|library|institute|university|company|corporation)\b",
    re.IGNORECASE,
)


def clean_title(title: str) -> str:
    """Shorten a catalogue title to how a reader would say it.

    Drops a trailing series or edition note in parentheses and the subtitle,
    and fixes ALL-CAPS titles.
    """
    title = title.split(":")[0].strip()
    title = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
    if title.isupper():
        title = title.title()
    return title


def is_person(author: str) -> bool:
    return (len(author.split()) >= 2
            and not any(ch.isdigit() for ch in author)
            and not NON_PERSON_AUTHOR.search(author))


def usable_books(books: list[dict]) -> list[dict]:
    """Books whose title and first author work well inside a sentence."""
    usable = []
    for book in books:
        short = clean_title(book["title"])
        if not (3 <= len(short) <= 60) or not book["authors"] or not is_person(book["authors"][0]):
            continue
        usable.append({**book, "short_title": short})
    return usable


def counts_for(n: int) -> dict[str, int]:
    """Split n queries by MIX; rounding leftovers go to the largest category."""
    counts = {category: round(share * n) for category, share in MIX.items()}
    counts["topic"] += n - sum(counts.values())
    return counts


def book_query(book: dict, templates: list[str], rng: random.Random) -> dict:
    author = book["authors"][0]
    return {
        "message":  rng.choice(templates).format(title=book["short_title"], author=author),
        "book_id":  book["book_id"],
        "title":    book["title"],
        "author":   author,
    }


def generate(n: int, seed: int, limit: int) -> list[dict]:
    rng   = random.Random(seed)
    books = usable_books(load_books(settings.books_csv_path, limit))
    count = counts_for(n)

    # Every book and author is used at most once across the whole set.
    book_pool = rng.sample(books, count["known_item"] + count["availability"] + count["reservation"])
    used_authors = {book["authors"][0] for book in book_pool}
    authors = sorted({book["authors"][0] for book in books} - used_authors)

    queries: list[dict] = []

    # Distinct topics while they last, each with a random phrasing.
    topics = rng.sample(TOPICS, len(TOPICS)) * (count["topic"] // len(TOPICS) + 1)
    for topic in topics[:count["topic"]]:
        queries.append({"category": "topic", "message": rng.choice(TOPIC_TEMPLATES).format(topic=topic),
                        "topic": topic})

    for author in rng.sample(authors, count["author"]):
        queries.append({"category": "author", "message": rng.choice(AUTHOR_TEMPLATES).format(author=author),
                        "author": author})

    for category, templates in (("known_item",   KNOWN_ITEM_TEMPLATES),
                                ("availability", AVAILABILITY_TEMPLATES),
                                ("reservation",  RESERVATION_TEMPLATES)):
        for _ in range(count[category]):
            queries.append({"category": category, **book_query(book_pool.pop(), templates, rng)})

    for message in rng.sample(GENERAL_MESSAGES, count["general"]):
        queries.append({"category": "general", "message": message})

    # Interleave the categories, as in real traffic.
    rng.shuffle(queries)
    return [{"id": f"L{i:03d}", **query} for i, query in enumerate(queries, start=1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=100, help="Number of queries (default: 100).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--limit", type=int, default=int(os.getenv("BOOK_LIMIT", "2000")),
                        help="Books in the ingested catalogue (default: $BOOK_LIMIT or 2000).")
    args = parser.parse_args()

    queries = generate(args.n, args.seed, args.limit)
    counts  = counts_for(args.n)
    OUTPUT_PATH.write_text(json.dumps({
        "_about": [
            f"Load test queries, generated by generate_load_queries.py (n={args.n}, seed={args.seed}, "
            f"limit={args.limit}). Do not edit by hand; regenerate instead.",
            "category: topic, author, known_item, availability, reservation or general.",
            "book_id / title / author: the catalogue book the query was made from (full title).",
            "The load runner approves every approval request, so lending actions write to its scratch database.",
        ],
        "mix": counts,
        "queries": queries,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {len(queries)} queries to {OUTPUT_PATH}")
    for category, k in counts.items():
        print(f"  {category:<13} {k:>3}")


if __name__ == "__main__":
    main()
