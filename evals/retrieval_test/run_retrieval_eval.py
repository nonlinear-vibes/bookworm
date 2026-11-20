"""Run the queries in retrieval_cases.json through four configurations and
save the top 5 results of each to search_results/<query id>.json.

Configurations:
  semantic  raw user query, dense vector search only
  lexical   raw user query, BM25 keyword search only
  hybrid    raw user query, both searches merged with Reciprocal Rank Fusion
  agentic   the full RAG subgraph: the Query Processor (LLM) writes the
            keyword query, the semantic query and the metadata filter, then
            both searches run and are fused, as in the live system

For metadata queries, the relevant set is computed from the catalogue and
saved with the results. Topic results are added to labels/topic_pool.csv
for hand labelling (existing labels are kept).

Usage, from the project root (see compose.yaml):
  docker compose run --rm eval python -m evals.retrieval_test.run_retrieval_eval
Then label the new rows in labels/topic_pool.csv and run generate_retrieval_report.py.
"""

import csv
import json
import os
import time
from pathlib import Path

EVALS_DIR   = Path(__file__).resolve().parent
RESULTS_DIR = EVALS_DIR / "search_results"
POOL_FILE   = EVALS_DIR / "labels" / "topic_pool.csv"
POOL_COLUMNS = ["query_id", "query", "book_id", "title", "author", "year", "genres", "description", "relevant"]
K = 5


def build_configs() -> dict:
    """Return name -> function(query) -> (top K books, retrieval plan or None)."""
    from langchain_core.messages import HumanMessage

    from bookworm.graph.nodes import result_fusion
    from bookworm.graph.rag_workflow import build_rag_graph
    from bookworm.llm.factory import create_llm
    from bookworm.retrieval import search

    rag_graph = build_rag_graph(create_llm())

    def hybrid(query):
        fused = result_fusion({
            "semantic_results": search.semantic_search(query),
            "lexical_results":  search.lexical_search(query),
        })
        return fused["fused_results"][:K], None

    def agentic(query):
        # The user's request doubles as the search task, which the Task
        # Planner would otherwise write; this isolates the RAG subgraph.
        result = rag_graph.invoke({"messages": [HumanMessage(content=query)], "search_task_description": query})
        return result.get("fused_results", [])[:K], result.get("retrieval_plan")

    return {
        "semantic": lambda query: (search.semantic_search(query)[:K], None),
        "lexical":  lambda query: (search.lexical_search(query)[:K], None),
        "hybrid":   hybrid,
        "agentic":  agentic,
    }


def metadata_relevant_ids(rule: dict) -> list[str]:
    """Every book among the ingested ones whose metadata satisfies the rule."""
    from bookworm.config import settings
    from bookworm.ingestion.loader import load_books

    relevant = []
    for book in load_books(settings.books_csv_path, int(os.getenv("BOOK_LIMIT", "2000"))):
        year = book["year"]
        if "author" in rule and rule["author"] not in book["authors"]:
            continue
        if "genre" in rule and rule["genre"] not in book["genres"]:
            continue
        if "year_from" in rule and (year is None or year < rule["year_from"]):
            continue
        if "year_to" in rule and (year is None or year > rule["year_to"]):
            continue
        relevant.append(book["book_id"])
    return relevant


def update_pool(topic_results: dict[str, dict]) -> int:
    """Add new (query, book) pairs to the labelling pool; return how many."""
    pool = {}
    if POOL_FILE.exists():
        with POOL_FILE.open(encoding="utf-8", newline="") as file:
            pool = {(row["query_id"], row["book_id"]): row for row in csv.DictReader(file)}

    added = 0
    for query_id, saved in topic_results.items():
        for run in saved["runs"].values():
            for book in run["results"]:
                if (query_id, book["book_id"]) not in pool:
                    pool[(query_id, book["book_id"])] = {
                        "query_id": query_id, "query": saved["query"], "relevant": "",
                        **{col: book[col] for col in ("book_id", "title", "author", "year", "genres", "description")},
                    }
                    added += 1

    POOL_FILE.parent.mkdir(exist_ok=True)
    with POOL_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=POOL_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(pool.values(), key=lambda r: (r["query_id"], r["title"].lower())))
    return added


def main() -> None:
    cases   = json.loads((EVALS_DIR / "retrieval_cases.json").read_text(encoding="utf-8"))
    configs = build_configs()
    topic_results = {}
    RESULTS_DIR.mkdir(exist_ok=True)

    for qtype in ("known_item", "metadata", "topic"):
        for case in cases[qtype]["cases"]:
            saved = {**case, "type": qtype, "runs": {}}
            if qtype == "metadata":
                saved["relevant_ids"] = metadata_relevant_ids(case["relevant_if"])

            for name, config in configs.items():
                start = time.perf_counter()
                books, plan = config(case["query"])
                saved["runs"][name] = {
                    "latency_s": round(time.perf_counter() - start, 3),
                    "plan":      plan,
                    "results":   books,
                }

            (RESULTS_DIR / f"{case['id']}.json").write_text(
                json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
            if qtype == "topic":
                topic_results[case["id"]] = saved
            print(f"{case['id']}: {case['query']}")

    added = update_pool(topic_results)
    print(f"\nAdded {added} new pairs to {POOL_FILE.relative_to(EVALS_DIR)}; label them (relevant = 1 or 0), "
          "then run retrieval_report.py.")


if __name__ == "__main__":
    main()
