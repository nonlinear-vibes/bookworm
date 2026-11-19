"""Retrieval evaluation: four configurations on three query types, at K = 5.

Configurations (all return the top K books):
  semantic  raw user query, dense vector search only
  lexical   raw user query, BM25 keyword search only
  hybrid    raw user query, both searches merged with Reciprocal Rank Fusion
  agentic   the full RAG subgraph: the Query Processor (LLM) writes the
            keyword query, the semantic query and the metadata filter, then
            both searches run and are fused, as in the live system

Query types (see retrieval_cases.json):
  known_item  one target book       -> Hit@1, Hit@5 (= Recall@5), MRR@5
  metadata    relevant set computed -> Precision@5, Recall@5
              from the catalogue
  topic       hand-labelled pool    -> Precision@5, pooled Recall@5

Usage (inside the eval container, see compose.yaml):
  docker compose run --rm eval python evals/eval_retrieval.py pool
      Run the topic queries and add every new (query, book) pair to
      labels/topic_pool.csv. Fill the `relevant` column with 1 or 0.
  docker compose run --rm eval python evals/eval_retrieval.py run
      Compute all metrics; writes results/retrieval.json and .md.
"""

import os
import csv
import sys
import time
import argparse

from common import (
    K, LABELS_DIR, load_json, markdown_table, match_owner, mean, setup_environment,
    write_results, write_text,
)

CONFIGS   = ["semantic", "lexical", "hybrid", "agentic"]
POOL_FILE = LABELS_DIR / "topic_pool.csv"
POOL_COLUMNS = ["query_id", "query", "book_id", "title", "author", "year", "genres", "description", "relevant"]


# -----------------------------------
#          Configurations
# -----------------------------------
def build_configs() -> dict:
    """Return name -> function(query) -> (top K books, retrieval plan or None)."""
    from langchain_core.messages import HumanMessage

    from bookworm.retrieval import search
    from bookworm.llm.factory import create_llm
    from bookworm.graph.nodes import result_fusion
    from bookworm.graph.rag_workflow import build_rag_graph

    rag_graph = build_rag_graph(create_llm())

    def semantic(query):
        return search.semantic_search(query)[:K], None

    def lexical(query):
        return search.lexical_search(query)[:K], None

    def hybrid(query):
        fused = result_fusion({
            "semantic_results": search.semantic_search(query),
            "lexical_results":  search.lexical_search(query),
        })
        return fused["fused_results"][:K], None

    def agentic(query):
        # The user's request doubles as the search task, which the Task
        # Planner would otherwise write; this isolates the RAG subgraph.
        result = rag_graph.invoke({
            "messages": [HumanMessage(content=query)],
            "search_task_description": query,
        })
        return result.get("fused_results", [])[:K], result.get("retrieval_plan")

    return {"semantic": semantic, "lexical": lexical, "hybrid": hybrid, "agentic": agentic}


def run_query(configs: dict, query: str) -> dict:
    runs = {}
    for name in CONFIGS:
        start = time.perf_counter()
        books, plan = configs[name](query)
        runs[name] = {
            "latency_s": round(time.perf_counter() - start, 3),
            "results":   [{"book_id": b["book_id"], "title": b["title"], "author": b["author"]} for b in books],
            "books":     books,
            "plan":      plan,
        }
    return runs


# -----------------------------------
#          Relevance sources
# -----------------------------------
def metadata_relevant_ids(rule: dict) -> set[str]:
    from bookworm.config import settings
    from bookworm.ingestion.loader import load_books

    limit = int(os.getenv("BOOK_LIMIT", "2000"))
    relevant = set()

    for book in load_books(settings.books_csv_path, limit):
        year = book["year"]
        if "author" in rule and rule["author"] not in book["authors"]:
            continue
        if "genre" in rule and rule["genre"] not in book["genres"]:
            continue
        if "year_from" in rule and (year is None or year < rule["year_from"]):
            continue
        if "year_to" in rule and (year is None or year > rule["year_to"]):
            continue
        relevant.add(book["book_id"])

    return relevant


def read_pool() -> dict[tuple[str, str], dict]:
    if not POOL_FILE.exists():
        return {}
    with POOL_FILE.open(encoding="utf-8", newline="") as file:
        return {(row["query_id"], row["book_id"]): row for row in csv.DictReader(file)}


def write_pool(pool: dict[tuple[str, str], dict]) -> None:
    POOL_FILE.parent.mkdir(exist_ok=True)
    rows = sorted(pool.values(), key=lambda r: (r["query_id"], r["title"].lower()))
    with POOL_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=POOL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    match_owner(POOL_FILE)


# -----------------------------------
#             Commands
# -----------------------------------
def command_pool() -> None:
    cases   = load_json("retrieval_cases.json")["topic"]["cases"]
    configs = build_configs()
    pool    = read_pool()
    added   = 0

    for case in cases:
        print(f"{case['id']}: {case['query']}")
        for name, run in run_query(configs, case["query"]).items():
            for book in run["books"]:
                key = (case["id"], book["book_id"])
                if key in pool:
                    continue
                pool[key] = {
                    "query_id": case["id"], "query": case["query"],
                    "book_id": book["book_id"], "title": book["title"], "author": book["author"],
                    "year": book["year"], "genres": book["genres"],
                    "description": book["description"], "relevant": "",
                }
                added += 1

    write_pool(pool)
    unlabelled = sum(1 for row in pool.values() if row["relevant"].strip() == "")
    print(f"\nAdded {added} new pairs; {len(pool)} in total, {unlabelled} still unlabelled.")
    print(f"Label them in {POOL_FILE.relative_to(POOL_FILE.parents[1])} (relevant = 1 or 0).")


def score_known_item(ids: list[str], target: str) -> dict:
    rank = ids.index(target) + 1 if target in ids else None
    return {
        "rank":   rank,
        "hit@1":  float(rank == 1),
        f"hit@{K}": float(rank is not None),
        "rr":     1 / rank if rank else 0.0,
    }


def score_set(ids: list[str], relevant: set[str]) -> dict:
    hits = sum(1 for book_id in ids if book_id in relevant)
    return {
        "hits":        hits,
        "relevant":    len(relevant),
        f"p@{K}":      hits / K,
        f"r@{K}":      hits / len(relevant) if relevant else None,
        f"max_p@{K}":  min(len(relevant), K) / K,
    }


def command_run() -> None:
    cases   = load_json("retrieval_cases.json")
    configs = build_configs()
    pool    = read_pool()
    report  = {"k": K, "configs": CONFIGS, "queries": []}

    for qtype in ("known_item", "metadata", "topic"):
        for case in cases[qtype]["cases"]:
            print(f"{case['id']}: {case['query']}")
            runs = run_query(configs, case["query"])

            if qtype == "metadata":
                relevant = metadata_relevant_ids(case["relevant_if"])
            elif qtype == "topic":
                relevant = {book_id for (qid, book_id), row in pool.items()
                            if qid == case["id"] and row["relevant"].strip() == "1"}

            for name, run in runs.items():
                ids = [r["book_id"] for r in run["results"]]
                if qtype == "known_item":
                    run["scores"] = score_known_item(ids, case["target"])
                else:
                    run["scores"] = score_set(ids, relevant)
                if qtype == "topic":
                    judged = {book_id for (qid, book_id), row in pool.items()
                              if qid == case["id"] and row["relevant"].strip() in ("0", "1")}
                    run["scores"]["unjudged"] = sum(1 for book_id in ids if book_id not in judged)
                del run["books"]

            report["queries"].append({**case, "type": qtype, "runs": runs})

    summary = summarize(report)
    report["summary"] = summary
    json_path = write_results("retrieval.json", report)
    md_path   = write_text(json_path.with_suffix(".md"), summary_markdown(report))

    print("\n" + summary_markdown(report))
    print(f"\nWrote {json_path.name} and {md_path.name} to {json_path.parent}")

    unjudged = sum(r["scores"].get("unjudged", 0) for q in report["queries"] for r in q["runs"].values())
    if unjudged:
        print(f"\nWARNING: {unjudged} topic results are unjudged (counted as not relevant). "
              "Run the `pool` command and label the new rows.", file=sys.stderr)


# -----------------------------------
#             Reporting
# -----------------------------------
METRICS = {
    "known_item": ["hit@1", f"hit@{K}", "rr"],
    "metadata":   [f"p@{K}", f"r@{K}"],
    "topic":      [f"p@{K}", f"r@{K}"],
}
METRIC_LABELS = {"hit@1": "Hit@1", f"hit@{K}": f"Hit@{K}", "rr": f"MRR@{K}",
                 f"p@{K}": f"P@{K}", f"r@{K}": f"R@{K}"}


def summarize(report: dict) -> dict:
    summary = {}
    for qtype, metrics in METRICS.items():
        queries = [q for q in report["queries"] if q["type"] == qtype]
        summary[qtype] = {
            name: {
                metric: mean([q["runs"][name]["scores"][metric] for q in queries
                              if q["runs"][name]["scores"].get(metric) is not None])
                for metric in metrics
            }
            for name in CONFIGS
        }
    summary["latency_s"] = {
        name: mean([q["runs"][name]["latency_s"] for q in report["queries"]]) for name in CONFIGS
    }
    return summary


def summary_markdown(report: dict) -> str:
    summary = report["summary"]
    names   = {"known_item": "Known item", "metadata": "Metadata", "topic": "Topic / mood (pooled)"}
    counts  = {t: sum(1 for q in report["queries"] if q["type"] == t) for t in METRICS}

    rows = [
        [f"{names[qtype]} (n={counts[qtype]})", METRIC_LABELS[metric],
         *[summary[qtype][name][metric] for name in CONFIGS]]
        for qtype, metrics in METRICS.items() for metric in metrics
    ]
    rows.append(["All queries", "Mean latency (s)", *[summary["latency_s"][n] for n in CONFIGS]])
    parts = ["## Summary", markdown_table(["Query type", "Metric", *CONFIGS], rows)]

    per_query = []
    for q in report["queries"]:
        main = METRICS[q["type"]][1] if q["type"] == "known_item" else METRICS[q["type"]][0]
        per_query.append([q["id"], q["variant"], METRIC_LABELS[main],
                          *[q["runs"][n]["scores"][main] for n in CONFIGS]])
    parts += ["## Per query", markdown_table(["ID", "Variant", "Metric", *CONFIGS], per_query)]

    plans = [[q["id"], q["runs"]["agentic"]["plan"]["keyword_query"],
              q["runs"]["agentic"]["plan"]["semantic_query"],
              {k: v for k, v in q["runs"]["agentic"]["plan"]["metadata_filter"].items() if v is not None} or ""]
             for q in report["queries"]]
    parts += ["## Query Processor plans (agentic)",
              markdown_table(["ID", "Keyword query", "Semantic query", "Filter"], plans)]

    return "\n\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["pool", "run"])
    args = parser.parse_args()

    setup_environment()
    command_pool() if args.command == "pool" else command_run()


if __name__ == "__main__":
    main()
