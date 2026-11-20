"""Score the saved search results and write retrieval.md.

  known_item  one target book        -> Hit@1, Hit@5, MRR@5
  metadata    relevant set from the  -> Precision@5, Recall@5
              catalogue (saved by run_retrieval.py)
  topic       hand labels in         -> Precision@5, pooled Recall@5
              labels/topic_pool.csv

Uses only the standard library, so it runs on the host as well:
  python3 evals/retrieval_report.py
"""

import csv
import sys
import json
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
CONFIGS   = ["semantic", "lexical", "hybrid", "agentic"]
K = 5

METRICS = {
    "known_item": ["Hit@1", "Hit@5", "MRR@5"],
    "metadata":   ["P@5", "R@5"],
    "topic":      ["P@5", "R@5"],
}
# Written for the reference run; the plans themselves may vary between runs (temperature 0.2).
PLANS_NOTE = """\
**Finding: unnecessary metadata filters.** The Query Processor (Mistral 7B) often adds
metadata filters the user never asked for, despite the prompt rule against it: genres
inferred from a topic or mood, genre names the catalogue does not use, and publication-year
ranges derived from the subject (e.g. 1939–1945 for books *about* the Second World War).
Author and year filters stated by the user help (see the metadata queries), but the
invented ones shrink the candidate set to irrelevant books, which is why agentic search
falls behind plain semantic and hybrid search on several topic queries. A reranker in the
Retrieval Evaluator node (currently a no-op) could catch and retry these result sets, and a
larger model would likely follow the filter rule more reliably."""

TYPE_NAMES ={"known_item": "Known item", "metadata": "Metadata", "topic": "Topic / mood (pooled)"}


def load_labels() -> dict[str, dict[str, str]]:
    """query_id -> {book_id: "1" / "0" / ""}"""
    labels: dict[str, dict[str, str]] = {}
    with (EVALS_DIR / "labels" / "topic_pool.csv").open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            labels.setdefault(row["query_id"], {})[row["book_id"]] = row["relevant"].strip()
    return labels


def score(query: dict, ids: list[str], labels: dict) -> dict:
    if query["type"] == "known_item":
        rank = ids.index(query["target"]) + 1 if query["target"] in ids else None
        return {"Hit@1": float(rank == 1), "Hit@5": float(rank is not None), "MRR@5": 1 / rank if rank else 0.0}

    if query["type"] == "metadata":
        relevant = set(query["relevant_ids"])
    else:
        relevant = {b for b, label in labels.get(query["id"], {}).items() if label == "1"}
    hits = sum(1 for b in ids if b in relevant)
    return {"P@5": hits / K, "R@5": hits / len(relevant) if relevant else None}


def mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def table(headers: list[str], rows: list[list]) -> str:
    def fmt(v):
        return f"{v:.2f}" if isinstance(v, float) else ("" if v is None else str(v))
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join(lines + ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows])


def main() -> None:
    cases   = json.loads((EVALS_DIR / "retrieval_cases.json").read_text(encoding="utf-8"))
    labels  = load_labels()
    queries = []
    for qtype in METRICS:
        for case in cases[qtype]["cases"]:
            path = EVALS_DIR / "search_results" / f"{case['id']}.json"
            if not path.exists():
                sys.exit(f"Missing {path.relative_to(EVALS_DIR)}: run run_retrieval.py first.")
            queries.append(json.loads(path.read_text(encoding="utf-8")))

    unjudged = 0
    for q in queries:
        for run in q["runs"].values():
            ids = [b["book_id"] for b in run["results"]]
            run["scores"] = score(q, ids, labels)
            if q["type"] == "topic":
                unjudged += sum(1 for b in ids if labels.get(q["id"], {}).get(b, "") not in ("0", "1"))

    summary = []
    for qtype, metrics in METRICS.items():
        of_type = [q for q in queries if q["type"] == qtype]
        for metric in metrics:
            summary.append([f"{TYPE_NAMES[qtype]} (n={len(of_type)})", metric,
                            *[mean([q["runs"][c]["scores"][metric] for q in of_type]) for c in CONFIGS]])
    summary.append(["All queries", "Mean latency (s)",
                    *[mean([q["runs"][c]["latency_s"] for q in queries]) for c in CONFIGS]])

    per_query = []
    for q in queries:
        metric = "Hit@5" if q["type"] == "known_item" else "P@5"
        per_query.append([q["id"], q["variant"], metric, *[q["runs"][c]["scores"][metric] for c in CONFIGS]])

    plans = []
    for q in queries:
        plan = q["runs"]["agentic"]["plan"] or {}
        filters = {k: v for k, v in (plan.get("metadata_filter") or {}).items() if v not in (None, "")}
        plans.append([q["id"], plan.get("keyword_query"), plan.get("semantic_query"),
                      ", ".join(f"{k}={v}" for k, v in filters.items())])

    md = "\n\n".join([
        "# Retrieval evaluation", "_Generated by `generate_retrieval_report.py`. Unjudged topic results count as not relevant._",
        "## Summary", table(["Query type", "Metric", *CONFIGS], summary),
        "## Per query", table(["ID", "Variant", "Metric", *CONFIGS], per_query),
        "## Query Processor plans (agentic)", table(["ID", "Keyword query", "Semantic query", "Filter"], plans),
        PLANS_NOTE,
    ]) + "\n"
    (EVALS_DIR / "retrieval.md").write_text(md, encoding="utf-8")
    print(md)

    if unjudged:
        print(f"WARNING: {unjudged} topic results are unlabelled in labels/topic_pool.csv.", file=sys.stderr)


if __name__ == "__main__":
    main()
