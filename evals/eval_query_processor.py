"""Manual scenarios for the Query Processor and the RAG subgraph.

Requires Ollama and Qdrant (with ingested books).

Run with:  uv run python evals/eval_query_processor.py [case_name ...]
With no arguments every case runs. Each case prints the retrieval plan
produced by the Query Processor, then the results of the RAG subgraph.
"""

import sys
from pprint import pprint

from langchain_core.messages import AIMessage, HumanMessage

from bookworm.graph.nodes import query_processor
from bookworm.graph.rag_workflow import build_rag_graph
from bookworm.llm.factory import create_llm


AI_CONTEXT = [
    HumanMessage(content="I'm looking for books about artificial intelligence."),
    AIMessage(content="Here are five books: 1. Book A, 2. Book B, 3. Book C, 4. Book D, 5. Book E."),
]

# name -> (messages, search task description, expected plan)
CASES = {
    "topic": (
        [HumanMessage(content="Can you find me some books about space exploration?")],
        "Search the catalogue for books about space exploration.",
        "keyword + semantic, no filter",
    ),
    "exact_title": (
        [HumanMessage(content="Do you have The Hobbit by J.R.R. Tolkien?")],
        "Find The Hobbit by J.R.R. Tolkien in the catalogue.",
        "keyword only (title), author filter",
    ),
    "author_only": (
        [HumanMessage(content="What books by Agatha Christie do you have?")],
        "Search the catalogue for books by Agatha Christie.",
        "keyword 'Agatha Christie', author filter",
    ),
    "mood": (
        [HumanMessage(content="I want something cozy and uplifting to read on a rainy weekend.")],
        "Find cozy, uplifting books.",
        "semantic only, no filter",
    ),
    "filters": (
        [HumanMessage(content="Recommend science fiction novels published after 2010.")],
        "Find science fiction novels published after 2010.",
        "genre 'Science fiction' + year_from 2011 filter",
    ),
    "follow_up": (
        [*AI_CONTEXT, HumanMessage(content="Do you have any newer ones, from 2020 or later?")],
        "Find books about artificial intelligence published in 2020 or later.",
        "AI topic resolved from history, year_from 2020",
    ),
}

# Simulates a rejected first attempt, as the Retrieval Evaluator would send it back.
RETRY_CASE = {
    "messages": [HumanMessage(content="I'd like a book on baking sourdough bread at home.")],
    "search_task_description": "Find books about baking sourdough bread at home.",
    "retry_retrieval": True,
    "retrieval_plan": {
        "keyword_query": "sourdough bread baking home",
        "semantic_query": "A practical guide to baking sourdough bread at home.",
        "metadata_filter": {"author": None, "year_from": 2022, "year_to": None},
    },
    "fused_results": [{"book_id": "BK-000000000999", "title": "Quick Weeknight Dinners", "author": "Author X"}],
    "retrieval_feedback": "The results are general cookbooks, not about bread or sourdough. The year filter may be too strict.",
    "retrieval_attempts": 1,
}


def run_case(llm, rag_graph, name: str) -> None:
    messages, task, expected = CASES[name]
    state = {"messages": messages, "search_task_description": task}

    print("\n" + "=" * 70)
    print(f"CASE: {name}")
    print(f"User: {messages[-1].content}")
    print(f"Expected: {expected}")
    print("=" * 70)

    print("\n--- Retrieval Plan ---")
    pprint(query_processor(llm, state)["retrieval_plan"], sort_dicts=False)

    result = rag_graph.invoke(state)
    for key in ("semantic_results", "lexical_results", "fused_results"):
        print(f"\n--- {key} ---")
        for book in result.get(key, []):
            print(f"  {book['title'][:60]} | {book['author'][:30]} | {book['year']} | {book['genres']}")


def run_retry_case(llm) -> None:
    print("\n" + "=" * 70)
    print("CASE: retry")
    print(f"User: {RETRY_CASE['messages'][-1].content}")
    print("Expected: a different plan that addresses the feedback (bread/sourdough focus, no year filter)")
    print("=" * 70)

    print("\n--- Previous Plan ---")
    pprint(RETRY_CASE["retrieval_plan"], sort_dicts=False)
    print(f"Feedback: {RETRY_CASE['retrieval_feedback']}")

    print("\n--- Revised Plan ---")
    pprint(query_processor(llm, RETRY_CASE)["retrieval_plan"], sort_dicts=False)


def main() -> None:
    names = sys.argv[1:] or [*CASES, "retry"]
    unknown = [n for n in names if n not in CASES and n != "retry"]
    if unknown:
        sys.exit(f"Unknown case(s): {unknown}. Available: {[*CASES, 'retry']}")

    llm = create_llm()
    rag_graph = build_rag_graph(llm)

    for name in names:
        if name == "retry":
            run_retry_case(llm)
        else:
            run_case(llm, rag_graph, name)


if __name__ == "__main__":
    main()
