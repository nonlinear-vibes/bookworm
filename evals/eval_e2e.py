"""Manual end-to-end scenarios for the main Bookworm graph.

Requires Ollama, Qdrant (with ingested books), and the SQLite database.
Lending cases change the database, so reset it before each full run:

    uv run python scripts/init_db.py --reset
    uv run python evals/eval_e2e.py [case_name ...]

With no arguments every case runs. Each case prints the nodes visited,
any approval request, and the final graph state.
"""

import sys
from pprint import pprint
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from bookworm.graph.workflow import build_graph


# Real books from the first 2000 ingested. Copies after init_db --reset:
# The Cruelest Months 3, Wings Above the Diamantina 1, Woman at the window 0.
SEARCH_CONTEXT = [
    HumanMessage(content="I'm looking for detective mystery novels."),
    AIMessage(content=(
        "Here are some mystery novels: 1. Trust No One by Debra Webb, "
        "2. The Cruelest Months by Louise Penny, 3. Wings Above the Diamantina "
        "by Arthur W. Upfield, 4. Woman at the window, a novel by A. J. Finn."
    )),
]

# name -> (messages, approval decision for any interrupt, expected route)
CASES = {
    "general": (
        [HumanMessage(content="Hi! What can you help me with?")],
        None,
        "planner -> response_generator -> evaluator",
    ),
    "search_only": (
        [HumanMessage(content="Can you find me some easy cookbooks for beginners?")],
        None,
        "planner -> rag -> response_generator -> evaluator",
    ),
    "availability_check": (
        [*SEARCH_CONTEXT, HumanMessage(content="Is The Cruelest Months available right now?")],
        None,
        "planner -> rag -> lending (no approval) -> response_generator -> evaluator",
    ),
    "reservation_approved": (
        [*SEARCH_CONTEXT, HumanMessage(content="Please reserve the second one for me.")],
        True,
        "planner -> rag -> lending (approval: yes) -> response_generator -> evaluator",
    ),
    "reservation_rejected": (
        [*SEARCH_CONTEXT, HumanMessage(content="Please reserve Wings Above the Diamantina for me.")],
        False,
        "planner -> rag -> lending (approval: no) -> response_generator -> evaluator",
    ),
    "reservation_unavailable": (
        [*SEARCH_CONTEXT, HumanMessage(content="Please reserve Woman at the window for me.")],
        None,
        "planner -> rag -> lending (no copies, no approval) -> response_generator -> evaluator",
    ),
    "cancellation": (
        [HumanMessage(content="Cancel my reservation for Whispers of the Wicked Saints.")],
        True,
        "planner -> rag -> lending (approval: yes) -> response_generator -> evaluator",
    ),
    "renewal": (
        [HumanMessage(content="Please renew my loan with ID loan-1.")],
        True,
        "planner -> lending (approval: yes) -> response_generator -> evaluator",
    ),
    "combined_search_and_reserve": (
        [HumanMessage(content="Find me a Betty Crocker cookbook with 4-ingredient dinners and reserve it.")],
        True,
        "planner -> rag -> lending (approval: yes) -> response_generator -> evaluator",
    ),
}


def run_case(graph, name: str) -> None:
    messages, approve, expected = CASES[name]
    config = {"configurable": {"thread_id": f"{name}-{uuid4()}"}}
    payload = {"user_id": "user-1", "messages": messages}

    print("\n" + "=" * 70)
    print(f"CASE: {name}")
    print(f"Expected route: {expected}")
    print("=" * 70)

    visited = []
    while payload is not None:
        resume = None
        for update in graph.stream(payload, config=config, stream_mode="updates"):
            for node, value in update.items():
                if node == "__interrupt__":
                    request = value[0].value
                    print(f"[interrupt] approval requested: {request} -> approved={approve}")
                    resume = Command(resume={"approved": bool(approve)})
                else:
                    visited.append(node)
                    print(f"[node] {node}")
        payload = resume

    print(f"\nActual route: {' -> '.join(visited)}")
    print("\n--- Final State ---")
    pprint(graph.get_state(config).values, sort_dicts=False)


def main() -> None:
    names = sys.argv[1:] or list(CASES)
    unknown = [n for n in names if n not in CASES]
    if unknown:
        sys.exit(f"Unknown case(s): {unknown}. Available: {list(CASES)}")

    graph = build_graph()
    for name in names:
        run_case(graph, name)


if __name__ == "__main__":
    main()
