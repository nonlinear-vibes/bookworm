"""Minimal single-turn run of the main graph.

Run with:  uv run python evals/eval_graph.py
"""

from pprint import pprint

from langchain_core.messages import HumanMessage, AIMessage

from bookworm.graph.workflow import build_graph


def main():
    graph = build_graph()

    # The demo user seeded by scripts/init_db.py.
    user_id = "user-1"

    result = graph.invoke({
        "user_id": user_id,
        "messages": [
            HumanMessage(
                content="I'm looking for books about artificial intelligence."
            ),
            AIMessage(
                content=(
                    "Here are five books available:\n"
                    "1. Book A\n"
                    "2. Book B\n"
                    "3. Book C\n"
                    "4. Book D\n"
                    "5. Book E"
                )
            ),
            HumanMessage(
                content="Can you reserve the second one for me?"
            ),
        ],
    })

    print("\n--- Final Graph State ---")
    pprint(result, sort_dicts=False)


if __name__ == "__main__":
    main()