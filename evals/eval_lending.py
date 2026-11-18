"""Manual scenario for the lending subgraph on its own.

Requires Ollama, Qdrant (with ingested books), and the SQLite database:

    uv run python scripts/init_db.py --reset
    uv run python evals/eval_lending.py
"""

from langchain_core.messages import HumanMessage, AIMessage

from bookworm.retrieval.search import lexical_search

from bookworm.graph.lending_workflow import build_lending_graph
from bookworm.llm.factory import create_llm
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver




def main():
    llm = create_llm()
    graph = build_lending_graph(llm, checkpointer=InMemorySaver())
    # The demo user seeded by scripts/init_db.py.
    user_id = "user-1"
    

    # Identify this conversation's graph execution thread.
    config = {
        "configurable": {
            "thread_id": "bookworm-test-session"
        }
    }

    state = {
            "user_id": user_id,
            "messages": [
                HumanMessage(
                    content="I'm looking for detective mystery novels."
                ),
                AIMessage(
                    content="Here are some: Trust No One, The Cruelest Months, "
                            "Wings Above the Diamantina."
                ),
                HumanMessage(
                    content="Can you reserve The Cruelest Months for me?"
                ),
            ],
            "lending_task_description": "Reserve The Cruelest Months for the user.",
            # Real search results, as the RAG subgraph would provide them.
            "search_results": lexical_search("The Cruelest Months"),
        }

    # Start execution.
    result = graph.invoke(state, config=config)

    # Handle a pending approval.
    while result.get("__interrupt__"):

        for interruption in result["__interrupt__"]:
            request = interruption.value

            print("\n" + "=" * 45)
            print("       BOOKWORM - APPROVAL REQUIRED")
            print("=" * 45)

            print(f"\nOperation: {request['tool_name']}")
            print("Arguments:")

            for key, value in request["arguments"].items():
                print(f"  {key}: {value}")

            print("\nDo you approve this operation?")
            answer = input("[y/n]: ").strip().lower()

            approved = answer == "y"

            # Resume the graph with the user's decision.
            result = graph.invoke(
                Command(resume={"approved": approved}),
                config=config,
            )

    print("\n--- Final Result ---")
    print(result.get("lending_result"))


if __name__ == "__main__":
    main()