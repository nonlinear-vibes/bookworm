from functools import partial

from langgraph.graph import StateGraph, START, END

from bookworm.graph.state import LendingState
from bookworm.graph.nodes import lending_agent, lending_approval, lending_executor


def route_after_agent(state: LendingState) -> str:
    """Decide whether approval is required."""

    if state.get("pending_lending_action"):
        return "approval"

    return "done"


def build_lending_graph(llm, checkpointer=None):
    builder = StateGraph(LendingState)

    builder.add_node("lending_agent", partial(lending_agent, llm))
    builder.add_node("approval", lending_approval)
    builder.add_node("executor", lending_executor)

    builder.add_edge(START, "lending_agent")
    builder.add_conditional_edges(
        "lending_agent", route_after_agent,
        {"approval": "approval",
         "done":      END},
    )
    builder.add_edge("approval", "executor")
    builder.add_edge("executor", END)

    return builder.compile(checkpointer=checkpointer)