from functools import partial

from langgraph.graph import END, START, StateGraph

from bookworm.graph.nodes import (
    lexical_search_node,
    query_processor,
    result_fusion,
    retrieval_evaluator,
    semantic_search_node,
)
from bookworm.graph.state import RagState


def route_after_query_processor(state: RagState) -> list[str]:
    """Run only the searches that have a query; both run in parallel."""

    plan     = state["retrieval_plan"]
    searches = []

    if plan["semantic_query"].strip():
        searches.append("semantic_search")
    if plan["keyword_query"].strip():
        searches.append("lexical_search")

    return searches


def route_after_retrieval_evaluator(state: RagState) -> str:
    """Retry the retrieval or finish with the current results."""

    if state.get("retry_retrieval"):
        return "retry"

    return "done"


def build_rag_graph(llm, checkpointer=None):
    builder = StateGraph(RagState)

    builder.add_node("query_processor", partial(query_processor, llm))
    builder.add_node("semantic_search", semantic_search_node)
    builder.add_node("lexical_search", lexical_search_node)
    builder.add_node("result_fusion", result_fusion)
    builder.add_node("retrieval_evaluator", retrieval_evaluator)

    builder.add_edge(START, "query_processor")

    builder.add_conditional_edges(
        "query_processor", route_after_query_processor,
        ["semantic_search", "lexical_search"],
    )

    builder.add_edge("semantic_search", "result_fusion")
    builder.add_edge("lexical_search",  "result_fusion")
    builder.add_edge("result_fusion",   "retrieval_evaluator")

    builder.add_conditional_edges(
        "retrieval_evaluator", route_after_retrieval_evaluator,
        {"retry": "query_processor",
         "done": END},
    )

    return builder.compile(checkpointer=checkpointer)
