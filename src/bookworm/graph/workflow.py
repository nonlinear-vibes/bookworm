from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from bookworm.graph.lending_workflow import build_lending_graph
from bookworm.graph.nodes import (
    evaluator,
    intent_detector,
    lending_node,
    rag_node,
    response_generator,
    task_planner,
)
from bookworm.graph.rag_workflow import build_rag_graph
from bookworm.graph.state import MainState
from bookworm.llm.factory import create_llm


def _planned_agents(state: MainState) -> set[str]:
    return {task["agent"] for task in state["task_plan"]}


def route_after_planner(state: MainState) -> str:
    agents = _planned_agents(state)

    if "search" in agents:
        return "rag"
    if "lending" in agents:
        return "lending"
    return "response"


def route_after_rag(state: MainState) -> str:
    if "lending" in _planned_agents(state):
        return "lending"
    return "response"


def route_after_evaluator(state: MainState) -> str:
    if state.get("revision_requested"):
        return "revise"
    return "end"


def build_graph(llm=None):
    if llm is None:
        llm = create_llm()

    builder       = StateGraph(MainState)
    lending_graph = build_lending_graph(llm)
    rag_graph     = build_rag_graph(llm)

    builder.add_node("intent_detector", partial(intent_detector, llm))
    builder.add_node("task_planner", partial(task_planner, llm))
    builder.add_node("rag", partial(rag_node, rag_graph))
    builder.add_node("lending", partial(lending_node, llm, lending_graph))
    builder.add_node("response_generator", partial(response_generator, llm))
    builder.add_node("evaluator", partial(evaluator, llm))

    builder.add_edge(START, "intent_detector")
    builder.add_edge("intent_detector", "task_planner")
    builder.add_conditional_edges(
        "task_planner", route_after_planner,
        {"rag":      "rag",
         "lending":  "lending",
         "response": "response_generator"},
    )
    builder.add_conditional_edges(
        "rag", route_after_rag,
        {"lending":  "lending",
         "response": "response_generator"},
    )
    builder.add_edge("lending", "response_generator")
    builder.add_edge("response_generator", "evaluator")
    builder.add_conditional_edges(
        "evaluator", route_after_evaluator,
        {"revise":   "response_generator",
         "end":       END}
    )

    return builder.compile(checkpointer=InMemorySaver())