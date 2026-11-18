from pprint import pformat

from langgraph.types import interrupt
from bookworm.tools.lending import LENDING_TOOLS, WRITE_TOOLS
from bookworm.tools.rag import lexical_search, semantic_search
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage, RemoveMessage

from bookworm.graph import prompts
from bookworm.graph.state import LendingState, MainState, RagState
from bookworm.graph.schemas import Evaluation, Intent, LendingAction, RetrievalPlan, TaskList


# Maximum number of times the Evaluator may send a response back for revision.
MAX_RESPONSE_RETRIES = 2

# Maximum number of times the Retrieval Evaluator may send the search back.
MAX_RETRIEVAL_RETRIES = 2

# Rank constant for Reciprocal Rank Fusion.
RRF_K = 60

# Number of fused results passed on to the rest of the graph.
MAX_SEARCH_RESULTS = 5

# -----------------------------------
#              Helpers
# -----------------------------------
def _last_user_text(state: MainState) -> str:
    """Get the latest user message."""

    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return str(message.content)

    return ""


def _history(state: MainState) -> str:
    """Format previous conversation messages (except the last one)."""

    messages = state["messages"][:-1]
    history  = []

    for message in messages:
        role    = message.type.upper()
        content = message.content

        if not isinstance(content, str):
            content = str(content)

        history.append(f"{role}: {content}")
    
    return "\n".join(history)


def get_lending_task(task_plan: list[dict]) -> str:
    for task in task_plan:
        if task.get("agent") == "lending":
            return task["description"]
    
    raise ValueError("No lending task found in task_plan.")


def get_search_task(task_plan: list[dict]) -> str:
    for task in task_plan:
        if task.get("agent") == "search":
            return task["description"]

    raise ValueError("No search task found in task_plan.")

# -----------------------------------
# --- Debug helpers (remove when done debugging) ---
# -----------------------------------
def _debug_print_llm_input(node_name: str, llm_input: list) -> None:
    """Print the messages sent to the LLM, skipping the system prompt."""

    print(f"\n{'=' * 80}\n[{node_name}] LLM INPUT\n{'=' * 80}")
    for message in llm_input:
        role, content = message if isinstance(message, tuple) else (message.type, message.content)
        if role == "system":
            continue
        print(f"--- {role.upper()} ---\n{content}")


def _debug_print_llm_output(node_name: str, result) -> None:
    """Print the LLM's reply (structured output or plain message)."""

    print(f"\n{'-' * 80}\n[{node_name}] LLM OUTPUT\n{'-' * 80}")
    if isinstance(result, BaseMessage):
        print(result.content)
    else:
        print(pformat(result.model_dump(), width=100, sort_dicts=False))



# -----------------------------------
#             Main nodes
# -----------------------------------
def intent_detector(llm, state: MainState) -> dict:
    """Classify the latest request and initialise state for the new turn."""

    msg = _last_user_text(state)

    llm_input = [
        ("system", prompts.INTENT_SYSTEM_PROMPT),
        ("human", "Conversation history:\n" + _history(state)),
        ("human", f"User message: {msg}"),
    ]
    # _debug_print_llm_input("intent_detector", llm_input)

    result = llm.with_structured_output(Intent).invoke(llm_input)
    # _debug_print_llm_output("intent_detector", result)

    return {
        "intent": result.intent,
        "task_plan": [],
        "lending_result": "",
        "pending_lending_action": None,
        "approval_granted": False,
        "response_attempts": 0,
        "evaluation_feedback": "",
        "revision_requested": False,
    }


def task_planner(llm, state: MainState) -> dict:
    """Create an ordered execution plan based on the detected intent."""

    intent = state["intent"]

    # General conversation
    if intent == "general":
        return {"task_plan": []}

    history = _history(state)

    llm_input = [
        ("system", prompts.TASK_PLANNER_SYSTEM_PROMPT),
        (
            "human",
            f"Detected intent: {intent}\n"
            f"Conversation context:\n{history}\n"
            f"Search results:\n{state.get("search_results")}\n"
            f"Latest user message: {_last_user_text(state)}\n"
        ),
    ]
    # _debug_print_llm_input("task_planner", llm_input)

    result = llm.with_structured_output(TaskList).invoke(llm_input)
    # _debug_print_llm_output("task_planner", result)

    return {"task_plan": [task.model_dump() for task in result.tasks]}


def rag_node(rag_graph, state: MainState) -> dict:
    """Run the RAG subgraph for the planned search task."""

    rag_state = {
        "messages": state["messages"],
        "search_task_description": get_search_task(state["task_plan"]),
    }

    result = rag_graph.invoke(rag_state)

    return {"search_results": result.get("fused_results", [])}


def lending_node(llm, lending_graph, state: MainState) -> dict:
    lending_state = {
        "user_id": state["user_id"],
        "messages": state["messages"],
        "lending_task_description": get_lending_task(state["task_plan"]),
        "search_results": state.get("search_results", []),
    }

    result = lending_graph.invoke(lending_state)

    return {
        "lending_result": result.get(
            "lending_result",
            "No lending result returned."
        )
    }


def _agent_results(state: MainState) -> str:
    """Format the conversation and agent results shared by the response nodes."""

    results = f"""
Conversation history:
{_history(state)}

Latest user message:
{_last_user_text(state)}

Catalogue search results:
{state.get("search_results", [])}
"""

    # Omit the section entirely when nothing was done, so the response
    # doesn't comment on the absence of a lending operation.
    if lending_result := state.get("lending_result"):
        results += f"""
Lending result:
{lending_result}
"""

    return results


def response_generator(llm, state: MainState) -> dict:
    """Generate the reply, revising it if the Evaluator requested changes."""

    context = _agent_results(state)

    if state.get("revision_requested"):
        context += f"""
Your previous response was rejected by the reviewer.
Reviewer feedback:
{state["evaluation_feedback"]}

Write a new response that fixes these issues.
"""

    llm_input = [
        SystemMessage(content=prompts.RESPONSE_GENERATOR_SYSTEM_PROMPT),
        HumanMessage(content=context),
    ]
    # _debug_print_llm_input("response_generator", llm_input)

    result = llm.invoke(llm_input)
    # _debug_print_llm_output("response_generator", result)

    return {
        "messages": [AIMessage(content=result.content)],
        "response_attempts": state.get("response_attempts", 0) + 1,
    }


def evaluator(llm, state: MainState) -> dict:
    """Check the generated response and request a revision if it is inadequate."""

    draft = state["messages"][-1]

    # The draft is the last message, so the context is built without it.
    context_state = {**state, "messages": state["messages"][:-1]}

    llm_input = [
        ("system", prompts.EVALUATOR_SYSTEM_PROMPT),
        (
            "human",
            f"{_agent_results(context_state)}\n"
            f"Response to evaluate:\n{draft.content}\n"
        ),
    ]
    # _debug_print_llm_input("evaluator", llm_input)

    result = llm.with_structured_output(Evaluation).invoke(llm_input)
    # _debug_print_llm_output("evaluator", result)

    retries_used = state.get("response_attempts", 1) - 1

    if result.passed or retries_used >= MAX_RESPONSE_RETRIES:
        # Passed, or out of retries: keep the latest draft as the final reply.
        return {
            "evaluation_feedback": "" if result.passed else result.feedback,
            "revision_requested": False,
        }

    # Remove the rejected draft so the next attempt replaces it in the history.
    return {
        "messages": [RemoveMessage(id=draft.id)],
        "evaluation_feedback": result.feedback,
        "revision_requested": True,
    }


# -----------------------------------
#         RAG subgraph nodes
# -----------------------------------
def query_processor(llm, state: RagState) -> dict:
    """Turn the search task into keyword, semantic, and metadata queries."""

    context = (
        f"Search task: {state['search_task_description']}\n"
        f"Latest user message: {_last_user_text(state)}\n"
        f"Conversation context:\n{_history(state)}\n"
    )

    if state.get("retry_retrieval"):
        context += (
            f"\nPrevious attempt:\n"
            f"Retrieval plan: {state.get('retrieval_plan')}\n"
            f"Results: {state.get('fused_results', [])}\n"
            f"Feedback: {state.get('retrieval_feedback', '')}\n"
        )

    llm_input = [
        ("system", prompts.QUERY_PROCESSOR_SYSTEM_PROMPT),
        ("human", context),
    ]
    # _debug_print_llm_input("query_processor", llm_input)

    plan = llm.with_structured_output(RetrievalPlan, method="json_schema").invoke(llm_input)
    # _debug_print_llm_output("query_processor", plan)

    # Guarantee that at least one search runs with task description fallback
    if not plan.keyword_query.strip() and not plan.semantic_query.strip():
        plan.semantic_query = state["search_task_description"]

    return {
        "retrieval_plan": plan.model_dump(),
        # Clear the previous attempt's results; a skipped search must not leave stale results behind.
        "semantic_results": [],
        "lexical_results": [],
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }


def semantic_search_node(state: RagState) -> dict:
    """Search the catalogue with the semantic query."""

    plan = state["retrieval_plan"]
    results = semantic_search.invoke({
        "query": plan["semantic_query"],
        "metadata_filter": plan["metadata_filter"],
    })

    return {"semantic_results": results}


def lexical_search_node(state: RagState) -> dict:
    """Search the catalogue with the keyword query."""

    plan = state["retrieval_plan"]
    results = lexical_search.invoke({
        "query": plan["keyword_query"],
        "metadata_filter": plan["metadata_filter"],
    })

    return {"lexical_results": results}


def result_fusion(state: RagState) -> dict:
    """Merge the search results with Reciprocal Rank Fusion."""

    scores: dict[str, float] = {}
    books:  dict[str, dict]  = {}

    for results in (state.get("semantic_results", []), state.get("lexical_results", [])):
        for rank, book in enumerate(results, start=1):
            book_id = book["book_id"]
            scores[book_id] = scores.get(book_id, 0.0) + 1 / (RRF_K + rank)
            books[book_id] = book

    ranked_ids = sorted(scores, key=scores.get, reverse=True)

    return {"fused_results": [books[book_id] for book_id in ranked_ids[:MAX_SEARCH_RESULTS]]}


def retrieval_evaluator(state: RagState) -> dict:
    """Judge whether the fused results are relevant enough to the search task."""

    # TODO: implement a cross-encoder reranker to score the results against the query.
    passed   = True
    feedback = ""

    retries_used = state.get("retrieval_attempts", 1) - 1

    if passed or retries_used >= MAX_RETRIEVAL_RETRIES:
        # Passed, or out of retries: keep the latest results.
        return {"retrieval_feedback": feedback, "retry_retrieval": False}

    return {"retrieval_feedback": feedback, "retry_retrieval": True}


# -----------------------------------
#       Lending subgraph nodes
# -----------------------------------
def lending_agent(llm, state: LendingState) -> dict:
    """Determine and prepare the required lending operation."""

    # Drop book descriptions: irrelevant for lending, and shorter context helps the small model.
    results_list = []
    for book in state.get("search_results", []):
        results_list.append({key: value for key, value in book.items() if key != "description"})

    # Prepare context for the agent.
    context = {
        "user_id": state["user_id"],
        "task": state["lending_task_description"],
        "search_results": results_list,
        "latest_user_message": _last_user_text(state),
    }

    messages = [
        SystemMessage(content=prompts.LENDING_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=str(context)),
    ]

    # workaround for plain-text function calls
    # original: llm_with_tools = llm.bind_tools(LENDING_TOOLS)
    structured_llm = llm.with_structured_output(
        LendingAction,
        method="json_schema",
    )
    tool_map = {tool.name: tool for tool in LENDING_TOOLS}

    max_iterations = 5

    for _ in range(max_iterations):
        # _debug_print_llm_input("lending_agent", messages)
        action = structured_llm.invoke(messages)
        # _debug_print_llm_output("lending_agent", action)

        # Keep the decision in the history so the model sees what it already did.
        messages.append(AIMessage(content=action.model_dump_json()))

        if action.operation == "no_action":
            return {
                "lending_result": action.reason,
                "pending_lending_action": None,
            }

        # Build the tool arguments ourselves; user_id always comes from state.
        if action.operation == "renew_loan":
            tool_args = {"user_id": state["user_id"], "loan_id": action.loan_id}
        elif action.operation == "check_availability":
            tool_args = {"book_id": action.book_id}
        else:  # make_reservation / cancel_reservation
            tool_args = {"user_id": state["user_id"], "book_id": action.book_id}

        # The schema allows missing IDs; stop instead of calling a tool with None.
        if any(value is None for value in tool_args.values()):
            return {
                "lending_result": f"Cannot perform {action.operation}: missing required ID.",
                "pending_lending_action": None,
            }

        if action.operation in WRITE_TOOLS:
            return {
                "pending_lending_action": {
                    "tool_name": action.operation,
                    "arguments": tool_args,
                }
            }

        tool_result = tool_map[action.operation].invoke(tool_args)
        messages.append(HumanMessage(content=f"Result of {action.operation}: {tool_result}"))

    return {
        "lending_result": (
            "The Lending Agent reached its maximum tool-call "
            "iterations without completing the task."
        ),
        "pending_lending_action": None,
    }


def lending_approval(state: LendingState) -> dict:
    """Ask the user to approve the proposed lending operation."""

    action = state["pending_lending_action"]

    decision = interrupt({
        "tool_name": action["tool_name"],
        "arguments": action["arguments"],
    })

    return {"approval_granted": (isinstance(decision, dict) and decision.get("approved") is True)}


def lending_executor(state: LendingState) -> dict:
    """Execute an approved lending operation or record its rejection."""

    action = state["pending_lending_action"]

    if not state.get("approval_granted", False):
        return {
            "lending_result": (
                "The user declined the operation. "
                "No changes were made."
            ),
            "pending_lending_action": None,
        }

    tool_map = {tool.name: tool for tool in LENDING_TOOLS}
    tool_name = action["tool_name"]

    selected_tool = tool_map[tool_name]
    result = selected_tool.invoke(action["arguments"])

    return {
        "lending_result": str(result),
        "pending_lending_action": None,
    }


