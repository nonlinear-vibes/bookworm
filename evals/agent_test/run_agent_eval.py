"""Run the end-to-end cases in agent_cases.json and save one trace per case.

Every case runs for a fresh demo user on a scratch copy of the SQLite
database, so the demo data is never changed. A trace (traces/AXX.json)
holds the case, every node update in order (subgraph nodes included),
the lending tool calls made inside the Lending Agent's loop, the
approval requests with the answers given, the user's loans and
reservations before and after the run, the final answer and the latency.
Score the traces by hand in agent.md.

Usage, from the project root (see compose.yaml):
  uv run python -m evals.agent_test.run_agent_eval [case_id ...]
  docker compose run --rm eval python -m evals.agent_test.run_agent_eval [case_id ...]
"""

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

from evals.common import account, prepare_user, run_turn, to_json, use_scratch_db

TEST_DIR   = Path(__file__).resolve().parent
TRACES_DIR = TEST_DIR / "traces"


def tool_recorder(steps: list):
    """Callback that adds lending tool calls to the trace.

    The Lending Agent calls check_availability inside a single node, so
    these calls do not appear as graph updates. The search tools are
    skipped: their results are already in the rag/* updates.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    class ToolRecorder(BaseCallbackHandler):
        def __init__(self):
            self.open = {}

        def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
            name = (serialized or {}).get("name", "")
            if name not in ("semantic_search", "lexical_search"):
                self.open[run_id] = {"tool_call": name, "input": kwargs.get("inputs") or input_str}
                steps.append(self.open[run_id])

        def on_tool_end(self, output, *, run_id, **kwargs):
            if run_id in self.open:
                self.open.pop(run_id)["output"] = getattr(output, "content", output)

    return ToolRecorder()


def run_case(graph, case: dict) -> dict:
    from langchain_core.messages import AIMessage, HumanMessage

    user    = prepare_user(case.get("setup"))
    before  = account(user["user_id"])
    message = case["message"].format(**user)
    history = [HumanMessage(content=text) if role == "human" else AIMessage(content=text)
               for role, text in case.get("history", [])]

    steps   = []
    config  = {"configurable": {"thread_id": f"eval-{case['id']}-{uuid4()}"},
               "callbacks": [tool_recorder(steps)]}
    payload = {"user_id": user["user_id"], "messages": [*history, HumanMessage(content=message)]}

    # An approval request pauses the graph; it is resumed with the case's answer.
    steps, latency = run_turn(graph, payload, config, approve=bool(case.get("approve", False)), steps=steps)
    state = graph.get_state(config).values

    return {
        "case":      {**case, "message": message},
        "user":      user,
        "latency_s": round(latency, 2),
        "final": {
            "intent":         state.get("intent"),
            "task_plan":      state.get("task_plan"),
            "lending_result": state.get("lending_result"),
            "answer":         state["messages"][-1].content,
        },
        "account_before": before,
        "account_after":  account(user["user_id"]),
        "steps":          steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case_ids", nargs="*", help="Run only these cases (default: all).")
    args = parser.parse_args()

    use_scratch_db()
    from bookworm.graph.workflow import build_graph

    cases = json.loads((TEST_DIR / "agent_cases.json").read_text(encoding="utf-8"))["cases"]
    if args.case_ids:
        unknown = set(args.case_ids) - {c["id"] for c in cases}
        if unknown:
            sys.exit(f"Unknown case id(s): {sorted(unknown)}")
        cases = [c for c in cases if c["id"] in args.case_ids]

    graph = build_graph()
    TRACES_DIR.mkdir(exist_ok=True)

    for case in cases:
        trace = run_case(graph, case)
        path  = TRACES_DIR / f"{case['id']}.json"
        path.write_text(json.dumps(trace, indent=2, ensure_ascii=False, default=to_json), encoding="utf-8")
        print(f"{case['id']}  {trace['latency_s']:>6.1f}s  {case['category']}")


if __name__ == "__main__":
    main()
