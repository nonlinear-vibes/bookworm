"""Run the end-to-end cases in agent_cases.json and save one trace per case.

Every case runs for a fresh demo user on a scratch copy of the SQLite
database, so the demo data is never changed. A trace (traces/AXX.json)
holds the case, every node update in order (subgraph nodes included),
the lending tool calls made inside the Lending Agent's loop, the
approval requests with the answers given, the user's loans and
reservations before and after the run, the final answer and the latency.
Score the traces by hand in agent.md.

Usage (see compose.yaml):
  docker compose run --rm eval python evals/run_agent.py [case_id ...]
"""

import os
import sys
import json
import time
import shutil
import argparse
import tempfile
from uuid import uuid4
from pathlib import Path
from datetime import datetime

EVALS_DIR  = Path(__file__).resolve().parent
TRACES_DIR = EVALS_DIR / "traces"


def use_scratch_db() -> None:
    """Point the app at a copy of the database.

    Must run before any other bookworm module is imported: settings are
    read once, and the other modules bind them at import time.
    """
    from bookworm import config

    scratch = Path(tempfile.mkdtemp(prefix="bookworm-eval-")) / "bookworm.db"
    shutil.copy(config.settings.sqlite_path, scratch)
    os.environ["SQLITE_PATH"] = str(scratch)
    config.settings = config.load_settings()


def prepare_user(case: dict) -> dict:
    """Create a demo user and apply the case's setup."""
    from bookworm.lending.db import connect
    from bookworm.lending.repository import HELD_COPIES_SQL
    from bookworm.lending.accounts import active_loans, create_demo_user

    user_id = create_demo_user()
    setup   = case.get("setup", {})

    with connect() as conn:
        # `copies` is the number of FREE copies. Demo users hold random
        # loans, so the total is raised by the copies already held.
        for book_id, copies in setup.get("copies", {}).items():
            conn.execute(
                f"UPDATE books AS b SET total_copies = ? + {HELD_COPIES_SQL} WHERE b.book_id = ?",
                (copies, book_id),
            )
        for book_id in setup.get("reservations", []):
            conn.execute(
                "INSERT INTO reservations (user_id, book_id, created_at) VALUES (?, ?, ?)",
                (user_id, book_id, datetime.now().isoformat(timespec="seconds")),
            )
        loan = active_loans(conn, user_id)[0]

    return {"user_id": user_id, "loan_id": loan["loan_id"]}


def account(user_id: str) -> dict:
    from bookworm.lending.db import connect

    with connect() as conn:
        loans = conn.execute(
            """SELECT l.loan_id, l.book_id, b.title, l.due_date, l.renewals, l.returned_at
               FROM loans l JOIN books b ON b.book_id = l.book_id
               WHERE l.user_id = ? ORDER BY l.loan_id""", (user_id,)).fetchall()
        reservations = conn.execute(
            """SELECT r.book_id, b.title, r.status
               FROM reservations r JOIN books b ON b.book_id = r.book_id
               WHERE r.user_id = ? ORDER BY r.reservation_id""", (user_id,)).fetchall()
    return {"loans": [dict(r) for r in loans], "reservations": [dict(r) for r in reservations]}


def to_json(value):
    """json.dumps fallback for LangChain messages and Pydantic models."""
    from langchain_core.messages import BaseMessage

    if isinstance(value, BaseMessage):
        data = {"type": value.type, "content": value.content}
        if getattr(value, "tool_calls", None):
            data["tool_calls"] = value.tool_calls
        return data
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


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
    from langgraph.types import Command
    from langchain_core.messages import AIMessage, HumanMessage

    user    = prepare_user(case)
    before  = account(user["user_id"])
    message = case["message"].format(**user)
    history = [HumanMessage(content=text) if role == "human" else AIMessage(content=text)
               for role, text in case.get("history", [])]

    steps   = []
    config  = {"configurable": {"thread_id": f"eval-{case['id']}-{uuid4()}"},
               "callbacks": [tool_recorder(steps)]}
    payload = {"user_id": user["user_id"], "messages": [*history, HumanMessage(content=message)]}
    approve = bool(case.get("approve", False))
    start   = time.perf_counter()

    # An approval request pauses the graph; resume it with the case's answer.
    while payload is not None:
        resume = None
        for namespace, update in graph.stream(payload, config=config, stream_mode="updates", subgraphs=True):
            path = [part.split(":")[0] for part in namespace]
            for node, value in update.items():
                if node == "__interrupt__":
                    if resume is None:   # reported by the subgraph and the parent
                        steps.append({"approval_request": value[0].value, "approved": approve})
                        resume = Command(resume={"approved": approve})
                else:
                    steps.append({"node": "/".join([*path, node]), "update": value})
        payload = resume

    latency = time.perf_counter() - start
    state   = graph.get_state(config).values

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

    cases = json.loads((EVALS_DIR / "agent_cases.json").read_text(encoding="utf-8"))["cases"]
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
