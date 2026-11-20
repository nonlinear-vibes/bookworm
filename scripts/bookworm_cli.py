"""Interactive command-line chat with the full Bookworm agent.

On start a new demo user (UR-XXXX) is created and "logged in" automatically,
with 2-3 randomly selected books on loan, the same way as in the web UI.

Requires Ollama, Qdrant (with ingested books), and the SQLite database
created by scripts/init_db.py.

Run with:  uv run python scripts/bookworm_cli.py [--debug] [--timings] [--timings-log PATH]
"""

import argparse
import json
import sys
import time
from datetime import datetime
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from bookworm.graph import nodes
from bookworm.graph.workflow import build_graph
from bookworm.lending.accounts import (
    active_loans,
    active_reservations,
    create_demo_user,
    describe_action,
)
from bookworm.lending.db import connect
from bookworm.observability import GraphObserver

# Gray (ANSI "bright black") for status and timing output; plain text when not a terminal.
_GRAY, _RESET = ("\033[90m", "\033[0m") if sys.stdout.isatty() else ("", "")


def gray(text: str) -> str:
    return f"{_GRAY}{text}{_RESET}"

HELP_TEXT = """Commands:
  /loans   show your loans and reservations
  /new     start a new conversation (forget the chat history)
  /help    show this help
  /quit    exit (also: /exit, Ctrl+D)"""


# --- Account ---
def print_account(user_id: str) -> None:
    with connect() as conn:
        loans = active_loans(conn, user_id)
        reservations = active_reservations(conn, user_id)

    print("Your loans:")
    if not loans:
        print("  (none)")
    for loan in loans:
        by = f" by {loan['authors']}" if loan["authors"] else ""
        print(f"  [{loan['loan_id']}] '{loan['title']}'{by} - due {loan['due_date']}"
              f" (renewed {loan['renewals']}x)")

    print("Your reservations:")
    if not reservations:
        print("  (none)")
    for res in reservations:
        by = f" by {res['authors']}" if res["authors"] else ""
        print(f"  '{res['title']}'{by} - reserved {res['created_at']}")


# --- Approval ---
def ask_approval(request: dict) -> bool:
    print(f"\n[approval needed] {describe_action(request)}")
    while True:
        try:
            answer = input("Approve? [y/n] ").strip().lower()
        except EOFError:
            return False
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


# --- Chat ---
def run_turn(graph, config: dict, user_id: str, text: str, observer: GraphObserver | None = None) -> str:
    """Send one user message through the graph, handling approval interrupts."""

    payload = {"user_id": user_id, "messages": [HumanMessage(content=text)]}
    run_config = {**config, "callbacks": [observer]} if observer else config

    while payload is not None:
        resume = None
        for update in graph.stream(payload, config=run_config, stream_mode="updates"):
            if "__interrupt__" in update:
                request = update["__interrupt__"][0].value
                resume = Command(resume={"approved": ask_approval(request)})
        payload = resume

    messages = graph.get_state(config).values.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage):
        return str(messages[-1].content)
    return "(no response)"


def report_timings(observer: GraphObserver, text: str, wall_s: float, args) -> None:
    """Print and/or log the timing spans of one turn.

    The wall time includes any time spent waiting at the approval prompt.
    """

    if args.timings:
        print(gray(f"\n[timings] turn wall time {wall_s:.2f}s"))
        print(gray(observer.summary()))
    if args.timings_log:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "query": text,
            "wall_s": round(wall_s, 4),
            "spans": observer.to_records(),
        }
        with open(args.timings_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def new_config(user_id: str) -> dict:
    return {"configurable": {"thread_id": f"{user_id}-{uuid4()}"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--debug", action="store_true", help="Print the state each graph node receives and returns.")
    parser.add_argument("--timings", action="store_true",
                        help="Print node, tool and LLM execution times after each reply.")
    parser.add_argument("--timings-log", metavar="PATH",
                        help="Append each turn's timing spans as a JSON line to PATH.")
    args = parser.parse_args()

    nodes.PRINT_STATE = args.debug

    try:
        user_id = create_demo_user()
    except RuntimeError as exc:
        sys.exit(str(exc))

    print("=" * 60)
    print(f"Welcome to Bookworm! Logged in as {user_id}.")
    print("=" * 60)
    print_account(user_id)
    print()
    print(HELP_TEXT)

    graph = build_graph()
    config = new_config(user_id)

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/help":
            print(HELP_TEXT)
            continue
        if text == "/loans":
            print_account(user_id)
            continue
        if text == "/new":
            config = new_config(user_id)
            print("Started a new conversation.")
            continue

        observer = GraphObserver(on_event=lambda message: print(gray(f"  {message}"), flush=True))
        started = time.perf_counter()
        try:
            reply = run_turn(graph, config, user_id, text, observer)
        except KeyboardInterrupt:
            print("\n(interrupted)")
            continue
        except Exception as exc:  # keep the session alive on model/service errors
            print(f"\n[error] {type(exc).__name__}: {exc}")
            continue

        print(f"\nBookworm: {reply}")
        report_timings(observer, text, time.perf_counter() - started, args)

    print("Goodbye!")


if __name__ == "__main__":
    main()
