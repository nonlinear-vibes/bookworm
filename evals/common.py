"""Helpers shared by the end-to-end evaluations (agent_test, load_test).

Run the evaluation scripts as modules from the project root, so this
package is importable:
  uv run python -m evals.agent_test.run_agent_eval
  docker compose run --rm eval python -m evals.agent_test.run_agent_eval

The bookworm imports are inside the functions on purpose: use_scratch_db()
must run before any bookworm module that reads the settings is imported.
"""

import os
import time
import shutil
import tempfile
from pathlib import Path
from datetime import datetime


def use_scratch_db() -> Path:
    """Point the app at a copy of the database, so the demo data is never changed.

    Must run before any other bookworm module is imported: settings are
    read once, and the other modules bind them at import time.
    """
    from bookworm import config

    scratch = Path(tempfile.mkdtemp(prefix="bookworm-eval-")) / "bookworm.db"
    shutil.copy(config.settings.sqlite_path, scratch)
    os.environ["SQLITE_PATH"] = str(scratch)
    config.settings = config.load_settings()
    return scratch


def prepare_user(setup: dict | None = None) -> dict:
    """Create a demo user and apply a case setup.

    setup.copies:       {book_id: number of FREE copies before the run}
    setup.reservations: [book_id, ...] active reservations the user already has
    Returns the user ID and the ID of the user's first active loan.
    """
    from bookworm.lending.db import connect
    from bookworm.lending.repository import HELD_COPIES_SQL
    from bookworm.lending.accounts import active_loans, create_demo_user

    user_id = create_demo_user()
    setup   = setup or {}

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
    """The user's loans and reservations, for before/after comparisons."""
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


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile of `values`, p in [0, 100]."""
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def run_turn(graph, payload: dict, config: dict, approve: bool,
             steps: list | None = None) -> tuple[list[dict], float]:
    """Stream one user turn through the graph until it ends.

    Every approval request is answered with `approve` and the graph is
    resumed. Returns the steps in order (node updates, subgraph nodes
    included, and approval requests) and the latency in seconds. Pass
    `steps` to share the list with a callback that records more events.
    """
    from langgraph.types import Command

    steps = [] if steps is None else steps
    start = time.perf_counter()

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

    return steps, time.perf_counter() - start
