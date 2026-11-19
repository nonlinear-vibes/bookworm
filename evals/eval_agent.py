"""End-to-end evaluation of the agentic workflow (main graph).

Every case in agent_cases.json runs for a fresh demo user on a scratch copy
of the SQLite database, so the demo data is never changed. Per case:

  intent     the Intent Detector's label is one of the accepted labels
  route      the agents that ran (rag / lending) match the expected route
  approval   an approval request appears exactly when expected, for the
             right operation and the right book or loan
  outcome    the database (or the lending result) ends up as expected, and
             nothing was written without an approved request
  task       route, approval and outcome are all correct
  grounded   every book title the final answer mentions is backed by the
             context: search results, the user's own loans/reservations,
             the lending result, or the conversation (heuristic, see
             common.extract_title_mentions)
  consistent the answer agrees with the lending result: it names the book
             the operation acted on, and does not report a failed
             operation as a success (heuristic)

Usage (inside the eval container, see compose.yaml):
  docker compose run --rm eval python evals/eval_agent.py [case_id ...]
Writes results/agent.json and results/agent.md.
"""

import re
import sys
import time
import argparse
from uuid import uuid4

from common import (
    extract_title_mentions, load_json, markdown_table, mean, normalize_title,
    setup_environment, titles_match, write_results, write_text,
)

AGENT_NODES = {"rag": "search", "lending": "lending"}


# -----------------------------------
#         Database helpers
# -----------------------------------
def prepare_user(case: dict) -> dict:
    """Create a demo user and apply the case's setup; return the user's context."""
    from datetime import datetime

    from bookworm.lending.db import connect
    from bookworm.lending.repository import HELD_COPIES_SQL
    from bookworm.lending.accounts import active_loans, create_demo_user

    user_id = create_demo_user()
    setup   = case.get("setup", {})

    with connect() as conn:
        # `copies` is the number of *available* copies. Demo users get random
        # loans, so another user (or this one) may already hold a copy; the
        # total is raised by whatever is held to keep cases independent.
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

    return {"user_id": user_id, "loan_id": loan["loan_id"], "loan_title": loan["title"]}


def account_snapshot(user_id: str) -> dict:
    from bookworm.lending.db import connect

    with connect() as conn:
        loans = conn.execute(
            "SELECT loan_id, book_id, due_date, renewals, returned_at FROM loans WHERE user_id = ? ORDER BY loan_id",
            (user_id,),
        ).fetchall()
        reservations = conn.execute(
            "SELECT book_id, status FROM reservations WHERE user_id = ? ORDER BY reservation_id",
            (user_id,),
        ).fetchall()
    return {"loans": [tuple(r) for r in loans], "reservations": [tuple(r) for r in reservations]}


def account_titles(user_id: str) -> list[str]:
    from bookworm.lending.db import connect

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT b.title FROM loans l JOIN books b ON b.book_id = l.book_id WHERE l.user_id = ?
            UNION
            SELECT b.title FROM reservations r JOIN books b ON b.book_id = r.book_id WHERE r.user_id = ?
            """,
            (user_id, user_id),
        ).fetchall()
    return [row["title"] for row in rows]


def catalogue_titles() -> list[str]:
    from bookworm.lending.db import connect

    with connect() as conn:
        return [row["title"] for row in conn.execute("SELECT title FROM books")]


def fill(value, user: dict):
    """Fill {loan_id} / {loan_title} placeholders in strings."""
    return value.format(**user) if isinstance(value, str) else value


# -----------------------------------
#              Checks
# -----------------------------------
def check_approval(expected, requests: list[dict], user: dict) -> tuple[bool, str]:
    if expected == "any":
        return len(requests) <= 1, f"{len(requests)} request(s)"
    if expected is None:
        return not requests, "none expected" if not requests else f"unexpected: {requests}"
    if len(requests) != 1:
        return False, f"expected 1 request, got {len(requests)}"

    request = requests[0]
    args    = request.get("arguments", {})
    if request.get("tool_name") != expected["tool"]:
        return False, f"wrong tool {request.get('tool_name')}"
    for key in ("book_id", "loan_id"):
        if key in expected and args.get(key) != fill(expected[key], user):
            return False, f"wrong {key} {args.get(key)}"
    return True, "ok"


def check_outcome(case: dict, user: dict, before: dict, lending_result: str) -> tuple[bool, list[str]]:
    from bookworm.lending.db import connect

    expect   = case["expect"]
    failures = []

    with connect() as conn:
        for check in expect.get("outcome", []):
            if "reservation_active" in check:
                row = conn.execute(
                    "SELECT 1 FROM reservations WHERE user_id = ? AND book_id = ? AND status = 'active'",
                    (user["user_id"], check["reservation_active"]),
                ).fetchone()
                if bool(row) != check["value"]:
                    failures.append(f"reservation_active({check['reservation_active']}) != {check['value']}")
            elif "loan_renewed" in check:
                row = conn.execute("SELECT renewals FROM loans WHERE loan_id = ?",
                                   (fill(check["loan_renewed"], user),)).fetchone()
                if (row is not None and row["renewals"] > 0) != check["value"]:
                    failures.append(f"loan_renewed != {check['value']}")
            elif "lending_result_contains" in check:
                if check["lending_result_contains"].lower() not in (lending_result or "").lower():
                    failures.append(f"lending_result lacks '{check['lending_result_contains']}'")

    # Without an approved, expected write, the account must stay unchanged.
    expects_write = isinstance(expect.get("approval"), dict) and case.get("approve", False)
    if not expects_write and expect.get("approval") != "any" and account_snapshot(user["user_id"]) != before:
        failures.append("unexpected database change")

    return not failures, failures


# The lending tools return messages like "Successfully reserved 'Title'." or
# "Reservation failed: ..."; a failed or impossible operation must not be
# reported to the user as a success.
FAILURE_MARKERS = ("failed", "cannot perform", "no book with id", "declined", "maximum tool-call")
SUCCESS_CLAIM   = re.compile(
    r"\b(yes\b|successfully|has been (reserved|renewed|cancelled)|"
    r"i(?:'ve| have) (reserved|renewed|cancelled)|is (currently )?available)", re.IGNORECASE)


def check_lending_consistency(answer: str, lending_result: str) -> tuple[bool | None, str]:
    """Does the answer agree with what the lending agent actually did?

    None when no lending operation ran. Checks that the book named in the
    lending result is the one the answer talks about, and that a failure is
    not presented as a success.
    """
    if not lending_result:
        return None, ""

    named = re.findall(r"'(.+?)'(?=[:.]|\s|$)", lending_result)
    if named and not any(titles_match(named[0], m) for m in extract_title_mentions(answer)) \
            and normalize_title(named[0]) not in normalize_title(answer):
        return False, f"answer does not mention '{named[0]}' from the lending result"

    if any(marker in lending_result.lower() for marker in FAILURE_MARKERS):
        claim = SUCCESS_CLAIM.search(answer)
        if claim:
            return False, f"reports success ('{claim.group(0)}') but the lending result is: {lending_result}"

    return True, ""


def check_grounding(answer: str, context_titles: list[str], context_text: str, catalogue: list[str]) -> dict:
    mentions   = extract_title_mentions(answer)
    norm_text  = normalize_title(context_text)
    grounded, ungrounded = [], []

    for mention in mentions:
        if any(titles_match(mention, t) for t in context_titles) or \
                (len(normalize_title(mention)) >= 8 and normalize_title(mention) in norm_text):
            grounded.append(mention)
        else:
            in_catalogue = any(titles_match(mention, t) for t in catalogue)
            ungrounded.append({"mention": mention, "in_catalogue": in_catalogue})

    return {"mentions": len(mentions), "grounded": grounded, "ungrounded": ungrounded}


# -----------------------------------
#               Run
# -----------------------------------
def run_case(graph, case: dict, catalogue: list[str]) -> dict:
    from langgraph.types import Command
    from langchain_core.messages import AIMessage, HumanMessage

    user   = prepare_user(case)
    before = account_snapshot(user["user_id"])
    expect = case["expect"]

    history  = [HumanMessage(content=t) if role == "human" else AIMessage(content=t)
                for role, t in case.get("history", [])]
    message  = fill(case["message"], user)
    config   = {"configurable": {"thread_id": f"eval-{case['id']}-{uuid4()}"}}
    payload  = {"user_id": user["user_id"], "messages": [*history, HumanMessage(content=message)]}

    visited, requests = [], []
    start = time.perf_counter()

    while payload is not None:
        resume = None
        for update in graph.stream(payload, config=config, stream_mode="updates"):
            for node, value in update.items():
                if node == "__interrupt__":
                    request = value[0].value
                    requests.append(request)
                    # Only an expected request gets the case's decision.
                    approve = bool(case.get("approve")) and expect.get("approval") is not None
                    resume  = Command(resume={"approved": approve})
                else:
                    visited.append(node)
        payload = resume

    latency = time.perf_counter() - start
    state   = graph.get_state(config).values

    agents = [AGENT_NODES[n] for n in ("rag", "lending") if n in visited]
    route  = "+".join(agents) or "none"
    answer = state["messages"][-1].content if state.get("messages") else ""

    approval_ok, approval_note = check_approval(expect.get("approval"), requests, user)
    outcome_ok, outcome_notes  = check_outcome(case, user, before, state.get("lending_result", ""))

    search_titles  = [b["title"] for b in state.get("search_results", [])]
    context_titles = search_titles + account_titles(user["user_id"])
    context_text   = " ".join([*(t for _, t in case.get("history", [])), message, state.get("lending_result", "")])
    grounding      = check_grounding(answer, context_titles, context_text, catalogue)
    consistent, consistency_note = check_lending_consistency(answer, state.get("lending_result", ""))

    route_ok = route == expect["route"]
    return {
        "id": case["id"], "category": case["category"], "message": message,
        "latency_s": round(latency, 2),
        "intent": state.get("intent"), "intent_ok": state.get("intent") in expect["intent"],
        "route": route, "route_ok": route_ok,
        "task_plan": state.get("task_plan"),
        "approval_requests": requests, "approval_ok": approval_ok, "approval_note": approval_note,
        "outcome_ok": outcome_ok, "outcome_notes": outcome_notes,
        "task_ok": route_ok and approval_ok and outcome_ok,
        "revisions": max(visited.count("response_generator") - 1, 0),
        "lending_result": state.get("lending_result", ""),
        "search_results": search_titles,
        "grounding": grounding,
        "lending_consistent": consistent, "consistency_note": consistency_note,
        "answer": answer,
    }


def report_markdown(results: list[dict]) -> str:
    def rate(key):
        return mean([float(r[key]) for r in results])

    mentions   = sum(r["grounding"]["mentions"] for r in results)
    grounded   = sum(len(r["grounding"]["grounded"]) for r in results)
    with_titles = [r for r in results if r["grounding"]["mentions"]]
    latencies  = sorted(r["latency_s"] for r in results)
    lending    = [r for r in results if r["lending_consistent"] is not None]

    summary = [
        ["Intent accuracy", rate("intent_ok")],
        ["Route accuracy", rate("route_ok")],
        ["Approval correctness", rate("approval_ok")],
        ["Outcome correctness", rate("outcome_ok")],
        ["Task success (route + approval + outcome)", rate("task_ok")],
        ["Grounded title mentions", f"{grounded}/{mentions}" + (f" ({grounded / mentions:.2f})" if mentions else "")],
        ["Answers consistent with the lending result",
         f"{sum(1 for r in lending if r['lending_consistent'])}/{len(lending)}"],
        ["Answers with only grounded titles",
         f"{sum(1 for r in with_titles if not r['grounding']['ungrounded'])}/{len(with_titles)}"],
        ["Responses revised by the Evaluator", f"{sum(1 for r in results if r['revisions'])}/{len(results)}"],
        ["Latency per case: median / max (s)", f"{latencies[len(latencies) // 2]:.1f} / {latencies[-1]:.1f}"],
    ]

    def mark(ok):
        return "✓" if ok else "✗"

    rows = [[r["id"], r["category"], f"{mark(r['intent_ok'])} {r['intent']}", f"{mark(r['route_ok'])} {r['route']}",
             mark(r["approval_ok"]), mark(r["outcome_ok"]),
             "" if r["lending_consistent"] is None else mark(r["lending_consistent"]),
             f"{len(r['grounding']['grounded'])}/{r['grounding']['mentions']}", r["latency_s"]]
            for r in results]

    failures = [f"- **{r['id']}**: " + "; ".join(
                    ([f"intent {r['intent']}"] if not r["intent_ok"] else []) +
                    ([f"route {r['route']}"] if not r["route_ok"] else []) +
                    ([f"approval: {r['approval_note']}"] if not r["approval_ok"] else []) +
                    r["outcome_notes"] +
                    ([r["consistency_note"]] if r["lending_consistent"] is False else []) +
                    [f"ungrounded '{u['mention']}'" + (" (in catalogue, not retrieved)" if u["in_catalogue"] else "")
                     for u in r["grounding"]["ungrounded"]])
                for r in results
                if not (r["intent_ok"] and r["task_ok"]) or r["grounding"]["ungrounded"]
                or r["lending_consistent"] is False]

    return "\n\n".join([
        "## Summary", markdown_table(["Metric", "Value"], summary),
        "## Per case", markdown_table(
            ["ID", "Category", "Intent", "Route", "Approval", "Outcome", "Consistent", "Grounded", "Latency (s)"],
            rows),
        "## Failures and ungrounded mentions", "\n".join(failures) or "None.",
    ]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case_ids", nargs="*", help="Run only these cases (default: all).")
    args = parser.parse_args()

    setup_environment(scratch_db=True)

    from bookworm.graph.workflow import build_graph

    cases = load_json("agent_cases.json")["cases"]
    if args.case_ids:
        unknown = set(args.case_ids) - {c["id"] for c in cases}
        if unknown:
            sys.exit(f"Unknown case id(s): {sorted(unknown)}")
        cases = [c for c in cases if c["id"] in args.case_ids]

    graph     = build_graph()
    catalogue = catalogue_titles()
    results   = []

    for case in cases:
        result = run_case(graph, case, catalogue)
        status = "PASS" if result["task_ok"] and result["intent_ok"] else "FAIL"
        print(f"{case['id']} {status} {result['route']:<15} {result['latency_s']:>6.1f}s  {case['category']}")
        results.append(result)

    md = report_markdown(results)
    # A partial run must not overwrite the full report.
    suffix = "" if not args.case_ids else "-partial"
    json_path = write_results(f"agent{suffix}.json", {"cases": results})
    write_text(json_path.with_suffix(".md"), md)

    print("\n" + md)
    print(f"Wrote {json_path.name} and {json_path.with_suffix('.md').name} to {json_path.parent}")


if __name__ == "__main__":
    main()
