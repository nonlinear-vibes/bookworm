"""User account helpers shared by the CLI and the web UI.

Create demo users, read a user's loans and reservations, and describe
pending lending actions for the approval prompt.
"""

import random
import sqlite3
import hashlib
from uuid import uuid4
from datetime import date, timedelta

from bookworm.config import settings
from bookworm.lending.db import connect, init_schema
from bookworm.lending.repository import HELD_COPIES_SQL, LOAN_PERIOD_DAYS


USER_ID_HEX_CHARS    = 4
MIN_LOANS, MAX_LOANS = 2, 3


def make_user_id(identity: str) -> str:
    """Build a short ID like 'UR-7F3A' from an identity string."""
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"UR-{digest[:USER_ID_HEX_CHARS].upper()}"


# --- Queries ---
def active_loans(conn: sqlite3.Connection, user_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT l.loan_id, l.book_id, l.due_date, l.renewals, b.title, b.authors
        FROM loans l JOIN books b ON b.book_id = l.book_id
        WHERE l.user_id = ? AND l.returned_at IS NULL
        ORDER BY l.loaned_at, l.loan_id
        """,
        (user_id,),
    ).fetchall()


def active_reservations(conn: sqlite3.Connection, user_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT r.book_id, r.created_at, b.title, b.authors
        FROM reservations r JOIN books b ON b.book_id = r.book_id
        WHERE r.user_id = ? AND r.status = 'active'
        ORDER BY r.created_at
        """,
        (user_id,),
    ).fetchall()


# --- Demo setup ---
def require_books(conn: sqlite3.Connection) -> None:
    """Raise RuntimeError if the books table is empty."""

    if not conn.execute("SELECT 1 FROM books LIMIT 1").fetchone():
        raise RuntimeError(
            f"No books in {settings.sqlite_path}. "
            "Run: uv run python scripts/init_db.py --limit 2000 --reset"
        )


def loan_random_books(conn: sqlite3.Connection, user_id: str, count: int) -> None:
    """Loan `count` random books to the user.

    Only books with at least two free copies are picked, so a copy stays
    available for others after the loan. Raises RuntimeError if there are
    not enough such books.
    """
    candidates = [
        row["book_id"]
        for row in conn.execute(
            f"""
            SELECT b.book_id
            FROM books b
            WHERE b.total_copies - {HELD_COPIES_SQL} >= 2
              AND b.book_id NOT IN (SELECT book_id FROM loans
                                    WHERE user_id = ? AND returned_at IS NULL)
            """,
            (user_id,),
        )
    ]
    if len(candidates) < count:
        raise RuntimeError("Not enough available books to create loans. Run scripts/init_db.py first.")

    today = date.today()
    due   = today + timedelta(days=LOAN_PERIOD_DAYS)
    for book_id in random.sample(candidates, count):
        conn.execute(
            "INSERT INTO loans (loan_id, user_id, book_id, loaned_at, due_date) VALUES (?, ?, ?, ?, ?)",
            (f"loan-{uuid4().hex[:8]}", user_id, book_id, today.isoformat(), due.isoformat()),
        )


def create_demo_user() -> str:
    """Create a new user with a fresh UR-XXXX ID and 2-3 random loans; return the ID.

    The user's name is the same as the ID. Raises RuntimeError if the
    database has no (or too few) books.
    """
    with connect() as conn:
        init_schema(conn)
        require_books(conn)

        # Short IDs can collide with an earlier session's user, so retry.
        for _ in range(100):
            user_id = make_user_id(uuid4().hex)
            if not conn.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone():
                break
        else:
            raise RuntimeError("Could not find a free user ID. Increase USER_ID_HEX_CHARS.")

        conn.execute("INSERT INTO users (user_id, name) VALUES (?, ?)", (user_id, user_id))
        loan_random_books(conn, user_id, random.randint(MIN_LOANS, MAX_LOANS))

    return user_id


# --- Approval ---
def describe_action(request: dict) -> str:
    """Turn an approval request into a readable sentence."""

    tool_name = request.get("tool_name", "unknown operation")
    args      = request.get("arguments", {})

    with connect() as conn:
        if "book_id" in args:
            row = conn.execute("SELECT title FROM books WHERE book_id = ?", (args["book_id"],)).fetchone()
            target = f"'{row['title']}'" if row else f"book {args['book_id']}"
        elif "loan_id" in args:
            row = conn.execute(
                "SELECT b.title FROM loans l JOIN books b ON b.book_id = l.book_id WHERE l.loan_id = ?",
                (args["loan_id"],),
            ).fetchone()
            target = f"loan {args['loan_id']}" + (f" ('{row['title']}')" if row else "")
        else:
            target = str(args)

    verbs = {
        "make_reservation": "Reserve",
        "cancel_reservation": "Cancel the reservation for",
        "renew_loan": "Renew",
    }
    return f"{verbs.get(tool_name, tool_name)} {target}"
