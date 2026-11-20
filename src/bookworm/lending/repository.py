"""Lending operations on the SQLite database.

Each function returns a short, user-facing result message, which the
lending tools pass back to the graph.
"""

import sqlite3
from datetime import date, datetime, timedelta

LOAN_PERIOD_DAYS = 21
MAX_RENEWALS     = 2

# Copies of book `b` that are taken: out on loan or held by an active
# reservation. Use with the books table aliased as `b`.
HELD_COPIES_SQL = """
    ((SELECT COUNT(*) FROM loans l
      WHERE l.book_id = b.book_id AND l.returned_at IS NULL)
   + (SELECT COUNT(*) FROM reservations r
      WHERE r.book_id = b.book_id AND r.status = 'active'))
"""


def _book_title(conn: sqlite3.Connection, book_id: str) -> str | None:
    row = conn.execute("SELECT title FROM books WHERE book_id = ?", (book_id,)).fetchone()
    return row["title"] if row else None


def available_copies(conn: sqlite3.Connection, book_id: str) -> str:
    row = conn.execute(
        f"""
        SELECT b.title, b.total_copies, {HELD_COPIES_SQL} AS held
        FROM books b
        WHERE b.book_id = ?
        """,
        (book_id,),
    ).fetchone()

    if row is None:
        return f"No book with ID {book_id} exists in the catalogue."

    # subtract them instead of storing a derived variable, 
    # single source of truth
    available = max(row["total_copies"] - row["held"], 0)
    if available == 0:
        return f"'{row['title']}': No available copies."
    return f"'{row['title']}': {available} of {row['total_copies']} copies available."


def make_reservation(conn: sqlite3.Connection, user_id: str, book_id: str) -> str:
    title = _book_title(conn, book_id)
    if title is None:
        return f"Reservation failed: no book with ID {book_id} exists in the catalogue."

    existing = conn.execute(
        "SELECT 1 FROM reservations WHERE user_id = ? AND book_id = ? AND status = 'active'",
        (user_id, book_id),
    ).fetchone()
    if existing:
        return f"Reservation failed: you already have an active reservation for '{title}'."

    # A reservation holds a copy, so it needs a free one. Checking and
    # inserting in one statement keeps two users from taking the last copy.
    cursor = conn.execute(
        f"""
        INSERT INTO reservations (user_id, book_id, created_at)
        SELECT ?, b.book_id, ?
        FROM books b
        WHERE b.book_id = ? AND b.total_copies - {HELD_COPIES_SQL} > 0
        """,
        (user_id, datetime.now().isoformat(timespec="seconds"), book_id),
    )
    if cursor.rowcount == 0:
        return f"Reservation failed: no copies of '{title}' are available."
    return f"Successfully reserved '{title}'."


def cancel_reservation(conn: sqlite3.Connection, user_id: str, book_id: str) -> str:
    title = _book_title(conn, book_id) or book_id

    cursor = conn.execute(
        "UPDATE reservations SET status = 'cancelled' "
        "WHERE user_id = ? AND book_id = ? AND status = 'active'",
        (user_id, book_id),
    )
    if cursor.rowcount == 0:
        return f"Cancellation failed: you have no active reservation for '{title}'."

    return f"Successfully cancelled the reservation for '{title}'."


def renew_loan(conn: sqlite3.Connection, user_id: str, loan_id: str) -> str:
    row = conn.execute(
        """
        SELECT l.due_date, l.renewals, l.returned_at, b.title
        FROM loans l JOIN books b ON b.book_id = l.book_id
        WHERE l.loan_id = ? AND l.user_id = ?
        """,
        (loan_id, user_id),
    ).fetchone()

    if row is None:
        return f"Renewal failed: you have no loan with ID {loan_id}."
    if row["returned_at"] is not None:
        return f"Renewal failed: '{row['title']}' has already been returned."
    if row["renewals"] >= MAX_RENEWALS:
        return f"Renewal failed: '{row['title']}' has already been renewed {MAX_RENEWALS} times."

    new_due = date.fromisoformat(row["due_date"]) + timedelta(days=LOAN_PERIOD_DAYS)
    conn.execute(
        "UPDATE loans SET due_date = ?, renewals = renewals + 1 WHERE loan_id = ?",
        (new_due.isoformat(), loan_id),
    )
    return f"Successfully renewed '{row['title']}'. New due date: {new_due.isoformat()}."
