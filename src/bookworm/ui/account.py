"""Reads loans, reservations and availability for the UI."""


import sqlite3

from bookworm.lending.db import connect
from bookworm.lending.repository import HELD_COPIES_SQL
from bookworm.lending.accounts import active_loans, active_reservations


def availability(book_ids: list[str]) -> dict[str, tuple[int, int]]:
    """Map book_id -> (available copies, total copies) for books in the database."""

    if not book_ids:
        return {}
    placeholders = ", ".join("?" * len(book_ids))
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT b.book_id, b.total_copies, {HELD_COPIES_SQL} AS held
            FROM books b
            WHERE b.book_id IN ({placeholders})
            """,
            book_ids,
        ).fetchall()
    return {
        row["book_id"]: (max(row["total_copies"] - row["held"], 0), row["total_copies"])
        for row in rows
    }


def load_account(user_id: str) -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
    with connect() as conn:
        return active_loans(conn, user_id), active_reservations(conn, user_id)
