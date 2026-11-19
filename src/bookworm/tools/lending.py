from langchain_core.tools import tool

from bookworm.lending import repository
from bookworm.lending.db import connect


@tool
def check_availability(book_id: str) -> str:
    """Check how many copies of a book are currently available.

    Args:
        book_id: The unique identifier of the book.
    """
    with connect() as conn:
        return repository.available_copies(conn, book_id)


@tool
def make_reservation(user_id: str, book_id: str) -> str:
    """Reserve a book for a library user.

    Args:
        user_id: The unique identifier of the user.
        book_id: The unique identifier of the book.
    """
    with connect() as conn:
        return repository.make_reservation(conn, user_id, book_id)


@tool
def cancel_reservation(user_id: str, book_id: str) -> str:
    """Cancel an existing book reservation.

    Args:
        user_id: The unique identifier of the user.
        book_id: The unique identifier of the book.
    """
    with connect() as conn:
        return repository.cancel_reservation(conn, user_id, book_id)


@tool
def renew_loan(user_id: str, loan_id: str) -> str:
    """Renew an existing library loan.

    Args:
        user_id: The unique identifier of the user.
        loan_id: The unique identifier of the loan.
    """
    with connect() as conn:
        return repository.renew_loan(conn, user_id, loan_id)


LENDING_TOOLS = [
    check_availability,
    make_reservation,
    cancel_reservation,
    renew_loan,
]

WRITE_TOOLS = {
    "make_reservation",
    "cancel_reservation",
    "renew_loan",
}
