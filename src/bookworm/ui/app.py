"""Bookworm web UI: account and search results on the left, chat on the right.

Each browser session is a new demo user (UR-XXXX) with 2-3 random loans.
The chat runs the same graph as scripts/bookworm_cli.py; lending operations
that change the database pause for approval with Approve/Decline buttons.

Requires Ollama, Qdrant (with ingested books), and the SQLite database
created by scripts/init_db.py.

Run with:  uv run streamlit run src/bookworm/ui/app.py
"""

from uuid import uuid4
from datetime import date, datetime
import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from bookworm.graph.workflow import build_graph
from bookworm.lending.accounts import create_demo_user, describe_action
from bookworm.ui.account import availability, load_account
from bookworm.ui.runner import run_until_pause


DESCRIPTION_PREVIEW_CHARS = 130

st.set_page_config(page_title="Bookworm", page_icon="📚", layout="wide")

# Fit both columns to the window so the page itself never scrolls and the chat
# input stays in view; the panels scroll inside instead. Heights are the window
# minus what sits above and below each panel (headings, chat input, padding).
st.html("""
<style>
[data-testid="stMainBlockContainer"] { padding-top: 3rem; padding-bottom: 1rem; }
.st-key-left_panel { height: max(20rem, calc(100dvh - 9.5rem)) !important; }
.st-key-chat       { height: max(16rem, calc(100dvh - 13.5rem)) !important; }
</style>
""")


@st.cache_resource(show_spinner="Loading the Bookworm agent...")
def get_graph():
    # One compiled graph for all sessions; each conversation has its own thread_id.
    return build_graph()


# --- Session state ---
def new_conversation() -> None:
    st.session_state.thread_id = f"{st.session_state.user_id}-{uuid4()}"
    st.session_state.pending_approval = None
    st.session_state.error = None


def decide(approved: bool) -> None:
    st.session_state.approval_decision = approved
    st.session_state.pending_approval = None


if "user_id" not in st.session_state:
    try:
        st.session_state.user_id = create_demo_user()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    new_conversation()

user_id = st.session_state.user_id
config  = {"configurable": {"thread_id": st.session_state.thread_id}}
graph   = get_graph()
values  = graph.get_state(config).values


# --- Left column: account and search results ---
def render_account() -> None:
    loans, reservations = load_account(user_id)

    with st.container(border=True):
        st.subheader("👤 Account", anchor=False)
        c1, c2, c3 = st.columns(3)
        c1.metric("User ID", user_id)
        c2.metric("Active loans", len(loans))
        c3.metric("Reservations", len(reservations))

        st.markdown("**Loans**")
        if loans:
            st.dataframe(
                [{"Title": l["title"], "Authors": l["authors"],
                  "Due": date.fromisoformat(l["due_date"]), "Renewals": l["renewals"],
                  "Loan ID": l["loan_id"]} for l in loans],
                hide_index=True,
                column_config={
                    "Due": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "Renewals": st.column_config.NumberColumn(width="small"),
                },
            )
        else:
            st.caption("No active loans.")

        st.markdown("**Reservations**")
        if reservations:
            st.dataframe(
                [{"Title": r["title"], "Authors": r["authors"],
                  "Reserved": datetime.fromisoformat(r["created_at"]), "Book ID": r["book_id"]}
                 for r in reservations],
                hide_index=True,
                column_config={"Reserved": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")},
            )
        else:
            st.caption("No active reservations.")


def render_search_results(results: list[dict]) -> None:
    with st.container(border=True):
        st.subheader("🔎 Latest search results", anchor=False)
        if not results:
            st.caption("Search results appear here once you ask Bookworm to find books.")
            return

        copies = availability([book["book_id"] for book in results])
        for book in results:
            with st.container(border=True):
                st.markdown(f"**{book['title']}**")
                byline = " · ".join(str(part) for part in (book.get("author"), book.get("year")) if part)
                if byline:
                    st.caption(byline)

                with st.container(horizontal=True, gap="small"):
                    if book["book_id"] in copies:
                        free, total = copies[book["book_id"]]
                        st.badge(f"{free} of {total} available", icon=":material/library_books:",
                                 color="green" if free else "red")
                    for genre in filter(None, (g.strip() for g in (book.get("genres") or "").split(","))):
                        st.badge(genre, color="gray")
                    st.badge(book["book_id"], color="blue")

                description = book.get("description") or ""
                if len(description) > DESCRIPTION_PREVIEW_CHARS:
                    with st.expander(description[:DESCRIPTION_PREVIEW_CHARS].rsplit(" ", 1)[0] + "..."):
                        st.write(description)
                elif description:
                    st.write(description)


# --- Right column: chat ---
def render_history(messages: list) -> None:
    if not messages:
        with st.chat_message("assistant", avatar="📚"):
            st.markdown(
                f"Hi **{user_id}**! I'm Bookworm, your library assistant. I can find books, "
                "check availability, reserve books and renew your loans. What are you looking for?"
            )
    for message in messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage) and message.content:
            with st.chat_message("assistant", avatar="📚"):
                st.markdown(message.content)


def render_approval(request: dict) -> None:
    with st.chat_message("assistant", avatar="📚"):
        st.warning(f"**Approval needed:** {describe_action(request)}", icon=":material/gavel:")
        with st.container(horizontal=True):
            st.button("Approve", icon=":material/check:", type="primary", on_click=decide, args=(True,))
            st.button("Decline", icon=":material/close:", on_click=decide, args=(False,))


def run_step(payload, label: str) -> None:
    """Run the graph until it finishes or pauses, showing live progress."""

    with st.chat_message("assistant", avatar="📚"):
        with st.status(label) as status:
            def on_status(message: str) -> None:
                status.update(label=message)
                st.write(message)

            try:
                st.session_state.pending_approval = run_until_pause(graph, payload, config, on_status)
            except Exception as exc:  # keep the session alive on model/service errors
                st.session_state.error = f"{type(exc).__name__}: {exc}"
                status.update(label="Something went wrong", state="error")
                return
            status.update(label="Done", state="complete")


left, right = st.columns([5, 6], gap="large")

with left:
    st.title("📚 Bookworm", anchor=False)
    with st.container(height=600, border=False, key="left_panel"):
        render_account()
        render_search_results(values.get("search_results", []))

with right:
    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.subheader("💬 Chat", anchor=False)
        st.button("New conversation", icon=":material/refresh:", on_click=new_conversation)

    chat = st.container(height=500, border=True, autoscroll=True, key="chat")
    pending = st.session_state.pending_approval
    with chat:
        render_history(values.get("messages", []))
        if st.session_state.error:
            st.error(st.session_state.error, icon=":material/error:")
        if pending:
            render_approval(pending)

    prompt = st.chat_input(
        "Approve or decline the pending action first" if pending
        else "Ask about books, loans or reservations...",
        disabled=pending is not None,
    )

    payload, label = None, ""
    decision = st.session_state.pop("approval_decision", None)
    if decision is not None:
        payload = Command(resume={"approved": decision})
        label = "Approved, continuing..." if decision else "Declined, continuing..."
    elif prompt and prompt.strip():
        payload = {"user_id": user_id, "messages": [HumanMessage(content=prompt.strip())]}
        label = "Thinking..."
        with chat, st.chat_message("user"):
            st.markdown(prompt.strip())

    if payload is not None:
        st.session_state.error = None
        with chat:
            run_step(payload, label)
        # Rerun so the history, account and search results all show the new state.
        st.rerun()
