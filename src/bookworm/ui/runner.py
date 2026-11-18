"""Run one graph step for the UI while reporting live progress.

The graph runs in a worker thread so the Streamlit script thread stays free
to show the GraphObserver's status messages as they arrive. Streamlit
elements must only be touched from the script thread, so the observer just
queues the messages and the script thread drains the queue.
"""

import queue
import threading
from collections.abc import Callable

from bookworm.observability import GraphObserver


POLL_INTERVAL_S = 0.1


def run_until_pause(graph, payload, config: dict, on_status: Callable[[str], None]) -> dict | None:
    """Stream `payload` through the graph until it finishes or asks for approval.

    `payload` is either new input or a langgraph Command that resumes an
    interrupted run. Returns the approval request if the graph was
    interrupted, else None. Exceptions from the graph are re-raised here.
    """
    events: queue.SimpleQueue[str] = queue.SimpleQueue()
    observer = GraphObserver(on_event=events.put)
    run_config = {**config, "callbacks": [observer]}
    outcome: dict = {}

    def worker() -> None:
        try:
            for update in graph.stream(payload, config=run_config, stream_mode="updates"):
                if "__interrupt__" in update:
                    outcome["interrupt"] = update["__interrupt__"][0].value
        except Exception as exc:  # handed over to the script thread
            outcome["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while thread.is_alive() or not events.empty():
        try:
            on_status(events.get(timeout=POLL_INTERVAL_S))
        except queue.Empty:
            pass
    thread.join()

    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("interrupt")
