"""Progress reporting and latency tracking for the Bookworm graphs.

``GraphObserver`` is a LangChain callback handler. Pass it in the run config
(``config={"callbacks": [observer]}``) and it receives events from every graph
node, including nodes of subgraphs, every tool call and every LLM call. It:

* reports a human-readable status line when a node or tool starts, through an
  ``on_event`` callback (``print`` in the CLI, a Streamlit widget in a UI);
* records the duration of each node, tool and LLM call, for latency and
  bottleneck analysis (``summary()`` / ``to_records()``).
"""

import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.errors import GraphInterrupt


NODE_LABELS: dict[str, str] = {
    # Main graph
    "intent_detector":     "Analysing intent...",
    "task_planner":        "Planning tasks...",
    "rag":                 "Searching the catalogue...",
    "lending":             "Handling lending request...",
    "response_generator":  "Generating response...",
    "evaluator":           "Checking answer...",
    # RAG subgraph
    "query_processor":     "Writing search query...",
    "semantic_search":     "Running semantic search...",
    "lexical_search":      "Running keyword search...",
    "result_fusion":       "Merging results...",
    "retrieval_evaluator": "Checking search results...",
    # Lending subgraph
    "lending_agent":       "Deciding on the lending action...",
    "approval":            "Waiting for approval...",
    "executor":            "Executing the lending action...",
    # Lending tools (the search tools share their node's label, so they are silent)
    "check_availability":  "Checking availability...",
    "make_reservation":    "Reserving the book...",
    "cancel_reservation":  "Cancelling the reservation...",
    "renew_loan":          "Renewing the loan...",
}

SILENT_TOOLS = {"semantic_search", "lexical_search"}

SpanKind = Literal["node", "tool", "llm"]


@dataclass
class Span:
    kind: SpanKind
    name: str
    node: str | None  # graph node the span ran in
    start: float  # time.perf_counter() value
    duration_s: float | None = None
    status: Literal["running", "ok", "error", "interrupted"] = "running"


class GraphObserver(BaseCallbackHandler):
    """Collects timing spans and emits status messages for a graph run."""

    def __init__(self, on_event: Callable[[str], None] | None = None) -> None:
        self.on_event = on_event
        self.spans: list[Span] = []
        self._open: dict[UUID, Span] = {}
        self._lock = threading.Lock()

    # --- helpers ---
    def _emit(self, name: str) -> None:
        if self.on_event is not None:
            self.on_event(NODE_LABELS.get(name, f"{name}..."))

    def _open_span(self, run_id: UUID, kind: SpanKind, name: str, node: str | None) -> None:
        span = Span(kind=kind, name=name, node=node, start=time.perf_counter())
        with self._lock:
            self._open[run_id] = span
            self.spans.append(span)

    def _close_span(self, run_id: UUID, status: str) -> None:
        with self._lock:
            span = self._open.pop(run_id, None)
        if span is not None:
            span.duration_s = time.perf_counter() - span.start
            span.status     = status

    # --- graph nodes ---
    def on_chain_start(self, serialized: dict[str, Any] | None, inputs: Any, *, run_id: UUID,
                       metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        node = (metadata or {}).get("langgraph_node")
        # Inner runnables, routers and subgraph wrappers carry the same metadata
        # but a different run name. Only the node runnable itself matches.
        if node is None or kwargs.get("name") != node:
            return
        self._open_span(run_id, "node", node, node)
        self._emit(node)

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._close_span(run_id, "ok")

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close_span(run_id, "interrupted" if isinstance(error, GraphInterrupt) else "error")

    # --- tools ---
    def on_tool_start(self, serialized: dict[str, Any] | None, input_str: str, *, run_id: UUID,
                      metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name", "tool")
        self._open_span(run_id, "tool", name, (metadata or {}).get("langgraph_node"))
        if name not in SILENT_TOOLS:
            self._emit(name)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._close_span(run_id, "ok")

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close_span(run_id, "error")

    # --- LLM calls ---
    def on_chat_model_start(self, serialized: dict[str, Any] | None, messages: Any, *, run_id: UUID,
                            metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        node = (metadata or {}).get("langgraph_node")
        self._open_span(run_id, "llm", f"llm@{node}" if node else "llm", node)

    def on_llm_start(self, serialized: dict[str, Any] | None, prompts: list[str], *, run_id: UUID,
                     metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self.on_chat_model_start(serialized, prompts, run_id=run_id, metadata=metadata, **kwargs)

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._close_span(run_id, "ok")

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close_span(run_id, "error")

    # --- reporting ---
    def to_records(self) -> list[dict[str, Any]]:
        """Finished spans as plain dicts, with start relative to the first span."""

        with self._lock:
            spans = [s for s in self.spans if s.duration_s is not None]
        if not spans:
            return []
        t0 = min(s.start for s in spans)
        return [{**asdict(s), "start": round(s.start - t0, 4), "duration_s": round(s.duration_s, 4)}
                for s in spans]

    def summary(self) -> str:
        """Table of count/total/mean/max seconds per (kind, name), slowest first."""

        groups: dict[tuple[str, str], list[float]] = defaultdict(list)
        for record in self.to_records():
            groups[(record["kind"], record["name"])].append(record["duration_s"])
        if not groups:
            return "(no timings recorded)"

        rows  = sorted(groups.items(), key=lambda item: sum(item[1]), reverse=True)
        width = max(len(name) for _, name in groups) + 2
        lines = [f"{'kind':<5} {'name':<{width}} {'n':>3} {'total s':>8} {'mean s':>8} {'max s':>8}"]
        for (kind, name), durations in rows:
            total = sum(durations)
            lines.append(f"{kind:<5} {name:<{width}} {len(durations):>3} {total:>8.2f} "
                         f"{total / len(durations):>8.2f} {max(durations):>8.2f}")
        return "\n".join(lines)
