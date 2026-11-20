"""Load test: send the queries in load_queries.json through the full graph.

Each query runs as one user turn for a fresh demo user on a scratch copy of
the SQLite database, through the same compiled graph the UI uses (no HTTP
layer: the UI runs the graph in-process too). Approval requests are always
approved, so lending actions really write to the (scratch) database.
`--concurrency` simulated users each send
their next query as soon as the previous one is answered (closed loop, no
think time).

One JSON line per request is appended to results/<scenario>.jsonl as soon as
it finishes, so an interrupted run keeps its results and can be continued
with --resume. Each line holds:
  * end-to-end latency, start/end offsets from the start of the run, worker;
  * the path taken: nodes run (subgraph nodes included), planned agents,
    approvals, response attempts (evaluator revisions);
  * spans from GraphObserver: duration of every node, tool and LLM call;
  * per LLM call, Ollama's token counts and its split of the call into model
    load, prompt processing (prefill) and generation (decode).

--retrieval-only skips the LLM: the raw query goes straight to the semantic
and keyword search and the fusion step, to measure the data side alone.

Usage, from the project root (see compose.yaml):
  docker compose run --rm eval python -m evals.load_test.run_load_test --scenario s1
  docker compose run --rm eval python -m evals.load_test.run_load_test --scenario s2 --concurrency 4
  docker compose run --rm eval python -m evals.load_test.run_load_test --retrieval-only
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from uuid import UUID, uuid4

from evals.common import percentile, prepare_user, run_turn, to_json, use_scratch_db

TEST_DIR     = Path(__file__).resolve().parent
QUERIES_FILE = TEST_DIR / "load_queries.json"
RESULTS_DIR  = TEST_DIR / "results"

# Run before the measured queries and not recorded: loads the chat and the
# embedding model into GPU memory (the first request pays ~45 s for that).
WARMUP_MESSAGES = ["Hi!", "Do you have any books about cooking?"]

NS = 1e9  # Ollama reports durations in nanoseconds


def ollama_stats(message) -> dict:
    """Token counts and timings Ollama returns with each response."""
    meta = getattr(message, "response_metadata", None) or {}

    def seconds(key):
        return round(meta[key] / NS, 4) if meta.get(key) is not None else None

    return {
        "prompt_tokens": meta.get("prompt_eval_count"),
        "output_tokens": meta.get("eval_count"),
        "load_s":        seconds("load_duration"),
        "prefill_s":     seconds("prompt_eval_duration"),
        "decode_s":      seconds("eval_duration"),
        "ollama_total_s": seconds("total_duration"),
    }


def llm_recorder(calls: list):
    """Callback that adds Ollama's stats for every LLM call to `calls`."""
    from langchain_core.callbacks import BaseCallbackHandler

    class LLMRecorder(BaseCallbackHandler):
        def __init__(self):
            self.nodes: dict[UUID, str | None] = {}

        def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
            self.nodes[run_id] = (metadata or {}).get("langgraph_node")

        def on_llm_end(self, response, *, run_id, **kwargs):
            generation = response.generations[0][0] if response.generations and response.generations[0] else None
            calls.append({"node": self.nodes.pop(run_id, None),
                          **ollama_stats(getattr(generation, "message", None))})

    return LLMRecorder()


def run_query(graph, query: dict, t0: float) -> dict:
    """Run one query as one user turn; never raises."""
    from langchain_core.messages import HumanMessage

    from bookworm.observability import GraphObserver

    record = {"id": query["id"], "category": query["category"], "message": query["message"],
              "worker": threading.current_thread().name}
    observer  = GraphObserver()
    llm_calls: list[dict] = []
    start = time.perf_counter()

    try:
        user    = prepare_user()
        config  = {"configurable": {"thread_id": f"load-{query['id']}-{uuid4()}"},
                   "callbacks": [observer, llm_recorder(llm_calls)]}
        payload = {"user_id": user["user_id"], "messages": [HumanMessage(content=query["message"])]}
        steps, latency = run_turn(graph, payload, config, approve=True)
        state = graph.get_state(config).values

        found = [book["book_id"] for book in state.get("search_results") or []]
        record.update({
            "status":            "ok",
            "latency_s":         round(latency, 3),
            "intent":            state.get("intent"),
            "agents":            sorted({task["agent"] for task in state.get("task_plan") or []}),
            "nodes":             [step["node"] for step in steps if "node" in step],
            "approvals":         sum("approval_request" in step for step in steps),
            "response_attempts": state.get("response_attempts"),
            "n_results":         len(found),
            "target_found":      query["book_id"] in found if "book_id" in query else None,
            "answer":            state["messages"][-1].content,
        })
    except Exception as exc:  # recorded, so one failure does not stop the run
        record.update({"status": "error", "latency_s": round(time.perf_counter() - start, 3),
                       "error": f"{type(exc).__name__}: {exc}"})

    record.update({
        "t_start":   round(start - t0, 3),
        "t_end":     round(time.perf_counter() - t0, 3),
        "llm_calls": llm_calls,
        "spans":     observer.to_records(),
    })
    return record


def run_retrieval_query(query: dict, t0: float) -> dict:
    """Search only: the raw query through semantic search, keyword search and fusion."""
    from bookworm.graph.nodes import result_fusion
    from bookworm.retrieval import search

    record = {"id": query["id"], "category": query["category"], "message": query["message"],
              "worker": threading.current_thread().name}
    start = time.perf_counter()
    try:
        timings = {}
        t = time.perf_counter()
        semantic = search.semantic_search(query["message"])
        timings["semantic_search_s"] = round(time.perf_counter() - t, 4)
        t = time.perf_counter()
        lexical = search.lexical_search(query["message"])
        timings["lexical_search_s"] = round(time.perf_counter() - t, 4)
        t = time.perf_counter()
        fused = result_fusion({"semantic_results": semantic, "lexical_results": lexical})["fused_results"]
        timings["fusion_s"] = round(time.perf_counter() - t, 4)

        found = [book["book_id"] for book in fused]
        record.update({"status": "ok", "latency_s": round(time.perf_counter() - start, 4), **timings,
                       "n_results": len(found),
                       "target_found": query["book_id"] in found if "book_id" in query else None})
    except Exception as exc:
        record.update({"status": "error", "latency_s": round(time.perf_counter() - start, 4),
                       "error": f"{type(exc).__name__}: {exc}"})

    record.update({"t_start": round(start - t0, 3), "t_end": round(time.perf_counter() - t0, 3)})
    return record


def print_summary(records: list[dict], wall_s: float) -> None:
    ok = [r for r in records if r["status"] == "ok"]
    print(f"\n{len(records)} requests, {len(ok)} ok, {len(records) - len(ok)} errors, "
          f"wall time {wall_s:.1f} s, throughput {len(records) / wall_s * 60:.2f} req/min")
    if not ok:
        return
    latencies = [r["latency_s"] for r in ok]
    print("latency s: " + "  ".join(f"p{p}={percentile(latencies, p):.2f}" for p in (50, 90, 95, 99))
          + f"  mean={mean(latencies):.2f}  max={max(latencies):.2f}")

    by_node: dict[str, list[float]] = {}
    for record in ok:
        for span in record.get("spans", []):
            if span["kind"] == "llm":
                by_node.setdefault(span["name"], []).append(span["duration_s"])
    if by_node:
        total = sum(latencies)
        print("LLM time by node (share of all end-to-end time):")
        for name, durations in sorted(by_node.items(), key=lambda item: -sum(item[1])):
            print(f"  {name:<28} n={len(durations):>4}  mean={mean(durations):6.2f} s  "
                  f"share={sum(durations) / total:6.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", help="Name of the results file (default: s1, or 'retrieval' "
                                           "with --retrieval-only).")
    parser.add_argument("--concurrency", type=int, default=1, help="Simulated users (default: 1).")
    parser.add_argument("--n", type=int, help="Run only the first N queries (default: all).")
    parser.add_argument("--retrieval-only", action="store_true", help="Search only, no LLM and no graph.")
    parser.add_argument("--no-warmup", action="store_true", help="Skip the warm-up requests.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true", help="Skip queries already in the results file.")
    group.add_argument("--overwrite", action="store_true", help="Delete the existing results file first.")
    args = parser.parse_args()

    scenario = args.scenario or ("retrieval" if args.retrieval_only else "s1")
    out_path = RESULTS_DIR / f"{scenario}.jsonl"
    queries  = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))["queries"][:args.n]

    RESULTS_DIR.mkdir(exist_ok=True)
    if out_path.exists():
        if args.overwrite:
            out_path.unlink()
        elif args.resume:
            done = {json.loads(line)["id"] for line in out_path.read_text(encoding="utf-8").splitlines() if line}
            queries = [q for q in queries if q["id"] not in done]
            print(f"Resuming: {len(done)} done, {len(queries)} left.")
        else:
            sys.exit(f"{out_path.relative_to(TEST_DIR)} exists: pass --resume or --overwrite.")

    use_scratch_db()

    if args.retrieval_only:
        def task(query, t0):
            return run_retrieval_query(query, t0)
        warmup = lambda: [run_retrieval_query({"id": "warmup", "category": "warmup", "message": m}, 0.0)
                          for m in WARMUP_MESSAGES]
    else:
        from bookworm.graph.workflow import build_graph
        graph = build_graph()

        def task(query, t0):
            return run_query(graph, query, t0)
        warmup = lambda: [run_query(graph, {"id": "warmup", "category": "warmup", "message": m}, 0.0)
                          for m in WARMUP_MESSAGES]

    if not args.no_warmup:
        print("Warming up...")
        for record in warmup():
            if record["status"] != "ok":
                sys.exit(f"Warm-up failed: {record.get('error')}")

    print(f"Scenario {scenario}: {len(queries)} queries, concurrency {args.concurrency} -> "
          f"{out_path.relative_to(TEST_DIR)}")
    records: list[dict] = []
    lock = threading.Lock()
    t0   = time.perf_counter()

    with out_path.open("a", encoding="utf-8") as out, \
         ThreadPoolExecutor(max_workers=args.concurrency, thread_name_prefix="user") as pool:
        futures = [pool.submit(task, query, t0) for query in queries]
        for future in as_completed(futures):
            record = {"scenario": scenario, "concurrency": args.concurrency, **future.result()}
            with lock:
                out.write(json.dumps(record, ensure_ascii=False, default=to_json) + "\n")
                out.flush()
                records.append(record)
            status = record["status"] if record["status"] == "ok" else f"ERROR {record.get('error')}"
            print(f"[{len(records):>3}/{len(queries)}] {record['id']}  {record['latency_s']:>7.2f} s  "
                  f"{record['category']:<12} {status}")

    print_summary(records, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
