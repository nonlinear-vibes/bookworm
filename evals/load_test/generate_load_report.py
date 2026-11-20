"""Summarise a load test run (results/<scenario>.jsonl) into load_<scenario>.md.

Sections:
  * latency: p50, p90, p95, p99, max and mean, wall time, throughput;
  * latency by request type;
  * share of the total end-to-end time per node: the six main-graph nodes
    split each request's latency between them, subgraph nodes are listed
    under the node that ran them (rag, lending), and LLM and tool calls are
    totalled separately, since they run inside nodes;
  * LLM calls per node: tokens and Ollama's prefill / decode split;
  * latency histogram;
  * lending actions taken, per request type;
  * if results/retrieval.jsonl exists, the search-only run for comparison.

Uses only the standard library, so it runs on the host as well:
  python3 -m evals.load_test.generate_load_report [--scenario s1]
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from evals.common import percentile

TEST_DIR    = Path(__file__).resolve().parent
RESULTS_DIR = TEST_DIR / "results"

MAIN_NODES = ["intent_detector", "task_planner", "rag", "lending", "response_generator", "evaluator"]
SUBGRAPH_NODES = {
    "rag":     ["query_processor", "semantic_search", "lexical_search", "result_fusion", "retrieval_evaluator"],
    "lending": ["lending_agent", "approval", "executor"],
}
CATEGORIES = ["topic", "author", "known_item", "availability", "reservation", "general"]

HISTOGRAM_BIN_S = 2
HISTOGRAM_WIDTH = 50


def load(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"Missing {path.relative_to(TEST_DIR)}: run run_load_test.py first.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def pct(part: float, whole: float) -> str:
    return f"{part / whole:.1%}" if whole else "–"


def latency_row(latencies: list[float]) -> list[str]:
    return [f"{percentile(latencies, p):.2f}" for p in (50, 90, 95, 99)] + \
           [f"{max(latencies):.2f}", f"{mean(latencies):.2f}"]


# --- sections ---

def summary_section(records: list[dict]) -> list[str]:
    ok        = [r for r in records if r["status"] == "ok"]
    latencies = [r["latency_s"] for r in ok]
    wall      = max(r["t_end"] for r in records) - min(r["t_start"] for r in records)

    lines = ["## Latency", ""]
    lines += table(["Requests", "Errors", "Concurrency", "Wall time", "Throughput"],
                   [[len(records), len(records) - len(ok), records[0]["concurrency"],
                     f"{wall / 60:.1f} min", f"{len(records) / wall * 60:.2f} req/min"]])
    lines += [""]
    lines += table(["p50 s", "p90 s", "p95 s", "p99 s", "max s", "mean s"], [latency_row(latencies)])
    return lines


def by_category_section(ok: list[dict]) -> list[str]:
    rows = []
    for category in CATEGORIES:
        group = [r for r in ok if r["category"] == category]
        if not group:
            continue
        latencies = [r["latency_s"] for r in group]
        rows.append([category, len(group),
                     f"{percentile(latencies, 50):.2f}", f"{percentile(latencies, 90):.2f}",
                     f"{max(latencies):.2f}", f"{mean(latencies):.2f}",
                     f"{mean(len(r['llm_calls']) for r in group):.1f}"])
    return ["## Latency by request type", ""] + \
        table(["Type", "n", "p50 s", "p90 s", "max s", "mean s", "LLM calls / request"], rows)


def node_section(ok: list[dict]) -> list[str]:
    total = sum(r["latency_s"] for r in ok)

    # Per node: total seconds and the number of requests that ran it. A node
    # interrupted for approval appears twice (before and after); both count.
    seconds: dict[str, float] = defaultdict(float)
    runs:    dict[str, int]   = defaultdict(int)
    for record in ok:
        seen = set()
        for span in record["spans"]:
            if span["kind"] == "node":
                seconds[span["name"]] += span["duration_s"]
                seen.add(span["name"])
        for name in seen:
            runs[name] += 1

    def row(label: str, name: str) -> list[str]:
        n = runs.get(name, 0)
        return [label, n, f"{seconds[name] / n:.3f}" if n else "–", f"{seconds[name]:.1f}", pct(seconds[name], total)]

    rows = []
    for node in MAIN_NODES:
        rows.append(row(f"**{node}**", node))
        for sub in SUBGRAPH_NODES.get(node, []):
            rows.append(row(f"&nbsp;&nbsp;↳ {sub}", sub))
    in_nodes = sum(seconds[node] for node in MAIN_NODES)
    rows.append(["Graph overhead (outside nodes)", len(ok), f"{(total - in_nodes) / len(ok):.3f}",
                 f"{total - in_nodes:.1f}", pct(total - in_nodes, total)])

    # The same time, split by what it was spent on.
    llm   = sum(s["duration_s"] for r in ok for s in r["spans"] if s["kind"] == "llm")
    tools: dict[str, float] = defaultdict(float)
    tool_calls: dict[str, int] = defaultdict(int)
    for record in ok:
        for span in record["spans"]:
            if span["kind"] == "tool":
                tools[span["name"]] += span["duration_s"]
                tool_calls[span["name"]] += 1
    kind_rows = [["LLM calls", sum(len(r["llm_calls"]) for r in ok), f"{llm:.1f}", pct(llm, total)]]
    for name, secs in sorted(tools.items(), key=lambda item: -item[1]):
        kind_rows.append([f"Tool: {name}", tool_calls[name], f"{secs:.2f}", pct(secs, total)])
    rest = total - llm - sum(tools.values())
    kind_rows.append(["Everything else (graph, state, parsing)", "–", f"{rest:.1f}", pct(rest, total)])

    lines  = ["## Where the time goes", "",
              f"Share of the total end-to-end time of all successful requests ({total:.0f} s). "
              "The main nodes (bold) and the graph overhead add up to 100%; subgraph nodes are part "
              "of their parent's time. *Runs* is the number of requests that ran the node, and "
              "*mean s / run* is the time per such request (all loop iterations included).", ""]
    lines += table(["Node", "Runs", "mean s / run", "total s", "share"], rows)
    lines += ["", "By kind of work (LLM and tool calls run inside the nodes above):", ""]
    lines += table(["Work", "Calls", "total s", "share"], kind_rows)
    return lines


def llm_section(ok: list[dict]) -> list[str]:
    total  = sum(r["latency_s"] for r in ok)
    llm_s: dict[str, list[float]] = defaultdict(list)
    for record in ok:
        for span in record["spans"]:
            if span["kind"] == "llm":
                llm_s[span["node"]].append(span["duration_s"])
    stats: dict[str, list[dict]] = defaultdict(list)
    for record in ok:
        for call in record["llm_calls"]:
            stats[call["node"]].append(call)

    def avg(calls: list[dict], key: str) -> float:
        values = [c[key] for c in calls if c.get(key) is not None]
        return mean(values) if values else float("nan")

    rows = []
    for node, durations in sorted(llm_s.items(), key=lambda item: -sum(item[1])):
        calls   = stats.get(node, [])
        decoded = sum(c["output_tokens"] or 0 for c in calls)
        decode  = sum(c["decode_s"] or 0 for c in calls)
        rows.append([node, len(durations), f"{mean(durations):.2f}",
                     f"{avg(calls, 'prompt_tokens'):.0f}", f"{avg(calls, 'output_tokens'):.0f}",
                     f"{avg(calls, 'prefill_s'):.2f}", f"{avg(calls, 'decode_s'):.2f}",
                     f"{decoded / decode:.1f}" if decode else "–",
                     pct(sum(durations), total)])

    calls    = [c for r in ok for c in r["llm_calls"]]
    prefill  = sum(c["prefill_s"] or 0 for c in calls)
    decode   = sum(c["decode_s"] or 0 for c in calls)
    load     = sum(c["load_s"] or 0 for c in calls)
    lines  = ["## LLM calls", "",
              "Per call, from Ollama's response: prompt and output tokens, and the time spent "
              "processing the prompt (prefill) and generating the output (decode).", ""]
    lines += table(["Node", "Calls", "mean s", "prompt tok", "output tok", "prefill s", "decode s",
                    "decode tok/s", "share of e2e"], rows)
    lines += ["", f"Across all {len(calls)} calls: decode {decode:.0f} s, prefill {prefill:.0f} s, "
                  f"model load {load:.1f} s (decode = {pct(decode, decode + prefill + load)} of the "
                  f"model's working time)."]
    return lines


def histogram_section(ok: list[dict]) -> list[str]:
    latencies = [r["latency_s"] for r in ok]
    bins      = math.ceil(max(latencies) / HISTOGRAM_BIN_S)
    counts    = [0] * bins
    for latency in latencies:
        counts[min(int(latency // HISTOGRAM_BIN_S), bins - 1)] += 1
    scale = HISTOGRAM_WIDTH / max(counts)

    lines = ["## Latency histogram", "", "```"]
    for i, count in enumerate(counts):
        lo = i * HISTOGRAM_BIN_S
        lines.append(f"{lo:>3}–{lo + HISTOGRAM_BIN_S:<3} s | {'█' * round(count * scale):<{HISTOGRAM_WIDTH}} {count}")
    lines.append("```")
    return lines


def lending_section(ok: list[dict]) -> list[str]:
    rows = []
    for category in CATEGORIES:
        group = [r for r in ok if r["category"] == category]
        if not group:
            continue
        rows.append([category, len(group),
                     sum("lending" in r["agents"] for r in group),
                     sum(r["approvals"] > 0 for r in group),
                     sum(any(s["kind"] == "node" and s["name"] == "executor" for s in r["spans"]) for r in group),
                     sum(r["target_found"] is True for r in group) if group[0].get("target_found") is not None else "–"])
    lines  = ["## Paths taken", "",
              "Every approval request was approved, so each one ended in a lending action "
              "(a database write). *Target found*: the book the query was made from is among "
              "the search results.", ""]
    lines += table(["Type", "n", "Lending planned", "Approval requested", "Action executed", "Target found"], rows)
    return lines


def retrieval_section() -> list[str]:
    path = RESULTS_DIR / "retrieval.jsonl"
    if not path.exists():
        return []
    ok = [r for r in load(path) if r["status"] == "ok"]
    rows = [["End to end (search + fusion)"] + latency_row([r["latency_s"] for r in ok])]
    for key, label in (("semantic_search_s", "Semantic search (embedding + Qdrant)"),
                       ("lexical_search_s", "Keyword search (BM25)"),
                       ("fusion_s", "Fusion (RRF)")):
        rows.append([label] + latency_row([r[key] for r in ok]))
    lines  = ["## Search only, without the LLM", "",
              f"The same {len(ok)} queries sent as they are to the search layer (results/retrieval.jsonl).", ""]
    lines += table(["Step", "p50 s", "p90 s", "p95 s", "p99 s", "max s", "mean s"], rows)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default="s1", help="Results file to summarise (default: s1).")
    args = parser.parse_args()

    records = load(RESULTS_DIR / f"{args.scenario}.jsonl")
    ok      = [r for r in records if r["status"] == "ok"]
    if not ok:
        sys.exit("No successful requests to summarise.")

    sections = [
        [f"# Load test: {args.scenario}", "",
         f"_Generated by `generate_load_report.py` from `results/{args.scenario}.jsonl`._"],
        summary_section(records),
        by_category_section(ok),
        node_section(ok),
        llm_section(ok),
        histogram_section(ok),
        lending_section(ok),
        retrieval_section(),
    ]
    errors = [r for r in records if r["status"] != "ok"]
    if errors:
        sections.append(["## Errors", ""] + table(["ID", "Type", "Error"],
                                                  [[r["id"], r["category"], r.get("error", "")] for r in errors]))

    md = "\n\n".join("\n".join(section) for section in sections if section) + "\n"
    out = TEST_DIR / f"load_{args.scenario}.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {out.relative_to(TEST_DIR)}")


if __name__ == "__main__":
    main()
