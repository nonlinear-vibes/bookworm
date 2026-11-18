# Bookworm

An AI-powered library assistant built with LangGraph, Ollama, Qdrant, and SQLite.

## Local setup

Requirements: [uv](https://docs.astral.sh/uv/), Ollama, and Qdrant running locally.

```bash
cp .env.example .env                      # then adjust if needed
ollama pull mistral
ollama pull qwen3-embedding:0.6b

uv sync
uv run python scripts/ingest_books.py --limit 2000 --recreate   # CSV -> embeddings -> Qdrant
uv run python scripts/init_db.py --limit 2000 --reset           # SQLite lending database
```

The raw dataset is expected at `data/raw/books.csv`. Use the same `--limit`
for both scripts so the SQLite books match the Qdrant collection.

## Web UI

```bash
uv run streamlit run src/bookworm/ui/app.py
```

Each browser session is a new demo user (`UR-XXXX`) with 2-3 random books on loan.

## Manual evals

The scripts in `evals/` call the real models and print their results for
manual inspection:

```bash
uv run python evals/eval_query_processor.py   # Query Processor + RAG subgraph
uv run python evals/eval_e2e.py               # full graph, all routes
```

Lending evals change the database; run `scripts/init_db.py --reset` first.

## Layout

```
src/bookworm/
├── config.py      all settings, read from environment variables
├── llm/           chat model factory
├── graph/         LangGraph workflows, nodes, prompts, schemas, state
├── retrieval/     embeddings, Qdrant access, search
├── ingestion/     CSV loading and the ingestion pipeline
├── lending/       SQLite schema and lending operations
├── tools/         LangChain tool wrappers used by the graph
└── ui/            Streamlit web UI
scripts/           command-line entry points (ingestion, database setup)
evals/             manual LLM scenarios
docker/            Docker files (stubs)
```
