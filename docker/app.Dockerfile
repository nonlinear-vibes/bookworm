# Python app image: LangGraph workflow, Streamlit UI, ingestion and DB scripts.
# The same image runs both the one-off init service and the UI service in
# compose.yaml. Runtime data (CSV, SQLite) lives in mounted volumes, not here.

FROM python:3.14-slim

# uv binary, pinned to the version used for local development.
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 1) Dependencies only. This layer is rebuilt only when the lockfile changes,
#    not on every source edit.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 2) Project source, installed in editable mode so config.PROJECT_ROOT
#    resolves to /app and relative data paths point at /app/data.
COPY README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY evals/ ./evals/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

EXPOSE 8501

CMD ["streamlit", "run", "src/bookworm/ui/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
