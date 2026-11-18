# Stub: Python app image (graph, ingestion scripts, SQLite access).
# TODO: finish and test once everything works locally.

FROM python:3.14-slim

# TODO: install uv, copy pyproject.toml + uv.lock, run `uv sync --frozen`,
#       copy src/ and scripts/, and set the command that starts the app.
# The SQLite database lives in the mounted /app/data volume, not the image.

WORKDIR /app
