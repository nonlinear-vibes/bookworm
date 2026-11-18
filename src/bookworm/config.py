"""Central configuration, read from environment variables.

Locally the values come from the project's .env file. In Docker the same
variables are set by compose.yaml, so only the hostnames change
(for example OLLAMA_BASE_URL=http://ollama:11434).
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _path(value: str) -> Path:
    """Resolve relative paths against the project root."""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    # Chat model
    llm_provider: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int

    # Ollama (serves both the chat and the embedding model)
    ollama_base_url: str
    embedding_model: str
    embedding_dimensions: int

    # Qdrant
    qdrant_url: str
    qdrant_collection: str

    # Data
    books_csv_path: Path
    sqlite_path: Path


def load_settings() -> Settings:
    return Settings(
        llm_provider         = os.getenv("LLM_PROVIDER", "ollama"),
        llm_model            = os.getenv("LLM_MODEL", "mistral"),
        llm_temperature      = float(os.getenv("LLM_TEMPERATURE", "0.2")),
        llm_max_tokens       = int(os.getenv("LLM_MAX_TOKENS", "512")),

        ollama_base_url      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        embedding_model      = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
        embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1024")),

        qdrant_url           = os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection    = os.getenv("QDRANT_COLLECTION", "books"),

        books_csv_path       = _path(os.getenv("BOOKS_CSV_PATH", "data/raw/books.csv")),
        sqlite_path          = _path(os.getenv("SQLITE_PATH", "data/sqlite/bookworm.db")),
    )


settings = load_settings()
