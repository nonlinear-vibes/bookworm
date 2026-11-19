"""Shared helpers for the evaluation scripts.

Call `setup_environment()` before importing any other bookworm module:
settings are read once, when bookworm.config is first imported, and the
other modules bind them at their own import time.
"""

import os
import re
import json
import shutil
import tempfile
from pathlib import Path
from difflib import SequenceMatcher

EVALS_DIR   = Path(__file__).resolve().parent
RESULTS_DIR = EVALS_DIR / "results"
LABELS_DIR  = EVALS_DIR / "labels"

# The RAG subgraph hands the top 5 fused results to the rest of the graph,
# so every retrieval metric is cut off at the same depth.
K = 5


def setup_environment(scratch_db: bool = False) -> None:
    """Make LLM output repeatable and, optionally, protect the demo database.

    - temperature 0 unless LLM_TEMPERATURE is already set (compose's eval
      service sets it explicitly),
    - with scratch_db, the SQLite database is copied to a temporary file and
      the run uses the copy, so lending cases never change the demo data.
    """
    os.environ.setdefault("LLM_TEMPERATURE", "0")

    from bookworm import config

    if scratch_db:
        scratch = Path(tempfile.mkdtemp(prefix="bookworm-eval-")) / "bookworm.db"
        shutil.copy(config.settings.sqlite_path, scratch)
        os.environ["SQLITE_PATH"] = str(scratch)

    # Rebuild the settings object before any other module imports it.
    config.settings = config.load_settings()


def load_json(name: str) -> dict:
    return json.loads((EVALS_DIR / name).read_text(encoding="utf-8"))


def match_owner(path: Path) -> None:
    """Give a written file the owner of evals/.

    The container runs as root, and evals/ is mounted from the host, so
    without this the host user could not edit labels or results.
    """
    owner = EVALS_DIR.stat()
    if (path.stat().st_uid, path.stat().st_gid) != (owner.st_uid, owner.st_gid):
        try:
            os.chown(path, owner.st_uid, owner.st_gid)
        except PermissionError:
            pass  # running as a normal user: the file is already ours


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(exist_ok=True)
    match_owner(path.parent)
    path.write_text(text, encoding="utf-8")
    match_owner(path)
    return path


def write_results(name: str, data: dict) -> Path:
    return write_text(RESULTS_DIR / name, json.dumps(data, indent=2, ensure_ascii=False))


def markdown_table(headers: list[str], rows: list[list]) -> str:
    def fmt(value) -> str:
        if isinstance(value, float):
            return f"{value:.2f}"
        return "" if value is None else str(value)

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


# -----------------------------------
#       Title matching (groundedness)
# -----------------------------------
def normalize_title(text: str) -> str:
    """Lowercase, drop subtitles/series notes and punctuation."""
    text = re.split(r"[:(\[]", text, maxsplit=1)[0]
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(text.split())


def titles_match(a: str, b: str) -> bool:
    a, b = normalize_title(a), normalize_title(b)
    if not a or not b:
        return False
    if a == b or (min(len(a), len(b)) >= 8 and (a in b or b in a)):
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.85


# Quoted or emphasised spans, and "1. Title by Author" list items.
_TITLE_PATTERNS = [
    r"\"([^\"\n]{3,120})\"",
    r"“([^”\n]{3,120})”",
    # Single quotes; inner apostrophes ("Frommer's") are allowed.
    r"(?<!\w)'([^\n]{3,120}?)'(?!\w)",
    r"\*\*([^*\n]{3,120})\*\*",
    r"(?m)^\s*\d+[.)]\s+\**\"?([^\"*\n]{3,120}?)\"?\**\s+by\s",
]


def extract_title_mentions(answer: str) -> list[str]:
    """Book titles the answer appears to mention (heuristic)."""
    found: list[str] = []
    for pattern in _TITLE_PATTERNS:
        for match in re.findall(pattern, answer):
            candidate = match.strip(" .,;:'\"*")
            if len(candidate) >= 3 and not any(titles_match(candidate, f) for f in found):
                found.append(candidate)
    return found
