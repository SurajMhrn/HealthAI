"""Per-query telemetry, appended to logs/queries.jsonl. Queries are redacted before they get here."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def log_query(path: Path, record: dict) -> None:
    """Append one telemetry record (a timestamp is added)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **record}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
