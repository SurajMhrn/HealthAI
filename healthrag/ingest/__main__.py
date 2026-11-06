"""Run the full ingest: ``python -m healthrag.ingest`` (or ``make ingest``)."""

from __future__ import annotations

import logging
import sys

from healthrag.config import get_settings
from healthrag.ingest.pipeline import run
from healthrag.redact.report import write_report


def main() -> int:
    """Build, chunk and redact all documents, then write chunks.jsonl and the redaction report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    result = run(settings)
    report = write_report(result, settings)
    logging.getLogger("ingest").info(
        "report: %s | leaks with deny-list: %d, without: %d",
        report,
        len(result.leaks.leaks),
        len(result.leaks_without_deny_list.leaks),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
