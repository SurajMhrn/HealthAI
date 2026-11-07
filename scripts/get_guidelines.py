"""Download the reviewed guideline PDFs listed in config.yaml.

Run with ``python -m scripts.get_guidelines`` (or ``make data``). Files go to
``data/raw/guidelines/<name>.pdf``. When a source has a pinned ``sha256`` in
config.yaml, the download must match it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

from healthrag.config import get_settings
from scripts.download import download, sha256_of

log = logging.getLogger("get_guidelines")


def main(argv: list[str] | None = None) -> int:
    """Download every configured guideline that isn't already present and write a manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="download again even if the file exists")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    out_dir = settings.paths.raw_guidelines
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for source in settings.guidelines.sources:
        dest = out_dir / f"{source.name}.pdf"
        if dest.exists() and not args.force:
            digest = sha256_of(dest)
            if source.sha256 and digest != source.sha256:
                log.error("%s does not match the pinned sha256; rerun with --force", dest)
                return 1
            log.info("%s already present", dest.name)
        else:
            digest = download(
                source.url,
                dest,
                settings.downloads.user_agent,
                settings.downloads.timeout_seconds,
                expected_sha256=source.sha256,
                attempts=settings.downloads.attempts,
            )
        with dest.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                log.error("%s is not a PDF (the server may have returned an error page)", dest)
                return 1
        manifest.append(
            {
                "name": source.name,
                "url": source.url,
                "license": source.license,
                "sha256": digest,
                "bytes": dest.stat().st_size,
                "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        log.info("%s: %d bytes, sha256 %s", dest.name, dest.stat().st_size, digest)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
