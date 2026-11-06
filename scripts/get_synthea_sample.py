"""Download Synthea's pre-generated sample population (CSV) as a second patient dataset.

Run with ``python -m scripts.get_synthea_sample`` (or ``make data``). Only the CSV files
that ingest reads are extracted, into ``data/raw/synthea_sample/csv``. The archive must
match the sha256 pinned in config.yaml.
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import PurePosixPath

from healthrag.config import get_settings
from scripts.download import download

log = logging.getLogger("get_synthea_sample")

NEEDED = ("patients.csv", "conditions.csv", "medications.csv", "encounters.csv")


def main(argv: list[str] | None = None) -> int:
    """Download the archive (unless present) and extract the CSVs ingest needs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="download and extract again")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    sample = settings.synthea_sample
    root = settings.paths.raw_synthea_sample
    csv_dir = root / "csv"
    if (csv_dir / "patients.csv").exists() and not args.force:
        log.info("%s already exists; skipping (use --force to redo)", csv_dir / "patients.csv")
        return 0

    archive = root / "synthea_sample_data_csv.zip"
    download(
        sample.url,
        archive,
        settings.downloads.user_agent,
        settings.downloads.timeout_seconds,
        expected_sha256=sample.sha256,
        attempts=settings.downloads.attempts,
    )
    csv_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = {PurePosixPath(info.filename).name: info for info in bundle.infolist() if not info.is_dir()}
        for name in NEEDED:
            if name not in members:
                log.error("%s is missing from the sample archive", name)
                return 1
            # Write by base name only, so paths inside the archive can't escape csv_dir.
            (csv_dir / name).write_bytes(bundle.read(members[name]))
    log.info("Extracted %s to %s (%s)", ", ".join(NEEDED), csv_dir, sample.description)
    return 0


if __name__ == "__main__":
    sys.exit(main())
