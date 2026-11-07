"""Download the pinned Synthea release and generate the synthetic patient population.

Run with ``python -m scripts.get_synthea`` (or ``make data``). Needs Java 17 or newer.
The output is CSV only, written to ``data/raw/synthea/csv``.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

from healthrag.config import Settings, get_settings
from scripts.download import download, sha256_of

log = logging.getLogger("get_synthea")


def java_major_version() -> int | None:
    """Major version of the ``java`` on PATH, or None if Java isn't installed."""
    if shutil.which("java") is None:
        return None
    result = subprocess.run(["java", "-version"], capture_output=True, text=True, check=False)
    match = re.search(r'version "(\d+)(?:\.(\d+))?', result.stderr + result.stdout)
    if not match:
        return None
    major = int(match.group(1))
    return int(match.group(2) or 0) if major == 1 else major  # "1.8" means Java 8


def ensure_jar(settings: Settings) -> Path:
    """Download the pinned Synthea jar into the vendor folder unless it is already there."""
    synthea = settings.synthea
    jar = settings.paths.vendor / "synthea" / f"synthea-with-dependencies-{synthea.version}.jar"
    if jar.exists() and sha256_of(jar) == synthea.jar_sha256:
        log.info("Using cached %s", jar)
        return jar
    download(
        synthea.jar_url,
        jar,
        settings.downloads.user_agent,
        settings.downloads.timeout_seconds,
        expected_sha256=synthea.jar_sha256,
        attempts=settings.downloads.attempts,
    )
    return jar


def synthea_command(jar: Path, settings: Settings, output_dir: Path) -> list[str]:
    """Command line for a reproducible, CSV-only Synthea run."""
    s = settings.synthea
    return [
        "java",
        "-jar",
        str(jar),
        "-s", str(s.seed),
        "-cs", str(s.clinician_seed),
        "-r", s.reference_date,
        "-p", str(s.population),
        f"--exporter.baseDirectory={output_dir.as_posix()}",
        "--exporter.csv.export=true",
        "--exporter.fhir.export=false",
        "--exporter.fhir_stu3.export=false",
        "--exporter.fhir_dstu2.export=false",
        "--exporter.ccda.export=false",
        "--exporter.hospital.fhir.export=false",
        "--exporter.practitioner.fhir.export=false",
        "--exporter.metadata.export=false",
        s.state,
    ]  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    """Generate the patients, unless the CSV output already exists (use --force to redo)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="delete existing output and generate again")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()

    output_dir = settings.paths.raw_synthea
    patients_csv = output_dir / "csv" / "patients.csv"
    if patients_csv.exists() and not args.force:
        log.info("%s already exists; skipping (use --force to regenerate)", patients_csv)
        return 0

    java = java_major_version()
    if java is None or java < settings.synthea.min_java_version:
        found = "no Java on PATH" if java is None else f"Java {java}"
        log.error(
            "Synthea needs Java %d or newer, found %s. Install a JDK (e.g. Eclipse Temurin %d) and retry.",
            settings.synthea.min_java_version,
            found,
            settings.synthea.min_java_version,
        )
        return 1

    jar = ensure_jar(settings)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    command = synthea_command(jar, settings, output_dir)
    log.info("Running Synthea %s (Java %d): %s", settings.synthea.version, java, " ".join(command))
    # Run from the vendor folder so Synthea's own log files don't land in the project root.
    subprocess.run(command, cwd=jar.parent, check=True)
    if not patients_csv.exists():
        log.error("Synthea finished but %s is missing", patients_csv)
        return 1
    log.info("Synthea CSVs written to %s", patients_csv.parent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
