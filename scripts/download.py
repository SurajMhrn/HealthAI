"""Small download helper shared by the data scripts."""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)


def sha256_of(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(
    url: str,
    dest: Path,
    user_agent: str,
    timeout: int,
    expected_sha256: str | None = None,
    attempts: int = 1,
) -> str:
    """Download ``url`` to ``dest`` and return the file's SHA-256.

    Connection errors are retried up to ``attempts`` times in total. If Python's own
    HTTPS client keeps failing and ``curl`` is installed, curl is tried once (on some
    Windows networks only the system TLS stack that curl uses gets through). The file is
    written to a temporary name first and only moved into place once the checksum
    (if given) matches, so a failed or tampered download never leaves a
    usable-looking file behind.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    for attempt in range(1, attempts + 1):
        log.info("Downloading %s (attempt %d/%d)", url, attempt, attempts)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out, length=1 << 20)
            break
        except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code < 500:
                raise
            if attempt == attempts:
                if shutil.which("curl") is None:
                    raise
                log.warning("Download failed (%s); falling back to curl", error)
                _curl(url, partial, user_agent, timeout, attempts)
                break
            log.warning("Download failed (%s); retrying in %d s", error, 5 * attempt)
            time.sleep(5 * attempt)
    actual = sha256_of(partial)
    if expected_sha256 and actual != expected_sha256.lower():
        partial.unlink()
        raise RuntimeError(f"Checksum mismatch for {url}: expected {expected_sha256}, got {actual}")
    partial.replace(dest)
    return actual


def _curl(url: str, dest: Path, user_agent: str, timeout: int, attempts: int) -> None:
    """Download with the curl command-line tool."""
    command = [
        "curl", "--fail", "--location", "--silent", "--show-error",
        "--retry", str(attempts), "--connect-timeout", str(timeout),
        "--user-agent", user_agent, "--output", str(dest), url,
    ]  # fmt: skip
    subprocess.run(command, check=True)
