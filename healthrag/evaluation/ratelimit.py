"""Waiting out provider rate limits during long evaluation runs.

Groq's free tier limits tokens per minute and per (rolling) day, and its 429 message says
when to retry ("Please try again in 5m58.56s"). Evaluation runs wait that long and carry
on, so a run that needs more than one day of quota finishes unattended.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")
_RETRY_IN = re.compile(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?")
DEFAULT_WAIT_SECONDS = 60.0
MARGIN_SECONDS = 5.0


def retry_after_seconds(error: Exception) -> float:
    """Seconds the provider asked us to wait, or a default when it didn't say."""
    match = _RETRY_IN.search(str(error))
    if not match or not any(match.groups()):
        return DEFAULT_WAIT_SECONDS
    hours, minutes, seconds = (float(g) if g else 0.0 for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds + MARGIN_SECONDS


def is_rate_limit(error: Exception) -> bool:
    """Whether an exception is a provider rate limit (HTTP 429)."""
    return "RateLimit" in type(error).__name__ or "429" in str(error) or "rate limit" in str(error).lower()


def call_with_patience(fn: Callable[[], T], what: str, max_wait_hours: float) -> T:
    """Call ``fn``, sleeping through rate limits until it succeeds or ``max_wait_hours`` have passed."""
    waited = 0.0
    while True:
        try:
            return fn()
        except Exception as error:  # noqa: BLE001 - only rate limits are retried
            if not is_rate_limit(error):
                raise
            wait = retry_after_seconds(error)
            if waited + wait > max_wait_hours * 3600:
                raise
            log.warning("%s: rate limited, waiting %.0f s", what, wait)
            time.sleep(wait)
            waited += wait
