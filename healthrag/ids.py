"""Deterministic identifiers for patients and chunks."""

from __future__ import annotations

import hashlib
import hmac


def patient_id(synthea_uuid: str, salt: str, length: int) -> str:
    """Pseudonymous patient ID: HMAC-SHA256 of the Synthea UUID keyed with the salt, truncated.

    The same UUID and salt always give the same ID. Without the salt the ID can't be
    linked back to the UUID.
    """
    if not salt:
        raise ValueError("salt must not be empty")
    if not 8 <= length <= 64:
        raise ValueError("length must be between 8 and 64 hex characters")
    digest = hmac.new(salt.encode("utf-8"), synthea_uuid.strip().lower().encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()[:length]


def chunk_id(source: str, ordinal: int, section: str, redacted_text: str) -> str:
    """Deterministic chunk ID from the chunk's position and its *redacted* text.

    Hashing redacted text, never raw text, means the ID carries no information
    about the identifiers that were removed.
    """
    payload = "\x1f".join([source, str(ordinal), section, redacted_text])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
