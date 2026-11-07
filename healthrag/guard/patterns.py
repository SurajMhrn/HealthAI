"""Pattern checks that need no model: prompt injection and attempts to re-identify patients.

The same injection patterns flag suspicious chunks at ingestion, and the pipeline drops
flagged chunks from the context before generation.
"""

from __future__ import annotations

import re

INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "ignore_instructions": re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}\b(?:previous|prior|above|earlier|all|your|the|system)\b"
        r"[^.\n]{0,20}\b(?:instructions?|rules?|prompts?|guidelines?|polic(?:y|ies)|guardrails?)\b",
        re.IGNORECASE,
    ),
    "reveal_prompt": re.compile(
        r"\b(?:reveal|show|print|repeat|output|leak|tell me)\b[^.\n]{0,30}\b(?:system|hidden|initial|original|your)\s+"
        r"(?:prompt|instructions|rules)\b",
        re.IGNORECASE,
    ),
    "role_change": re.compile(
        r"\b(?:you are now|from now on,? you|act as|pretend (?:to be|you are)|roleplay as|developer mode|jailbreak|"
        r"do anything now|no restrictions)\b",
        re.IGNORECASE,
    ),
    "fake_turn_markers": re.compile(
        r"(?:</?(?:system|assistant|instructions?)>|\[/?INST\]|<\|im_(?:start|end)\|>|^\s*(?:system|assistant)\s*:)",
        re.IGNORECASE | re.MULTILINE,
    ),
    "new_instructions": re.compile(r"\b(?:new|updated|additional|important)\s+(?:instructions?|rules?)\s*:", re.IGNORECASE),
}

# Identifiers other than names. A bare "name" is too common ("the name of the drug"), so names only
# count when qualified ("full name") or possessive ("the patient's name").
_IDENTIFIER_CORE = (
    r"(?:surname|identity|ssn|social security(?: number)?|(?:home\s+)?address|phone(?: number)?|date of birth|"
    r"dob|birthday|passport(?: number)?|driver'?s licen[cs]e|licen[cs]e number|record id|mrn|"
    r"medical record number|uuid|email)"
)
_QUALIFIED_NAME = r"(?:real|full|actual|true|first|last|legal|patient)\s+name"
_IDENTIFIER_WORDS = rf"(?:{_IDENTIFIER_CORE}|{_QUALIFIED_NAME})"
_POSSESSIVE_IDENTIFIER = rf"(?:{_IDENTIFIER_CORE}|{_QUALIFIED_NAME}|name)"
_PLACEHOLDER = r"<(?:PERSON|DATE_OF_BIRTH|ADDRESS|PHONE_NUMBER|RECORD_ID)>"

IDENTITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "unredact": re.compile(
        r"\b(?:un-?redact|de-?anonymi[sz]e|re-?identify|unmask)\w*", re.IGNORECASE
    ),
    "placeholder_lookup": re.compile(
        rf"(?:who|what|which)\b[^.?\n]{{0,40}}{_PLACEHOLDER}"
        rf"|{_PLACEHOLDER}[^.?\n]{{0,30}}\b(?:stands? for|really|actually|hid(?:e|den))",
        re.IGNORECASE,
    ),
    "ask_identifier": re.compile(
        rf"\b(?:what(?:'s| is| are)|tell me|give me|show me|find|look up|reveal|list|share)\b[^.?\n]{{0,40}}"
        rf"\b{_IDENTIFIER_WORDS}\b[^.?\n]{{0,30}}\b(?:patient|patients|person|this|her|his|their|them|<PERSON>)",
        re.IGNORECASE,
    ),
    "possessive_identifier": re.compile(
        rf"\b(?:patient'?s|person'?s|her|his|their|<PERSON>'?s?)\s+{_POSSESSIVE_IDENTIFIER}\b", re.IGNORECASE
    ),
}


def matched(patterns: dict[str, re.Pattern[str]], text: str) -> list[str]:
    """Names of the patterns that match ``text``."""
    return [name for name, pattern in patterns.items() if pattern.search(text)]


def injection_matches(text: str) -> list[str]:
    """Prompt-injection patterns found in a user message or a retrieved chunk."""
    return matched(INJECTION_PATTERNS, text)


def identity_matches(text: str) -> list[str]:
    """Patterns suggesting an attempt to learn who a patient is."""
    return matched(IDENTITY_PATTERNS, text)
