"""NeMo Guardrails wrapper: input rails before retrieval, output rails after generation.

Input rails (in order): prompt-injection patterns, identity-reveal patterns, then an LLM
self-check on the small model covering individual medical advice, off-topic questions,
injection and identity reveal. Output rail: Presidio on the answer; any identifier blocks it.

The rails config lives in ``healthrag/guard/rails/``. The model and its endpoint are added
at runtime from config.yaml and .env, so no key is ever written to the rails config.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from healthrag.config import Settings
from healthrag.guard.patterns import identity_matches, injection_matches
from healthrag.redact.engine import Redactor

RAILS_DIR = Path(__file__).parent / "rails"


@dataclass(frozen=True)
class GuardResult:
    """Outcome of one rails check."""

    blocked: bool
    rail: str | None
    message: str
    latency_ms: float


class Guard:
    """Runs the configured input and output rails through NeMo Guardrails."""

    def __init__(self, settings: Settings, redactor: Redactor) -> None:
        from nemoguardrails import LLMRails, RailsConfig
        from nemoguardrails.rails.llm.config import Model

        if not (settings.small_base_url and settings.small_api_key):
            raise RuntimeError("Guardrails need the small tier: set SMALL_BASE_URL and SMALL_API_KEY in .env")
        llm = settings.llm
        config = RailsConfig.from_path(str(RAILS_DIR))
        config.models = [
            Model(
                type="main",
                engine="openai",
                model=llm.small_model,
                parameters={
                    "base_url": settings.small_base_url,
                    "api_key": settings.small_api_key.get_secret_value(),
                    "temperature": llm.temperature,
                    "max_tokens": llm.max_tokens,
                    **llm.extra_params.get(llm.small_model, {}),
                },
            )
        ]
        self.redactor = redactor
        self.rails = LLMRails(config)
        self.rails.register_action(self._check_injection, name="check_injection_patterns")
        self.rails.register_action(self._check_identity, name="check_identity_reveal")
        self.rails.register_action(self._check_output_identifiers, name="check_output_identifiers")

    @staticmethod
    async def _check_injection(text: str = "") -> bool:
        return bool(injection_matches(text or ""))

    @staticmethod
    async def _check_identity(text: str = "") -> bool:
        return bool(identity_matches(text or ""))

    async def _check_output_identifiers(self, text: str = "") -> bool:
        return bool(self.redactor.redact(text or "").detections)

    def _run(self, messages: list[dict[str, str]], rail_type: str) -> GuardResult:
        from nemoguardrails.rails.llm.options import RailStatus, RailType

        started = time.perf_counter()
        result = self.rails.check(messages, rail_types=[RailType(rail_type)])
        latency = (time.perf_counter() - started) * 1000
        blocked = result.status == RailStatus.BLOCKED
        return GuardResult(blocked, result.rail if blocked else None, result.content, latency)

    def check_input(self, question: str) -> GuardResult:
        """Run the input rails on a (redacted) user question."""
        return self._run([{"role": "user", "content": question}], "input")

    def check_output(self, question: str, answer: str) -> GuardResult:
        """Run the output rails on a generated answer."""
        return self._run([{"role": "user", "content": question}, {"role": "assistant", "content": answer}], "output")
