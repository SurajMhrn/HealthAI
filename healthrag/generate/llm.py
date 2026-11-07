"""One OpenAI-SDK client for every tier, with an on-disk response cache.

Each tier is an OpenAI-compatible endpoint (``*_BASE_URL`` / ``*_API_KEY`` in .env).
``LARGE_PROVIDER=azure`` switches the large tier to Azure OpenAI.
Responses are cached by a hash of the full request (endpoint, model, messages and
options), so re-running an evaluation doesn't spend quota again.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

from healthrag.config import Settings

log = logging.getLogger(__name__)

Tier = Literal["small", "large", "judge"]


@dataclass(frozen=True)
class LlmResponse:
    """A completion plus what it cost to get it."""

    text: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cached: bool = False


class ChatClient(Protocol):
    """What the rest of the code needs from an LLM client (tests use a fake)."""

    def chat(self, tier: Tier, messages: list[dict[str, str]], json_mode: bool = False) -> LlmResponse:
        """Send messages to the model of ``tier`` and return its reply."""
        ...

    def model_name(self, tier: Tier) -> str:
        """The configured model for a tier."""
        ...


class DiskCache:
    """JSON files under ``cache_dir``, one per request hash."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    @staticmethod
    def key(request: dict) -> str:
        """Stable hash of a request."""
        return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        """Cached response for ``key``, or None."""
        path = self.cache_dir / key[:2] / f"{key}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def put(self, key: str, value: dict) -> None:
        """Store a response."""
        path = self.cache_dir / key[:2] / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class LlmClient:
    """Chat completions for the small, large and judge tiers."""

    def __init__(self, settings: Settings) -> None:
        from openai import AzureOpenAI, OpenAI  # imported here so offline code paths don't need it

        c = settings.llm
        self.settings = settings
        self.cache = DiskCache(c.cache_dir)
        self._models: dict[str, str] = {"small": c.small_model, "large": c.large_model, "judge": c.judge_model}
        self._endpoints: dict[str, str] = {}
        self._clients: dict[str, OpenAI] = {}
        common = {"timeout": c.timeout_seconds, "max_retries": c.max_retries}
        for tier, url, key in (
            ("small", settings.small_base_url, settings.small_api_key),
            ("large", settings.large_base_url, settings.large_api_key),
            ("judge", settings.judge_base_url, settings.judge_api_key),
        ):
            if tier == "large" and settings.large_provider == "azure":
                if not (settings.azure_openai_endpoint and settings.azure_openai_api_key):
                    continue
                self._clients[tier] = AzureOpenAI(
                    azure_endpoint=settings.azure_openai_endpoint,
                    api_key=settings.azure_openai_api_key.get_secret_value(),
                    api_version=settings.azure_openai_api_version,
                    **common,
                )
                self._models[tier] = settings.azure_openai_deployment or self._models[tier]
                self._endpoints[tier] = settings.azure_openai_endpoint
            elif url and key:
                self._clients[tier] = OpenAI(base_url=url, api_key=key.get_secret_value(), **common)
                self._endpoints[tier] = url

    def model_name(self, tier: Tier) -> str:
        """The configured model for a tier."""
        return self._models[tier]

    def chat(self, tier: Tier, messages: list[dict[str, str]], json_mode: bool = False) -> LlmResponse:
        """Send a chat request (or return the cached reply for an identical request)."""
        if tier not in self._clients:
            raise RuntimeError(f"No endpoint configured for the {tier} tier. Set {tier.upper()}_BASE_URL/_API_KEY.")
        c = self.settings.llm
        model = self._models[tier]
        options: dict = {"temperature": c.temperature, "max_tokens": c.max_tokens, **c.extra_params.get(model, {})}
        if json_mode:
            options["response_format"] = {"type": "json_object"}
        request = {"endpoint": self._endpoints[tier], "model": model, "messages": messages, "options": options}
        key = self.cache.key(request)
        hit = self.cache.get(key)
        if hit is not None:
            return LlmResponse(**{**hit, "cached": True})

        extra = {k: v for k, v in options.items() if k not in ("temperature", "max_tokens", "response_format")}
        started = time.perf_counter()
        completion = self._clients[tier].chat.completions.create(
            model=model,
            messages=messages,
            temperature=options["temperature"],
            max_tokens=options["max_tokens"],
            response_format=options.get("response_format"),
            extra_body=extra or None,
        )
        latency = (time.perf_counter() - started) * 1000
        usage = completion.usage
        response = LlmResponse(
            text=(completion.choices[0].message.content or "").strip(),
            model=model,
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
            latency_ms=latency,
        )
        self.cache.put(key, {k: v for k, v in asdict(response).items() if k != "cached"})
        return response


def estimated_cost(settings: Settings, model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimated USD cost at the configured reference price (0.0 if the model has no price)."""
    price = settings.llm.prices.get(model)
    if price is None:
        return 0.0
    return (tokens_in * price.input + tokens_out * price.output) / 1_000_000
