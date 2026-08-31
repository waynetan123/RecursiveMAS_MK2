"""Async OpenRouter client for screening generation.

Three things here are load-bearing for the validity of the screen:

1. The provider is pinned and asserted constant. OpenRouter otherwise routes
   across providers serving different quantizations, and a score averaged over
   mixed weights is attributable to nothing.
2. Reasoning is disabled with {"enabled": False}, never {"exclude": True} — the
   latter still runs and still bills the reasoning, so the thinking-OFF arm would
   silently be a thinking-ON arm.
3. Every response is cached on disk, so a interrupted sweep resumes instead of
   being paid for twice.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from screening.config import TOP_P, ModelSpec

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Status codes worth retrying: rate limits and transient upstream failures.
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
MAX_RETRIES = 5
REQUEST_TIMEOUT_S = 900.0  # AIME at 16k tokens is genuinely slow

# Markers a model might emit if reasoning leaked into the visible content.
THINKING_TAGS = ("<think>", "<thinking>", "<reasoning>")


# One completion, plus the diagnostics the screen depends on.
@dataclass(frozen=True)
class Response:
    text: str
    finish_reason: str
    provider: str
    model: str = ""  # what OpenRouter says it served, not what we asked for
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float = 0.0
    had_reasoning_field: bool = False
    from_cache: bool = False


# Raised when a run's responses contradict the thinking mode that was requested.
class ThinkingModeViolation(RuntimeError):
    pass


# Hash a request into a stable cache filename.
def _cache_key(payload: dict[str, Any], rollout_idx: int) -> str:
    # sort_keys makes the digest independent of dict ordering across runs.
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(f"{canonical}::{rollout_idx}".encode()).hexdigest()


# Pull the reasoning-token count out of OpenRouter's usage block.
def _reasoning_tokens(usage: dict[str, Any]) -> int:
    details = usage.get("completion_tokens_details") or {}
    return int(details.get("reasoning_tokens") or 0)


# Async OpenRouter caller with bounded concurrency and an on-disk response cache.
class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        cache_dir: Path,
        concurrency: int = 8,
        use_cache: bool = True,
        require_cache: bool = False,
    ) -> None:
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        # --rescore-only sets this: a cache miss is an error, never an API call.
        self.require_cache = require_cache
        self._semaphore = asyncio.Semaphore(concurrency)
        self._client: httpx.AsyncClient | None = None

        # Providers actually seen per model. Tracked per model, not globally,
        # because each model is pinned to a different provider by design.
        self.observed_providers: dict[str, set[str]] = defaultdict(set)

        # Models OpenRouter reports serving, per model we requested. A mismatch
        # means a slug was aliased or redirected and the run is not what it claims.
        self.observed_models: dict[str, set[str]] = defaultdict(set)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def __aenter__(self) -> OpenRouterClient:
        self._client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._client is not None:
            await self._client.aclose()

    # Build the request body for one completion.
    def _build_payload(
        self,
        model: ModelSpec,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: int,
        thinking: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "model": model.model_id,
            "messages": messages,
            "temperature": temperature,
            "top_p": TOP_P,
            "max_tokens": max_tokens,
            "seed": seed,
            "provider": model.provider,
            "reasoning": thinking,
            "usage": {"include": True},
        }

    # Read a cached response, returning None on a miss.
    def _read_cache(self, key: str) -> Response | None:
        if not self.use_cache:
            return None

        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None

        # A truncated cache file from an interrupted write is treated as a miss.
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return Response(**{**data, "from_cache": True})

    # Persist a response for resumability.
    def _write_cache(self, key: str, response: Response) -> None:
        if not self.use_cache:
            return

        payload = {k: v for k, v in response.__dict__.items() if k != "from_cache"}
        path = self.cache_dir / f"{key}.json"

        # Write via a temp file so an interrupt cannot leave a half-written entry.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    # Issue one request, retrying transient failures with exponential backoff.
    async def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self._client is not None, "use OpenRouterClient as an async context manager"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                reply = await self._client.post(OPENROUTER_URL, json=payload, headers=headers)

                # Non-retryable 4xx means the request itself is wrong: fail loud.
                if reply.status_code not in RETRYABLE_STATUS and reply.status_code >= 400:
                    raise RuntimeError(
                        f"OpenRouter returned HTTP {reply.status_code}: {reply.text[:500]}"
                    )
                if reply.status_code in RETRYABLE_STATUS:
                    raise httpx.HTTPStatusError(
                        f"retryable HTTP {reply.status_code}",
                        request=reply.request,
                        response=reply,
                    )
                return reply.json()

            except (httpx.HTTPStatusError, httpx.TransportError, json.JSONDecodeError) as exc:
                last_error = exc
                await asyncio.sleep(2.0 * (attempt + 1))

        raise RuntimeError(f"OpenRouter request failed after {MAX_RETRIES} retries: {last_error}")

    # Generate one completion, serving from cache when possible.
    async def complete(
        self,
        model: ModelSpec,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: int,
        thinking: dict[str, Any],
        rollout_idx: int = 0,
    ) -> Response:
        payload = self._build_payload(model, messages, temperature, max_tokens, seed, thinking)
        key = _cache_key(payload, rollout_idx)

        cached = self._read_cache(key)
        if cached is not None:
            self._record_routing(model, cached)
            return cached

        # In --rescore-only mode a miss means the generation step was incomplete.
        if self.require_cache:
            raise RuntimeError(
                f"Cache miss for {model.model_id} in --rescore-only mode (key {key[:12]}). "
                "Run the generation sweep first."
            )

        # Bound concurrency around the network call only, not the cache lookup.
        async with self._semaphore:
            body = await self._post_with_retries(payload)

        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = body.get("usage") or {}

        response = Response(
            text=str(message.get("content") or ""),
            finish_reason=str(choice.get("finish_reason") or ""),
            provider=str(body.get("provider") or "unknown"),
            model=str(body.get("model") or ""),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            reasoning_tokens=_reasoning_tokens(usage),
            cost=float(usage.get("cost") or 0.0),
            had_reasoning_field=bool(message.get("reasoning")),
        )

        self._record_routing(model, response)
        self._write_cache(key, response)
        return response

    # Note which provider and which served model this response actually came from.
    def _record_routing(self, model: ModelSpec, response: Response) -> None:
        self.observed_providers[model.model_id].add(response.provider)

        # Older cache entries predate the model field; skip rather than flag them.
        if response.model:
            self.observed_models[model.model_id].add(response.model)

    # Fail if a model's responses did not all come from the model and provider asked for.
    def assert_routing(self, model: ModelSpec) -> None:
        # A served id may carry a routing suffix (":nitro", ":floor"); the base slug
        # is what has to match, since that is what identifies the weights.
        served = {name.split(":", 1)[0] for name in self.observed_models.get(model.model_id, set())}
        unexpected = served - {model.model_id}
        if unexpected:
            raise RuntimeError(
                f"Requested {model.model_id} but OpenRouter served {sorted(unexpected)}. "
                "The slug has been aliased or redirected; results do not describe the "
                "model named in the manifest."
            )

        providers = self.observed_providers.get(model.model_id, set())
        if len(providers) > 1:
            raise RuntimeError(
                f"Provider changed mid-run for {model.model_id}, so its scores mix "
                f"different weights: {sorted(providers)}. "
                "Check allow_fallbacks is False in the model's provider pin."
            )


# Verify a sample of responses really had reasoning disabled.
def assert_thinking_disabled(responses: list[Response], benchmark: str) -> None:
    violations: list[str] = []

    for idx, response in enumerate(responses):
        # A populated reasoning field means the model reasoned despite enabled=False.
        if response.had_reasoning_field:
            violations.append(f"response {idx}: message.reasoning was populated")

        # Billed reasoning tokens are the unambiguous signal, and the costly one.
        if response.reasoning_tokens > 0:
            violations.append(
                f"response {idx}: {response.reasoning_tokens} reasoning tokens billed"
            )

        # Some providers emit thinking inline rather than in a separate field.
        lowered = response.text.lower()
        for tag in THINKING_TAGS:
            if tag in lowered:
                violations.append(f"response {idx}: found {tag} in content")
                break

    if violations:
        raise ThinkingModeViolation(
            f"Thinking was requested OFF for '{benchmark}' but the provider ignored it. "
            "The gate column would be silently thinking-ON and every exclusion decision "
            "could invert. Violations:\n  " + "\n  ".join(violations[:10])
        )
