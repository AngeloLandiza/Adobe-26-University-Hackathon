"""Hosted LLMs for node 1: Claude and Gemini behind one `complete()` call.

They only work with an API key and network access. Stage-one scoring runs
with neither, so there node 1 falls back to the heuristic ranker. Clients
are created on first use, so building a provider never needs a key.
"""

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import anthropic
from google import genai
from google.genai import types

from engine.pipeline.priority.pricing import cost_usd
from engine.pipeline.types import LLMCallStats

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
CLAUDE_MAX_TOKENS = 4096  # room for (always-on) thinking plus the JSON
# Server-side refusal fallback: if Claude declines, Anthropic re-runs the request on its recommended model.
CLAUDE_BETAS = ["server-side-fallback-2026-07-01"]


class ProviderError(Exception):
    """The provider answered, but not with a usable plan (refusal, cut off, not JSON)."""


@dataclass
class LLMResult:
    data: dict
    stats: LLMCallStats


def _parse_json(text: str | None, provider: str) -> dict:
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProviderError(f"{provider}: response is not JSON") from exc
    if not isinstance(data, dict):
        raise ProviderError(f"{provider}: response is not a JSON object")
    return data


class ClaudeProvider:
    name = "claude"

    def __init__(self, model: str | None = None, timeout_s: float = 60.0):
        self.model = model or os.environ.get("CHAMELEON_CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL
        self.timeout_s = timeout_s
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = anthropic.Anthropic(timeout=self.timeout_s, max_retries=1)
        return self._client

    def complete(self, system: str, user: str, schema: dict) -> LLMResult:
        start = time.perf_counter()
        # No temperature: Opus 5.5 rejects it. Low effort is enough for labelling.
        response = self._get_client().beta.messages.create(
            model=self.model,
            max_tokens=CLAUDE_MAX_TOKENS,
            betas=CLAUDE_BETAS,
            fallbacks="default",
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
        )
        latency = time.perf_counter() - start
        if response.stop_reason in ("refusal", "max_tokens"):
            raise ProviderError(f"claude stopped with {response.stop_reason}")
        text = next((block.text for block in response.content if block.type == "text"), None)
        data = _parse_json(text, "claude")
        usage = response.usage
        return LLMResult(data, LLMCallStats(
            provider=self.name, model=self.model, latency_s=latency,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            cost_usd=cost_usd(self.model, usage.input_tokens, usage.output_tokens),
        ))


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str | None = None, timeout_s: float = 60.0):
        self.model = model or os.environ.get("CHAMELEON_GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        self.timeout_s = timeout_s
        self._client = None

    def _get_client(self):
        if self._client is None:
            # Reads GEMINI_API_KEY / GOOGLE_API_KEY. HttpOptions.timeout is in milliseconds.
            self._client = genai.Client(http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)))
        return self._client

    def complete(self, system: str, user: str, schema: dict) -> LLMResult:
        start = time.perf_counter()
        response = self._get_client().models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_json_schema=schema,
            ),
        )
        latency = time.perf_counter() - start
        data = _parse_json(response.text, "gemini")
        usage = response.usage_metadata
        input_tokens = (usage.prompt_token_count or 0) if usage else 0
        # Thinking tokens bill as output.
        output_tokens = ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0
        return LLMResult(data, LLMCallStats(
            provider=self.name, model=self.model, latency_s=latency,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost_usd(self.model, input_tokens, output_tokens),
        ))


def cached_complete(provider, system: str, user: str, schema: dict) -> LLMResult:
    """`provider.complete`, memoised on disk when env CHAMELEON_LLM_CACHE names
    a directory, so repeated dev runs over the same designs don't pay again.
    Only successful answers are stored. A hit reports zero cost and latency
    (this call spent nothing) but keeps the original token counts."""
    cache_dir = os.environ.get("CHAMELEON_LLM_CACHE")
    if not cache_dir:
        return provider.complete(system, user, schema)
    key = "|".join([provider.name, provider.model, system, user, json.dumps(schema, sort_keys=True)])
    path = Path(cache_dir) / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        stats = LLMCallStats(**stored["stats"])
        stats.cached, stats.cost_usd, stats.latency_s = True, 0.0, 0.0
        return LLMResult(stored["data"], stats)
    result = provider.complete(system, user, schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"data": result.data, "stats": asdict(result.stats)}), encoding="utf-8")
    return result


def get_provider(name: str | None = None) -> ClaudeProvider | GeminiProvider | None:
    """`name` (or env CHAMELEON_LLM, default "auto"): claude | gemini | heuristic | auto.
    "auto" picks Claude if ANTHROPIC_API_KEY is set, else Gemini if
    GEMINI_API_KEY / GOOGLE_API_KEY is set, else None (the heuristic)."""
    choice = (name or os.environ.get("CHAMELEON_LLM") or "auto").lower()
    if choice == "heuristic":
        return None
    if choice == "claude":
        return ClaudeProvider()
    if choice == "gemini":
        return GeminiProvider()
    if choice == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            return ClaudeProvider()
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            return GeminiProvider()
        return None
    raise ValueError(f"unknown LLM provider {choice!r}; expected claude, gemini, heuristic or auto")
