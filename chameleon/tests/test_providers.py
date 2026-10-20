import json
import math
import re
import pytest
from types import SimpleNamespace as NS
from engine.pipeline.priority import providers, prompt
from engine.pipeline.priority.pricing import cost_usd
from engine.pipeline.priority.providers import ClaudeProvider, GeminiProvider, ProviderError, get_provider

PAYLOAD = {"summary": "s", "roles": [{"unit_id": "a", "role": "headline"}], "ranking": ["a"]}
KEYS = ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "CHAMELEON_LLM", "CHAMELEON_CLAUDE_MODEL", "CHAMELEON_GEMINI_MODEL")


class FakeAnthropic:
    def __init__(self, *a, stop="end_turn", text=json.dumps(PAYLOAD), content=None, **k):
        self.client_kwargs = k
        self.kw = None; self.beta = NS(messages=NS(create=self._create)); self._stop, self._text = stop, text
        self._content = content
    def _create(self, **kw):
        self.kw = kw
        return NS(stop_reason=self._stop, content=self._content or [NS(type="text", text=self._text)],
                  usage=NS(input_tokens=1000, output_tokens=200))


def fake_gemini(monkeypatch, seen):
    def generate_content(model, contents, config):
        seen.update(model=model, contents=contents, config=config)
        return NS(text=json.dumps(PAYLOAD), usage_metadata=NS(prompt_token_count=1000, candidates_token_count=150, thoughts_token_count=50))
    def client(*a, **k):
        seen["client_kwargs"] = k
        return NS(models=NS(generate_content=generate_content))
    monkeypatch.setattr(providers.genai, "Client", client)


def test_schema_enums_unit_ids():
    s = prompt.response_schema(["a", "b"])
    assert s["properties"]["ranking"]["items"]["enum"] == ["a", "b"]
    assert s["additionalProperties"] is False


def test_claude_request_shape_and_cost(monkeypatch):
    fake = FakeAnthropic()
    monkeypatch.setattr(providers.anthropic, "Anthropic", lambda *a, **k: fake)
    res = ClaudeProvider(model="claude-opus-5-5").complete("sys", "user", prompt.response_schema(["a"]))
    kw = fake.kw
    assert "temperature" not in kw and kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["output_config"]["format"]["type"] == "json_schema" and kw["output_config"]["effort"] == "low"
    assert res.data == PAYLOAD and res.stats.cost_usd == pytest.approx(1000 * 4 / 1e6 + 200 * 20 / 1e6)


@pytest.mark.parametrize("stop,text", [("refusal", ""), ("max_tokens", '{"summ'), ("end_turn", "not json")])
def test_claude_refusal_and_truncation_raise(monkeypatch, stop, text):
    monkeypatch.setattr(providers.anthropic, "Anthropic", lambda *a, **k: FakeAnthropic(stop=stop, text=text))
    with pytest.raises(ProviderError):
        ClaudeProvider().complete("sys", "user", prompt.response_schema(["a"]))


def test_gemini_request_shape_and_cost(monkeypatch):
    seen = {}
    fake_gemini(monkeypatch, seen)
    res = GeminiProvider(model="gemini-3.8-flash").complete("sys", "user", {"type": "object"})
    assert seen["config"].response_mime_type == "application/json"
    assert seen["config"].response_json_schema == {"type": "object"}
    assert res.stats.output_tokens == 200
    assert res.stats.cost_usd == pytest.approx(1000 * 0.75 / 1e6 + 200 * 3.75 / 1e6)


def test_get_provider_auto(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    assert get_provider() is None
    monkeypatch.setenv("GEMINI_API_KEY", "x"); assert get_provider().name == "gemini"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x"); assert get_provider().name == "claude"
    assert get_provider("heuristic") is None


@pytest.mark.live
@pytest.mark.parametrize("name", ["claude", "gemini"])
def test_live_provider_on_l2(name, l2):
    from engine.pipeline.priority.units import build_units
    from engine.pipeline.priority.features import unit_features
    units = build_units(l2["source"]); ids = [u.unit_id for u in units]
    res = get_provider(name).complete(prompt.SYSTEM_PROMPT, prompt.build_user_message(unit_features(l2["source"], units)), prompt.response_schema(ids))
    assert {r["unit_id"] for r in res.data["roles"]} <= set(ids)
    print(res.stats)


# --- beyond the brief ---

def test_claude_reads_the_text_block_after_thinking(monkeypatch):
    # Opus 5.5 always thinks, so a thinking block comes before the JSON text block
    content = [NS(type="thinking", thinking=""), NS(type="text", text=json.dumps(PAYLOAD))]
    monkeypatch.setattr(providers.anthropic, "Anthropic", lambda *a, **k: FakeAnthropic(content=content))
    assert ClaudeProvider().complete("sys", "user", {}).data == PAYLOAD


def test_claude_passes_prompt_and_client_settings(monkeypatch):
    fake = FakeAnthropic()
    made = {}
    def make(*a, **k):
        made.update(k); return fake
    monkeypatch.setattr(providers.anthropic, "Anthropic", make)
    res = ClaudeProvider(model="claude-haiku-5-5", timeout_s=12).complete("SYS", "USER", {"x": 1})
    assert made == {"timeout": 12, "max_retries": 1}
    assert fake.kw["model"] == "claude-haiku-5-5" and fake.kw["system"] == "SYS" and fake.kw["max_tokens"] == 4096
    assert fake.kw["messages"] == [{"role": "user", "content": "USER"}]
    assert fake.kw["output_config"]["format"]["schema"] == {"x": 1}
    assert (res.stats.provider, res.stats.model, res.stats.input_tokens, res.stats.output_tokens) == ("claude", "claude-haiku-5-5", 1000, 200)


def test_gemini_passes_prompt_and_timeout_in_ms(monkeypatch):
    seen = {}
    fake_gemini(monkeypatch, seen)
    GeminiProvider(model="gemini-3.5-flash-lite", timeout_s=12).complete("SYS", "USER", {"type": "object"})
    assert seen["client_kwargs"]["http_options"].timeout == 12000
    assert seen["model"] == "gemini-3.5-flash-lite" and seen["contents"] == "USER"
    assert seen["config"].system_instruction == "SYS"


def test_providers_create_clients_lazily(monkeypatch):
    # constructing a provider must not need a key: get_provider("claude") with no key may not raise
    def boom(*a, **k): raise RuntimeError("client created too early")
    monkeypatch.setattr(providers.anthropic, "Anthropic", boom)
    monkeypatch.setattr(providers.genai, "Client", boom)
    ClaudeProvider(); GeminiProvider()


def test_get_provider_explicit_names(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("CHAMELEON_LLM", "gemini")
    assert get_provider().name == "gemini"
    assert get_provider("claude").model == "claude-opus-5-5"
    monkeypatch.setenv("CHAMELEON_CLAUDE_MODEL", "claude-sonnet-5-5")
    assert get_provider("claude").model == "claude-sonnet-5-5"
    with pytest.raises(ValueError):
        get_provider("gpt")


def test_env_example_ships_no_keys_and_names_priced_models(monkeypatch):
    from pathlib import Path
    from engine.pipeline.priority.pricing import PRICES_PER_MTOK
    monkeypatch.setattr(providers.os, "environ", {})  # only what .env.example sets
    providers._load_env_file(Path(".env.example"))
    assert get_provider() is None  # no key in the example, so "auto" means the heuristic
    for var in ("CHAMELEON_CLAUDE_MODEL", "CHAMELEON_GEMINI_MODEL"):
        assert providers.os.environ.get(var) in PRICES_PER_MTOK, var


def test_cost_of_unknown_model_is_nan():
    assert math.isnan(cost_usd("some-new-model", 100, 100))
    assert cost_usd("claude-haiku-5-5", 2_000_000, 1_000_000) == pytest.approx(0.70)


def test_user_message_leaves_background_out(l2):
    from engine.pipeline.priority.units import build_units
    from engine.pipeline.priority.features import unit_features
    msg = prompt.build_user_message(unit_features(l2["source"], build_units(l2["source"])))
    data = json.loads(msg)
    assert data["background_unit_ids"] == ["bg"]
    assert "bg" not in [u["unit_id"] for u in data["units"]] and len(data["units"]) == 8
    assert '": ' not in msg and '", "' not in msg  # compact separators keep the prompt short
    assert list(data["units"][0]) == sorted(data["units"][0])  # sorted keys: same features -> same prompt bytes
    assert not re.search(r"\d\.\d{4,}", msg)  # floats rounded to 3 places


# --- real SDKs end to end, with only the network replaced by a mock transport ---

def test_claude_through_the_real_sdk(monkeypatch):
    import httpx2
    seen = {}
    def handler(request):
        seen.update(url=str(request.url), beta=request.headers.get("anthropic-beta", ""), body=json.loads(request.content))
        return httpx2.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "thinking", "thinking": "", "signature": "x"}, {"type": "text", "text": json.dumps(PAYLOAD)}],
            "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 1000, "output_tokens": 200}})
    real = providers.anthropic.Anthropic
    monkeypatch.setattr(providers.anthropic, "Anthropic", lambda **k: real(
        api_key="test", http_client=httpx2.Client(transport=httpx2.MockTransport(handler)), **k))
    res = ClaudeProvider().complete("SYS", "USER", prompt.response_schema(["a"]))
    body = seen["body"]
    assert "/v1/messages" in seen["url"] and "server-side-fallback-2026-07-01" in seen["beta"]
    assert body["fallbacks"] == "default" and "temperature" not in body and body["max_tokens"] == 4096
    assert body["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": prompt.response_schema(["a"])}}
    assert res.data == PAYLOAD and (res.stats.input_tokens, res.stats.output_tokens) == (1000, 200)


def test_gemini_through_the_real_sdk(monkeypatch):
    import httpx
    from google.genai import types
    seen = {}
    def handler(request):
        seen.update(url=str(request.url), body=json.loads(request.content))
        return httpx.Response(200, json={
            "candidates": [{"content": {"role": "model", "parts": [{"text": json.dumps(PAYLOAD)}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 150, "thoughtsTokenCount": 50, "totalTokenCount": 1200}})
    real = providers.genai.Client
    def make(http_options):
        return real(api_key="test", http_options=types.HttpOptions(
            timeout=http_options.timeout, httpx_client=httpx.Client(transport=httpx.MockTransport(handler))))
    monkeypatch.setattr(providers.genai, "Client", make)
    res = GeminiProvider().complete("SYS", "USER", prompt.response_schema(["a"]))
    gen = seen["body"]["generationConfig"]
    assert "gemini-3.8-flash:generateContent" in seen["url"]
    assert gen["responseMimeType"] == "application/json" and gen["responseJsonSchema"] == prompt.response_schema(["a"])
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "SYS"
    assert res.data == PAYLOAD and (res.stats.input_tokens, res.stats.output_tokens) == (1000, 200)
