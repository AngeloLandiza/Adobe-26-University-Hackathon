"""Per-token prices for the provider benchmark (USD per 1M tokens, input and
output, as of 2026-10-13). Google lists rises for Gemini 4 Argon from
2026-12-01 and Gemini 3.8 Flash from 2027-01-01; update this table if those
dates pass. Thinking tokens bill as output on both providers."""

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-4-argon": (2.00, 10.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of one call; NaN for a model missing from the table, so the
    benchmark shows "unknown" rather than a misleading $0."""
    if model not in PRICES_PER_MTOK:
        return float("nan")
    price_in, price_out = PRICES_PER_MTOK[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
