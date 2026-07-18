"""
Thin wrapper around the Anthropic API. Centralized here so:
  - the model name/version lives in one place
  - it's trivial to mock in unit tests (see tests/)
  - both the multi-agent pipeline and the baseline single-LLM-call system
    use the exact same client/model, keeping the thesis comparison fair.

Requires ANTHROPIC_API_KEY in the environment (.env file supported via
python-dotenv, loaded in app.py / evaluate.py entry points).
"""
from __future__ import annotations

import json
import os

import anthropic

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY from env
    return _client


def call_claude_json(system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> dict:
    """Calls Claude and parses a JSON object from the response text.

    Prompts used with this helper must explicitly instruct the model to
    respond with ONLY a JSON object, no markdown fences, no preamble.
    """
    client = get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Claude nuk ktheu JSON të vlefshëm: {text[:300]}") from e
