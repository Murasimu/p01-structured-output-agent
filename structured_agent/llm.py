"""Live LLM wiring via any OpenAI-compatible API (OpenAI, Groq, DeepSeek,
local vLLM/llama.cpp servers, etc.). Kept separate from the agent core so
tests never touch the network.

Configuration (environment variables):
  LLM_API_KEY   required
  LLM_BASE_URL  default https://api.openai.com/v1
  LLM_MODEL     default gpt-4o-mini
"""

from __future__ import annotations

import os
from typing import Callable

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


def make_llm(
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
) -> Callable[[list[dict[str, str]]], str]:
    """Return a `messages -> text` callable backed by an OpenAI-compatible chat API."""
    key = api_key or os.environ.get("LLM_API_KEY")
    if not key:
        raise RuntimeError("LLM_API_KEY is not set (see .env.example)")
    from openai import OpenAI  # imported lazily: demo-only dependency

    client = OpenAI(
        base_url=base_url or os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
        api_key=key,
    )
    resolved_model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)

    def llm(messages: list[dict[str, str]]) -> str:
        resp = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    return llm
