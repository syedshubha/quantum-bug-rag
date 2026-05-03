"""LLM client for the v6 forced-choice taxonomy track.

Mirrors the notebook's lightweight client (no project-wide config plumbing):
``MockLLM`` for offline development, ``OpenAILLM`` for production runs.

Both clients expose a single ``complete(messages, **kw) -> str`` method and a
``parse(raw) -> dict`` helper that strips markdown fencing before
``json.loads``.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Optional

from .schemas import TAXONOMY_FORCED

DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 800


class BaseLLM:
    def complete(self, messages: list[dict], **kw) -> str:
        raise NotImplementedError

    def parse(self, raw: str) -> dict:
        try:
            t = raw.strip()
            if t.startswith("```"):
                lines = t.splitlines()
                t = "\n".join(lines[1:-1]) if len(lines) > 2 else t
            return json.loads(t)
        except json.JSONDecodeError as e:
            return {"_raw": raw[:500], "_parse_error": str(e)}


class MockLLM(BaseLLM):
    """Deterministic mock for offline development. Emits valid JSON every call."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def complete(self, messages: list[dict], **kw) -> str:
        scores = {c: round(self.rng.uniform(0.1, 0.95), 2) for c in TAXONOMY_FORCED}
        return json.dumps({
            "scores": scores,
            "taxonomy_class": max(scores, key=scores.get),
            "suspected_location": "[Mock]",
            "justification": "[Mock]",
        })


class OpenAILLM(BaseLLM):
    """OpenAI ChatCompletions client with rate-limit aware retry loop."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict], **kw) -> str:
        from openai import RateLimitError, APIError
        for attempt in range(6):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=kw.get("temperature", self.temperature),
                    max_completion_tokens=kw.get(
                        "max_completion_tokens", self.max_tokens
                    ),
                )
                return resp.choices[0].message.content or ""
            except RateLimitError:
                wait = min(2 ** attempt * 5, 120)
                print(f"  [openai] rate-limited, sleeping {wait}s")
                time.sleep(wait)
            except APIError:
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt * 3, 60))
        raise RuntimeError("OpenAI retries exhausted")


def build_llm(use_mock: bool = False, model: str = DEFAULT_MODEL) -> BaseLLM:
    """Construct the v6 LLM client from environment configuration."""
    if use_mock:
        return MockLLM(seed=42)
    api_key: Optional[str] = None
    try:  # Kaggle environment
        from kaggle_secrets import UserSecretsClient  # type: ignore
        api_key = UserSecretsClient().get_secret("OPENAI_API_KEY")
    except Exception:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing — set the env var or use mock mode.")
    return OpenAILLM(api_key=api_key, model=model)
