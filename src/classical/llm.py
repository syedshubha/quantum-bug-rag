"""LLM client for the classical-vs-quantum binary track.

This client is independent of ``src.llm_client`` because the v6 notebook's
backend has slightly different invocation conventions (e.g.
``max_completion_tokens`` rather than ``max_tokens``) and a tailored mock
that biases its score by counting quantum-vs-classical token signatures.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Optional

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
    """Heuristic mock that biases ``score_quantum`` by token signatures.

    Used for offline development; produces valid JSON every call.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def complete(self, messages: list[dict], **kw) -> str:
        text = messages[-1]["content"].lower()
        q = sum(text.count(t) for t in (
            "qiskit", "qubit", "circuit.", "qml.", "measure", ".cx(", ".h(", "statevector",
        ))
        c = sum(text.count(t) for t in (
            "except", "raise", "tolerance", "assertion", "test_", "logging.",
        ))
        score = max(0.05, min(0.95, 0.5 + 0.05 * (q - c) + self.rng.uniform(-0.03, 0.03)))
        score = round(score, 3)
        return json.dumps({
            "reasoning": "[mock]",
            "score_quantum": score,
            "predicted": "quantum" if score >= 0.5 else "classical",
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
    if use_mock:
        return MockLLM(seed=42)
    api_key: Optional[str] = None
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        api_key = UserSecretsClient().get_secret("OPENAI_API_KEY")
    except Exception:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing — set the env var or use mock mode.")
    return OpenAILLM(api_key=api_key, model=model)
