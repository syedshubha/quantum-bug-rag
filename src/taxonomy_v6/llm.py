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
from typing import Any, Optional

from dotenv import load_dotenv

from .schemas import TAXONOMY_FORCED

DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 800


class BaseLLM:
    def complete(self, messages: list[dict], **kw) -> str:
        raise NotImplementedError

    def parse(self, raw: str) -> dict:
        try:
            if isinstance(raw, dict):
                return raw
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
        retrieved_ids = list(kw.get("retrieved_ids", []))
        return json.dumps({
            "scores": scores,
            "taxonomy_class": max(scores, key=scores.get),
            "suspected_location": "[Mock]",
            "justification": "[Mock]",
            "evidence_ids": retrieved_ids[:1],
        })


def build_response_format(retrieved_ids: list[str]) -> dict[str, Any]:
    evidence_schema: dict[str, Any] = {
        "type": "array",
        "items": {"type": "string"},
    }
    if retrieved_ids:
        evidence_schema["items"] = {"type": "string", "enum": retrieved_ids}
    else:
        evidence_schema["maxItems"] = 0

    score_properties = {
        cls: {"type": "number", "minimum": 0.0, "maximum": 1.0}
        for cls in TAXONOMY_FORCED
    }
    schema = {
        "name": "taxonomy_v6_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "scores": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": score_properties,
                    "required": list(TAXONOMY_FORCED),
                },
                "taxonomy_class": {
                    "type": "string",
                    "enum": list(TAXONOMY_FORCED),
                },
                "suspected_location": {"type": "string"},
                "justification": {"type": "string"},
                "evidence_ids": evidence_schema,
            },
            "required": [
                "scores",
                "taxonomy_class",
                "suspected_location",
                "justification",
                "evidence_ids",
            ],
        },
    }
    return {"type": "json_schema", "json_schema": schema}


def parsed_response_is_complete(parsed: dict) -> bool:
    if not isinstance(parsed, dict):
        return False
    if parsed.get("_parse_error"):
        return False
    if parsed.get("taxonomy_class") not in TAXONOMY_FORCED:
        return False
    scores = parsed.get("scores")
    if not isinstance(scores, dict):
        return False
    for cls in TAXONOMY_FORCED:
        if cls not in scores:
            return False
        if not isinstance(scores[cls], (int, float)):
            return False
    evidence_ids = parsed.get("evidence_ids")
    return isinstance(evidence_ids, list)


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
        load_dotenv()
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict], **kw) -> str:
        from openai import RateLimitError, APIError
        for attempt in range(6):
            try:
                request = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": kw.get("temperature", self.temperature),
                    "max_completion_tokens": kw.get(
                        "max_completion_tokens", self.max_tokens
                    ),
                }
                response_format = kw.get("response_format")
                if response_format is not None:
                    request["response_format"] = response_format
                resp = self.client.chat.completions.create(
                    **request,
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
    load_dotenv()
    api_key: Optional[str] = None
    try:  # Kaggle environment
        from kaggle_secrets import UserSecretsClient  # type: ignore
        api_key = UserSecretsClient().get_secret("OPENAI_API_KEY")
    except Exception:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing — set the env var or use mock mode.")
    return OpenAILLM(api_key=api_key, model=model)
