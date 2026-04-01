"""
llm_client.py – LLM abstraction layer.

We define a BaseLLMClient interface and provide:
  - MockLLMClient    : deterministic stub for testing; no API calls.
  - OpenAILLMClient  : thin wrapper around the OpenAI chat completions API.
  - GeminiLLMClient  : thin wrapper around the Google Generative AI API.

OpenAI and Gemini clients are imported lazily so that their SDK packages are
optional dependencies.  API keys are read from environment variables only;
never hard-code secrets.
"""

from __future__ import annotations

import json
import os
import random
from abc import ABC, abstractmethod
from typing import Any

from .utils import get_logger

logger = get_logger(__name__)

# ── Base interface ────────────────────────────────────────────────────────────

class BaseLLMClient(ABC):
    """Abstract base class for all LLM backends."""

    @abstractmethod
    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """
        Send *messages* (OpenAI-style chat format) to the LLM and return the
        raw response string.
        """

    def complete_and_parse(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, Any]:
        """
        Call complete() and attempt to parse the response as JSON.

        Returns a dict; on parse failure returns a dict with key ``_raw`` set
        to the raw string and ``_parse_error`` set to the error message.
        """
        raw = self.complete(messages, **kwargs)
        return self._parse_raw(raw)

    def complete_and_parse_n(
        self,
        messages: list[dict[str, str]],
        n: int = 3,
        temperature: float = 0.4,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Call complete() *n* times at the given *temperature* and parse each
        response independently.  Used for self-consistency decoding.
        """
        results: list[dict[str, Any]] = []
        for i in range(n):
            raw = self.complete(messages, temperature=temperature, **kwargs)
            parsed = self._parse_raw(raw)
            results.append(parsed)
            logger.debug("Self-consistency pass %d/%d: %s", i + 1, n, parsed.get("taxonomy_class", "?"))
        return results

    @staticmethod
    def _parse_raw(raw: str) -> dict[str, Any]:
        """Parse a raw LLM response string as JSON."""
        try:
            text = raw.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse LLM response as JSON: %s", exc)
            return {"_raw": raw, "_parse_error": str(exc)}


# ── Mock client ───────────────────────────────────────────────────────────────

_MOCK_TAXONOMY_CLASSES = [
    "incorrect_operator",
    "incorrect_qubit_mapping",
    "missing_barrier",
    "wrong_initial_state",
    "measurement_error",
    "unknown",
]


class MockLLMClient(BaseLLMClient):
    """
    Deterministic stub LLM client.

    Returns a plausible-looking JSON diagnostic without making any API calls.
    Useful for unit tests and pipeline smoke-tests.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        taxonomy_class = self._rng.choice(_MOCK_TAXONOMY_CLASSES)
        bug_likelihood = round(self._rng.uniform(0.4, 0.95), 2)
        response = {
            "bug_likelihood": bug_likelihood,
            "taxonomy_class": taxonomy_class,
            "suspected_location": "qc.cx(0, 1)",
            "justification": (
                f"[Mock] Detected pattern consistent with '{taxonomy_class}'. "
                "This is a synthetic response for testing purposes."
            ),
        }
        return json.dumps(response)


# ── OpenAI client ─────────────────────────────────────────────────────────────

class OpenAILLMClient(BaseLLMClient):
    """
    LLM client backed by the OpenAI chat completions API.

    Requires the ``openai`` package and the ``OPENAI_API_KEY`` environment
    variable.
    """

    def __init__(self, model: str = "gpt-4o", max_tokens: int = 1024, temperature: float = 0.0) -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAILLMClient. "
                "Install it with: pip install openai"
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set."
            )
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        import time
        import openai as _openai  # noqa: PLC0415

        max_retries = 6
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=kwargs.get("max_tokens", self._max_tokens),
                    temperature=kwargs.get("temperature", self._temperature),
                )
                return response.choices[0].message.content or ""
            except _openai.RateLimitError as exc:
                wait = min(2 ** attempt * 5, 120)
                logger.warning(
                    "Rate limited (429). Retry %d/%d in %ds. %s",
                    attempt + 1, max_retries, wait, exc,
                )
                time.sleep(wait)

        # Final attempt – let it raise on failure
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
            temperature=kwargs.get("temperature", self._temperature),
        )
        return response.choices[0].message.content or ""


# ── Gemini client ─────────────────────────────────────────────────────────────

class GeminiLLMClient(BaseLLMClient):
    """
    LLM client backed by the Google Generative AI (Gemini) API.

    Requires the ``google-generativeai`` package and the ``GOOGLE_API_KEY``
    environment variable.
    """

    def __init__(self, model: str = "gemini-1.5-pro", max_tokens: int = 1024, temperature: float = 0.0) -> None:
        try:
            import google.generativeai as genai  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'google-generativeai' package is required for GeminiLLMClient. "
                "Install it with: pip install google-generativeai"
            ) from exc

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY environment variable is not set."
            )
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        # Convert OpenAI-style message list to a single prompt string.
        from .prompt_builder import format_messages_as_text  # noqa: PLC0415

        prompt = format_messages_as_text(messages)
        response = self._model.generate_content(prompt)
        return response.text or ""

# ── GitHub Models client ─────────────────────────────────────────────────────
class GitHubModelsLLMClient(BaseLLMClient):
    """
    LLM client using GitHub Models (Azure inference endpoint).
    Requires GITHUB_TOKEN environment variable.
    """

    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 1024, temperature: float = 0.0) -> None:
        import requests  # local import to avoid hard dependency

        token = os.environ.get("GITHUB_MODELS_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if not token:
            raise RuntimeError(
                "Neither GITHUB_MODELS_TOKEN nor GITHUB_TOKEN environment variable is set."
            )

        self._url = "https://models.inference.ai.azure.com/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._requests = requests

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        import time

        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "temperature": kwargs.get("temperature", self._temperature),
        }

        max_retries = 6
        for attempt in range(max_retries):
            response = self._requests.post(self._url, headers=self._headers, json=payload)
            if response.status_code == 429:
                wait = min(2 ** attempt * 5, 120)
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = max(wait, int(retry_after))
                    except ValueError:
                        pass
                logger.warning(
                    "Rate limited (429). Retry %d/%d in %ds.",
                    attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"] or ""

        # Final attempt – let it raise on failure
        response = self._requests.post(self._url, headers=self._headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"] or ""


# ── Factory ───────────────────────────────────────────────────────────────────

def build_llm_client(config: dict) -> BaseLLMClient:
    """
    Construct an LLM client from a configuration dict.

    Expected keys under ``config["llm"]``:
      - backend: "mock" | "openai" | "gemini"
      - openai.model, openai.max_tokens, openai.temperature
      - gemini.model, gemini.max_tokens, gemini.temperature
    """
    llm_cfg = config.get("llm", {})
    backend = llm_cfg.get("backend", "mock")

    if backend == "mock":
        return MockLLMClient()
    if backend == "openai":
        cfg = llm_cfg.get("openai", {})
        return OpenAILLMClient(
            model=cfg.get("model", "gpt-4o"),
            max_tokens=cfg.get("max_tokens", 1024),
            temperature=cfg.get("temperature", 0.0),
        )
    if backend == "gemini":
        cfg = llm_cfg.get("gemini", {})
        return GeminiLLMClient(
            model=cfg.get("model", "gemini-1.5-pro"),
            max_tokens=cfg.get("max_tokens", 1024),
            temperature=cfg.get("temperature", 0.0),
        )
    if backend == "github_models":
        cfg = llm_cfg.get("github_models", {})
        return GitHubModelsLLMClient(
            model=cfg.get("model", "gpt-4o-mini"),
            max_tokens=cfg.get("max_tokens", 1024),
            temperature=cfg.get("temperature", 0.0),
      )

    raise ValueError(
        f"Unknown LLM backend '{backend}'. Choose from: mock, openai, gemini, github_models"
    )
