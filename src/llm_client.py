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
        try:
            # Strip markdown code fences if present.
            text = raw.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                # Remove opening and closing fence lines.
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

    raise ValueError(
        f"Unknown LLM backend '{backend}'. Choose from: mock, openai, gemini."
    )
