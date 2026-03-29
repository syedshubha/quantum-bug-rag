"""
llm_client.py – LLM abstraction with a mock backend and pluggable real backends.

I define a base ``LLMClient`` interface and two implementations:

* ``MockLLMClient``  – returns deterministic dummy responses; lets me test
  the full pipeline without any API calls or costs.
* ``OpenAILLMClient`` – wraps the OpenAI chat-completions API.
* ``GeminiLLMClient`` – wraps the Google Generative AI (Gemini) API.

I read API keys from environment variables and never embed them in code.
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .schemas import BugTaxonomyClass, DiagnosticResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """I define the contract that every LLM backend must satisfy."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """
        I send *prompt* to the model and return the raw text response.

        Implementors should raise ``LLMClientError`` on API failures.
        """

    def analyse(self, program_id: str, prompt: str, mode: str = "unknown") -> DiagnosticResult:
        """
        I call ``complete``, parse the JSON response, and return a
        ``DiagnosticResult``.  I fall back gracefully if parsing fails.
        """
        raw = self.complete(prompt)
        return _parse_response(program_id, raw, mode)


class LLMClientError(RuntimeError):
    """I represent an unrecoverable LLM API error."""


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def _parse_response(program_id: str, raw: str, mode: str) -> DiagnosticResult:
    """
    I extract the JSON object from *raw* and return a ``DiagnosticResult``.

    I tolerate the model wrapping the JSON in markdown code fences.
    """
    # Strip markdown fences if present.
    text = re.sub(r"```(?:json)?", "", raw).strip()

    # Try to locate the outermost JSON object.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        logger.warning("Could not extract JSON from LLM response for %s; using defaults.", program_id)
        return DiagnosticResult(
            program_id=program_id,
            bug_likelihood=0.5,
            taxonomy_class=BugTaxonomyClass.UNKNOWN,
            justification=raw[:500],
            mode=mode,
        )

    try:
        data: Dict[str, Any] = json.loads(match.group())
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error for %s: %s", program_id, exc)
        return DiagnosticResult(
            program_id=program_id,
            bug_likelihood=0.5,
            taxonomy_class=BugTaxonomyClass.UNKNOWN,
            justification=raw[:500],
            mode=mode,
        )

    # Normalise the taxonomy class safely.
    raw_class = data.get("taxonomy_class", BugTaxonomyClass.UNKNOWN)
    try:
        taxonomy_class = BugTaxonomyClass(raw_class)
    except ValueError:
        taxonomy_class = BugTaxonomyClass.UNKNOWN

    return DiagnosticResult(
        program_id=program_id,
        bug_likelihood=float(data.get("bug_likelihood", 0.5)),
        taxonomy_class=taxonomy_class,
        suspected_location=data.get("suspected_location"),
        justification=data.get("justification", ""),
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Mock client (no API calls)
# ---------------------------------------------------------------------------


class MockLLMClient(LLMClient):
    """
    I return deterministic dummy responses so I can validate the full
    pipeline without incurring API costs.

    I vary the response based on simple heuristics (e.g. the word
    ``measure`` appearing in the code) to make tests more meaningful.
    """

    def __init__(self, bug_likelihood: float = 0.7) -> None:
        self._default_likelihood = bug_likelihood
        logger.info("MockLLMClient initialised (bug_likelihood=%.2f)", bug_likelihood)

    def complete(self, prompt: str) -> str:
        # I look for simple signals in the prompt to vary the response.
        likelihood = self._default_likelihood
        taxonomy = BugTaxonomyClass.UNKNOWN.value

        if "measure" not in prompt.lower():
            likelihood = max(likelihood, 0.75)
            taxonomy = BugTaxonomyClass.MISSING_MEASUREMENT.value
        elif "cx" in prompt.lower() or "cnot" in prompt.lower():
            taxonomy = BugTaxonomyClass.WRONG_QUBIT_ORDER.value

        response = {
            "bug_likelihood": likelihood,
            "taxonomy_class": taxonomy,
            "suspected_location": None,
            "justification": (
                "Mock analysis: this is a placeholder response from the mock LLM client. "
                "Replace MockLLMClient with a real backend for production use."
            ),
        }
        return json.dumps(response)


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------


class OpenAILLMClient(LLMClient):
    """
    I wrap the OpenAI chat-completions API.

    I read the API key from the ``OPENAI_API_KEY`` environment variable.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        try:
            import openai  # type: ignore
        except ImportError as exc:
            raise ImportError("Install openai: `pip install openai`") from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMClientError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export it before running: export OPENAI_API_KEY=<your-key>"
            )

        self._client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info("OpenAILLMClient initialised (model=%s)", model)

    def complete(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMClientError(f"OpenAI API error: {exc}") from exc


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------


class GeminiLLMClient(LLMClient):
    """
    I wrap the Google Generative AI (Gemini) API.

    I read the API key from the ``GEMINI_API_KEY`` environment variable.
    """

    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.0,
    ) -> None:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Install google-generativeai: `pip install google-generativeai`"
            ) from exc

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise LLMClientError(
                "GEMINI_API_KEY environment variable is not set. "
                "Export it before running: export GEMINI_API_KEY=<your-key>"
            )

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)
        self._generation_config = {"temperature": temperature}
        logger.info("GeminiLLMClient initialised (model=%s)", model)

    def complete(self, prompt: str) -> str:
        try:
            response = self._model.generate_content(
                prompt,
                generation_config=self._generation_config,
            )
            return response.text or ""
        except Exception as exc:
            raise LLMClientError(f"Gemini API error: {exc}") from exc


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_llm_client(backend: str = "mock", **kwargs: Any) -> LLMClient:
    """
    I instantiate and return the requested LLM backend.

    Parameters
    ----------
    backend:
        One of ``"mock"``, ``"openai"``, ``"gemini"``.
    **kwargs:
        Passed through to the chosen client's ``__init__``.
    """
    backend = backend.lower()
    if backend == "mock":
        return MockLLMClient(**kwargs)
    if backend == "openai":
        return OpenAILLMClient(**kwargs)
    if backend == "gemini":
        return GeminiLLMClient(**kwargs)
    raise ValueError(f"Unknown LLM backend '{backend}'. Choose 'mock', 'openai', or 'gemini'.")
