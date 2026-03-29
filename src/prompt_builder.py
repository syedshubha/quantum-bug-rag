"""
prompt_builder.py – Constructing prompts for the LLM.

I build prompt strings from a program's source code, optional retrieved
context, and a system instruction.  Keeping prompt construction separate
from the LLM client makes it easy to iterate on prompts independently.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .schemas import BugPatternEntry, BugTaxonomyClass


# ---------------------------------------------------------------------------
# System instruction
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTION = """\
You are an expert Qiskit / quantum-circuit debugger.
Your task is to analyse Python source code that uses Qiskit and determine
whether it contains a bug.

Always respond in the following JSON format (no additional text):
{
  "bug_likelihood": <float 0.0-1.0>,
  "taxonomy_class": "<one of the allowed classes>",
  "suspected_location": "<file:line or null>",
  "justification": "<concise explanation>"
}

Allowed taxonomy classes:
""" + "\n".join(f"  - {c.value}" for c in BugTaxonomyClass)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_prompt_only(
    program_id: str,
    source_code: str,
    description: Optional[str] = None,
) -> str:
    """
    I build a prompt for the prompt-only (no retrieval) mode.

    Parameters
    ----------
    program_id:
        Identifier for the program (used only for labelling in the prompt).
    source_code:
        Raw Qiskit Python source code to analyse.
    description:
        Optional natural-language description of the program.
    """
    parts = [_SYSTEM_INSTRUCTION, "\n---\n"]
    parts.append(f"Program ID: {program_id}\n")
    if description:
        parts.append(f"Description: {description}\n")
    parts.append("\nSource code:\n```python\n")
    parts.append(source_code)
    parts.append("\n```\n")
    parts.append("\nAnalyse the code above and respond in the required JSON format.")
    return "".join(parts)


def build_rag_prompt(
    program_id: str,
    source_code: str,
    retrieved_patterns: List[BugPatternEntry],
    description: Optional[str] = None,
) -> str:
    """
    I build a RAG prompt that includes retrieved bug-pattern context.

    Parameters
    ----------
    program_id:
        Identifier for the program.
    source_code:
        Raw Qiskit Python source code to analyse.
    retrieved_patterns:
        Bug-pattern entries returned by the retriever.
    description:
        Optional natural-language description of the program.
    """
    parts = [_SYSTEM_INSTRUCTION, "\n---\n"]
    parts.append(f"Program ID: {program_id}\n")
    if description:
        parts.append(f"Description: {description}\n")

    if retrieved_patterns:
        parts.append("\n## Retrieved Bug Patterns (use as reference context)\n")
        for i, pattern in enumerate(retrieved_patterns, start=1):
            parts.append(f"\n### Pattern {i}: {pattern.title}\n")
            parts.append(f"Category: {pattern.taxonomy_class}\n")
            parts.append(f"Description: {pattern.description}\n")
            if pattern.code_snippet:
                parts.append(f"Example:\n```python\n{pattern.code_snippet}\n```\n")

    parts.append("\n---\n")
    parts.append("\nSource code:\n```python\n")
    parts.append(source_code)
    parts.append("\n```\n")
    parts.append(
        "\nUsing the retrieved patterns as context, analyse the code above "
        "and respond in the required JSON format."
    )
    return "".join(parts)


def format_few_shot_examples(examples: List[Dict[str, Any]]) -> str:
    """
    I format a list of labelled examples as few-shot demonstrations.

    Each example dict must have ``source_code`` and ``expected_output`` keys.
    """
    lines = ["\n## Few-shot Examples\n"]
    for i, ex in enumerate(examples, start=1):
        lines.append(f"\n### Example {i}\n")
        lines.append(f"```python\n{ex['source_code']}\n```\n")
        lines.append(f"Expected output:\n```json\n{json.dumps(ex['expected_output'], indent=2)}\n```\n")
    return "".join(lines)
