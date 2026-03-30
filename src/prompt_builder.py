"""
prompt_builder.py – Construct LLM prompts for each experimental mode.

Three modes are supported:
  - prompt_only : code snippet only; no retrieved context.
  - rag         : code snippet + retrieved bug-pattern context.

The static mode does not use an LLM; see baselines.py.
"""

from __future__ import annotations

from .schemas import BugPattern, BugSample, TaxonomyEntry

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert in quantum software engineering and Qiskit programming. "
    "Your task is to analyse a Qiskit Python code snippet and produce a structured "
    "bug diagnostic. Return exactly one valid JSON object and nothing else."
    "Do not include markdown or extra text.\n\n"

    "The JSON schema is:\n"
    "{\n"
    '  "bug_likelihood": <float 0.0–1.0>,\n'
    '  "taxonomy_class": <string>,\n'
    '  "suspected_location": <string>,\n'
    '  "justification": <string>\n'
    "}\n\n"
    "If you are unsure, set bug_likelihood to 0.5 and taxonomy_class to 'unknown'."
)

# ── Prompt builders ───────────────────────────────────────────────────────────

def build_prompt_only(sample: BugSample) -> list[dict[str, str]]:
    """
    Build a chat message list for prompt-only mode.

    The model receives only the raw code snippet.
    """
    user_content = (
        "Please analyse the following Qiskit code snippet and produce a bug diagnostic.\n\n"
        f"```python\n{sample.code}\n```"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_rag_prompt(
    sample: BugSample,
    retrieved_patterns: list[BugPattern],
    taxonomy_entries: list[TaxonomyEntry] | None = None,
) -> list[dict[str, str]]:
    """
    Build a chat message list for RAG mode.

    Prepends retrieved bug-pattern context before the code snippet.
    """
    context_parts: list[str] = []

    if retrieved_patterns:
        context_parts.append("## Retrieved Bug Patterns\n")
        for i, p in enumerate(retrieved_patterns, start=1):
            context_parts.append(
                f"### Pattern {i}: {p.name} [{p.taxonomy_class}]\n"
                f"{p.description}\n"
                + (f"Fix hint: {p.fix_hint}\n" if p.fix_hint else "")
            )

    if taxonomy_entries:
        context_parts.append("\n## Relevant Taxonomy Classes\n")
        for te in taxonomy_entries:
            context_parts.append(f"- **{te.class_id}**: {te.description}")

    context_block = "\n".join(context_parts)

    user_content = (
        f"{context_block}\n\n"
        "## Code Snippet to Analyse\n\n"
        f"```python\n{sample.code}\n```\n\n"
        "Using the retrieved context above, produce a structured bug diagnostic. Use the retrieved context as reference only.s"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def format_messages_as_text(messages: list[dict[str, str]]) -> str:
    """
    Flatten a chat-message list into a single string.

    Useful for backends that accept plain text rather than message lists.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "").capitalize()
        content = msg.get("content", "")
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)
