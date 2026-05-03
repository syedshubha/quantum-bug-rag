"""Forced-choice classifier prompt and prompt builders for the v6 track.

The system message instructs the LLM that the input is *guaranteed* to be
buggy (no "no bug" outcome) and asks for one of five taxonomy classes plus
a per-class score in [0, 1]. We post-process the response argmax-style so
even a borderline score-set yields a valid prediction.
"""

from __future__ import annotations

from .schemas import BugPattern, BugSample

CLASSIFIER_PROMPT = """You are a quantum bug classifier. The given code DEFINITELY contains a bug. Your job is to identify which of these 5 categories the bug belongs to.

Categories (one of these MUST apply — there is no "no bug" option):
- incorrect_operator: wrong gate, wrong gate args (degrees vs radians, wrong matrix), deprecated/removed API, wrong API call, wrong type, wrong numerical computation, post-transpile mutation, oversized circuit, wrong sub-gate composition
- incorrect_qubit_mapping: wrong qubit/cbit index, off-by-one in register, MSB/LSB reversal, CNOT control/target swap, wrong identifier, wrong wires assignment
- missing_barrier: a barrier() is needed to prevent transpiler reordering but is absent
- wrong_initial_state: initialize() with wrong-size vector (must be 2^n), non-normalized amplitudes, wrong basis encoding, wrong state preparation
- measurement_error: missing measure before execute, double measure, gate-after-measure, c_if on unmeasured cbit, wrong qubit-to-cbit count

Score each category from 0.0 to 1.0 by how well it explains the bug. Scores DO NOT need to sum to 1.0. Pick the category with the HIGHEST score as the final classification.

Output JSON only, no markdown:
{"scores": {"incorrect_operator": <0-1>, "incorrect_qubit_mapping": <0-1>, "missing_barrier": <0-1>, "wrong_initial_state": <0-1>, "measurement_error": <0-1>}, "taxonomy_class": "<category with highest score>", "suspected_location": "<code fragment>", "justification": "<one sentence>", "evidence_ids": ["<retrieved reference id>", "..."]}"""

MAX_CODE_CHARS = 4000
MAX_REF_CHARS = 350


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "\n# [truncated]"


def build_prompt_only(sample: BugSample) -> list[dict]:
    user = (
        "No external references are provided for this sample, so evidence_ids "
        "must be an empty array.\n\n"
        f"Code (definitely buggy):\n```python\n{_truncate(sample.code, MAX_CODE_CHARS)}\n```"
    )
    return [
        {"role": "system", "content": CLASSIFIER_PROMPT},
        {"role": "user", "content": user},
    ]


def build_rag_prompt(sample: BugSample, retrieved: list[BugPattern]) -> list[dict]:
    refs = [
        (
            f"Reference ID {p.pattern_id} [{p.taxonomy_class}]: "
            f"{_truncate(p.description, MAX_REF_CHARS)}"
        )
        for p in retrieved
    ]
    refs_text = "\n".join(refs) if refs else "(no relevant references)"
    user = (
        "Reference bug-fix patterns (from validated quantum-framework release notes — "
        "use these as category guidance). If you cite evidence, "
        "only use the provided reference IDs in evidence_ids:\n"
        f"{refs_text}\n\n"
        f"Code (definitely buggy):\n```python\n{_truncate(sample.code, MAX_CODE_CHARS)}\n```"
    )
    return [
        {"role": "system", "content": CLASSIFIER_PROMPT},
        {"role": "user", "content": user},
    ]
