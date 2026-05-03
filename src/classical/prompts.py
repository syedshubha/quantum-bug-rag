"""Binary classifier prompts (quantum vs classical) for the classical track.

Design choices:

  - Equal-length descriptions for both classes (the v6 issue was that one
    class had nine sub-cases described while others had one).
  - One canonical example per class, drawn from BQCP's ``comment`` field
    so it reflects the actual labelling intent of the dataset.
  - Chain-of-thought field requested first; the model writes reasoning
    before assigning a probability.
  - Output is a single probability ``score_quantum ∈ [0, 1]``, from which
    we derive the discrete prediction with a 0.5 threshold. A binary
    task with a continuous score makes calibration analysis tractable.
"""

from __future__ import annotations

from .schemas import BugSample, KBEntry

SYSTEM_PROMPT = """You analyze a buggy code snippet from a quantum-software repository
and decide whether the bug is in QUANTUM logic or in CLASSICAL logic.

Definitions:

- QUANTUM bug: the defect is in code that manipulates quantum state. Examples:
  wrong gate is applied, wrong qubit indices, malformed state preparation,
  incorrect measurement, miscompiled quantum circuit, wrong basis, wrong
  parameterization of a quantum operator, errors in transpilation passes
  that affect quantum semantics.
  Example from Bugs in Quantum Computing Platforms (BQCP-302): a Y90 pulse
  is generated with the opposite sign, producing the wrong unitary.

- CLASSICAL bug: the defect is in surrounding software that does not depend
  on quantum reasoning. Examples: missing error handling around a solver
  call, wrong numerical tolerance in an equality check, type confusion,
  test misconfiguration, file I/O failure, build/packaging issue, generic
  Python exception bug.
  Example from BQCP-3: cvx_fit does not check whether prob.solve() succeeded;
  fix adds a retry loop and an error path. The defect is in classical
  control flow, not in any quantum operation.

A bug is CLASSICAL even if it occurs inside a quantum-framework repository,
provided the buggy logic itself does not involve quantum-specific reasoning.

Output a single JSON object (no markdown):
{
  "reasoning":     "<2-3 sentences identifying which logic the defect is in>",
  "score_quantum": <float in [0, 1]: probability the bug is quantum>,
  "predicted":     "<\\"quantum\\" or \\"classical\\">"
}
"""

MAX_CODE_CHARS = 5000
MAX_REF_CHARS = 350


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "\n# [truncated]"


def build_prompt_only(sample: BugSample) -> list[dict]:
    user = f"Buggy snapshot:\n```\n{_truncate(sample.code, MAX_CODE_CHARS)}\n```"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_rag(sample: BugSample, retrieved: list[KBEntry]) -> list[dict]:
    refs = [
        f"  Ref{i} [{e.domain}/{e.framework}]: {_truncate(e.description, MAX_REF_CHARS)}"
        for i, e in enumerate(retrieved, 1)
    ]
    refs_block = "\n".join(refs) if refs else "  (no references retrieved)"
    user = (
        "Reference bug-fix patterns (retrieved from documentation; each is "
        "labeled with its domain — use the labels as evidence, not as the "
        "answer):\n"
        f"{refs_block}\n\n"
        f"Buggy snapshot:\n```\n{_truncate(sample.code, MAX_CODE_CHARS)}\n```"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
