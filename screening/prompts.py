"""Single-agent prompts for screening.

Screening measures one untrained backbone, so the multi-agent role framing in
`RecursiveMAS/inference/prompts.py` does not apply. The final-answer instructions
do: `answer_utils.compare_answers` extracts from `\\boxed{}` and from "Final
Choice:" lines, so changing that tail would change the score without changing the
model. Every instruction string below is lifted verbatim from the harness.
"""

from __future__ import annotations

import re

from screening.config import BenchmarkSpec, add_recursivemas_to_path
from screening.datasets import EvalRecord

add_recursivemas_to_path()

from prompts import SYSTEM_PROMPT, build_code_interface_prompt  # noqa: E402

# Detects a lettered option list, matching build_math_solver_prompt's own test.
CHOICE_QUESTION_PATTERN = re.compile(r"(?mi)^\s*[A-D]\s*[\.\):\-]\s+")

# GPQA runs with choice_old_prompt=2 in run.py; every other dataset uses 0.
GPQA_CHOICE_PROMPT_MODE = 2

# The four final-answer instructions, verbatim from prompts.py.
INSTRUCTION_CHOICE_TERSE = "Final Choice: put only the option letter in \\boxed{}, e.g., \\boxed{A}."
INSTRUCTION_CHOICE_DEFAULT = (
    "Solve the question and put the final choice inside \\boxed{}, for example \\boxed{A}."
)
INSTRUCTION_FREEFORM = (
    "Solve the question given information and put the final answer inside "
    "\\boxed{}, for example \\boxed{1}."
)
INSTRUCTION_CODE = (
    "Solve the problem and put the final code inside one markdown code block, "
    "for example ```python\\n<your solution code>\\n```."
)


# Pick the final-answer instruction the harness would use for this question.
def _final_instruction(question: str, choice_prompt_mode: int) -> tuple[str, str]:
    is_choice = bool(CHOICE_QUESTION_PATTERN.search(question))

    # Mode 2 is GPQA's terser wording; the harness also tightens the separator for it.
    if is_choice and choice_prompt_mode == GPQA_CHOICE_PROMPT_MODE:
        return INSTRUCTION_CHOICE_TERSE, "\n"
    if is_choice:
        return INSTRUCTION_CHOICE_DEFAULT, "\n\n"
    return INSTRUCTION_FREEFORM, "\n\n"


# Build the code prompt: interface line, problem, then the code-block instruction.
def _build_code_prompt(record: EvalRecord) -> str:
    meta = record.meta or {}
    interface = build_code_interface_prompt(
        meta.get("task_type", "complete"),
        fn_name=meta.get("fn_name"),
    )
    return (
        f"{interface}\n"
        "\n---\nThe programming problem is:\n"
        f"{record.question}\n"
        f"{INSTRUCTION_CODE}"
    )


# Build the user message for one record.
def build_prompt(record: EvalRecord, spec: BenchmarkSpec) -> str:
    if spec.scorer == "code":
        return _build_code_prompt(record)

    # GPQA is the only dataset the harness gives the terse choice wording to.
    choice_prompt_mode = GPQA_CHOICE_PROMPT_MODE if spec.key == "gpqa" else 0
    instruction, separator = _final_instruction(record.question, choice_prompt_mode)

    return f"Question:\n{record.question}{separator}{instruction}"


# Assemble the chat messages for one record.
def build_messages(record: EvalRecord, spec: BenchmarkSpec) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(record, spec)},
    ]
