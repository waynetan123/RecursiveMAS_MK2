"""Benchmark loading for Step 1 screening.

Delegates to the vendored RecursiveMAS loaders rather than reimplementing them.
Those loaders encode paper-specific choices — GPQA option shuffling, MedQA option
formatting, MBPP+ prompt-test selection — that decide whether our scores land on
the same scale as the published table. The only new loading code here is HLE,
which the paper does not use.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from screening.config import BenchmarkSpec, ScreenConfig, add_recursivemas_to_path

add_recursivemas_to_path()

from inference_utils.inference_mas import load_eval_questions_and_answers  # noqa: E402

# MBPP+ prompt-test count, matching run.py's --mbppplus_num_prompt_tests.
MBPPPLUS_NUM_PROMPT_TESTS = 3

# LiveCodeBench scores against public plus hidden tests, matching run.py's default.
LCB_USE_PRIVATE_TESTS = True

# HLE items whose answer is a letter choice rather than free text.
HLE_MULTIPLE_CHOICE = "multiplechoice"


# One evaluation item, normalised across every benchmark.
@dataclass(frozen=True)
class EvalRecord:
    idx: int
    question: str
    gold: str
    dataset_name: str  # the loader's own name, passed straight to compare_answers
    meta: dict[str, Any] | None = None  # code test cases, HLE answer type, etc.


# Read the first non-empty value among several candidate keys.
def _first_text(sample: Mapping, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = sample.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


# True when an HLE row carries an image and so cannot be screened text-only.
def _hle_is_multimodal(sample: Mapping) -> bool:
    # The hub stores an empty string rather than null for text-only rows.
    for key in ("image", "image_preview"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return True
    return False


# Load the text-only subset of Humanity's Last Exam.
def _load_hle(spec: BenchmarkSpec, config: ScreenConfig) -> tuple[str, list[EvalRecord]]:
    from datasets import load_dataset

    dataset = load_dataset("cais/hle", split=spec.split, token=config.hf_token or None)

    # Drop multimodal items: Step 3 disables the vision path, so a score that
    # depended on images would not describe the system actually being built.
    text_only = [row for row in dataset if not _hle_is_multimodal(row)]
    if not text_only:
        raise ValueError("HLE text-only subset is empty — check the dataset schema.")

    if spec.shuffle:
        import random

        random.Random(config.seed).shuffle(text_only)
    if spec.num_samples > 0:
        text_only = text_only[: spec.num_samples]

    # Multiple-choice items need the choice instruction appended, same as MedQA.
    records = []
    for idx, row in enumerate(text_only):
        question = _first_text(row, ("question", "problem"))
        answer_type = str(row.get("answer_type") or "").replace(" ", "").lower()
        records.append(
            EvalRecord(
                idx=idx,
                question=question,
                gold=_first_text(row, ("answer",)),
                dataset_name="hle",
                meta={
                    "answer_type": answer_type,
                    "is_multiple_choice": answer_type == HLE_MULTIPLE_CHOICE,
                    "category": row.get("category"),
                    "raw_subject": row.get("raw_subject"),
                },
            )
        )
    return "hle", records


# Load one benchmark, returning the loader's dataset name and normalised records.
def load_benchmark(
    spec: BenchmarkSpec,
    config: ScreenConfig,
    num_samples_override: int = -1,
) -> tuple[str, list[EvalRecord]]:
    # A CLI cap (--num-samples) overrides the spec, for canary and debug runs.
    num_samples = num_samples_override if num_samples_override > 0 else spec.num_samples

    if spec.key == "hle":
        dataset_name, records = _load_hle(spec, config)
        return dataset_name, records[:num_samples] if num_samples > 0 else records

    # Everything else comes from the vendored loader, with the same arguments
    # inference_mas.main() passes, so questions and golds are byte-identical.
    dataset_name, questions, golds, metadata = load_eval_questions_and_answers(
        dataset=spec.loader_arg,
        dataset_split=spec.split,
        num_samples=num_samples,
        shuffle=spec.shuffle,
        seed=config.seed,
        gpqa_shuffle_options=True,
        return_metadata=True,
        lcb_use_private_tests=LCB_USE_PRIVATE_TESTS,
        mbppplus_subset="",
        mbppplus_cache_dir="",
        mbppplus_num_prompt_tests=MBPPPLUS_NUM_PROMPT_TESTS,
    )

    # Code benchmarks carry per-item test cases in metadata; others carry none.
    records = [
        EvalRecord(
            idx=idx,
            question=question,
            gold=str(gold),
            dataset_name=dataset_name,
            meta=metadata[idx] if metadata else None,
        )
        for idx, (question, gold) in enumerate(zip(questions, golds))
    ]
    return dataset_name, records
