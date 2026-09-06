"""Benchmark loading for Step 1 screening.

Delegates to the vendored RecursiveMAS loaders rather than reimplementing them.
Those loaders encode paper-specific choices — GPQA option shuffling, MedQA option
formatting, MBPP+ prompt-test selection — that decide whether our scores land on
the same scale as the published table.

The benchmarks the paper does not use are loaded here instead: HLE, and the four
that replaced the cells the first screen retired — Omni-MATH-2-Filtered,
MedXpertQA, and the two controls, BeyondAIME and Medbullets op4/op5.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from screening.config import BenchmarkSpec, ScreenConfig, add_recursivemas_to_path

add_recursivemas_to_path()

from inference_utils.inference_mas import (  # noqa: E402
    _stable_shuffle_indices,
    load_eval_questions_and_answers,
)

# MBPP+ prompt-test count, matching run.py's --mbppplus_num_prompt_tests.
MBPPPLUS_NUM_PROMPT_TESTS = 3

# LiveCodeBench scores against public plus hidden tests, matching run.py's default.
LCB_USE_PRIVATE_TESTS = True

# HLE items whose answer is a letter choice rather than free text.
HLE_MULTIPLE_CHOICE = "multiplechoice"

# Source repositories for the replacement benchmarks. The vendored loaders know
# none of them, so all four are loaded here the way HLE already is.
#
# Omni-MATH-2, never KbsdJames/Omni-MATH: the 2024 original was audited in
# arXiv:2601.19532, which found 647 problems requiring edits and 247 that
# exact-answer verification cannot score — 61 of them referencing images the
# text-only screen could never see.
OMNI_MATH_REPO = "martheballon/Omni-MATH-2"
MEDXPERTQA_REPO = "TsinghuaC3I/MedXpertQA"
MEDXPERTQA_CONFIG = "Text"  # the MM config is multimodal; Step 3 disables vision
BEYONDAIME_REPO = "ByteDance-Seed/BeyondAIME"
MEDBULLETS_REPOS = {
    "medbullets_op4": "meoconxinhxan/Medical-Eval-MedBullets_op4",
    "medbullets_op5": "meoconxinhxan/Medical-Eval-MedBullets_op5",
}

# Omni-MATH-2 ships the audit's verdict per item in `tags`. An untagged row is one
# the audit judged answerable and exactly verifiable; the tagged ones request
# proofs or estimates, reference missing images, or are degenerate. Dropping them
# is what "Omni-MATH-2-Filtered" means — there is no separate split on the hub, so
# the filter is applied here and its size asserted below.
OMNI_MATH_EXPECTED_FILTERED_SIZE = 4174

# Where MedXpertQA's stem ends and its inline option list begins. Every one of the
# 2,450 rows carries this marker on its own line.
MEDXPERTQA_CHOICE_MARKER = "\nAnswer Choices:"

# compare_answers only takes its math branch for dataset names it recognises, and
# the vendored key list is never edited. BeyondAIME's positive-integer answers need
# exactly that branch — it is the only maths benchmark here with no judge in the
# loop — so the loader hands the scorer a name from the list. The cell is still
# recorded under its own key; nothing but the extraction cascade sees this.
MATH_CASCADE_DATASET_NAME = "aime25"


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


# Append the choice instruction, widened past the vendored A-D wording.
#
# `ensure_choice_instruction` hardcodes "(A/B/C/D)". Telling a model answering a
# ten-option MedXpertQA item to choose between four is not a formatting detail: it
# invites an answer outside the option set on every item where the gold is E-J.
def _choice_instruction(question: str, letters: str) -> str:
    question = question.rstrip()

    # Same idempotency guard as the vendored function, so a question that already
    # carries the instruction is not given a second, contradictory one.
    if "choose the correct option" in question.lower():
        return question
    return f"{question}\n\nChoose the correct option ({'/'.join(letters)})."


# Build a lettered option block and report where the gold ended up.
#
# Options are re-lettered from a stable hash of the item rather than left in their
# shipped order. Medmarks found smaller models measurably sensitive to answer
# position, and Medbullets ships a skewed gold distribution (op4: A 87, D 68 of
# 308), so without this a position effect could be read as distractor sensitivity.
def _lettered_options(
    option_texts: list[str],
    gold_position: int,
    letters: str,
    seed_text: str,
    shuffle: bool,
) -> tuple[str, str]:
    count = len(option_texts)
    if count > len(letters):
        raise ValueError(f"{count} options but only {len(letters)} letters available.")

    # The same md5-seeded permutation GPQA's options go through, so both choice
    # benchmarks are shuffled by one mechanism rather than two.
    order = _stable_shuffle_indices(count, seed_text) if shuffle else list(range(count))

    lines = []
    gold_letter = letters[0]
    for position, original in enumerate(order):
        letter = letters[position]
        lines.append(f"{letter}. {option_texts[original]}")
        if original == gold_position:
            gold_letter = letter
    return "\n".join(lines), gold_letter


# Stable identity for one Medbullets question, shared by its op4 and op5 versions.
#
# The two variants ship the same 308 questions in different row orders, so
# sample_idx cannot pair them. The op4->op5 drop is a paired difference on
# identical items, and without a join key it is not computable at all.
def _medbullets_pair_key(stem: str, gold_text: str) -> str:
    import hashlib

    normalised = f"{' '.join(stem.split())}||{' '.join(gold_text.split())}"
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()[:16]


# Load Omni-MATH-2-Filtered, optionally cut to a difficulty floor.
def _load_omni_math(spec: BenchmarkSpec, config: ScreenConfig) -> tuple[str, list[EvalRecord]]:
    from datasets import load_dataset

    dataset = load_dataset(OMNI_MATH_REPO, split=spec.split, token=config.hf_token or None)

    # The Filtered subset: drop every item the audit tagged as unverifiable.
    filtered = [row for row in dataset if not (row.get("tags") or [])]
    if len(filtered) != OMNI_MATH_EXPECTED_FILTERED_SIZE:
        raise ValueError(
            f"Omni-MATH-2 filtered to {len(filtered)} items, expected "
            f"{OMNI_MATH_EXPECTED_FILTERED_SIZE}. The tag schema has moved; re-check "
            "which items are being scored before spending on the cell."
        )

    # Two items survive the audit's tags with an empty reference answer. A judged
    # item with no gold asks the judge whether the prediction matches nothing, which
    # it cannot answer and which is scored wrong either way, so they are dropped and
    # the count reported rather than left to sit silently in the denominator.
    scoreable = [row for row in filtered if str(row.get("answer") or "").strip()]
    if len(scoreable) != len(filtered):
        print(
            f"omni_math: dropped {len(filtered) - len(scoreable)} items with no reference "
            f"answer, leaving {len(scoreable)}."
        )
    filtered = scoreable

    # The difficulty cut is a real trade-off — raising it buys headroom and costs n,
    # and both are required — so it is left at 0.0 until the gating cell produces the
    # score-versus-difficulty curve that can settle it. Applying it later is free:
    # the generations are cached, so --rescore-only re-scores a subset at no cost.
    if spec.min_difficulty > 0.0:
        filtered = [
            row
            for row in filtered
            if float(row.get("difficulty") or 0.0) >= spec.min_difficulty
        ]
        if not filtered:
            raise ValueError(
                f"No Omni-MATH item reaches difficulty {spec.min_difficulty}."
            )

    if spec.shuffle:
        import random

        random.Random(config.seed).shuffle(filtered)
    if spec.num_samples > 0:
        filtered = filtered[: spec.num_samples]

    # Difficulty is carried per item because Step 2 exists to read the cut off it;
    # domain and source let a later slice ask whether an effect concentrates in one
    # tier rather than spreading evenly.
    records = [
        EvalRecord(
            idx=idx,
            question=_first_text(row, ("problem", "question")),
            gold=_first_text(row, ("answer",)),
            dataset_name="omni_math",
            meta={
                "omni_id": row.get("id"),
                "difficulty": row.get("difficulty"),
                "domain": row.get("domain"),
                "source": row.get("source"),
            },
        )
        for idx, row in enumerate(filtered)
    ]
    return "omni_math", records


# Load the text split of MedXpertQA.
def _load_medxpertqa(spec: BenchmarkSpec, config: ScreenConfig) -> tuple[str, list[EvalRecord]]:
    from datasets import load_dataset

    dataset = load_dataset(
        MEDXPERTQA_REPO,
        MEDXPERTQA_CONFIG,
        split=spec.split,
        token=config.hf_token or None,
    )

    records = []
    for idx, row in enumerate(dataset):
        # The shipped question already ends in "Answer Choices: (A) ... (J) ...", all
        # on one line. prompts.py detects a choice question by looking for a lettered
        # option at the start of a line, so left as shipped the item would be given
        # the free-form "put the final answer inside \boxed{}" instruction instead of
        # the choice one. Rebuild the block the way every other choice benchmark
        # presents it: stem, then one option per line.
        stem = str(row.get("question") or "").split(MEDXPERTQA_CHOICE_MARKER)[0].rstrip()
        options = row.get("options") or {}
        letters = sorted(letter for letter, text in options.items() if str(text).strip())
        block = "\n".join(f"{letter}. {options[letter]}" for letter in letters)

        records.append(
            EvalRecord(
                idx=idx,
                # Golds are already uniform over all ten letters (222-263 each), so
                # shuffling would buy nothing and would only break the item ids.
                question=_choice_instruction(f"{stem}\n{block}", spec.choice_letters),
                gold=str(row.get("label") or "").strip().upper(),
                dataset_name="medxpertqa",
                meta={
                    "item_id": row.get("id"),
                    "medical_task": row.get("medical_task"),
                    "body_system": row.get("body_system"),
                    "question_type": row.get("question_type"),
                    "num_options": len(letters),
                },
            )
        )

    _assert_golds_in_range(records, spec.choice_letters, spec.key)
    return "medxpertqa", records


# Load BeyondAIME, the maths control.
def _load_beyondaime(spec: BenchmarkSpec, config: ScreenConfig) -> tuple[str, list[EvalRecord]]:
    from datasets import load_dataset

    dataset = load_dataset(BEYONDAIME_REPO, split=spec.split, token=config.hf_token or None)

    records = [
        EvalRecord(
            idx=idx,
            question=_first_text(row, ("problem", "question")),
            gold=str(row.get("answer")).strip(),
            dataset_name=MATH_CASCADE_DATASET_NAME,
            meta={"answer_type": "integer"},
        )
        for idx, row in enumerate(dataset)
    ]
    return MATH_CASCADE_DATASET_NAME, records


# Load one Medbullets variant, re-lettered and ordered to pair with the other.
def _load_medbullets(spec: BenchmarkSpec, config: ScreenConfig) -> tuple[str, list[EvalRecord]]:
    from datasets import load_dataset

    repo = MEDBULLETS_REPOS[spec.key]
    dataset = load_dataset(repo, split=spec.split, token=config.hf_token or None)

    records = []
    for idx, row in enumerate(dataset):
        options = row.get("options") or {}
        original_letters = sorted(options)
        option_texts = [str(options[letter]) for letter in original_letters]

        # answer_idx is the authoritative gold letter, so the gold is located by
        # index rather than by matching the answer text back against the options.
        gold_letter = str(row.get("answer_idx") or "").strip().upper()
        if gold_letter not in original_letters:
            raise ValueError(f"{spec.key} row {idx}: answer_idx {gold_letter!r} is not an option.")

        stem = str(row.get("question") or "").rstrip()
        pair_key = _medbullets_pair_key(stem, str(row.get("answer") or ""))
        block, shuffled_gold = _lettered_options(
            option_texts=option_texts,
            gold_position=original_letters.index(gold_letter),
            letters=spec.choice_letters,
            # Keyed on the item, not the row: the ten questions each variant ships
            # twice then get the same permutation, so the duplicate pair produces one
            # prompt and the second copy is served from cache.
            seed_text=f"{config.seed}::{pair_key}",
            shuffle=True,
        )

        records.append(
            EvalRecord(
                idx=idx,
                question=_choice_instruction(f"{stem}\n{block}", spec.choice_letters),
                gold=shuffled_gold,
                dataset_name=spec.key,
                meta={
                    "pair_key": pair_key,
                    "variant": spec.key.rsplit("_", 1)[-1],
                    "num_options": len(option_texts),
                    "shipped_gold_letter": gold_letter,
                },
            )
        )

    # Ordered by the join key so op4 and op5 present the same item at the same
    # position, which makes the paired op4->op5 difference a column subtraction
    # rather than a reconciliation. Indices are reassigned after sorting.
    records.sort(key=lambda record: (record.meta or {}).get("pair_key", ""))
    records = [replace(record, idx=idx) for idx, record in enumerate(records)]

    _assert_golds_in_range(records, spec.choice_letters, spec.key)
    return spec.key, records


# Fail loud on a gold letter outside the benchmark's declared range.
#
# Silent is the dangerous mode here: an out-of-range gold is scored wrong on every
# item, which reads as a hard benchmark rather than a broken loader.
def _assert_golds_in_range(records: list[EvalRecord], letters: str, key: str) -> None:
    allowed = set(letters)
    bad = [record.idx for record in records if record.gold not in allowed]
    if bad:
        raise ValueError(
            f"{key}: {len(bad)} rows have a gold outside {letters} (first: {bad[:5]}). "
            "The dataset schema has moved."
        )


# Benchmarks loaded here rather than by the vendored harness, which knows none of them.
CUSTOM_LOADERS = {
    "hle": _load_hle,
    "omni_math": _load_omni_math,
    "medxpertqa": _load_medxpertqa,
    "beyondaime": _load_beyondaime,
    "medbullets_op4": _load_medbullets,
    "medbullets_op5": _load_medbullets,
}


# Load one benchmark, returning the loader's dataset name and normalised records.
def load_benchmark(
    spec: BenchmarkSpec,
    config: ScreenConfig,
    num_samples_override: int = -1,
) -> tuple[str, list[EvalRecord]]:
    # A CLI cap (--num-samples) overrides the spec, for canary and debug runs.
    num_samples = num_samples_override if num_samples_override > 0 else spec.num_samples

    custom_loader = CUSTOM_LOADERS.get(spec.key)
    if custom_loader is not None:
        dataset_name, records = custom_loader(spec, config)
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
