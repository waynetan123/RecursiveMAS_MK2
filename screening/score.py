"""Scoring for Step 1 screening.

Dispatches to the vendored RecursiveMAS scorers rather than reimplementing them,
so a screening score means the same thing as the corresponding number in the
paper's table. Adds two diagnostics the harness does not report:

- Parse-failure rate. `answer_utils.compare_answers` scores an unparseable
  multiple-choice prediction as "A" rather than as wrong, which hands back ~25%
  of unparsed items as free credit. An accuracy built on many unparsed items is
  not a measurement, so it is tracked separately.
- Truncation rate. A high rate means the token cap bound the score, not the
  model, so the number is a floor and is never grounds to retire a benchmark.
- Failure rate. The union of the two above with provider errors: every generation
  that could not yield a scoreable answer. Each of those is scored wrong when it
  might have been right, so score + failure_rate is the highest score the cell
  could have reached, which is what lets report.py decide a benchmark whose
  diagnostics are dirty but whose ceiling still sits below the gate.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from screening.config import AGGREGATION_MEAN, BenchmarkSpec, add_recursivemas_to_path
from screening.datasets import EvalRecord
from screening.openrouter import Response

add_recursivemas_to_path()

from inference_utils.answer_utils import (  # noqa: E402
    compare_answers,
    extract_boxed_answer,
)
from inference_utils.lcb_utils import (  # noqa: E402
    clean_raw_output,
    evaluate_generated_code,
    extract_python_code,
)

# 95% confidence, used for the Wilson interval around every score.
Z_95 = 1.959964

# Choice extraction, widened past the vendored A-D range.
#
# `answer_utils._extract_choice_core` hardcodes `[A-Da-d]`. On a benchmark whose
# options run wider — MedXpertQA answers A-J, Medbullets op5 answers A-E — a
# correct "E" matches nothing, falls through `compare_answers`'s default of "A",
# and is scored CORRECT on every item whose gold happens to be A. The parse-failure
# diagnostic cannot catch it either: the answer parsed, just to the wrong letter.
# Both patterns below are the vendored ones with the character class parameterised,
# so a benchmark declaring choice_letters="ABCD" is scored bit-for-bit as before.
_CHOICE_KEYWORDS = (
    r"(?:final\s*(?:choice|answer)|correct\s*(?:choice|option|answer)|choice|option|answer)"
)
_FINAL_LINE_PATTERN = re.compile(
    r"Final\s*(?:Choice|Answer)\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE
)


# The two extraction patterns for one option range, compiled once per range.
def _choice_patterns(letters: str) -> tuple[re.Pattern, re.Pattern]:
    letter_class = f"[{letters}{letters.lower()}]"
    return (
        re.compile(
            rf"{_CHOICE_KEYWORDS}\s*[:\-]?\s*[\(\[]?\s*({letter_class})\b",
            re.IGNORECASE,
        ),
        re.compile(rf"^\s*[\(\[]?\s*({letter_class})(?:\s*[\)\]]|\s*[:\.\-]|$)"),
    )


# Find the option letter in one candidate string, or None.
def _extract_choice_core(text: str, letters: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None

    keyword_pattern, direct_pattern = _choice_patterns(letters)

    # Prefer an explicit "choice/answer/option" mention over a bare leading letter,
    # which is what the vendored extractor does and why "A. Because ..." in a
    # preamble does not win against a "Final Choice: C" further down.
    keyword_match = keyword_pattern.search(text)
    if keyword_match:
        return keyword_match.group(1).upper()

    # Accept short direct forms like "A", "(B)", "C: ...".
    direct_match = direct_pattern.match(text)
    if direct_match:
        return direct_match.group(1).upper()

    return None


# Extract the chosen option from a full response, in the harness's own order.
def extract_choice(text: str, letters: str, default: str | None = None) -> str | None:
    text = text or ""

    # Boxed first, then the last "Final Choice:" line, then the whole response —
    # the same cascade, in the same order, as answer_utils.extract_choice_answer.
    candidates = []
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        candidates.append(boxed)
    final_lines = _FINAL_LINE_PATTERN.findall(text)
    if final_lines:
        candidates.append(final_lines[-1])
    candidates.append(text.strip())

    for candidate in candidates:
        letter = _extract_choice_core(candidate, letters)
        if letter is not None:
            return letter

    # default=None makes an extraction failure observable, which is what the
    # parse-failure diagnostic needs. Scoring keeps the harness's default of "A"
    # so that accuracy stays on the published table's scale, and relies on that
    # diagnostic to say how much of the score is chance credit.
    if default is None:
        return None
    fallback = _extract_choice_core(default, letters)
    return fallback if fallback is not None else letters[0]


# Score one choice item, mirroring compare_answers' choice branch at any width.
def _compare_choice(gold: str, pred_text: str, letters: str) -> tuple[str, str, bool]:
    gold_choice = extract_choice(gold, letters, default=letters[0]) or letters[0]
    pred_choice = extract_choice(pred_text, letters, default=letters[0]) or letters[0]
    return gold_choice, pred_choice, gold_choice == pred_choice


# The outcome for one item in one rollout.
@dataclass(frozen=True)
class ItemResult:
    idx: int
    rollout_idx: int
    correct: bool
    parse_ok: bool
    truncated: bool
    api_error: bool = False
    pred_parsed: str = ""
    gold_parsed: str = ""


# True when the provider returned nothing usable, rather than a wrong answer.
# These must be counted separately: scoring an empty response as incorrect would
# blame the model for a provider failure and understate the benchmark's score.
def _is_api_error(response: Response) -> bool:
    return response.finish_reason == "error" or not response.text.strip()


# The aggregate result for one (benchmark, model, thinking mode).
@dataclass
class BenchmarkScore:
    benchmark: str
    model: str
    thinking: str
    metric_name: str
    score: float
    n: int
    num_correct: int
    parse_failure_rate: float
    truncation_rate: float
    error_rate: float
    failure_rate: float  # union of the three above, per generation
    wilson_low: float
    wilson_high: float
    items: list[ItemResult] = field(default_factory=list)


# Wilson score interval for a binomial proportion, in percentage points.
def wilson_interval(num_correct: int, n: int, z: float = Z_95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0

    # Wilson is used rather than the normal approximation because it stays
    # well-behaved near 0 and 100, which is exactly where the 90 gate sits.
    proportion = num_correct / n
    denominator = 1.0 + (z**2) / n
    center = (proportion + (z**2) / (2 * n)) / denominator
    spread = z * math.sqrt(proportion * (1 - proportion) / n + (z**2) / (4 * n**2)) / denominator

    return max(0.0, 100.0 * (center - spread)), min(100.0, 100.0 * (center + spread))


# True when the model emitted an answer in the form the extractor expects.
def _parsed_cleanly(text: str, spec: BenchmarkSpec) -> bool:
    if spec.scorer == "code":
        return bool(extract_python_code(clean_raw_output(text)).strip())

    # For choice questions, default=None makes extraction failure observable;
    # compare_answers would otherwise silently substitute "A".
    if spec.scorer == "choice":
        return extract_choice(text, spec.choice_letters, default=None) is not None

    # Math answers must be boxed. extract_pred_answer falls back to the whole
    # response, which would count an unparsed answer as parsed.
    return extract_boxed_answer(text) is not None


# Score one item by executing its generated code against the real test cases.
def _score_code_item(record: EvalRecord, response: Response, timeout_s: int) -> bool:
    meta = record.meta or {}
    eval_sample = meta.get("eval_sample")
    if not eval_sample:
        return False

    code = extract_python_code(clean_raw_output(response.text))
    result = evaluate_generated_code(code, eval_sample, timeout_s=timeout_s)
    return bool(result.get("all_passed", False))


# Score every item in one rollout for a non-judged benchmark.
def _score_rollout(
    spec: BenchmarkSpec,
    dataset_name: str,
    records: list[EvalRecord],
    responses: list[Response],
    rollout_idx: int,
) -> list[ItemResult]:
    results = []
    for record, response in zip(records, responses):
        truncated = response.finish_reason == "length"
        parse_ok = _parsed_cleanly(response.text, spec)

        # Code executes; choice questions use the widened extractor above; the rest
        # go through the vendored normalisation cascade unchanged.
        if spec.scorer == "code":
            correct = _score_code_item(record, response, spec.code_timeout_s)
            pred_parsed, gold_parsed = "", ""
        elif spec.scorer == "choice":
            gold_parsed, pred_parsed, correct = _compare_choice(
                record.gold, response.text, spec.choice_letters
            )
        else:
            gold_parsed, pred_parsed, correct, _, _ = compare_answers(
                record.gold, response.text, dataset_name
            )

        results.append(
            ItemResult(
                idx=record.idx,
                rollout_idx=rollout_idx,
                correct=bool(correct),
                parse_ok=parse_ok,
                truncated=truncated,
                api_error=_is_api_error(response),
                pred_parsed=str(pred_parsed or ""),
                gold_parsed=str(gold_parsed or ""),
            )
        )
    return results


# Score a judged benchmark by sending every item to the configured LLM judge.
def _score_judged(
    records: list[EvalRecord],
    responses: list[Response],
    rollout_idx: int,
) -> list[ItemResult]:
    # Imported late so load_dotenv has already populated the judge's env vars.
    from inference_utils.llm_judge import judge_samples

    payloads = [
        {
            "question": record.question,
            "gold_answer": record.gold,
            "pred_answer": extract_boxed_answer(response.text) or response.text[-500:],
            "raw_output": response.text,
        }
        for record, response in zip(records, responses)
    ]
    verdicts = judge_samples(payloads)

    return [
        ItemResult(
            idx=record.idx,
            rollout_idx=rollout_idx,
            correct=bool(verdict),
            parse_ok=True,  # open-ended answers have no fixed format to parse
            truncated=response.finish_reason == "length",
            api_error=_is_api_error(response),
        )
        for record, response, verdict in zip(records, responses, verdicts)
    ]


# Score one benchmark across all its rollouts.
def score_benchmark(
    spec: BenchmarkSpec,
    dataset_name: str,
    records: list[EvalRecord],
    responses_by_rollout: list[list[Response]],
    model_key: str,
    thinking: str,
) -> BenchmarkScore:
    all_items: list[ItemResult] = []
    for rollout_idx, responses in enumerate(responses_by_rollout):
        if spec.scorer == "judge":
            all_items.extend(_score_judged(records, responses, rollout_idx))
        else:
            all_items.extend(
                _score_rollout(spec, dataset_name, records, responses, rollout_idx)
            )

    n = len(records)
    total_generations = len(all_items) or 1

    # With one rollout the metric is plain accuracy. With several it depends on the
    # spec: pass@k counts an item correct if ANY rollout solved it, which is the
    # paper's protocol and is also what makes a benchmark look saturated when it is
    # not — ten attempts at a problem the model solves 68% of the time is close to a
    # guarantee. avg@k takes the mean of independent attempts instead and inflates
    # nothing.
    if spec.rollouts == 1:
        metric_name = "accuracy"
        num_correct = sum(1 for item in all_items if item.correct)
        score = 100.0 * num_correct / n if n else 0.0
    elif spec.aggregation == AGGREGATION_MEAN:
        metric_name = f"avg@{spec.rollouts}"
        num_correct = sum(1 for item in all_items if item.correct)
        score = 100.0 * num_correct / total_generations
        # The interval is taken at item scale, not generation scale. Rollouts of the
        # same item are not independent draws, so treating all n*k generations as a
        # binomial would report an interval roughly sqrt(k) too narrow; item
        # sampling is what dominates the uncertainty at n in the thousands anyway.
        num_correct = round(score * n / 100.0)
    else:
        metric_name = f"pass@{spec.rollouts}"
        num_correct = len({item.idx for item in all_items if item.correct})
        score = 100.0 * num_correct / n if n else 0.0

    wilson_low, wilson_high = wilson_interval(num_correct, n)

    return BenchmarkScore(
        benchmark=spec.key,
        model=model_key,
        thinking=thinking,
        metric_name=metric_name,
        score=score,
        n=n,
        num_correct=num_correct,
        parse_failure_rate=sum(1 for i in all_items if not i.parse_ok) / total_generations,
        truncation_rate=sum(1 for i in all_items if i.truncated) / total_generations,
        error_rate=sum(1 for i in all_items if i.api_error) / total_generations,
        # The union, not the sum: the three failure modes overlap heavily, since a
        # truncated response usually also fails to parse. Summing them would
        # double-count and overstate how much headroom is unaccounted for.
        failure_rate=sum(
            1 for i in all_items if i.truncated or not i.parse_ok or i.api_error
        )
        / total_generations,
        wilson_low=wilson_low,
        wilson_high=wilson_high,
        items=all_items,
    )
