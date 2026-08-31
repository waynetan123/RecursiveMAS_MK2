"""Aggregate screening results into the Step 1 deliverables.

Reads whatever `run_screen.py` has written and produces:

- headroom.md    the locked benchmark list; Steps 2 and 8 read it to know what to run
- exclusions.md  a written note per exclusion, required by Step 1's success criteria
- manifest.json  models, pinned providers, seeds and realised cost, for reproducibility

This module never calls the API. Run it as often as you like.

    python -m screening.report
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from screening.config import (
    BENCHMARKS,
    INFEASIBLE_CANDIDATES,
    MAX_TRUSTED_ERROR_RATE,
    MAX_TRUSTED_PARSE_FAILURE_RATE,
    MAX_TRUSTED_TRUNCATION_RATE,
    MIN_SURVIVING_BENCHMARKS,
    DEFAULT_JUDGE_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    MODELS,
    OUT_OF_SCOPE_BENCHMARKS,
    SATURATION_THRESHOLD,
    BENCHMARKS_DIR,
    DEFAULT_SEED,
)

# Decisions a benchmark can receive from the gate.
RETAIN = "RETAIN"
RETIRE = "RETIRE"
UNDECIDED = "UNDECIDED"


# One benchmark's screening outcome across both models and both thinking modes.
@dataclass
class BenchmarkVerdict:
    benchmark: str
    scores_off: dict[str, float]  # model key -> score
    scores_on: dict[str, float]
    best_off: float
    best_off_model: str
    wilson_low: float
    wilson_high: float
    headroom: float
    parse_failure_rate: float
    truncation_rate: float
    error_rate: float
    metric_name: str
    n: int
    decision: str
    reason: str


# Read every summary line the runner has written.
def load_summaries(results_path: Path) -> list[dict]:
    if not results_path.exists():
        raise FileNotFoundError(
            f"No results at {results_path}. Run `python -m screening.run_screen --all` first."
        )

    summaries = []
    with results_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") == "summary":
                summaries.append(row)

    # A cell re-run appends a fresh summary, so keep only the newest of each.
    latest: dict[tuple[str, str, str], dict] = {}
    for row in summaries:
        latest[(row["benchmark"], row["model"], row["thinking"])] = row
    return list(latest.values())


# Apply the gate to one benchmark's summaries.
def build_verdict(benchmark: str, rows: list[dict]) -> BenchmarkVerdict:
    scores_off = {r["model"]: r["score"] for r in rows if r["thinking"] == "off"}
    scores_on = {r["model"]: r["score"] for r in rows if r["thinking"] == "on"}

    # The gate reads the strongest backbone with thinking disabled.
    best_off_model = max(scores_off, key=lambda m: scores_off[m]) if scores_off else ""
    best_off = scores_off.get(best_off_model, 0.0)
    gating_row = next(
        (r for r in rows if r["thinking"] == "off" and r["model"] == best_off_model), rows[0]
    )

    # Diagnostics must come from the thinking-OFF cells only, because those are the
    # cells the gate reads. Taking the max across both modes would let a thinking-ON
    # run — where reasoning tokens routinely exhaust the cap — invalidate a
    # thinking-OFF decision that was never truncated at all.
    off_rows = [r for r in rows if r["thinking"] == "off"] or rows
    parse_failure_rate = max(r["parse_failure_rate"] for r in off_rows)
    truncation_rate = max(r["truncation_rate"] for r in off_rows)
    error_rate = max(r.get("error_rate", 0.0) for r in off_rows)

    decision, reason = _decide(
        scorer=BENCHMARKS[benchmark].scorer,
        best_off=best_off,
        wilson_low=gating_row["wilson_low"],
        wilson_high=gating_row["wilson_high"],
        n=gating_row["n"],
        parse_failure_rate=parse_failure_rate,
        truncation_rate=truncation_rate,
        error_rate=error_rate,
    )

    return BenchmarkVerdict(
        benchmark=benchmark,
        scores_off=scores_off,
        scores_on=scores_on,
        best_off=best_off,
        best_off_model=best_off_model,
        wilson_low=gating_row["wilson_low"],
        wilson_high=gating_row["wilson_high"],
        headroom=100.0 - best_off,
        parse_failure_rate=parse_failure_rate,
        truncation_rate=truncation_rate,
        error_rate=error_rate,
        metric_name=gating_row["metric_name"],
        n=gating_row["n"],
        decision=decision,
        reason=reason,
    )


# Explain what a parse failure means for each kind of scorer.
PARSE_FAILURE_EFFECT = {
    "choice": (
        "unparsed items are scored as 'A' rather than as wrong, so the accuracy is "
        "inflated by chance credit"
    ),
    "math": (
        "unparsed items produced no boxed answer, so the score reflects formatting "
        "compliance as much as reasoning"
    ),
    "code": (
        "no code block could be extracted from these responses, so they were scored as "
        "failures without ever being executed"
    ),
    "judge": "the judge received no parsed answer and fell back to raw output",
}


# Decide retain / retire / undecided, accounting for the scoring traps.
#
# The confidence interval decides direction, symmetrically: a benchmark is only
# retired when the whole interval sits at or above the threshold, and only retained
# when the whole interval sits below it. An interval that straddles the threshold is
# undecided regardless of where the point estimate falls — that is precisely the
# case where the sample is too small to support either decision.
def _decide(
    scorer: str,
    best_off: float,
    wilson_low: float,
    wilson_high: float,
    n: int,
    parse_failure_rate: float,
    truncation_rate: float,
    error_rate: float,
) -> tuple[str, str]:
    # Provider failures are not model failures. Empty responses scored as wrong
    # depress the score and would make a saturated benchmark look retainable.
    if error_rate > MAX_TRUSTED_ERROR_RATE:
        return UNDECIDED, (
            f"{error_rate:.1%} of thinking-OFF generations returned a provider error or empty "
            "content, which is scored as incorrect and so understates the benchmark. "
            "Re-run the affected cells before deciding."
        )

    # Truncation cuts an answer off before it is emitted, so the generation is scored
    # wrong and the observed score is a LOWER bound on the true one. The bias runs one
    # way, which decides where the guard belongs: a lower bound already above the gate
    # still justifies retiring, because the true score can only be higher. A lower
    # bound below the gate cannot justify retaining, because the true score may well
    # sit above it — that is the case where a saturated benchmark would slip through
    # and cost weeks of Step 8 compute.
    if truncation_rate > MAX_TRUSTED_TRUNCATION_RATE and wilson_low < SATURATION_THRESHOLD:
        return UNDECIDED, (
            f"{truncation_rate:.0%} of thinking-OFF generations hit the token cap, so the "
            f"observed {best_off:.1f} is a lower bound rather than a measurement, and the "
            f"interval reaches down to {wilson_low:.1f}. The true score may sit above the "
            f"{SATURATION_THRESHOLD:.0f} threshold. Raise the cap and re-run before retaining."
        )

    # A high parse-failure rate means the score is measuring output format, not ability.
    if parse_failure_rate > MAX_TRUSTED_PARSE_FAILURE_RATE:
        effect = PARSE_FAILURE_EFFECT.get(scorer, "the score cannot be interpreted")
        return UNDECIDED, (
            f"Parse-failure rate {parse_failure_rate:.1%} exceeds "
            f"{MAX_TRUSTED_PARSE_FAILURE_RATE:.0%}: {effect}. The score cannot support a "
            "decision until the prompt elicits the expected answer format."
        )

    # Confidently saturated: even the bottom of the interval leaves under 10 points.
    if wilson_low >= SATURATION_THRESHOLD:
        return RETIRE, (
            f"Strongest backbone scores {best_off:.1f} (95% CI [{wilson_low:.1f}, "
            f"{wilson_high:.1f}], n={n}) with thinking disabled, under the paper's own harness "
            f"and decoding settings. Headroom {100.0 - best_off:.1f} points, below the 10-point "
            "threshold, and the whole interval sits above it. No recursion effect could be "
            "resolved above this floor."
        )

    # Confidently unsaturated: the whole interval leaves room above the backbone.
    if wilson_high < SATURATION_THRESHOLD:
        return RETAIN, (
            f"Headroom {100.0 - best_off:.1f} points above the strongest single backbone "
            f"(95% CI [{wilson_low:.1f}, {wilson_high:.1f}], n={n})."
        )

    # The interval straddles the threshold, so neither decision is supported yet.
    return UNDECIDED, (
        f"Point estimate {best_off:.1f} with 95% CI [{wilson_low:.1f}, {wilson_high:.1f}] at "
        f"n={n}, which straddles the {SATURATION_THRESHOLD:.0f} threshold. The sample is too "
        "small to decide; run the full set."
    )


# Render the locked benchmark list.
def write_headroom(verdicts: list[BenchmarkVerdict], path: Path) -> None:
    model_keys = sorted(MODELS)
    header = (
        "| Benchmark | Metric | n | "
        + " | ".join(f"{k} OFF" for k in model_keys)
        + " | max OFF | Wilson 95% | max ON | Headroom | Parse-fail | Trunc | Err | Decision |"
    )
    # Ten fixed columns plus one per model, or the table will not render.
    divider = "|" + "---|" * (11 + len(model_keys))

    lines = [
        f"# Benchmark screening — {date.today().isoformat()}",
        "",
        f"Gate: RETIRE when the strongest backbone scores >= {SATURATION_THRESHOLD:.0f} with "
        "thinking disabled (fewer than 10 points of headroom).",
        "",
        header,
        divider,
    ]

    for verdict in sorted(verdicts, key=lambda v: v.best_off, reverse=True):
        per_model = " | ".join(
            f"{verdict.scores_off.get(k, float('nan')):.1f}" for k in model_keys
        )
        best_on = max(verdict.scores_on.values()) if verdict.scores_on else float("nan")

        # Paper benchmarks that survive are the continuity anchors Step 1 requires.
        label = verdict.decision
        if verdict.decision == RETAIN and BENCHMARKS[verdict.benchmark].is_paper_benchmark:
            label = "RETAIN (anchor)"

        # The interval is asymmetric, so both bounds come from the score itself
        # rather than being mirrored around the point estimate.
        lines.append(
            f"| {verdict.benchmark} | {verdict.metric_name} | {verdict.n} | {per_model} "
            f"| {verdict.best_off:.1f} | [{verdict.wilson_low:.1f}, {verdict.wilson_high:.1f}] "
            f"| {best_on:.1f} | {verdict.headroom:.1f} "
            f"| {verdict.parse_failure_rate:.1%} | {verdict.truncation_rate:.1%} "
            f"| {verdict.error_rate:.1%} | {label} |"
        )

    retained = [v for v in verdicts if v.decision == RETAIN]
    undecided = [v for v in verdicts if v.decision == UNDECIDED]
    anchors = [v for v in retained if BENCHMARKS[v.benchmark].is_paper_benchmark]

    lines += [
        "",
        f"**Retained:** {len(retained)} — "
        + (
            "stop rule not triggered."
            if len(retained) >= MIN_SURVIVING_BENCHMARKS
            else f"**STOP RULE TRIGGERED** (fewer than {MIN_SURVIVING_BENCHMARKS} survivors). "
            "Pause the project; benchmark construction becomes a separately scoped task."
        ),
        f"**Anchors carried from the paper:** {len(anchors)} — "
        + (", ".join(v.benchmark for v in anchors) if anchors else "none, which fails Step 1's continuity requirement."),
    ]
    if undecided:
        lines.append(
            f"**Undecided:** {len(undecided)} — "
            + ", ".join(v.benchmark for v in undecided)
            + ". Not locked; see exclusions.md for what each needs."
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Render the written note explaining every exclusion.
def write_exclusions(verdicts: list[BenchmarkVerdict], path: Path) -> None:
    lines = [
        f"# Benchmark exclusions — {date.today().isoformat()}",
        "",
        "Required by Step 1's success criteria. Recorded at screening time, before any",
        "training, so that no exclusion rests on a retrospective justification.",
        "",
    ]

    # Benchmarks that failed the gate empirically.
    for verdict in sorted(verdicts, key=lambda v: v.benchmark):
        if verdict.decision == RETAIN:
            continue
        status = "RETIRED (saturated)" if verdict.decision == RETIRE else "UNDECIDED (not locked)"
        lines += [
            f"## {verdict.benchmark} — {status}",
            "",
            verdict.reason,
            "",
            f"Scores with thinking disabled: "
            + ", ".join(f"{m}={s:.1f}" for m, s in sorted(verdict.scores_off.items()))
            + f". Metric {verdict.metric_name} over n={verdict.n}.",
            "",
        ]

    # Paper benchmarks the architecture under study cannot run at all.
    lines += ["## Paper benchmarks out of architectural scope", ""]
    for name, reason in OUT_OF_SCOPE_BENCHMARKS.items():
        lines += [f"### {name} — NOT SCREENED (out of scope)", "", reason, ""]

    # Candidates ruled out before screening, for reasons that are not scores.
    lines += ["## Replacement candidates not screened", ""]
    for name, reason in INFEASIBLE_CANDIDATES.items():
        lines += [f"### {name} — NOT SCREENED", "", reason, ""]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Record what produced these numbers, so the locked list is reproducible.
def write_manifest(summaries: list[dict], results_path: Path, path: Path) -> None:
    providers: dict[str, set[str]] = defaultdict(set)
    served_models: dict[str, set[str]] = defaultdict(set)
    cost = 0.0

    # Providers and cost come from the per-sample rows, not the summaries.
    with results_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "summary":
                continue
            providers[row["model"]].add(row.get("provider", "unknown"))
            if row.get("served_model"):
                served_models[row["model"]].add(row["served_model"])
            cost += float(row.get("cost") or 0.0)

    manifest = {
        "generated": date.today().isoformat(),
        "seed": DEFAULT_SEED,
        "saturation_threshold": SATURATION_THRESHOLD,
        # The judge decides every HLE score, so it belongs in the reproducibility
        # record. Resolved the same way run_screen resolves it.
        "judge": {
            "model": os.environ.get("API_MODEL") or DEFAULT_JUDGE_MODEL,
            "base_url": os.environ.get("API_BASE_URL") or DEFAULT_JUDGE_BASE_URL,
        },
        "models": {
            key: {
                "model_id": spec.model_id,
                "provider_pin": spec.provider,
                "providers_observed": sorted(providers.get(key, [])),
                # What OpenRouter reported serving, so the claim is checkable
                # after the fact rather than only asserted during the run.
                "models_served": sorted(served_models.get(key, [])),
            }
            for key, spec in MODELS.items()
        },
        "benchmarks": {
            key: {
                "loader_arg": spec.loader_arg,
                "split": spec.split,
                "scorer": spec.scorer,
                "max_tokens": spec.max_tokens,
                "temperature": spec.temperature,
                "rollouts": spec.rollouts,
                "notes": spec.notes,
            }
            for key, spec in BENCHMARKS.items()
        },
        "cells_completed": len(summaries),
        "realised_cost_usd": round(cost, 4),
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    results_path = BENCHMARKS_DIR / "screening_results.jsonl"
    summaries = load_summaries(results_path)

    # Group by benchmark so each gets one verdict across all four of its cells.
    by_benchmark: dict[str, list[dict]] = defaultdict(list)
    for row in summaries:
        by_benchmark[row["benchmark"]].append(row)

    verdicts = [build_verdict(name, rows) for name, rows in by_benchmark.items()]

    write_headroom(verdicts, BENCHMARKS_DIR / "headroom.md")
    write_exclusions(verdicts, BENCHMARKS_DIR / "exclusions.md")
    write_manifest(summaries, results_path, BENCHMARKS_DIR / "manifest.json")

    retained = sum(1 for v in verdicts if v.decision == RETAIN)
    print(f"Wrote headroom.md, exclusions.md, manifest.json to {BENCHMARKS_DIR}")
    print(f"Retained {retained} of {len(verdicts)} screened benchmarks.")

    if retained < MIN_SURVIVING_BENCHMARKS:
        print(
            f"\nSTOP RULE: fewer than {MIN_SURVIVING_BENCHMARKS} benchmarks survived. "
            "Pause the project per Step 1."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
