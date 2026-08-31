"""Screening runner: generates completions and scores them.

This is the entry point. `report.py` only aggregates what this writes to disk.

Generation and scoring are separable on purpose. Generation is slow, network-bound
and expensive, so it is cached; scoring is cheap, local, and the half you will
iterate on. `--rescore-only` re-scores cached responses without spending anything.

    python -m screening.run_screen --canary
    python -m screening.run_screen --benchmark gpqa --model qwen --thinking off
    python -m screening.run_screen --all
    python -m screening.run_screen --all --rescore-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from screening.config import (
    BENCHMARKS,
    MODELS,
    THINKING_MODES,
    BenchmarkSpec,
    ModelSpec,
    ScreenConfig,
    load_config,
)
from screening.datasets import EvalRecord, load_benchmark
from screening.openrouter import OpenRouterClient, Response, assert_thinking_disabled
from screening.prompts import build_messages
from screening.score import BenchmarkScore, score_benchmark

# Items generated and checked before the rest of a benchmark proceeds.
THINKING_PROBE_SIZE = 20

# --canary limits: enough to exercise every path, cheap enough to run freely.
# Two rollouts still exercise the pass@k aggregation; a 2,000-token cap keeps AIME
# from dominating the bill and incidentally exercises the truncation column too.
CANARY_SAMPLES = 10
CANARY_ROLLOUTS = 2
CANARY_MAX_TOKENS = 2000

# Rough average output tokens per item, for the --dry-run estimate only.
DRY_RUN_OUTPUT_FRACTION = {"off": 0.35, "on": 0.85}
DRY_RUN_PROMPT_TOKENS = 500

# Judge cost per graded item: the prompt carries the question plus a 4,000-char
# raw output, and llm_judge caps its own reply at 256 tokens.
JUDGE_PROMPT_TOKENS = 1500
JUDGE_OUTPUT_TOKENS = 256
JUDGE_PRICE_IN_PER_M = 2.00
JUDGE_PRICE_OUT_PER_M = 6.00


# Generate one item.
async def _generate_one(
    client: OpenRouterClient,
    model: ModelSpec,
    record: EvalRecord,
    spec: BenchmarkSpec,
    seed: int,
    thinking: dict,
    rollout_idx: int,
) -> Response:
    return await client.complete(
        model=model,
        messages=build_messages(record, spec),
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        seed=seed,
        thinking=thinking,
        rollout_idx=rollout_idx,
    )


# Generate every item for one rollout, probing the thinking mode first.
async def _generate_rollout(
    client: OpenRouterClient,
    model: ModelSpec,
    records: list[EvalRecord],
    spec: BenchmarkSpec,
    seed: int,
    thinking_key: str,
    rollout_idx: int,
    verify_thinking: bool,
) -> list[Response]:
    thinking = THINKING_MODES[thinking_key]

    async def generate(record: EvalRecord) -> Response:
        return await _generate_one(
            client, model, record, spec, seed, thinking, rollout_idx
        )

    # Probe a small prefix first so a provider ignoring enabled=False is caught
    # before the full benchmark is paid for.
    if verify_thinking and thinking_key == "off":
        probe = records[:THINKING_PROBE_SIZE]
        probe_responses = await asyncio.gather(*(generate(r) for r in probe))
        assert_thinking_disabled(list(probe_responses), spec.key)

        remaining = await asyncio.gather(*(generate(r) for r in records[THINKING_PROBE_SIZE:]))
        return list(probe_responses) + list(remaining)

    return list(await asyncio.gather(*(generate(r) for r in records)))


# Run one (benchmark, model, thinking mode) cell end to end.
async def run_cell(
    client: OpenRouterClient,
    config: ScreenConfig,
    spec: BenchmarkSpec,
    model: ModelSpec,
    thinking_key: str,
    num_samples: int,
) -> BenchmarkScore:
    dataset_name, records = load_benchmark(spec, config, num_samples_override=num_samples)
    print(
        f"[run] {spec.key:<10} model={model.key:<6} thinking={thinking_key:<3} "
        f"n={len(records)} rollouts={spec.rollouts}"
    )

    # Each rollout gets its own seed so pass@k samples differ from one another.
    responses_by_rollout = []
    for rollout_idx in range(spec.rollouts):
        responses = await _generate_rollout(
            client=client,
            model=model,
            records=records,
            spec=spec,
            seed=config.seed + rollout_idx,
            thinking_key=thinking_key,
            rollout_idx=rollout_idx,
            verify_thinking=(rollout_idx == 0),
        )
        responses_by_rollout.append(responses)

    client.assert_routing(model)

    score = score_benchmark(
        spec=spec,
        dataset_name=dataset_name,
        records=records,
        responses_by_rollout=responses_by_rollout,
        model_key=model.key,
        thinking=thinking_key,
    )

    _write_results(config, spec, model, thinking_key, records, responses_by_rollout, score)
    return score


# Append per-sample records and the cell summary to the audit trail.
def _write_results(
    config: ScreenConfig,
    spec: BenchmarkSpec,
    model: ModelSpec,
    thinking_key: str,
    records: list[EvalRecord],
    responses_by_rollout: list[list[Response]],
    score: BenchmarkScore,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / "screening_results.jsonl"

    with path.open("a", encoding="utf-8") as handle:
        for rollout_idx, responses in enumerate(responses_by_rollout):
            for record, response in zip(records, responses):
                handle.write(
                    json.dumps(
                        {
                            "benchmark": spec.key,
                            "model": model.key,
                            "thinking": thinking_key,
                            "rollout_idx": rollout_idx,
                            "sample_idx": record.idx,
                            "question": record.question,
                            "gold_answer": record.gold,
                            "response": response.text,
                            "finish_reason": response.finish_reason,
                            "provider": response.provider,
                            "served_model": response.model,
                            "prompt_tokens": response.prompt_tokens,
                            "completion_tokens": response.completion_tokens,
                            "reasoning_tokens": response.reasoning_tokens,
                            "cost": response.cost,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        # The summary line is what report.py reads; items stay in the per-sample rows.
        summary = asdict(score)
        summary.pop("items", None)
        summary["type"] = "summary"
        summary["timestamp"] = time.time()
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")


# Resolve how many items a cell will actually run.
def _effective_n(spec: BenchmarkSpec, num_samples: int) -> int:
    # A CLI cap wins, then the spec's own subsample, then the documented full size.
    if num_samples > 0:
        return min(num_samples, spec.full_set_size or num_samples)
    if spec.num_samples > 0:
        return spec.num_samples
    return spec.full_set_size


# Estimate the cost of a set of cells without calling the API.
def _dry_run(cells: list[tuple[BenchmarkSpec, ModelSpec, str]], num_samples: int) -> None:
    generation_total = 0.0
    judge_total = 0.0
    print(f"{'benchmark':<11}{'model':<8}{'think':<7}{'gens':>8}{'gen $':>9}{'judge $':>9}")

    for spec, model, thinking_key in cells:
        generations = _effective_n(spec, num_samples) * spec.rollouts

        output_tokens = generations * spec.max_tokens * DRY_RUN_OUTPUT_FRACTION[thinking_key]
        input_tokens = generations * DRY_RUN_PROMPT_TOKENS
        generation_cost = (
            output_tokens / 1e6 * model.price_out_per_m
            + input_tokens / 1e6 * model.price_in_per_m
        )

        # Open-ended benchmarks pay again to have every generation graded.
        judge_cost = 0.0
        if spec.scorer == "judge":
            judge_cost = (
                generations * JUDGE_PROMPT_TOKENS / 1e6 * JUDGE_PRICE_IN_PER_M
                + generations * JUDGE_OUTPUT_TOKENS / 1e6 * JUDGE_PRICE_OUT_PER_M
            )

        generation_total += generation_cost
        judge_total += judge_cost
        print(
            f"{spec.key:<11}{model.key:<8}{thinking_key:<7}{generations:>8}"
            f"{generation_cost:>9.2f}{judge_cost:>9.2f}"
        )

    print(f"\nGeneration: ${generation_total:.2f}")
    print(f"LLM judge:  ${judge_total:.2f}")
    print(f"Total:      ${generation_total + judge_total:.2f}")
    print("\nRough: assumes average output is a fixed fraction of the cap, which is the")
    print("largest source of error. Realised cost comes from each response's reported")
    print("usage and is written to manifest.json.")


# Shrink a spec for --canary, so the plumbing runs without a real bill.
def _to_canary(spec: BenchmarkSpec) -> BenchmarkSpec:
    return replace(
        spec,
        rollouts=min(spec.rollouts, CANARY_ROLLOUTS),
        max_tokens=min(spec.max_tokens, CANARY_MAX_TOKENS),
    )


# Expand the CLI selection into the list of cells to run.
def _select_cells(args: argparse.Namespace) -> list[tuple[BenchmarkSpec, ModelSpec, str]]:
    benchmarks = list(BENCHMARKS.values()) if not args.benchmark else [BENCHMARKS[args.benchmark]]
    models = list(MODELS.values()) if not args.model else [MODELS[args.model]]
    thinking_modes = list(THINKING_MODES) if not args.thinking else [args.thinking]

    if args.canary:
        benchmarks = [_to_canary(spec) for spec in benchmarks]

    return [(b, m, t) for b in benchmarks for m in models for t in thinking_modes]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RecursiveMAS MK2 Step 1 benchmark screening.")
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), help="Default: all.")
    parser.add_argument("--model", choices=sorted(MODELS), help="Default: all.")
    parser.add_argument("--thinking", choices=sorted(THINKING_MODES), help="Default: both.")
    parser.add_argument("--all", action="store_true", help="Run the full sweep.")
    parser.add_argument(
        "--canary",
        action="store_true",
        help=(
            f"Every cell at {CANARY_SAMPLES} samples, {CANARY_ROLLOUTS} rollouts and a "
            f"{CANARY_MAX_TOKENS}-token cap, to exercise every path cheaply."
        ),
    )
    parser.add_argument("--num-samples", type=int, default=-1, help="Cap items per benchmark.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Print a cost estimate and exit.")
    parser.add_argument("--no-cache", action="store_true", help="Force regeneration.")
    parser.add_argument(
        "--rescore-only",
        action="store_true",
        help="Re-score cached responses without calling the API.",
    )
    return parser


async def main_async(args: argparse.Namespace) -> int:
    cells = _select_cells(args)
    num_samples = CANARY_SAMPLES if args.canary else args.num_samples

    if args.dry_run:
        _dry_run(cells, num_samples)
        return 0

    config = load_config(seed=args.seed)

    scores: list[BenchmarkScore] = []
    async with OpenRouterClient(
        api_key=config.openrouter_api_key,
        cache_dir=config.cache_dir,
        concurrency=args.concurrency,
        use_cache=not args.no_cache,
        require_cache=args.rescore_only,
    ) as client:
        for spec, model, thinking_key in cells:
            score = await run_cell(
                client, config, spec, model, thinking_key, num_samples
            )
            scores.append(score)
            print(
                f"       -> {score.metric_name}={score.score:.2f}% "
                f"[{score.wilson_low:.1f}, {score.wilson_high:.1f}] "
                f"parse_fail={score.parse_failure_rate:.1%} "
                f"trunc={score.truncation_rate:.1%}"
            )

    print(f"\nCompleted {len(scores)} cells. Run `python -m screening.report` to aggregate.")
    return 0


def main() -> int:
    args = build_parser().parse_args()

    # Require an explicit selection so a bare invocation cannot start a $130 sweep.
    if not (args.all or args.canary or args.benchmark or args.model or args.thinking):
        build_parser().error("Select --all, --canary, or a --benchmark / --model / --thinking.")

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
