# Benchmark exclusions — 2026-09-06

Required by Step 1's success criteria. Recorded at screening time, before any
training, so that no exclusion rests on a retrospective justification.

## aime25 — UNDECIDED (not locked)

Point estimate 96.7 with 95% CI [83.3, 99.4] at n=30, which straddles the 90 threshold. The sample is too small to decide; run the full set.

Scores with thinking disabled: gemma=96.7, qwen=90.0. Metric pass@10 over n=30.

## aime26 — UNDECIDED (not locked)

Point estimate 90.0 with 95% CI [74.4, 96.5] at n=30, which straddles the 90 threshold. The sample is too small to decide; run the full set.

Scores with thinking disabled: gemma=90.0, qwen=90.0. Metric pass@10 over n=30.

## math500 — RETIRED (saturated)

Strongest backbone scores 96.2 (95% CI [94.1, 97.6], n=500) with thinking disabled, under the paper's own harness and decoding settings. Headroom 3.8 points, below the 10-point threshold, and the whole interval sits above it. No recursion effect could be resolved above this floor.

Scores with thinking disabled: gemma=96.2, qwen=83.4. Metric accuracy over n=500.

## medqa — UNDECIDED (not locked)

Point estimate 92.3 with 95% CI [88.8, 94.8] at n=300, which straddles the 90 threshold. The sample is too small to decide; run the full set.

Scores with thinking disabled: gemma=92.3, qwen=90.0. Metric accuracy over n=300.

## Controls — screened but not gated

### beyondaime — CONTROL (has range)

Not gated: n=100 gives +/-9.8, which compares signs and rough magnitudes and nothing finer. Three jobs. (1) Contamination: Omni-MATH's problems went public in October 2024 and both backbones postdate that, so a gain there could be retrieval; BeyondAIME is the clean twin, same domain and answer format and difficulty band, every problem manually rewritten. (2) Judge independence: positive-integer answers go through compare_answers with no judge in the loop, the only maths benchmark in the set with that property. (3) Threshold calibration for omni_math's difficulty cut, whose labels are self-reported where BeyondAIME's band is externally anchored. CC0.

Scores with thinking disabled: gemma=59.0, qwen=55.0. Metric accuracy over n=100.

### medbullets_op4 — CONTROL (has range)

Not gated: Gemma's canary already scored 90.0, the same neighbourhood as the medqa it would replace, and the questions come from a public X account posting between April 2022 and December 2023 and so are heavily exposed to pretraining. It earns its place as the baseline arm of a paired probe, where both objections cancel: a memorised question is memorised in op4 and op5 alike and contributes nothing to the difference between them. Options are re-shuffled by stable hash because the shipped gold distribution is skewed (A 87, D 68 of 308), and a position effect must not be readable as distractor sensitivity.

Scores with thinking disabled: gemma=84.7, qwen=84.1. Metric accuracy over n=308.

### medbullets_op5 — CONTROL (has range)

The same 308 questions as medbullets_op4 with one extra distractor and nothing else changed, so the op4->op5 drop isolates one variable: what an additional attractive wrong option costs. Rejecting one is the Critic's job, giving a falsifiable prediction independent of headline accuracy — RecursiveMAS's drop should be smaller than the single-agent baseline's, and should shrink monotonically across depths 1-2-3. The comparison is within one scale across Step 8 rows, never 4B against 31B, since larger models are less distractible regardless. Paired on identical items, so item difficulty cancels and the +/-4.3 marginal interval does not apply; simulated power at 308 items and five runs detects a 3-point change 98% of the time and a 2-point change 68%. Screening it now asks only whether the instrument has any range: a baseline drop under ~2 points means there is none and the probe should be dropped.

Scores with thinking disabled: gemma=81.2, qwen=78.9. Metric accuracy over n=308.

## Paper benchmarks out of architectural scope

### HotpotQA — NOT SCREENED (out of scope)

Search-QA, supported only under the Deliberation style. run.py:validate_style_dataset rejects it for every other style, because without tools it falls through to string-match scoring and reports a number that does not reflect the task. Sequential Style is Planner/Critic/Solver with no Tool-Caller agent, so it cannot search, and the paper reports no Sequential-Scaled figure for this benchmark to compare against.

### Bamboogle — NOT SCREENED (out of scope)

Search-QA, supported only under the Deliberation style, for the same reasons as HotpotQA. RELEASE_RECOMMENDED_SETTINGS sanctions ('deliberation', 'bamboogle') and no sequential pairing.

## Replacement candidates not screened

### FrontierMath — NOT SCREENED

12 of 338 problems are public; the remainder is held privately by Epoch AI, with OpenAI holding exclusive access to a subset. Cannot be run locally at any n.

### Agents' Last Exam — NOT SCREENED

153 computer-use agent tasks with reference outputs gated behind manual approval. The Sequential Style emits one text answer with no computer-use loop, and the average full pass rate is below 1%, so the benchmark has a floor problem as well as a scaffold problem.

### SWE-Bench Verified — NOT SCREENED

Requires Docker, repository checkouts and an agentic edit/test loop that the Sequential Style cannot express. Screening a bare model would measure the scaffold rather than the backbone.

