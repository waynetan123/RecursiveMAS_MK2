
**Reference paper:** _Recursive Multi-Agent Systems_ (arXiv:2604.25917), Yang, Zou et al. — UIUC, Stanford, NVIDIA, MIT **Compute:** one 8×H100-80GB node **Estimated duration:** 9–13 weeks (primary arm), plus 2–3 weeks for the optional second arm **Estimated compute:** 425–795 node-hours (≈3,400–6,400 H100-hours)

---

## 1. Overview

### What RecursiveMAS is

When several language models work together as a team, they normally communicate in writing. One agent produces a paragraph, the next reads it, and so on. This is wasteful in two ways: converting internal representations into words discards information, and every conversion costs time and tokens.

RecursiveMAS removes the writing step. Each agent passes its raw internal representation — the vector it produces just before it would ordinarily choose a word — directly to the next agent. A small adapter called the **RecursiveLink** translates between representation spaces. The chain then loops back to the first agent and repeats for several rounds. Only in the final round does the last agent produce actual text.

The underlying models are never altered. Only the adapters are trained, which is why the method is cheap: the paper trains 13.12M parameters, 0.31% of the system, at a peak of 15.29 GB and an estimated cost of $4.27.

Training happens in two stages. The **inner loop** teaches each agent to produce a representation that its own adapter can map onto a sensible target. The **outer loop** then unrolls the whole chain across all rounds and trains the cross-agent adapters end to end, so the system learns to collaborate as a single unit.

The paper reports an average accuracy gain of 8.3% over comparable baselines, an end-to-end speed-up of 1.2×–2.4×, and a reduction in token use of 34.6%–75.6%.

### How we wish to extend it

The paper's largest sequential configuration uses three models of roughly 3–4B parameters (Gemma3-4B-it, Llama3.2-3B-Instruct, Qwen3.5-4B). Its largest model anywhere is 9B.

We will rebuild the same architecture with models roughly seven times larger, and answer two questions the original work could not.

**Question 1 — does the architecture still help at scale?**

> Does RecursiveMAS provide a benefit once the underlying models are strong enough to solve most of the tasks on their own?

Both answers are informative. If the gain persists, the architecture contributes something independent of raw model capability. If it disappears, the method is a technique for extracting more from weak models — still useful, but a narrower claim than the paper implies.

**Question 2 — how much of the benefit is recursion, and how much is model diversity?**

The paper's configurations mix model families and describe them as offering "complementary strengths". This raises an obvious objection: perhaps some of the reported gain comes from combining different models at all, rather than from recursive collaboration specifically. The paper cannot separate these, because every sequential configuration it tests is heterogeneous.

We can separate them by running two arms.

### Experimental arms

**Arm A (primary) — homogeneous.** Three instances of Gemma 4 31B Dense, distinguished only by role prompt and adapter.

|Role|Model|
|---|---|
|Planner|Gemma 4 31B Dense|
|Critic|Gemma 4 31B Dense|
|Solver|Gemma 4 31B Dense|

Because all three agents are frozen and identical, one copy of the weights serves the whole system — roughly 62 GB resident rather than the ~160 GB a mixed roster would require. Since our largest anticipated failure mode is memory exhaustion during the unrolled backward pass, this headroom is valuable. Nothing is lost by sharing, as the chain executes sequentially in any case.

Any gain measured in this arm cannot be attributed to combining model families, because there is only one. It must come from recursion and role specialisation. This is a tighter answer to Question 1 than the paper's own setup permits, and it is closer to the paper's theoretical analysis, which assumes the cross-agent projection is square.

There is precedent: the paper's own Deliberation Style pattern pairs Qwen3.5-4B with Qwen3.5-4B. Identical backbones in different roles is a configuration the authors themselves used.

**Arm B (secondary, budget permitting) — heterogeneous.**

|Role|Model|
|---|---|
|Planner|Gemma 4 31B Dense|
|Critic|Gemma 4 26B-A4B (mixture-of-experts, ~3.8B active)|
|Solver|Qwen3.8-27B|

This restores the paper's diversity claim. The difference between Arm A and Arm B answers Question 2 directly.

**Control arm (diagnostic only).** Qwen3.6-27B substituted for the Solver in Arm B. Used solely to distinguish a genuine null result from an implementation fault — see Step 3.

We scope the work to the **Sequential Style** collaboration pattern only. The three other patterns in the paper are out of scope.

### Why homogeneous first

Three reasons, in order of weight:

1. **It removes the hardest engineering task.** Qwen3.8-27B uses a running-summary attention mechanism on 48 of its 64 layers, which creates a serious gradient-flow problem described under Step 3. Gemma 4 does not. Arm A avoids the issue entirely.
2. **Tooling maturity.** Gemma 4 has been publicly available since April 2026 under a permissive licence, so training-time support is well exercised. Qwen3.8-27B is a matter of weeks old.
3. **Memory headroom**, as above.

Arm A is by a considerable margin the lowest-risk build. It should be completed and evaluated before Arm B begins.

---

## 2. Success criteria

The project succeeds if it produces a defensible answer to both research questions, **whether or not the answers favour the method**. Specifically:

1. **The pipeline is validated.** The paper's own configuration reproduces to within two accuracy points at recursion depth 3, using the published models and settings.
2. **Gradient flow is proven correct.** A finite-difference check passes across the full unrolled chain at the target scale, not merely on a short test case.
3. **The comparison is fair.** All baselines run on the same weights, the same evaluation harness, and the same decoding settings. No result rests on a comparison against an external API.
4. **The benchmarks discriminate.** Every retained benchmark has measurable headroom above the strongest single backbone. Saturated benchmarks are excluded before any compute is spent.
5. **The result is statistically meaningful.** Five independent runs per configuration. The paper reports a standard deviation of ±0.0041 on accuracy, so any claimed effect must exceed roughly half a point.
6. **Accuracy and efficiency are reported separately.** It is entirely possible for one to survive scaling and the other not to.
7. **Recursion is separated from diversity**, if Arm B is completed.

A null result — "the architectural advantage compresses at scale whilst the efficiency advantage holds" — counts as success. A result that cannot distinguish a null finding from a broken implementation counts as failure.

---

## 3. Plan at a glance

|#|Step|Duration|Compute|
|---|---|---|---|
|1|API screening; lock benchmark list|3–5 days|negligible (~$30)|
|2|Replicate at the paper's original scale|1.5–2 weeks|~25 node-hours|
|3|Port the RecursiveLink; verify gradients|1–1.5 weeks|~8 node-hours|
|4|Rewrite training data into role-specific targets|3–5 days|negligible (API)|
|5|Inner-loop training|2 days|~10 node-hours|
|6|Outer-loop training|1 week|20–30 node-hours|
|7|Train self-hosted baselines|1–2 weeks|60–120 node-hours|
|8|Evaluation under a single harness|3–4 weeks|300–600 node-hours|
|9|Second arm — heterogeneous roster (optional)|2–3 weeks|100–200 node-hours|

Steps 3 to 8 describe Arm A. Step 9 repeats the necessary subset for Arm B.

Evaluation, not training, dominates the budget. This is counter-intuitive but follows directly from the design: training touches a few thousand curated examples once, whereas evaluation covers every benchmark, at three recursion depths, across five runs, for every baseline.

---

## Step 1 — API screening and benchmark selection

### Resources needed

- API access to Gemma 4 31B and Qwen3.8-27B
- The paper's nine benchmarks: MATH500, AIME2025, AIME2026, GPQA-Diamond, MedQA, LiveCodeBench-v6, MBPP Plus, HotpotQA, Bamboogle
- Candidate replacement benchmarks: HLE, Agents' Last Exam, FrontierMath, SWE-Bench Verified
- Budget: approximately $30

### Explanation

The paper's best system reaches 88.0 on MATH500, 86.7 on AIME2026, 66.2 on GPQA-Diamond and 42.9 on LiveCodeBench. Published figures for our chosen backbones suggest a single untrained model already exceeds several of these on its own — by more than forty points on code in at least one case.

If that holds, those benchmarks cannot measure what we want to measure. There is no room above the floor for the architecture to demonstrate anything. Running them anyway would waste weeks of compute and produce an uninterpretable table.

Screen both backbones even though Arm A uses only Gemma, since Arm B needs Qwen figures and the screening is nearly free.

### Implementation

1. Run each backbone, unmodified, across all nine original benchmarks via API, using the paper's decoding settings.
2. Record the score for each model on each benchmark.
3. Compute headroom: the gap between the strongest single model and a perfect score.
4. Retire any benchmark where headroom falls below roughly ten points.
5. Screen the replacement candidates the same way; retain those with adequate headroom.
6. Keep two original benchmarks regardless, as continuity anchors permitting partial comparison with the published results. GPQA-Diamond and MedQA are the likeliest survivors.

### Success criteria

- A locked list of six to eight benchmarks, each with recorded headroom.
- At least two benchmarks carried over from the original paper.
- A written note explaining every exclusion, for the eventual write-up.

**Stop rule:** if fewer than four benchmarks survive screening, pause the project. Benchmark construction then becomes a research task in its own right and must be scoped separately.

---

## Step 2 — Replicate at the paper's original scale

### Resources needed

- The official implementation, from the project's public repository
- Gemma3-4B-it, Llama3.2-3B-Instruct, Qwen3.5-4B
- The paper's original training data sources: s1K, m1K, OpenCodeReasoning, ARPO-SFT
- Approximately 25 node-hours

### Explanation

Before changing anything, reproduce something known to work. This gives a reference implementation to compare against when the larger version misbehaves — and it will misbehave.

Without this step, a failure at 31B is ambiguous: the method might be wrong, the port might be wrong, or the training data might be wrong. With it, the search space narrows considerably. Debugging at 4B costs hours; the same fault at 31B costs days.

Note that this step is deliberately heterogeneous, matching the paper exactly. Do not substitute the homogeneous roster here — the purpose is to reproduce published numbers, not to test our own hypothesis.

### Implementation

1. Stand up the Sequential Style (Scaled) configuration exactly as published.
2. Use the paper's settings throughout: AdamW at a learning rate of 5e-4, cosine schedule, batch size 4, maximum sequence length 4,096, latent thought length of 80 steps.
3. Train the inner loop, then the outer loop, at recursion depths 1, 2 and 3.
4. Evaluate against the paper's reported figures.

### Success criteria

Reproduce the following within two accuracy points at recursion depth 3:

|Benchmark|Published|
|---|---|
|MATH500|88.0|
|GPQA-Diamond|66.2|
|LiveCodeBench|42.9|
|MedQA|79.3|

The efficiency claims should also hold directionally: a speed-up over the text-based variant that grows with recursion depth, and a substantial reduction in tokens.

**Stop rule:** if reproduction fails by more than five points after two weeks, raise it with the authors before attempting the scaled version.

---

## Step 3 — Port the RecursiveLink and verify gradient flow

### Resources needed

- Model weights for Gemma 4 31B Dense
- Working knowledge of the model's configuration and decoder implementation
- Approximately 8 node-hours, mostly for testing
- **For Arm B only:** Gemma 4 26B-A4B, Qwen3.8-27B, Qwen3.6-27B, and a substantially larger time allowance

### Explanation

Two jobs for Arm A. A third appears only in Arm B.

**Resizing the adapters.** The adapter takes the form `W₃h + W₂σ(W₁h)` — a direct projection added to a narrow bottleneck path. The direct path preserves the meaning of the incoming representation; the bottleneck path learns the difference between conventions. With a homogeneous roster all three agents share the same representation width, so `W₃` is square. This matches the paper's theoretical analysis, which only covers the case where that projection is the identity, and removes a source of error. Adapter size scales with the square of the width, so trainable parameters rise from the paper's 13.12M to roughly 50M per link — still negligible beside 31B of frozen backbone.

**Disabling the vision path.** Gemma 4 can process images. Requesting "the last layer's representation" may therefore return something from a multimodal fusion path rather than the text decoder. Address the text decoder explicitly and confirm the extracted vector is the one the paper means.

**Weight sharing.** All three agents are frozen and identical, so load one copy and attach three adapter sets and three role prompts. Verify that adapters do not accidentally share parameters — this is the most likely bug in the homogeneous setup, and it will silently collapse three roles into one.

**The Arm B complication, deferred.** Qwen3.8-27B keeps a single running summary that is updated at each step rather than storing every previous token. It is faster, which is why it was chosen. But standard implementations overwrite that summary in place, because text generation never needs the history. Training does: the outer loop must trace a gradient backwards through roughly 720 sequential updates. If the history is discarded, the gradient dies before reaching the upstream agents.

That failure is silent and, critically, it looks exactly like a negative research finding. The loss plateaus, the Solver's adapter trains whilst the Planner's and Critic's receive nothing, and the natural conclusion is "the architecture does not help at this scale". That conclusion would be wrong in a way that is very hard to detect after the fact. Arm A avoids the problem entirely; Arm B must confront it under Step 9.

### Implementation

1. Read the representation width from the model configuration; construct the adapters accordingly. Retain the two-layer design with the direct path — the paper's ablation shows it outperforms all three alternatives tested.
2. Load the text decoder explicitly; confirm the extracted representation originates from the intended layer.
3. Attach three independent adapter sets to the single shared backbone. Assert that their parameters are distinct.
4. Perturb one input value slightly and confirm the observed change in output matches the analytical gradient.
5. Repeat the check across the full 80-step unroll. A short check can pass whilst a long one silently degrades.
6. Confirm gradients reach the Planner's adapter — the furthest point upstream — before proceeding.

### Success criteria

- Finite-difference agreement across the full unrolled chain, not merely a short test case.
- Non-zero, non-degenerate gradients reaching the Planner's adapter.
- The three adapter sets are verifiably independent.

**Stop rule:** do not proceed to Step 6 until the full-length gradient check passes. Outer-loop training on an unverified backward pass produces results that cannot be interpreted.

---

## Step 4 — Rewrite training data into role-specific targets

### Resources needed

- s1K, m1K, OpenCodeReasoning, ARPO-SFT
- API access to a large model for rewriting (the paper used Qwen3.5-397B-A17B)
- Approximately 3–5 days, mostly waiting on API calls

### Explanation

Each agent needs its own supervision target. The Planner is trained towards an initial plan, the Critic towards a revised plan, and the Solver towards the final answer.

It is worth being precise about what the rewriting model does, because this is commonly misunderstood. It is **not** solving the problems. Every problem already has a known-correct answer. The model reformats that answer into three role-appropriate views. A rewriting model weaker than the agents is therefore perfectly acceptable — nobody is asking it to reason.

For a competition mathematics problem with the answer 246, the rewriter produces an initial plan setting out the approach, a revised plan correcting an omission in that approach, and the original worked solution unchanged.

This dataset is shared by both arms and every baseline, so build it once and version it.

### Implementation

1. Collect question-and-answer pairs from all four sources.
2. For each pair, request two rewrites: an initial step-by-step plan, and a revised plan incorporating a correction.
3. Assign the initial plan to the Planner, the revised plan to the Critic, and the original answer to the Solver.
4. Spot-check a sample of 100 by hand. Rewriting failures are common and quiet.
5. Cache everything.

### Success criteria

- Approximately 4,000 role-tagged triples spanning mathematics, medicine, science and code.
- Manual inspection confirms the Critic targets genuinely differ from the Planner targets — a rewriter that simply copies produces a dataset that teaches nothing.
- The dataset is version-controlled and reused identically across all arms.

---

## Step 5 — Inner-loop training

### Resources needed

- The shared frozen backbone with three adapter sets attached
- The dataset from Step 4
- Approximately 10 node-hours

### Explanation

This stage warms up each role independently. Each learns to shape its internal representation so that its adapter can map it onto the correct target. The objective compares the adapter's output against what the model's own input layer would produce for the correct answer.

In a homogeneous roster this stage carries an additional burden: it is where role specialisation is established. The three agents share weights and differ only in prompt and adapter, so if the inner loop fails to differentiate them, the system reduces to one model called three times. Check for this explicitly.

### Implementation

1. Train each role's inner adapter separately against the shared backbone.
2. Use a latent thought length of 80 steps. The paper's ablation shows performance stabilising around this point, with little gain beyond.
3. Run the three roles concurrently across the node — the freed memory makes this straightforward.

### Success criteria

- All three adapters converge; the objective falls steadily rather than plateauing immediately.
- **Interpretability check:** pass a representation through each adapter and decode the result as though it were an ordinary input. The Planner's output should be recognisably plan-shaped, the Solver's answer-shaped. Not noise.
- **Differentiation check:** the three adapters produce measurably different outputs for the same input. If they converge to near-identical mappings, role specialisation has failed and the homogeneous arm cannot test what it is meant to test.

**Stop rule:** if the decoded output is noise, return to Step 3 — this almost always indicates a problem with which layer's representation is being extracted. If the differentiation check fails, revisit the role prompts before proceeding.

---

## Step 6 — Outer-loop training

### Resources needed

- The shared backbone plus warmed-up inner adapters
- 20–30 node-hours

### Explanation

This is where the system is trained as a single unit.

For one problem at recursion depth 3, the Planner reads the question and produces 80 internal steps rather than words. These pass through an adapter into the Critic's representation space. The Critic produces 80 of its own. Another adapter, then the Solver, and 80 more. The Solver's output returns to the Planner and the cycle repeats twice more. Only at the very end does the Solver produce actual text.

That is 720 sequential steps with the entire computation graph retained, so gradients can travel back through all of them. Memory pressure comes from retaining those intermediate states, not from the weights.

Weight sharing helps materially here. With roughly 100 GB freed relative to a mixed roster, activation offload may prove unnecessary and a larger effective batch becomes possible — which is why this step is estimated below the mixed-roster figure.

### Implementation

1. Freeze the backbone; train only the cross-agent adapters.
2. Unroll the chain for the chosen recursion depth; compute the objective against the correct answer using the final text output only.
3. Train at depths 1, 2 and 3 if the training-depth analysis is wanted; otherwise depth 3 alone.
4. Checkpoint frequently. A failure ten hours into a run is expensive.

### Success criteria

- Training completes without divergence.
- Accuracy improves monotonically with recursion depth on a held-out set — the paper's central claim, and the first genuine signal the method works at this scale.
- Gradients reaching the Planner's adapter remain non-trivial throughout.

---

## Step 7 — Train self-hosted baselines

### Resources needed

- The same Gemma 4 31B weights used throughout
- Sharded training capability for the full fine-tuning baseline
- 60–120 node-hours

### Explanation

The baselines carry the entire fairness argument, and they must run on our own hardware. API results cannot substitute: providers typically serve reduced-precision weights, endpoints change silently between runs, and default reasoning behaviour is often unclear. Since the research question is precisely whether the architecture adds anything beyond raw model capability, an unexplained capability shift in the baseline is the worst possible confound.

A homogeneous roster makes these baselines unusually clean, because "the single agent" is unambiguous — it is the same model, in the same precision, that all three roles are running.

Five rows are required:

- **Single agent, untrained.** Gemma 4 31B alone. Not in the original paper, and the row that answers Question 1 directly.
- **Single agent, adapter fine-tuning.** Same model, same data.
- **Single agent, full fine-tuning.** The most expensive single item in the project. At 31B this requires sharded optimiser state, and per step it costs more than RecursiveMAS training does.
- **Text-based recursive system.** Identical topology, agents passing text instead of representations. The direct comparison for the efficiency claims, and roughly 2.4× slower to evaluate than the method itself.
- **Simple ensemble.** The same model queried three times with the three role prompts, results combined by majority vote, no recursion and no adapters. This is the cheapest possible explanation for any gain we observe, and ruling it out is worth the modest cost.

### Implementation

1. Train each baseline on the Step 4 dataset, matching training budget as closely as the methods allow.
2. Use identical decoding settings across every row.
3. Record memory use and cost per row, mirroring the paper's cost table.

### Success criteria

- All five rows train to completion.
- Comparable trainable-parameter budgets where the comparison is meaningful.
- Every row reproducible from a recorded configuration file.

---

## Step 8 — Evaluate everything under one harness

### Resources needed

- All trained systems and baselines
- An API-based judge for open-ended answers, matching the paper's protocol
- 300–600 node-hours
- 3–4 weeks

### Explanation

The single largest cost in the project. The multiplication is unforgiving: benchmarks × recursion depths × five runs × every baseline.

One harness, one set of settings, one evaluation script. Any deviation between rows undermines the comparison.

The only appropriate use of an API here is the judge for open-ended search answers, which matches the paper's own protocol and does not touch the systems being compared.

### Implementation

1. Fix decoding settings across all rows: top-p 0.95, temperature 0.6 for reasoning tasks and 0.2 for code.
2. Set generation limits by task difficulty — approximately 4,000 tokens for science and medicine, 16,000 for competition mathematics.
3. Use the same sampling protocol as the paper where benchmarks are carried over.
4. Execute code in a sandbox with per-test timeouts.
5. Five independent runs per configuration; report mean and standard deviation.
6. Report accuracy and efficiency separately. They may well diverge.

### Success criteria

- Every retained benchmark evaluated across every row at recursion depth 3.
- Standard deviations comparable to the paper's ±0.0041; wider variance indicates a harness problem.
- A clear statement of whether the architectural advantage survives at this scale, with the ensemble baseline ruling out the cheapest alternative explanation.

---

## Step 9 — Second arm: heterogeneous roster (optional)

### Resources needed

- Gemma 4 26B-A4B, Qwen3.8-27B
- Qwen3.6-27B for the diagnostic control
- Everything already built in Steps 4 to 8
- 100–200 node-hours, 2–3 weeks

### Explanation

Arm A establishes whether recursion helps when every agent is the same model. Arm B asks how much more it helps when the agents differ. The gap between the two is the measurement of how much of the paper's reported benefit came from model diversity rather than from recursive collaboration — a question the paper cannot address, since every configuration it tests is heterogeneous.

This arm re-introduces the engineering risk deferred from Step 3. Qwen3.8-27B's running-summary attention must be made differentiable across the full unroll. Options, in ascending order of effort: retain every intermediate state, which exhausts memory; recompute them on the backward pass, roughly 30% slower but reliable; or derive the gradient analytically and write a custom backward pass, fastest but with the greatest scope for subtle error. Recomputation is the recommended default.

It also loses the memory advantage: three distinct backbones must be resident simultaneously, so activation offload will very likely be required.

### Implementation

1. Complete the DeltaNet gradient work and pass the same full-length finite-difference check required in Step 3.
2. Rebuild the adapters with non-square projections to bridge the differing representation widths.
3. Reuse the Step 4 dataset unchanged.
4. Repeat Steps 5, 6 and 8. Baselines from Step 7 remain valid where the backbone is unchanged; the untrained single-agent row must be repeated for Qwen3.8-27B.
5. Report Arm A and Arm B side by side.

### Success criteria

- Arm B trains and evaluates cleanly under the same harness.
- The Arm A / Arm B difference is reported explicitly as the diversity contribution.

**Final safeguard:** if Arm B shows no architectural advantage where Arm A did, re-run the headline comparison with Qwen3.6-27B substituted for the Solver before drawing any conclusion. If the advantage reappears under conventional attention, the finding was an implementation artefact and must not be published as a scaling result.

---

## 4. Risk summary

|Risk|Likelihood|Mitigation|
|---|---|---|
|Benchmarks saturated|Very high|Step 1 screening, before any compute is committed|
|Homogeneous roster under-reports the paper's effect|Moderate|Arm B measures the diversity contribution directly|
|Role specialisation fails to establish|Moderate|Differentiation check at the end of Step 5|
|Evaluation over-runs the budget|Moderate|Reduce to three runs; evaluate baselines at depth 3 only|
|Full fine-tuning baseline under-scoped|Moderate|Budget sharded training explicitly from the outset|
|Rewriting model produces degenerate targets|Moderate|Manual inspection of 100 samples in Step 4|
|Adapters accidentally share parameters|Moderate|Explicit assertion in Step 3; differentiation check in Step 5|
|Memory exhaustion during outer-loop training|Low for Arm A, high for Arm B|Weight sharing in Arm A; activation offload planned for Arm B|
|Gradient failure through running-summary attention|Not applicable to Arm A, high for Arm B|Deferred entirely to Step 9; full-length check plus control arm|

Choosing a homogeneous primary arm removes the project's two most serious technical risks — gradient failure through Qwen's attention mechanism, and memory exhaustion during the unrolled backward pass — and converts the third, immature tooling, into a non-issue. It introduces one new risk in exchange: that a homogeneous system understates the effect the paper measured. Arm B exists to quantify precisely that, which turns the trade-off into a finding rather than a limitation.

The two steps most likely to overrun remain Step 8 and, if attempted, Step 9. Neither is limited by compute — one is a scheduling problem, the other an engineering problem. Plan the buffer there.