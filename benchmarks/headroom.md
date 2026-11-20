# Benchmark screening — 2026-09-06

Gate: RETIRE when the strongest backbone scores >= 90 with thinking disabled (fewer than 10 points of headroom).

| Benchmark | Metric | n | gemma OFF | qwen OFF | max OFF | Wilson 95% | max ON | Headroom | Parse-fail | Trunc | Err | Decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aime25 | pass@10 | 30 | 96.7 | 90.0 | 96.7 | [83.3, 99.4] | nan | 3.3 | 4.0% | 3.0% | 0.0% | UNDECIDED |
| math500 | accuracy | 500 | 96.2 | 83.4 | 96.2 | [94.1, 97.6] | nan | 3.8 | 1.2% | 1.2% | 0.0% | RETIRE |
| medqa | accuracy | 300 | 92.3 | 90.0 | 92.3 | [88.8, 94.8] | nan | 7.7 | 1.3% | 0.0% | 0.0% | UNDECIDED |
| aime26 | pass@10 | 30 | 90.0 | 90.0 | 90.0 | [74.4, 96.5] | nan | 10.0 | 0.3% | 0.7% | 0.0% | UNDECIDED |
| medbullets_op4 | accuracy | 308 | 84.7 | 84.1 | 84.7 | [80.3, 88.3] | nan | 15.3 | 0.0% | 0.0% | 0.0% | CONTROL (has range) |
| lcb | accuracy | 1055 | 82.4 | 75.5 | 82.4 | [80.0, 84.6] | nan | 17.6 | 0.9% | 0.8% | 0.0% | RETAIN (anchor) |
| mbppplus | accuracy | 378 | 81.7 | 79.1 | 81.7 | [77.5, 85.3] | nan | 18.3 | 0.5% | 0.5% | 0.0% | RETAIN (anchor) |
| medbullets_op5 | accuracy | 308 | 81.2 | 78.9 | 81.2 | [76.4, 85.1] | nan | 18.8 | 0.0% | 0.0% | 0.0% | CONTROL (has range) |
| gpqa | accuracy | 198 | 74.2 | 79.3 | 79.3 | [73.1, 84.4] | nan | 20.7 | 0.0% | 0.0% | 0.5% | RETAIN (anchor) |
| omni_math | accuracy | 4172 | 74.3 | 67.6 | 74.3 | [72.9, 75.6] | nan | 25.7 | 0.0% | 0.6% | 0.0% | RETAIN |
| beyondaime | accuracy | 100 | 59.0 | 55.0 | 59.0 | [49.2, 68.1] | nan | 41.0 | 1.0% | 1.0% | 0.0% | CONTROL (has range) |
| medxpertqa | accuracy | 2450 | 55.3 | 38.3 | 55.3 | [53.3, 57.2] | nan | 44.7 | 0.0% | 0.0% | 0.0% | RETAIN |
| hle | accuracy | 2158 | 10.8 | 12.2 | 12.2 | [10.9, 13.6] | nan | 87.8 | 0.0% | 0.3% | 1.6% | RETAIN |

**Retained:** 6 — stop rule not triggered.
**Anchors carried from the paper:** 3 — gpqa, mbppplus, lcb
**Undecided:** 3 — medqa, aime25, aime26. Not locked; see exclusions.md for what each needs.
**Controls (not gated, not counted):** 3 — beyondaime, medbullets_op4, medbullets_op5.

## Critic probe — baseline op4 -> op5 drop

op4 and op5 are the same 308 questions; op5 adds one plausible-but-wrong
option and changes nothing else, so the drop is what one extra distractor
costs. Rejecting one is the Critic's job. These are single-agent baselines:
the prediction they will later be tested against is that RecursiveMAS's drop
is smaller, and shrinks monotonically across depths 1-2-3.

| Model | op4 | op5 | drop | Verdict |
|---|---|---|---|---|
| gemma | 84.7 | 81.2 | +3.6 | usable range |
| qwen | 84.1 | 78.9 | +5.2 | usable range |

A baseline drop under 2 points means the probe has no
dynamic range at this scale. That is a finding, not a failure: it is recorded
as a limitation in the write-up and the Step 8 row is not spent.

## Caveats on how these columns read

**`qwen/off`.** Not a true no-reasoning baseline. reasoning.enabled=false does suppress the labelled thinking channel, and all three probe signals agree it is absent: no thinking tag appears in any of the 11,718 recorded responses, no message.reasoning field is populated, and no reasoning tokens are billed beyond a single outlier in hle. The deliberation is nonetheless still performed — it has moved into the visible answer as ordinary prose. Qwen emits 1.8-3.4x Gemma's output length, and self-correction discourse markers appear in 68.7% of gpqa, 76.7% of lcb and 84.4% of hle responses, against 18-41% for Gemma on the same items. The thinking-OFF arm therefore compares Gemma answering directly against Qwen reasoning unlabelled, and the two columns are not the same experimental condition. This is recorded rather than corrected: suppressing the working would require a prompt change, and the prompt is held verbatim from the harness so that scores stay on the published table's scale. It does not affect any gate decision, because Qwen scores below Gemma on all eight benchmarks and so sets no gating score.

