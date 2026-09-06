"""Central configuration for Step 1 benchmark screening.

Every other module in `screening/` reads its settings from here, so that decoding
parameters, provider pins and benchmark definitions live in exactly one place and
end up recorded verbatim in the run manifest.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Repository layout. The vendored upstream checkout is never written to.
REPO_ROOT = Path(__file__).resolve().parent.parent
RECURSIVEMAS_DIR = REPO_ROOT / "RecursiveMAS"
INFERENCE_DIR = RECURSIVEMAS_DIR / "inference"
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
CACHE_DIR = REPO_ROOT / "screening" / ".cache"

# Decoding settings shared by every benchmark, from the paper's protocol.
TOP_P = 0.95
DEFAULT_SEED = 42

# LLM judge for open-ended answers. The paper used Qwen3.5-397B-A17B; this is the
# nearest available frontier Qwen on OpenRouter. Routed through the same account as
# generation, so no second key is needed.
DEFAULT_JUDGE_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_JUDGE_MODEL = "qwen/qwen3.8-2.4t-a95b"

# The exclusion gate: retire a benchmark when the strongest backbone reaches this
# score with thinking disabled, i.e. when fewer than 10 points of headroom remain.
SATURATION_THRESHOLD = 90.0

# Minimum benchmarks that must survive screening, or Step 1's stop rule fires.
MIN_SURVIVING_BENCHMARKS = 4

# --canary limits: enough to exercise every path, cheap enough to run freely.
# Two rollouts still exercise the multi-rollout aggregation; a 2,000-token cap keeps
# the 16k benchmarks from dominating the bill and incidentally exercises the
# truncation column too. Read by report.py as well, to recognise a canary summary
# written before the flag existed.
CANARY_SAMPLES = 10
CANARY_ROLLOUTS = 2
CANARY_MAX_TOKENS = 2000

# Parse-failure rate above which a choice benchmark's accuracy is not trustworthy,
# because answer_utils scores unparseable predictions as "A" rather than as wrong.
MAX_TRUSTED_PARSE_FAILURE_RATE = 0.05

# Truncation above this makes a saturated-looking score a floor rather than a measurement.
MAX_TRUSTED_TRUNCATION_RATE = 0.10

# Provider errors and empty completions are scored as wrong, so more than a few
# of them understate the benchmark and the cell needs re-running.
MAX_TRUSTED_ERROR_RATE = 0.02


# One model under screening, pinned to a single OpenRouter provider.
# Prices are USD per million tokens and are used only for the --dry-run estimate;
# realised cost always comes from each response's reported usage.
@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    provider: dict[str, Any]
    price_in_per_m: float = 0.0
    price_out_per_m: float = 0.0


# Provider pins are deliberate and must not be relaxed to automatic routing:
# OpenRouter otherwise load-balances across providers serving different
# quantizations, which would make a score unattributable to any specific weights.
MODELS: dict[str, ModelSpec] = {
    # Novita is the only bf16 Gemma endpoint whose completion cap (131k) clears
    # AIME's 16k requirement; most bf16 providers cap at 8,192 tokens.
    "gemma": ModelSpec(
        key="gemma",
        model_id="google/gemma-4-31b-it",
        provider={
            "order": ["novita"],
            "allow_fallbacks": False,
            "quantizations": ["bf16"],
            "require_parameters": True,
        },
        price_in_per_m=0.08,
        price_out_per_m=0.35,
    ),
    # No Qwen3.8-27B provider declares bf16, so we pin the first-party Alibaba
    # endpoint as the closest available proxy. See the plan's precision spot-check.
    "qwen": ModelSpec(
        key="qwen",
        model_id="qwen/qwen3.8-27b",
        provider={
            "order": ["alibaba"],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        price_in_per_m=0.425,
        price_out_per_m=2.55,
    ),
}

# Reasoning toggles. Never use {"exclude": True} — OpenRouter still runs and still
# bills the reasoning, so the "off" arm would silently be an "on" arm.
THINKING_MODES: dict[str, dict[str, Any]] = {
    "off": {"enabled": False},
    "on": {"enabled": True, "effort": "high"},
}

# Qualifications on how a cell's numbers may be read, keyed by "<model>/<mode>".
# Written into manifest.json and headroom.md so a quoted score cannot be lifted
# out of the record without the caveat that applies to it.
THINKING_MODE_CAVEATS: dict[str, str] = {
    "qwen/off": (
        "Not a true no-reasoning baseline. reasoning.enabled=false does suppress the "
        "labelled thinking channel, and all three probe signals agree it is absent: no "
        "thinking tag appears in any of the 11,718 recorded responses, no message.reasoning "
        "field is populated, and no reasoning tokens are billed beyond a single outlier in "
        "hle. The deliberation is nonetheless still performed — it has moved into the "
        "visible answer as ordinary prose. Qwen emits 1.8-3.4x Gemma's output length, and "
        "self-correction discourse markers appear in 68.7% of gpqa, 76.7% of lcb and 84.4% "
        "of hle responses, against 18-41% for Gemma on the same items. The thinking-OFF arm "
        "therefore compares Gemma answering directly against Qwen reasoning unlabelled, and "
        "the two columns are not the same experimental condition. This is recorded rather "
        "than corrected: suppressing the working would require a prompt change, and the "
        "prompt is held verbatim from the harness so that scores stay on the published "
        "table's scale. It does not affect any gate decision, because Qwen scores below "
        "Gemma on all eight benchmarks and so sets no gating score."
    ),
}


# How a multi-rollout benchmark's generations become one score. pass@k counts an
# item correct if any rollout solved it, which is the paper's protocol; avg@k takes
# the mean of independent attempts. The difference is not cosmetic: at a true
# per-attempt skill of 30%, pass@10 reports 97.2. That is what saturated aime25 and
# aime26 at the item level while their problems still had headroom, so any benchmark
# whose headroom carries a decision is aggregated as a mean.
AGGREGATION_PASS_AT_K = "pass_at_k"
AGGREGATION_MEAN = "mean"


# One benchmark: where to load it, how to generate for it, and how to score it.
@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    loader_arg: str  # dataset name understood by the vendored loaders
    split: str
    scorer: str  # choice | math | code | judge
    max_tokens: int
    temperature: float = 0.6
    rollouts: int = 1  # >1 reports pass@k or avg@k, per `aggregation`
    aggregation: str = AGGREGATION_PASS_AT_K  # only read when rollouts > 1
    num_samples: int = -1  # -1 loads the full set
    full_set_size: int = 0  # documented size, used only for the --dry-run estimate
    shuffle: bool = False  # the harness defaults to False; only true where we subsample
    is_paper_benchmark: bool = True  # False for replacement candidates
    code_timeout_s: int = 10
    # Option range this benchmark's answers use. The vendored extractor hardcodes
    # A-D, which silently mis-scores anything wider; screening/score.py reads this
    # instead. See MedXpertQA (A-J) and Medbullets op5 (A-E).
    choice_letters: str = "ABCD"
    # Item difficulty floor, for benchmarks that ship a difficulty label. 0.0 keeps
    # every item, which is what a first pass must do: the cut is chosen from the
    # observed score-versus-difficulty curve, not guessed before the curve exists.
    min_difficulty: float = 0.0
    # Controls are instrument checks, not gated benchmarks. They report headroom so
    # a saturated control is visible, but produce no retain/retire verdict and are
    # excluded from the surviving-benchmark tally the stop rule reads.
    is_control: bool = False
    # Set only by --canary. Recorded on every summary the canary writes so a
    # 10-item smoke-test score can never be mistaken for a measurement.
    is_canary: bool = False
    notes: str = ""


# The seven benchmarks Sequential Style supports, plus HLE as a replacement
# candidate. Seven, not nine: RELEASE_RECOMMENDED_SETTINGS in inference_mas.py
# sanctions exactly these for ("sequential_scaled", ...), and the two search-QA
# benchmarks appear only under ("deliberation", ...) — see OUT_OF_SCOPE_BENCHMARKS.
#
# Generation caps and temperatures mirror run.py:infer_max_new_tokens and
# run.py:infer_temperature exactly, so scores land on the same scale as the
# published table.
BENCHMARKS: dict[str, BenchmarkSpec] = {
    "math500": BenchmarkSpec(
        key="math500",
        full_set_size=500,
        loader_arg="math500",
        split="test",
        scorer="math",
        max_tokens=2000,
    ),
    "gpqa": BenchmarkSpec(
        key="gpqa",
        full_set_size=198,
        loader_arg="gpqa",
        split="train",
        scorer="choice",
        max_tokens=4000,
        notes=(
            "Gated dataset — accept the terms at huggingface.co/datasets/Idavidrein/gpqa "
            "and set HF_TOKEN. Those terms forbid publishing examples in plain text, so "
            "screening_results.jsonl must stay git-ignored."
        ),
    ),
    "medqa": BenchmarkSpec(
        key="medqa",
        full_set_size=300,
        loader_arg=str(INFERENCE_DIR / "dataset" / "medqa.json"),
        split="train",
        scorer="choice",
        max_tokens=4000,
        notes="300-item local subset shipped with the repo, not the full 1,273-item test set.",
    ),
    "mbppplus": BenchmarkSpec(
        key="mbppplus",
        full_set_size=378,
        loader_arg="mbppplus",
        split="test",
        scorer="code",
        max_tokens=4000,
        temperature=0.2,
        code_timeout_s=10,
    ),
    "aime25": BenchmarkSpec(
        key="aime25",
        full_set_size=30,
        loader_arg="aime25",
        split="test",
        scorer="math",
        max_tokens=16000,
        rollouts=10,
    ),
    "aime26": BenchmarkSpec(
        key="aime26",
        full_set_size=30,
        loader_arg="aime26",
        split="train",
        scorer="math",
        max_tokens=16000,
        rollouts=10,
    ),
    "lcb": BenchmarkSpec(
        key="lcb",
        full_set_size=1055,
        loader_arg="lcb",
        split="test",
        scorer="code",
        max_tokens=4096,
        temperature=0.2,
        code_timeout_s=6,
        notes="Loader concatenates all six release files: cumulative v1-v6 lite, ~1,055 problems.",
    ),
    "hle": BenchmarkSpec(
        key="hle",
        full_set_size=2158,
        loader_arg="hle",
        split="test",
        scorer="judge",
        max_tokens=4000,
        is_paper_benchmark=False,
        notes=(
            "Gated dataset — accept the terms at huggingface.co/datasets/cais/hle. "
            "Text-only subset; multimodal items dropped because Step 3 disables the "
            "vision path. Carries a canary string and must not reach training corpora."
        ),
    ),
    # Restores mathematics, which retiring math500, aime25 and aime26 emptied.
    # The repaired 2026 edition, never the 2024 original: an audit of that release
    # found 647 problems needing edits and 247 that exact-answer verification cannot
    # score at all — 61 of them referencing images Step 3's disabled vision path
    # could never see.
    "omni_math": BenchmarkSpec(
        key="omni_math",
        # The Filtered subset — every item the audit tagged non-standard removed —
        # less the two that survive the tags with no reference answer at all.
        full_set_size=4172,
        loader_arg="omni_math",
        split="train",
        scorer="judge",
        # Raised from 8,000: Qwen averaged 13,511 completion tokens on these problems
        # in the canary, so the old cap turned most of its cell into truncations.
        max_tokens=16000,
        rollouts=5,
        aggregation=AGGREGATION_MEAN,
        is_paper_benchmark=False,
        notes=(
            "Omni-MATH-2-Filtered (martheballon/Omni-MATH-2, non-standard tags removed). "
            "Scored by the judge pinned in this file and NEVER by Omni-Judge, which the "
            "same audit found wrong in 96.4% of judge disagreements, its dominant failure "
            "being exactly the equivalence-checking it exists to perform; re-scoring "
            "identical generations moved one frontier model 5.5 points. This project "
            "measures effects of about half a point, so (dataset, judge) is one frozen "
            "unit. avg@k, never pass@k: pass@10 at a true per-attempt skill of 30% reports "
            "97.2, which is how aime25 and aime26 saturated while their problems had not. "
            "Screening runs at --rollouts 1, where the metric is plain accuracy and the "
            "trap cannot reappear by accident. No difficulty cut is set in advance: "
            "min_difficulty stays 0.0 until the gating cell's score-versus-difficulty "
            "curve says where the cut maximises n while leaving 25 points of headroom."
        ),
    ),
    # Replaces medqa, which screened at 92.3 with only 23 of 300 items unsolved.
    # Hardest of the 19 verifiable medical benchmarks in Medmarks: 0.236 mean across
    # 61 models against medqa's 0.784. MIT-licensed and ungated.
    "medxpertqa": BenchmarkSpec(
        key="medxpertqa",
        full_set_size=2450,
        loader_arg="medxpertqa",
        split="test",
        scorer="choice",
        # 8,000 rather than medqa's 4,000. Qwen truncated 34% of gpqa at 4,000 and
        # these items are harder, so the old cap would produce a floor, not a score.
        max_tokens=8000,
        # Ten options, A-J. The vendored extractor stops at D and defaults the rest
        # to "A", which scores them correct whenever A is the gold.
        choice_letters="ABCDEFGHIJ",
        is_paper_benchmark=False,
        notes=(
            "Text split only; the MM split is multimodal and Step 3 disables the vision "
            "path. Golds are uniform over all ten letters (222-263 each), so no option "
            "shuffle is needed and none is applied."
        ),
    ),
    # CONTROL. Contamination and judge-independence check for omni_math (§5.1).
    "beyondaime": BenchmarkSpec(
        key="beyondaime",
        full_set_size=100,
        loader_arg="beyondaime",
        split="test",
        scorer="math",
        max_tokens=16000,
        is_paper_benchmark=False,
        is_control=True,
        notes=(
            "Not gated: n=100 gives +/-9.8, which compares signs and rough magnitudes and "
            "nothing finer. Three jobs. (1) Contamination: Omni-MATH's problems went "
            "public in October 2024 and both backbones postdate that, so a gain there "
            "could be retrieval; BeyondAIME is the clean twin, same domain and answer "
            "format and difficulty band, every problem manually rewritten. (2) Judge "
            "independence: positive-integer answers go through compare_answers with no "
            "judge in the loop, the only maths benchmark in the set with that property. "
            "(3) Threshold calibration for omni_math's difficulty cut, whose labels are "
            "self-reported where BeyondAIME's band is externally anchored. CC0."
        ),
    ),
    # CONTROL. Critic mechanism probe, first half of the pair (§5.2).
    "medbullets_op4": BenchmarkSpec(
        key="medbullets_op4",
        full_set_size=308,
        loader_arg="medbullets_op4",
        split="train",
        scorer="choice",
        max_tokens=4000,
        is_paper_benchmark=False,
        is_control=True,
        notes=(
            "Not gated: Gemma's canary already scored 90.0, the same neighbourhood as the "
            "medqa it would replace, and the questions come from a public X account posting "
            "between April 2022 and December 2023 and so are heavily exposed to pretraining. "
            "It earns its place as the baseline arm of a paired probe, where both objections "
            "cancel: a memorised question is memorised in op4 and op5 alike and contributes "
            "nothing to the difference between them. Options are re-shuffled by stable hash "
            "because the shipped gold distribution is skewed (A 87, D 68 of 308), and a "
            "position effect must not be readable as distractor sensitivity."
        ),
    ),
    # CONTROL. Critic mechanism probe, second half of the pair (§5.2).
    "medbullets_op5": BenchmarkSpec(
        key="medbullets_op5",
        full_set_size=308,
        loader_arg="medbullets_op5",
        split="train",
        scorer="choice",
        max_tokens=4000,
        # A-E: op5 is op4 plus one extra plausible-but-wrong option.
        choice_letters="ABCDE",
        is_paper_benchmark=False,
        is_control=True,
        notes=(
            "The same 308 questions as medbullets_op4 with one extra distractor and nothing "
            "else changed, so the op4->op5 drop isolates one variable: what an additional "
            "attractive wrong option costs. Rejecting one is the Critic's job, giving a "
            "falsifiable prediction independent of headline accuracy — RecursiveMAS's drop "
            "should be smaller than the single-agent baseline's, and should shrink "
            "monotonically across depths 1-2-3. The comparison is within one scale across "
            "Step 8 rows, never 4B against 31B, since larger models are less distractible "
            "regardless. Paired on identical items, so item difficulty cancels and the "
            "+/-4.3 marginal interval does not apply; simulated power at 308 items and five "
            "runs detects a 3-point change 98% of the time and a 2-point change 68%. "
            "Screening it now asks only whether the instrument has any range: a baseline "
            "drop under ~2 points means there is none and the probe should be dropped."
        ),
    ),
}

# Paper benchmarks the Sequential Style cannot run. Not a saturation judgement:
# these are excluded by the project's own scoping decision, and are documented in
# exclusions.md so the write-up can say why only seven of nine were screened.
OUT_OF_SCOPE_BENCHMARKS: dict[str, str] = {
    "HotpotQA": (
        "Search-QA, supported only under the Deliberation style. "
        "run.py:validate_style_dataset rejects it for every other style, because "
        "without tools it falls through to string-match scoring and reports a number "
        "that does not reflect the task. Sequential Style is Planner/Critic/Solver "
        "with no Tool-Caller agent, so it cannot search, and the paper reports no "
        "Sequential-Scaled figure for this benchmark to compare against."
    ),
    "Bamboogle": (
        "Search-QA, supported only under the Deliberation style, for the same reasons "
        "as HotpotQA. RELEASE_RECOMMENDED_SETTINGS sanctions ('deliberation', "
        "'bamboogle') and no sequential pairing."
    ),
}

# Benchmarks excluded before screening, recorded here so report.py can write them
# into exclusions.md alongside the ones that fail the gate empirically.
INFEASIBLE_CANDIDATES: dict[str, str] = {
    "FrontierMath": (
        "12 of 338 problems are public; the remainder is held privately by Epoch AI, "
        "with OpenAI holding exclusive access to a subset. Cannot be run locally at any n."
    ),
    "Agents' Last Exam": (
        "153 computer-use agent tasks with reference outputs gated behind manual approval. "
        "The Sequential Style emits one text answer with no computer-use loop, and the "
        "average full pass rate is below 1%, so the benchmark has a floor problem as well "
        "as a scaffold problem."
    ),
    "SWE-Bench Verified": (
        "Requires Docker, repository checkouts and an agentic edit/test loop that the "
        "Sequential Style cannot express. Screening a bare model would measure the "
        "scaffold rather than the backbone."
    ),
}


# Resolved secrets and paths for one screening run.
@dataclass(frozen=True)
class ScreenConfig:
    openrouter_api_key: str
    hf_token: str
    judge_model: str
    judge_base_url: str
    seed: int = DEFAULT_SEED
    cache_dir: Path = field(default=CACHE_DIR)
    output_dir: Path = field(default=BENCHMARKS_DIR)


# Load .env and resolve the keys, failing loud and listing everything that is missing.
def load_config(seed: int = DEFAULT_SEED) -> ScreenConfig:
    load_dotenv(REPO_ROOT / ".env")

    # Only two secrets are genuinely needed. The judge's three settings are derived
    # below rather than being asked for again.
    required = ["OPENROUTER_API_KEY", "HF_TOKEN"]

    # Report every missing key at once rather than one per failed run.
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if missing:
        raise RuntimeError(
            "Missing required .env keys: "
            + ", ".join(missing)
            + f"\nCopy {REPO_ROOT / '.env.example'} to .env and fill them in."
        )

    openrouter_api_key = os.environ["OPENROUTER_API_KEY"].strip()

    # llm_judge.require_config() reads API_KEY / API_BASE_URL / API_MODEL straight from
    # the environment, and those names are fixed in the vendored module. Rather than
    # making the user paste the same OpenRouter key twice, populate them here. setdefault
    # means an explicitly set value still wins, so the judge can be pointed at a
    # different provider if that is ever wanted.
    os.environ.setdefault("API_KEY", openrouter_api_key)
    os.environ.setdefault("API_BASE_URL", DEFAULT_JUDGE_BASE_URL)
    os.environ.setdefault("API_MODEL", DEFAULT_JUDGE_MODEL)

    return ScreenConfig(
        openrouter_api_key=openrouter_api_key,
        hf_token=os.environ["HF_TOKEN"].strip(),
        judge_model=os.environ["API_MODEL"].strip(),
        judge_base_url=os.environ["API_BASE_URL"].strip(),
        seed=seed,
    )


# Make the vendored RecursiveMAS inference package importable.
def add_recursivemas_to_path() -> None:
    import sys

    # inference_mas imports torchvision unless told not to; screening is text-only,
    # so we set the same flag run.py sets before any vendored module is loaded.
    os.environ.setdefault("MAS_FORCE_DISABLE_TORCHVISION", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # inference/ is the package root: its modules import each other by bare name.
    for path in (INFERENCE_DIR, RECURSIVEMAS_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
