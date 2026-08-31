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


# One benchmark: where to load it, how to generate for it, and how to score it.
@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    loader_arg: str  # dataset name understood by the vendored loaders
    split: str
    scorer: str  # choice | math | code | judge
    max_tokens: int
    temperature: float = 0.6
    rollouts: int = 1  # >1 reports pass@k instead of accuracy
    num_samples: int = -1  # -1 loads the full set
    full_set_size: int = 0  # documented size, used only for the --dry-run estimate
    shuffle: bool = False  # the harness defaults to False; only true where we subsample
    is_paper_benchmark: bool = True  # False for replacement candidates
    code_timeout_s: int = 10
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
