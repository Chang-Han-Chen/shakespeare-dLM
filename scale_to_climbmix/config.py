"""Single source of truth for the ClimbMix block-diffusion IsoFLOP study."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
TOKENIZED_DIR = DATA_ROOT / "tokenized"
TOKENIZER_PATH = DATA_ROOT / "tokenizer.json"
TOKENIZER_META_PATH = DATA_ROOT / "tokenizer_meta.json"
DATA_MANIFEST_PATH = TOKENIZED_DIR / "manifest.json"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
WANDB_PROJECT = "climbmix-isoflop-scaleup"

# NVIDIA's official files contain GPT-2 token IDs. This shuffled raw-text
# conversion makes it practical to train and apply our own BPE tokenizer.
DATASET_REPO = "karpathy/climbmix-400b-shuffle"
DATASET_REVISION = "915333b4f8b8684f39aeaafea600fea6f43fb703"
VALIDATION_SHARDS = (0,)
# Ninety raw-text shards provide about 5.8B tokens with the fixed 8K tokenizer.
# This covers adaptive downward 1e18 model-size extensions through 12.3M
# without reusing training tokens.
TRAIN_SHARDS = tuple(range(1, 91))
TOKENIZER_TRAIN_SHARDS = (1, 2, 3, 4)
MIN_PREPARED_TRAIN_TOKENS = 5_600_000_000

# 8,192 ordinary/EOT tokens plus a diffusion-only [MASK] token.
BASE_VOCAB_SIZE = 8_192
VOCAB_SIZE = 8_193
EOT_TOKEN = "<|endoftext|>"
MASK_TOKEN = "<|mask|>"
MASK_ID = BASE_VOCAB_SIZE

SEQ_LEN = 256
BLOCK_LEN = 4
# Historical completed-study batch; keep this fixed for reproducing old runs.
BATCH_SIZE = 64
TOKENS_PER_STEP = BATCH_SIZE * SEQ_LEN
# Scale-up runs use the larger global batch for better H100 utilization.
SCALEUP_BATCH_SIZE = 128
SCALEUP_TOKENS_PER_STEP = SCALEUP_BATCH_SIZE * SEQ_LEN

FLOP_MULTIPLIER = 12
COMPUTE_ACCOUNTING = "dense_dual_stream_attention_v1"
COMPUTE_BUDGETS = (1e14, 3e14, 1e15, 3e15, 1e16)
MIN_STEPS = 150
MAX_STEPS = 25_000
MAX_SCALEUP_STEPS = 200_000

# Full-budget 3x sweeps. Boundary winners are extended until locally bracketed.
LEARNING_RATES = (3e-4, 9e-4, 2.7e-3, 8.1e-3)
WEIGHT_DECAY = 0.1
WARMUP_FRACTION = 0.05
DECAY_FRACTION = 0.15
GRAD_CLIP = 1.0
SEED = 1337
MASK_EPS = 1e-3
EVAL_BATCH_SIZE = 16
EVAL_BATCHES = 32


def round_up(value: float, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def llama_ffn_dim(d_model: int) -> int:
    """Llama's approximately 8d/3 SwiGLU hidden width."""
    return round_up(8 * d_model / 3, 8)


def counted_parameters(d_model: int, n_layer: int, d_ff: int) -> int:
    """Exclude input embeddings; include the untied LM head and all norms."""
    per_layer = 4 * d_model**2 + 3 * d_model * d_ff + 2 * d_model
    return n_layer * per_layer + VOCAB_SIZE * d_model + d_model


def layer_matrix_parameters(d_model: int, n_layer: int, d_ff: int) -> int:
    return n_layer * (4 * d_model**2 + 3 * d_model * d_ff)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    n_layer: int
    d_model: int
    n_head: int

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def d_ff(self) -> int:
        return llama_ffn_dim(self.d_model)

    @property
    def layer_matrix_params(self) -> int:
        return layer_matrix_parameters(self.d_model, self.n_layer, self.d_ff)

    @property
    def n_params(self) -> int:
        return counted_parameters(self.d_model, self.n_layer, self.d_ff)

    @property
    def training_flops_per_clean_token(self) -> int:
        """Leading forward/backward matmul FLOPs per clean token.

        Transformer matrices see both L-token streams, dense attention sees
        the full 2L sequence, and the LM head is evaluated only on noisy
        positions. Pointwise operations are omitted.
        """
        block_linears = FLOP_MULTIPLIER * self.layer_matrix_params
        dense_attention = 48 * SEQ_LEN * self.n_layer * self.d_model
        lm_head = 6 * self.d_model * VOCAB_SIZE
        return block_linears + dense_attention + lm_head

    @property
    def autoregressive_training_flops_per_clean_token(self) -> int:
        """Historical dense-matrix accounting for a causal AR update."""
        block_linears = 6 * self.layer_matrix_params
        dense_attention = 12 * SEQ_LEN * self.n_layer * self.d_model
        lm_head = 6 * self.d_model * VOCAB_SIZE
        return block_linears + dense_attention + lm_head

    @property
    def flash_causal_training_flops_per_clean_token(self) -> int:
        """Leading matmul FLOPs for triangular causal FlashAttention.

        There are L(L+1)/2 attended query-key pairs.  QK and PV together
        cost four FLOPs per pair and hidden dimension in the forward pass;
        training is three times forward, giving 6(L+1)d per clean token.
        """
        block_linears = 6 * self.layer_matrix_params
        causal_attention = 6 * (SEQ_LEN + 1) * self.n_layer * self.d_model
        lm_head = 6 * self.d_model * VOCAB_SIZE
        return block_linears + causal_attention + lm_head

    @property
    def effective_compute_parameters(self) -> float:
        return self.training_flops_per_clean_token / FLOP_MULTIPLIER


# Smooth width/depth growth with head dimension fixed at 16.
MODEL_SPECS = (
    ModelSpec("0.14M", n_layer=2, d_model=16, n_head=1),
    ModelSpec("0.29M", n_layer=2, d_model=32, n_head=2),
    ModelSpec("0.5M", n_layer=4, d_model=48, n_head=3),
    ModelSpec("1M", n_layer=4, d_model=80, n_head=5),
    ModelSpec("2M", n_layer=7, d_model=112, n_head=7),
    ModelSpec("4M", n_layer=11, d_model=144, n_head=9),
    ModelSpec("8M", n_layer=12, d_model=208, n_head=13),
)
# First adaptive scale-up wave around the current 3e16 prediction (4.30M).
# Keep these separate so the completed historical grid cannot change.
SCALEUP_MODEL_SPECS = (
    ModelSpec("3.4M", n_layer=9, d_model=144, n_head=9),
    ModelSpec("4.4M", n_layer=10, d_model=160, n_head=10),
    ModelSpec("5.6M", n_layer=11, d_model=176, n_head=11),
    ModelSpec("6.9M", n_layer=12, d_model=192, n_head=12),
    ModelSpec("8.5M", n_layer=13, d_model=208, n_head=13),
    ModelSpec("10.3M", n_layer=14, d_model=224, n_head=14),
    ModelSpec("12.3M", n_layer=15, d_model=240, n_head=15),
    ModelSpec("15.5M", n_layer=17, d_model=256, n_head=16),
    ModelSpec("19.1M", n_layer=19, d_model=272, n_head=17),
    ModelSpec("22.3M", n_layer=20, d_model=288, n_head=18),
    ModelSpec("24.3M", n_layer=22, d_model=288, n_head=18),
    ModelSpec("28.1M", n_layer=23, d_model=304, n_head=19),
    ModelSpec("35.3M", n_layer=24, d_model=336, n_head=21),
)
# Larger causal-AR candidates continue the same fixed-16 head dimension.  They
# are spaced by about 1.25x so the Chinchilla D/N=20 seed and either adjacent
# direction can be evaluated without changing architecture families mid-curve.
AR_LARGE_MODEL_SPECS = (
    ModelSpec("43.7M", n_layer=25, d_model=368, n_head=23),
    ModelSpec("54.5M", n_layer=29, d_model=384, n_head=24),
    ModelSpec("68.5M", n_layer=29, d_model=432, n_head=27),
    ModelSpec("85.8M", n_layer=34, d_model=448, n_head=28),
    ModelSpec("107.7M", n_layer=35, d_model=496, n_head=31),
    ModelSpec("134.8M", n_layer=39, d_model=528, n_head=33),
    # One-shot p_AR=0.4 extrapolation at 1e19 from the final four profiles.
    ModelSpec("176.0M", n_layer=43, d_model=576, n_head=36),
)
# Targeted batch-64 historical refinements. The first two points are robust
# to the remaining uncertainty in the 1e18 vertex; later points are added
# only after the completed rolling law fixes their targets.
REFINEMENT_MODEL_SPECS = (
    ModelSpec("0.21M", n_layer=2, d_model=24, n_head=1),
    ModelSpec("0.35M", n_layer=7, d_model=32, n_head=2),
    ModelSpec("0.45M", n_layer=2, d_model=48, n_head=3),
    ModelSpec("0.73M", n_layer=4, d_model=64, n_head=4),
    ModelSpec("1.20M", n_layer=7, d_model=80, n_head=5),
    ModelSpec("1.34M", n_layer=5, d_model=96, n_head=6),
    ModelSpec("1.56M", n_layer=7, d_model=96, n_head=6),
    ModelSpec("1.67M", n_layer=8, d_model=96, n_head=6),
    ModelSpec("2.60M", n_layer=11, d_model=112, n_head=7),
)
# Experimental extension for the fixed-token block-size study. It is kept out
# of MODEL_SPECS so the completed 1/2/4/8M IsoFLOP grid cannot silently grow.
EXPERIMENTAL_MODEL_SPECS = (
    ModelSpec("10M", n_layer=13, d_model=224, n_head=14),
    # Fixed architecture for the 25M-unique-token data-efficiency study.
    ModelSpec("50.0M", n_layer=29, d_model=368, n_head=23),
)
ALL_MODEL_SPECS = tuple(
    sorted(
        MODEL_SPECS
        + SCALEUP_MODEL_SPECS
        + AR_LARGE_MODEL_SPECS
        + REFINEMENT_MODEL_SPECS
        + EXPERIMENTAL_MODEL_SPECS,
        key=lambda spec: spec.n_params,
    )
)
MODEL_BY_LABEL = {spec.label: spec for spec in ALL_MODEL_SPECS}


def steps_for(budget: float, spec: ModelSpec) -> int:
    return int(budget / (spec.training_flops_per_clean_token * TOKENS_PER_STEP))


def is_feasible(budget: float, spec: ModelSpec) -> bool:
    return MIN_STEPS <= steps_for(budget, spec) <= MAX_STEPS


def realized_tokens(steps: int) -> int:
    return steps * TOKENS_PER_STEP


def realized_flops(steps: int, spec: ModelSpec) -> int:
    return realized_tokens(steps) * spec.training_flops_per_clean_token


def budget_slug(budget: float) -> str:
    return f"{budget:.0e}".replace("+", "")


def lr_slug(lr: float) -> str:
    return f"{lr:.1e}".replace("+", "")


def validate_config() -> None:
    assert BASE_VOCAB_SIZE + 1 == VOCAB_SIZE
    assert SEQ_LEN % BLOCK_LEN == 0
    assert len(set(spec.label for spec in ALL_MODEL_SPECS)) == len(ALL_MODEL_SPECS)
    for spec in ALL_MODEL_SPECS:
        assert spec.n_layer >= 2
        assert spec.d_model % spec.n_head == 0
        if spec in REFINEMENT_MODEL_SPECS:
            assert spec.head_dim in (16, 24)
        else:
            assert spec.head_dim == 16
    for left, right in zip(ALL_MODEL_SPECS, ALL_MODEL_SPECS[1:]):
        assert left.n_params < right.n_params
    coverage = [
        sum(is_feasible(budget, spec) for spec in MODEL_SPECS)
        for budget in COMPUTE_BUDGETS
    ]
    assert min(coverage) >= 3, coverage


validate_config()
