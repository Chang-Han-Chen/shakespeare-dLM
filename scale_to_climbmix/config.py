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

# NVIDIA's official files contain GPT-2 token IDs. This shuffled raw-text
# conversion makes it practical to train and apply our own BPE tokenizer.
DATASET_REPO = "karpathy/climbmix-400b-shuffle"
DATASET_REVISION = "915333b4f8b8684f39aeaafea600fea6f43fb703"
VALIDATION_SHARDS = (0,)
TRAIN_SHARDS = tuple(range(1, 26))
TOKENIZER_TRAIN_SHARDS = (1, 2, 3, 4)

# 8,192 ordinary/EOT tokens plus a diffusion-only [MASK] token.
BASE_VOCAB_SIZE = 8_192
VOCAB_SIZE = 8_193
EOT_TOKEN = "<|endoftext|>"
MASK_TOKEN = "<|mask|>"
MASK_ID = BASE_VOCAB_SIZE

SEQ_LEN = 256
BLOCK_LEN = 4
BATCH_SIZE = 64
TOKENS_PER_STEP = BATCH_SIZE * SEQ_LEN

FLOP_MULTIPLIER = 12
COMPUTE_ACCOUNTING = "dense_dual_stream_attention_v1"
COMPUTE_BUDGETS = (1e14, 3e14, 1e15, 3e15, 1e16)
MIN_STEPS = 150
MAX_STEPS = 25_000

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
        """Leading matmul FLOPs for a single-stream causal AR update."""
        block_linears = 6 * self.layer_matrix_params
        dense_attention = 12 * SEQ_LEN * self.n_layer * self.d_model
        lm_head = 6 * self.d_model * VOCAB_SIZE
        return block_linears + dense_attention + lm_head

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
# Experimental extension for the fixed-token block-size study. It is kept out
# of MODEL_SPECS so the completed 1/2/4/8M IsoFLOP grid cannot silently grow.
EXPERIMENTAL_MODEL_SPECS = (
    ModelSpec("10M", n_layer=13, d_model=224, n_head=14),
)
ALL_MODEL_SPECS = MODEL_SPECS + EXPERIMENTAL_MODEL_SPECS
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
        assert spec.head_dim == 16
    for left, right in zip(ALL_MODEL_SPECS, ALL_MODEL_SPECS[1:]):
        assert left.n_params < right.n_params
    coverage = [
        sum(is_feasible(budget, spec) for spec in MODEL_SPECS)
        for budget in COMPUTE_BUDGETS
    ]
    assert min(coverage) >= 3, coverage


validate_config()
