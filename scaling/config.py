"""Single source of truth for the TinyShakespeare IsoFLOP experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DATA_PATH = REPO_ROOT / "data.txt"
# The original proxy-C sweep remains in results/ and figures/. The active
# experiment uses architecture-aware dense-attention FLOPs and writes to
# separate directories so the two accounting schemes remain auditable.
RESULTS_DIR = ROOT / "results_dense_attention"
FIGURES_DIR = ROOT / "figures_dense_attention"

VOCAB_SIZE = 66
SEQ_LEN = 256
BLOCK_LEN = 4
BATCH_SIZE = 128
TOKENS_PER_STEP = BATCH_SIZE * SEQ_LEN

# Three times forward matmul FLOPs approximates forward + backward. Transformer
# projections/MLPs process both streams; dense attention processes the full
# 2L-by-2L matrix; the LM head is evaluated only on the noisy stream.
FLOP_MULTIPLIER = 12
COMPUTE_ACCOUNTING = "dense_dual_stream_attention_v1"
COMPUTE_BUDGETS = (1e13, 3e13, 1e14, 3e14, 1e15)
MIN_STEPS = 150
MAX_STEPS = 25_000

# Four peak LRs with an exact 3x spacing. Every feasible (C, N) point is run
# at all four values; analysis selects the lowest validation NELBO.
LEARNING_RATES = (1e-3, 3e-3, 9e-3, 2.7e-2)
WEIGHT_DECAY = 0.1
WARMUP_FRACTION = 0.05
DECAY_FRACTION = 0.15
MIN_LR_RATIO = 0.0
GRAD_CLIP = 1.0
SEED = 1337
MASK_EPS = 1e-3
EVAL_BATCH_SIZE = 128
EVAL_BATCHES = 20


def round_up(value: float, multiple: int) -> int:
    return int((int(value + multiple - 1) // multiple) * multiple)


def llama_ffn_dim(d_model: int) -> int:
    """Llama's 8d/3 SwiGLU width, scaled to a multiple of eight."""
    return round_up(8 * d_model / 3, 8)


def counted_parameters(d_model: int, n_layer: int, d_ff: int) -> int:
    """Exclude token embeddings; include untied LM head and learned norms."""
    per_layer = 4 * d_model**2 + 3 * d_model * d_ff + 2 * d_model
    return n_layer * per_layer + VOCAB_SIZE * d_model + d_model


def layer_matrix_parameters(d_model: int, n_layer: int, d_ff: int) -> int:
    """Attention projections plus SwiGLU matrices; excludes norms and head."""
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
    def n_params(self) -> int:
        return counted_parameters(self.d_model, self.n_layer, self.d_ff)

    @property
    def layer_matrix_params(self) -> int:
        return layer_matrix_parameters(self.d_model, self.n_layer, self.d_ff)

    @property
    def training_flops_per_clean_token(self) -> int:
        """Leading matmul FLOPs for forward + backward per clean token.

        12P: projections and MLPs over the two L-token streams.
        48Lnd: dense QK^T and AV attention over sequence length 2L.
        6dV: LM head over only the L noisy positions.
        """
        block_linears = FLOP_MULTIPLIER * self.layer_matrix_params
        dense_attention = 48 * SEQ_LEN * self.n_layer * self.d_model
        lm_head = 6 * self.d_model * VOCAB_SIZE
        return block_linears + dense_attention + lm_head

    @property
    def autoregressive_training_flops_per_clean_token(self) -> int:
        """Leading matmul FLOPs for a single-stream causal AR update.

        The transformer matrices process one L-token stream, dense causal
        attention still executes an L-by-L kernel, and the LM head predicts
        every clean token.
        """
        block_linears = 6 * self.layer_matrix_params
        dense_attention = 12 * SEQ_LEN * self.n_layer * self.d_model
        lm_head = 6 * self.d_model * VOCAB_SIZE
        return block_linears + dense_attention + lm_head

    @property
    def effective_compute_parameters(self) -> float:
        return self.training_flops_per_clean_token / FLOP_MULTIPLIER


MODEL_SPECS = (
    ModelSpec("0.002M", n_layer=2, d_model=8, n_head=1),
    ModelSpec("0.005M", n_layer=2, d_model=12, n_head=2),
    ModelSpec("0.01M", n_layer=3, d_model=16, n_head=2),
    ModelSpec("0.02M", n_layer=3, d_model=24, n_head=3),
    ModelSpec("0.04M", n_layer=3, d_model=32, n_head=4),
    ModelSpec("0.1M", n_layer=4, d_model=48, n_head=3),
    ModelSpec("0.2M", n_layer=4, d_model=64, n_head=4),
    ModelSpec("0.4M", n_layer=5, d_model=80, n_head=5),
    ModelSpec("0.8M", n_layer=7, d_model=96, n_head=6),
    ModelSpec("1.6M", n_layer=8, d_model=128, n_head=8),
)
MODEL_BY_LABEL = {spec.label: spec for spec in MODEL_SPECS}

# Adaptive model-size coverage: this extra-small model is only required to
# bracket the lowest-compute IsoFLOP minimum.
TARGETED_BUDGETS = {
    "0.002M": (1e13,),
    "0.005M": (1e13,),
}


def steps_for(budget: float, spec: ModelSpec) -> int:
    return int(budget / (spec.training_flops_per_clean_token * TOKENS_PER_STEP))


def is_feasible(budget: float, spec: ModelSpec) -> bool:
    if spec.label in TARGETED_BUDGETS and budget not in TARGETED_BUDGETS[spec.label]:
        return False
    return MIN_STEPS <= steps_for(budget, spec) <= MAX_STEPS


def realized_tokens(steps: int) -> int:
    return steps * TOKENS_PER_STEP


def realized_flops(steps: int, spec: ModelSpec) -> int:
    return spec.training_flops_per_clean_token * realized_tokens(steps)


def budget_slug(budget: float) -> str:
    return f"{budget:.0e}".replace("+", "")


def lr_slug(lr: float) -> str:
    return f"{lr:.1e}".replace("+", "")


def validate_config() -> None:
    labels = set()
    for spec in MODEL_SPECS:
        assert spec.label not in labels
        labels.add(spec.label)
        assert spec.n_layer >= 2
        assert spec.d_model % spec.n_head == 0
        assert spec.head_dim % 2 == 0
        if spec.n_params > 40_000:
            assert spec.head_dim == 16
    for left, right in zip(MODEL_SPECS, MODEL_SPECS[1:]):
        assert left.n_params < right.n_params


validate_config()
