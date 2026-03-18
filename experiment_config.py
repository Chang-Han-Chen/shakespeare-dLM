"""
experiment_config.py — Shared configuration for all experiment runners.

Centralizes MODEL_SIZES, OPTIMAL_LR, model metadata, and command building
so that sweep.py, run_scaling.py, run_isoflop.py, and generate_samples.py
all import from one place.
"""

import os
import sys

# ---------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------

MODEL_MODULE_MAP = {
    "remasked":            "model_remasked",
    "mdlm":                "model_MDLM",
    "edit_one_pass":       "model_edit_one_pass",
    "edit_two_pass":       "model_edit_two_pass",
    "ar":                  "model_AR",
    "block_remasked":      "model_block_remasked",
    "block_mdlm":          "model_block_MDLM",
    "block_edit_one_pass": "model_block_edit_one_pass",
    "block_edit_two_pass": "model_block_edit_two_pass",
}

DIFFUSION_MODELS = [
    "remasked", "mdlm", "edit_one_pass", "edit_two_pass",
]
BLOCK_MODELS = [
    "block_remasked", "block_mdlm", "block_edit_one_pass", "block_edit_two_pass",
]
ALL_MODELS = DIFFUSION_MODELS + ["ar"] + BLOCK_MODELS

BLOCK_MODEL_SET = set(BLOCK_MODELS)


def is_block_model(model):
    return model in BLOCK_MODEL_SET


# ---------------------------------------------------------------
# Model sizes: label -> (n_embd, n_layer, approx N_params)
# ---------------------------------------------------------------

MODEL_SIZES = {
    "0.1M":  (64,   2,  0.107e6),
    "0.3M":  (96,   3,  0.345e6),
    "0.5M":  (120,  3,  0.535e6),
    "1M":    (128,  5,  1.002e6),
    "2M":    (200,  4,  1.949e6),
    "3M":    (256,  4,  3.182e6),
}

ALL_SIZES = list(MODEL_SIZES.keys())

DEFAULT_BLOCK_LENS = [4, 16, 64]


# ---------------------------------------------------------------
# FLOP accounting
# ---------------------------------------------------------------

def flop_multiplier(model):
    """
    Approximate C in FLOPs = C * N * tokens_processed.

    Standard diffusion: C = 6 (one fwd+bwd).
    edit_two_pass: C = 12 (two fwd+bwd per step).
    Block models during training use a 2L-length input (dual-stream),
    so attention cost roughly doubles: C ≈ 12.
    block_edit_two_pass: C ≈ 24.
    """
    if "edit_two_pass" in model:
        return 24 if is_block_model(model) else 12
    return 12 if is_block_model(model) else 6


def dropout_for_model(model, default_dropout=0.1):
    """AR uses 0.2 by default; diffusion models use 0.1."""
    return 0.2 if model == "ar" else default_dropout


# ---------------------------------------------------------------
# Optimal LR per (model, size)
# ---------------------------------------------------------------

# From sweep at dropout=0.1 (diffusion) / 0.2 (AR).
# New sizes (0.5M, 2M) interpolated from neighbors.
# Block models initially inherit from their non-block counterparts.

_BASE_LR = {
    # remasked
    ("remasked",      "0.1M"): 1e-2,
    ("remasked",      "0.3M"): 3e-3,
    ("remasked",      "0.5M"): 3e-3,
    ("remasked",      "1M"):   3e-3,
    ("remasked",      "2M"):   3e-3,
    ("remasked",      "3M"):   3e-3,

    # mdlm
    ("mdlm",          "0.1M"): 1e-2,
    ("mdlm",          "0.3M"): 3e-3,
    ("mdlm",          "0.5M"): 3e-3,
    ("mdlm",          "1M"):   3e-3,
    ("mdlm",          "2M"):   3e-3,
    ("mdlm",          "3M"):   3e-3,

    # edit_one_pass
    ("edit_one_pass",  "0.1M"): 1e-2,
    ("edit_one_pass",  "0.3M"): 1e-2,
    ("edit_one_pass",  "0.5M"): 1e-2,
    ("edit_one_pass",  "1M"):   1e-2,
    ("edit_one_pass",  "2M"):   3e-3,
    ("edit_one_pass",  "3M"):   3e-3,

    # edit_two_pass
    ("edit_two_pass",  "0.1M"): 1e-2,
    ("edit_two_pass",  "0.3M"): 3e-3,
    ("edit_two_pass",  "0.5M"): 3e-3,
    ("edit_two_pass",  "1M"):   3e-3,
    ("edit_two_pass",  "2M"):   3e-3,
    ("edit_two_pass",  "3M"):   3e-3,

    # ar
    ("ar",            "0.1M"): 1e-2,
    ("ar",            "0.3M"): 1e-2,
    ("ar",            "0.5M"): 1e-2,
    ("ar",            "1M"):   1e-2,
    ("ar",            "2M"):   3e-3,
    ("ar",            "3M"):   3e-3,
}

# Block models inherit from their non-block counterpart.
_BLOCK_PREFIX_MAP = {
    "block_remasked":      "remasked",
    "block_mdlm":          "mdlm",
    "block_edit_one_pass": "edit_one_pass",
    "block_edit_two_pass": "edit_two_pass",
}

OPTIMAL_LR = dict(_BASE_LR)
for bm, base in _BLOCK_PREFIX_MAP.items():
    for size in ALL_SIZES:
        if (base, size) in _BASE_LR:
            OPTIMAL_LR[(bm, size)] = _BASE_LR[(base, size)]


def get_optimal_lr(model, size):
    return OPTIMAL_LR.get((model, size))


# ---------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------

def build_command(
    model,
    size,
    out_dir,
    *,
    max_iters=4000,
    batch_size=128,
    block_size=256,
    block_len=None,
    dropout=None,
    lr=None,
    eval_interval=300,
    eval_iters=50,
    warmup_iters=100,
    gpt2_eval_interval=0,
    gpt2_eval_samples=0,
    save_interval=1000,
    num_final_samples=5,
    sample_interval=0,
):
    n_embd, n_layer, _ = MODEL_SIZES[size]
    if lr is None:
        lr = get_optimal_lr(model, size)
    if lr is None:
        raise ValueError(f"No LR for ({model}, {size})")
    min_lr = lr / 10.0
    if dropout is None:
        dropout = dropout_for_model(model)

    loss_path = os.path.join(out_dir, "loss.pkl")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")

    cmd = [
        sys.executable, "train.py",
        "--model", model,
        "--n_embd", str(n_embd),
        "--n_layer", str(n_layer),
        "--n_head", "4",
        "--dropout", str(dropout),
        "--learning_rate", str(lr),
        "--min_lr", str(min_lr),
        "--batch_size", str(batch_size),
        "--block_size", str(block_size),
        "--max_iters", str(max_iters),
        "--eval_interval", str(eval_interval),
        "--eval_iters", str(eval_iters),
        "--warmup_iters", str(warmup_iters),
        "--gpt2_eval_interval", str(gpt2_eval_interval),
        "--gpt2_eval_samples", str(gpt2_eval_samples),
        "--sample_interval", str(sample_interval),
        "--save_interval", str(save_interval) if save_interval > 0 else str(max_iters + 1),
        "--num_final_samples", str(num_final_samples),
        "--loss_log_path", loss_path,
        "--checkpoint_path", ckpt_path,
    ]

    if block_len is not None:
        cmd.extend(["--block_len", str(block_len)])

    return cmd
