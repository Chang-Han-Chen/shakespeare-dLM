"""Fast low-p_AR diagnostic for C=1e15, N=0.5M."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from config import (
    BATCH_SIZE,
    BLOCK_LEN,
    COMPUTE_ACCOUNTING,
    MODEL_BY_LABEL,
    SEED,
    SEQ_LEN,
    TOKENS_PER_STEP,
    WEIGHT_DECAY,
)
from curriculum_ar_trunk import ar_loss, evaluate_ar, set_lr
from curriculum_config import phase_steps_for, realized_flops
from data import ClimbMixData, corrupt, sample_mask_probabilities
from model import BlockDiffusionTransformer
from train import (
    atomic_json_dump,
    autocast_context,
    diffusion_nelbo,
    evaluate,
    optimizer_for,
    set_seed,
    wsd_learning_rate,
)


BUDGET = 1e15
SIZE = "0.5M"
DEFAULT_PEAK_LR = 8.1e-3
MILESTONES = (0, 500, 1_000, 2_000)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-ar", type=float, required=True)
    parser.add_argument("--ar-lr", type=float, default=DEFAULT_PEAK_LR)
    parser.add_argument("--bd-lr", type=float, default=DEFAULT_PEAK_LR)
    parser.add_argument(
        "--ar-no-decay",
        action="store_true",
        help="Keep AR at peak LR after warmup instead of applying the final 15% decay",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def preserving_rng(function):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        return function()
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def main() -> None:
    args = parse_args()
    if not 0 < args.p_ar < 1:
        raise ValueError("p_ar must be between zero and one")
    spec = MODEL_BY_LABEL[SIZE]
    ar_steps, bd_steps = phase_steps_for(BUDGET, spec, args.p_ar)
    device = torch.device(args.device)
    set_seed(SEED)
    dataset = ClimbMixData.load(device)
    model = BlockDiffusionTransformer(spec).to(device)

    ar_optimizer = optimizer_for(model, args.ar_lr)
    ar_trace = []
    ar_log_interval = max(1, ar_steps // 10)
    model.train()
    started = time.monotonic()
    ar_warmup_steps = max(1, round(0.05 * ar_steps))
    for step in range(ar_steps):
        learning_rate = (
            args.ar_lr * min(1.0, (step + 1) / ar_warmup_steps)
            if args.ar_no_decay
            else wsd_learning_rate(step, ar_steps, args.ar_lr)
        )
        set_lr(ar_optimizer, learning_rate)
        inputs, targets = dataset.autoregressive_train_batch(step, BATCH_SIZE)
        with autocast_context(device):
            loss = ar_loss(model.forward_ar(inputs), targets)
        ar_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        ar_optimizer.step()
        completed = step + 1
        if step == 0 or completed % ar_log_interval == 0 or completed == ar_steps:
            ar_trace.append(
                {
                    "step": completed,
                    "train_ar_ce": loss.item(),
                    "learning_rate": learning_rate,
                    "grad_norm": float(grad_norm),
                }
            )
    ar_duration = time.monotonic() - started
    val_ar_ce = preserving_rng(lambda: evaluate_ar(model, dataset, device))

    del ar_optimizer
    bd_optimizer = optimizer_for(model, args.bd_lr)
    bd_trace = []
    validation_trace = []

    def record_validation(step):
        val_nelbo, val_masked_ce = preserving_rng(
            lambda: evaluate(model, dataset, device)
        )
        validation_trace.append(
            {
                "bd_step": step,
                "val_nelbo": val_nelbo,
                "val_masked_ce_t0.5": val_masked_ce,
            }
        )
        model.train()
        print(
            f"p={args.p_ar:.2f} BD={step:>5}/{bd_steps} "
            f"val={val_nelbo:.5f}",
            flush=True,
        )

    record_validation(0)
    bd_log_interval = max(1, bd_steps // 20)
    started = time.monotonic()
    last_bd_loss = math.nan
    for step in range(bd_steps):
        learning_rate = wsd_learning_rate(step, bd_steps, args.bd_lr)
        set_lr(bd_optimizer, learning_rate)
        x0 = dataset.train_batch(ar_steps + step, BATCH_SIZE)
        probabilities = sample_mask_probabilities(BATCH_SIZE, device)
        xt, masked, token_probability = corrupt(x0, probabilities)
        with autocast_context(device):
            logits = model(xt, x0)
            loss = diffusion_nelbo(logits, x0, masked, token_probability)
        bd_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        bd_optimizer.step()
        last_bd_loss = loss.item()
        completed = step + 1
        if step == 0 or completed % bd_log_interval == 0 or completed == bd_steps:
            bd_trace.append(
                {
                    "step": completed,
                    "train_nelbo": last_bd_loss,
                    "learning_rate": learning_rate,
                    "grad_norm": float(grad_norm),
                }
            )
        if completed in MILESTONES:
            record_validation(completed)
    bd_duration = time.monotonic() - started
    if validation_trace[-1]["bd_step"] != bd_steps:
        record_validation(bd_steps)

    payload = {
        "status": "complete",
        "purpose": "low_p_ar_chinchilla20_diagnostic",
        "budget": BUDGET,
        "size": SIZE,
        "n_params": spec.n_params,
        "p_ar": args.p_ar,
        "ar_learning_rate": args.ar_lr,
        "bd_learning_rate": args.bd_lr,
        "weight_decay": WEIGHT_DECAY,
        "optimizer_reset_at_transition": True,
        "ar_schedule": (
            "phase_local_5pct_warmup_then_stable_no_decay"
            if args.ar_no_decay
            else "phase_local_wsd_5_80_15"
        ),
        "bd_schedule": "phase_local_wsd_5_80_15",
        "ar_steps": ar_steps,
        "bd_steps": bd_steps,
        "total_steps": ar_steps + bd_steps,
        "ar_tokens_per_parameter": ar_steps * TOKENS_PER_STEP / spec.n_params,
        "clean_tokens": (ar_steps + bd_steps) * TOKENS_PER_STEP,
        "realized_flops": realized_flops(ar_steps, bd_steps, spec),
        "compute_accounting": COMPUTE_ACCOUNTING,
        "sequence_length": SEQ_LEN,
        "block_len": BLOCK_LEN,
        "batch_size": BATCH_SIZE,
        "val_ar_ce_at_transition": val_ar_ce,
        "val_nelbo": validation_trace[-1]["val_nelbo"],
        "val_masked_ce_t0.5": validation_trace[-1]["val_masked_ce_t0.5"],
        "last_bd_train_nelbo": last_bd_loss,
        "ar_duration_seconds": ar_duration,
        "bd_duration_seconds": bd_duration,
        "ar_trace": ar_trace,
        "bd_trace": bd_trace,
        "validation_trace": validation_trace,
        "seed": SEED,
    }
    atomic_json_dump(payload, args.output)
    print(
        f"complete p={args.p_ar:.2f} D_AR/N={payload['ar_tokens_per_parameter']:.1f} "
        f"val={payload['val_nelbo']:.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
