"""Compute-matched BD-prefix control for the p_AR=0.06 diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from config import BATCH_SIZE, MODEL_BY_LABEL, SEED, TOKENS_PER_STEP
from curriculum_ar_trunk import set_lr
from curriculum_config import phase_steps_for
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
P_AR_REFERENCE = 0.06
PEAK_LR = 8.1e-3
MILESTONES = (0, 500, 1_000, 2_000)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--random-tail",
        action="store_true",
        help="Skip the compute-matched BD prefix and train the common tail from init",
    )
    return parser.parse_args()


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def preserving_rng(function):
    state = capture_rng()
    try:
        return function()
    finally:
        restore_rng(state)


def main() -> None:
    args = parse_args()
    spec = MODEL_BY_LABEL[SIZE]
    ar_steps, bd_tail_steps = phase_steps_for(BUDGET, spec, P_AR_REFERENCE)
    bd_prefix_steps = (
        0
        if args.random_tail
        else math.floor(
            ar_steps
            * spec.autoregressive_training_flops_per_clean_token
            / spec.training_flops_per_clean_token
        )
    )
    device = torch.device("cuda")
    set_seed(SEED)
    dataset = ClimbMixData.load(device)
    model = BlockDiffusionTransformer(spec).to(device)
    tail_rng_state = capture_rng()

    prefix_duration = 0.0
    if bd_prefix_steps:
        prefix_optimizer = optimizer_for(model, PEAK_LR)
        model.train()
        prefix_started = time.monotonic()
        for step in range(bd_prefix_steps):
            learning_rate = wsd_learning_rate(step, bd_prefix_steps, PEAK_LR)
            set_lr(prefix_optimizer, learning_rate)
            x0 = dataset.train_batch(step, BATCH_SIZE)
            probabilities = sample_mask_probabilities(BATCH_SIZE, device)
            xt, masked, token_probability = corrupt(x0, probabilities)
            with autocast_context(device):
                logits = model(xt, x0)
                loss = diffusion_nelbo(logits, x0, masked, token_probability)
            prefix_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            prefix_optimizer.step()
        prefix_duration = time.monotonic() - prefix_started
        del prefix_optimizer
    tail_optimizer = optimizer_for(model, PEAK_LR)
    # AR consumes no model-side randomness, so restore the post-init state to
    # give this control the same corruption stream as the AR->BD branch.
    restore_rng(tail_rng_state)
    validation_trace = []
    bd_trace = []

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
            f"BD-prefix control tail={step:>5}/{bd_tail_steps} "
            f"val={val_nelbo:.5f}",
            flush=True,
        )

    record_validation(0)
    tail_started = time.monotonic()
    last_loss = math.nan
    log_interval = max(1, bd_tail_steps // 20)
    for step in range(bd_tail_steps):
        learning_rate = wsd_learning_rate(step, bd_tail_steps, PEAK_LR)
        set_lr(tail_optimizer, learning_rate)
        # Match the AR branch's tail data exactly; the skipped interval is the
        # additional clean-token exposure bought by cheaper AR steps.
        x0 = dataset.train_batch(ar_steps + step, BATCH_SIZE)
        probabilities = sample_mask_probabilities(BATCH_SIZE, device)
        xt, masked, token_probability = corrupt(x0, probabilities)
        with autocast_context(device):
            logits = model(xt, x0)
            loss = diffusion_nelbo(logits, x0, masked, token_probability)
        tail_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        tail_optimizer.step()
        last_loss = loss.item()
        completed = step + 1
        if step == 0 or completed % log_interval == 0 or completed == bd_tail_steps:
            bd_trace.append(
                {
                    "step": completed,
                    "train_nelbo": last_loss,
                    "learning_rate": learning_rate,
                    "grad_norm": float(grad_norm),
                }
            )
        if completed in MILESTONES:
            record_validation(completed)
    tail_duration = time.monotonic() - tail_started
    if validation_trace[-1]["bd_step"] != bd_tail_steps:
        record_validation(bd_tail_steps)

    realized_flops = (
        (bd_prefix_steps + bd_tail_steps)
        * TOKENS_PER_STEP
        * spec.training_flops_per_clean_token
    )
    payload = {
        "status": "complete",
        "purpose": (
            "random_init_common_bd_tail_control"
            if args.random_tail
            else "compute_matched_bd_prefix_control"
        ),
        "budget": BUDGET,
        "size": SIZE,
        "p_ar_reference": P_AR_REFERENCE,
        "ar_prefix_steps_replaced": ar_steps,
        "bd_prefix_steps": bd_prefix_steps,
        "bd_tail_steps": bd_tail_steps,
        "learning_rate": PEAK_LR,
        "optimizer_reset": True,
        "prefix_schedule": "phase_local_wsd_5_80_15",
        "tail_schedule": "phase_local_wsd_5_80_15",
        "realized_flops": realized_flops,
        "val_nelbo": validation_trace[-1]["val_nelbo"],
        "val_masked_ce_t0.5": validation_trace[-1]["val_masked_ce_t0.5"],
        "last_train_nelbo": last_loss,
        "prefix_duration_seconds": prefix_duration,
        "tail_duration_seconds": tail_duration,
        "validation_trace": validation_trace,
        "bd_trace": bd_trace,
        "seed": SEED,
    }
    output = Path(
        "results_diagnostics/low_p_1e15_0p5M/"
        + (
            "random_init_common_bd_tail_p_ar_0p06.json"
            if args.random_tail
            else "compute_matched_bd_prefix_p_ar_0p06.json"
        )
    )
    atomic_json_dump(payload, output)
    print(
        f"complete BD-prefix control prefix={bd_prefix_steps} "
        f"tail={bd_tail_steps} val={payload['val_nelbo']:.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
