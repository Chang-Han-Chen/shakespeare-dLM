"""Finish one AR decay branch, reset AdamW, and train the BD phase."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

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
from curriculum_ar_trunk import ar_loss, set_lr
from curriculum_config import (
    ar_decay_start,
    ar_decay_steps,
    is_feasible,
    phase_steps_for,
    realized_flops,
)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--size", choices=tuple(MODEL_BY_LABEL), required=True)
    parser.add_argument("--p-ar", type=float, required=True)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Fix AR+BD optimizer steps instead of deriving them from FLOPs",
    )
    parser.add_argument("--trunk-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bd-lr",
        type=float,
        default=None,
        help="BD peak-LR override; defaults to the AR trunk LR",
    )
    parser.add_argument(
        "--ar-no-decay",
        action="store_true",
        help="Hold the AR LR at its peak through the final 15 percent",
    )
    parser.add_argument("--block-len", type=int, default=BLOCK_LEN)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.p_ar < 1.0:
        raise ValueError("p-ar must be strictly between zero and one")
    if args.block_len < 1 or SEQ_LEN % args.block_len:
        raise ValueError("block-len must be a positive divisor of sequence length")
    spec = MODEL_BY_LABEL[args.size]
    fixed_total_steps = args.total_steps is not None
    if fixed_total_steps:
        if args.total_steps < 1:
            raise ValueError("total-steps must be positive")
        ar_steps = round(args.p_ar * args.total_steps)
        bd_steps = args.total_steps - ar_steps
    else:
        if not is_feasible(args.budget, spec, args.p_ar):
            raise ValueError("Infeasible curriculum point")
        ar_steps, bd_steps = phase_steps_for(args.budget, spec, args.p_ar)
    decay_start = ar_decay_start(ar_steps)
    decay_count = ar_decay_steps(ar_steps)
    trunk = json.loads((args.trunk_dir / "trunk_result.json").read_text())
    expected_mode = "fixed_total_steps" if fixed_total_steps else "fixed_compute"
    if trunk.get("comparison_mode", "fixed_compute") != expected_mode:
        raise ValueError("AR trunk comparison mode mismatch")
    if fixed_total_steps and trunk.get("fixed_total_steps") != args.total_steps:
        raise ValueError("AR trunk total-step mismatch")
    ar_peak_lr = float(trunk["learning_rate"])
    bd_peak_lr = args.bd_lr if args.bd_lr is not None else ar_peak_lr
    checkpoint_path = Path(trunk["checkpoints_by_step"][str(decay_start)])

    device = torch.device(args.device)
    set_seed(SEED)
    dataset = ClimbMixData.load(device)
    total_steps = ar_steps + bd_steps
    if total_steps * TOKENS_PER_STEP + 1 > dataset.train.total_tokens:
        raise RuntimeError("Curriculum run would exceed one data epoch")
    model = BlockDiffusionTransformer(spec, block_len=args.block_len).to(device)
    ar_optimizer = optimizer_for(model, ar_peak_lr)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    ar_optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint["completed_ar_steps"] != decay_start:
        raise ValueError("AR checkpoint step mismatch")

    model.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ar_started = time.monotonic()
    last_ar_loss = math.nan
    for local, global_step in enumerate(range(decay_start, ar_steps)):
        if args.ar_no_decay:
            learning_rate = ar_peak_lr
        elif decay_count == 1:
            learning_rate = 0.0
        else:
            learning_rate = ar_peak_lr * (1.0 - local / (decay_count - 1))
        set_lr(ar_optimizer, learning_rate)
        inputs, targets = dataset.autoregressive_train_batch(global_step, BATCH_SIZE)
        with autocast_context(device):
            loss = ar_loss(model.forward_ar(inputs), targets)
        ar_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        ar_optimizer.step()
        last_ar_loss = loss.item()
        if not math.isfinite(last_ar_loss):
            raise FloatingPointError("Non-finite AR decay loss")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ar_decay_duration = time.monotonic() - ar_started

    # The requested transition resets all optimizer state.
    del ar_optimizer
    bd_optimizer = optimizer_for(model, bd_peak_lr)
    bd_trace = []
    log_interval = max(1, bd_steps // 20)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    bd_started = time.monotonic()
    last_bd_loss = math.nan
    for local_step in range(bd_steps):
        learning_rate = wsd_learning_rate(local_step, bd_steps, bd_peak_lr)
        set_lr(bd_optimizer, learning_rate)
        x0 = dataset.train_batch(ar_steps + local_step, BATCH_SIZE)
        probabilities = sample_mask_probabilities(
            BATCH_SIZE,
            device,
            args.block_len,
        )
        xt, masked, token_probability = corrupt(x0, probabilities)
        with autocast_context(device):
            logits = model(xt, x0)
            loss = diffusion_nelbo(logits, x0, masked, token_probability)
        bd_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        bd_optimizer.step()
        last_bd_loss = loss.item()
        if not math.isfinite(last_bd_loss):
            raise FloatingPointError("Non-finite BD loss")
        completed = local_step + 1
        if local_step == 0 or completed % log_interval == 0 or completed == bd_steps:
            bd_trace.append(
                {
                    "step": completed,
                    "train_nelbo": last_bd_loss,
                    "learning_rate": learning_rate,
                    "grad_norm": float(grad_norm),
                }
            )
            print(
                f"BD {completed:>6}/{bd_steps} loss={last_bd_loss:.4f} "
                f"lr={learning_rate:.6g}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    bd_duration = time.monotonic() - bd_started
    val_nelbo, val_masked_ce = evaluate(
        model,
        dataset,
        device,
        args.block_len,
    )

    payload = {
        "status": "complete",
        "comparison_mode": expected_mode,
        "budget": args.budget,
        "size": spec.label,
        "n_params": spec.n_params,
        "p_ar": args.p_ar,
        "learning_rate": bd_peak_lr,
        "ar_learning_rate": ar_peak_lr,
        "bd_learning_rate": bd_peak_lr,
        "weight_decay": WEIGHT_DECAY,
        "optimizer_reset_at_transition": True,
        "ar_end_decay": not args.ar_no_decay,
        "shared_ar_trunk": True,
        "shared_ar_warmup_steps": trunk["shared_warmup_steps"],
        "ar_steps": ar_steps,
        "ar_decay_start": decay_start,
        "ar_decay_steps": decay_count,
        "bd_steps": bd_steps,
        "total_steps": total_steps,
        "sequence_length": SEQ_LEN,
        "block_len": args.block_len,
        "batch_size": BATCH_SIZE,
        "clean_tokens": total_steps * TOKENS_PER_STEP,
        "effective_train_epochs": (
            total_steps * TOKENS_PER_STEP / dataset.train.total_tokens
        ),
        "realized_flops": realized_flops(ar_steps, bd_steps, spec),
        "compute_accounting": COMPUTE_ACCOUNTING,
        "autoregressive_training_flops_per_clean_token": (
            spec.autoregressive_training_flops_per_clean_token
        ),
        "block_diffusion_training_flops_per_clean_token": (
            spec.training_flops_per_clean_token
        ),
        "last_ar_train_ce": last_ar_loss,
        "last_bd_train_nelbo": last_bd_loss,
        "val_nelbo": val_nelbo,
        "val_masked_ce_t0.5": val_masked_ce,
        "ar_decay_duration_seconds": ar_decay_duration,
        "bd_training_duration_seconds": bd_duration,
        "bd_seconds_per_step": bd_duration / bd_steps,
        "bd_trace": bd_trace,
        "seed": SEED,
    }
    atomic_json_dump(payload, args.output)
    print(
        f"complete p={args.p_ar:.1f} val={val_nelbo:.5f} "
        f"AR_decay_s={ar_decay_duration:.1f} BD_s={bd_duration:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
