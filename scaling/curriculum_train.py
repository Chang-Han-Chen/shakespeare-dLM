"""Train one AR-warm-start to block-diffusion curriculum configuration."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from config import (
    BATCH_SIZE,
    BLOCK_LEN,
    DECAY_FRACTION,
    GRAD_CLIP,
    MODEL_BY_LABEL,
    SEED,
    SEQ_LEN,
    TOKENS_PER_STEP,
    VOCAB_SIZE,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
)
from curriculum_config import (
    COMPUTE_ACCOUNTING,
    average_training_flops_per_clean_token,
    is_feasible,
    realized_flops,
    split_phase_steps,
    steps_for,
)
from data import ShakespeareData, corrupt, sample_mask_probabilities
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
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--lr-source", default="unspecified")
    parser.add_argument(
        "--ar-weight-decay",
        type=float,
        default=WEIGHT_DECAY,
        help="AdamW decay for matrix parameters during the AR phase.",
    )
    parser.add_argument(
        "--bd-weight-decay",
        type=float,
        default=WEIGHT_DECAY,
        help="AdamW decay for matrix parameters during the BD phase.",
    )
    parser.add_argument(
        "--bd-warmup-fraction",
        type=float,
        default=WARMUP_FRACTION,
        help="Block-diffusion restart warmup; AR remains at the configured 5%%.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None, help="Test-only override")
    parser.add_argument("--block-len", type=int, default=BLOCK_LEN)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def ar_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.float().reshape(-1, VOCAB_SIZE),
        targets.reshape(-1),
    )


def train_ar_step(model, optimizer, dataset, device):
    inputs, targets = dataset.autoregressive_batch("train", BATCH_SIZE)
    with autocast_context(device):
        logits = model.forward_ar(inputs)
        loss = ar_cross_entropy(logits, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    return float(loss.item()), float(grad_norm)


def train_bd_step(model, optimizer, dataset, device, block_len):
    x0 = dataset.batch("train", BATCH_SIZE)
    probabilities = sample_mask_probabilities(BATCH_SIZE, device, block_len)
    xt, masked, token_probability = corrupt(x0, probabilities)
    with autocast_context(device):
        logits = model(xt, x0)
        loss = diffusion_nelbo(logits, x0, masked, token_probability)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    return float(loss.item()), float(grad_norm)


def run_phase(
    *,
    phase,
    phase_steps,
    model,
    optimizer,
    dataset,
    device,
    peak_lr,
    warmup_fraction,
    global_offset,
    trace,
    block_len=BLOCK_LEN,
):
    train_step = (
        train_ar_step
        if phase == "ar"
        else lambda model, optimizer, dataset, device: train_bd_step(
            model,
            optimizer,
            dataset,
            device,
            block_len,
        )
    )
    log_interval = max(1, phase_steps // 10)
    trace_interval = max(1, phase_steps // 100)
    last_loss = math.nan
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    phase_started = time.monotonic()
    for local_step in range(phase_steps):
        lr = wsd_learning_rate(
            local_step,
            phase_steps,
            peak_lr,
            warmup_fraction=warmup_fraction,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        last_loss, grad_norm = train_step(model, optimizer, dataset, device)
        if not math.isfinite(last_loss):
            raise FloatingPointError(
                f"Non-finite {phase} loss at local step {local_step}: {last_loss}"
            )
        should_trace = (
            local_step == 0
            or (local_step + 1) % trace_interval == 0
            or local_step + 1 == phase_steps
        )
        if should_trace:
            trace.append(
                {
                    "global_step": global_offset + local_step + 1,
                    "phase": phase,
                    "phase_step": local_step + 1,
                    "loss": last_loss,
                    "learning_rate": lr,
                    "grad_norm": grad_norm,
                }
            )
        if (
            local_step == 0
            or (local_step + 1) % log_interval == 0
            or local_step + 1 == phase_steps
        ):
            print(
                f"phase={phase} step {local_step + 1:>6}/{phase_steps} "
                f"global={global_offset + local_step + 1} "
                f"loss={last_loss:.4f} lr={lr:.6g} grad={grad_norm:.3f}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    phase_duration = time.monotonic() - phase_started
    return last_loss, phase_duration


def main() -> None:
    args = parse_args()
    if not 0.0 < args.p_ar < 1.0:
        raise ValueError("curriculum p_ar must be strictly between zero and one")
    if args.block_len < 1 or SEQ_LEN % args.block_len:
        raise ValueError("block-len must be a positive divisor of sequence length")
    if args.ar_weight_decay < 0 or args.bd_weight_decay < 0:
        raise ValueError("weight decay must be non-negative")
    spec = MODEL_BY_LABEL[args.size]
    planned_steps = steps_for(args.budget, spec, args.p_ar)
    if args.steps is None and not is_feasible(args.budget, spec, args.p_ar):
        raise ValueError(f"Infeasible run: {planned_steps} total steps")
    total_steps = args.steps if args.steps is not None else planned_steps
    ar_steps, bd_steps = split_phase_steps(total_steps, args.p_ar)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_seed(SEED)
    dataset = ShakespeareData.load(device)
    model = BlockDiffusionTransformer(spec, block_len=args.block_len).to(device)
    if model.counted_parameter_count() != spec.n_params:
        raise RuntimeError(
            f"Parameter mismatch: model={model.counted_parameter_count()}, config={spec.n_params}"
        )

    started = time.monotonic()
    model.train()
    trace = []
    optimizer = optimizer_for(model, args.lr, weight_decay=args.ar_weight_decay)
    ar_last_loss, ar_duration = run_phase(
        phase="ar",
        phase_steps=ar_steps,
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        device=device,
        peak_lr=args.lr,
        warmup_fraction=WARMUP_FRACTION,
        global_offset=0,
        trace=trace,
        block_len=args.block_len,
    )

    # Deliberately discard both Adam moments and step counters at the
    # objective/attention-mask transition. The BD WSD schedule is phase-local.
    del optimizer
    optimizer = optimizer_for(model, args.lr, weight_decay=args.bd_weight_decay)
    print(
        f"transition global_step={ar_steps} optimizer_reset=true schedule_reset=true",
        flush=True,
    )
    bd_last_loss, bd_duration = run_phase(
        phase="bd",
        phase_steps=bd_steps,
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        device=device,
        peak_lr=args.lr,
        warmup_fraction=args.bd_warmup_fraction,
        global_offset=ar_steps,
        trace=trace,
        block_len=args.block_len,
    )

    val_nelbo, val_masked_ce = evaluate(
        model,
        dataset,
        device,
        args.block_len,
    )
    duration = time.monotonic() - started
    exact_flops = realized_flops(total_steps, spec, args.p_ar)
    realized_p_ar = ar_steps / total_steps
    result = {
        "status": "complete",
        "budget": args.budget,
        "size": spec.label,
        "n_params": spec.n_params,
        "n_layer": spec.n_layer,
        "d_model": spec.d_model,
        "n_head": spec.n_head,
        "head_dim": spec.head_dim,
        "d_ff": spec.d_ff,
        "block_len": args.block_len,
        "sequence_length": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "steps": total_steps,
        "ar_steps": ar_steps,
        "bd_steps": bd_steps,
        "p_ar": args.p_ar,
        "realized_p_ar": realized_p_ar,
        "learning_rate": args.lr,
        "learning_rate_source": args.lr_source,
        "weight_decay": (
            args.ar_weight_decay
            if args.ar_weight_decay == args.bd_weight_decay
            else None
        ),
        "ar_weight_decay": args.ar_weight_decay,
        "bd_weight_decay": args.bd_weight_decay,
        "phase_local_wsd": True,
        "ar_warmup_fraction": WARMUP_FRACTION,
        "bd_warmup_fraction": args.bd_warmup_fraction,
        "warmup_fraction_per_phase": (
            WARMUP_FRACTION
            if args.bd_warmup_fraction == WARMUP_FRACTION
            else None
        ),
        "ar_stable_fraction": 1.0 - WARMUP_FRACTION - DECAY_FRACTION,
        "bd_stable_fraction": 1.0 - args.bd_warmup_fraction - DECAY_FRACTION,
        "stable_fraction_per_phase": (
            1.0 - WARMUP_FRACTION - DECAY_FRACTION
            if args.bd_warmup_fraction == WARMUP_FRACTION
            else None
        ),
        "decay_fraction_per_phase": DECAY_FRACTION,
        "optimizer_reset_at_transition": True,
        "schedule_reset_at_transition": True,
        "ar_train_loss": ar_last_loss,
        "train_loss": bd_last_loss,
        "ar_duration_seconds": ar_duration,
        "bd_duration_seconds": bd_duration,
        "ar_seconds_per_step": ar_duration / ar_steps,
        "bd_seconds_per_step": bd_duration / bd_steps,
        "wall_time_ratio_ar_to_bd": (
            (ar_duration / ar_steps) / (bd_duration / bd_steps)
        ),
        "theoretical_flop_ratio_ar_to_bd": (
            spec.autoregressive_training_flops_per_clean_token
            / spec.training_flops_per_clean_token
        ),
        "val_nelbo": val_nelbo,
        "val_masked_ce_t0.5": val_masked_ce,
        "clean_tokens": total_steps * TOKENS_PER_STEP,
        "ar_clean_tokens": ar_steps * TOKENS_PER_STEP,
        "bd_clean_tokens": bd_steps * TOKENS_PER_STEP,
        "realized_flops": exact_flops,
        "compute_undershoot_fraction": (args.budget - exact_flops) / args.budget,
        "compute_accounting": COMPUTE_ACCOUNTING,
        "ar_training_flops_per_clean_token": (
            spec.autoregressive_training_flops_per_clean_token
        ),
        "bd_training_flops_per_clean_token": spec.training_flops_per_clean_token,
        "nominal_average_training_flops_per_clean_token": (
            average_training_flops_per_clean_token(spec, args.p_ar)
        ),
        "realized_average_training_flops_per_clean_token": (
            exact_flops / (total_steps * TOKENS_PER_STEP)
        ),
        "effective_compute_parameters": (
            exact_flops / (12 * total_steps * TOKENS_PER_STEP)
        ),
        "train_trace": trace,
        "seed": SEED,
        "duration_seconds": duration,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    atomic_json_dump(result, args.output)
    print(
        f"complete val_nelbo={val_nelbo:.5f} val_ce={val_masked_ce:.5f} "
        f"seconds={duration:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
