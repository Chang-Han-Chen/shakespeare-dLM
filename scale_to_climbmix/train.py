"""Train one full-budget ClimbMix block-diffusion configuration."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from config import (
    BASE_VOCAB_SIZE,
    BATCH_SIZE,
    BLOCK_LEN,
    COMPUTE_ACCOUNTING,
    DECAY_FRACTION,
    EVAL_BATCHES,
    EVAL_BATCH_SIZE,
    GRAD_CLIP,
    MASK_EPS,
    MODEL_BY_LABEL,
    SEED,
    SEQ_LEN,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
    is_feasible,
    realized_flops,
    realized_tokens,
    steps_for,
)
from data import (
    ClimbMixData,
    corrupt,
    sample_mask_probabilities,
    stratified_mask_probabilities,
)
from model import BlockDiffusionTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--size", choices=tuple(MODEL_BY_LABEL), required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None, help="Test/benchmark override")
    parser.add_argument("--block-len", type=int, default=BLOCK_LEN)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wsd_learning_rate(
    step: int,
    total_steps: int,
    peak_lr: float,
    warmup_fraction: float = WARMUP_FRACTION,
    decay_fraction: float = DECAY_FRACTION,
) -> float:
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if not 0.0 < decay_fraction < 1.0:
        raise ValueError("decay_fraction must be in (0, 1)")
    if warmup_fraction + decay_fraction >= 1.0:
        raise ValueError("warmup and decay must leave a stable phase")
    warmup_steps = round(warmup_fraction * total_steps)
    decay_steps = max(1, round(decay_fraction * total_steps))
    decay_start = total_steps - decay_steps
    if warmup_steps and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step < decay_start:
        return peak_lr
    if decay_steps == 1:
        return 0.0
    progress = (step - decay_start) / (decay_steps - 1)
    return peak_lr * max(0.0, 1.0 - progress)


def diffusion_nelbo(logits, targets, masked, mask_probability):
    token_ce = F.cross_entropy(
        logits[:, :, :BASE_VOCAB_SIZE].float().reshape(-1, BASE_VOCAB_SIZE),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    importance = masked.float() / mask_probability.clamp_min(MASK_EPS)
    return (token_ce * importance).mean()


def masked_ce(logits, targets, masked):
    token_ce = F.cross_entropy(
        logits[:, :, :BASE_VOCAB_SIZE].float().reshape(-1, BASE_VOCAB_SIZE),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    return (token_ce * masked).sum() / masked.sum().clamp_min(1)


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.inference_mode()
def evaluate(model, dataset, device, block_len: int = BLOCK_LEN):
    model.eval()
    set_seed(SEED + 10_000)
    nelbo_values, ce_values = [], []
    for batch_index in range(EVAL_BATCHES):
        x0 = dataset.val_batch(batch_index, EVAL_BATCH_SIZE)
        probability = stratified_mask_probabilities(
            EVAL_BATCH_SIZE,
            device,
            batch_index,
            block_len,
        )
        xt, masked, token_probability = corrupt(x0, probability)
        with autocast_context(device):
            logits = model(xt, x0)
        nelbo_values.append(diffusion_nelbo(logits, x0, masked, token_probability).item())

        fixed_probability = torch.full_like(probability, 0.5)
        xt_fixed, masked_fixed, _ = corrupt(x0, fixed_probability)
        with autocast_context(device):
            logits_fixed = model(xt_fixed, x0)
        ce_values.append(masked_ce(logits_fixed, x0, masked_fixed).item())
    return float(np.mean(nelbo_values)), float(np.mean(ce_values))


def optimizer_for(model, lr):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = {"lr": lr, "betas": (0.9, 0.95), "eps": 1e-8}
    if "fused" in torch.optim.AdamW.__init__.__code__.co_varnames:
        kwargs["fused"] = next(model.parameters()).is_cuda
    return torch.optim.AdamW(groups, **kwargs)


def atomic_json_dump(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.block_len < 1 or SEQ_LEN % args.block_len:
        raise ValueError("block-len must be a positive divisor of sequence length")
    spec = MODEL_BY_LABEL[args.size]
    planned_steps = steps_for(args.budget, spec)
    if args.steps is None and not is_feasible(args.budget, spec):
        raise ValueError(f"Infeasible run: {planned_steps} steps")
    total_steps = args.steps if args.steps is not None else planned_steps
    if total_steps < 1:
        raise ValueError("steps must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_seed(SEED)
    dataset = ClimbMixData.load(device)
    required_tokens = realized_tokens(total_steps)
    if required_tokens > dataset.train.total_tokens:
        raise RuntimeError(
            f"Run needs {required_tokens:,} unique tokens, dataset has "
            f"{dataset.train.total_tokens:,}"
        )
    if EVAL_BATCHES * EVAL_BATCH_SIZE * SEQ_LEN > dataset.val.total_tokens:
        raise RuntimeError("Validation shard is too small")

    model = BlockDiffusionTransformer(spec, block_len=args.block_len).to(device)
    if model.counted_parameter_count() != spec.n_params:
        raise RuntimeError(
            f"Parameter mismatch: model={model.counted_parameter_count()}, "
            f"config={spec.n_params}"
        )
    optimizer = optimizer_for(model, args.lr)

    started = time.monotonic()
    model.train()
    last_loss = math.nan
    log_interval = max(1, total_steps // 20)
    trace = []
    for step in range(total_steps):
        lr = wsd_learning_rate(step, total_steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x0 = dataset.train_batch(step, BATCH_SIZE)
        probabilities = sample_mask_probabilities(
            BATCH_SIZE,
            device,
            args.block_len,
        )
        xt, masked, token_probability = corrupt(x0, probabilities)
        with autocast_context(device):
            logits = model(xt, x0)
            loss = diffusion_nelbo(logits, x0, masked, token_probability)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        last_loss = loss.item()
        if not math.isfinite(last_loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {last_loss}")
        if step == 0 or (step + 1) % log_interval == 0 or step + 1 == total_steps:
            record = {
                "step": step + 1,
                "train_nelbo": last_loss,
                "learning_rate": lr,
                "grad_norm": float(grad_norm),
            }
            trace.append(record)
            print(
                f"step {step + 1:>6}/{total_steps} loss={last_loss:.4f} "
                f"lr={lr:.6g} grad={float(grad_norm):.3f}",
                flush=True,
            )

    val_nelbo, val_masked_ce = evaluate(
        model,
        dataset,
        device,
        args.block_len,
    )
    duration = time.monotonic() - started
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
        "learning_rate": args.lr,
        "weight_decay": WEIGHT_DECAY,
        "warmup_fraction": WARMUP_FRACTION,
        "stable_fraction": 1.0 - WARMUP_FRACTION - DECAY_FRACTION,
        "decay_fraction": DECAY_FRACTION,
        "train_loss": last_loss,
        "train_trace": trace,
        "val_nelbo": val_nelbo,
        "val_masked_ce_t0.5": val_masked_ce,
        "clean_tokens": required_tokens,
        "realized_flops": realized_flops(total_steps, spec),
        "compute_accounting": COMPUTE_ACCOUNTING,
        "training_flops_per_clean_token": spec.training_flops_per_clean_token,
        "effective_compute_parameters": spec.effective_compute_parameters,
        "train_dataset_tokens": dataset.train.total_tokens,
        "effective_train_epochs": required_tokens / dataset.train.total_tokens,
        "seed": SEED,
        "duration_seconds": duration,
        "tokens_per_second": required_tokens / duration,
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
