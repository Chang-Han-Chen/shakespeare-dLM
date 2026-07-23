"""Train one shared AR trunk, save transition checkpoints, and finish pure AR."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from config import (
    BASE_VOCAB_SIZE,
    BATCH_SIZE,
    COMPUTE_BUDGETS,
    EVAL_BATCHES,
    EVAL_BATCH_SIZE,
    MODEL_BY_LABEL,
    RESULTS_DIR,
    SEED,
    TOKENS_PER_STEP,
    budget_slug,
    is_feasible as baseline_is_feasible,
)
from curriculum_config import (
    P_AR_VALUES,
    ar_decay_start,
    ar_decay_steps,
    phase_steps_for,
    pure_ar_steps_for,
    shared_ar_warmup_steps,
)
from data import ClimbMixData
from model import BlockDiffusionTransformer
from train import (
    atomic_json_dump,
    autocast_context,
    optimizer_for,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--size", choices=tuple(MODEL_BY_LABEL), required=True)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Fixed-total-step mode; the full AR endpoint also uses this many steps",
    )
    parser.add_argument(
        "--skip-pure-ar-endpoint",
        action="store_true",
        help="Stop after the last shared branch checkpoint",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="AR peak-LR override; defaults to the selected pure-BD LR",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def selected_lr(budget, spec):
    path = (
        RESULTS_DIR
        / "runs"
        / budget_slug(budget)
        / spec.label
        / "lr_bracket.json"
    )
    bracket = json.loads(path.read_text())
    if bracket["status"] != "locally_bracketed":
        raise ValueError(f"Invalid LR bracket: {path}")
    return float(bracket["selected_lr"])


def ar_loss(logits, targets):
    return F.cross_entropy(
        logits[:, :, :BASE_VOCAB_SIZE].float().reshape(-1, BASE_VOCAB_SIZE),
        targets.reshape(-1),
    )


def set_lr(optimizer, learning_rate):
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def save_checkpoint(path, model, optimizer, step, learning_rate):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "completed_ar_steps": step,
            "learning_rate": learning_rate,
        },
        temporary,
    )
    os.replace(temporary, path)


@torch.inference_mode()
def evaluate_ar(model, dataset, device):
    model.eval()
    values = []
    for batch_index in range(EVAL_BATCHES):
        inputs, targets = dataset.autoregressive_val_batch(
            batch_index,
            EVAL_BATCH_SIZE,
        )
        with autocast_context(device):
            logits = model.forward_ar(inputs)
        values.append(ar_loss(logits, targets).item())
    return float(np.mean(values))


def main() -> None:
    args = parse_args()
    spec = MODEL_BY_LABEL[args.size]
    fixed_total_steps = args.total_steps is not None
    if not fixed_total_steps and args.budget not in COMPUTE_BUDGETS:
        raise ValueError("Standard curriculum budget is not in COMPUTE_BUDGETS")
    if not fixed_total_steps and not baseline_is_feasible(args.budget, spec):
        raise ValueError("AR trunks are only built for feasible pure-BD points")
    if fixed_total_steps and args.total_steps < 1:
        raise ValueError("total-steps must be positive")
    if fixed_total_steps and args.lr is None:
        raise ValueError("Fixed-total-step mode requires an explicit LR")
    peak_lr = args.lr if args.lr is not None else selected_lr(args.budget, spec)
    device = torch.device(args.device)
    set_seed(SEED)
    dataset = ClimbMixData.load(device)
    pure_steps = (
        args.total_steps
        if fixed_total_steps
        else pure_ar_steps_for(args.budget, spec)
    )
    if (pure_steps * TOKENS_PER_STEP + 1) > dataset.train.total_tokens:
        raise RuntimeError("Pure AR run would exceed one pass over training data")

    if fixed_total_steps:
        p_ar_steps = {
            f"{p_ar:.1f}": round(p_ar * args.total_steps)
            for p_ar in P_AR_VALUES
        }
    else:
        p_ar_steps = {
            f"{p_ar:.1f}": phase_steps_for(args.budget, spec, p_ar)[0]
            for p_ar in P_AR_VALUES
        }
    branch_starts = {
        key: ar_decay_start(steps)
        for key, steps in p_ar_steps.items()
    }
    pure_decay_start = ar_decay_start(pure_steps)
    checkpoint_steps = set(branch_starts.values())
    if not args.skip_pure_ar_endpoint:
        checkpoint_steps.add(pure_decay_start)
    trunk_stop = (
        max(branch_starts.values())
        if args.skip_pure_ar_endpoint
        else pure_decay_start
    )
    warmup_steps = (
        max(1, round(0.05 * min(p_ar_steps.values())))
        if fixed_total_steps
        else shared_ar_warmup_steps(args.budget, spec)
    )

    model = BlockDiffusionTransformer(spec).to(device)
    optimizer = optimizer_for(model, peak_lr)
    checkpoints = {}
    trace = []
    log_interval = max(1, trunk_stop // 20)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    trunk_started = time.monotonic()
    model.train()
    for step in range(trunk_stop):
        learning_rate = peak_lr * min(1.0, (step + 1) / warmup_steps)
        set_lr(optimizer, learning_rate)
        inputs, targets = dataset.autoregressive_train_batch(step, BATCH_SIZE)
        with autocast_context(device):
            loss = ar_loss(model.forward_ar(inputs), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"Non-finite AR loss at step {step}")
        completed = step + 1
        if completed in checkpoint_steps:
            path = args.output_dir / "checkpoints" / f"ar_step_{completed}.pt"
            save_checkpoint(path, model, optimizer, completed, learning_rate)
            checkpoints[str(completed)] = str(path)
        if step == 0 or completed % log_interval == 0 or completed == trunk_stop:
            trace.append(
                {
                    "step": completed,
                    "train_ar_ce": loss.item(),
                    "learning_rate": learning_rate,
                    "grad_norm": float(grad_norm),
                }
            )
            print(
                f"AR trunk {completed:>6}/{trunk_stop} "
                f"loss={loss.item():.4f} lr={learning_rate:.6g}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    trunk_duration = time.monotonic() - trunk_started

    trunk_result = {
        "status": "complete",
        "comparison_mode": (
            "fixed_total_steps" if fixed_total_steps else "fixed_compute"
        ),
        "budget": args.budget,
        "size": spec.label,
        "n_params": spec.n_params,
        "learning_rate": peak_lr,
        "shared_warmup_steps": warmup_steps,
        "trunk_steps": trunk_stop,
        "trunk_duration_seconds": trunk_duration,
        "trunk_seconds_per_step": trunk_duration / trunk_stop,
        "p_ar_steps": p_ar_steps,
        "fixed_total_steps": args.total_steps,
        "branch_starts": branch_starts,
        "checkpoints_by_step": checkpoints,
        "trace": trace,
        "pure_ar_endpoint_skipped": args.skip_pure_ar_endpoint,
    }
    atomic_json_dump(trunk_result, args.output_dir / "trunk_result.json")
    if args.skip_pure_ar_endpoint:
        print(
            f"complete shared_AR_trunk steps={trunk_stop} "
            f"seconds={trunk_duration:.1f}",
            flush=True,
        )
        return

    # Continue the same optimizer/model through the pure-AR 15% decay.
    decay_count = ar_decay_steps(pure_steps)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    decay_started = time.monotonic()
    for local, step in enumerate(range(pure_decay_start, pure_steps)):
        if decay_count == 1:
            learning_rate = 0.0
        else:
            learning_rate = peak_lr * (1.0 - local / (decay_count - 1))
        set_lr(optimizer, learning_rate)
        inputs, targets = dataset.autoregressive_train_batch(step, BATCH_SIZE)
        with autocast_context(device):
            loss = ar_loss(model.forward_ar(inputs), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    decay_duration = time.monotonic() - decay_started
    val_ar_ce = evaluate_ar(model, dataset, device)
    final_checkpoint = args.output_dir / "pure_ar_final.pt"
    save_checkpoint(final_checkpoint, model, optimizer, pure_steps, 0.0)

    pure_result = {
        "status": "complete",
        "objective": "autoregressive",
        "comparison_mode": (
            "fixed_total_steps" if fixed_total_steps else "fixed_compute"
        ),
        "budget": args.budget,
        "size": spec.label,
        "n_params": spec.n_params,
        "learning_rate": peak_lr,
        "steps": pure_steps,
        "fixed_total_steps": args.total_steps,
        "clean_tokens": pure_steps * TOKENS_PER_STEP,
        "effective_train_epochs": pure_steps * TOKENS_PER_STEP / dataset.train.total_tokens,
        "warmup_steps": warmup_steps,
        "stable_steps": pure_decay_start - warmup_steps,
        "decay_steps": decay_count,
        "val_ar_ce": val_ar_ce,
        "training_duration_seconds": trunk_duration + decay_duration,
        "seconds_per_step": (trunk_duration + decay_duration) / pure_steps,
        "trunk_duration_seconds": trunk_duration,
        "decay_duration_seconds": decay_duration,
        "final_checkpoint": str(final_checkpoint),
        "autoregressive_training_flops_per_clean_token": (
            spec.autoregressive_training_flops_per_clean_token
        ),
    }
    atomic_json_dump(pure_result, args.output_dir / "pure_ar_result.json")
    print(
        f"complete pure_AR val_ce={val_ar_ce:.5f} "
        f"seconds={trunk_duration + decay_duration:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
