"""Train one full-budget pure-autoregressive ClimbMix configuration."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from config import (
    BASE_VOCAB_SIZE,
    DECAY_FRACTION,
    EVAL_BATCHES,
    EVAL_BATCH_SIZE,
    GRAD_CLIP,
    MAX_SCALEUP_STEPS,
    MIN_STEPS,
    MODEL_BY_LABEL,
    SCALEUP_BATCH_SIZE,
    SEED,
    SEQ_LEN,
    WANDB_PROJECT,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
)
from data import ClimbMixData
from model import BlockDiffusionTransformer
from train import (
    atomic_json_dump,
    attention_backend_context,
    autocast_context,
    optimizer_for,
    set_seed,
    wsd_learning_rate,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--size", choices=tuple(MODEL_BY_LABEL), required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None, help="Test/benchmark override")
    parser.add_argument("--batch-size", type=int, default=SCALEUP_BATCH_SIZE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the model with TorchInductor",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("auto", "flash", "cudnn", "efficient", "math"),
        default="flash",
        help="Causal SDPA backend; FlashAttention is recommended on H100",
    )
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-group")
    return parser.parse_args()


def ar_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :, :BASE_VOCAB_SIZE].float().reshape(-1, BASE_VOCAB_SIZE),
        targets.reshape(-1),
    )


@torch.inference_mode()
def evaluate_ar(
    model: BlockDiffusionTransformer,
    dataset: ClimbMixData,
    device: torch.device,
) -> float:
    model.eval()
    values = []
    for batch_index in range(EVAL_BATCHES):
        inputs, targets = dataset.autoregressive_val_batch(
            batch_index,
            EVAL_BATCH_SIZE,
        )
        with autocast_context(device):
            logits = model.forward_ar(inputs)
        values.append(ar_cross_entropy(logits, targets).item())
    return float(np.mean(values))


def main() -> None:
    args = parse_args()
    spec = MODEL_BY_LABEL[args.size]
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    tokens_per_step = args.batch_size * SEQ_LEN
    planned_steps = int(
        args.budget
        / (spec.flash_causal_training_flops_per_clean_token * tokens_per_step)
    )
    if args.steps is None and not MIN_STEPS <= planned_steps <= MAX_SCALEUP_STEPS:
        raise ValueError(f"Infeasible AR run: {planned_steps} steps")
    total_steps = args.steps if args.steps is not None else planned_steps
    if total_steps < 1:
        raise ValueError("steps must be positive")

    set_seed(SEED)
    dataset = ClimbMixData.load(device)
    required_tokens = total_steps * tokens_per_step
    if required_tokens > dataset.train.total_tokens:
        raise RuntimeError(
            f"Run needs {required_tokens:,} unique tokens, dataset has "
            f"{dataset.train.total_tokens:,}"
        )
    model = BlockDiffusionTransformer(spec).to(device)
    if model.counted_parameter_count() != spec.n_params:
        raise RuntimeError("Model/config parameter mismatch")
    ar_forward = model.forward_ar
    if args.compile:
        ar_forward = torch.compile(
            model.forward_ar,
            mode="reduce-overhead",
            fullgraph=False,
        )
    optimizer = optimizer_for(model, args.lr)

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            group=args.wandb_group,
            job_type="pure_ar_train",
            config={
                "objective": "autoregressive",
                "budget": args.budget,
                "size": spec.label,
                "n_params": spec.n_params,
                "n_layer": spec.n_layer,
                "d_model": spec.d_model,
                "n_head": spec.n_head,
                "d_ff": spec.d_ff,
                "sequence_length": SEQ_LEN,
                "batch_size": args.batch_size,
                "steps": total_steps,
                "learning_rate": args.lr,
                "weight_decay": WEIGHT_DECAY,
                "warmup_fraction": WARMUP_FRACTION,
                "decay_fraction": DECAY_FRACTION,
                "seed": SEED,
                "compiled": args.compile,
                "attention_backend": args.attention_backend,
                "compute_accounting": "triangular_flash_causal_v2",
            },
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.monotonic()
    trace = []
    last_loss = math.nan
    log_interval = max(1, total_steps // 20)
    model.train()
    with attention_backend_context(args.attention_backend, device):
        for step in range(total_steps):
            learning_rate = wsd_learning_rate(step, total_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            inputs, targets = dataset.autoregressive_train_batch(
                step,
                args.batch_size,
            )
            with autocast_context(device):
                loss = ar_cross_entropy(ar_forward(inputs), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
                foreach=True,
            )
            optimizer.step()

            completed = step + 1
            if (
                step == 0
                or completed % log_interval == 0
                or completed == total_steps
            ):
                last_loss = loss.item()
                if not math.isfinite(last_loss):
                    raise FloatingPointError(
                        f"Non-finite AR loss at step {completed}"
                    )
                record = {
                    "step": completed,
                    "train_ar_ce": last_loss,
                    "learning_rate": learning_rate,
                    "grad_norm": float(grad_norm),
                }
                trace.append(record)
                print(
                    f"step {completed:>6}/{total_steps} "
                    f"ar_ce={last_loss:.4f} lr={learning_rate:.6g} "
                    f"grad={float(grad_norm):.3f}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/ar_ce": last_loss,
                            "train/learning_rate": learning_rate,
                            "train/grad_norm": float(grad_norm),
                            "train/clean_tokens": completed * tokens_per_step,
                        },
                        step=completed,
                    )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_duration = time.monotonic() - started
        evaluation_started = time.monotonic()
        val_ar_ce = evaluate_ar(model, dataset, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        evaluation_duration = time.monotonic() - evaluation_started

    duration = time.monotonic() - started
    wandb_url = wandb_run.url if wandb_run is not None else None
    wandb_run_id = wandb_run.id if wandb_run is not None else None
    realized_flops = (
        required_tokens * spec.flash_causal_training_flops_per_clean_token
    )
    result = {
        "status": "complete",
        "objective": "autoregressive",
        "budget": args.budget,
        "size": spec.label,
        "n_params": spec.n_params,
        "n_layer": spec.n_layer,
        "d_model": spec.d_model,
        "n_head": spec.n_head,
        "head_dim": spec.head_dim,
        "d_ff": spec.d_ff,
        "sequence_length": SEQ_LEN,
        "batch_size": args.batch_size,
        "steps": total_steps,
        "learning_rate": args.lr,
        "weight_decay": WEIGHT_DECAY,
        "warmup_fraction": WARMUP_FRACTION,
        "stable_fraction": 1.0 - WARMUP_FRACTION - DECAY_FRACTION,
        "decay_fraction": DECAY_FRACTION,
        "train_ar_ce": last_loss,
        "train_trace": trace,
        "val_ar_ce": val_ar_ce,
        "clean_tokens": required_tokens,
        "realized_flops": realized_flops,
        "compute_accounting": "triangular_flash_causal_v2",
        "flash_causal_training_flops_per_clean_token": (
            spec.flash_causal_training_flops_per_clean_token
        ),
        "train_dataset_tokens": dataset.train.total_tokens,
        "effective_train_epochs": required_tokens / dataset.train.total_tokens,
        "tokens_per_parameter": required_tokens / spec.n_params,
        "seed": SEED,
        "compiled": args.compile,
        "attention_backend": args.attention_backend,
        "training_duration_seconds": training_duration,
        "evaluation_duration_seconds": evaluation_duration,
        "duration_seconds": duration,
        "training_tokens_per_second": required_tokens / training_duration,
        "accounted_h100_bf16_mfu": (
            realized_flops / training_duration / 989e12
        ),
        "tokens_per_second": required_tokens / duration,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "wandb_run_id": wandb_run_id,
        "wandb_url": wandb_url,
    }
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "validation/ar_ce": val_ar_ce,
                "train/training_duration_seconds": training_duration,
                "validation/duration_seconds": evaluation_duration,
                "train/duration_seconds": duration,
                "train/training_tokens_per_second": (
                    required_tokens / training_duration
                ),
                "train/realized_flops": result["realized_flops"],
                "train/accounted_h100_bf16_mfu": (
                    result["accounted_h100_bf16_mfu"]
                ),
                "result_path": str(args.output),
            }
        )
        wandb_run.finish()
    atomic_json_dump(result, args.output)
    print(
        f"complete val_ar_ce={val_ar_ce:.5f} "
        f"seconds={duration:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
