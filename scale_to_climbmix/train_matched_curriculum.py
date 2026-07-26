"""Train a fixed-p_AR curriculum matched to pure-BD steps and tokens."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from config import (
    BLOCK_LEN,
    COMPUTE_ACCOUNTING,
    DECAY_FRACTION,
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
from data import ClimbMixData, corrupt, sample_mask_probabilities
from model import BlockDiffusionTransformer
from train import (
    atomic_json_dump,
    attention_backend_context,
    autocast_context,
    diffusion_nelbo,
    evaluate,
    initialize_distributed,
    optimizer_for,
    set_seed,
    wsd_learning_rate,
)
from train_ar import ar_cross_entropy


DEFAULT_P_AR = 0.4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--size", choices=tuple(MODEL_BY_LABEL), required=True)
    parser.add_argument("--ar-lr", type=float, required=True)
    parser.add_argument("--bd-lr", type=float, required=True)
    parser.add_argument(
        "--p-ar",
        type=float,
        default=DEFAULT_P_AR,
        help="Fraction of matched steps assigned to the AR phase",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None, help="Test/benchmark override")
    parser.add_argument("--batch-size", type=int, default=SCALEUP_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--block-len", type=int, default=BLOCK_LEN)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ar-attention-backend",
        choices=("auto", "flash", "cudnn", "efficient", "math"),
        default="flash",
    )
    parser.add_argument(
        "--bd-attention-backend",
        choices=("auto", "cudnn", "efficient", "math"),
        default="cudnn",
    )
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-group")
    return parser.parse_args()


def set_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if not 0.0 < args.p_ar < 1.0:
        raise ValueError("p-ar must be strictly between zero and one")
    if args.block_len < 1 or SEQ_LEN % args.block_len:
        raise ValueError("block-len must be a positive divisor of sequence length")
    spec = MODEL_BY_LABEL[args.size]
    global_tokens_per_step = args.batch_size * SEQ_LEN
    planned_steps = int(
        args.budget
        / (spec.training_flops_per_clean_token * global_tokens_per_step)
    )
    if args.steps is None and not MIN_STEPS <= planned_steps <= MAX_SCALEUP_STEPS:
        raise ValueError(f"Infeasible matched-step run: {planned_steps} steps")
    total_steps = args.steps if args.steps is not None else planned_steps
    if total_steps < 2:
        raise ValueError("matched-step run needs at least two total steps")
    ar_steps = round(args.p_ar * total_steps)
    bd_steps = total_steps - ar_steps
    if ar_steps < 1 or bd_steps < 1:
        raise ValueError("both curriculum phases must be nonempty")

    distributed = initialize_distributed(args.device)
    device = distributed.device
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if args.batch_size % distributed.world_size:
        raise ValueError(
            f"Global batch {args.batch_size} is not divisible by "
            f"world size {distributed.world_size}"
        )
    local_batch_size = args.batch_size // distributed.world_size

    # Initialize all ranks identically before DDP broadcasts parameters.
    set_seed(args.seed)
    dataset = ClimbMixData.load(device)
    clean_tokens = total_steps * global_tokens_per_step
    if clean_tokens > dataset.train.total_tokens:
        raise RuntimeError(
            f"Run needs {clean_tokens:,} unique tokens, dataset has "
            f"{dataset.train.total_tokens:,}"
        )
    base_model = BlockDiffusionTransformer(
        spec,
        block_len=args.block_len,
    ).to(device)
    if base_model.counted_parameter_count() != spec.n_params:
        raise RuntimeError("Model/config parameter mismatch")
    if args.compile:
        base_model.compile(mode="reduce-overhead", fullgraph=False)
    model = (
        DistributedDataParallel(
            base_model,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        if distributed.world_size > 1
        else base_model
    )

    wandb_run = None
    if args.wandb_project and distributed.is_primary:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            group=args.wandb_group,
            job_type=f"matched_step_p_ar_{args.p_ar:g}",
            config={
                "objective": "ar_to_bd",
                "comparison_mode": "matched_pure_bd_steps_tokens",
                "p_ar": args.p_ar,
                "nominal_bd_budget": args.budget,
                "size": spec.label,
                "n_params": spec.n_params,
                "n_layer": spec.n_layer,
                "d_model": spec.d_model,
                "n_head": spec.n_head,
                "d_ff": spec.d_ff,
                "sequence_length": SEQ_LEN,
                "block_len": args.block_len,
                "batch_size": args.batch_size,
                "global_batch_size": args.batch_size,
                "local_batch_size": local_batch_size,
                "world_size": distributed.world_size,
                "total_steps": total_steps,
                "ar_steps": ar_steps,
                "bd_steps": bd_steps,
                "ar_learning_rate": args.ar_lr,
                "bd_learning_rate": args.bd_lr,
                "weight_decay": WEIGHT_DECAY,
                "warmup_fraction_per_phase": WARMUP_FRACTION,
                "decay_fraction_per_phase": DECAY_FRACTION,
                "optimizer_reset_at_transition": True,
                "seed": args.seed,
                "compiled": args.compile,
                "ar_attention_backend": args.ar_attention_backend,
                "bd_attention_backend": args.bd_attention_backend,
                "nominal_compute_accounting": COMPUTE_ACCOUNTING,
                "ar_compute_accounting": "triangular_flash_causal_v2",
                "nominal_pure_bd_flops": (
                    total_steps
                    * global_tokens_per_step
                    * spec.training_flops_per_clean_token
                ),
                "planned_realized_flops": (
                    global_tokens_per_step
                    * (
                        ar_steps
                        * spec.flash_causal_training_flops_per_clean_token
                        + bd_steps
                        * spec.training_flops_per_clean_token
                    )
                ),
            },
        )

    # Masks differ by rank while model initialization remains identical.
    set_seed(args.seed + distributed.rank)
    if distributed.world_size > 1:
        dist.barrier()
    trace = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.monotonic()
    model.train()

    ar_optimizer = optimizer_for(model, args.ar_lr)
    ar_log_interval = max(1, ar_steps // 20)
    last_ar_loss = math.nan
    with attention_backend_context(args.ar_attention_backend, device):
        ar_started = time.monotonic()
        ar_last_log_time = ar_started
        ar_last_log_step = 0
        for step in range(ar_steps):
            learning_rate = wsd_learning_rate(step, ar_steps, args.ar_lr)
            set_lr(ar_optimizer, learning_rate)
            inputs, targets = dataset.autoregressive_train_batch(
                step,
                local_batch_size,
                rank=distributed.rank,
                world_size=distributed.world_size,
            )
            with autocast_context(device):
                loss = ar_cross_entropy(model(inputs), targets)
            ar_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
                foreach=True,
            )
            ar_optimizer.step()
            completed = step + 1
            if (
                step == 0
                or completed % ar_log_interval == 0
                or completed == ar_steps
            ):
                logged_loss = loss.detach()
                if distributed.world_size > 1:
                    dist.all_reduce(logged_loss, op=dist.ReduceOp.SUM)
                    logged_loss = logged_loss / distributed.world_size
                last_ar_loss = logged_loss.item()
                if not math.isfinite(last_ar_loss):
                    raise FloatingPointError("Non-finite AR curriculum loss")
                logged_at = time.monotonic()
                ar_elapsed = logged_at - ar_started
                ar_mfu = (
                    completed
                    * global_tokens_per_step
                    * spec.flash_causal_training_flops_per_clean_token
                    / ar_elapsed
                    / (989e12 * distributed.world_size)
                )
                ar_window_mfu = (
                    (completed - ar_last_log_step)
                    * global_tokens_per_step
                    * spec.flash_causal_training_flops_per_clean_token
                    / (logged_at - ar_last_log_time)
                    / (989e12 * distributed.world_size)
                )
                ar_last_log_time = logged_at
                ar_last_log_step = completed
                if distributed.is_primary:
                    record = {
                        "phase": "ar",
                        "phase_step": completed,
                        "global_step": completed,
                        "train_loss": last_ar_loss,
                        "learning_rate": learning_rate,
                        "grad_norm": float(grad_norm),
                        "accounted_h100_bf16_mfu": ar_mfu,
                        "window_accounted_h100_bf16_mfu": ar_window_mfu,
                    }
                    trace.append(record)
                    print(
                        f"AR {completed:>6}/{ar_steps} "
                        f"ce={last_ar_loss:.4f} lr={learning_rate:.6g} "
                        f"mfu={100 * ar_mfu:.1f}% "
                        f"window_mfu={100 * ar_window_mfu:.1f}%",
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/ar_ce": last_ar_loss,
                                "train/learning_rate": learning_rate,
                                "train/grad_norm": float(grad_norm),
                                "train/phase": 0,
                                "train/clean_tokens": (
                                    completed * global_tokens_per_step
                                ),
                                "performance/ar_accounted_h100_bf16_mfu": (
                                    ar_mfu
                                ),
                                "performance/ar_window_h100_bf16_mfu": (
                                    ar_window_mfu
                                ),
                                "performance/aggregate_accounted_tflops": (
                                    ar_mfu
                                    * distributed.world_size
                                    * 989
                                ),
                            },
                            step=completed,
                        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if distributed.world_size > 1:
            dist.barrier()
        ar_duration = time.monotonic() - ar_started

    del ar_optimizer
    bd_optimizer = optimizer_for(model, args.bd_lr)
    bd_log_interval = max(1, bd_steps // 20)
    last_bd_loss = math.nan
    with attention_backend_context(args.bd_attention_backend, device):
        bd_started = time.monotonic()
        bd_last_log_time = bd_started
        bd_last_log_step = 0
        for step in range(bd_steps):
            learning_rate = wsd_learning_rate(step, bd_steps, args.bd_lr)
            set_lr(bd_optimizer, learning_rate)
            x0 = dataset.train_batch(
                ar_steps + step,
                local_batch_size,
                rank=distributed.rank,
                world_size=distributed.world_size,
            )
            probability = sample_mask_probabilities(
                local_batch_size,
                device,
                args.block_len,
            )
            xt, masked, token_probability = corrupt(x0, probability)
            with autocast_context(device):
                logits = model(xt, x0)
                loss = diffusion_nelbo(
                    logits,
                    x0,
                    masked,
                    token_probability,
                )
            bd_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
                foreach=True,
            )
            bd_optimizer.step()
            completed = step + 1
            global_step = ar_steps + completed
            if (
                step == 0
                or completed % bd_log_interval == 0
                or completed == bd_steps
            ):
                logged_loss = loss.detach()
                if distributed.world_size > 1:
                    dist.all_reduce(logged_loss, op=dist.ReduceOp.SUM)
                    logged_loss = logged_loss / distributed.world_size
                last_bd_loss = logged_loss.item()
                if not math.isfinite(last_bd_loss):
                    raise FloatingPointError("Non-finite BD curriculum loss")
                logged_at = time.monotonic()
                bd_elapsed = logged_at - bd_started
                bd_mfu = (
                    completed
                    * global_tokens_per_step
                    * spec.training_flops_per_clean_token
                    / bd_elapsed
                    / (989e12 * distributed.world_size)
                )
                bd_window_mfu = (
                    (completed - bd_last_log_step)
                    * global_tokens_per_step
                    * spec.training_flops_per_clean_token
                    / (logged_at - bd_last_log_time)
                    / (989e12 * distributed.world_size)
                )
                bd_last_log_time = logged_at
                bd_last_log_step = completed
                total_elapsed = logged_at - started
                completed_flops = global_tokens_per_step * (
                    ar_steps
                    * spec.flash_causal_training_flops_per_clean_token
                    + completed * spec.training_flops_per_clean_token
                )
                total_mfu = (
                    completed_flops
                    / total_elapsed
                    / (989e12 * distributed.world_size)
                )
                if distributed.is_primary:
                    record = {
                        "phase": "bd",
                        "phase_step": completed,
                        "global_step": global_step,
                        "train_loss": last_bd_loss,
                        "learning_rate": learning_rate,
                        "grad_norm": float(grad_norm),
                        "accounted_h100_bf16_mfu": bd_mfu,
                        "window_accounted_h100_bf16_mfu": bd_window_mfu,
                        "total_accounted_h100_bf16_mfu": total_mfu,
                    }
                    trace.append(record)
                    print(
                        f"BD {completed:>6}/{bd_steps} "
                        f"nelbo={last_bd_loss:.4f} lr={learning_rate:.6g} "
                        f"mfu={100 * bd_mfu:.1f}% "
                        f"window_mfu={100 * bd_window_mfu:.1f}% "
                        f"total_mfu={100 * total_mfu:.1f}%",
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/bd_nelbo": last_bd_loss,
                                "train/learning_rate": learning_rate,
                                "train/grad_norm": float(grad_norm),
                                "train/phase": 1,
                                "train/clean_tokens": (
                                    global_step * global_tokens_per_step
                                ),
                                "performance/bd_accounted_h100_bf16_mfu": (
                                    bd_mfu
                                ),
                                "performance/bd_window_h100_bf16_mfu": (
                                    bd_window_mfu
                                ),
                                "performance/total_accounted_h100_bf16_mfu": (
                                    total_mfu
                                ),
                                "performance/aggregate_accounted_tflops": (
                                    total_mfu
                                    * distributed.world_size
                                    * 989
                                ),
                            },
                            step=global_step,
                        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if distributed.world_size > 1:
            dist.barrier()
        bd_duration = time.monotonic() - bd_started

        evaluation_started = time.monotonic()
        if distributed.is_primary:
            val_nelbo, val_masked_ce = evaluate(
                base_model,
                dataset,
                device,
                args.block_len,
            )
        else:
            val_nelbo, val_masked_ce = math.nan, math.nan
        if distributed.world_size > 1:
            dist.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        evaluation_duration = time.monotonic() - evaluation_started

    duration = time.monotonic() - started
    nominal_bd_flops = (
        clean_tokens * spec.training_flops_per_clean_token
    )
    realized_flops = global_tokens_per_step * (
        ar_steps * spec.flash_causal_training_flops_per_clean_token
        + bd_steps * spec.training_flops_per_clean_token
    )
    if distributed.is_primary:
        wandb_url = wandb_run.url if wandb_run is not None else None
        wandb_run_id = wandb_run.id if wandb_run is not None else None
        result = {
            "status": "complete",
            "objective": "ar_to_bd",
            "comparison_mode": "matched_pure_bd_steps_tokens",
            "p_ar": args.p_ar,
            "budget": args.budget,
            "nominal_bd_budget": args.budget,
            "size": spec.label,
            "n_params": spec.n_params,
            "n_layer": spec.n_layer,
            "d_model": spec.d_model,
            "n_head": spec.n_head,
            "head_dim": spec.head_dim,
            "d_ff": spec.d_ff,
            "block_len": args.block_len,
            "sequence_length": SEQ_LEN,
            "batch_size": args.batch_size,
            "local_batch_size": local_batch_size,
            "world_size": distributed.world_size,
            "total_steps": total_steps,
            "ar_steps": ar_steps,
            "bd_steps": bd_steps,
            "ar_learning_rate": args.ar_lr,
            "bd_learning_rate": args.bd_lr,
            "weight_decay": WEIGHT_DECAY,
            "optimizer_reset_at_transition": True,
            "warmup_fraction_per_phase": WARMUP_FRACTION,
            "decay_fraction_per_phase": DECAY_FRACTION,
            "last_ar_train_ce": last_ar_loss,
            "last_bd_train_nelbo": last_bd_loss,
            "train_trace": trace,
            "val_nelbo": val_nelbo,
            "val_masked_ce_t0.5": val_masked_ce,
            "clean_tokens": clean_tokens,
            "nominal_pure_bd_flops": nominal_bd_flops,
            "realized_flops": realized_flops,
            "realized_to_nominal_compute": (
                realized_flops / nominal_bd_flops
            ),
            "accounted_h100_bf16_mfu": (
                realized_flops
                / (ar_duration + bd_duration)
                / (989e12 * distributed.world_size)
            ),
            "ar_accounted_h100_bf16_mfu": (
                ar_steps
                * global_tokens_per_step
                * spec.flash_causal_training_flops_per_clean_token
                / ar_duration
                / (989e12 * distributed.world_size)
            ),
            "bd_accounted_h100_bf16_mfu": (
                bd_steps
                * global_tokens_per_step
                * spec.training_flops_per_clean_token
                / bd_duration
                / (989e12 * distributed.world_size)
            ),
            "aggregate_accounted_tflops": (
                realized_flops / (ar_duration + bd_duration) / 1e12
            ),
            "nominal_compute_accounting": COMPUTE_ACCOUNTING,
            "ar_compute_accounting": "triangular_flash_causal_v2",
            "flash_causal_training_flops_per_clean_token": (
                spec.flash_causal_training_flops_per_clean_token
            ),
            "block_diffusion_training_flops_per_clean_token": (
                spec.training_flops_per_clean_token
            ),
            "train_dataset_tokens": dataset.train.total_tokens,
            "effective_train_epochs": (
                clean_tokens / dataset.train.total_tokens
            ),
            "seed": args.seed,
            "compiled": args.compile,
            "ar_attention_backend": args.ar_attention_backend,
            "bd_attention_backend": args.bd_attention_backend,
            "ar_duration_seconds": ar_duration,
            "bd_duration_seconds": bd_duration,
            "evaluation_duration_seconds": evaluation_duration,
            "duration_seconds": duration,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "wandb_run_id": wandb_run_id,
            "wandb_url": wandb_url,
        }
        if wandb_run is not None:
            wandb_run.summary.update(
                {
                    "validation/nelbo": val_nelbo,
                    "validation/masked_ce_t0.5": val_masked_ce,
                    "train/ar_duration_seconds": ar_duration,
                    "train/bd_duration_seconds": bd_duration,
                    "train/duration_seconds": duration,
                    "train/realized_flops": realized_flops,
                    "train/nominal_pure_bd_flops": nominal_bd_flops,
                    "train/realized_to_nominal_compute": (
                        realized_flops / nominal_bd_flops
                    ),
                    "performance/accounted_h100_bf16_mfu": (
                        result["accounted_h100_bf16_mfu"]
                    ),
                    "performance/ar_accounted_h100_bf16_mfu": (
                        result["ar_accounted_h100_bf16_mfu"]
                    ),
                    "performance/bd_accounted_h100_bf16_mfu": (
                        result["bd_accounted_h100_bf16_mfu"]
                    ),
                    "performance/aggregate_accounted_tflops": (
                        result["aggregate_accounted_tflops"]
                    ),
                    "result_path": str(args.output),
                }
            )
            wandb_run.finish()
        atomic_json_dump(result, args.output)
        print(
            f"complete p_ar={args.p_ar:g} val_nelbo={val_nelbo:.5f} "
            f"realized/nominal={realized_flops / nominal_bd_flops:.3f} "
            f"mfu={100 * result['accounted_h100_bf16_mfu']:.1f}% "
            f"seconds={duration:.1f} world_size={distributed.world_size}",
            flush=True,
        )
    if distributed.world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
