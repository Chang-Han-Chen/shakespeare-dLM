"""Limited-data full-BD and matched-curriculum experiments.

The full-BD run keeps one stable-learning-rate trunk, saves a resumable
checkpoint every five epochs, and lets independent jobs decay each checkpoint
for one fifth as many epochs.  Once the best decayed endpoint fixes the total
horizon, curriculum jobs train from scratch for exactly that many steps.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from config import (
    BLOCK_LEN,
    COMPUTE_ACCOUNTING,
    DECAY_FRACTION,
    GRAD_CLIP,
    MODEL_BY_LABEL,
    SCALEUP_BATCH_SIZE,
    SEED,
    SEQ_LEN,
    WANDB_PROJECT,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
)
from data import (
    ClimbMixData,
    FixedTokenEpochData,
    corrupt,
    sample_mask_probabilities,
)
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


DEFAULT_SIZE = "50.0M"
# Nearest whole number of batch-128, length-256 batches to 25M tokens.
DEFAULT_UNIQUE_TOKENS = 25_001_984
DEFAULT_BD_LR = 9e-4
DEFAULT_AR_LR = 2.7e-3
DEFAULT_CHECKPOINT_EVERY = 5
DEFAULT_TRUNK_EPOCHS = 50
DEFAULT_WARMUP_EPOCHS = 1.0
H100_BF16_FLOPS = 989e12


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", choices=tuple(MODEL_BY_LABEL), default=DEFAULT_SIZE)
    parser.add_argument("--batch-size", type=int, default=SCALEUP_BATCH_SIZE)
    parser.add_argument("--unique-tokens", type=int, default=DEFAULT_UNIQUE_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--block-len", type=int, default=BLOCK_LEN)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--bd-attention-backend",
        choices=("auto", "cudnn", "efficient", "math"),
        default="cudnn",
    )
    parser.add_argument(
        "--ar-attention-backend",
        choices=("auto", "flash", "cudnn", "efficient", "math"),
        default="flash",
    )
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-group", default="data-efficiency-25m-50m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    trunk = commands.add_parser("bd-trunk")
    add_common_arguments(trunk)
    trunk.add_argument("--output-dir", type=Path, required=True)
    trunk.add_argument("--epochs", type=int, default=DEFAULT_TRUNK_EPOCHS)
    trunk.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
    )
    trunk.add_argument("--lr", type=float, default=DEFAULT_BD_LR)
    trunk.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    trunk.add_argument("--warmup-epochs", type=float, default=DEFAULT_WARMUP_EPOCHS)
    trunk.add_argument("--resume", type=Path)

    decay = commands.add_parser("bd-decay")
    add_common_arguments(decay)
    decay.add_argument("--checkpoint", type=Path, required=True)
    decay.add_argument("--output", type=Path, required=True)
    decay.add_argument(
        "--decay-epochs",
        type=int,
        help="Defaults to one fifth of the checkpoint epoch",
    )

    curriculum = commands.add_parser("curriculum")
    add_common_arguments(curriculum)
    curriculum.add_argument("--total-steps", type=int, required=True)
    curriculum.add_argument("--p-ar", type=float, default=0.4)
    curriculum.add_argument("--ar-lr", type=float, default=DEFAULT_AR_LR)
    curriculum.add_argument("--bd-lr", type=float, default=DEFAULT_BD_LR)
    curriculum.add_argument(
        "--ar-weight-decay",
        type=float,
        required=True,
    )
    curriculum.add_argument(
        "--bd-weight-decay",
        type=float,
        default=WEIGHT_DECAY,
    )
    curriculum.add_argument("--output", type=Path, required=True)
    curriculum.add_argument("--save-checkpoint", type=Path)
    return parser.parse_args()


def configure_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return device


def validate_common(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.unique_tokens < 1:
        raise ValueError("unique-tokens must be positive")
    if args.block_len < 1 or SEQ_LEN % args.block_len:
        raise ValueError("block-len must be a positive divisor of sequence length")


def build_data(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ClimbMixData, FixedTokenEpochData]:
    dataset = ClimbMixData.load(device)
    fixed = FixedTokenEpochData(
        source=dataset,
        unique_tokens=args.unique_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    return dataset, fixed


def build_model(
    args: argparse.Namespace,
    device: torch.device,
    *,
    local_rank: int = 0,
    world_size: int = 1,
) -> tuple[BlockDiffusionTransformer, torch.nn.Module]:
    spec = MODEL_BY_LABEL[args.size]
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
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        if world_size > 1
        else base_model
    )
    return base_model, model


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.load(map_location=<cuda>) also moves serialized CPU RNG tensors;
    # both RNG APIs require their state byte tensor to reside on CPU.
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device)


def gather_rng_states(
    device: torch.device,
    world_size: int,
) -> list[dict[str, Any]]:
    local_state = capture_rng_state(device)
    if world_size == 1:
        return [local_state]
    states: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(states, local_state)
    if any(state is None for state in states):
        raise RuntimeError("Failed to gather distributed RNG states")
    return [state for state in states if state is not None]


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def warmup_then_stable_lr(
    step: int,
    warmup_steps: int,
    peak_lr: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    return peak_lr


def linear_decay_lr(step: int, total_steps: int, peak_lr: float) -> float:
    if total_steps < 1:
        raise ValueError("decay needs at least one step")
    if total_steps == 1:
        return 0.0
    return peak_lr * max(0.0, 1.0 - step / (total_steps - 1))


def wandb_init(
    args: argparse.Namespace,
    *,
    job_type: str,
    config: dict[str, Any],
    run_id: str | None = None,
):
    if not args.wandb_project:
        return None
    import wandb

    kwargs: dict[str, Any] = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_name,
        "group": args.wandb_group,
        "job_type": job_type,
        "config": config,
    }
    if run_id is not None:
        kwargs.update({"id": run_id, "resume": "allow"})
    return wandb.init(**kwargs)


def train_bd_steps(
    *,
    model: torch.nn.Module,
    fixed: FixedTokenEpochData,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    block_len: int,
    attention_backend: str,
    start_step: int,
    steps: int,
    learning_rate_for_step,
    spec,
    wandb_run,
    log_prefix: str,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[float, float, list[dict[str, Any]]]:
    model.train()
    log_interval = max(1, steps // 20)
    last_loss = math.nan
    trace: list[dict[str, Any]] = []
    local_batch_size = fixed.local_batch_size(world_size)
    if world_size > 1:
        dist.barrier()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.monotonic()
    with attention_backend_context(attention_backend, device):
        for local_step in range(steps):
            global_step = start_step + local_step
            learning_rate = learning_rate_for_step(local_step, global_step)
            set_optimizer_lr(optimizer, learning_rate)
            clean = fixed.train_batch(
                global_step,
                rank=rank,
                world_size=world_size,
            )
            probability = sample_mask_probabilities(
                local_batch_size,
                device,
                block_len,
            )
            noisy, masked, token_probability = corrupt(clean, probability)
            with autocast_context(device):
                loss = diffusion_nelbo(
                    model(noisy, clean),
                    clean,
                    masked,
                    token_probability,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
                foreach=True,
            )
            optimizer.step()

            completed = local_step + 1
            if (
                local_step == 0
                or completed % log_interval == 0
                or completed == steps
            ):
                logged_loss = loss.detach()
                if world_size > 1:
                    dist.all_reduce(logged_loss, op=dist.ReduceOp.SUM)
                    logged_loss = logged_loss / world_size
                last_loss = float(logged_loss)
                if not math.isfinite(last_loss):
                    raise FloatingPointError("Non-finite BD loss")
                elapsed = time.monotonic() - started
                mfu = (
                    completed
                    * fixed.tokens_per_batch
                    * spec.training_flops_per_clean_token
                    / elapsed
                    / (H100_BF16_FLOPS * world_size)
                )
                record = {
                    "step": global_step + 1,
                    "local_step": completed,
                    "train_nelbo": last_loss,
                    "learning_rate": learning_rate,
                    "grad_norm": float(grad_norm),
                    "accounted_h100_bf16_mfu": mfu,
                }
                if rank == 0:
                    trace.append(record)
                    print(
                        f"{log_prefix} {completed:>6}/{steps} "
                        f"global={global_step + 1} nelbo={last_loss:.4f} "
                        f"lr={learning_rate:.6g} mfu={100 * mfu:.1f}%",
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/bd_nelbo": last_loss,
                                "train/learning_rate": learning_rate,
                                "train/grad_norm": float(grad_norm),
                                "train/clean_tokens": (
                                    (global_step + 1) * fixed.tokens_per_batch
                                ),
                                "train/effective_epochs": (
                                    (global_step + 1) / fixed.steps_per_epoch
                                ),
                                "performance/accounted_h100_bf16_mfu": mfu,
                            },
                            step=global_step + 1,
                        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return last_loss, time.monotonic() - started, trace


def checkpoint_payload(
    *,
    kind: str,
    args: argparse.Namespace,
    base_model: BlockDiffusionTransformer,
    optimizer: torch.optim.Optimizer,
    fixed: FixedTokenEpochData,
    next_step: int,
    peak_lr: float,
    weight_decay: float,
    warmup_epochs: float,
    wandb_run,
    rng_states: list[dict[str, Any]],
    world_size: int,
) -> dict[str, Any]:
    spec = MODEL_BY_LABEL[args.size]
    return {
        "format_version": 1,
        "kind": kind,
        "size": args.size,
        "n_params": spec.n_params,
        "block_len": args.block_len,
        "batch_size": args.batch_size,
        "unique_tokens": args.unique_tokens,
        "seed": args.seed,
        "steps_per_epoch": fixed.steps_per_epoch,
        "next_step": next_step,
        "epoch": next_step / fixed.steps_per_epoch,
        "peak_lr": peak_lr,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "model_state": base_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": rng_states[0],
        "rng_states": rng_states,
        "world_size": world_size,
        "wandb_run_id": wandb_run.id if wandb_run is not None else None,
    }


def assert_checkpoint_matches(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    fixed: FixedTokenEpochData,
) -> None:
    expected = {
        "size": args.size,
        "block_len": args.block_len,
        "batch_size": args.batch_size,
        "unique_tokens": args.unique_tokens,
        "seed": args.seed,
        "steps_per_epoch": fixed.steps_per_epoch,
    }
    for key, value in expected.items():
        if checkpoint[key] != value:
            raise ValueError(
                f"Checkpoint {key}={checkpoint[key]!r}, requested {value!r}"
            )


def run_bd_trunk(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.checkpoint_every < 1 or args.epochs % args.checkpoint_every:
        raise ValueError("epochs must be divisible by checkpoint-every")
    if args.warmup_epochs < 0.0:
        raise ValueError("warmup-epochs must be nonnegative")
    distributed = initialize_distributed(args.device)
    device = configure_device(str(distributed.device))
    _dataset, fixed = build_data(args, device)
    local_batch_size = fixed.local_batch_size(distributed.world_size)
    set_seed(args.seed)
    base_model, model = build_model(
        args,
        device,
        local_rank=distributed.local_rank,
        world_size=distributed.world_size,
    )
    optimizer = optimizer_for(model, args.lr, args.weight_decay)
    start_step = 0
    prior_training_duration = 0.0
    resume_run_id = None
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        assert_checkpoint_matches(checkpoint, args, fixed)
        base_model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        rng_states = checkpoint.get(
            "rng_states",
            [checkpoint["rng_state"]],
        )
        if len(rng_states) != distributed.world_size:
            raise ValueError(
                "Resume checkpoint world size does not match this launch"
            )
        restore_rng_state(rng_states[distributed.rank], device)
        start_step = int(checkpoint["next_step"])
        prior_training_duration = float(
            checkpoint.get("training_duration_seconds", 0.0)
        )
        resume_run_id = checkpoint.get("wandb_run_id")
        if not math.isclose(float(checkpoint["peak_lr"]), args.lr):
            raise ValueError("Cannot change peak LR when resuming a trunk")
        if not math.isclose(
            float(checkpoint["weight_decay"]),
            args.weight_decay,
        ):
            raise ValueError("Cannot change weight decay when resuming a trunk")

    total_steps = args.epochs * fixed.steps_per_epoch
    if start_step >= total_steps:
        raise ValueError("resume checkpoint is already at or beyond target epochs")
    if start_step % fixed.steps_per_epoch:
        raise ValueError("trunk resume must be at an epoch boundary")
    spec = MODEL_BY_LABEL[args.size]
    if distributed.is_primary:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "study": "limited_data_efficiency",
        "schedule": "shared_bd_stable_trunk",
        "seed": args.seed,
        "size": args.size,
        "n_params": spec.n_params,
        "unique_tokens": args.unique_tokens,
        "fixed_data_selection": "training_split_prefix_from_offset_zero",
        "batch_size": args.batch_size,
        "local_batch_size": local_batch_size,
        "world_size": distributed.world_size,
        "steps_per_epoch": fixed.steps_per_epoch,
        "total_steps": total_steps,
        "target_epochs": args.epochs,
        "start_epoch": start_step / fixed.steps_per_epoch,
        "checkpoint_every_epochs": args.checkpoint_every,
        "peak_learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "validation_during_trunk": False,
        "compiled": args.compile,
        "attention_backend": args.bd_attention_backend,
        "compute_accounting": COMPUTE_ACCOUNTING,
    }
    run = (
        wandb_init(
            args,
            job_type="data_efficiency_bd_trunk",
            config=config,
            run_id=resume_run_id,
        )
        if distributed.is_primary
        else None
    )
    # Masks are independent across DDP ranks; initialization was shared.
    if args.resume is None:
        set_seed(args.seed + distributed.rank)
    warmup_steps = round(args.warmup_epochs * fixed.steps_per_epoch)
    trace: list[dict[str, Any]] = []
    durations: list[float] = []
    last_loss = math.nan
    next_checkpoint_step = (
        (start_step // (args.checkpoint_every * fixed.steps_per_epoch)) + 1
    ) * args.checkpoint_every * fixed.steps_per_epoch
    while start_step < total_steps:
        end_step = min(next_checkpoint_step, total_steps)
        chunk_steps = end_step - start_step
        last_loss, duration, chunk_trace = train_bd_steps(
            model=model,
            fixed=fixed,
            optimizer=optimizer,
            device=device,
            block_len=args.block_len,
            attention_backend=args.bd_attention_backend,
            start_step=start_step,
            steps=chunk_steps,
            learning_rate_for_step=lambda _local, global_step: (
                warmup_then_stable_lr(global_step, warmup_steps, args.lr)
            ),
            spec=spec,
            wandb_run=run,
            log_prefix="BD-trunk",
            rank=distributed.rank,
            world_size=distributed.world_size,
        )
        durations.append(duration)
        trace.extend(chunk_trace)
        start_step = end_step
        epoch = start_step // fixed.steps_per_epoch
        checkpoint_path = (
            args.output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
        )
        rng_states = gather_rng_states(
            device,
            distributed.world_size,
        )
        payload = checkpoint_payload(
            kind="full_bd_stable_trunk",
            args=args,
            base_model=base_model,
            optimizer=optimizer,
            fixed=fixed,
            next_step=start_step,
            peak_lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            wandb_run=run,
            rng_states=rng_states,
            world_size=distributed.world_size,
        )
        payload["training_duration_seconds"] = (
            prior_training_duration + sum(durations)
        )
        if distributed.is_primary:
            atomic_torch_save(payload, checkpoint_path)
            print(f"saved {checkpoint_path}", flush=True)
        if distributed.world_size > 1:
            dist.barrier()
        next_checkpoint_step += (
            args.checkpoint_every * fixed.steps_per_epoch
        )

    training_duration = prior_training_duration + sum(durations)
    result = {
        "status": "complete",
        **config,
        "last_train_nelbo": last_loss,
        "total_steps": total_steps,
        "clean_token_exposures": total_steps * fixed.tokens_per_batch,
        "effective_epochs": total_steps / fixed.steps_per_epoch,
        "training_duration_seconds": training_duration,
        "accounted_h100_bf16_mfu": (
            total_steps
            * fixed.tokens_per_batch
            * spec.training_flops_per_clean_token
            / training_duration
            / (H100_BF16_FLOPS * distributed.world_size)
        ),
        "train_trace": trace,
        "wandb_run_id": run.id if run is not None else None,
        "wandb_url": run.url if run is not None else None,
    }
    if distributed.is_primary:
        output = args.output_dir / "trunk_result.json"
        atomic_json_dump(result, output)
        if run is not None:
            run.summary.update(
                {
                    "train/last_nelbo": last_loss,
                    "train/duration_seconds": training_duration,
                    "performance/accounted_h100_bf16_mfu": (
                        result["accounted_h100_bf16_mfu"]
                    ),
                    "result_path": str(output),
                }
            )
            run.finish()
        print(
            f"complete trunk epochs={args.epochs} mfu="
            f"{100 * result['accounted_h100_bf16_mfu']:.1f}%",
            flush=True,
        )
    if distributed.world_size > 1:
        dist.destroy_process_group()


def run_bd_decay(args: argparse.Namespace) -> None:
    distributed = initialize_distributed(args.device)
    device = configure_device(str(distributed.device))
    dataset, fixed = build_data(args, device)
    local_batch_size = fixed.local_batch_size(distributed.world_size)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    assert_checkpoint_matches(checkpoint, args, fixed)
    if checkpoint["kind"] != "full_bd_stable_trunk":
        raise ValueError("bd-decay requires a full-BD trunk checkpoint")
    start_step = int(checkpoint["next_step"])
    checkpoint_epoch = start_step // fixed.steps_per_epoch
    if checkpoint_epoch * fixed.steps_per_epoch != start_step:
        raise ValueError("checkpoint must lie on an epoch boundary")
    decay_epochs = (
        args.decay_epochs
        if args.decay_epochs is not None
        else checkpoint_epoch // 5
    )
    if checkpoint_epoch % 5 and args.decay_epochs is None:
        raise ValueError("automatic decay requires a multiple-of-five checkpoint")
    if decay_epochs < 1:
        raise ValueError("decay-epochs must be positive")
    decay_steps = decay_epochs * fixed.steps_per_epoch
    set_seed(args.seed)
    base_model, model = build_model(
        args,
        device,
        local_rank=distributed.local_rank,
        world_size=distributed.world_size,
    )
    base_model.load_state_dict(checkpoint["model_state"])
    optimizer = optimizer_for(
        model,
        float(checkpoint["peak_lr"]),
        float(checkpoint["weight_decay"]),
    )
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    checkpoint_rng_states = checkpoint.get(
        "rng_states",
        [checkpoint["rng_state"]],
    )
    if distributed.world_size > 1:
        if len(checkpoint_rng_states) != distributed.world_size:
            raise ValueError(
                "Decay world size must match the trunk checkpoint world size"
            )
        restore_rng_state(
            checkpoint_rng_states[distributed.rank],
            device,
        )
    else:
        restore_rng_state(checkpoint["rng_state"], device)
    spec = MODEL_BY_LABEL[args.size]
    total_steps = start_step + decay_steps
    config = {
        "study": "limited_data_efficiency",
        "schedule": "full_bd_checkpoint_proportional_decay",
        "seed": args.seed,
        "size": args.size,
        "n_params": spec.n_params,
        "unique_tokens": args.unique_tokens,
        "fixed_data_selection": "training_split_prefix_from_offset_zero",
        "batch_size": args.batch_size,
        "local_batch_size": local_batch_size,
        "world_size": distributed.world_size,
        "steps_per_epoch": fixed.steps_per_epoch,
        "checkpoint_epoch": checkpoint_epoch,
        "decay_epochs": decay_epochs,
        "decay_to_checkpoint_ratio": decay_epochs / checkpoint_epoch,
        "total_horizon_epochs": total_steps / fixed.steps_per_epoch,
        "total_steps": total_steps,
        "peak_learning_rate": float(checkpoint["peak_lr"]),
        "weight_decay": float(checkpoint["weight_decay"]),
        "source_trunk_world_size": int(checkpoint.get("world_size", 1)),
        "validation_only_after_decay": True,
        "compiled": args.compile,
        "attention_backend": args.bd_attention_backend,
        "compute_accounting": COMPUTE_ACCOUNTING,
    }
    run = (
        wandb_init(
            args,
            job_type="data_efficiency_bd_decay",
            config=config,
        )
        if distributed.is_primary
        else None
    )
    last_loss, training_duration, trace = train_bd_steps(
        model=model,
        fixed=fixed,
        optimizer=optimizer,
        device=device,
        block_len=args.block_len,
        attention_backend=args.bd_attention_backend,
        start_step=start_step,
        steps=decay_steps,
        learning_rate_for_step=lambda local, _global: linear_decay_lr(
            local,
            decay_steps,
            float(checkpoint["peak_lr"]),
        ),
        spec=spec,
        wandb_run=run,
        log_prefix=f"BD-decay@{checkpoint_epoch}",
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    rng_states = gather_rng_states(
        device,
        distributed.world_size,
    )
    endpoint = args.output.with_name(args.output.stem + "_checkpoint.pt")
    if distributed.is_primary:
        evaluation_started = time.monotonic()
        val_nelbo, val_masked_ce = evaluate(
            base_model,
            dataset,
            device,
            args.block_len,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        evaluation_duration = time.monotonic() - evaluation_started
        atomic_torch_save(
            {
                **checkpoint_payload(
                    kind="full_bd_decayed_endpoint",
                    args=args,
                    base_model=base_model,
                    optimizer=optimizer,
                    fixed=fixed,
                    next_step=total_steps,
                    peak_lr=float(checkpoint["peak_lr"]),
                    weight_decay=float(checkpoint["weight_decay"]),
                    warmup_epochs=float(checkpoint["warmup_epochs"]),
                    wandb_run=run,
                    rng_states=rng_states,
                    world_size=distributed.world_size,
                ),
                "source_checkpoint": str(args.checkpoint),
                "checkpoint_epoch": checkpoint_epoch,
                "decay_epochs": decay_epochs,
            },
            endpoint,
        )
        result = {
            "status": "complete",
            **config,
            "last_train_nelbo": last_loss,
            "val_nelbo": val_nelbo,
            "val_masked_ce_t0.5": val_masked_ce,
            "clean_token_exposures": total_steps * fixed.tokens_per_batch,
            "training_duration_seconds": training_duration,
            "evaluation_duration_seconds": evaluation_duration,
            "accounted_h100_bf16_mfu": (
                decay_steps
                * fixed.tokens_per_batch
                * spec.training_flops_per_clean_token
                / training_duration
                / (H100_BF16_FLOPS * distributed.world_size)
            ),
            "source_checkpoint": str(args.checkpoint),
            "endpoint_checkpoint": str(endpoint),
            "train_trace": trace,
            "wandb_run_id": run.id if run is not None else None,
            "wandb_url": run.url if run is not None else None,
        }
        atomic_json_dump(result, args.output)
        if run is not None:
            run.summary.update(
                {
                    "validation/nelbo": val_nelbo,
                    "validation/masked_ce_t0.5": val_masked_ce,
                    "train/duration_seconds": training_duration,
                    "validation/duration_seconds": evaluation_duration,
                    "performance/accounted_h100_bf16_mfu": (
                        result["accounted_h100_bf16_mfu"]
                    ),
                    "result_path": str(args.output),
                    "endpoint_checkpoint": str(endpoint),
                }
            )
            run.finish()
        print(
            f"complete BD endpoint checkpoint={checkpoint_epoch} "
            f"decay={decay_epochs} "
            f"total={config['total_horizon_epochs']:.1f} "
            f"val_nelbo={val_nelbo:.5f}",
            flush=True,
        )
    if distributed.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def train_ar_phase(
    *,
    model: torch.nn.Module,
    fixed: FixedTokenEpochData,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    attention_backend: str,
    steps: int,
    peak_lr: float,
    spec,
    wandb_run,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[float, float, list[dict[str, Any]]]:
    model.train()
    log_interval = max(1, steps // 20)
    last_loss = math.nan
    trace: list[dict[str, Any]] = []
    if world_size > 1:
        dist.barrier()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.monotonic()
    with attention_backend_context(attention_backend, device):
        for step in range(steps):
            learning_rate = wsd_learning_rate(step, steps, peak_lr)
            set_optimizer_lr(optimizer, learning_rate)
            inputs, targets = fixed.autoregressive_train_batch(
                step,
                rank=rank,
                world_size=world_size,
            )
            with autocast_context(device):
                loss = ar_cross_entropy(model(inputs), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
                foreach=True,
            )
            optimizer.step()
            completed = step + 1
            if step == 0 or completed % log_interval == 0 or completed == steps:
                logged_loss = loss.detach()
                if world_size > 1:
                    dist.all_reduce(logged_loss, op=dist.ReduceOp.SUM)
                    logged_loss = logged_loss / world_size
                last_loss = float(logged_loss)
                if not math.isfinite(last_loss):
                    raise FloatingPointError("Non-finite AR loss")
                elapsed = time.monotonic() - started
                mfu = (
                    completed
                    * fixed.tokens_per_batch
                    * spec.flash_causal_training_flops_per_clean_token
                    / elapsed
                    / (H100_BF16_FLOPS * world_size)
                )
                record = {
                    "phase": "ar",
                    "phase_step": completed,
                    "global_step": completed,
                    "train_ce": last_loss,
                    "learning_rate": learning_rate,
                    "grad_norm": float(grad_norm),
                    "accounted_h100_bf16_mfu": mfu,
                }
                if rank == 0:
                    trace.append(record)
                    print(
                        f"AR {completed:>6}/{steps} ce={last_loss:.4f} "
                        f"lr={learning_rate:.6g} mfu={100 * mfu:.1f}%",
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/ar_ce": last_loss,
                                "train/learning_rate": learning_rate,
                                "train/grad_norm": float(grad_norm),
                                "train/clean_tokens": (
                                    completed * fixed.tokens_per_batch
                                ),
                                "performance/ar_accounted_h100_bf16_mfu": mfu,
                            },
                            step=completed,
                        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return last_loss, time.monotonic() - started, trace


def run_curriculum(args: argparse.Namespace) -> None:
    if args.total_steps < 2:
        raise ValueError("curriculum needs at least two total steps")
    if not 0.0 < args.p_ar < 1.0:
        raise ValueError("p-ar must be strictly between zero and one")
    if args.ar_weight_decay < 0.0 or args.bd_weight_decay < 0.0:
        raise ValueError("weight decay must be nonnegative")
    distributed = initialize_distributed(args.device)
    device = configure_device(str(distributed.device))
    dataset, fixed = build_data(args, device)
    local_batch_size = fixed.local_batch_size(distributed.world_size)
    ar_steps = round(args.p_ar * args.total_steps)
    bd_steps = args.total_steps - ar_steps
    if ar_steps < 1 or bd_steps < 1:
        raise ValueError("both curriculum phases must be nonempty")
    spec = MODEL_BY_LABEL[args.size]
    set_seed(args.seed)
    base_model, model = build_model(
        args,
        device,
        local_rank=distributed.local_rank,
        world_size=distributed.world_size,
    )
    config = {
        "study": "limited_data_efficiency",
        "schedule": "fixed_horizon_ar_to_bd",
        "comparison_mode": "exact_full_bd_selected_total_steps",
        "seed": args.seed,
        "size": args.size,
        "n_params": spec.n_params,
        "unique_tokens": args.unique_tokens,
        "fixed_data_selection": "training_split_prefix_from_offset_zero",
        "batch_size": args.batch_size,
        "local_batch_size": local_batch_size,
        "world_size": distributed.world_size,
        "steps_per_epoch": fixed.steps_per_epoch,
        "total_steps": args.total_steps,
        "total_horizon_epochs": args.total_steps / fixed.steps_per_epoch,
        "p_ar": args.p_ar,
        "ar_steps": ar_steps,
        "bd_steps": bd_steps,
        "ar_learning_rate": args.ar_lr,
        "bd_learning_rate": args.bd_lr,
        "ar_weight_decay": args.ar_weight_decay,
        "bd_weight_decay": args.bd_weight_decay,
        "optimizer_reset_at_transition": True,
        "warmup_fraction_per_phase": WARMUP_FRACTION,
        "decay_fraction_per_phase": DECAY_FRACTION,
        "compiled": args.compile,
        "ar_attention_backend": args.ar_attention_backend,
        "bd_attention_backend": args.bd_attention_backend,
        "validation_only_at_endpoint": True,
    }
    run = (
        wandb_init(
            args,
            job_type="data_efficiency_curriculum",
            config=config,
        )
        if distributed.is_primary
        else None
    )
    if distributed.world_size > 1:
        set_seed(args.seed + distributed.rank)
    ar_optimizer = optimizer_for(
        model,
        args.ar_lr,
        args.ar_weight_decay,
    )
    last_ar_loss, ar_duration, ar_trace = train_ar_phase(
        model=model,
        fixed=fixed,
        optimizer=ar_optimizer,
        device=device,
        attention_backend=args.ar_attention_backend,
        steps=ar_steps,
        peak_lr=args.ar_lr,
        spec=spec,
        wandb_run=run,
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    del ar_optimizer
    bd_optimizer = optimizer_for(
        model,
        args.bd_lr,
        args.bd_weight_decay,
    )
    last_bd_loss, bd_duration, bd_trace = train_bd_steps(
        model=model,
        fixed=fixed,
        optimizer=bd_optimizer,
        device=device,
        block_len=args.block_len,
        attention_backend=args.bd_attention_backend,
        start_step=ar_steps,
        steps=bd_steps,
        learning_rate_for_step=lambda local, _global: wsd_learning_rate(
            local,
            bd_steps,
            args.bd_lr,
        ),
        spec=spec,
        wandb_run=run,
        log_prefix="BD-curriculum",
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    rng_states = gather_rng_states(
        device,
        distributed.world_size,
    )
    realized_flops = fixed.tokens_per_batch * (
        ar_steps * spec.flash_causal_training_flops_per_clean_token
        + bd_steps * spec.training_flops_per_clean_token
    )
    training_duration = ar_duration + bd_duration
    checkpoint_path = args.save_checkpoint
    if checkpoint_path is None:
        checkpoint_path = args.output.with_name(
            args.output.stem + "_checkpoint.pt"
        )
    if distributed.is_primary:
        evaluation_started = time.monotonic()
        val_nelbo, val_masked_ce = evaluate(
            base_model,
            dataset,
            device,
            args.block_len,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        evaluation_duration = time.monotonic() - evaluation_started
        atomic_torch_save(
            checkpoint_payload(
                kind="fixed_horizon_curriculum_endpoint",
                args=args,
                base_model=base_model,
                optimizer=bd_optimizer,
                fixed=fixed,
                next_step=args.total_steps,
                peak_lr=args.bd_lr,
                weight_decay=args.bd_weight_decay,
                warmup_epochs=0.0,
                wandb_run=run,
                rng_states=rng_states,
                world_size=distributed.world_size,
            ),
            checkpoint_path,
        )
        result = {
            "status": "complete",
            **config,
            "last_ar_train_ce": last_ar_loss,
            "last_bd_train_nelbo": last_bd_loss,
            "val_nelbo": val_nelbo,
            "val_masked_ce_t0.5": val_masked_ce,
            "clean_token_exposures": (
                args.total_steps * fixed.tokens_per_batch
            ),
            "realized_flops": realized_flops,
            "realized_to_full_bd_compute": (
                realized_flops
                / (
                    args.total_steps
                    * fixed.tokens_per_batch
                    * spec.training_flops_per_clean_token
                )
            ),
            "ar_duration_seconds": ar_duration,
            "bd_duration_seconds": bd_duration,
            "training_duration_seconds": training_duration,
            "evaluation_duration_seconds": evaluation_duration,
            "accounted_h100_bf16_mfu": (
                realized_flops
                / training_duration
                / (H100_BF16_FLOPS * distributed.world_size)
            ),
            "train_trace": ar_trace + bd_trace,
            "endpoint_checkpoint": str(checkpoint_path),
            "wandb_run_id": run.id if run is not None else None,
            "wandb_url": run.url if run is not None else None,
        }
        atomic_json_dump(result, args.output)
        if run is not None:
            run.summary.update(
                {
                    "validation/nelbo": val_nelbo,
                    "validation/masked_ce_t0.5": val_masked_ce,
                    "train/ar_duration_seconds": ar_duration,
                    "train/bd_duration_seconds": bd_duration,
                    "train/duration_seconds": training_duration,
                    "train/realized_flops": realized_flops,
                    "performance/accounted_h100_bf16_mfu": (
                        result["accounted_h100_bf16_mfu"]
                    ),
                    "result_path": str(args.output),
                    "endpoint_checkpoint": str(checkpoint_path),
                }
            )
            run.finish()
        print(
            f"complete curriculum wd={args.ar_weight_decay:g} "
            f"horizon={config['total_horizon_epochs']:.1f} "
            f"val_nelbo={val_nelbo:.5f}",
            flush=True,
        )
    if distributed.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    validate_common(args)
    if args.command == "bd-trunk":
        run_bd_trunk(args)
    elif args.command == "bd-decay":
        run_bd_decay(args)
    elif args.command == "curriculum":
        run_curriculum(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
