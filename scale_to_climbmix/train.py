"""Train one full-budget ClimbMix block-diffusion configuration."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from config import (
    BASE_VOCAB_SIZE,
    BLOCK_LEN,
    COMPUTE_ACCOUNTING,
    DECAY_FRACTION,
    EVAL_BATCHES,
    EVAL_BATCH_SIZE,
    GRAD_CLIP,
    MASK_EPS,
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
    parser.add_argument("--batch-size", type=int, default=SCALEUP_BATCH_SIZE)
    parser.add_argument("--block-len", type=int, default=BLOCK_LEN)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the model with TorchInductor (recommended for long H100 runs)",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("auto", "cudnn", "efficient", "math"),
        default="auto",
        help="Dense SDPA backend; cuDNN supports the non-causal BD mask on H100",
    )
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-group")
    return parser.parse_args()


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def initialize_distributed(device_arg: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return DistributedContext(0, 0, 1, torch.device(device_arg))
    if not device_arg.startswith("cuda"):
        raise ValueError("Multi-process DDP requires a CUDA device")
    if not dist.is_nccl_available():
        raise RuntimeError("NCCL is required for CUDA DDP")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    return DistributedContext(rank, local_rank, world_size, device)


def attention_backend_context(name: str, device: torch.device):
    if device.type != "cuda" or name == "auto":
        return nullcontext()
    backends = {
        "cudnn": SDPBackend.CUDNN_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }
    return sdpa_kernel(backends[name])


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
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    global_tokens_per_step = args.batch_size * SEQ_LEN
    planned_steps = int(
        args.budget
        / (spec.training_flops_per_clean_token * global_tokens_per_step)
    )
    if args.steps is None and not MIN_STEPS <= planned_steps <= MAX_SCALEUP_STEPS:
        raise ValueError(f"Infeasible run: {planned_steps} steps")
    total_steps = args.steps if args.steps is not None else planned_steps
    if total_steps < 1:
        raise ValueError("steps must be positive")

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

    # Initialize every rank identically before DDP broadcasts parameters.
    set_seed(SEED)
    dataset = ClimbMixData.load(device)
    required_tokens = total_steps * global_tokens_per_step
    if required_tokens > dataset.train.total_tokens:
        raise RuntimeError(
            f"Run needs {required_tokens:,} unique tokens, dataset has "
            f"{dataset.train.total_tokens:,}"
        )
    if EVAL_BATCHES * EVAL_BATCH_SIZE * SEQ_LEN > dataset.val.total_tokens:
        raise RuntimeError("Validation shard is too small")

    base_model = BlockDiffusionTransformer(spec, block_len=args.block_len).to(device)
    if base_model.counted_parameter_count() != spec.n_params:
        raise RuntimeError(
            f"Parameter mismatch: model={base_model.counted_parameter_count()}, "
            f"config={spec.n_params}"
        )
    if args.compile:
        base_model.compile(mode="reduce-overhead", fullgraph=False)
    model = (
        DistributedDataParallel(
            base_model,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        if distributed.world_size > 1
        else base_model
    )
    optimizer = optimizer_for(model, args.lr)

    wandb_run = None
    if args.wandb_project and distributed.is_primary:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            group=args.wandb_group,
            job_type="pure_bd_train",
            config={
                "budget": args.budget,
                "size": spec.label,
                "n_params": spec.n_params,
                "n_layer": spec.n_layer,
                "d_model": spec.d_model,
                "n_head": spec.n_head,
                "d_ff": spec.d_ff,
                "block_len": args.block_len,
                "sequence_length": SEQ_LEN,
                "global_batch_size": args.batch_size,
                "local_batch_size": local_batch_size,
                "world_size": distributed.world_size,
                "steps": total_steps,
                "learning_rate": args.lr,
                "weight_decay": WEIGHT_DECAY,
                "seed": SEED,
                "compiled": args.compile,
                "attention_backend": args.attention_backend,
                "compute_accounting": COMPUTE_ACCOUNTING,
            },
        )

    # Masks should be independent across ranks even though initialization is shared.
    set_seed(SEED + distributed.rank)
    if distributed.world_size > 1:
        dist.barrier()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.monotonic()
    model.train()
    last_loss = math.nan
    log_interval = max(1, total_steps // 20)
    trace = []
    with attention_backend_context(args.attention_backend, device):
        for step in range(total_steps):
            lr = wsd_learning_rate(step, total_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            x0 = dataset.train_batch(
                step,
                local_batch_size,
                rank=distributed.rank,
                world_size=distributed.world_size,
            )
            probabilities = sample_mask_probabilities(
                local_batch_size,
                device,
                args.block_len,
            )
            xt, masked, token_probability = corrupt(x0, probabilities)
            with autocast_context(device):
                logits = model(xt, x0)
                loss = diffusion_nelbo(logits, x0, masked, token_probability)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
                foreach=True,
            )
            optimizer.step()

            should_log = (
                step == 0
                or (step + 1) % log_interval == 0
                or step + 1 == total_steps
            )
            if should_log:
                logged_loss = loss.detach()
                if distributed.world_size > 1:
                    dist.all_reduce(logged_loss, op=dist.ReduceOp.SUM)
                    logged_loss = logged_loss / distributed.world_size
                last_loss = logged_loss.item()
                if not math.isfinite(last_loss):
                    raise FloatingPointError(
                        f"Non-finite loss at step {step}: {last_loss}"
                    )
                if distributed.is_primary:
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
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/nelbo": last_loss,
                                "train/learning_rate": lr,
                                "train/grad_norm": float(grad_norm),
                                "train/clean_tokens": (step + 1)
                                * global_tokens_per_step,
                            },
                            step=step + 1,
                        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if distributed.world_size > 1:
            dist.barrier()
        training_duration = time.monotonic() - started
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
        evaluation_duration = time.monotonic() - evaluation_started

    duration = time.monotonic() - started
    if distributed.is_primary:
        wandb_url = wandb_run.url if wandb_run is not None else None
        wandb_run_id = wandb_run.id if wandb_run is not None else None
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
            "batch_size": args.batch_size,
            "local_batch_size": local_batch_size,
            "world_size": distributed.world_size,
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
            "realized_flops": (
                required_tokens * spec.training_flops_per_clean_token
            ),
            "compute_accounting": COMPUTE_ACCOUNTING,
            "training_flops_per_clean_token": spec.training_flops_per_clean_token,
            "effective_compute_parameters": spec.effective_compute_parameters,
            "train_dataset_tokens": dataset.train.total_tokens,
            "effective_train_epochs": required_tokens / dataset.train.total_tokens,
            "seed": SEED,
            "compiled": args.compile,
            "attention_backend": args.attention_backend,
            "training_duration_seconds": training_duration,
            "evaluation_duration_seconds": evaluation_duration,
            "duration_seconds": duration,
            "training_tokens_per_second": required_tokens / training_duration,
            "tokens_per_second": required_tokens / duration,
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
                    "train/training_duration_seconds": training_duration,
                    "validation/duration_seconds": evaluation_duration,
                    "train/duration_seconds": duration,
                    "train/training_tokens_per_second": (
                        required_tokens / training_duration
                    ),
                    "train/tokens_per_second": required_tokens / duration,
                    "train/realized_flops": (
                        required_tokens
                        * spec.training_flops_per_clean_token
                    ),
                    "result_path": str(args.output),
                }
            )
            wandb_run.finish()
        atomic_json_dump(result, args.output)
        print(
            f"complete val_nelbo={val_nelbo:.5f} val_ce={val_masked_ce:.5f} "
            f"seconds={duration:.1f} world_size={distributed.world_size}",
            flush=True,
        )
    if distributed.world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
