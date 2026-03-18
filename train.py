import os
import sys
import time
import math
import pickle
import argparse
import importlib
import inspect
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v}")


parser = argparse.ArgumentParser()

# Model selection
parser.add_argument(
    "--model", type=str, default="remasked",
    choices=["remasked", "mdlm", "edit_one_pass", "edit_two_pass", "ar"],
    help="Which model variant to train",
)

# Training
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--block_size", type=int, default=256)
parser.add_argument("--max_iters", type=int, default=4000)
parser.add_argument("--eval_interval", type=int, default=300)
parser.add_argument("--warmup_iters", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=3e-3)
parser.add_argument("--min_lr", type=float, default=3e-4)
parser.add_argument("--eval_iters", type=int, default=50)
parser.add_argument("--save_interval", type=int, default=200)
parser.add_argument("--grad_clip", type=float, default=1.0)

# Model
parser.add_argument("--n_embd", type=int, default=256)
parser.add_argument("--n_head", type=int, default=4)
parser.add_argument("--n_layer", type=int, default=6)
parser.add_argument("--dropout", type=float, default=0.1)

# Diffusion / corruption
parser.add_argument("--T", type=int, default=100)
parser.add_argument("--t_min", type=float, default=0.1,
                    help="Lower clipping bound for t/T (0.0 = no lower clip)")
parser.add_argument("--t_max", type=float, default=0.9,
                    help="Upper clipping bound for t/T (1.0 = no upper clip)")
parser.add_argument(
    "--noise_schedule",
    type=str,
    default="linear",
    choices=["linear", "cosine"],
)
parser.add_argument("--corrupt_prob", type=float, default=0.15,
                    help="Fraction of visible tokens corrupted (edit_one_pass)")
parser.add_argument("--lambda_corr", type=float, default=1.0,
                    help="Weight of correction loss (edit_two_pass)")

# Eval
parser.add_argument("--eval_t_frac", type=float, default=0.6,
                    help="Fixed time fraction for validation denoising CE")
parser.add_argument("--gpt2_eval_samples", type=int, default=50,
                    help="Number of samples for GPT2-large CE metric")
parser.add_argument(
    "--gpt2_eval_interval",
    type=int,
    default=600,
    help="How often to run GPT2-large evaluation (0 disables)",
)
parser.add_argument(
    "--sample_interval",
    type=int,
    default=None,
    help="How often to print a generated sample during training (defaults to eval_interval, 0 disables)",
)

# Data / misc
parser.add_argument("--train_split_ratio", type=float, default=0.9)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--checkpoint_path", type=str, default=None,
                    help="Defaults to ckpt_{model}.pt")
parser.add_argument("--loss_log_path", type=str, default=None,
                    help="Defaults to loss_{model}.pkl")
parser.add_argument("--prompt_len", type=int, default=16)
parser.add_argument("--num_final_samples", type=int, default=10)
parser.add_argument("--use_compile", type=str2bool, default=True)

args = parser.parse_args()

if args.checkpoint_path is None:
    args.checkpoint_path = f"ckpt_{args.model}.pt"
if args.loss_log_path is None:
    args.loss_log_path = f"loss_{args.model}.pkl"

batch_size = args.batch_size
block_size = args.block_size
max_iters = args.max_iters
eval_interval = args.eval_interval
warmup_iters = args.warmup_iters
learning_rate = args.learning_rate
min_lr = args.min_lr
eval_iters = args.eval_iters
save_interval = args.save_interval
grad_clip = args.grad_clip
gpt2_eval_interval = (
    eval_interval if args.gpt2_eval_interval is None else args.gpt2_eval_interval
)
sample_interval = (
    eval_interval if args.sample_interval is None else args.sample_interval
)

n_embd = args.n_embd
n_head = args.n_head
n_layer = args.n_layer
dropout = args.dropout
assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
head_dim = n_embd // n_head

T = args.T
t_min = args.t_min
t_max = args.t_max
noise_schedule = args.noise_schedule
checkpoint_path = args.checkpoint_path
loss_log_path = args.loss_log_path
eval_t_frac = args.eval_t_frac
gpt2_eval_samples = args.gpt2_eval_samples
prompt_len = args.prompt_len
num_final_samples = args.num_final_samples

# --- Dynamic model import ---
MODEL_MODULE_MAP = {
    "remasked": "model_remasked",
    "mdlm": "model_MDLM",
    "edit_one_pass": "model_edit_one_pass",
    "edit_two_pass": "model_edit_two_pass",
    "ar": "model_AR",
}
model_module = importlib.import_module(MODEL_MODULE_MAP[args.model])
Model = model_module.Model
generate_from = model_module.generate_from
model_get_batch = model_module.get_batch
model_compute_loss = model_module.compute_loss
model_compute_eval_loss = model_module.compute_eval_loss
model_get_eval_batch = getattr(model_module, "get_eval_batch", None)

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

torch.manual_seed(args.seed)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def autocast_ctx():
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def token_epochs_from_steps(num_steps, num_train_tokens):
    return (num_steps * batch_size * block_size) / max(1, num_train_tokens)


def print_run_info(model):
    n_params = count_parameters(model)
    train_tokens_per_step = batch_size * block_size

    token_epoch_per_step = token_epochs_from_steps(1, len(train_data))
    token_epoch_warmup = token_epochs_from_steps(warmup_iters, len(train_data))
    token_epoch_eval = token_epochs_from_steps(eval_interval, len(train_data))
    token_epoch_total = token_epochs_from_steps(max_iters, len(train_data))

    print("=" * 80)
    print("Training config")
    print("=" * 80)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print(f"device: {device}")
    print(f"vocab_size: {vocab_size}")
    print(f"train_tokens: {len(train_data):,}")
    print(f"val_tokens: {len(val_data):,}")
    print(f"tokens_per_step: {train_tokens_per_step:,}")
    print(f"model_parameters: {n_params:,} ({n_params / 1e6:.3f}M)")
    print("-" * 80)
    print("Token-epoch summary")
    print("-" * 80)
    print("token_epoch := processed_train_tokens / len(train_data)")
    print(f"token_epochs_per_step: {token_epoch_per_step:.4f}")
    print(f"token_epochs_per_eval_interval: {token_epoch_eval:.2f}")
    print(f"token_epochs_in_warmup: {token_epoch_warmup:.2f}")
    print(f"expected_total_token_epochs: {token_epoch_total:.2f}")
    print("=" * 80)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
chars = ["_"] + chars  # "_" is MASK
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

mask_token_id = stoi["_"]

print(f"Vocab size: {vocab_size}")

def encode(s):
    return [stoi[ch] for ch in s]

def decode(l):
    return "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(args.train_split_ratio * len(data))
train_data = data[:n]
val_data = data[n:]


# ---------------------------------------------------------------------
# Diffusion / corruption schedule
# ---------------------------------------------------------------------

def time_fraction_tensor(t_steps: torch.Tensor) -> torch.Tensor:
    """
    Map integer timesteps t in {1, ..., T} to a float in [t_min, t_max].
    """
    t_frac = t_steps.float() / T
    t_frac = t_frac.clamp(min=t_min, max=t_max)
    return t_frac


def time_fraction_scalar(t_step: int) -> float:
    t_frac = float(t_step) / T
    t_frac = min(max(t_frac, t_min), t_max)
    return t_frac


def survival_prob_tensor(t_steps: torch.Tensor) -> torch.Tensor:
    """
    Probability a token stays unmasked at timestep t.
    """
    t_frac = time_fraction_tensor(t_steps)
    if noise_schedule == "linear":
        a_t = 1.0 - t_frac
    elif noise_schedule == "cosine":
        a_t = torch.cos(0.5 * math.pi * t_frac)
    else:
        raise ValueError(f"unknown noise_schedule: {noise_schedule}")
    return a_t.clamp(0.0, 1.0)


def survival_prob_scalar(t_step: int) -> float:
    t_frac = time_fraction_scalar(t_step)
    if noise_schedule == "linear":
        a_t = 1.0 - t_frac
    elif noise_schedule == "cosine":
        a_t = math.cos(0.5 * math.pi * t_frac)
    else:
        raise ValueError(f"unknown noise_schedule: {noise_schedule}")
    return max(0.0, min(1.0, a_t))


# ---------------------------------------------------------------------
# Eval: fixed-t batch for validation
# ---------------------------------------------------------------------

def get_eval_batch(split):
    """Like get_batch but with a fixed noise level (eval_t_frac)."""
    data_split = train_data if split == "train" else val_data
    idx = torch.randint(len(data_split) - block_size, (batch_size,))
    x0 = torch.stack([data_split[i : i + block_size] for i in idx])

    # Fixed survival probability from eval_t_frac
    if noise_schedule == "linear":
        a_t = 1.0 - eval_t_frac
    elif noise_schedule == "cosine":
        a_t = math.cos(0.5 * math.pi * eval_t_frac)
    else:
        raise ValueError(f"unknown noise_schedule: {noise_schedule}")
    a_t = max(0.0, min(1.0, a_t))

    token_mask = torch.rand(batch_size, block_size) > a_t
    xt = x0.clone()
    xt[token_mask] = mask_token_id

    return xt.to(device), x0.to(device), token_mask.to(device)


def get_model_eval_batch(split):
    """
    Allow model-specific eval batches when the training objective differs
    from the diffusion denoising setup used by the shared fixed-t eval.
    """
    if model_get_eval_batch is not None:
        return model_get_eval_batch(split, cfg)
    return get_eval_batch(split)


@torch.no_grad()
def estimate_loss(run_model):
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb, mb = get_model_eval_batch(split)
            with autocast_ctx():
                loss = model_compute_eval_loss(run_model, xb, yb, mb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    return out


# ---------------------------------------------------------------------
# Eval: GPT2-large CE on generated samples
# ---------------------------------------------------------------------

@torch.no_grad()
def estimate_gpt2_ce(diffusion_model, gpt2_model, gpt2_tokenizer,
                     num_samples=100):
    """
    Generate samples from the diffusion model, then score the generated
    (non-prefix) portion under GPT2-large.  Returns average CE in
    GPT2's BPE token space.
    """
    diffusion_model.eval()
    gpt2_model.eval()

    total_ce = 0.0
    total_tokens = 0

    for i in range(num_samples):
        # Pick a prefix from val_data (same logic as final sampling)
        prompt_start = i * block_size
        prompt_start = prompt_start % (len(val_data) - block_size)

        x = torch.full((1, block_size), mask_token_id, device=device)
        x[0, :prompt_len] = val_data[
            prompt_start : prompt_start + prompt_len
        ].to(device)

        prompt_mask = torch.zeros(
            (1, block_size), dtype=torch.bool, device=device
        )
        prompt_mask[:, :prompt_len] = True

        sample_text = generate_from(
            diffusion_model, x, prompt_mask,
            T=T, block_size=block_size, vocab_size=vocab_size,
            mask_token_id=mask_token_id,
            survival_prob_scalar=survival_prob_scalar,
            decode=decode,
        )

        # Tokenize full text with offset mappings so we can find the
        # exact BPE-token boundary between prefix and continuation.
        # BPE is NOT additive: BPE(a+b) != BPE(a) || BPE(b), so we
        # cannot just tokenize the prefix separately to find the split.
        enc = gpt2_tokenizer(
            sample_text, return_offsets_mapping=True, return_tensors="pt",
        )
        full_ids = enc["input_ids"][0]           # (L,)
        offsets = enc["offset_mapping"][0]        # (L, 2) — char spans

        # First token whose char span starts strictly at or after the
        # prompt boundary.  Tokens straddling the boundary are excluded.
        cont_start = None
        for idx_tok, (char_start, char_end) in enumerate(offsets.tolist()):
            if char_start >= prompt_len:
                cont_start = idx_tok
                break
        if cont_start is None or cont_start >= len(full_ids):
            continue  # entire text falls within / straddles the prefix

        if len(full_ids) <= cont_start + 1:
            continue  # too few continuation tokens to score

        input_ids = full_ids.unsqueeze(0).to(device)
        logits = gpt2_model(input_ids).logits  # (1, L, V_gpt2)

        # CE for next-token prediction, scored only on continuation.
        # logits[0, i] predicts token i+1, so to score tokens from
        # cont_start onward we use logits[cont_start-1 : -1].
        start = max(cont_start - 1, 0)
        target_ids = full_ids[start + 1:].to(device)
        pred_logits = logits[0, start : start + len(target_ids)]

        if len(target_ids) == 0:
            continue

        ce = F.cross_entropy(pred_logits, target_ids, reduction="sum").item()
        total_ce += ce
        total_tokens += len(target_ids)

    if total_tokens == 0:
        return float("inf")
    return total_ce / total_tokens


# ---------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------

def generate(model, prompt_len=16):
    model.eval()

    x = torch.full((1, block_size), mask_token_id, device=device)
    x[0, :prompt_len] = data[:prompt_len].to(device)

    prompt_mask = torch.zeros((1, block_size), dtype=torch.bool, device=device)
    prompt_mask[:, :prompt_len] = True

    return generate_from(
        model, x, prompt_mask,
        T=T, block_size=block_size, vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        survival_prob_scalar=survival_prob_scalar,
        decode=decode,
    )


# ---------------------------------------------------------------------
# Model config dict (passed to model's get_batch / compute_loss)
# ---------------------------------------------------------------------

cfg = {
    "train_data": train_data,
    "val_data": val_data,
    "batch_size": batch_size,
    "block_size": block_size,
    "T": T,
    "mask_token_id": mask_token_id,
    "vocab_size": vocab_size,
    "device": device,
    "survival_prob_tensor": survival_prob_tensor,
    # edit_one_pass specific
    "corrupt_prob": args.corrupt_prob,
    # edit_two_pass specific
    "lambda_corr": args.lambda_corr,
}


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters

    decay_ratio = (it - warmup_iters) / max(1, (max_iters - warmup_iters))
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    model = Model(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        head_dim=head_dim,
        block_size=block_size,
        dropout=dropout,
    ).to(device)
    adamw_kwargs = {"lr": min_lr}
    if "fused" in inspect.signature(torch.optim.AdamW).parameters:
        adamw_kwargs["fused"] = (device == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)

    use_compile = (device == "cuda") and hasattr(torch, "compile") and args.use_compile
    compiled_model = torch.compile(model) if use_compile else model

    gpt2_model = None
    gpt2_tokenizer = None
    gpt2_enabled = gpt2_eval_samples > 0 and gpt2_eval_interval > 0
    if gpt2_enabled:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        # Load GPT2-large for evaluation only when that metric is enabled.
        print("Loading GPT2-large for evaluation...")
        gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")
        gpt2_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        gpt2_model = GPT2LMHeadModel.from_pretrained(
            "gpt2-large", torch_dtype=gpt2_dtype
        ).to(device)
        gpt2_model.eval()
        print(f"GPT2-large loaded ({gpt2_dtype}).")

    print_run_info(model)

    train_losses = []
    val_losses = []
    gpt2_ces = []

    model.train()

    for iter in range(max_iters):
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        should_run_loss_eval = eval_interval > 0 and (iter % eval_interval == 0)
        should_run_gpt2_eval = gpt2_enabled and (iter % gpt2_eval_interval == 0)
        should_print_sample = sample_interval > 0 and (iter % sample_interval == 0)

        if should_run_loss_eval or should_run_gpt2_eval or should_print_sample:
            model.eval()
            metrics = None
            gpt2_ce = None

            if should_run_loss_eval:
                metrics = estimate_loss(compiled_model)
                train_losses.append((iter, metrics["train"]))
                val_losses.append((iter, metrics["val"]))

            if should_run_gpt2_eval:
                gpt2_ce = estimate_gpt2_ce(
                    model, gpt2_model, gpt2_tokenizer,
                    num_samples=gpt2_eval_samples,
                )
                gpt2_ces.append((iter, gpt2_ce))

            current_token_epoch = token_epochs_from_steps(iter, len(train_data))
            log_parts = [
                f"step {iter}",
                f"tok_epoch {current_token_epoch:.2f}",
            ]
            if metrics is not None:
                log_parts.append(f"train {metrics['train']:.4f}")
                log_parts.append(f"val {metrics['val']:.4f}")
            if gpt2_ce is not None:
                log_parts.append(f"gpt2_ce {gpt2_ce:.4f}")
            log_parts.append(f"lr {lr:.6f}")
            print(" | ".join(log_parts))

            if should_print_sample:
                print("Generating sample...")
                print(generate(model, prompt_len=prompt_len))
            model.train()

        batch = model_get_batch("train", cfg)

        with autocast_ctx():
            loss = model_compute_loss(compiled_model, batch, cfg)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if iter > 0 and iter % save_interval == 0:
            torch.save(
                {
                    "iter": iter,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                    "args": vars(args),
                    "vocab_size": vocab_size,
                    "stoi": stoi,
                    "itos": itos,
                },
                checkpoint_path,
            )
            print(f"saved checkpoint to {checkpoint_path} at step {iter}")

        if iter % 100 == 0:
            current_token_epoch = token_epochs_from_steps(iter, len(train_data))
            print(
                f"step {iter} | tok_epoch {current_token_epoch:.2f} | "
                f"loss {loss.item():.4f} | grad norm {grad_norm:.4f}"
            )

    torch.save(
        {
            "iter": max_iters,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
            "args": vars(args),
            "vocab_size": vocab_size,
            "stoi": stoi,
            "itos": itos,
        },
        checkpoint_path,
    )
    print(f"saved final checkpoint to {checkpoint_path}")

    print("\n" + "=" * 60)
    print(f"Generating {num_final_samples} samples")
    print("=" * 60)
    model.eval()

    samples = []
    for i in range(num_final_samples):
        prompt_start = i * block_size
        prompt_start = prompt_start % (len(val_data) - block_size)

        x = torch.full((1, block_size), mask_token_id, device=device)
        x[0, :prompt_len] = val_data[prompt_start : prompt_start + prompt_len].to(device)

        prompt_mask = torch.zeros((1, block_size), dtype=torch.bool, device=device)
        prompt_mask[:, :prompt_len] = True

        sample_text = generate_from(
            model, x, prompt_mask,
            T=T, block_size=block_size, vocab_size=vocab_size,
            mask_token_id=mask_token_id,
            survival_prob_scalar=survival_prob_scalar,
            decode=decode,
        )
        samples.append(sample_text)

        print(f"\n--- Sample {i + 1} ---")
        print(sample_text)

    with open(loss_log_path, "wb") as f:
        pickle.dump({
            "train": train_losses,
            "val": val_losses,
            "gpt2_ce": gpt2_ces,
        }, f)
