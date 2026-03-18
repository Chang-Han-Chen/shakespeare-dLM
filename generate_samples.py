"""
generate_samples.py — Load checkpoints and generate prefix-continuation samples.

Loads saved checkpoints from scaling/{model}/{size}/ckpt.pt, then generates
text samples using validation-set prefixes for qualitative comparison.

Usage:
    python generate_samples.py                              # all models, 3M size
    python generate_samples.py --models ar remasked --size 3M
    python generate_samples.py --prompt_len 32 --num_samples 10
    python generate_samples.py --custom_prompt "HAMLET:\nTo be, or not to be"
"""

import argparse
import importlib
import math
import os
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------

MODEL_MODULE_MAP = {
    "remasked": "model_remasked",
    "mdlm": "model_MDLM",
    "edit_one_pass": "model_edit_one_pass",
    "edit_two_pass": "model_edit_two_pass",
    "ar": "model_AR",
}

MODEL_SIZES = {
    "0.1M": (64, 2),
    "0.3M": (96, 3),
    "1M":   (128, 5),
    "3M":   (256, 4),
}

ALL_MODELS = list(MODEL_MODULE_MAP.keys())


# ---------------------------------------------------------------
# Noise schedule (matches train.py)
# ---------------------------------------------------------------

def make_survival_prob_scalar(T, t_min, t_max, noise_schedule):
    def survival_prob_scalar(t_step):
        t_frac = float(t_step) / T
        t_frac = min(max(t_frac, t_min), t_max)
        if noise_schedule == "linear":
            a_t = 1.0 - t_frac
        elif noise_schedule == "cosine":
            a_t = math.cos(0.5 * math.pi * t_frac)
        else:
            raise ValueError(f"unknown noise_schedule: {noise_schedule}")
        return max(0.0, min(1.0, a_t))
    return survival_prob_scalar


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate samples from saved checkpoints")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS)
    parser.add_argument("--size", type=str, default="3M",
                        choices=list(MODEL_SIZES.keys()))
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples per model")
    parser.add_argument("--prompt_len", type=int, default=32,
                        help="Number of prefix characters to condition on")
    parser.add_argument("--custom_prompt", type=str, default=None,
                        help="Use a custom prompt instead of validation data")
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--t_min", type=float, default=0.1)
    parser.add_argument("--t_max", type=float, default=0.9)
    parser.add_argument("--noise_schedule", type=str, default="linear")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Save output to file (default: print to stdout)")
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    # Load data + vocab
    with open("data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    chars = ["_"] + chars  # "_" is MASK
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    mask_token_id = stoi["_"]

    def encode(s):
        return [stoi.get(ch, mask_token_id) for ch in s]

    def decode(l):
        return "".join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    val_data = data[n:]

    survival_prob_scalar = make_survival_prob_scalar(
        args.T, args.t_min, args.t_max, args.noise_schedule
    )

    # Determine prompt positions from validation data
    torch.manual_seed(args.seed)
    prompt_starts = []
    for i in range(args.num_samples):
        start = torch.randint(0, len(val_data) - args.block_size, (1,)).item()
        prompt_starts.append(start)

    n_embd, n_layer = MODEL_SIZES[args.size]

    output_lines = []

    def log(s=""):
        output_lines.append(s)
        print(s)

    log("=" * 80)
    log(f"SAMPLE COMPARISON — size={args.size}, prompt_len={args.prompt_len}, "
        f"block_size={args.block_size}, T={args.T}")
    log("=" * 80)

    # Show prompts first
    log("\n" + "-" * 80)
    log("PROMPTS (from validation set)")
    log("-" * 80)
    for i, start in enumerate(prompt_starts):
        if args.custom_prompt is not None:
            prompt_text = args.custom_prompt[:args.prompt_len]
        else:
            prompt_tokens = val_data[start : start + args.prompt_len]
            prompt_text = decode(prompt_tokens.tolist())
        log(f"\nPrompt {i+1}: |{repr(prompt_text)}|")

    # Generate from each model
    for model_name in args.models:
        ckpt_path = f"scaling/{model_name}/{args.size}/ckpt.pt"

        if not os.path.exists(ckpt_path):
            log(f"\n{'=' * 80}")
            log(f"SKIPPING {model_name} {args.size} — no checkpoint at {ckpt_path}")
            continue

        log(f"\n{'=' * 80}")
        log(f"MODEL: {model_name} ({args.size})")
        log("=" * 80)

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_args = ckpt.get("args", {})

        # Import model module
        model_module = importlib.import_module(MODEL_MODULE_MAP[model_name])
        Model = model_module.Model
        generate_fn = model_module.generate_from

        # Build model with checkpoint's architecture
        ckpt_n_embd = ckpt_args.get("n_embd", n_embd)
        ckpt_n_layer = ckpt_args.get("n_layer", n_layer)
        ckpt_n_head = ckpt_args.get("n_head", 4)
        head_dim = ckpt_n_embd // ckpt_n_head

        model = Model(
            vocab_size=vocab_size,
            n_embd=ckpt_n_embd,
            n_head=ckpt_n_head,
            n_layer=ckpt_n_layer,
            head_dim=head_dim,
            block_size=args.block_size,
            dropout=0.0,  # no dropout at inference
        ).to(device)

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        log(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
        log(f"  Checkpoint step: {ckpt.get('iter', '?')}")
        log(f"  Checkpoint loss: {ckpt.get('loss', '?'):.4f}" if isinstance(ckpt.get('loss'), float) else "")

        for i, start in enumerate(prompt_starts):
            torch.manual_seed(args.seed + i)  # reproducible per sample

            if args.custom_prompt is not None:
                prompt_tokens = torch.tensor(
                    encode(args.custom_prompt[:args.prompt_len]),
                    dtype=torch.long
                )
            else:
                prompt_tokens = val_data[start : start + args.prompt_len]

            # Prepare input
            x = torch.full((1, args.block_size), mask_token_id, device=device)
            x[0, :len(prompt_tokens)] = prompt_tokens.to(device)

            prompt_mask = torch.zeros(
                (1, args.block_size), dtype=torch.bool, device=device
            )
            prompt_mask[:, :len(prompt_tokens)] = True

            # Generate
            sample_text = generate_fn(
                model, x, prompt_mask,
                T=args.T,
                block_size=args.block_size,
                vocab_size=vocab_size,
                mask_token_id=mask_token_id,
                survival_prob_scalar=survival_prob_scalar,
                decode=decode,
            )

            prompt_text = decode(prompt_tokens.tolist())
            continuation = sample_text[len(prompt_text):]

            log(f"\n  --- Sample {i+1} ---")
            log(f"  Prompt:       |{repr(prompt_text)}|")
            log(f"  Continuation: |{repr(continuation[:200])}|")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Save if requested
    if args.output:
        with open(args.output, "w") as f:
            f.write("\n".join(output_lines))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
